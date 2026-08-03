// NOTE(claude): Post-processing ray tracer for Mosscap-coupled Dex output.
// Reads a single combined Mosscap snapshot netcdf (Mosscap's dense conserved
// MHD state `Q` + Dex's sparse populations/diagnostics on the same
// block/morton grid) and traces rays through it exactly like DexRT's own
// dexrt_ray, whose shared ray-marching/output code lives in
// DexRT/source/PostProcessingCore.hpp.
//
// Density/velocity/pressure are derived fresh from `Q` via Mosscap's own
// cons_to_prim, using the per-run "i*" conserved-variable index attributes
// written alongside `Q` -- this keeps the reader correct for hydro vs
// MHD/GLM-MHD inputs without hardcoding which fluid type produced the file.
// `n_e` is trusted as-is from the file (it isn't energy-derived, so
// re-deriving it buys nothing); temperature/pressure are energy-derived and
// so are recomputed from `Q` rather than trusted.
// NOTE(claude): This file lives in source/ (Mosscap's own tree), which has
// its own Types.hpp/Config.hpp -- DexRT headers of the same name must be
// included by explicit relative path here to avoid silently picking up
// Mosscap's versions instead (matching the convention already used by
// source/DexInterface.cpp for its DexRT includes).
#include "../DexRT/source/Types.hpp"
#include <argparse/argparse.hpp>
#include <string>
#include <vector>
#include <algorithm>
#include <fmt/core.h>
#include <yaml-cpp/yaml.h>
#include <YAKL_netcdf.h>
#include "../DexRT/source/Utils.hpp"
#include "../DexRT/source/DexrtConfig.hpp"
#include "../DexRT/source/CrtafParser.hpp"
#include "../DexRT/source/Populations.hpp"
#include "../DexRT/source/BlockMap.hpp"
#include "../DexRT/source/MortonCodes.hpp"
#include "../DexRT/source/Constants.hpp"
#include "../DexRT/source/PostProcessingCore.hpp"
#include <sstream>
#include "../DexRT/thirdparty/tqdm.hpp"

#include "Eos.hpp"
#include "State.hpp"
#include "AtmosCommon.hpp"
#include "MosscapConfig.hpp"

int get_dexrt_dimensionality() {
    return 2;
}

/// Post-process a range of snapshots (same mosscap_config_path/muz/mux/
/// wavelength for all of them) instead of a single mosscap_output_path. The
/// two patterns are fmt::format strings taking the snapshot index as their
/// sole positional argument, e.g. "condensation_mhd_restart_HCa_{:05d}.nc"
/// reproduces Mosscap's own zero-padded naming convention
/// (Output.cpp's get_filename).
struct BatchConfig {
    bool enable = false;
    std::string mosscap_output_pattern;
    std::string ray_output_pattern;
    int start = 0;
    int end = 0;
    int stride = 1;
};

struct MosscapRayConfig {
    fp_t mem_pool_gb = FP(2.0);
    std::string own_path;
    /// Path to the yaml that ran the Mosscap simulation (its "dex" and "eos"
    /// sections are reused verbatim).
    std::string mosscap_config_path;
    /// Path to a single Mosscap output snapshot netcdf to ray-trace. Ignored
    /// (and not required) when batch.enable is true.
    std::string mosscap_output_path;
    std::string ray_output_path;
    BatchConfig batch;
    std::vector<fp_t> muz;
    std::vector<fp_t> mux;
    std::vector<fp_t> wavelength;
    bool rotate_aabb = true;
    bool output_cfn = false;
    bool output_eta_chi = false;
    DexrtConfig dexrt;
    Mosscap::fp_t gamma = 5.0 / 3.0;
    Mosscap::fp_t mass_per_h = 1.0;
    Mosscap::fp_t total_abund = 1.0;
};

MosscapRayConfig parse_mosscap_ray_config(const std::string& path) {
    MosscapRayConfig config;
    config.own_path = path;
    config.ray_output_path = "mosscap_ray_output.nc";

    YAML::Node file = YAML::LoadFile(path);
    auto require_key = [&] (const std::string& key) {
        if (!file[key]) {
            throw std::runtime_error(fmt::format("{} key must be present in config file.", key));
        }
    };
    require_key("mosscap_config_path");
    config.mosscap_config_path = file["mosscap_config_path"].as<std::string>();

    if (file["batch"] && file["batch"]["enable"] && file["batch"]["enable"].as<bool>()) {
        auto batch_node = file["batch"];
        auto require_batch_key = [&] (const std::string& key) {
            if (!batch_node[key]) {
                throw std::runtime_error(fmt::format("batch.{} key must be present when batch.enable is true.", key));
            }
        };
        require_batch_key("mosscap_output_pattern");
        require_batch_key("ray_output_pattern");
        require_batch_key("start");
        require_batch_key("end");
        config.batch.enable = true;
        config.batch.mosscap_output_pattern = batch_node["mosscap_output_pattern"].as<std::string>();
        config.batch.ray_output_pattern = batch_node["ray_output_pattern"].as<std::string>();
        config.batch.start = batch_node["start"].as<int>();
        config.batch.end = batch_node["end"].as<int>();
        if (batch_node["stride"]) {
            config.batch.stride = batch_node["stride"].as<int>();
        }
        if (config.batch.stride <= 0) {
            throw std::runtime_error("batch.stride must be positive.");
        }
        if (config.batch.end < config.batch.start) {
            throw std::runtime_error("batch.end must be >= batch.start.");
        }
    } else {
        require_key("mosscap_output_path");
        config.mosscap_output_path = file["mosscap_output_path"].as<std::string>();
        if (file["ray_output_path"]) {
            config.ray_output_path = file["ray_output_path"].as<std::string>();
        }
    }

    if (file["rotate_aabb"]) {
        config.rotate_aabb = file["rotate_aabb"].as<bool>();
    }
    if (file["output_cfn"]) {
        config.output_cfn = file["output_cfn"].as<bool>();
    }
    if (file["output_eta_chi"]) {
        config.output_eta_chi = file["output_eta_chi"].as<bool>();
    }
    if (file["system"]) {
        auto system = file["system"];
        if (system["mem_pool_gb"]) {
            config.mem_pool_gb = system["mem_pool_gb"].as<fp_t>();
        }
    }

    auto parse_one_or_more_float_to_vector = [&] (const std::string& key) {
        std::vector<fp_t> result;
        if (!file[key]) {
            return result;
        }

        if (file[key].IsSequence()) {
            result.reserve(file[key].size());
            for (const auto& v : file[key]) {
                result.push_back(v.as<fp_t>());
            }
        } else {
            result.push_back(file[key].as<fp_t>());
        }
        return result;
    };
    require_key("muz");
    require_key("mux");
    config.muz = parse_one_or_more_float_to_vector("muz");
    config.mux = parse_one_or_more_float_to_vector("mux");
    config.wavelength = parse_one_or_more_float_to_vector("wavelength");
    if ((config.muz.size() != config.mux.size()) || config.muz.size() == 0) {
        throw std::runtime_error("muz and mux must be provided and have the same number of entries (non-zero).");
    }

    // NOTE(cmo): The Mosscap run's own yaml already contains a "dex" section
    // with the exact schema DexrtConfig expects (atom paths, boundary type,
    // threshold_temperature, ...) -- it's passed unmodified to
    // parse_dexrt_config the same way Mosscap's own DexInterface does. Only
    // `atmos_path` (the Promweaver boundary file) and the atom/boundary/mode
    // settings are used from it; `output_path` is a dead key in Mosscap's
    // coupled output.
    YAML::Node mosscap_file = YAML::LoadFile(config.mosscap_config_path);
    if (!mosscap_file["dex"]) {
        throw std::runtime_error(fmt::format("No \"dex\" section found in {}", config.mosscap_config_path));
    }
    YAML::Node dex_node = mosscap_file["dex"];
    config.dexrt = parse_dexrt_config(config.mosscap_config_path, dex_node);

    config.gamma = Mosscap::get_or<Mosscap::fp_t>(mosscap_file, "eos.gamma", 5.0 / 3.0);
    config.mass_per_h = Mosscap::get_or<Mosscap::fp_t>(mosscap_file, "eos.mass_per_h", 1.0);
    config.total_abund = Mosscap::get_or<Mosscap::fp_t>(mosscap_file, "eos.total_abund", 1.0);

    return config;
}

void load_wavelength_if_missing(MosscapRayConfig* cfg) {
    MosscapRayConfig& config = *cfg;
    if (config.wavelength.size() == 0) {
        const std::string path = config.batch.enable
            ? fmt::format(fmt::runtime(config.batch.mosscap_output_pattern), config.batch.start)
            : config.mosscap_output_path;
        yakl::Array<f32, 1, yakl::memHost> wavelengths;
        yakl::SimpleNetCDF nc;
        nc.open(path, yakl::NETCDF_MODE_READ);
        nc.read(wavelengths, "wavelength");
        config.wavelength.reserve(wavelengths.extent(0));
        for (int i = 0; i < wavelengths.extent(0); ++i) {
            config.wavelength.push_back(wavelengths(i));
        }
        nc.close();
    }
}

namespace MosscapRayImpl {
    /// Presence-tested conserved-variable slot indices, mirroring
    /// write_cons_header in source/Output.cpp. Absent entries are left at -1.
    struct ConsIndices {
        int irho = -1;
        int imx = -1;
        int imy = -1;
        int imz = -1;
        int iene = -1;
        int iione = -1;
        int ibx = -1;
        int iby = -1;
        int ibz = -1;
        int ipsi = -1;
        int iheat = -1;
    };

    inline bool has_global_attr(int ncid, const char* name) {
        nc_type dtype;
        size_t len;
        return nc_inq_att(ncid, NC_GLOBAL, name, &dtype, &len) == NC_NOERR;
    }

    inline int read_global_int_attr(int ncid, const char* name) {
        int value;
        if (nc_get_att_int(ncid, NC_GLOBAL, name, &value) != NC_NOERR) {
            throw std::runtime_error(fmt::format("Failed to read required global attribute \"{}\"", name));
        }
        return value;
    }

    inline ConsIndices read_cons_indices(int ncid) {
        ConsIndices idx;
        idx.irho = read_global_int_attr(ncid, "irho");
        idx.imx = read_global_int_attr(ncid, "imx");
        if (has_global_attr(ncid, "imy")) {
            idx.imy = read_global_int_attr(ncid, "imy");
        }
        if (has_global_attr(ncid, "imz")) {
            idx.imz = read_global_int_attr(ncid, "imz");
        }
        idx.iene = read_global_int_attr(ncid, "iene");
        idx.iione = read_global_int_attr(ncid, "iione");
        if (has_global_attr(ncid, "ibx")) {
            idx.ibx = read_global_int_attr(ncid, "ibx");
            idx.iby = read_global_int_attr(ncid, "iby");
            idx.ibz = read_global_int_attr(ncid, "ibz");
        }
        if (has_global_attr(ncid, "ipsi")) {
            idx.ipsi = read_global_int_attr(ncid, "ipsi");
        }
        if (has_global_attr(ncid, "iheat")) {
            idx.iheat = read_global_int_attr(ncid, "iheat");
        }
        return idx;
    }

    /// Derives nh_tot/pressure/temperature/vturb/vx/vz for every active cell
    /// from Q via cons_to_prim, storing them into atmos. Must be a standalone
    /// function template (not a lambda) -- nvcc forbids defining an extended
    /// __host__ __device__ lambda (the dex_parallel_for/KOKKOS_LAMBDA below)
    /// inside a generic lambda, which is what invoke_fluid_traits's dispatch
    /// callback is.
    template <typename FTraits>
    void derive_atmosphere_from_Q(
        const SparseAtmosphere& atmos,
        const MultiResBlockMap<BLOCK_SIZE, ENTRY_SIZE, 2>& mr_block_map,
        const yakl::Array<Mosscap::fp_t, 4, yakl::memDevice>& Q,
        const ConsIndices& cons_idx,
        i32 num_ghost,
        Mosscap::fp_t gamma,
        Mosscap::fp_t mu0,
        Mosscap::fp_t mass_per_h,
        Mosscap::fp_t total_abund
    ) {
        using Cons = typename FTraits::cons;
        using Prim = typename FTraits::prim;
        constexpr int n_hydro = FTraits::num_vars;
        constexpr Mosscap::fp_t m_p = ConstantsF64::u;

        dex_parallel_for(
            "Derive atmosphere from Q",
            mr_block_map.block_map.loop_bounds(),
            KOKKOS_LAMBDA (i64 tile_idx, i32 block_idx) {
                IdxGen idx_gen(mr_block_map);
                i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
                Coord2 coord = idx_gen.loop_coord(tile_idx, block_idx);

                // NOTE(cmo): dex z == mosscap y.
                const i32 qi = coord.x + num_ghost;
                const i32 qj = coord.z + num_ghost;

                yakl::SArray<Mosscap::fp_t, 1, n_hydro> q;
                q(Mosscap::I(Cons::Rho)) = Q(cons_idx.irho, 0, qj, qi);
                q(Mosscap::I(Cons::MomX)) = Q(cons_idx.imx, 0, qj, qi);
                if constexpr (FTraits::is_mhd || FTraits::num_dim > 1) {
                    q(Mosscap::I(Cons::MomY)) = Q(cons_idx.imy, 0, qj, qi);
                }
                if constexpr (FTraits::is_mhd || FTraits::num_dim > 2) {
                    q(Mosscap::I(Cons::MomZ)) = Q(cons_idx.imz, 0, qj, qi);
                }
                q(Mosscap::I(Cons::Ene)) = Q(cons_idx.iene, 0, qj, qi);
                q(Mosscap::I(Cons::IonE)) = Q(cons_idx.iione, 0, qj, qi);
                if constexpr (FTraits::is_mhd) {
                    q(Mosscap::I(Cons::Bx)) = Q(cons_idx.ibx, 0, qj, qi);
                    q(Mosscap::I(Cons::By)) = Q(cons_idx.iby, 0, qj, qi);
                    q(Mosscap::I(Cons::Bz)) = Q(cons_idx.ibz, 0, qj, qi);
                    if constexpr (Mosscap::is_instance(FTraits::fluid_type, Mosscap::FluidType::GlmMhd)) {
                        q(Mosscap::I(Cons::Psi)) = Q(cons_idx.ipsi, 0, qj, qi);
                    }
                    if constexpr (FTraits::has_hypertc) {
                        q(Mosscap::I(Cons::HeatF)) = Q(cons_idx.iheat, 0, qj, qi);
                    }
                }

                yakl::SArray<Mosscap::fp_t, 1, n_hydro> w;
                Mosscap::cons_to_prim<FTraits>(gamma, mu0, q, w);

                const Mosscap::fp_t rho = w(Mosscap::I(Prim::Rho));
                const Mosscap::fp_t nh_tot = rho / (mass_per_h * m_p);
                const Mosscap::fp_t ne = Mosscap::fp_t(atmos.ne(ks));
                const Mosscap::fp_t y = ne / nh_tot;
                const Mosscap::fp_t pressure = w(Mosscap::I(Prim::Pres));
                const Mosscap::fp_t temperature = Mosscap::temperature_si(pressure, nh_tot, total_abund, y);

                atmos.nh_tot(ks) = fp_t(nh_tot);
                atmos.pressure(ks) = fp_t(pressure);
                atmos.temperature(ks) = fp_t(temperature);
                atmos.vturb(ks) = fp_t(Mosscap::vturb_fn(temperature, nh_tot, ne));
                atmos.vx(ks) = fp_t(w(Mosscap::I(Prim::Vx)));
                atmos.vz(ks) = fp_t(w(Mosscap::I(Prim::Vy)));
            }
        );
    }

    /// Detect the FluidType that produced this file purely from which
    /// conserved-variable attributes are present -- mirrors the conditionals
    /// in write_cons_header (source/Output.cpp), inverted. MhdHyperTc is
    /// reported in preference to HyperTcOnly when ambiguous; the two are
    /// indistinguishable from attribute presence alone, but behave
    /// identically in cons_to_prim (they only diverge in prim_to_flux, which
    /// isn't used here).
    inline Mosscap::FluidType detect_fluid_type(int ncid) {
        const bool is_mhd = has_global_attr(ncid, "ibx");
        if (!is_mhd) {
            return Mosscap::FluidType::Hydro;
        }
        const bool is_glm = has_global_attr(ncid, "ipsi");
        const bool has_hypertc = has_global_attr(ncid, "iheat");
        if (is_glm && has_hypertc) {
            return Mosscap::FluidType::GlmMhdHyperTc;
        }
        if (is_glm) {
            return Mosscap::FluidType::GlmMhd;
        }
        if (has_hypertc) {
            return Mosscap::FluidType::MhdHyperTc;
        }
        return Mosscap::FluidType::Mhd;
    }
}

/// Load the sparse atmosphere + populations for a Mosscap-coupled snapshot
/// directly into a DexRayState, deriving density/velocity/pressure/
/// temperature/vturb fresh from `Q`. Mirrors the structure of
/// DexInterface::init_atmosphere/update_atmosphere (source/DexInterface.cpp),
/// but reads a static netcdf snapshot instead of live simulation state, and
/// trusts the file's `ne` rather than recomputing an ionisation fraction.
void load_mosscap_atmosphere(const MosscapRayConfig& config, const std::string& mosscap_output_path, DexRayState* state) {
    using namespace MosscapRayImpl;

    yakl::SimpleNetCDF nc;
    nc.open(mosscap_output_path, yakl::NETCDF_MODE_READ);
    int ncid = nc.file.ncid;

    const int block_size = read_global_int_attr(ncid, "block_size");
    if (block_size != BLOCK_SIZE) {
        throw std::runtime_error(
            fmt::format(
                "block_size in Mosscap output ({}) does not match compiled BLOCK_SIZE ({}).",
                block_size,
                BLOCK_SIZE
            )
        );
    }
    const int num_x_blocks = read_global_int_attr(ncid, "num_x_blocks");
    const int num_z_blocks = read_global_int_attr(ncid, "num_z_blocks");
    const int num_ghost = read_global_int_attr(ncid, "num_ghost");
    double voxel_scale_d = 0.0;
    if (nc_get_att_double(ncid, NC_GLOBAL, "voxel_scale", &voxel_scale_d) != NC_NOERR) {
        throw std::runtime_error("Failed to read required global attribute \"voxel_scale\"");
    }

    const i32 num_x = num_x_blocks * BLOCK_SIZE;
    const i32 num_z = num_z_blocks * BLOCK_SIZE;

    // NOTE(cmo): Build the BlockMap from the same morton_tiles/num_active_tiles
    // convention DexRT's own sparse atmosphere/output format uses (see
    // setup_block_map_sparse_atmos, DexRT/source/BlockMap.cpp) -- Mosscap's
    // dex output follows it exactly, just under different dim/attr names.
    BlockMap<BLOCK_SIZE, 2> block_map;
    block_map.num_x_tiles() = num_x_blocks;
    block_map.num_z_tiles() = num_z_blocks;
    block_map.num_active_tiles = nc.getDimSize("num_active_tiles");
    block_map.bbox.min = 0;
    block_map.bbox.max(0) = num_x;
    block_map.bbox.max(1) = num_z;

    const i64 num_total_tiles = i64(num_x_blocks) * i64(num_z_blocks);
    yakl::Array<uint32_t, 1, yakl::memHost> morton_order("morton_traversal_order", num_total_tiles);
    i64 morton_idx = 0;
    for (int z = 0; z < num_z_blocks; ++z) {
        for (int x = 0; x < num_x_blocks; ++x) {
            morton_order(morton_idx++) = encode_morton<2>(Coord2{.x = x, .z = z});
        }
    }
    std::sort(morton_order.begin(), morton_order.end());
    block_map.morton_traversal_order = morton_order.createDeviceCopy();

    block_map.lookup.init(Dims<2>{.x = num_x_blocks, .z = num_z_blocks});
    nc.read(block_map.active_tiles, "morton_tiles");
    if (block_map.active_tiles.extent(0) != block_map.num_active_tiles) {
        throw std::runtime_error("num_active_tiles attribute doesn't match the length of morton_tiles.");
    }
    dex_parallel_for(
        "Set up BlockMap lookup from morton_tiles",
        FlatLoop<1>(block_map.active_tiles.extent(0)),
        KOKKOS_LAMBDA (i64 active_idx) {
            Coord2 tile_coord = decode_morton<2>(block_map.active_tiles(active_idx));
            block_map.lookup(tile_coord) = active_idx;
        }
    );
    Kokkos::fence();

    state->mr_block_map.init(block_map, /*max_mip_level=*/0);
    configure_mr_block_map(state->mr_block_map);

    const i64 num_active_cells = i64(block_map.num_active_tiles) * DexImpl::int_pow<2>(BLOCK_SIZE);

    // NOTE(cmo): voxel_scale/offsets aren't persisted as such by Mosscap's
    // writer; voxel_scale comes from the dex attribute (== sim.state.dx), and
    // the domain's physical origin comes straight from the original run's own
    // grid config (dex's z axis is Mosscap's y axis, see DexInterface.cpp's
    // "z in dex is y in mosscap" convention).
    YAML::Node mosscap_file = YAML::LoadFile(config.mosscap_config_path);
    fp_t offset_x = FP(0.0);
    fp_t offset_z = FP(0.0);
    if (mosscap_file["grid"]) {
        auto grid = mosscap_file["grid"];
        if (grid["x_start"]) {
            offset_x = grid["x_start"].as<fp_t>();
        }
        if (grid["y_start"]) {
            offset_z = grid["y_start"].as<fp_t>();
        }
    }

    state->atmos = SparseAtmosphere{
        .voxel_scale = fp_t(voxel_scale_d),
        .offset_x = offset_x,
        .offset_y = FP(0.0),
        .offset_z = offset_z,
        .num_x = num_x,
        .num_y = 0,
        .num_z = num_z,
        .moving = true,
        .temperature = Fp1d("temperature", num_active_cells),
        .pressure = Fp1d("pressure", num_active_cells),
        .ne = Fp1d("ne", num_active_cells),
        .nh_tot = Fp1d("nh_tot", num_active_cells),
        .nh0 = Fp1d("nh0", num_active_cells),
        .vturb = Fp1d("vturb", num_active_cells),
        .vx = Fp1d("vx", num_active_cells),
        .vy = Fp1d("vy", num_active_cells),
        .vz = Fp1d("vz", num_active_cells)
    };
    state->atmos.nh0 = FP(0.0);
    state->atmos.vy = FP(0.0);
    Kokkos::fence();

    // NOTE(claude): n_e isn't energy-derived, so the file's own (conservation
    // -adjusted) value is trusted directly rather than recomputed.
    nc.read(state->atmos.ne, "ne");
    nc.read(state->pops, "pops");

    const int var_dim_size = nc.getDimSize("var");
    const ConsIndices cons_idx = read_cons_indices(ncid);
    const Mosscap::FluidType fluid_type = detect_fluid_type(ncid);

    yakl::Array<Mosscap::fp_t, 4, yakl::memDevice> Q;
    nc.read(Q, "Q");

    const Mosscap::fp_t gamma = config.gamma;
    const Mosscap::fp_t mass_per_h = config.mass_per_h;
    const Mosscap::fp_t total_abund = config.total_abund;
    const Mosscap::fp_t mu0 = 4.0e-7 * 3.14159265358979312; // NOTE(cmo): matches Mosscap::State's default, not currently config-overridable

    const auto& atmos = state->atmos;
    const auto& mr_block_map = state->mr_block_map;

    Mosscap::invoke_fluid_traits(2, fluid_type, [&]<typename FTraits>(FTraits) {
        // NOTE(claude): Q's "var" dim can be larger than FTraits::num_vars --
        // when CMA tracer-advection is enabled (advected populations/ion
        // fraction), Mosscap appends extra tracer columns after the base
        // hydro/MHD slots. The irho/imx/... attributes describe exactly
        // where the base fields live regardless, so only check they fit.
        if (FTraits::num_vars > var_dim_size) {
            throw std::runtime_error(
                fmt::format(
                    "Detected fluid type implies at least {} conserved variables, but Q only has {}.",
                    FTraits::num_vars,
                    var_dim_size
                )
            );
        }
        derive_atmosphere_from_Q<FTraits>(
            atmos,
            mr_block_map,
            Q,
            cons_idx,
            num_ghost,
            gamma,
            mu0,
            mass_per_h,
            total_abund
        );
    });
    Kokkos::fence();
}

/// Ray-trace a single Mosscap snapshot (mosscap_output_path) into
/// ray_output_path, using the already-constructed atom-model/EOS state
/// (adata/phi/nh_lte). Reloads the atmosphere+pops and reopens the output
/// file, but reuses everything else -- this is the per-snapshot unit of work
/// shared by both single-file and batch mode.
void process_snapshot(
    const MosscapRayConfig& config,
    const std::string& mosscap_output_path,
    const std::string& ray_output_path,
    DexRayState& state,
    bool quiet
) {
    load_mosscap_atmosphere(config, mosscap_output_path, &state);

    state.eta = Fp1d(
        "eta",
        state.atmos.temperature.extent(0)
    );
    state.chi = Fp1d(
        "chi",
        state.atmos.temperature.extent(0)
    );

    auto out = setup_output(ray_output_path, config, state.atmos);

    auto mu_iterator = tq::trange(config.muz.size());
    std::ostringstream ostream_redirect;
    if (quiet) {
        mu_iterator.set_ostream(ostream_redirect);
    }
    for (int mu : mu_iterator) {
        state.ray_set = compute_ray_set<yakl::memDevice>(config, state.atmos, mu);
        PwBc<> pw_bc = load_bc(
            config.dexrt.atmos_path,
            state.ray_set.wavelength,
            config.dexrt.boundary,
            PromweaverResampleType::Interpolation
        );

        if (
            !state.ray_I.initialized()
            || (state.ray_I.extent(0) != state.ray_set.wavelength.extent(0))
            || (state.ray_I.extent(1) != state.ray_set.start_coord.extent(0))
        ) {
            state.ray_I = Fp2d(
                "I",
                state.ray_set.wavelength.extent(0),
                state.ray_set.start_coord.extent(0)
            );
            state.ray_tau = Fp2d(
                "tau",
                state.ray_set.wavelength.extent(0),
                state.ray_set.start_coord.extent(0)
            );
        }
        DexRayStateAndBc<PwBc<>> ray_state{
            .state = state,
            .bc = pw_bc
        };
        compute_ray_intensity(&ray_state, config);
        write_output_plane(out, ray_state.state, config, mu);
    }
}

int main(int argc, char** argv) {
    argparse::ArgumentParser program("Mosscap Ray");
    program.add_argument("--config")
        .default_value(std::string("mosscap_ray.yaml"))
        .help("Path to config file")
        .metavar("FILE");
    program.add_argument("--quiet")
        .default_value(false)
        .implicit_value(true)
        .help("Whether to print progress");
    program.add_epilog("Single-pass formal solver for post-processing Mosscap-coupled Dex models.");

    program.parse_known_args(argc, argv);

    MosscapRayConfig config = parse_mosscap_ray_config(program.get<std::string>("--config"));
    bool quiet = program.get<bool>("--quiet");
    Kokkos::initialize(argc, argv);
    yakl::init(
        yakl::InitConfig()
            .set_pool_size_mb(config.mem_pool_gb * 1024)
    );
    {
        load_wavelength_if_missing(&config);
        if (config.dexrt.mode == DexrtMode::GivenFs) {
            throw std::runtime_error(fmt::format("Models run in GivenFs mode not supported by {}", argv[0]));
        }
        if (config.dexrt.boundary != BoundaryType::Promweaver) {
            throw std::runtime_error(fmt::format("Only promweaver boundaries are supported by {}", argv[0]));
        }
        std::vector<ModelAtom<f64>> crtaf_models;
        crtaf_models.reserve(config.dexrt.atom_paths.size());
        for (int i = 0; i < config.dexrt.atom_paths.size(); ++i) {
            const auto& p = config.dexrt.atom_paths[i];
            const auto& model_config = config.dexrt.atom_configs[i];
            crtaf_models.emplace_back(parse_crtaf_model<f64>(p, model_config));
        }
        AtomicDataHostDevice<fp_t> atomic_data = to_atomic_data<fp_t, f64>(
            crtaf_models,
            ToAtomicDataOptions{
                .limit_line_edge_bins=false
            }
        );

        DexRayState state{
            .adata = atomic_data.device,
            .phi = VoigtProfile<fp_t>(
                VoigtProfile<fp_t>::Linspace{FP(0.0), FP(0.15), 1024},
                VoigtProfile<fp_t>::Linspace{FP(0.0), FP(1.5e3), 64 * 1024}
            ),
            .nh_lte = HPartFn(),
        };

        if (config.batch.enable) {
            int snapshot_count = (config.batch.end - config.batch.start) / config.batch.stride + 1;
            int snapshot_num = 0;
            for (int idx = config.batch.start; idx <= config.batch.end; idx += config.batch.stride) {
                ++snapshot_num;
                const std::string mosscap_output_path = fmt::format(fmt::runtime(config.batch.mosscap_output_pattern), idx);
                const std::string ray_output_path = fmt::format(fmt::runtime(config.batch.ray_output_pattern), idx);
                if (!quiet) {
                    fmt::println("Snapshot {} ({}/{}): {}", idx, snapshot_num, snapshot_count, mosscap_output_path);
                }
                process_snapshot(config, mosscap_output_path, ray_output_path, state, quiet);
            }
        } else {
            process_snapshot(config, config.mosscap_output_path, config.ray_output_path, state, quiet);
        }
    }
    yakl::finalize();
    Kokkos::finalize();

    return 0;
}
