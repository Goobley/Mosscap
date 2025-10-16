#define JASNAH_NO_FIF
#include "DexInterface.hpp"
#include "Simulation.hpp"
#include "JasPP.hpp"
// NOTE(cmo): The reason Kokkos sort was failing before was due to the (never
// used If, Then, Else in JasPP)
#include "Kokkos_Sort.hpp"

#include "../DexRT/source/CrtafParser.hpp"
#include "../DexRT/source/Populations.hpp"
#include "../DexRT/source/RcUtilsModes.hpp"
#include "../DexRT/source/WavelengthParallelisation.hpp"
#include "../DexRT/source/Collisions.hpp"
#include "../DexRT/source/ChargeConservation.hpp"
#include "../DexRT/source/PressureConservation.hpp"
#include "../DexRT/source/NgAcceleration.hpp"
#include "../DexRT/source/ProfileNormalisation.hpp"
#include "../DexRT/source/DynamicFormalSolution.hpp"
#include "../DexRT/source/MiscSparse.hpp"
#include "../DexRT/source/InitialPops.hpp"

// TODO(cmo): Figure out how to deal with 3D down the line.
int get_dexrt_dimensionality() {
    return 2;
}

namespace Mosscap {

// NOTE(cmo): Direct transplant from dexrt
static void allocate_J(DexState& state) {
    JasUnpack(state, config, mr_block_map, c0_size, adata);
    const auto& block_map = mr_block_map.block_map;
    const bool sparse = config.sparse_calculation;
    i64 num_cells = mr_block_map.block_map.get_num_active_cells();
    i32 wave_dim = adata.wavelength.extent(0);

    if (!sparse) {
        num_cells = i64(block_map.num_x_tiles()) * block_map.num_z_tiles() * square(BLOCK_SIZE);
    }

    if (config.store_J_on_cpu) {
        state.J = decltype(state.J)("J", yakl::DimsT<i64>(c0_size.wave_batch, num_cells));
        state.J_cpu = decltype(state.J_cpu)("JHost", yakl::DimsT<i64>(wave_dim, num_cells));
    } else {
        state.J = decltype(state.J)("J", yakl::DimsT<i64>(wave_dim, num_cells));
    }
    state.J = 0;
    // TODO(cmo): If we have scattering terms and are updating J, the old
    // contents should probably be moved first, but we don't have these terms yet.
}

static void allocate_cell_count_based_terms(DexState& state, i64 num_active_cells) {
    const int n_level_total = state.adata.energy.extent(0);
    state.pops = decltype(state.pops)("pops", n_level_total, num_active_cells);
    state.Gamma.clear();
    for (int ia = 0; ia < state.adata_host.num_level.extent(0); ++ia) {
        const int n_level = state.adata_host.num_level(ia);
        state.Gamma.emplace_back(
            decltype(state.Gamma)::value_type("Gamma", n_level, n_level, num_active_cells)
        );
    }
    state.wphi = decltype(state.wphi)("wphi", state.adata.lines.extent(0), num_active_cells);

    // TODO(cmo): Maybe no J unless requested.
    allocate_J(state);
}

bool DexInterface::update_atmosphere(Simulation& sim) {
    constexpr i32 num_dim = 2;
    constexpr i32 block_size = BLOCK_SIZE;
    constexpr fp_t m_p = ConstantsF64::u;
    auto& block_map = state.mr_block_map.block_map;
    const i32 num_x = block_map.num_x_tiles() * block_size;
    const i32 num_z = block_map.num_z_tiles() * block_size;
    const auto& sz = sim.state.sz;

    const auto& Q = sim.state.Q;
    const auto& eos = sim.eos;
    auto cutoff_temperature = state.config.threshold_temperature;

    constexpr u32 sentinel = std::numeric_limits<u32>::max();
    yakl::Array<u32, 1, yakl::memDevice> active_tiles("active tiles", block_map.morton_traversal_order.extent(0));

    i32 num_active_tiles = 0;
    dex_parallel_reduce(
        "Compute active tiles",
        FlatLoop<1>(block_map.morton_traversal_order.extent(0)),
        KOKKOS_LAMBDA (i64 tile_idx, i32& num_active_tiles) {
            u32 code = block_map.morton_traversal_order(tile_idx);
            Coord<num_dim> coord = decode_morton<num_dim>(code);
            const i32 xt = coord.x;
            const i32 zt = coord.z;

            constexpr int n_hydro = N_HYDRO_VARS<num_dim>;
            yakl::SArray<fp_t, 1, n_hydro> w;
            using Prim = Prim<num_dim>;

            for (int z = zt * block_size; z < (zt + 1) * block_size; ++z) {
                for (int x = xt * block_size; x < (xt + 1) * block_size; ++x) {
                    CellIndex idx{.i = x + sz.ng, .j = z + sz.ng, .k = 0};
                    const auto q = QtyView(Q, idx);
                    cons_to_prim<num_dim>(eos.gamma, q, w);

                    fp_t n_baryon = w(I(Prim::Rho)) / (eos.avg_mass * m_p);
                    fp_t y = eos.y;
                    if (!eos.is_constant) {
                        y = eos.y_space(idx.k, idx.j, idx.i);
                    }
                    auto temp = temperature_si(w(I(Prim::Pres)), n_baryon, y);
                    if (temp <= cutoff_temperature) {
                        num_active_tiles += 1;
                        active_tiles(tile_idx) = code;
                        return;
                    }
                }
            }
            active_tiles(tile_idx) = sentinel;
        },
        Kokkos::Sum<i32>(num_active_tiles)
    );
    fmt::println("num_active_tiles: {}", num_active_tiles);

    block_map.lookup.entries = -1;
    block_map.num_active_tiles = num_active_tiles;
    block_map.active_tiles = decltype(block_map.active_tiles)("active tiles", num_active_tiles);
    KView<u32*> active_tiles_view(active_tiles.data(), active_tiles.size());
    Kokkos::sort(active_tiles_view);
    Kokkos::fence();

    dex_parallel_for(
        "Setup active tiles",
        FlatLoop<1>(num_active_tiles),
        KOKKOS_LAMBDA (i32 idx) {
            u32 code = active_tiles_view(idx);
            block_map.active_tiles(idx) = code;
            Coord2 coord = decode_morton<num_dim>(code);
            block_map.lookup(coord) = idx;
        }
    );
    Kokkos::fence();
    state.mr_block_map.init(block_map, interface_config.max_mip_level);

    using dfp_t = Dex::fp_t;
    i64 num_active_cells = num_active_tiles * ::DexImpl::int_pow<num_dim>(block_size);
    state.atmos = SparseAtmosphere{
        .voxel_scale = dfp_t(sim.state.dx),
        .offset_x = dfp_t(sim.state.loc.x),
        .offset_y = FP(0.0),
        .offset_z = dfp_t(sim.state.loc.y),
        .num_x = num_x,
        .num_y = 0,
        .num_z = num_z,
        .moving = true,
        .temperature = yakl::Array<dfp_t, 1, yakl::memDevice>("temperature", num_active_cells),
        .pressure = yakl::Array<dfp_t, 1, yakl::memDevice>("pressure", num_active_cells),
        .ne = yakl::Array<dfp_t, 1, yakl::memDevice>("ne", num_active_cells),
        .nh_tot = yakl::Array<dfp_t, 1, yakl::memDevice>("nh_tot", num_active_cells),
        .nh0 = yakl::Array<dfp_t, 1, yakl::memDevice>("nh0", num_active_cells),
        .vturb = yakl::Array<dfp_t, 1, yakl::memDevice>("vturb", num_active_cells),
        .vx = yakl::Array<dfp_t, 1, yakl::memDevice>("vx", num_active_cells),
        .vy = yakl::Array<dfp_t, 1, yakl::memDevice>("vy", num_active_cells),
        .vz = yakl::Array<dfp_t, 1, yakl::memDevice>("vz", num_active_cells)
    };
    const auto& atmos = state.atmos;
    dex_parallel_for(
        "Copy atmos qtys",
        block_map.loop_bounds(),
        KOKKOS_LAMBDA (i64 tile_idx, i32 block_idx) {
            IdxGen idx_gen(block_map);
            i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
            Coord2 coord = idx_gen.loop_coord(tile_idx, block_idx);

            constexpr i32 n_hydro = N_HYDRO_VARS<num_dim>;
            CellIndex idx{.i = coord.x + sz.ng, .j = coord.z + sz.ng, .k = 0};
            yakl::SArray<fp_t, 1, n_hydro> w;
            QtyView q(Q, idx);
            cons_to_prim<num_dim>(eos.gamma, q, w);
            using Prim = Prim<num_dim>;

            atmos.pressure(ks) = w(I(Prim::Pres));
            const fp_t nh = w(I(Prim::Rho)) / (eos.avg_mass * m_p);
            atmos.nh_tot(ks) = nh;
            fp_t y = eos.y;
            if (!eos.is_constant) {
                y = eos.y_space(idx.k, idx.j, idx.i);
            }
            atmos.ne(ks) = atmos.nh_tot(ks) * y;
            const fp_t temperature = temperature_si(w(I(Prim::Pres)), nh, y);
            atmos.temperature(ks) = temperature;
            atmos.nh0(ks) = FP(0.0);
            atmos.vturb(ks) = vturb_fn(temperature, nh, y * nh);
            atmos.vx(ks) = w(I(Prim::Vx));
            atmos.vy(ks) = FP(0.0);
            atmos.vz(ks) = w(I(Prim::Vy));
        }
    );
    Kokkos::fence();


    allocate_cell_count_based_terms(state, num_active_cells);

    // TODO(cmo): We can reduce the amount of work performed here.
    // casc_state.init(state, state.config.max_cascade);
    const bool sparse_calc = state.config.sparse_calculation;
    CascadeStorage c0 = state.c0_size;
    std::vector<yakl::Array<i32, 2, yakl::memDevice>> active_probes;
    if (sparse_calc) {
        active_probes = compute_active_probe_lists(state, state.config.max_cascade);
    }
    casc_state.probes_to_compute.init(c0, sparse_calc, active_probes);
    casc_state.mip_chain.init(state, state.mr_block_map.buffer_len(), c0.wave_batch);

    if (interface_config.advect) {
        copy_pops_from_aux_fields(sim);
    }
    fmt::println("Update atmosphere at {:.3f} s", sim.time);

    return true;
}

bool DexInterface::init_atmosphere(Simulation& sim, i32 max_mip_level) {
    constexpr int num_dim = 2;
    constexpr fp_t m_p = ConstantsF64::u;

    auto& map = state.mr_block_map.block_map;
    const auto& sz = sim.state.sz;
    const auto& Q = sim.state.Q;
    const auto& eos = sim.eos;
    auto cutoff_temperature = state.config.threshold_temperature;

    constexpr i32 block_size = BLOCK_SIZE;
    const i32 num_x = sz.xc - 2 * sz.ng;
    // NOTE(cmo): z in dex is y in mosscap
    const i32 num_z = sz.yc - 2 * sz.ng;
    if (num_x % block_size != 0 || num_z % block_size != 0) {
        throw std::runtime_error("Inner grid is not a multiple of BLOCK_SIZE");
    }
    map.num_x_tiles() = num_x / block_size;
    map.num_z_tiles() = num_z / block_size;
    if (
        map.num_x_tiles() >= std::numeric_limits<u16>::max() ||
        map.num_z_tiles() >= std::numeric_limits<u16>::max()
    ) {
        throw std::runtime_error("Too many tiles for Morton code/overlaps with sentinel");
    }

    map.bbox.min = 0;
    map.bbox.max(0) = num_x;
    map.bbox.max(1) = num_z;

    map.lookup.init(Dims<2>{.x = map.num_x_tiles(), .z = map.num_z_tiles()});
    yakl::Array<u32, 2, yakl::memDevice> morton_order(
        "morton_traversal_order",
        map.num_z_tiles(),
        map.num_x_tiles()
    );
    yakl::Array<u32, 2, yakl::memDevice> active_2d(
        "active_2d",
        map.num_z_tiles(),
        map.num_x_tiles()
    );
    constexpr u32 sentinel = std::numeric_limits<u32>::max();
    active_2d = sentinel;
    Kokkos::fence();
    i32 num_active_tiles = 0;

    dex_parallel_reduce(
        "compute valid morton tiles",
        FlatLoop<2>(map.num_z_tiles(), map.num_x_tiles()),
        KOKKOS_LAMBDA (i32 zt, i32 xt, i32& num_active_tiles) {
            u32 code = encode_morton<2>(Coord2{.x = xt, .z = zt});
            morton_order(zt, xt) = code;

            constexpr int n_hydro = N_HYDRO_VARS<num_dim>;
            yakl::SArray<fp_t, 1, n_hydro> w;
            using Prim = Prim<num_dim>;

            for (int z = zt * block_size; z < (zt + 1) * block_size; ++z) {
                for (int x = xt * block_size; x < (xt + 1) * block_size; ++x) {
                    CellIndex idx{.i = x + sz.ng, .j = z + sz.ng, .k = 0};
                    const auto q = QtyView(Q, idx);
                    cons_to_prim<num_dim>(eos.gamma, q, w);

                    fp_t n_baryon = w(I(Prim::Rho)) / (eos.avg_mass * m_p);
                    fp_t y = eos.y;
                    if (!eos.is_constant) {
                        y = eos.y_space(idx.k, idx.j, idx.i);
                    }
                    auto temp = temperature_si(w(I(Prim::Pres)), n_baryon, y);
                    if (temp <= cutoff_temperature) {
                        num_active_tiles += 1;
                        active_2d(zt, xt) = code;
                        return;
                    }
                }
            }
        },
        Kokkos::Sum<i32>(num_active_tiles)
    );

    KView<u32*> morton_order_view(morton_order.data(), morton_order.size());
    Kokkos::sort(morton_order_view);
    KView<u32*> active_tiles_view(active_2d.data(), active_2d.size());
    Kokkos::sort(active_tiles_view);
    Kokkos::fence();

    map.num_active_tiles = num_active_tiles;
    map.morton_traversal_order = morton_order.reshape(morton_order.size());
    map.active_tiles = decltype(map.active_tiles)("active tiles", num_active_tiles);

    dex_parallel_for(
        "Setup active tiles",
        FlatLoop<1>(num_active_tiles),
        KOKKOS_LAMBDA (i32 idx) {
            u32 code = active_tiles_view(idx);
            map.active_tiles(idx) = code;
            Coord2 coord = decode_morton<num_dim>(code);
            map.lookup(coord) = idx;
        }
    );
    Kokkos::fence();

    state.mr_block_map.init(map, max_mip_level);

    using dfp_t = Dex::fp_t;
    i64 num_active_cells = num_active_tiles * ::DexImpl::int_pow<num_dim>(block_size);
    state.atmos = SparseAtmosphere{
        .voxel_scale = dfp_t(sim.state.dx),
        .offset_x = dfp_t(sim.state.loc.x),
        .offset_y = FP(0.0),
        .offset_z = dfp_t(sim.state.loc.y),
        .num_x = num_x,
        .num_y = 0,
        .num_z = num_z,
        .moving = true,
        .temperature = yakl::Array<dfp_t, 1, yakl::memDevice>("temperature", num_active_cells),
        .pressure = yakl::Array<dfp_t, 1, yakl::memDevice>("pressure", num_active_cells),
        .ne = yakl::Array<dfp_t, 1, yakl::memDevice>("ne", num_active_cells),
        .nh_tot = yakl::Array<dfp_t, 1, yakl::memDevice>("nh_tot", num_active_cells),
        .nh0 = yakl::Array<dfp_t, 1, yakl::memDevice>("nh0", num_active_cells),
        .vturb = yakl::Array<dfp_t, 1, yakl::memDevice>("vturb", num_active_cells),
        .vx = yakl::Array<dfp_t, 1, yakl::memDevice>("vx", num_active_cells),
        .vy = yakl::Array<dfp_t, 1, yakl::memDevice>("vy", num_active_cells),
        .vz = yakl::Array<dfp_t, 1, yakl::memDevice>("vz", num_active_cells)
    };
    const auto& atmos = state.atmos;
    dex_parallel_for(
        "Copy atmos qtys",
        map.loop_bounds(),
        KOKKOS_LAMBDA (i64 tile_idx, i32 block_idx) {
            IdxGen idx_gen(map);
            i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
            Coord2 coord = idx_gen.loop_coord(tile_idx, block_idx);

            constexpr i32 n_hydro = N_HYDRO_VARS<num_dim>;
            CellIndex idx{.i = coord.x + sz.ng, .j = coord.z + sz.ng, .k = 0};
            yakl::SArray<fp_t, 1, n_hydro> w;
            QtyView q(Q, idx);
            cons_to_prim<num_dim>(eos.gamma, q, w);
            using Prim = Prim<num_dim>;

            atmos.pressure(ks) = w(I(Prim::Pres));
            const fp_t nh = w(I(Prim::Rho)) / (eos.avg_mass * m_p);
            atmos.nh_tot(ks) = nh;
            fp_t y = eos.y;
            if (!eos.is_constant) {
                y = eos.y_space(idx.k, idx.j, idx.i);
            }
            atmos.ne(ks) = atmos.nh_tot(ks) * y;
            const fp_t temperature = temperature_si(w(I(Prim::Pres)), nh, y);
            atmos.temperature(ks) = temperature;
            atmos.nh0(ks) = FP(0.0);
            atmos.vturb(ks) = vturb_fn(temperature, nh, y * nh);
            atmos.vx(ks) = w(I(Prim::Vx));
            atmos.vy(ks) = FP(0.0);
            atmos.vz(ks) = w(I(Prim::Vy));
        }
    );
    Kokkos::fence();

    return true;
}

bool DexInterface::init_config(Simulation& sim, YAML::Node& cfg, const std::string& config_path) {
    auto dex_config = cfg["dex"];
    state.config = parse_dexrt_config(config_path, dex_config);

    setup_comm(&state);

    using dfp_t = Dex::fp_t;

    const auto& config = state.config;
    std::vector<ModelAtom<f64>> crtaf_models;
    crtaf_models.reserve(config.atom_paths.size());
    for (int i = 0; i < config.atom_paths.size(); ++i) {
        const auto& p = config.atom_paths[i];
        const auto& model_config = config.atom_configs[i];
        crtaf_models.emplace_back(parse_crtaf_model<f64>(p, model_config));
    }
    AtomicDataHostDevice<dfp_t> atomic_data = to_atomic_data<dfp_t, f64>(crtaf_models);
    state.adata = atomic_data.device;
    state.adata_host = atomic_data.host;
    state.have_h = atomic_data.have_h_model;
    state.atoms = extract_atoms(atomic_data.device, atomic_data.host);
    GammaAtomsAndMapping gamma_atoms = extract_atoms_with_gamma_and_mapping(atomic_data.device, atomic_data.host);
    state.atoms_with_gamma = gamma_atoms.atoms;
    state.atoms_with_gamma_mapping = gamma_atoms.mapping;

    interface_config.enable = true;
    return true;
}

bool DexInterface::init(Simulation& sim, YAML::Node& cfg) {
    auto dex_config = cfg["dex"];
    // state.config = parse_dexrt_config("mosscap", dex_config);

    // setup_comm(&state);

    using dfp_t = Dex::fp_t;

    const auto& config = state.config;
    // std::vector<ModelAtom<f64>> crtaf_models;
    // crtaf_models.reserve(config.atom_paths.size());
    // for (int i = 0; i < config.atom_paths.size(); ++i) {
    //     const auto& p = config.atom_paths[i];
    //     const auto& model_config = config.atom_configs[i];
    //     crtaf_models.emplace_back(parse_crtaf_model<f64>(p, model_config));
    // }
    // AtomicDataHostDevice<dfp_t> atomic_data = to_atomic_data<dfp_t, f64>(crtaf_models);
    // state.adata = atomic_data.device;
    // state.adata_host = atomic_data.host;
    // state.have_h = atomic_data.have_h_model;
    // state.atoms = extract_atoms(atomic_data.device, atomic_data.host);
    // GammaAtomsAndMapping gamma_atoms = extract_atoms_with_gamma_and_mapping(atomic_data.device, atomic_data.host);
    // state.atoms_with_gamma = gamma_atoms.atoms;
    // state.atoms_with_gamma_mapping = gamma_atoms.mapping;

    i32 max_mip_level = 0;
    for (int i = 0; i <= config.max_cascade; ++i) {
        max_mip_level = std::max(max_mip_level, config.mip_config.mip_levels[i]);
    }
    if (state.config.mode != DexrtMode::GivenFs && LINE_SCHEME == LineCoeffCalc::Classic) {
        max_mip_level = 0;
        state.println("Mips not supported with LineCoeffCalc::Classic");
    }
    interface_config.max_mip_level = max_mip_level;
    init_atmosphere(sim, max_mip_level);

    state.phi = VoigtProfile<dfp_t>();
    state.nh_lte = HPartFn();
    state.println("DexRT Scale: {} m", state.atmos.voxel_scale);


    // NOTE(cmo): We just have one of these chained for each boundary type -- they don't do anything if this configuration doesn't need them to.
    state.pw_bc = load_bc(config.atmos_path, state.adata.wavelength, config.boundary, PromweaverResampleType::FluxConserving);
    state.boundary = config.boundary;

    // NOTE(cmo): This doesn't actually know that things will be allocated sparse
    CascadeRays c0_rays;
    c0_rays.num_probes(0) = state.atmos.num_x;
    c0_rays.num_probes(1) = state.atmos.num_z;
    c0_rays.num_flat_dirs = PROBE0_NUM_RAYS;
    c0_rays.num_incl = NUM_INCL;
    c0_rays.wave_batch = WAVE_BATCH;
    constexpr int RcMode = RC_flags_storage_2d();
    state.c0_size = cascade_rays_to_storage<RcMode>(c0_rays);

    const auto& block_map = state.mr_block_map.block_map;
    state.max_block_mip = decltype(state.max_block_mip)(
        "max_block_mip",
        (state.adata.wavelength.extent(0) + c0_rays.wave_batch - 1) / c0_rays.wave_batch,
        block_map.num_z_tiles(),
        block_map.num_x_tiles()
    );
    state.max_block_mip = -1;

    yakl::Array<dfp_t, 1, yakl::memHost> muy("muy", NUM_INCL);
    yakl::Array<dfp_t, 1, yakl::memHost> wmuy("wmuy", NUM_INCL);
    for (int i = 0; i < NUM_INCL; ++i) {
        muy(i) = INCL_RAYS[i];
        wmuy(i) = INCL_WEIGHTS[i];
    }
    state.incl_quad.muy = muy.createDeviceCopy();
    state.incl_quad.wmuy = wmuy.createDeviceCopy();

    i64 num_active_cells = state.mr_block_map.get_num_active_cells();
    allocate_cell_count_based_terms(state, num_active_cells);
    casc_state.init(state, state.config.max_cascade);

    return true;
}

static void setup_wavelength_batch(const DexState& state, int la_start, int la_end) {
    if (state.config.store_J_on_cpu) {
        state.J = FP(0.0);
        Kokkos::fence();
    }
}

/// Called to copy J from GPU to plane of host array if config.store_J_on_cpu
static void copy_J_plane_to_host(const DexState& state, int la_start, int la_end) {
    int wave_batch = la_end - la_start;
    const auto J_copy = state.J.createHostCopy();
    // TODO(cmo): Replace with a memcpy?
    for (int wave = 0; wave < wave_batch; ++wave) {
        for (i64 ks = 0; ks < J_copy.extent(1); ++ks) {
            state.J_cpu(la_start + wave, ks) = J_copy(wave, ks);
        }
    }
}

static void finalise_wavelength_batch(const DexState& state, int la_start, int la_end) {
    if (state.config.store_J_on_cpu) {
        copy_J_plane_to_host(state, la_start, la_end);
    }

    const i32 wave_batch_idx = la_start / state.c0_size.wave_batch;
    JasUnpack(state, max_block_mip, mr_block_map);
    dex_parallel_for(
        "Copy max mip",
        FlatLoop<1>(state.mr_block_map.block_map.loop_bounds().dim(0)),
        YAKL_LAMBDA (i64 tile_idx) {
            MRIdxGen idx_gen(mr_block_map);
            Coord2 coord = idx_gen.loop_coord(0, tile_idx, 0);
            Coord2 tile_coord = idx_gen.compute_tile_coord(tile_idx);
            i32 mip_level = idx_gen.get_sample_level(coord);
            max_block_mip(wave_batch_idx, tile_coord.z, tile_coord.x) = mip_level;
        }
    );
    yakl::fence();
}

bool DexInterface::iterate(const DexConvergence& tol, bool first_iter) {
    JasUnpack(state, config);

    const bool conserve_charge = config.conserve_charge;
    const bool actually_conserve_charge = state.have_h && conserve_charge;
    if (!actually_conserve_charge && conserve_charge) {
        throw std::runtime_error("Charge conservation enabled without a model H!");
    }
    const bool conserve_pressure = config.conserve_pressure;
    if (conserve_pressure && !conserve_charge) {
        throw std::runtime_error("Cannot enable pressure conservation without charge conservation.");
    }
    const bool actually_conserve_pressure = actually_conserve_charge && conserve_pressure;
    const int initial_lambda_iterations = config.initial_lambda_iterations;
    const int max_iters = config.max_iter;

    auto& waves = state.adata_host.wavelength;
    WavelengthDistributor wave_dist;
    wave_dist.init(state.mpi_state, waves.extent(0), state.c0_size.wave_batch);

    int i = 0;
    if ((first_iter || !interface_config.advect) && actually_conserve_charge) {
        // TODO(cmo): Make all of these parameters configurable
        state.println("-- Iterating LTE n_e/pressure --");
        fp_t lte_max_change = FP(1.0);
        int lte_i = 0;
        while ((lte_max_change > FP(1e-5) || lte_i < 6) && lte_i < max_iters) {
            lte_i += 1;
            compute_nh0(state);
            compute_collisions_to_gamma(&state);
            lte_max_change = stat_eq(&state, StatEqOptions{
                .ignore_change_below_ntot_frac=FP(1e-7)
            });
            if (lte_i < 2) {
                continue;
            }
            // NOTE(cmo): Ignore what the lte_change actually is
            // from stat eq... it will "converge" essentially
            // instantly due to linearity, so whilst the error may
            // be above a threshold, it's unlikely to get
            // meaningfully better after the second iteration
            fp_t nr_update = nr_post_update(&state, NrPostUpdateOptions{
                .ignore_change_below_ntot_frac = FP(1e-7),
                .conserve_pressure = actually_conserve_pressure,
                .total_abund = FP(1.0),
            });
            lte_max_change = nr_update;
            // if (actually_conserve_pressure) {
            //     fp_t nh_tot_update = simple_conserve_pressure(&state);
            //     lte_max_change = std::max(nh_tot_update, lte_max_change);
            // }
        }
        state.println("Ran for {} iterations", lte_i);
        set_initial_pops_special(&state);
    }

    // state.println("-- Non-LTE Iterations ({} wavelengths) --", state.adata_host.wavelength.extent(0));
    NgAccelerator ng;
    if (config.ng.enable) {
        ng.init(
            NgAccelArgs{
                .num_level=(i64)state.pops.extent(0),
                .num_space=(i64)state.pops.extent(1),
                .accel_tol=config.ng.threshold,
                .lower_tol=config.ng.lower_threshold
            }
        );
        ng.accelerate(state, FP(1.0));
    }
    bool first_inner_iter = true;
    bool accelerated = false;
    fp_t max_change = 1.0_fp;
    while (((max_change > tol.convergence || i < (initial_lambda_iterations+1)) && i < max_iters) || accelerated) {
        state.println("==== FS {} ====", i);
        compute_nh0(state);

        if (state.mpi_state.rank == 0) {
            compute_collisions_to_gamma(&state);
        } else {
            for (int ia = 0; ia < state.Gamma.size(); ++ia) {
                state.Gamma[ia] = FP(0.0);
            }
            yakl::fence();
        }

        bool print_worst_wphi = first_inner_iter;
        compute_profile_normalisation(state, casc_state, print_worst_wphi);
        state.J = FP(0.0);
        if (config.store_J_on_cpu) {
            state.J_cpu = FP(0.0);
        }
        yakl::fence();
        WavelengthBatch wave_batch;
        wave_dist.wait_for_all(state.mpi_state);
        wave_dist.reset();
        while (wave_dist.next_batch(state.mpi_state, &wave_batch)) {
            setup_wavelength_batch(state, wave_batch.la_start, wave_batch.la_end);
            bool lambda_iterate = i < initial_lambda_iterations;
            dynamic_formal_sol_rc(
                state,
                casc_state,
                lambda_iterate,
                wave_batch.la_start,
                wave_batch.la_end
            );
            finalise_wavelength_batch(state, wave_batch.la_start, wave_batch.la_end);
        }
        yakl::fence();
        wave_dist.wait_for_all(state.mpi_state);

        state.println("  == Statistical equilibrium ==");
        wave_dist.reduce_Gamma(&state);
        max_change = stat_eq(
            &state,
            StatEqOptions{
                .ignore_change_below_ntot_frac=std::min(FP(1e-6), tol.convergence)
            }
        );
        if (actually_conserve_charge) {
            fp_t nr_update = nr_post_update(
                &state,
                NrPostUpdateOptions{
                    .ignore_change_below_ntot_frac = std::min(FP(1e-6), tol.convergence),
                    .conserve_pressure = actually_conserve_pressure,
                    .total_abund = FP(1.0)
                }
            );
            wave_dist.update_ne(&state);
            max_change = std::max(nr_update, max_change);
            if (actually_conserve_pressure) {
                wave_dist.update_nh_tot(&state);
            }
        }
        if (config.ng.enable) {
            accelerated = ng.accelerate(state, max_change);
            if (accelerated) {
                state.println("  ~~ Ng Acceleration! (📉 or 💣) ~~");
            }
        }
        i += 1;
        first_inner_iter = false;
    }

    if (first_iter) {
        config.conserve_pressure = false;
    }
    num_iter = i;

    return true;
}

/// Add Dex's metadata to the file using attributes. The netcdf layer needs extending to do this, so I'm just throwing it in manually.
void add_netcdf_attributes(const DexState& state, const yakl::SimpleNetCDF& file) {
    const auto ncwrap = [&] (int ierr, int line) {
        if (ierr != NC_NOERR) {
            state.println("NetCDF Error writing attributes at main.cpp:{}", line);
            state.println("{}",nc_strerror(ierr));
            yakl::yakl_throw(nc_strerror(ierr));
        }
    };
    int ncid = file.file.ncid;
    if (ncid == -999) {
        throw std::runtime_error("File appears to have been closed before writing attributes!");
    }

    std::string name = "dexrt (2d)";
    ncwrap(
        nc_put_att_text(ncid, NC_GLOBAL, "program", name.size(), name.c_str()),
        __LINE__
    );

    std::string precision = "f64";
#ifdef DEXRT_SINGLE_PREC
    precision = "f32";
#endif
    ncwrap(
        nc_put_att_text(ncid, NC_GLOBAL, "rt_precision", precision.size(), precision.c_str()),
        __LINE__
    );
    std::string method(RcConfigurationNames[int(RC_CONFIG)]);
    ncwrap(
        nc_put_att_text(ncid, NC_GLOBAL, "RC_method", method.size(), method.c_str()),
        __LINE__
    );

    if (RC_CONFIG == RcConfiguration::ParallaxFixInner) {
        i32 inner_parallax_merge_lim = INNER_PARALLAX_MERGE_ABOVE_CASCADE;
        ncwrap(
            nc_put_att_int(ncid, NC_GLOBAL, "inner_parallax_merge_above_cascade", NC_INT, 1, &inner_parallax_merge_lim),
            __LINE__
        );
    }
    if (RC_CONFIG == RcConfiguration::ParallaxFix) {
        i32 parallax_merge_lim = PARALLAX_MERGE_ABOVE_CASCADE;
        ncwrap(
            nc_put_att_int(ncid, NC_GLOBAL, "parallax_merge_above_cascade", NC_INT, 1, &parallax_merge_lim),
            __LINE__
        );
    }

    std::string raymarch_type(RaymarchTypeNames[int(RAYMARCH_TYPE)]);
    ncwrap(
        nc_put_att_text(ncid, NC_GLOBAL, "raymarch_type", raymarch_type.size(), raymarch_type.c_str()),
        __LINE__
    );
    if (RAYMARCH_TYPE == RaymarchType::LineSweep) {
        i32 line_sweep_on_and_above = LINE_SWEEP_START_CASCADE;
        ncwrap(
            nc_put_att_int(ncid, NC_GLOBAL, "line_sweep_start_cascade", NC_INT, 1, &line_sweep_on_and_above),
            __LINE__
        );
    }

    f64 probe0_length = PROBE0_LENGTH;
    ncwrap(
        nc_put_att_double(ncid, NC_GLOBAL, "probe0_length", NC_DOUBLE, 1, &probe0_length),
        __LINE__
    );
    i32 probe0_num_rays = PROBE0_NUM_RAYS;
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "probe0_num_rays", NC_INT, 1, &probe0_num_rays),
        __LINE__
    );
    i32 probe0_spacing = PROBE0_SPACING;
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "probe0_spacing", NC_INT, 1, &probe0_spacing),
        __LINE__
    );
    i32 cascade_branching = CASCADE_BRANCHING_FACTOR;
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "cascade_branching_factor", NC_INT, 1, &cascade_branching),
        __LINE__
    );
    i32 multiple_branching_factors = VARY_BRANCHING_FACTOR;
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "multiple_branching_factors", NC_INT, 1, &multiple_branching_factors),
        __LINE__
    );
    if (VARY_BRANCHING_FACTOR) {
        i32 upper_branching = UPPER_BRANCHING_FACTOR;
        ncwrap(
            nc_put_att_int(ncid, NC_GLOBAL, "upper_branching_factor", NC_INT, 1, &upper_branching),
            __LINE__
        );
        i32 branch_switch = BRANCHING_FACTOR_SWITCH;
        ncwrap(
            nc_put_att_int(ncid, NC_GLOBAL, "branching_factor_switch", NC_INT, 1, &branch_switch),
            __LINE__
        );
    }
    i32 max_cascade = state.config.max_cascade;
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "max_cascade", NC_INT, 1, &max_cascade),
        __LINE__
    );
    i32 last_casc_to_inf = LAST_CASCADE_TO_INFTY;
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "last_casc_to_infty", NC_INT, 1, &last_casc_to_inf),
        __LINE__
    );
    f64 last_casc_dist = LAST_CASCADE_MAX_DIST;
    ncwrap(
        nc_put_att_double(ncid, NC_GLOBAL, "last_cascade_max_distance", NC_DOUBLE, 1, &last_casc_dist),
        __LINE__
    );
    i32 preaverage = PREAVERAGE;
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "preaverage", NC_INT, 1, &preaverage),
        __LINE__
    );
    i32 dir_by_dir = DIR_BY_DIR;
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "dir_by_dir", NC_INT, 1, &dir_by_dir),
        __LINE__
    );
    i32 pingpong = PINGPONG_BUFFERS;
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "pingpong_buffers", NC_INT, 1, &pingpong),
        __LINE__
    );
    i32 store_tau_cascades = STORE_TAU_CASCADES;
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "store_tau_cascades", NC_INT, 1, &store_tau_cascades),
        __LINE__
    );
    f64 thermal_vel_frac = ANGLE_INVARIANT_THERMAL_VEL_FRAC;
    ncwrap(
        nc_put_att_double(ncid, NC_GLOBAL, "angle_invariant_thermal_vel_frac", NC_DOUBLE, 1, &thermal_vel_frac),
        __LINE__
    );
    i32 conserve_pressure_nr = CONSERVE_PRESSURE_NR;
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "conserve_pressure_nr", NC_INT, 1, &conserve_pressure_nr),
        __LINE__
    );
    i32 extra_safe_source_fn = EXTRA_SAFE_SOURCE_FN;
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "extra_safe_source_fn", NC_INT, 1, &extra_safe_source_fn),
        __LINE__
    );
    i32 report_nan_intensity = REPORT_NAN_INTENSITY;
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "report_nan_intensity", NC_INT, 1, &report_nan_intensity),
        __LINE__
    );


    i32 warp_size = DEXRT_WARP_SIZE;
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "warp_size", NC_INT, 1, &warp_size),
        __LINE__
    );
    i32 wave_batch = WAVE_BATCH;
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "wave_batch", NC_INT, 1, &wave_batch),
        __LINE__
    );
    i32 num_incl = NUM_INCL;
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "num_incl", NC_INT, 1, &num_incl),
        __LINE__
    );
    f64 incl_rays[NUM_INCL];
    f64 incl_weights[NUM_INCL];
    for (int i = 0; i < NUM_INCL; ++i) {
        incl_rays[i] = INCL_RAYS[i];
        incl_weights[i] = INCL_WEIGHTS[i];
    }
    ncwrap(
        nc_put_att_double(ncid, NC_GLOBAL, "incl_rays", NC_DOUBLE, num_incl, incl_rays),
        __LINE__
    );
    ncwrap(
        nc_put_att_double(ncid, NC_GLOBAL, "incl_weights", NC_DOUBLE, num_incl, incl_weights),
        __LINE__
    );
    i32 num_atom = state.adata_host.num_level.extent(0);
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "num_atom", NC_INT, 1, &num_atom),
        __LINE__
    );
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "num_level", NC_INT, num_atom, state.adata_host.num_level.get_data()),
        __LINE__
    );
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "num_line", NC_INT, state.adata_host.num_line.extent(0), state.adata_host.num_line.get_data()),
        __LINE__
    );
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "line_start", NC_INT, state.adata_host.line_start.extent(0), state.adata_host.line_start.get_data()),
        __LINE__
    );
    yakl::Array<f64, 1, yakl::memHost> lambda0("lambda0", state.adata_host.lines.extent(0));
    for (int i = 0; i < lambda0.extent(0); ++i) {
        lambda0(i) = state.adata_host.lines(i).lambda0;
    }
    ncwrap(
        nc_put_att_double(ncid, NC_GLOBAL, "lambda0", NC_DOUBLE, lambda0.extent(0), lambda0.get_data()),
        __LINE__
    );

    // NOTE(cmo): Hack to save timing data. These functions only print to stdout -- want to redirect that.
    auto cout_buf = std::cout.rdbuf();
    std::ostringstream timer_buffer;
    std::cout.rdbuf(timer_buffer.rdbuf());
    yakl::timer_finalize();
    std::cout.rdbuf(cout_buf);
    std::string timer_data = timer_buffer.str();
    ncwrap(
        nc_put_att_text(ncid, NC_GLOBAL, "timing", timer_data.size(), timer_data.c_str()),
        __LINE__
    );
    // ncwrap(
    //     nc_put_att_int(ncid, NC_GLOBAL, "num_iter", NC_INT, 1, &num_iter),
    //     __LINE__
    // );

    std::string output_format = state.config.output.sparse ? "sparse" : "full";
    ncwrap(
        nc_put_att_text(ncid, NC_GLOBAL, "output_format", output_format.size(), output_format.c_str()),
        __LINE__
    );
    i32 final_dense_fs = state.config.final_dense_fs;
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "final_dense_fs", NC_INT, 1, &final_dense_fs),
        __LINE__
    );

    std::string line_scheme_name(LineCoeffCalcNames[int(LINE_SCHEME)]);
    ncwrap(
        nc_put_att_text(ncid, NC_GLOBAL, "line_calculation_scheme", line_scheme_name.size(), line_scheme_name.c_str()),
        __LINE__
    );

    i32 block_size = BLOCK_SIZE;
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "block_size", NC_INT, 1, &block_size),
        __LINE__
    );
    i32 nx_blocks = state.mr_block_map.block_map.num_x_tiles();
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "num_x_blocks", NC_INT, 1, &nx_blocks),
        __LINE__
    );
    i32 nz_blocks = state.mr_block_map.block_map.num_z_tiles();
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "num_z_blocks", NC_INT, 1, &nz_blocks),
        __LINE__
    );

    if constexpr (LINE_SCHEME == LineCoeffCalc::VelocityInterp) {
        i32 interp_bins = INTERPOLATE_DIRECTIONAL_BINS;
        ncwrap(
            nc_put_att_int(ncid, NC_GLOBAL, "interpolate_directional_bins", NC_INT, 1, &interp_bins),
            __LINE__
        );

        f64 interp_max_width = INTERPOLATE_DIRECTIONAL_MAX_THERMAL_WIDTH;
        ncwrap(
            nc_put_att_double(ncid, NC_GLOBAL, "interpolate_direction_max_thermal_width", NC_DOUBLE, 1, &interp_max_width),
            __LINE__
        );
    }

    ncwrap(
        nc_put_att_int(
            ncid, NC_GLOBAL, "mip_levels", NC_INT,
            state.config.max_cascade+1,
            state.config.mip_config.mip_levels.data()
        ),
        __LINE__
    );

    // std::string git_hash(GIT_HASH);
    // ncwrap(
    //     nc_put_att_text(ncid, NC_GLOBAL, "git_hash", git_hash.size(), git_hash.c_str()),
    //     __LINE__
    // );

    f64 voxel_scale = state.atmos.voxel_scale;
    ncwrap(
        nc_put_att_double(ncid, NC_GLOBAL, "voxel_scale", NC_DOUBLE, 1, &voxel_scale),
        __LINE__
    );

    const auto& config_path(state.config.own_path);
    ncwrap(
        nc_put_att_text(ncid, NC_GLOBAL, "config_path", config_path.size(), config_path.c_str()),
        __LINE__
    );
}

void save_results(const DexState& state, yakl::SimpleNetCDF& nc, bool single_file, i32 num_iter) {
    const auto& config = state.config;
    const auto& out_cfg = config.output;
    if (state.mpi_state.rank != 0) {
        return;
    }

    i32 time_idx = 0;
    if (single_file) {
        // NOTE(cmo): This is called after mosscap has already extended the time axis
        time_idx = nc.getDimSize("time") - 1;
        time_idx = std::max(time_idx, 0);
    }
    const auto& block_map = state.mr_block_map.block_map;

    if (single_file) {
        nc.write1(num_iter, "dex_num_iter", time_idx, "time");
    } else {
        nc.write(num_iter, "dex_num_iter");
    }

    bool sparse_J = state.config.sparse_calculation && (state.J.extent(1) == state.atmos.temperature.extent(0));
    auto convert_name = [&](const std::string& name) {
        if (single_file) {
            return fmt::format("{}_{}", name, time_idx);
        }
        return name;
    };

    auto maybe_rehydrate_and_write = [&](
        auto arr,
        const std::string& name,
        std::vector<std::string> leading_dim_names
    ) {
        auto& dim_names = leading_dim_names;
        if (out_cfg.sparse) {
            dim_names.insert(dim_names.end(), {convert_name("ks")});
            nc.write(arr, name, dim_names);
        } else {
            auto hydrated = rehydrate_sparse_quantity(block_map, arr);
            dim_names.insert(dim_names.end(), {"z_dex", "x_dex"});
            nc.write(hydrated, name, dim_names);
        }
    };

    if (out_cfg.J) {
        if (config.store_J_on_cpu) {
            if (sparse_J) {
                maybe_rehydrate_and_write(state.J_cpu, convert_name("J"), {"wavelength"});
            } else {
                auto J_full = state.J_cpu.reshape(state.J_cpu.extent(0), block_map.num_z_tiles() * BLOCK_SIZE, block_map.num_x_tiles() * BLOCK_SIZE);
                nc.write(J_full, convert_name("J"), {"wavelength", "z_dex", "x_dex"});
            }
        } else {
            if (sparse_J) {
                maybe_rehydrate_and_write(state.J, convert_name("J"), {"wavelength"});
            } else {
                auto J_full = state.J.reshape(state.J.extent(0), block_map.num_z_tiles() * BLOCK_SIZE, block_map.num_x_tiles() * BLOCK_SIZE);
                nc.write(J_full, convert_name("J"), {"wavelength", "z_dex", "x_dex"});
            }
        }
        nc.write(state.max_block_mip, convert_name("max_mip_block"), {"wavelength_batch", "tile_z", "tile_x"});
    }

    if (out_cfg.wavelength && state.adata.wavelength.initialized()) {
        nc.write(state.adata.wavelength, "wavelength", {"wavelength"});
    }
    if (out_cfg.pops && state.pops.initialized()) {
        maybe_rehydrate_and_write(state.pops, convert_name("pops"), {"level"});
    }
    if (out_cfg.lte_pops) {
        auto lte_pops = state.pops.createDeviceObject();
        compute_lte_pops(&state, lte_pops);
        yakl::fence();
        maybe_rehydrate_and_write(lte_pops, convert_name("lte_pops"), {"level"});
    }
    if (out_cfg.ne && state.atmos.ne.initialized()) {
        maybe_rehydrate_and_write(state.atmos.ne, convert_name("ne"), {});
    }
    if (out_cfg.nh_tot && state.atmos.nh_tot.initialized()) {
        maybe_rehydrate_and_write(state.atmos.nh_tot, convert_name("nh_tot"), {});
        maybe_rehydrate_and_write(state.atmos.temperature, convert_name("temperature"), {});
    }
    // if (out_cfg.psi_star && casc_state.psi_star.initialized()) {
    //     nc.write(casc_state.psi_star, convert_name("psi_star"), {"casc_shape"});
    // }
    if (out_cfg.active) {
        // NOTE(cmo): Currently active is always written dense
        const auto& active_char = reify_active_c0(block_map);
        nc.write(active_char, convert_name("active"), {"z_dex", "x_dex"});
    }
    // for (int casc : out_cfg.cascades) {
    //     // NOTE(cmo): The validity of these + necessary warning were checked/output in the config parsing step
    //     std::string name = fmt::format("I_C{}", casc);
    //     std::string shape = fmt::format("casc_shape_{}", casc);
    //     nc.write(casc_state.i_cascades[casc], name, {shape});
    //     if constexpr (STORE_TAU_CASCADES) {
    //         name = fmt::format("tau_C{}", casc);
    //         nc.write(casc_state.tau_cascades[casc], name, {shape});
    //     }
    // }
    if (out_cfg.sparse) {
        nc.write(block_map.active_tiles, convert_name("morton_tiles"), {convert_name("num_active_tiles")});
    }
}

void DexInterface::write_output(const Simulation& sim, yakl::SimpleNetCDF& nc) {
    const auto& cfg = sim.out_cfg;
    if (cfg.prev_output_time < 0.0_fp || !cfg.single_file) {
        add_netcdf_attributes(state, nc);
    }
    save_results(state, nc, cfg.single_file, num_iter);
}

void DexInterface::copy_nhtot_to_rho(const Simulation& sim) {
    if (!interface_config.enable) {
        return;
    }

    JasUnpack(state, mr_block_map, atmos);
    const auto& block_map = mr_block_map.block_map;
    const auto& Q = sim.state.Q;
    const auto& sz = sim.state.sz;

    constexpr fp_t m_p = ConstantsF64::u;
    const auto& eos = sim.eos;
    constexpr i32 num_dim = 2;
    using Cons = Cons<num_dim>;

    dex_parallel_for(
        "nhtot -> rho",
        FlatLoop<2>(block_map.loop_bounds()),
        KOKKOS_LAMBDA (i64 tile_idx, i32 block_idx) {
            IdxGen idx_gen(mr_block_map);
            const i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
            Coord2 coord = idx_gen.loop_coord(tile_idx, block_idx);
            CellIndex idx{.i = coord.x + sz.ng, .j = coord.z + sz.ng, .k = 0};

            Q(I(Cons::Rho), idx.k, idx.j, idx.i) = atmos.nh_tot(ks) * eos.avg_mass * m_p;
        }
    );
    Kokkos::fence();
}

void DexInterface::copy_pops_to_aux_fields(const Simulation& sim) {
    if (!interface_config.advect || !interface_config.enable) {
        return;
    }

    JasUnpack(state, mr_block_map, atmos, pops);
    const auto& block_map = mr_block_map.block_map;
    const auto& Q = sim.state.Q;
    const auto& sz = sim.state.sz;

    const i32 start_idx = interface_config.field_start_idx;
    dex_parallel_for(
        "Pops -> Tracers",
        FlatLoop<2>(block_map.loop_bounds()),
        KOKKOS_LAMBDA (i64 tile_idx, i32 block_idx) {
            IdxGen idx_gen(mr_block_map);
            const i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
            Coord2 coord = idx_gen.loop_coord(tile_idx, block_idx);
            CellIndex idx{.i = coord.x + sz.ng, .j = coord.z + sz.ng, .k = 0};

            Q(start_idx, idx.k, idx.j, idx.i) = atmos.ne(ks);
            for (int v = start_idx + 1; v < Q.extent(0); ++v) {
                Q(v, idx.k, idx.j, idx.i) = pops(v - (start_idx + 1), ks);
            }
        }
    );
    Kokkos::fence();
}

void DexInterface::copy_pops_from_aux_fields(const Simulation& sim) {
    if (!interface_config.advect || !interface_config.enable) {
        return;
    }

    JasUnpack(state, mr_block_map, atmos, pops);
    const auto& block_map = mr_block_map.block_map;
    const auto& Q = sim.state.Q;
    const auto& sz = sim.state.sz;

    const i32 start_idx = interface_config.field_start_idx;
    dex_parallel_for(
        "Tracers -> Pops",
        FlatLoop<2>(block_map.loop_bounds()),
        KOKKOS_LAMBDA (i64 tile_idx, i32 block_idx) {
            IdxGen idx_gen(mr_block_map);
            const i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
            Coord2 coord = idx_gen.loop_coord(tile_idx, block_idx);
            CellIndex idx{.i = coord.x + sz.ng, .j = coord.z + sz.ng, .k = 0};

            atmos.ne(ks) = Q(start_idx, idx.k, idx.j, idx.i);
            for (int v = start_idx + 1; v < Q.extent(0); ++v) {
                pops(v - (start_idx + 1), ks) = Q(v, idx.k, idx.j, idx.i);
            }
        }
    );
    Kokkos::fence();

    using dfp_t = Dex::fp_t;
    for (int ia = 0; ia < state.atoms.size(); ++ia) {
        const dfp_t abundance = state.adata_host.abundance(ia);
        const i32 Z = state.adata_host.Z(ia);
        const auto& nh_tot = state.atmos.nh_tot;
        const i32 pops_start = state.adata_host.level_start(ia);
        const i32 num_level = state.adata_host.num_level(ia);

        dex_parallel_for(
            "Rescale pops",
            block_map.loop_bounds(),
            KOKKOS_LAMBDA (i64 tile_idx, i32 block_idx) {
                IdxGen idx_gen(mr_block_map);
                const i64 ks = idx_gen.loop_idx(tile_idx, block_idx);

                const dfp_t n_total_k = abundance * nh_tot(ks);
                dfp_t n_sum = FP(0.0);
                for (int i = 0; i < num_level; ++i) {
                    n_sum += pops(pops_start + i, ks);
                }
                const dfp_t ratio = n_total_k / n_sum;
                for (int i = 0; i < num_level; ++i) {
                    pops(pops_start + i, ks) *= ratio;
                }
                if (Z == 1) {
                    atmos.ne(ks) *= ratio;
                }
            }
        );
    }
    Kokkos::fence();
}

void DexInterface::lte_init_aux_fields(const Simulation& sim) {
    if (!interface_config.advect || !interface_config.enable) {
        return;
    }

    constexpr fp_t m_p = ConstantsF64::u;
    constexpr i32 num_dim = 2;

    const auto& Q = sim.state.Q;
    const auto& sz = sim.state.sz;
    const auto& eos = sim.eos;

    const i32 tracer_start = interface_config.field_start_idx;
    for (int ia = 0; ia < state.atoms.size(); ++ia) {
        const auto& atom = state.atoms[ia];
        const auto flat_pops = std::remove_cvref_t<decltype(Q)>(
            "flat_tracer_pops",
            &Q(tracer_start + 1, 0, 0, 0),
            atom.energy.size(),
            Q.extent(1),
            Q.extent(2),
            Q.extent(3)
        ).reshape(atom.energy.size(), Q.extent(1)*Q.extent(2)*Q.extent(3));

        dex_parallel_for(
            "LTE tracers",
            FlatLoop<3>(sz.zc, sz.yc, sz.xc),
            KOKKOS_LAMBDA (i32 k, i32 j, i32 i) {
                constexpr i32 n_hydro = N_HYDRO_VARS<num_dim>;
                CellIndex idx{.i = i, .j = j, .k = k};
                yakl::SArray<fp_t, 1, n_hydro> w;
                QtyView q(Q, idx);
                cons_to_prim<num_dim>(eos.gamma, q, w);
                using Prim = Prim<num_dim>;

                const i64 flat_idx = i + j * sz.xc + k * sz.yc * sz.xc;

                const fp_t pressure = w(I(Prim::Pres));
                const fp_t nh = w(I(Prim::Rho)) / (eos.avg_mass * m_p);
                fp_t y = eos.y;
                if (!eos.is_constant) {
                    y = eos.y_space(idx.k, idx.j, idx.i);
                }
                const fp_t ne = nh * y;
                const fp_t temperature = temperature_si(w(I(Prim::Pres)), nh, y);

                Q(tracer_start, k, j, i) = ne;
                lte_pops(
                    atom.energy,
                    atom.g,
                    atom.stage,
                    temperature,
                    ne,
                    nh,
                    flat_pops,
                    flat_idx
                );
            }
        );
    }
}

}