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

// TODO(cmo): Figure out how to deal with 3D down the line.
int get_dexrt_dimensionality() {
    return 2;
}

namespace Mosscap {

// NOTE(cmo): Direct transplant from dexrt
static void allocate_J(DexState* state) {
    JasUnpack((*state), config, mr_block_map, c0_size, adata);
    const auto& block_map = mr_block_map.block_map;
    const bool sparse = config.sparse_calculation;
    i64 num_cells = mr_block_map.block_map.get_num_active_cells();
    i32 wave_dim = adata.wavelength.extent(0);
    if (config.mode == DexrtMode::GivenFs) {
        wave_dim = state->given_state.emis.extent(2);
    }

    if (!sparse) {
        num_cells = i64(block_map.num_x_tiles()) * block_map.num_z_tiles() * square(BLOCK_SIZE);
    }

    if (config.store_J_on_cpu) {
        state->J = decltype(state->J)("J", yakl::DimsT<i64>(c0_size.wave_batch, num_cells));
        state->J_cpu = decltype(state->J_cpu)("JHost", yakl::DimsT<i64>(wave_dim, num_cells));
    } else {
        state->J = decltype(state->J)("J", yakl::DimsT<i64>(wave_dim, num_cells));
    }
    state->J = 0;
    // TODO(cmo): If we have scattering terms and are updating J, the old
    // contents should probably be moved first, but we don't have these terms yet.
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
    if (map.num_x_tiles() >= 65535 || map.num_z_tiles() >= 65535) {
        throw std::runtime_error("Too many tiles for Morton code/overlaps with sentinel");
    }

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
                    CellIndex idx{.i = x, .j = z, .k = 0};
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
            CellIndex idx{.i = coord.x, .j = coord.z, .k = 0};
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
}

bool DexInterface::init(Simulation& sim, YAML::Node& cfg) {
    auto dex_config = cfg["dex"];
    state.config = parse_dexrt_config("mosscap", dex_config);

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

    i32 max_mip_level = 0;
    for (int i = 0; i <= config.max_cascade; ++i) {
        max_mip_level = std::max(max_mip_level, config.mip_config.mip_levels[i]);
    }
    if (state.config.mode != DexrtMode::GivenFs && LINE_SCHEME == LineCoeffCalc::Classic) {
        max_mip_level = 0;
        state.println("Mips not supported with LineCoeffCalc::Classic");
    }
    // TODO(cmo): Atmos.
    // TODO(cmo): How are we handling num_active_cells for allocations? Do we
    // assume only some fraction by default?
    // Allocate only what we need each step then set up a new BlockMap and migrate bits

    state.phi = VoigtProfile<dfp_t>();
    state.nh_lte = HPartFn();
    state.println("DexRT Scale: {} m", state.atmos.voxel_scale);

    i64 num_active_cells = state.mr_block_map.get_num_active_cells();

    const int n_level_total = state.adata.energy.extent(0);
    state.pops = decltype(state.pops)("pops", n_level_total, num_active_cells);
    if (config.mode == DexrtMode::NonLte) {
        for (int ia = 0; ia < state.adata_host.num_level.extent(0); ++ia) {
            const int n_level = state.adata_host.num_level(ia);
            state.Gamma.emplace_back(
                decltype(state.Gamma)::value_type("Gamma", n_level, n_level, num_active_cells)
            );
        }
        state.wphi = decltype(state.wphi)("wphi", state.adata.lines.extent(0), num_active_cells);
    }

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

    yakl::Array<dfp_t, 1, yakl::memHost> muy("muy", NUM_INCL);
    yakl::Array<dfp_t, 1, yakl::memHost> wmuy("wmuy", NUM_INCL);
    for (int i = 0; i < NUM_INCL; ++i) {
        muy(i) = INCL_RAYS[i];
        wmuy(i) = INCL_WEIGHTS[i];
    }
    state.incl_quad.muy = muy.createDeviceCopy();
    state.incl_quad.wmuy = wmuy.createDeviceCopy();

    allocate_J(&state);

    // TODO: setup cascstate
    // KView<dfp_t*> a(state.atmos.temperature.data(), state.atmos.temperature.size());
    // Kokkos::sort(a);

    return true;
}

}