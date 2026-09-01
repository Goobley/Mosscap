#define JASNAH_NO_FIF
#include "DexInterface.hpp"
#include "Simulation.hpp"
#include "JasPP.hpp"
#include "TimeDepPopulations.hpp"
#include "MosscapConfig.hpp"
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
#include "../DexRT/source/EnergyConservation.hpp"
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

template <typename Lambda, typename ...Args>
static auto invoke_fluid_traits_2d(
    int num_dim,
    FluidType fluid_type,
    Lambda&& fn,
    Args&&... args)
-> decltype(fn(FluidTraits<2, FluidType::Hydro>{}, std::forward<Args>(args)...)) {
    auto invoke_dim = [...args = std::forward<Args>(args), fn, num_dim]<FluidType FType>(FluidType fluid_type) {
        // switch (num_dim) {
            // case 1: {
            //     return fn(FluidTraits<1, FType>{}, std::forward<Args>(args)...);
            // };
            // case 2: {
            //     return fn(FluidTraits<2, FType>{}, std::forward<Args>(args)...);
            // };
            // case 3: {
            //     return fn(FluidTraits<3, FType>{}, std::forward<Args>(args)...);
            // };
        // }
        return fn(FluidTraits<2, FType>{}, std::forward<Args>(args)...);
    };

    switch (fluid_type) {
        case FluidType::Hydro: {
            return invoke_dim.template operator()<FluidType::Hydro>(FluidType::Hydro);
        };
        case FluidType::Mhd: {
            return invoke_dim.template operator()<FluidType::Mhd>(FluidType::Mhd);
        };
        case FluidType::MhdHyperTc: {
            return invoke_dim.template operator()<FluidType::MhdHyperTc>(FluidType::MhdHyperTc);
        };
        case FluidType::GlmMhd: {
            return invoke_dim.template operator()<FluidType::GlmMhd>(FluidType::GlmMhd);
        };
        case FluidType::GlmMhdHyperTc: {
            return invoke_dim.template operator()<FluidType::GlmMhdHyperTc>(FluidType::GlmMhdHyperTc);
        };
    }
    return invoke_dim.template operator()<FluidType::Hydro>(FluidType::Hydro);
}

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

void allocate_rad_loss(DexState& state) {
    JasUnpack(state, config, mr_block_map, c0_size, adata);
    if (config.rad_loss == RadLossType::None) {
        return;
    }
    i64 num_cells = mr_block_map.block_map.get_num_active_cells();
    i32 wave_dim = adata.wavelength.extent(0);

    int rad_loss_leading_dim = wave_dim;
    if (config.store_J_on_cpu) {
        rad_loss_leading_dim = c0_size.wave_batch;
    }
    int rad_loss_cpu_leading_dim = wave_dim;
    if (config.rad_loss == RadLossType::Integrated) {
        rad_loss_leading_dim = 1;
        rad_loss_cpu_leading_dim = 1;
    }

    state.rad_loss = decltype(state.rad_loss)("L", yakl::DimsT<i64>(rad_loss_leading_dim, num_cells));
    if (config.store_J_on_cpu) {
        state.rad_loss_cpu = decltype(state.rad_loss_cpu)("LHost", yakl::DimsT<i64>(rad_loss_cpu_leading_dim, num_cells));
        state.rad_loss_cpu = FP(0.0);
    }
    state.rad_loss = FP(0.0);
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
    state.rate_diag.allocate(state.adata_host, state.config.output, num_active_cells);
    state.wphi = decltype(state.wphi)("wphi", state.adata.lines.extent(0), num_active_cells);

    // TODO(cmo): Maybe no J unless requested.
    allocate_J(state);
    allocate_rad_loss(state);
}

DexInterface::~DexInterface() {
#ifdef HAVE_MPI
    if (state.mpi_state.rank == 0) {
        int should_continue = false;
        MPI_Bcast(&should_continue, 1, MPI_INT, 0, state.mpi_state.comm);
    }
#endif
}

void DexInterface::broadcast_atmosphere() {
#ifdef HAVE_MPI
    constexpr int block_size = BLOCK_SIZE;
    using dfp_t = Dex::fp_t;

    i64 dims[7];
    dfp_t atmos_params[4];
    if (state.mpi_state.rank == 0) {
        dims[0] = state.mr_block_map.get_num_active_cells();
        dims[1] = state.mr_block_map.block_map.num_active_tiles;
        dims[2] = state.mr_block_map.block_map.num_x_tiles();
        dims[3] = state.mr_block_map.block_map.num_z_tiles();
        dims[4] = state.mr_block_map.max_mip_level;
        dims[5] = state.config.conserve_charge;
        dims[6] = state.config.conserve_pressure;
        atmos_params[0] = state.atmos.voxel_scale;
        atmos_params[1] = state.atmos.offset_x;
        atmos_params[2] = state.atmos.offset_y;
        atmos_params[3] = state.atmos.offset_z;
    }
    MPI_Bcast(&dims, 7, MPI_INT64_T, 0, state.mpi_state.comm);
    MPI_Bcast(&atmos_params, 4, get_FpMpi(), 0, state.mpi_state.comm);
    if (state.mpi_state.rank != 0) {
        state.config.conserve_charge = dims[5];
        state.config.conserve_pressure = dims[6];
        i64 num_active_cells = dims[0];
        i32 num_active_tiles = dims[1];
        i32 num_x = dims[2] * block_size;
        i32 num_z = dims[3] * block_size;

        auto& block_map = state.mr_block_map.block_map;
        if (interface_config.bbox_crop) {
            // NOTE(claude): Moving bbox -> the tile geometry changes each step,
            // so rebuild it on workers from the broadcast tile counts before the
            // lookup entries are received (lookup.init sizes + clears entries).
            block_map.num_x_tiles() = dims[2];
            block_map.num_z_tiles() = dims[3];
            block_map.bbox.min = 0;
            block_map.bbox.max(0) = num_x;
            block_map.bbox.max(1) = num_z;
            block_map.lookup.init(Dims<2>{.x = block_map.num_x_tiles(), .z = block_map.num_z_tiles()});
        } else {
            block_map.lookup.entries = -1;
        }
        block_map.num_active_tiles = num_active_tiles;
        block_map.active_tiles = decltype(block_map.active_tiles)("active tiles", num_active_tiles);

        dfp_t voxel_scale = atmos_params[0];
        dfp_t offset_x = atmos_params[1];
        dfp_t offset_y = atmos_params[2];
        dfp_t offset_z = atmos_params[3];

        state.atmos = SparseAtmosphere{
            .voxel_scale = voxel_scale,
            .offset_x = offset_x,
            .offset_y = offset_y,
            .offset_z = offset_z,
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
            .vz = yakl::Array<dfp_t, 1, yakl::memDevice>("vz", num_active_cells),
            .e_int = yakl::Array<dfp_t, 1, yakl::memDevice>("e_int", num_active_cells)
        };
        allocate_cell_count_based_terms(state, num_active_cells);
        Kokkos::fence();
    }

    // NOTE(cmo): broadcast all the things
    auto& comm = state.mpi_state.comm;
    auto& block_map = state.mr_block_map.block_map;
    auto& entries = block_map.lookup.entries;
    MPI_Bcast(entries.data(), entries.size(), MPI_INT64_T, 0, comm);
    MPI_Bcast(block_map.active_tiles.data(), block_map.active_tiles.size(), MPI_UINT32_T, 0, comm);

    auto& atmos = state.atmos;
    MPI_Bcast(atmos.temperature.data(), atmos.temperature.size(), get_FpMpi(), 0, comm);
    MPI_Bcast(atmos.pressure.data(), atmos.pressure.size(), get_FpMpi(), 0, comm);
    MPI_Bcast(atmos.ne.data(), atmos.ne.size(), get_FpMpi(), 0, comm);
    MPI_Bcast(atmos.nh_tot.data(), atmos.nh_tot.size(), get_FpMpi(), 0, comm);
    MPI_Bcast(atmos.nh0.data(), atmos.nh0.size(), get_FpMpi(), 0, comm);
    MPI_Bcast(atmos.vturb.data(), atmos.vturb.size(), get_FpMpi(), 0, comm);
    MPI_Bcast(atmos.vx.data(), atmos.vx.size(), get_FpMpi(), 0, comm);
    MPI_Bcast(atmos.vy.data(), atmos.vy.size(), get_FpMpi(), 0, comm);
    MPI_Bcast(atmos.vz.data(), atmos.vz.size(), get_FpMpi(), 0, comm);
    MPI_Bcast(atmos.e_int.data(), atmos.e_int.size(), get_FpMpi(), 0, comm);
    MPI_Bcast(state.pops.data(), state.pops.size(), get_FpMpi(), 0, comm);
    Kokkos::fence();


    if (state.mpi_state.rank != 0) {
        i32 max_mip_level = dims[4];
        state.mr_block_map.init(state.mr_block_map.block_map, max_mip_level);

        if (interface_config.bbox_crop) {
            // NOTE(claude): The box dimensions changed, so the cascade storage
            // (c0_size, max_block_mip, cascade buffers) must be resized to match.
            // Do not reallocate the per-active-cell terms here: pops was just
            // received above and reallocating would zero it.
            reallocate_cascade_storage();
        } else {
            const bool sparse_calc = state.config.sparse_calculation;
            CascadeStorage c0 = state.c0_size;
            std::vector<yakl::Array<i32, 2, yakl::memDevice>> active_probes;
            if (sparse_calc) {
                active_probes = compute_active_probe_lists(state, state.config.max_cascade);
            }
            casc_state.probes_to_compute.init(c0, sparse_calc, active_probes);
            casc_state.mip_chain.init(state, state.mr_block_map.buffer_len(), c0.wave_batch);
        }

        if (interface_config.time_dependent_updates) {
            prev_pops = state.pops.createDeviceCopy();
        }
    }


#endif
}


void DexInterface::initial_worker_atmos_setup() {
#ifdef HAVE_MPI
    using dfp_t = Dex::fp_t;
    constexpr i32 block_size = BLOCK_SIZE;

    i64 dims[5];
    dfp_t atmos_params[4];
    if (state.mpi_state.rank == 0) {
        dims[0] = state.mr_block_map.get_num_active_cells();
        dims[1] = state.mr_block_map.block_map.num_active_tiles;
        dims[2] = state.mr_block_map.block_map.num_x_tiles();
        dims[3] = state.mr_block_map.block_map.num_z_tiles();
        dims[4] = state.mr_block_map.max_mip_level;
        atmos_params[0] = state.atmos.voxel_scale;
        atmos_params[1] = state.atmos.offset_x;
        atmos_params[2] = state.atmos.offset_y;
        atmos_params[3] = state.atmos.offset_z;
    }

    MPI_Bcast(&dims, 5, MPI_INT64_T, 0, state.mpi_state.comm);
    MPI_Bcast(&atmos_params, 4, get_FpMpi(), 0, state.mpi_state.comm);
    auto& map = state.mr_block_map.block_map;
    if (state.mpi_state.rank != 0) {
        i64 num_active_cells = dims[0];
        i32 num_active_tiles = dims[1];
        i32 num_x = dims[2] * block_size;
        i32 num_z = dims[3] * block_size;

        dfp_t voxel_scale = atmos_params[0];
        dfp_t offset_x = atmos_params[1];
        dfp_t offset_y = atmos_params[2];
        dfp_t offset_z = atmos_params[3];

        map.num_x_tiles() = dims[2];
        map.num_z_tiles() = dims[3];

        map.bbox.min = 0;
        map.bbox.max(0) = num_x;
        map.bbox.max(1) = num_z;
        map.lookup.init(Dims<2>{.x = map.num_x_tiles(), .z = map.num_z_tiles()});

        map.num_active_tiles = dims[1];
        map.morton_traversal_order = yakl::Array<u32, 1, yakl::memDevice>(
            "morton_traversal_order",
            map.num_z_tiles() * map.num_x_tiles()
        );
        map.active_tiles = decltype(map.active_tiles)("active tiles", map.num_active_tiles);

        auto& block_map = state.mr_block_map.block_map;
        block_map.lookup.entries = -1;
        block_map.num_active_tiles = num_active_tiles;
        block_map.active_tiles = decltype(block_map.active_tiles)("active tiles", num_active_tiles);

        state.atmos = SparseAtmosphere{
            .voxel_scale = voxel_scale,
            .offset_x = offset_x,
            .offset_y = offset_y,
            .offset_z = offset_z,
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
            .vz = yakl::Array<dfp_t, 1, yakl::memDevice>("vz", num_active_cells),
            .e_int = yakl::Array<dfp_t, 1, yakl::memDevice>("e_int", num_active_cells)
        };
        allocate_cell_count_based_terms(state, num_active_cells);
    }

    auto& comm = state.mpi_state.comm;
    auto& entries = map.lookup.entries;
    MPI_Bcast(entries.data(), entries.size(), MPI_INT64_T, 0, comm);
    MPI_Bcast(map.morton_traversal_order.data(), map.morton_traversal_order.size(), MPI_UINT32_T, 0, comm);
    MPI_Bcast(map.active_tiles.data(), map.active_tiles.size(), MPI_UINT32_T, 0, comm);

    auto& atmos = state.atmos;
    MPI_Bcast(atmos.temperature.data(), atmos.temperature.size(), get_FpMpi(), 0, comm);
    MPI_Bcast(atmos.pressure.data(), atmos.pressure.size(), get_FpMpi(), 0, comm);
    MPI_Bcast(atmos.ne.data(), atmos.ne.size(), get_FpMpi(), 0, comm);
    MPI_Bcast(atmos.nh_tot.data(), atmos.nh_tot.size(), get_FpMpi(), 0, comm);
    MPI_Bcast(atmos.nh0.data(), atmos.nh0.size(), get_FpMpi(), 0, comm);
    MPI_Bcast(atmos.vturb.data(), atmos.vturb.size(), get_FpMpi(), 0, comm);
    MPI_Bcast(atmos.vx.data(), atmos.vx.size(), get_FpMpi(), 0, comm);
    MPI_Bcast(atmos.vy.data(), atmos.vy.size(), get_FpMpi(), 0, comm);
    MPI_Bcast(atmos.vz.data(), atmos.vz.size(), get_FpMpi(), 0, comm);
    MPI_Bcast(atmos.e_int.data(), atmos.e_int.size(), get_FpMpi(), 0, comm);


    if (state.mpi_state.rank != 0) {
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


        i64 num_active_cells = state.mr_block_map.get_num_active_cells();
        allocate_cell_count_based_terms(state, num_active_cells);
        casc_state.init(state, state.config.max_cascade);
    }
#endif
}

void DexInterface::run_worker_loop() {
#ifdef HAVE_MPI
    if (state.mpi_state.rank == 0) {
        return;
    }

    initial_worker_atmos_setup();

    i32 time_idx = 0;
    while (true) {
        int should_continue;
        MPI_Bcast(&should_continue, 1, MPI_INT, 0, state.mpi_state.comm);

        if (!should_continue) {
            break;
        }

        broadcast_atmosphere();
        f64 float_args[3];
        i32 int_args[2];
        MPI_Bcast(int_args, 2, MPI_INT, 0, state.mpi_state.comm);
        MPI_Bcast(float_args, 3, MPI_DOUBLE, 0, state.mpi_state.comm);
        DexConvergence conv{
            .convergence = Dex::fp_t(float_args[0]),
            .max_iter = int_args[0]
        };
        IterateArgs args{
            .dt = float_args[1],
            .theta = float_args[2],
            .first_iter = bool(int_args[1])
        };
        iterate(conv, args);
        time_idx += 1;
    }
    yakl::finalize();
    Kokkos::finalize();
    exit(0);
#endif
}

template <typename FTraits>
bool DexInterface::update_atmosphere(Simulation& sim) {
    if (interface_config.bbox_crop) {
        // NOTE(claude): Moving bbox -> recompute the box and rebuild the whole
        // solve geometry (block map, atmosphere, cascade storage) each update,
        // then run the same advect/time-dependent tail as the full-grid path.
        const TileBbox box = compute_active_tile_bbox<FTraits>(sim);
        rebuild_block_map_and_atmos<FTraits>(sim, box, interface_config.max_mip_level);
        reallocate_solver_state();
        fmt::println(
            "num_active_tiles: {} (bbox {}x{} tiles at ({}, {}))",
            state.mr_block_map.block_map.num_active_tiles, box.bnx, box.bnz, box.tx0, box.tz0
        );

        if (interface_config.advect) {
            copy_pops_from_aux_fields(sim);
        }
        if (interface_config.time_dependent_updates) {
            prev_pops = state.pops.createDeviceCopy();
        }
        fmt::println("Update atmosphere at {:.3f} s", sim.time);
        return true;
    }

    constexpr i32 num_dim = FTraits::num_dim;
    constexpr i32 block_size = BLOCK_SIZE;
    constexpr fp_t m_p = ConstantsF64::u;
    auto& block_map = state.mr_block_map.block_map;
    const i32 num_x = block_map.num_x_tiles() * block_size;
    const i32 num_z = block_map.num_z_tiles() * block_size;
    const auto& sz = sim.state.sz;

    const auto& Q = sim.state.Q;
    const auto& eos = sim.eos;
    const fp_t mu0 = sim.state.mu0;
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

            constexpr int n_hydro = FTraits::num_vars;
            yakl::SArray<fp_t, 1, n_hydro> w;
            using Prim = typename FTraits::prim;

            for (int z = zt * block_size; z < (zt + 1) * block_size; ++z) {
                for (int x = xt * block_size; x < (xt + 1) * block_size; ++x) {
                    CellIndex idx{.i = x + sz.ng, .j = z + sz.ng, .k = 0};
                    const auto q = QtyView(Q, idx);
                    cons_to_prim<FTraits>(eos.gamma, mu0, q, w);

                    fp_t nh_tot = w(I(Prim::Rho)) / (eos.mass_per_h * m_p);
                    fp_t y = eos.y;
                    if (!eos.is_constant) {
                        y = eos.y_space(idx.k, idx.j, idx.i);
                    }
                    auto temp = temperature_si(w(I(Prim::Pres)), nh_tot, eos.total_abund, y);
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
        .vz = yakl::Array<dfp_t, 1, yakl::memDevice>("vz", num_active_cells),
        .e_int = yakl::Array<dfp_t, 1, yakl::memDevice>("e_int", num_active_cells)
    };
    const bool ignore_rt_velocities = interface_config.ignore_rt_velocities;
    const auto& atmos = state.atmos;
    const DexToMhdGrid dex_to_mhd = dex_to_mhd_grid(sz.ng);
    dex_parallel_for(
        "Copy atmos qtys",
        block_map.loop_bounds(),
        KOKKOS_LAMBDA (i64 tile_idx, i32 block_idx) {
            IdxGen idx_gen(block_map);
            i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
            Coord2 coord = idx_gen.loop_coord(tile_idx, block_idx);

            constexpr i32 n_hydro = FTraits::num_vars;
            const CellIndex idx = dex_to_mhd(coord);
            yakl::SArray<fp_t, 1, n_hydro> w;
            QtyView q(Q, idx);
            cons_to_prim<FTraits>(eos.gamma, mu0, q, w);
            using Prim = typename FTraits::prim;
            using Cons = typename FTraits::cons;

            atmos.pressure(ks) = w(I(Prim::Pres));
            // Fixed-density material energy target for forced SE: thermal plus
            // the modeled atomic reservoir carried by IonE.
            atmos.e_int(ks) = (
                atmos.pressure(ks) / (eos.gamma - 1.0_fp)
                + q(I(Cons::IonE)) * w(I(Prim::Rho))
            );
            const fp_t nh = w(I(Prim::Rho)) / (eos.mass_per_h * m_p);
            atmos.nh_tot(ks) = nh;
            fp_t y = eos.y;
            if (!eos.is_constant) {
                y = eos.y_space(idx.k, idx.j, idx.i);
            }
            atmos.ne(ks) = atmos.nh_tot(ks) * y;
            const fp_t temperature = temperature_si(w(I(Prim::Pres)), nh, eos.total_abund, y);
            atmos.temperature(ks) = temperature;
            atmos.nh0(ks) = FP(0.0);
            atmos.vturb(ks) = vturb_fn(temperature, nh, y * nh);
            if (ignore_rt_velocities) {
                atmos.vx(ks) = FP(0.0);
                atmos.vy(ks) = FP(0.0);
                atmos.vz(ks) = FP(0.0);
            } else {
                atmos.vx(ks) = w(I(Prim::Vx));
                atmos.vy(ks) = FP(0.0);
                atmos.vz(ks) = w(I(Prim::Vy));
            }
        }
    );
    Kokkos::fence();


    allocate_cell_count_based_terms(state, num_active_cells);
    allocate_reservoir_terms(num_active_cells);

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
    if (interface_config.time_dependent_updates) {
        prev_pops = state.pops.createDeviceCopy();
    }
    fmt::println("Update atmosphere at {:.3f} s", sim.time);

    return true;
}

bool DexInterface::update_atmosphere(Simulation& sim) {
    return invoke_fluid_traits_2d(sim.num_dim, sim.fluid_type, [&]<typename FTraits>(FTraits) {
        return this->update_atmosphere<FTraits>(sim);
    });
}

template <typename FTraits>
TileBbox DexInterface::compute_active_tile_bbox(Simulation& sim) {
    constexpr i32 block_size = BLOCK_SIZE;
    constexpr fp_t m_p = ConstantsF64::u;

    const auto& sz = sim.state.sz;
    const i32 num_x = sz.xc - 2 * sz.ng;
    const i32 num_z = sz.yc - 2 * sz.ng;
    const i32 full_nx_tiles = num_x / block_size;
    const i32 full_nz_tiles = num_z / block_size;

    // NOTE(claude): Default (no crop) -> couple the whole inner grid.
    const TileBbox full_grid{.tx0 = 0, .tz0 = 0, .bnx = full_nx_tiles, .bnz = full_nz_tiles};
    if (!interface_config.bbox_crop) {
        return full_grid;
    }

    const auto& Q = sim.state.Q;
    const auto& eos = sim.eos;
    const fp_t mu0 = sim.state.mu0;
    const auto cutoff_temperature = state.config.threshold_temperature;

    // NOTE(claude): Per-tile active mask (any cell in the tile below the RT
    // cutoff temperature), using the same test as init_atmosphere.
    yakl::Array<u8, 2, yakl::memDevice> active_tile("active_tile_mask", full_nz_tiles, full_nx_tiles);
    dex_parallel_for(
        "Compute active tile mask",
        FlatLoop<2>(full_nz_tiles, full_nx_tiles),
        KOKKOS_LAMBDA (i32 zt, i32 xt) {
            constexpr int n_hydro = FTraits::num_vars;
            yakl::SArray<fp_t, 1, n_hydro> w;
            using Prim = typename FTraits::prim;
            u8 active = 0;
            for (int z = zt * block_size; z < (zt + 1) * block_size; ++z) {
                for (int x = xt * block_size; x < (xt + 1) * block_size; ++x) {
                    CellIndex idx{.i = x + sz.ng, .j = z + sz.ng, .k = 0};
                    const auto q = QtyView(Q, idx);
                    cons_to_prim<FTraits>(eos.gamma, mu0, q, w);
                    const fp_t nh_tot = w(I(Prim::Rho)) / (eos.mass_per_h * m_p);
                    fp_t y = eos.y;
                    if (!eos.is_constant) {
                        y = eos.y_space(idx.k, idx.j, idx.i);
                    }
                    const auto temp = temperature_si(w(I(Prim::Pres)), nh_tot, eos.total_abund, y);
                    if (temp <= cutoff_temperature) {
                        active = 1;
                    }
                }
            }
            active_tile(zt, xt) = active;
        }
    );
    Kokkos::fence();

    // NOTE(claude): Reduce to the tile-space bounding box of active tiles.
    // Inactive tiles contribute the identity of each reducer so they drop out.
    i32 min_xt, max_xt, min_zt, max_zt;
    i64 num_active;
    dex_parallel_reduce(
        "bbox min_xt", FlatLoop<2>(full_nz_tiles, full_nx_tiles),
        KOKKOS_LAMBDA (i32 zt, i32 xt, i32& v) { if (active_tile(zt, xt)) v = std::min(v, xt); },
        Kokkos::Min<i32>(min_xt)
    );
    dex_parallel_reduce(
        "bbox max_xt", FlatLoop<2>(full_nz_tiles, full_nx_tiles),
        KOKKOS_LAMBDA (i32 zt, i32 xt, i32& v) { if (active_tile(zt, xt)) v = std::max(v, xt); },
        Kokkos::Max<i32>(max_xt)
    );
    dex_parallel_reduce(
        "bbox min_zt", FlatLoop<2>(full_nz_tiles, full_nx_tiles),
        KOKKOS_LAMBDA (i32 zt, i32 xt, i32& v) { if (active_tile(zt, xt)) v = std::min(v, zt); },
        Kokkos::Min<i32>(min_zt)
    );
    dex_parallel_reduce(
        "bbox max_zt", FlatLoop<2>(full_nz_tiles, full_nx_tiles),
        KOKKOS_LAMBDA (i32 zt, i32 xt, i32& v) { if (active_tile(zt, xt)) v = std::max(v, zt); },
        Kokkos::Max<i32>(max_zt)
    );
    dex_parallel_reduce(
        "bbox count", FlatLoop<2>(full_nz_tiles, full_nx_tiles),
        KOKKOS_LAMBDA (i32 zt, i32 xt, i64& v) { if (active_tile(zt, xt)) v += 1; },
        Kokkos::Sum<i64>(num_active)
    );

    if (num_active == 0) {
        // NOTE(claude): Nothing active -> fall back to the full grid; the
        // active-tile selection then finds zero and iterate() short-circuits.
        return full_grid;
    }

    const i32 halo_tiles = (interface_config.bbox_halo_cells + block_size - 1) / block_size;
    i32 tx0 = std::max(0, min_xt - halo_tiles);
    i32 tz0 = std::max(0, min_zt - halo_tiles);
    i32 tx1 = std::min(full_nx_tiles - 1, max_xt + halo_tiles);
    i32 tz1 = std::min(full_nz_tiles - 1, max_zt + halo_tiles);

    // NOTE(claude): Hyperblocking (HYPERBLOCK2x2) requires an even number of
    // tiles per axis. Expand the (inclusive) box by one tile on whichever side
    // stays in the grid to make each extent even. The full grid is already
    // even-tiled (else the uncropped path would throw), so this always fits.
    if constexpr (HYPERBLOCK2x2) {
        if (((tx1 - tx0 + 1) & 1) != 0) {
            if (tx0 > 0) { tx0 -= 1; } else { tx1 += 1; }
        }
        if (((tz1 - tz0 + 1) & 1) != 0) {
            if (tz0 > 0) { tz0 -= 1; } else { tz1 += 1; }
        }
    }
    return TileBbox{.tx0 = tx0, .tz0 = tz0, .bnx = tx1 - tx0 + 1, .bnz = tz1 - tz0 + 1};
}

template <typename FTraits>
bool DexInterface::rebuild_block_map_and_atmos(Simulation& sim, const TileBbox& box, i32 max_mip_level) {
    constexpr fp_t m_p = ConstantsF64::u;
    constexpr i32 block_size = BLOCK_SIZE;

    auto& map = state.mr_block_map.block_map;
    const auto& sz = sim.state.sz;
    const auto& Q = sim.state.Q;
    const auto& eos = sim.eos;
    const fp_t mu0 = sim.state.mu0;
    const auto cutoff_temperature = state.config.threshold_temperature;

    // NOTE(claude): The block map is built in a box-relative, 0-origin frame:
    // tile (0,0) is the box's lower-left tile, which sits at full-grid cell
    // origin (tx0, tz0)*BLOCK_SIZE. The physical placement is restored via the
    // atmosphere offsets below, and the box->full-grid mapping is recorded for
    // output.
    const i32 tx0 = box.tx0;
    const i32 tz0 = box.tz0;
    const DexToMhdGrid dex_to_mhd{
        .cell_offset_x = tx0 * block_size,
        .cell_offset_z = tz0 * block_size,
        .num_ghost = sz.ng
    };
    const i32 num_x = box.bnx * block_size;
    const i32 num_z = box.bnz * block_size;

    // NOTE(claude): Full-grid tile counts, for the output promotion.
    full_num_x_tiles = (sz.xc - 2 * sz.ng) / block_size;
    full_num_z_tiles = (sz.yc - 2 * sz.ng) / block_size;
    box_tile_origin_x = tx0;
    box_tile_origin_z = tz0;

    map.num_x_tiles() = box.bnx;
    map.num_z_tiles() = box.bnz;
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
    yakl::Array<u32, 2, yakl::memDevice> morton_order("morton_traversal_order", map.num_z_tiles(), map.num_x_tiles());
    yakl::Array<u32, 2, yakl::memDevice> active_2d("active_2d", map.num_z_tiles(), map.num_x_tiles());
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

            constexpr int n_hydro = FTraits::num_vars;
            yakl::SArray<fp_t, 1, n_hydro> w;
            using Prim = typename FTraits::prim;

            for (int z = zt * block_size; z < (zt + 1) * block_size; ++z) {
                for (int x = xt * block_size; x < (xt + 1) * block_size; ++x) {
                    const CellIndex idx = dex_to_mhd(Coord2{.x = x, .z = z});
                    const auto q = QtyView(Q, idx);
                    cons_to_prim<FTraits>(eos.gamma, mu0, q, w);

                    fp_t nh_tot = w(I(Prim::Rho)) / (eos.mass_per_h * m_p);
                    fp_t y = eos.y;
                    if (!eos.is_constant) {
                        y = eos.y_space(idx.k, idx.j, idx.i);
                    }
                    auto temp = temperature_si(w(I(Prim::Pres)), nh_tot, eos.total_abund, y);
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
            Coord2 coord = decode_morton<FTraits::num_dim>(code);
            map.lookup(coord) = idx;
        }
    );
    Kokkos::fence();

    state.mr_block_map.init(map, max_mip_level);
    const bool ignore_rt_velocities = interface_config.ignore_rt_velocities;

    using dfp_t = Dex::fp_t;
    i64 num_active_cells = num_active_tiles * ::DexImpl::int_pow<FTraits::num_dim>(block_size);
    state.atmos = SparseAtmosphere{
        .voxel_scale = dfp_t(sim.state.dx),
        // NOTE(claude): Shift the offsets by the box tile origin so the active
        // region keeps its absolute physical placement (and hence its PromWeaver
        // boundary geometry) despite the cropped, translated block map.
        .offset_x = dfp_t(sim.state.loc.x + dex_to_mhd.cell_offset_x * sim.state.dx),
        .offset_y = FP(0.0),
        .offset_z = dfp_t(sim.state.loc.y + dex_to_mhd.cell_offset_z * sim.state.dx),
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
        .vz = yakl::Array<dfp_t, 1, yakl::memDevice>("vz", num_active_cells),
        .e_int = yakl::Array<dfp_t, 1, yakl::memDevice>("e_int", num_active_cells)
    };
    const auto& atmos = state.atmos;
    dex_parallel_for(
        "Copy atmos qtys",
        map.loop_bounds(),
        KOKKOS_LAMBDA (i64 tile_idx, i32 block_idx) {
            IdxGen idx_gen(map);
            i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
            Coord2 coord = idx_gen.loop_coord(tile_idx, block_idx);

            constexpr i32 n_hydro = FTraits::num_vars;
            const CellIndex idx = dex_to_mhd(coord);
            yakl::SArray<fp_t, 1, n_hydro> w;
            QtyView q(Q, idx);
            cons_to_prim<FTraits>(eos.gamma, mu0, q, w);
            using Prim = typename FTraits::prim;
            using Cons = typename FTraits::cons;

            atmos.pressure(ks) = w(I(Prim::Pres));
            atmos.e_int(ks) = (
                atmos.pressure(ks) / (eos.gamma - 1.0_fp)
                + q(I(Cons::IonE)) * w(I(Prim::Rho))
            );
            const fp_t nh = w(I(Prim::Rho)) / (eos.mass_per_h * m_p);
            atmos.nh_tot(ks) = nh;
            fp_t y = eos.y;
            if (!eos.is_constant) {
                y = eos.y_space(idx.k, idx.j, idx.i);
            }
            atmos.ne(ks) = atmos.nh_tot(ks) * y;
            const fp_t temperature = temperature_si(w(I(Prim::Pres)), nh, eos.total_abund, y);
            atmos.temperature(ks) = temperature;
            atmos.nh0(ks) = FP(0.0);
            atmos.vturb(ks) = vturb_fn(temperature, nh, y * nh);
            if (ignore_rt_velocities) {
                atmos.vx(ks) = FP(0.0);
                atmos.vy(ks) = FP(0.0);
                atmos.vz(ks) = FP(0.0);
            } else {
                atmos.vx(ks) = w(I(Prim::Vx));
                atmos.vy(ks) = FP(0.0);
                atmos.vz(ks) = w(I(Prim::Vy));
            }
        }
    );
    Kokkos::fence();

    return true;
}

void DexInterface::reallocate_cascade_storage() {
    // NOTE(claude): Under a moving bbox the box dimensions (and hence cascade
    // storage) change each RT update, so the c0 storage, max_block_mip and the
    // cascade buffers must be resized. This does not touch the per-active-cell
    // terms, so it is safe on workers after pops has been broadcast.
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

    // NOTE(claude): casc_state.init push_back's into i_cascades/tau_cascades, so
    // reconstruct it first to avoid accumulating buffers across updates.
    casc_state = DexCascState{};
    casc_state.init(state, state.config.max_cascade);
    Kokkos::fence();
}

void DexInterface::reallocate_solver_state() {
    i64 num_active_cells = state.mr_block_map.get_num_active_cells();
    allocate_cell_count_based_terms(state, num_active_cells);
    allocate_reservoir_terms(num_active_cells);
    reallocate_cascade_storage();
}

template <typename FTraits>
bool DexInterface::init_atmosphere(Simulation& sim, i32 max_mip_level) {
    constexpr fp_t m_p = ConstantsF64::u;

    if (interface_config.bbox_crop) {
        const TileBbox box = compute_active_tile_bbox<FTraits>(sim);
        return rebuild_block_map_and_atmos<FTraits>(sim, box, max_mip_level);
    }

    auto& map = state.mr_block_map.block_map;
    const auto& sz = sim.state.sz;
    const auto& Q = sim.state.Q;
    const auto& eos = sim.eos;
    const fp_t mu0 = sim.state.mu0;
    auto cutoff_temperature = state.config.threshold_temperature;

    constexpr i32 block_size = BLOCK_SIZE;
    const i32 num_x = sz.xc - 2 * sz.ng;
    // NOTE(cmo): z in dex is y in mosscap
    const i32 num_z = sz.yc - 2 * sz.ng;
    // NOTE(claude): Identity box->full-grid mapping when not cropping, so the
    // output promotion is a no-op.
    box_tile_origin_x = 0;
    box_tile_origin_z = 0;
    full_num_x_tiles = num_x / block_size;
    full_num_z_tiles = num_z / block_size;
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

            constexpr int n_hydro = FTraits::num_vars;
            yakl::SArray<fp_t, 1, n_hydro> w;
            using Prim = typename FTraits::prim;

            for (int z = zt * block_size; z < (zt + 1) * block_size; ++z) {
                for (int x = xt * block_size; x < (xt + 1) * block_size; ++x) {
                    CellIndex idx{.i = x + sz.ng, .j = z + sz.ng, .k = 0};
                    const auto q = QtyView(Q, idx);
                    cons_to_prim<FTraits>(eos.gamma, mu0, q, w);

                    fp_t nh_tot = w(I(Prim::Rho)) / (eos.mass_per_h * m_p);
                    fp_t y = eos.y;
                    if (!eos.is_constant) {
                        y = eos.y_space(idx.k, idx.j, idx.i);
                    }
                    auto temp = temperature_si(w(I(Prim::Pres)), nh_tot, eos.total_abund, y);
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
            Coord2 coord = decode_morton<FTraits::num_dim>(code);
            map.lookup(coord) = idx;
        }
    );
    Kokkos::fence();

    state.mr_block_map.init(map, max_mip_level);
    const bool ignore_rt_velocities = interface_config.ignore_rt_velocities;

    using dfp_t = Dex::fp_t;
    i64 num_active_cells = num_active_tiles * ::DexImpl::int_pow<FTraits::num_dim>(block_size);
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
        .vz = yakl::Array<dfp_t, 1, yakl::memDevice>("vz", num_active_cells),
        .e_int = yakl::Array<dfp_t, 1, yakl::memDevice>("e_int", num_active_cells)
    };
    const auto& atmos = state.atmos;
    const DexToMhdGrid dex_to_mhd = dex_to_mhd_grid(sz.ng);
    dex_parallel_for(
        "Copy atmos qtys",
        map.loop_bounds(),
        KOKKOS_LAMBDA (i64 tile_idx, i32 block_idx) {
            IdxGen idx_gen(map);
            i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
            Coord2 coord = idx_gen.loop_coord(tile_idx, block_idx);

            constexpr i32 n_hydro = FTraits::num_vars;
            const CellIndex idx = dex_to_mhd(coord);
            yakl::SArray<fp_t, 1, n_hydro> w;
            QtyView q(Q, idx);
            cons_to_prim<FTraits>(eos.gamma, mu0, q, w);
            using Prim = typename FTraits::prim;
            using Cons = typename FTraits::cons;

            atmos.pressure(ks) = w(I(Prim::Pres));
            atmos.e_int(ks) = (
                atmos.pressure(ks) / (eos.gamma - 1.0_fp)
                + q(I(Cons::IonE)) * w(I(Prim::Rho))
            );
            const fp_t nh = w(I(Prim::Rho)) / (eos.mass_per_h * m_p);
            atmos.nh_tot(ks) = nh;
            fp_t y = eos.y;
            if (!eos.is_constant) {
                y = eos.y_space(idx.k, idx.j, idx.i);
            }
            atmos.ne(ks) = atmos.nh_tot(ks) * y;
            const fp_t temperature = temperature_si(w(I(Prim::Pres)), nh, eos.total_abund, y);
            atmos.temperature(ks) = temperature;
            atmos.nh0(ks) = FP(0.0);
            atmos.vturb(ks) = vturb_fn(temperature, nh, y * nh);
            if (ignore_rt_velocities) {
                atmos.vx(ks) = FP(0.0);
                atmos.vy(ks) = FP(0.0);
                atmos.vz(ks) = FP(0.0);
            } else {
                atmos.vx(ks) = w(I(Prim::Vx));
                atmos.vy(ks) = FP(0.0);
                atmos.vz(ks) = w(I(Prim::Vy));
            }
        }
    );
    Kokkos::fence();

    return true;
}

bool DexInterface::init_atmosphere(Simulation& sim, i32 max_mip_level) {
    return invoke_fluid_traits_2d(sim.num_dim, sim.fluid_type, [&]<typename FTraits>(FTraits) {
        return this->init_atmosphere<FTraits>(sim, max_mip_level);
    });
}

bool DexInterface::init_config(Simulation& sim, YAML::Node& cfg, const std::string& config_path) {
    auto dex_config = cfg["dex"];
    state.config = parse_dexrt_config(config_path, dex_config);
    state.config.total_abund = sim.eos.total_abund;

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
    AtomicDataHostDevice<dfp_t> atomic_data = to_atomic_data<dfp_t, f64>(
        crtaf_models,
        ToAtomicDataOptions{
            .limit_line_edge_bins=state.config.limit_line_edge_bins
        }
    );
    state.adata = atomic_data.device;
    state.adata_host = atomic_data.host;
    state.have_h = atomic_data.have_h_model;
    state.atoms = extract_atoms(atomic_data.device, atomic_data.host);
    GammaAtomsAndMapping gamma_atoms = extract_atoms_with_gamma_and_mapping(atomic_data.device, atomic_data.host);
    state.atoms_with_gamma = gamma_atoms.atoms;
    state.atoms_with_gamma_mapping = gamma_atoms.mapping;
    state.rate_diag.init_stage_index(state.adata_host);
    init_reservoir_luts();

    i32 max_mip_level = 0;
    for (int i = 0; i <= config.max_cascade; ++i) {
        max_mip_level = std::max(max_mip_level, config.mip_config.mip_levels[i]);
    }
    if (state.config.mode != DexrtMode::GivenFs && LINE_SCHEME == LineCoeffCalc::Classic) {
        max_mip_level = 0;
        state.println("Mips not supported with LineCoeffCalc::Classic");
    }
    interface_config.max_mip_level = max_mip_level;
    state.phi = VoigtProfile<dfp_t>();
    state.nh_lte = HPartFn();
    state.pw_bc = load_bc(config.atmos_path, state.adata.wavelength, config.boundary, PromweaverResampleType::FluxConserving);
    state.boundary = config.boundary;

    yakl::Array<dfp_t, 1, yakl::memHost> muy("muy", NUM_INCL);
    yakl::Array<dfp_t, 1, yakl::memHost> wmuy("wmuy", NUM_INCL);
    for (int i = 0; i < NUM_INCL; ++i) {
        muy(i) = INCL_RAYS[i];
        wmuy(i) = INCL_WEIGHTS[i];
    }
    state.incl_quad.muy = muy.createDeviceCopy();
    state.incl_quad.wmuy = wmuy.createDeviceCopy();

    interface_config.enable = true;
    interface_config.temperature_floor = get_or<fp_t>(cfg, "eos.min_temperature", 2e3_fp);

    run_worker_loop();
    return true;
}

template <typename FTraits>
bool DexInterface::init(Simulation& sim, YAML::Node& cfg) {
    auto dex_config = cfg["dex"];

    using dfp_t = Dex::fp_t;

    init_atmosphere<FTraits>(sim, interface_config.max_mip_level);
    state.println("DexRT Scale: {} m", state.atmos.voxel_scale);

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

    i64 num_active_cells = state.mr_block_map.get_num_active_cells();
    allocate_cell_count_based_terms(state, num_active_cells);
    casc_state.init(state, state.config.max_cascade);

    initial_worker_atmos_setup();

    // NOTE(cmo): Fill the tracer arrays for dex if they're not already allocated
    if (sim.state.cma.apply && !sim.state.cma.fluid_start_idx.initialized()) {
        constexpr fp_t m_p = ConstantsF64::u;
        yakl::Array<i32, 1, yakl::memHost> start_idx("cma_start", state.adata_host.Z.size());
        yakl::Array<i32, 1, yakl::memHost> end_idx("cma_end", state.adata_host.Z.size());
        yakl::Array<fp_t, 1, yakl::memHost> inv_sum("cma_inv_sum", state.adata_host.Z.size());

        // n_e not included
        i32 start = interface_config.field_start_idx + 1;
        for (i32 ia = 0; ia < state.adata_host.Z.size(); ++ia) {
            start += state.adata_host.level_start(ia);
            i32 end = start + state.adata_host.num_level(ia);
            start_idx(ia) = start;
            end_idx(ia) = end;
            inv_sum(ia) = (state.adata_host.abundance(ia) / (m_p * sim.eos.mass_per_h));
        }
        sim.state.cma.fluid_start_idx = start_idx.createDeviceCopy();
        sim.state.cma.fluid_end_idx = end_idx.createDeviceCopy();
        sim.state.cma.fluid_inv_sum = inv_sum.createDeviceCopy();
    }

    return true;
}

bool DexInterface::init(Simulation& sim, YAML::Node& cfg) {
    return invoke_fluid_traits_2d(sim.num_dim, sim.fluid_type, [&]<typename FTraits>(FTraits ftraits) {
        return this->init<FTraits>(sim, cfg);
    });
}

static void setup_wavelength_batch(const DexState& state, int la_start, int la_end) {
    if (state.config.store_J_on_cpu) {
        state.J = FP(0.0);
        if (state.rad_loss.initialized() && state.config.rad_loss != RadLossType::Integrated) {
            state.rad_loss = FP(0.0);
        }
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

/// Called to copy rad_loss from GPU to plane of host array if config.store_J_on_cpu
static void copy_rad_loss_plane_to_host(const DexState& state, int la_start, int la_end) {
    if (!state.rad_loss.initialized()) {
        return;
    }

    // TODO(cmo): This whole function is silly if we're in integrated.
    int wave_batch = la_end - la_start;
    const auto rad_loss_copy = state.rad_loss.createHostCopy();
    if (state.config.rad_loss == RadLossType::PerWavelength) {
        // TODO(cmo): Replace with a memcpy?
        for (int wave = 0; wave < wave_batch; ++wave) {
            for (i64 ks = 0; ks < rad_loss_copy.extent(1); ++ks) {
                state.rad_loss_cpu(la_start + wave, ks) = rad_loss_copy(wave, ks);
            }
        }
    } else {
        KOKKOS_ASSERT(rad_loss_copy.extent(0) == 1);
        for (i64 ks = 0; ks < rad_loss_copy.extent(1); ++ks) {
            state.rad_loss_cpu(0, ks) = rad_loss_copy(0, ks);
        }
    }
}

static void finalise_wavelength_batch(const DexState& state, int la_start, int la_end) {
    if (state.config.store_J_on_cpu) {
        copy_J_plane_to_host(state, la_start, la_end);
        if (state.rad_loss.initialized()) {
            copy_rad_loss_plane_to_host(state, la_start, la_end);
        }
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

bool DexInterface::iterate(const DexConvergence& tol, const IterateArgs& args) {
    JasUnpack(state, config);
#ifdef HAVE_MPI
    if (state.mpi_state.rank == 0) {
        int should_continue = true;
        MPI_Bcast(&should_continue, 1, MPI_INT, 0, state.mpi_state.comm);

        broadcast_atmosphere();

        f64 float_args[3];
        i32 int_args[2];
        float_args[0] = tol.convergence;
        float_args[1] = args.dt;
        float_args[2] = args.theta;
        int_args[0] = tol.max_iter;
        int_args[1] = args.first_iter;

        MPI_Bcast(int_args, 2, MPI_INT, 0, state.mpi_state.comm);
        MPI_Bcast(float_args, 3, MPI_DOUBLE, 0, state.mpi_state.comm);
    }
#endif
    if (state.mr_block_map.get_num_active_cells() == 0) {
        return true;
    }

    const bool conserve_charge = config.conserve_charge;
    const bool actually_conserve_charge = state.have_h && conserve_charge;
    if (!actually_conserve_charge && conserve_charge) {
        throw std::runtime_error("Charge conservation enabled without a model H!");
    }
    const bool conserve_pressure = config.conserve_pressure;
    if (conserve_pressure && !conserve_charge) {
        throw std::runtime_error("Cannot enable pressure conservation without charge conservation.");
    }
    bool time_dependent = false;
    if (args.dt != 0.0_fp && interface_config.time_dependent_updates) {
        time_dependent = true;
        if (!prev_pops.initialized()) {
            throw std::runtime_error("Time dependent update requested, but prev_pops not available (call update_atmosphere).");
        }
        if (conserve_pressure) {
            throw std::runtime_error(
                "dex.conserve_pressure is not supported with time-dependent population updates; "
                "the TD solve conserves the advected mass density."
            );
        }
    }
    const bool actually_conserve_pressure = actually_conserve_charge && conserve_pressure;
    const int initial_lambda_iterations = 1;
    const int max_iters = tol.max_iter;

    auto& waves = state.adata_host.wavelength;
    WavelengthDistributor wave_dist;
    wave_dist.init(state.mpi_state, waves.extent(0), state.c0_size.wave_batch);

    ::Fp2d predicted_pops;
    if (args.theta < FP(1.0)) {
        predicted_pops = state.pops.createDeviceObject();
    }

    int i = 0;
    if ((args.first_iter || !interface_config.advect) && actually_conserve_charge && !time_dependent) {
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
                .conserve_pressure = actually_conserve_pressure
            });
            lte_max_change = nr_update;
            // if (actually_conserve_pressure) {
            //     fp_t nh_tot_update = simple_conserve_pressure(&state);
            //     lte_max_change = std::max(nh_tot_update, lte_max_change);
            // }
        }
        state.println("Ran for {} iterations", lte_i);
        // NOTE(claude): stat_eq/nr_post_update above only actually solve
        // (and mutate pops/ne/nh_tot) on rank 0 under MPI, broadcasting just
        // the scalar change back to workers each call -- so the populations
        // themselves need broadcasting once here before anything downstream
        // (e.g. set_initial_pops_special, the FS loop) reads them.
        wave_dist.update_pops(&state);
        wave_dist.update_ne(&state);
        if (actually_conserve_pressure) {
            wave_dist.update_nh_tot(&state);
        }
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

        if (false) {
            // NOTE(cmo): This is done after reducing Gamma now in case it
            // reduced the precision in the accumulation of the radiative terms
            compute_collisions_to_gamma(&state);
        } else {
            for (int ia = 0; ia < state.Gamma.size(); ++ia) {
                state.Gamma[ia] = FP(0.0);
            }
            yakl::fence();
        }
        state.rate_diag.zero();

        bool print_worst_wphi = first_inner_iter;
        compute_profile_normalisation(state, casc_state, print_worst_wphi);
        state.J = FP(0.0);
        if (config.store_J_on_cpu) {
            state.J_cpu = FP(0.0);
        }
        if (state.rad_loss.initialized()) {
            state.rad_loss = FP(0.0);
            if (config.store_J_on_cpu) {
                state.rad_loss_cpu = FP(0.0);
            }
        }
        yakl::fence();
        WavelengthBatch wave_batch;
        wave_dist.reset();
        wave_dist.wait_for_all(state.mpi_state);
        while (wave_dist.next_batch(state.mpi_state, &wave_batch)) {
            setup_wavelength_batch(state, wave_batch.la_start, wave_batch.la_end);
            bool lambda_iterate = i < initial_lambda_iterations;
            dynamic_formal_sol_rc(
                state,
                casc_state,
                DynamicFormalSolRcOptions {
                    .la_start = wave_batch.la_start,
                    .la_end = wave_batch.la_end,
                    .lambda_iterate = lambda_iterate,
                    .compute_rad_loss = interface_config.rad_loss
                }
            );
            finalise_wavelength_batch(state, wave_batch.la_start, wave_batch.la_end);
        }
        yakl::fence();
        wave_dist.wait_for_all(state.mpi_state);

        if (time_dependent) {
            state.println("  == Pops update ==");
            wave_dist.reduce_Gamma(&state);
            if (state.mpi_state.rank == 0) {
                compute_collisions_to_gamma(
                    &state,
                    ComputeCollisionsOptions{
                        .zero_gamma = false
                    }
                );
            }
            max_change = time_dep_update(
                state,
                prev_pops,
                KineticEqOptions {
                    .dt = Dex::fp_t(args.dt),
                    .theta = Dex::fp_t(args.theta),
                    .initial_iter = i == 0,
                    .predicted_pops = predicted_pops,
                    .ignore_change_below_ntot_frac=std::min(FP(1e-6), tol.convergence)
                }
            );
            if (actually_conserve_charge) {
                fp_t nr_update = time_dep_nr_post_update(
                    state,
                    prev_pops,
                    TimeDepNrPostUpdateOptions{
                        .dt = Dex::fp_t(args.dt),
                        .theta = Dex::fp_t(args.theta),
                        .predicted_pops = predicted_pops,
                        .ignore_change_below_ntot_frac = std::min(FP(1e-6), tol.convergence)
                    }
                );
                wave_dist.update_ne(&state);
                max_change = std::max(nr_update, max_change);
            }
        } else {
            state.println("  == Statistical equilibrium ==");
            wave_dist.reduce_Gamma(&state);
            if (state.mpi_state.rank == 0) {
                compute_collisions_to_gamma(
                    &state,
                    ComputeCollisionsOptions{
                        .zero_gamma = false
                    }
                );
            }
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
                        .conserve_pressure = actually_conserve_pressure
                    }
                );
                wave_dist.update_ne(&state);
                max_change = std::max(nr_update, max_change);
                if (actually_conserve_pressure) {
                    wave_dist.update_nh_tot(&state);
                }

                // Pressure conservation changes nh_tot, whereas this path is
                // the fixed-density material-energy solve. Keep the two
                // mutually exclusive, as in DexRT's standalone driver.
                if (!actually_conserve_pressure) {
                    fp_t temp_update = simple_conserve_energy(&state);
                    wave_dist.update_temperature(&state);
                    max_change = std::max(temp_update, max_change);
                }
            }
        }
        if (config.ng.enable) {
            accelerated = ng.accelerate(state, max_change);
            if (accelerated) {
                state.println("  ~~ Ng Acceleration! (📉 or 💣) ~~");
            }
        }
        wave_dist.update_pops(&state);
        i += 1;
        first_inner_iter = false;
    }
    // NOTE(cmo): We only need to do this after the final iteration.
    wave_dist.reduce_rad_loss(&state);
    wave_dist.reduce_J(&state);
    wave_dist.reduce_rate_diagnostics(&state);

    num_iter = i;

    return max_change <= tol.convergence;
}

/// NOTE(claude): Takes a mutable state because collisional_rates output is
/// computed into the (by now dead) Gamma buffer.
void save_results(
    DexState& state,
    yakl::SimpleNetCDF& nc,
    bool single_file,
    i32 num_iter,
    i32 time_idx
) {
    const auto& config = state.config;
    const auto& out_cfg = config.output;
    if (state.mpi_state.rank != 0) {
        return;
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
        maybe_rehydrate_and_write(state.atmos.ne, convert_name("ne"), {});
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

    if (out_cfg.rad_loss) {
        std::string leading_dim = config.rad_loss == RadLossType::Integrated ? "wavelength_integrated" : "wavelength";
        // NOTE(claude): rad_loss is held in kW/m3 (and applied as such), but is
        // output in W/m3 to match the other rates. Scale a copy -- the live
        // array is still read by min_characteristic_cooling_time and
        // update_temperature_rad_eq.
        if (config.store_J_on_cpu) {
            const auto& src = state.rad_loss_cpu;
            auto scaled = src.createHostObject();
            for (i32 la = 0; la < src.extent(0); ++la) {
                for (i64 ks = 0; ks < src.extent(1); ++ks) {
                    scaled(la, ks) = src(la, ks) * 1e3_fp;
                }
            }
            maybe_rehydrate_and_write(scaled, convert_name("rad_loss"), {leading_dim});
        } else {
            const auto& src = state.rad_loss;
            auto scaled = src.createDeviceObject();
            dex_parallel_for(
                "rad_loss -> W m-3",
                FlatLoop<2>(src.extent(0), src.extent(1)),
                KOKKOS_LAMBDA (i32 la, i64 ks) {
                    scaled(la, ks) = src(la, ks) * 1e3_fp;
                }
            );
            Kokkos::fence();
            maybe_rehydrate_and_write(scaled, convert_name("rad_loss"), {leading_dim});
        }
    }

    const std::vector<GammaMat>* collisional = nullptr;
    if (out_cfg.collisional_rates) {
        // NOTE(claude): Refills Gamma with the raw collisional rates (no
        // fixup_gamma, so the diagonal stays zero). Gamma is dead here: it is
        // re-zeroed at the top of the next iterate.
        compute_collisions_to_gamma(&state);
        collisional = &state.Gamma;
    }
    write_rate_diagnostics(
        state,
        nc,
        RateDiagOutputOpts{
            .sparse = out_cfg.sparse,
            .suffix = single_file ? fmt::format("_{}", time_idx) : std::string(""),
            .z_dim = "z_dex",
            .x_dim = "x_dex",
            // NOTE(claude): kW m-3 -> W m-3, to match the other rates here.
            .energy_scale = RadLossFp(1e3)
        },
        collisional
    );
}

/// Write one of the per-active-cell reservoir diagnostics, following the same
/// sparse/dense and naming conventions as the rest of the Dex output.
static void write_sparse_diagnostic(
    const DexState& state,
    yakl::SimpleNetCDF& nc,
    bool single_file,
    i32 time_idx,
    const DexFp1d& arr,
    const std::string& base_name
) {
    if (!arr.initialized()) {
        return;
    }

    auto convert_name = [&](const std::string& name) {
        if (single_file) {
            return fmt::format("{}_{}", name, time_idx);
        }
        return name;
    };

    const std::string name = convert_name(base_name);
    if (state.config.output.sparse) {
        nc.write(arr, name, {convert_name("ks")});
    } else {
        auto hydrated = rehydrate_sparse_quantity(state.mr_block_map.block_map, arr);
        nc.write(hydrated, name, {"z_dex", "x_dex"});
    }
}

void DexInterface::write_output(const Simulation& sim, yakl::SimpleNetCDF& nc) {
    const auto& cfg = sim.out_cfg;

    // NOTE(claude): Under bbox_crop the solve block map is box-relative. Every
    // grid-dependent writer derives its extents/positions from
    // block_map.num_*_tiles() and block_map.active_tiles (via decode_morton),
    // and max_block_mip is a pre-filled per-tile array. So to write output
    // as-if the full grid were coupled we temporarily "promote" those to the
    // full grid: re-encode the active-tile morton codes with the box origin
    // (preserving order, so the ks<->tile mapping of the data arrays is
    // unchanged), pad max_block_mip to full-grid tile dims, and set the full
    // tile counts. Everything is restored afterwards. When not cropping this is
    // an identity and is skipped.
    auto& block_map = state.mr_block_map.block_map;
    const bool promote = interface_config.bbox_crop && (
        box_tile_origin_x != 0 || box_tile_origin_z != 0 ||
        block_map.num_x_tiles() != full_num_x_tiles ||
        block_map.num_z_tiles() != full_num_z_tiles
    );
    const i32 saved_nx = block_map.num_x_tiles();
    const i32 saved_nz = block_map.num_z_tiles();
    auto saved_active_tiles = block_map.active_tiles;
    auto saved_max_block_mip = state.max_block_mip;
    if (promote) {
        const i32 tx0 = box_tile_origin_x;
        const i32 tz0 = box_tile_origin_z;

        auto full_active = decltype(block_map.active_tiles)(
            "active tiles (full grid)", saved_active_tiles.extent(0)
        );
        dex_parallel_for(
            "Promote active tiles",
            FlatLoop<1>(saved_active_tiles.extent(0)),
            KOKKOS_LAMBDA (i32 i) {
                Coord2 c = decode_morton<2>(saved_active_tiles(i));
                full_active(i) = encode_morton<2>(Coord2{.x = c.x + tx0, .z = c.z + tz0});
            }
        );
        Kokkos::fence();

        if (state.max_block_mip.initialized()) {
            const auto mbm = state.max_block_mip;
            auto full_mbm = decltype(state.max_block_mip)(
                "max_block_mip (full grid)", mbm.extent(0), full_num_z_tiles, full_num_x_tiles
            );
            full_mbm = -1;
            Kokkos::fence();
            dex_parallel_for(
                "Promote max_block_mip",
                FlatLoop<3>(mbm.extent(0), mbm.extent(1), mbm.extent(2)),
                KOKKOS_LAMBDA (i32 w, i32 zt, i32 xt) {
                    full_mbm(w, zt + tz0, xt + tx0) = mbm(w, zt, xt);
                }
            );
            Kokkos::fence();
            state.max_block_mip = full_mbm;
        }

        block_map.num_x_tiles() = full_num_x_tiles;
        block_map.num_z_tiles() = full_num_z_tiles;
        block_map.active_tiles = full_active;
    }

    if (cfg.prev_output_time < 0.0_fp || !cfg.single_file) {
        add_netcdf_attributes(state, nc);
    }
    save_results(state, nc, cfg.single_file, num_iter, sim.out_cfg.output_count);

    if (state.mpi_state.rank == 0) {
        const i32 time_idx = cfg.output_count;
        write_sparse_diagnostic(state, nc, cfg.single_file, time_idx, g_ion, "g_ion");
        write_sparse_diagnostic(state, nc, cfg.single_file, time_idx, g_exc, "g_exc");
        if (interface_config.rad_loss) {
            write_sparse_diagnostic(state, nc, cfg.single_file, time_idx, temp_floor_heat, "temp_floor_heat");
        }
    }

    if (promote) {
        block_map.num_x_tiles() = saved_nx;
        block_map.num_z_tiles() = saved_nz;
        block_map.active_tiles = saved_active_tiles;
        state.max_block_mip = saved_max_block_mip;
    }
}

template <typename FTraits>
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
    using Cons = typename FTraits::cons;

    const DexToMhdGrid dex_to_mhd = dex_to_mhd_grid(sz.ng);
    dex_parallel_for(
        "nhtot -> rho",
        FlatLoop<2>(block_map.loop_bounds()),
        KOKKOS_LAMBDA (i64 tile_idx, i32 block_idx) {
            IdxGen idx_gen(mr_block_map);
            const i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
            Coord2 coord = idx_gen.loop_coord(tile_idx, block_idx);
            const CellIndex idx = dex_to_mhd(coord);

            Q(I(Cons::Rho), idx.k, idx.j, idx.i) = atmos.nh_tot(ks) * eos.mass_per_h * m_p;
        }
    );
    Kokkos::fence();
}

void DexInterface::copy_nhtot_to_rho(const Simulation& sim) {
    return invoke_fluid_traits_2d(sim.num_dim, sim.fluid_type, [&]<typename FTraits>(FTraits) {
        return this->copy_nhtot_to_rho<FTraits>(sim);
    });
}

template <typename FTraits>
void DexInterface::copy_pops_to_aux_fields(const Simulation& sim) {
    if (!interface_config.advect || !interface_config.enable) {
        return;
    }

    JasUnpack(state, mr_block_map, atmos, pops);
    const auto& block_map = mr_block_map.block_map;
    const auto& Q = sim.state.Q;
    const auto& sz = sim.state.sz;

    // NOTE(cmo): This is a bit of a hack, but we really do need to update the
    // tracers to match the thermodynamic state even if they're not present
    // outside the RT active cells
    lte_init_aux_fields<FTraits>(sim);

    // Now to the actual promised copying
    const i32 start_idx = interface_config.field_start_idx;
    const DexToMhdGrid dex_to_mhd = dex_to_mhd_grid(sz.ng);
    dex_parallel_for(
        "Pops -> Tracers",
        FlatLoop<2>(block_map.loop_bounds()),
        KOKKOS_LAMBDA (i64 tile_idx, i32 block_idx) {
            IdxGen idx_gen(mr_block_map);
            const i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
            Coord2 coord = idx_gen.loop_coord(tile_idx, block_idx);
            const CellIndex idx = dex_to_mhd(coord);

            Q(start_idx, idx.k, idx.j, idx.i) = atmos.ne(ks);
            for (int v = start_idx + 1; v < Q.extent(0); ++v) {
                Q(v, idx.k, idx.j, idx.i) = pops(v - (start_idx + 1), ks);
            }
        }
    );
    Kokkos::fence();
}

void DexInterface::copy_pops_to_aux_fields(const Simulation& sim) {
    return invoke_fluid_traits_2d(sim.num_dim, sim.fluid_type, [&]<typename FTraits>(FTraits) {
        return this->copy_pops_to_aux_fields<FTraits>(sim);
    });
}

void DexInterface::init_reservoir_luts() {
    const auto& adata_host = state.adata_host;
    const i32 num_atom = adata_host.num_level.extent(0);
    const i64 total_n_level = adata_host.energy.extent(0);

    yakl::Array<Dex::fp_t, 1, yakl::memHost> chi("chi_lut", total_n_level);
    yakl::Array<Dex::fp_t, 1, yakl::memHost> e_exc("e_exc_lut", total_n_level);

    for (i32 ia = 0; ia < num_atom; ++ia) {
        const i32 start = adata_host.level_start(ia);
        const i32 n_level = adata_host.num_level(ia);
        for (i32 l = start; l < start + n_level; ++l) {
            // NOTE(claude): The ground state of this level's ion stage, i.e. the
            // lowest energy level of this atom sharing its stage. Levels are
            // energy ordered within an atom.
            const auto stage_l = adata_host.stage(l);
            i32 ground = l;
            for (i32 m = start; m < start + n_level; ++m) {
                if (adata_host.stage(m) == stage_l) {
                    ground = m;
                    break;
                }
            }
            // NOTE(claude): Energies are relative to level 0 of the model atom, so
            // chi is measured from the atom's own lowest stage: for a CaII
            // model, the Ca II levels carry chi = 0 and only the Ca II -> III
            // step is counted. chi + e_exc == energy, so g_ion + g_exc
            // reproduces the IonE reservoir exactly.
            // adata energies are stored in eV, so convert to J here.
            chi(l) = (adata_host.energy(ground) - adata_host.energy(start)) * ConstantsF64::eV;
            e_exc(l) = (adata_host.energy(l) - adata_host.energy(ground)) * ConstantsF64::eV;
        }
    }

    chi_lut = chi.createDeviceCopy();
    e_exc_lut = e_exc.createDeviceCopy();
}

void DexInterface::allocate_reservoir_terms(i64 num_active_cells) {
    if (res_e_ion_pre.initialized() && res_e_ion_pre.extent(0) == num_active_cells) {
        return;
    }

    res_e_ion_pre = DexFp1d("E_ion_pre", num_active_cells);
    res_e_exc_pre = DexFp1d("E_exc_pre", num_active_cells);
    g_ion = DexFp1d("g_ion", num_active_cells);
    g_exc = DexFp1d("g_exc", num_active_cells);
    temp_floor_heat = DexFp1d("temp_floor_heat", num_active_cells);

    res_e_ion_pre = FP(0.0);
    res_e_exc_pre = FP(0.0);
    g_ion = FP(0.0);
    g_exc = FP(0.0);
    temp_floor_heat = FP(0.0);
    Kokkos::fence();
}

void DexInterface::compute_reservoir_energies(const DexFp1d& e_ion, const DexFp1d& e_exc) {
    JasUnpack(state, pops);
    const auto& chi = chi_lut;
    const auto& exc = e_exc_lut;

    // NOTE(claude): pops has every level of every atom along its leading
    // dimension, so this sums over all species.
    dex_parallel_for(
        "Reservoir energy",
        FlatLoop<1>(pops.extent(1)),
        KOKKOS_LAMBDA (i64 ks) {
            Dex::fp_t ion = FP(0.0);
            Dex::fp_t excitation = FP(0.0);
            for (int l = 0; l < pops.extent(0); ++l) {
                ion += pops(l, ks) * chi(l);
                excitation += pops(l, ks) * exc(l);
            }
            e_ion(ks) = ion;
            e_exc(ks) = excitation;
        }
    );
    Kokkos::fence();
}

void DexInterface::snapshot_reservoir_energies() {
    if (!interface_config.enable || state.mr_block_map.get_num_active_cells() == 0) {
        return;
    }

    allocate_reservoir_terms(state.pops.extent(1));
    compute_reservoir_energies(res_e_ion_pre, res_e_exc_pre);
}

void DexInterface::evaluate_reservoir_rates(fp_t dt) {
    if (!interface_config.enable || state.mr_block_map.get_num_active_cells() == 0) {
        return;
    }
    if (!res_e_ion_pre.initialized() || res_e_ion_pre.extent(0) != state.pops.extent(1)) {
        throw std::runtime_error(
            "The active cell count changed across the NEQ solve; reservoir rates can't be differenced."
        );
    }

    // NOTE(claude): Compute the post-solve energies into g_ion/g_exc, then
    // difference in place. rho is unchanged across the solve.
    compute_reservoir_energies(g_ion, g_exc);

    const auto& e_ion_pre = res_e_ion_pre;
    const auto& e_exc_pre = res_e_exc_pre;
    const auto& gi = g_ion;
    const auto& ge = g_exc;
    const Dex::fp_t inv_dt = (dt > 0.0_fp) ? Dex::fp_t(FP(1.0) / dt) : FP(0.0);

    dex_parallel_for(
        "Reservoir rates",
        FlatLoop<1>(gi.extent(0)),
        KOKKOS_LAMBDA (i64 ks) {
            // Positive when recombining / de-exciting, i.e. releasing energy
            gi(ks) = (e_ion_pre(ks) - gi(ks)) * inv_dt;
            ge(ks) = (e_exc_pre(ks) - ge(ks)) * inv_dt;
        }
    );
    Kokkos::fence();
}

template <typename FTraits>
void DexInterface::integrate_rad_loss_split(const Simulation& sim) {
    if (!interface_config.enable || !interface_config.rad_loss) {
        return;
    }

    using Cons = typename FTraits::cons;
    using Prim = typename FTraits::prim;

    const fp_t temperature_floor = interface_config.temperature_floor;
    const fp_t total_abund = sim.eos.total_abund;
    JasUnpack(state, mr_block_map, atmos, pops, adata);
    const auto& block_map = mr_block_map.block_map;
    const auto& Q = sim.state.Q;
    const auto& sz = sim.state.sz;
    JasUnpack(sim, dt, eos);
    const auto& gamma = eos.gamma;
    const auto& mu0 = sim.state.mu0;

    yakl::Array<RadLossFp, 1, yakl::memDevice> rad_loss;
    if (state.config.rad_loss == RadLossType::Integrated) {
        if (state.config.store_J_on_cpu) {
            // NOTE(cmo): In an MPI case, the rad_loss from the workers is
            // reduced into rad_loss_cpu, _not_ rad_loss, so ensure we're using
            // the right array here.
            rad_loss = state.rad_loss_cpu.createDeviceCopy().reshape(state.rad_loss.extent(1));
        } else {
            rad_loss = yakl::Array<RadLossFp, 1, yakl::memDevice>("rad loss sum", state.rad_loss.data(), state.rad_loss.extent(1));
        }
    } else {
        if (state.config.store_J_on_cpu) {
            yakl::Array<RadLossFp, 1, yakl::memHost> rad_loss_h("rad loss sum", state.rad_loss.extent(1));
            rad_loss_h = FP(0.0);
            JasUnpack(state, rad_loss_cpu);

            dex_parallel_for<Kokkos::DefaultHostExecutionSpace>(
                "Sum rad loss over wavelength",
                FlatLoop<1>(rad_loss_h.extent(0)),
                KOKKOS_LAMBDA (i64 ks) {
                    for (int la = 0; la < rad_loss_cpu.extent(0); ++la) {
                        rad_loss_h(ks) += rad_loss_cpu(la, ks);
                    }
                }
            );
            Kokkos::fence();
            rad_loss = rad_loss_h.createDeviceCopy();
        } else {
            rad_loss = yakl::Array<RadLossFp, 1, yakl::memDevice>("rad loss sum", state.rad_loss.extent(1));
            const auto rad_loss_w = state.rad_loss;
            rad_loss = FP(0.0);
            dex_parallel_for(
                "Sum rad loss over wavelength",
                FlatLoop<1>(rad_loss.extent(0)),
                KOKKOS_LAMBDA (i64 ks) {
                    for (int la = 0; la < rad_loss_w.extent(0); ++la) {
                        rad_loss(ks) += rad_loss_w(la, ks);
                    }
                }
            );
            Kokkos::fence();
        }
    }

    const bool update_ion_e = interface_config.update_ion_e;
    allocate_reservoir_terms(state.pops.extent(1));
    const auto& floor_heat = temp_floor_heat;
    // NOTE(claude): The EOS floor fires over the whole domain during the RK
    // stages and accumulates into this dense field in J m-3; here we pull out
    // the dex-active cells and convert to W m-3 over the full step.
    const auto& eos_floor_heat = sim.eos.floor_heat;
    const bool has_eos_floor_heat = eos_floor_heat.initialized();
    const fp_t inv_dt = (dt > 0.0_fp) ? (1.0_fp / dt) : 0.0_fp;
    typedef Kokkos::MinLoc<fp_t, i64> MinLoc;
    MinLoc::value_type minloc;
    const DexToMhdGrid dex_to_mhd = dex_to_mhd_grid(sz.ng);
    dex_parallel_reduce(
        "Integrate rad loss",
        FlatLoop<2>(block_map.loop_bounds()),
        KOKKOS_LAMBDA (i64 tile_idx, i32 block_idx, MinLoc::value_type& min_loc) {
            IdxGen idx_gen(mr_block_map);
            const i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
            Coord2 coord = idx_gen.loop_coord(tile_idx, block_idx);
            const CellIndex idx = dex_to_mhd(coord);

            // Calculated in kW/m3;
            fp_t delta_E = 0.0_fp;
            delta_E = -rad_loss(ks) * 1e3_fp * dt;
            floor_heat(ks) = has_eos_floor_heat
                ? eos_floor_heat(idx.k, idx.j, idx.i) * inv_dt
                : 0.0_fp;

            if (update_ion_e) {
                QtyView Qv(Q, idx);
                Qv(I(Cons::Ene)) += delta_E;
                fp_t ion_e = 0.0_fp;
                for (int l = 0; l < pops.extent(0); ++l) {
                    ion_e += pops(l, ks) * adata.energy(l) * ConstantsF64::eV;
                }
                Qv(I(Cons::IonE)) = ion_e / Qv(I(Cons::Rho));
                yakl::SArray<fp_t, 1, FTraits::num_vars> w(0.0_fp);
                cons_to_prim<FTraits>(gamma, mu0, Qv, w);
                const fp_t temperature_post = w(I(Prim::Pres)) / (ConstantsF64::k_B * (total_abund * atmos.nh_tot(ks) + atmos.ne(ks)));
                if (temperature_post < temperature_floor) {
                    const fp_t temperature_deficit = temperature_floor - temperature_post;
                    const fp_t pressure_deficit = temperature_deficit * (ConstantsF64::k_B * (total_abund * atmos.nh_tot(ks) + atmos.ne(ks)));
                    const fp_t energy_deficit = pressure_deficit / (gamma - 1.0_fp);
                    Qv(I(Cons::Ene)) += energy_deficit;
                    floor_heat(ks) += energy_deficit * inv_dt;
                }

                if (temperature_post < min_loc.val) {
                    min_loc.val = temperature_post;
                    min_loc.loc = ks;
                }

            } else {
                // NOTE(cmo): This path is not recomended, but left for posterity.
                const fp_t eint_pre = atmos.pressure(ks) / (gamma - 1.0_fp);
                fp_t eint_post = eint_pre + delta_E;
                fp_t pressure_post = eint_post * (gamma - 1.0_fp);
                fp_t temperature_post = pressure_post / (ConstantsF64::k_B * (total_abund * atmos.nh_tot(ks) + atmos.ne(ks)));
                const fp_t eint_unclamped = eint_post;
                temperature_post = std::max(temperature_post, temperature_floor);
                pressure_post = (ConstantsF64::k_B * (total_abund * atmos.nh_tot(ks) + atmos.ne(ks))) * temperature_post;
                eint_post = pressure_post / (gamma - 1.0_fp);
                floor_heat(ks) += (eint_post - eint_unclamped) * inv_dt;

                if (temperature_post < min_loc.val) {
                    min_loc.val = temperature_post;
                    min_loc.loc = ks;
                }

                Q(I(Cons::Ene), idx.k, idx.j, idx.i) += (eint_post - eint_pre);
            }
        },
        MinLoc(minloc)
    );
    Kokkos::fence();

    // NOTE(claude): Drained now, so a second output in the same step cannot
    // double-count it. main.cpp also zeroes it before each step, which covers
    // the case where rad_loss is off and we never get here.
    sim.eos.reset_floor_heat();

    fmt::println("Min temperature {:.3e} K @ ks={}", minloc.val, minloc.loc);
}

void DexInterface::integrate_rad_loss_split(const Simulation& sim) {
    if (state.mr_block_map.get_num_active_cells() == 0) {
        return;
    }
    return invoke_fluid_traits_2d(sim.num_dim, sim.fluid_type, [&]<typename FTraits>(FTraits){
        return this->integrate_rad_loss_split<FTraits>(sim);
    });
}

template <typename FTraits>
void DexInterface::copy_pops_from_aux_fields(const Simulation& sim) {
    if (!interface_config.advect || !interface_config.enable) {
        return;
    }

    JasUnpack(state, mr_block_map, atmos, pops);
    const auto& block_map = mr_block_map.block_map;
    const auto& Q = sim.state.Q;
    const auto& sz = sim.state.sz;

    const i32 start_idx = interface_config.field_start_idx;
    const DexToMhdGrid dex_to_mhd = dex_to_mhd_grid(sz.ng);
    dex_parallel_for(
        "Tracers -> Pops",
        FlatLoop<2>(block_map.loop_bounds()),
        KOKKOS_LAMBDA (i64 tile_idx, i32 block_idx) {
            IdxGen idx_gen(mr_block_map);
            const i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
            Coord2 coord = idx_gen.loop_coord(tile_idx, block_idx);
            const CellIndex idx = dex_to_mhd(coord);

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

void DexInterface::copy_pops_from_aux_fields(const Simulation& sim) {
    return invoke_fluid_traits_2d(sim.num_dim, sim.fluid_type, [&]<typename FTraits>(FTraits) {
        return this->copy_pops_from_aux_fields<FTraits>(sim);
    });
}

template <typename FTraits>
void DexInterface::copy_to_eos(const Simulation& sim) {
    if (!interface_config.advect || !interface_config.enable) {
        return;
    }

    JasUnpack(state, mr_block_map, atmos, pops);
    const auto& block_map = mr_block_map.block_map;
    const auto& sz = sim.state.sz;
    const auto& eos = sim.eos;

    if (!sim.eos.y_space.initialized()) {
        return;
    }

    const DexToMhdGrid dex_to_mhd = dex_to_mhd_grid(sz.ng);
    dex_parallel_for(
        "Pops -> y",
        FlatLoop<2>(block_map.loop_bounds()),
        KOKKOS_LAMBDA (i64 tile_idx, i32 block_idx) {
            IdxGen idx_gen(mr_block_map);
            const i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
            Coord2 coord = idx_gen.loop_coord(tile_idx, block_idx);
            const CellIndex idx = dex_to_mhd(coord);

            const fp_t y = atmos.ne(ks) / atmos.nh_tot(ks);

            eos.y_space(idx.k, idx.j, idx.i) = y;
        }
    );
    Kokkos::fence();
}

void DexInterface::copy_to_eos(const Simulation& sim) {
    return invoke_fluid_traits_2d(sim.num_dim, sim.fluid_type, [&]<typename FTraits>(FTraits) {
        return this->copy_to_eos<FTraits>(sim);
    });
}

template <typename FTraits>
void DexInterface::lte_init_aux_fields(const Simulation& sim) {
    if (!interface_config.advect || !interface_config.enable) {
        return;
    }

    constexpr fp_t m_p = ConstantsF64::u;

    const auto& Q = sim.state.Q;
    const auto& sz = sim.state.sz;
    const auto& eos = sim.eos;
    const fp_t mu0 = sim.state.mu0;

    const i32 tracer_start = interface_config.field_start_idx;
    for (int ia = 0; ia < state.atoms.size(); ++ia) {
        const auto& atom = state.atoms[ia];
        const auto& level_start = state.adata_host.level_start(ia);
        const auto flat_pops = std::remove_cvref_t<decltype(Q)>(
            "flat_tracer_pops",
            &Q(tracer_start + level_start + 1, 0, 0, 0),
            atom.energy.size(),
            Q.extent(1),
            Q.extent(2),
            Q.extent(3)
        ).reshape(atom.energy.size(), Q.extent(1)*Q.extent(2)*Q.extent(3));

        dex_parallel_for(
            "LTE tracers",
            FlatLoop<3>(sz.zc, sz.yc, sz.xc),
            KOKKOS_LAMBDA (i32 k, i32 j, i32 i) {
                constexpr i32 n_hydro = FTraits::num_vars;
                CellIndex idx{.i = i, .j = j, .k = k};
                yakl::SArray<fp_t, 1, n_hydro> w;
                QtyView q(Q, idx);
                cons_to_prim<FTraits>(eos.gamma, mu0, q, w);
                using Prim = typename FTraits::prim;

                const i64 flat_idx = i + j * sz.xc + k * sz.yc * sz.xc;

                const fp_t pressure = w(I(Prim::Pres));
                const fp_t nh = w(I(Prim::Rho)) / (eos.mass_per_h * m_p);
                fp_t y = eos.y;
                if (!eos.is_constant) {
                    y = eos.y_space(idx.k, idx.j, idx.i);
                }
                const fp_t ne = nh * y;
                const fp_t temperature = temperature_si(w(I(Prim::Pres)), nh, eos.total_abund, y);

                Q(tracer_start, k, j, i) = ne;
                lte_pops(
                    atom.energy,
                    atom.g,
                    atom.stage,
                    temperature,
                    ne,
                    nh * atom.abundance,
                    flat_pops,
                    flat_idx
                );
            }
        );
    }
}

void DexInterface::lte_init_aux_fields(const Simulation& sim) {
    return invoke_fluid_traits_2d(sim.num_dim, sim.fluid_type, [&]<typename FTraits>(FTraits) {
        return this->lte_init_aux_fields<FTraits>(sim);
    });
}

fp_t DexInterface::mean_temperature() {
    JasUnpack(state, atmos);
    const auto& temperature = atmos.temperature;

    const fp_t threshold = state.config.threshold_temperature;
    const bool have_temperature_threshold = threshold > FP(0.0);
    f64 result;
    dex_parallel_reduce(
        "Temperature sum",
        FlatLoop<1>(temperature.extent(0)),
        KOKKOS_LAMBDA (i64 ks, f64& running_sum) {
            const fp_t T = temperature(ks);
            if (have_temperature_threshold && T >= threshold) {
                return;
            }
            running_sum += temperature(ks);
        },
        Kokkos::Sum<f64>(result)
    );
    i64 num_active;
    if (!have_temperature_threshold) {
        num_active = temperature.extent(0);
    } else {
        dex_parallel_reduce(
            "Active cell sum",
            FlatLoop<1>(temperature.extent(0)),
            KOKKOS_LAMBDA (i64 ks, i64& running_sum) {
                if (temperature(ks) < threshold) {
                    running_sum += 1;
                }
            },
            Kokkos::Sum<i64>(num_active)
        );
    }
    return result / num_active;
}

fp_t DexInterface::min_characteristic_cooling_time() {
    JasUnpack(state, atmos, pops, adata, rad_loss);

    constexpr fp_t gamma = (FP(5.0) / FP(3.0));
    constexpr fp_t igm1 = FP(1.0) / (gamma - FP(1.0));
    using namespace ConstantsFP;

    if (state.config.rad_loss == RadLossType::PerWavelength) {
        throw std::runtime_error("Currently only supporting Integrated rad loss.");
    }

    fp_t result;
    dex_parallel_reduce(
        "Max characteristic cooling",
        FlatLoop<1>(pops.extent(1)),
        KOKKOS_LAMBDA (i64 ks, fp_t& running_min) {
            fp_t e_int = igm1 * atmos.pressure(ks);
            for (int i = 0; i < adata.energy.extent(0); ++i) {
                e_int += pops(i, ks) * adata.energy(i) * eV;
            }
            fp_t cooling_time = e_int / std::abs(rad_loss(0, ks) * 1e3_fp);
            running_min = std::min(running_min, cooling_time);
        },
        Kokkos::Min<fp_t>(result)
    );
    return result;
}

void DexInterface::update_temperature_rad_eq(fp_t delta_t) {
    JasUnpack(state, atmos);
    const fp_t threshold = state.config.threshold_temperature;

    if (state.config.rad_loss == RadLossType::PerWavelength) {
        throw std::runtime_error("Currently only supporting Integrated rad loss.");
    }

    decltype(state.rad_loss) rad_loss = state.rad_loss;
    if (state.config.store_J_on_cpu) {
        rad_loss = state.rad_loss_cpu.createDeviceCopy();
    }

    fp_t max_temp_change;
    dex_parallel_reduce(
        "Update temperature (rad loss)",
        FlatLoop<1>(rad_loss.extent(1)),
        KOKKOS_LAMBDA (i64 ks, fp_t& running_max) {
            const fp_t L = rad_loss(0, ks) * 1e3_fp; // Calculated in kW/m3
            const fp_t T = atmos.temperature(ks);

            const fp_t temperature_update = (FP(2.0) / FP(5.0)) * L * T / atmos.pressure(ks) * delta_t;
            if (threshold > FP(0.0) && T < threshold) {
                atmos.temperature(ks) -= temperature_update;
                running_max = std::max(running_max, std::abs(temperature_update));
            }
        },
        Kokkos::Max<fp_t>(max_temp_change)
    );
    fmt::println("Max temperature change: {} K", max_temp_change);
}

}
