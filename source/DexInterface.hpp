#if !defined(MOSSCAP_DEX_INTERFACE)
#define MOSSCAP_DEX_INTERFACE

#include "Config.hpp"
#include "AtmosCommon.hpp"
#include "../DexRT/source/State.hpp"
#include "../DexRT/source/CascadeState.hpp"

#ifdef DEXRT_USE_MAGMA
#error "MAGMA not supported for Dex in the Mosscap integration"
#endif

namespace YAML { struct Node; }

namespace Mosscap {

// NOTE(cmo): Only supporting 2D for now
using DexState = ::State;
using DexCascState = ::CascadeState;
using DexFp1d = ::Fp1d;
using DexFp2d = ::Fp2d;

struct DexMosscapConfig {
    bool advect = false;
    bool enable = false;
    bool rad_loss = false;
    bool time_dependent_updates = false;
    bool update_ion_e = true;
    bool ignore_rt_velocities = false;
    i32 max_mip_level = 0;
    i32 field_start_idx = 0;
    fp_t theta = 1.0_fp;
    fp_t temperature_floor = 2e3_fp;
    // NOTE(claude): When set, only couple a bounding box around the active
    // (cool) tiles to DexRT rather than the whole inner grid, so rays escape and
    // sample the boundary condition at a nearer edge. The box is recomputed each
    // RT update and halo'd out by bbox_halo_cells (snapped to BLOCK_SIZE, clamped
    // to the grid). Output is still written as-if the full grid were coupled.
    bool bbox_crop = false;
    i32 bbox_halo_cells = 16;
};

struct DexConvergence {
    Dex::fp_t convergence;
    i32 max_iter;
};

/// A tile-space bounding box in full-grid tile coordinates: origin (tx0, tz0)
/// and box tile dimensions (bnx, bnz). Used by the bbox_crop coupling.
struct TileBbox {
    i32 tx0;
    i32 tz0;
    i32 bnx;
    i32 bnz;
};

struct IterateArgs {
    fp_t dt = 0.0_fp; /// 0.0 implies stat eq
    fp_t theta = 1.0_fp; /// Theta in semi-implicit Euler method
    bool first_iter = false;
};

struct Simulation;

struct DexInterface {
    DexMosscapConfig interface_config;
    DexState state;
    DexCascState casc_state;
    i32 num_iter;
    DexFp2d prev_pops; /// Used for time dependent updates

    /// Mapping from the (possibly cropped) solve block map back to the full
    /// inner grid, used only at output time to write results as-if the whole
    /// grid were coupled. When bbox_crop is off these are (0, 0) and the full
    /// tile counts, so the output promotion is an identity. Set on rank 0 by
    /// the (re)build of the atmosphere geometry.
    i32 box_tile_origin_x = 0;
    i32 box_tile_origin_z = 0;
    i32 full_num_x_tiles = 0;
    i32 full_num_z_tiles = 0;

    /// Per-level decomposition of the energy stored in the atomic reservoir,
    /// relative to the ground state of each model atom's own lowest stage.
    /// chi_lut + e_exc_lut == adata.energy by construction, converted from the
    /// eV adata stores to J. [J]
    DexFp1d chi_lut; /// Ionisation energy of the level's ion stage
    DexFp1d e_exc_lut; /// Excitation energy above the level's own stage ground
    /// Reservoir energies sampled immediately before the NEQ solve [J m-3]
    DexFp1d res_e_ion_pre;
    DexFp1d res_e_exc_pre;
    /// Rate at which the NEQ solve released reservoir energy to the gas
    /// (positive when recombining / de-exciting) [W m-3]
    DexFp1d g_ion;
    DexFp1d g_exc;
    /// Energy injected by the temperature floor in integrate_rad_loss_split,
    /// positive when the floor is heating the gas [W m-3]
    DexFp1d temp_floor_heat;

    ~DexInterface();

    bool init_config(Simulation& sim, YAML::Node& config, const std::string& config_path);

    bool init(Simulation& sim, YAML::Node& config);
    template <typename FTraits>
    bool init(Simulation& sim, YAML::Node& config);

    bool init_atmosphere(Simulation& sim, i32 max_mip_level);
    template <typename FTraits>
    bool init_atmosphere(Simulation& sim, i32 max_mip_level);

    bool update_atmosphere(Simulation& sim);
    template <typename FTraits>
    bool update_atmosphere(Simulation& sim);

    /// Compute the active-cell tile bounding box (in full-grid tile coords) for
    /// the bbox_crop mode, halo'd by bbox_halo_cells and clamped/snapped to the
    /// grid. When bbox_crop is off, or nothing is active, this returns the full
    /// grid so the solve degrades to the uncropped behaviour.
    template <typename FTraits>
    TileBbox compute_active_tile_bbox(Simulation& sim);

    /// Rebuild the sparse block map + sparse atmosphere over the given box,
    /// shifting the atmosphere offsets so the active region keeps its absolute
    /// physical placement. Used by the bbox_crop paths of
    /// init_atmosphere/update_atmosphere. Records the box->full-grid mapping
    /// (box_tile_origin_*, full_num_*_tiles) for output.
    template <typename FTraits>
    bool rebuild_block_map_and_atmos(Simulation& sim, const TileBbox& box, i32 max_mip_level);

    /// (Re)allocate the geometry-sized cascade storage: c0_size, max_block_mip
    /// and the cascade buffers. Does NOT touch the per-active-cell terms (pops
    /// etc.), so it is safe to call on MPI workers after pops has been received.
    /// Reads state.atmos/mr_block_map.
    void reallocate_cascade_storage();

    /// (Re)allocate the full solver-side state whose size tracks the atmosphere
    /// geometry: the per-active-cell terms and reservoir terms, plus
    /// reallocate_cascade_storage. Needed each step under a moving bbox because
    /// the box dimensions (and hence cascade storage) change. Reads
    /// state.atmos/mr_block_map.
    void reallocate_solver_state();

    bool iterate(const DexConvergence& tol, const IterateArgs& args = IterateArgs());

    void copy_nhtot_to_rho(const Simulation& sim);
    template <typename FTraits>
    void copy_nhtot_to_rho(const Simulation& sim);

    void copy_pops_to_aux_fields(const Simulation&);
    template <typename FTraits>
    void copy_pops_to_aux_fields(const Simulation&);

    void copy_pops_from_aux_fields(const Simulation&);
    template <typename FTraits>
    void copy_pops_from_aux_fields(const Simulation&);

    void copy_to_eos(const Simulation&);
    template <typename FTraits>
    void copy_to_eos(const Simulation&);

    void lte_init_aux_fields(const Simulation&);
    template <typename FTraits>
    void lte_init_aux_fields(const Simulation&);

    void write_output(const Simulation&, yakl::SimpleNetCDF&);

    void integrate_rad_loss_split(const Simulation& sim);
    template <typename FTraits>
    void integrate_rad_loss_split(const Simulation& sim);

    /// Build chi_lut/e_exc_lut from adata_host. Called once, at init.
    void init_reservoir_luts();
    /// (Re)allocate the per-active-cell reservoir diagnostics. The active cell
    /// count changes as the atmosphere is updated.
    void allocate_reservoir_terms(i64 num_active_cells);
    /// Overwrite e_ion/e_exc with the reservoir energy currently held in
    /// state.pops, summed over the levels of every atom [J m-3]
    void compute_reservoir_energies(const DexFp1d& e_ion, const DexFp1d& e_exc);
    /// Sample the reservoir energies immediately before the NEQ solve
    void snapshot_reservoir_energies();
    /// Difference against the pre-solve sample to form g_ion/g_exc [W m-3]
    void evaluate_reservoir_rates(fp_t dt);

    void run_worker_loop();
    void initial_worker_atmos_setup();
    void broadcast_atmosphere();

    fp_t min_characteristic_cooling_time();
    fp_t mean_temperature();
    void update_temperature_rad_eq(fp_t delta_t);
};

template <typename T, typename U, int rank, int mem_space>
inline
yakl::Array<U, rank, mem_space> maybe_convert_fp_array(const yakl::Array<T, rank, mem_space>& in) {
    if constexpr (std::is_same<T, U>::value) {
        return in;
    }

    yakl::Array<U, rank, mem_space> result(in.label(), yakl::DimsT<i64>(in.get_dimensions()));
    Kokkos::parallel_for<
        std::conditional_t<
            mem_space == yakl::memDevice,
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultHostExecutionSpace
        >
    >(
        "Convert and copy array",
        in.size(),
        KOKKOS_LAMBDA (i64 i) {
            result.data()[i] = U(in.data()[i]);
        }
    );
    Kokkos::fence();
    return result;
}

}

#else
#endif