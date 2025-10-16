#if !defined(MOSSCAP_DEXRT_EOS_HPP)
#define MOSSCAP_DEXRT_EOS_HPP

#include "Types.hpp"
#include "Simulation.hpp"

namespace Mosscap {

struct DexPressureEos {
    bool init() {
        return true;
    }

    void update_eos(const Simulation& sim) const {
        if (!sim.dex.interface_config.enable || !sim.dex.interface_config.advect) {
            return;
        }

        JasUnpack(sim.dex.state, mr_block_map);
        const auto& block_map = mr_block_map.block_map;
        const auto& Q = sim.state.Q;
        const auto& sz = sim.state.sz;
        const i32 ne_idx = sim.dex.interface_config.field_start_idx;

        constexpr fp_t m_p = ConstantsF64::u;
        // constexpr fp_t k_B = ConstantsF64::k_B;
        const auto& eos = sim.eos;
        constexpr i32 num_dim = 2;
        using Cons = Cons<num_dim>;

        // TODO(cmo): Pull this out and set it somewhere
        constexpr fp_t total_abund = 1.0_fp;

        // NOTE(cmo): Idea to only update blocks where dex is active
        dex_parallel_for(
            "update eos",
            FlatLoop<2>(block_map.loop_bounds()),
            KOKKOS_LAMBDA (i64 tile_idx, i32 block_idx) {
                IdxGen idx_gen(mr_block_map);
                const i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
                Coord2 coord = idx_gen.loop_coord(tile_idx, block_idx);
                CellIndex idx{.i = coord.x + sz.ng, .j = coord.z + sz.ng, .k = 0};

                // const fp_t pressure = (atmos.nh_tot(ks) * total_abund + atmos.ne(ks)) * k_B * atmos.temperature(ks);
                // const fp_t y = atmos.ne(ks) / (atmos.nh_tot(ks) * total_abund);
                const fp_t nh_tot = Q(I(Cons::Rho), idx.k, idx.j, idx.i) / (eos.avg_mass * m_p);
                const fp_t prev_y = eos.y_space(idx.k, idx.j, idx.i);
                const fp_t y = Q(ne_idx, idx.k, idx.j, idx.i) / (nh_tot * total_abund);
                // const fp_t pressure_ratio = (1.0_fp + y / total_abund) / (1.0_fp + prev_y / total_abund);
                // NOTE(cmo): Accounts for pressure change due to ionisation
                const fp_t delta_E_factor = (y - prev_y) / (total_abund + prev_y);
                eos.y_space(idx.k, idx.j, idx.i) = y;

                const fp_t rho = Q(I(Cons::Rho), idx.k, idx.j, idx.i);
                fp_t mom2_sum = square(Q(I(Cons::MomX), idx.k, idx.j, idx.i));
                mom2_sum += square(Q(I(Cons::MomY), idx.k, idx.j, idx.i));
                const fp_t e_kin = 0.5_fp * mom2_sum / rho;
                const fp_t eint = Q(I(Cons::Ene), idx.k, idx.j, idx.i) - e_kin;
                const fp_t delta_eint = eint * delta_E_factor;

                fp_t ene_pre = Q(I(Cons::Ene), idx.k, idx.j, idx.i);
                Q(I(Cons::Ene), idx.k, idx.j, idx.i) += delta_eint;
            }
        );
        Kokkos::fence();

    }
};

}

#else
#endif