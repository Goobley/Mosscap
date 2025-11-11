#if !defined(MOSSCAP_DEXRT_EOS_HPP)
#define MOSSCAP_DEXRT_EOS_HPP

#include "Types.hpp"
#include "Simulation.hpp"

namespace Mosscap {

struct DexPressureEos {
    fp_t temperature_threshold = 2.5e3_fp;

    bool init(fp_t temperature_threshold_ = 2.5e3_fp) {
        temperature_threshold = temperature_threshold_;
        return true;
    }

    template <typename FTraits>
    void update_eos(const Simulation& sim) const {
        if (!sim.dex.interface_config.enable || !sim.dex.interface_config.advect) {
            return;
        }

        JasUnpack(sim.dex.state, mr_block_map);
        const auto& block_map = mr_block_map.block_map;
        const auto& Q = sim.state.Q;
        const auto& sz = sim.state.sz;
        const i32 ne_idx = sim.dex.interface_config.field_start_idx;
        const fp_t mu0 = sim.state.mu0;

        constexpr fp_t m_p = ConstantsF64::u;
        constexpr fp_t k_B = ConstantsF64::k_B;
        const auto& eos = sim.eos;
        const auto& temperature_threshold = this->temperature_threshold;
        using Cons = typename FTraits::cons;

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
                QtyView Qv(Q, idx);

                // const fp_t pressure = (atmos.nh_tot(ks) * total_abund + atmos.ne(ks)) * k_B * atmos.temperature(ks);
                // const fp_t y = atmos.ne(ks) / (atmos.nh_tot(ks) * total_abund);
                const fp_t nh_tot = Qv(I(Cons::Rho)) / (eos.avg_mass * m_p);
                const fp_t prev_y = eos.y_space(idx.k, idx.j, idx.i);
                const fp_t y = Qv(ne_idx) / nh_tot;
                // const fp_t pressure_ratio = (1.0_fp + y / total_abund) / (1.0_fp + prev_y / total_abund);
                // NOTE(cmo): Accounts for pressure change due to ionisation
                // const fp_t delta_E_factor = (y - prev_y) / (total_abund + prev_y);
                eos.y_space(idx.k, idx.j, idx.i) = y;

                const fp_t rho = Qv(I(Cons::Rho));
                fp_t mom2_sum = square(Qv(I(Cons::MomX)));
                if constexpr (FTraits::is_mhd || FTraits::num_dim > 1) {
                    mom2_sum += square(Qv(I(Cons::MomY)));
                }
                if constexpr (FTraits::is_mhd || FTraits::num_dim > 2) {
                    mom2_sum += square(Qv(I(Cons::MomZ)));
                }
                const fp_t e_kin = 0.5_fp * mom2_sum / rho;
                fp_t e_mag = 0.0_fp;
                JasUse(mu0);
                if constexpr (FTraits::is_mhd) {
                    e_mag = (square(Qv(I(Cons::Bx))) + square(Qv(I(Cons::By))) + square(Qv(I(Cons::Bz)))) / (2.0_fp * mu0);
                }
                const fp_t eint = Q(I(Cons::Ene), idx.k, idx.j, idx.i) - e_kin - e_mag;
                const fp_t prev_pressure = (eos.gamma - 1.0_fp) * eint;
                const fp_t temperature = temperature_si(prev_pressure, total_abund * nh_tot, prev_y);
                const fp_t limited_temperature = std::max(temperature, temperature_threshold);
                // const fp_t delta_eint = eint * delta_E_factor;
                const fp_t new_pressure = nh_tot * (total_abund + y) * limited_temperature * k_B;

                Q(I(Cons::Ene), idx.k, idx.j, idx.i) = new_pressure / (eos.gamma - 1.0_fp) + e_kin + e_mag;
            }
        );
        Kokkos::fence();

    }
};

}

#else
#endif