#if !defined(MOSSCAP_DEXRT_EOS_HPP)
#define MOSSCAP_DEXRT_EOS_HPP

#include "Types.hpp"
#include "Simulation.hpp"

namespace Mosscap {

struct DexPressureEosOptions {
    bool use_tracer_energy = true;
    bool use_tracer_charge = true;
    bool renorm_tracer_hpops = true;
    bool fully_rebuild_etot = true;
};

struct DexPressureEos {
    fp_t temperature_threshold = 2.5e3_fp;
    DexPressureEosOptions opts;
    Fp1d tracer_energy;
    Fp1d tracer_charge;
    yakl::Array<i32, 1, yakl::memDevice> tracer_is_h;

    bool init(
        const Simulation& sim,
        fp_t temperature_threshold_ = 2.5e3_fp,
        const DexPressureEosOptions& opts_ = DexPressureEosOptions()
    ) {
        temperature_threshold = temperature_threshold_;
        opts = opts_;
        const auto& adata = sim.dex.state.adata_host;
        i32 num_tracers = adata.energy.extent(0) + 1;
        Fp1dHost t_energy("tracer_energy", num_tracers);
        Fp1dHost t_charge("tracer_charge", num_tracers);
        yakl::Array<i32, 1, yakl::memHost> t_is_h("tracer_is_h", num_tracers);
        for (int i = 0; i < t_energy.extent(0); ++i) {
            t_energy(i) = 0.0_fp;
            t_charge(i) = 0.0_fp;
            t_is_h(i) = 0;
        }
        // NOTE(cmo): ne is essentially ignored here, and we use the other terms
        // to reconstruct energy etc.
        i32 h_start_level = 0;
        i32 h_end_level = 0;
        for (int ia = 0; ia < adata.Z.extent(0); ++ia) {
            if (adata.Z(ia) != 1) {
                continue;
            }

            h_start_level = adata.level_start(ia);
            h_end_level = adata.level_start(ia) + adata.num_level(ia);
            break;
        }
        for (int i = 1; i < t_energy.extent(0); ++i) {
            t_energy(i) = adata.energy(i-1) * ConstantsF64::eV;
            t_charge(i) = adata.stage(i-1) - 1.0_fp;
            t_is_h(i) = ((i - 1) >= h_start_level) && ((i - 1) < h_end_level);
        }

        tracer_energy = t_energy.createDeviceCopy();
        tracer_charge = t_charge.createDeviceCopy();
        tracer_is_h = t_is_h.createDeviceCopy();
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
        constexpr fp_t chi_H = 2.178710282685096e-18_fp; // [J]
        const auto& eos = sim.eos;
        JasUse((*this), temperature_threshold, tracer_energy, tracer_charge, tracer_is_h);
        const auto& temperature_threshold = this->temperature_threshold;
        const auto& opts = this->opts;
        const auto& tracer_energy = this->tracer_energy;
        const auto& tracer_charge = this->tracer_charge;
        const auto& tracer_is_h = this->tracer_is_h;
        using Cons = typename FTraits::cons;

        const fp_t total_abund = eos.total_abund;

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

                const fp_t nh_tot = Qv(I(Cons::Rho)) / (eos.avg_mass * m_p);

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
                const fp_t prev_y = eos.y_space(idx.k, idx.j, idx.i);
                const fp_t prev_pressure = (eos.gamma - 1.0_fp) * (Qv(I(Cons::Ene)) - e_kin - e_mag - Qv(I(Cons::IonE)) * rho);
                const fp_t temperature = temperature_si(prev_pressure, total_abund * nh_tot, prev_y);

                if (opts.renorm_tracer_hpops) {
                    fp_t nh_from_tracer = 0.0_fp;
                    for (int l = 0; l < tracer_is_h.extent(0); ++l) {
                        if (tracer_is_h(l) == 1) {
                            nh_from_tracer += Qv(l + ne_idx);
                        }
                    }
                    const fp_t tracer_err = nh_tot / nh_from_tracer;
                    for (int l = 0; l < tracer_is_h.extent(0); ++l) {
                        Qv(l + ne_idx) *= tracer_err;
                    }
                }

                if (opts.use_tracer_charge) {
                    fp_t ne = 0.0_fp;
                    for (int l = 0; l < tracer_charge.extent(0); ++l) {
                        ne += tracer_charge(l) * Qv(l + ne_idx);
                    }
                    Qv(ne_idx) = ne;
                }
                if (opts.use_tracer_energy) {
                    fp_t ion_e = 0.0_fp;
                    for (int l = 0; l < tracer_energy.extent(0); ++l) {
                        ion_e = tracer_energy(l) * Qv(l + ne_idx);
                    }
                    Qv(I(Cons::IonE)) = ion_e / rho;
                } else {
                    Qv(I(Cons::IonE)) = Qv(ne_idx) * chi_H / rho;
                }

                const fp_t y = Qv(ne_idx) / nh_tot;
                eos.y_space(idx.k, idx.j, idx.i) = y;

                if (opts.fully_rebuild_etot) {
                    const fp_t limited_temperature = std::max(temperature_threshold, temperature);
                    Qv(I(Cons::Ene)) = (
                        1.0_fp / (eos.gamma - 1.0_fp) * (total_abund + y) * nh_tot * k_B * limited_temperature
                        + Qv(I(Cons::IonE)) * rho
                        + e_kin
                        + e_mag
                    );
                } else {
                    const fp_t eint = Q(I(Cons::Ene), idx.k, idx.j, idx.i) - e_kin - e_mag;
                    // NOTE(cmo): Minimum change to make energy consistent with
                    // ionisation state (energy changes handled in radiative losses)
                    const fp_t new_eint = eint * (1.0_fp + y) / (1.0_fp + prev_y);
                    Qv(I(Cons::Ene)) = new_eint + e_kin + e_mag;
                }
            }
        );
        Kokkos::fence();

    }
};

}

#else
#endif