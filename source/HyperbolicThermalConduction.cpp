#include "HyperbolicThermalConduction.hpp"
#include "Simulation.hpp"
#include "MosscapConfig.hpp"
#include "SourceTerms.hpp"
#include "Boundaries.hpp"

namespace Mosscap {

    template <typename FTraits>
    void hypertc_update_heatf(const Simulation& sim) {
        JasUnpack(sim, state, eos, sources, dt, max_cfl);
        JasUnpack(state, sz, W, p_mass, cond, dx, glm_ch);
        const auto& S = sources.S;
        const fp_t inv_dx = 1.0_fp / dx;
        Fp3d temperature;
        if (sim.eos.is_constant) {
            temperature = Fp3d("temperature", sz.zc, sz.yc, sz.xc);
            dex_parallel_for(
                "Compute temperature",
                FlatLoop<3>(sz.zc, sz.yc, sz.xc),
                KOKKOS_LAMBDA (int k, int j, int i) {
                    using Prim = typename FTraits::prim;
                    const fp_t nh_tot = W(I(Prim::Rho), k, j, i) / (eos.avg_mass * p_mass);
                    const fp_t pressure = W(I(Prim::Pres), k, j, i);
                    temperature(k, j, i) = temperature_si(pressure, nh_tot, eos.y);
                }
            );
            Kokkos::fence();
        } else if (sim.eos.T_space.initialized()) {
            temperature = sim.eos.T_space;
        } else {
            temperature = Fp3d("temperature", sz.zc, sz.yc, sz.xc);
            dex_parallel_for(
                "Compute temperature",
                FlatLoop<3>(sz.zc, sz.yc, sz.xc),
                KOKKOS_LAMBDA (int k, int j, int i) {
                    using Prim = typename FTraits::prim;
                    const fp_t nh_tot = W(I(Prim::Rho), k, j, i) / (eos.avg_mass * p_mass);
                    const fp_t pressure = W(I(Prim::Pres), k, j, i);
                    const fp_t y = eos.y_space(k, j, i);
                    temperature(k, j, i) = temperature_si(pressure, nh_tot, y);
                }
            );
            Kokkos::fence();
        }

        int nx = sz.xc - 2 * sz.ng;
        int ny = std::max(sz.yc - 2 * sz.ng, 1);
        int nz = std::max(sz.zc - 2 * sz.ng, 1);
        dex_parallel_for(
            "Compute q (HeatF) source",
            FlatLoop<3>(nz, ny, nx),
            KOKKOS_LAMBDA (int ki, int ji, int ii) {
                using Prim = typename FTraits::prim;
                using Cons = typename FTraits::cons;
                const int k = nz == 1 ? ki : ki + sz.ng;
                const int j = ny == 1 ? ji : ji + sz.ng;
                const int i = ii + sz.ng;

                const fp_t temp = temperature(k, j, i);
                fp_t sigma_T_52 = cond.hypertc_kappa;
                if (cond.spitzer) {
                    sigma_T_52 *= square(temp) * std::sqrt(temp);
                }

                // NOTE(cmo): Use central 4th order differences to
                // evaluate grad T. This is similar
                // to MURaM's approach.
                // In this case grad T_i = 8/(12dx) (T_{i+1} - T_{i-1}) - 1/(12dx) (T_{i+2} - T_{i-2})

                constexpr fp_t w1 = 8.0_fp / 12.0_fp;
                constexpr fp_t w2 = 1.0_fp / 12.0_fp;
                fp_t B_gradT = W(I(Prim::Bx), k, j, i) * inv_dx * (
                    w1 * (temperature(k, j, i+1) - temperature(k, j, i-1)) -
                    w2 * (temperature(k, j, i+2) - temperature(k, j, i-2))
                );
                fp_t b2 = square(W(I(Prim::Bx), k, j, i));
                if constexpr (FTraits::num_dim > 1) {
                    B_gradT += W(I(Prim::By), k, j, i) * inv_dx * (
                        w1 * (temperature(k, j+1, i) - temperature(k, j-1, i)) -
                        w2 * (temperature(k, j+2, i) - temperature(k, j-2, i))
                    );
                    b2 += square(W(I(Prim::By), k, j, i));
                }
                if constexpr (FTraits::num_dim > 2) {
                    B_gradT += W(I(Prim::Bz), k, j, i) * inv_dx * (
                        w1 * (temperature(k+1, j, i) - temperature(k-1, j, i)) -
                        w2 * (temperature(k+2, j, i) - temperature(k-2, j, i))
                    );
                    b2 += square(W(I(Prim::Bz), k, j, i));
                }
                const fp_t inv_b_norm = 1.0_fp / std::max(std::sqrt(b2), 1e-60_fp);
                B_gradT *= inv_b_norm;
                const fp_t sigma_T_72 = temp * sigma_T_52;
                fp_t tau = std::max(
                    4.0_fp * dt,
                    // NOTE(cmo): glm_ch _is_ the max wave propagation speed, which is what is needed here.
                    // TODO(cmo): Limiting?
                    sigma_T_72 * square(max_cfl) * (eos.gamma - 1.0_fp) / (W(I(Prim::Pres), k, j, i) * square(glm_ch))
                );
                S(I(Cons::HeatF), k, j, i) -= (sigma_T_52 * B_gradT + W(I(Cons::HeatF), k, j, i)) / tau;
                if constexpr (!HYPERTC_IN_FLUX_VECTOR) {
                    // NOTE(cmo): Add energy term consistent with the 4th order
                    // FD scheme used for evaluating the source of q.
                    auto Bq = [&] (int B, int k, int j, int i) {
                        return W(B, k, j, i) * W(I(Prim::HeatF), k, j, i);
                    };
                    fp_t ene_res = w1 * (Bq(Prim::Bx, k, j, i+1) - Bq(Prim::Bx, k, j, i-1)) - w2 * (Bq(Prim::Bx, k, j, i+2) - Bq(Prim::Bx, k, j, i-2));
                    if constexpr (FTraits::num_dim > 1) {
                        ene_res += w1 * (Bq(Prim::By, k, j+1, i) - Bq(Prim::By, k, j-1, i)) - w2 * (Bq(Prim::By, k, j+2, i) - Bq(Prim::By, k, j-2, i));
                    }
                    if constexpr (FTraits::num_dim > 2) {
                        ene_res += w1 * (Bq(Prim::Bz, k+1, j, i) - Bq(Prim::Bz, k-1, j, i)) - w2 * (Bq(Prim::Bz, k+2, j, i) - Bq(Prim::Bz, k-2, j, i));
                    }
                    ene_res *= inv_dx * inv_b_norm;
                    S(I(Cons::Ene), k, j, i) -= ene_res;
                }
            }
        );
    }

    void setup_hyperbolic_tc(Simulation& sim, YAML::Node& config) {
        FluidTraitsRt traits(sim.num_dim, sim.fluid_type);
        if (!traits.has_hypertc) {
            return;
        }

        auto hypertc_source_fn = select_fluid_traits<const Simulation&>(
            sim.num_dim,
            sim.fluid_type,
            [] <typename FTraits> (FTraits, const Simulation& sim) {
                return hypertc_update_heatf<FTraits>(sim);
            }
        );
        sim.compute_source_terms.push_back(SourceTerm{
            .name = "Hyperbolic thermal conduction",
            .fn = hypertc_source_fn
        });
    }
}