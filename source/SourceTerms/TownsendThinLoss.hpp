#if !defined(MOSSCAP_TOWNSEND_THIN_LOSS_HPP)
#define MOSSCAP_TOWNSEND_THIN_LOSS_HPP
#include "../Simulation.hpp"


namespace YAML { class Node; };

namespace Mosscap {

struct ThinLossContext {
    Fp1d temps;
    Fp1d lambdas;
    Fp1d Y_k;
    Fp1d alpha_k;
    fp_t min_temperature;
};

void setup_thin_loss(Simulation& sim, YAML::Node& config);

template <typename FTraits, typename QType>
fp_t thin_loss_single_val(
    const Simulation& sim,
    const ThinLossContext& ctx,
    const QType& q,
    const fp_t ion_frac=1.0_fp
) {
    using Cons = typename FTraits::cons;
    using Prim = typename FTraits::prim;
    constexpr int n_hydro = FTraits::num_vars;
    constexpr fp_t m_p = ConstantsF64::u;
    constexpr fp_t k_B = ConstantsF64::k_B;

    JasUnpack(sim, state, eos, dt_sub);
    JasUnpack(state, mu0);
    const int n_temps = ctx.temps.extent(0);
    const int n_bins = ctx.alpha_k.extent(0);

    Fp1d result("thin loss result", 1);
    Kokkos::fence();
    dex_parallel_for(
        "Compute thin loss",
        FlatLoop<1>(1),
        KOKKOS_LAMBDA (int i) {
            yakl::SArray<fp_t, 1, n_hydro> w;
            cons_to_prim<FTraits>(eos.gamma, mu0, q, w);

            const fp_t nh_tot = w(I(Prim::Rho)) / (eos.mass_per_h * m_p);
            fp_t y = eos.y;
            if (!eos.is_constant) {
                y = ion_frac;
            }
            auto temperature = temperature_si(w(I(Prim::Pres)), nh_tot, eos.total_abund, y);
            fp_t ne = y * nh_tot;
            if (temperature < ctx.min_temperature) {
                return;
            }

            // Find temperature bin
            int idx = 0;
            while ((idx < n_bins - 1) && (ctx.temps(idx + 1) < temperature)) {
                idx += 1;
            }

            const fp_t alpha_k_m1 = ctx.alpha_k(idx) - 1.0_fp;
            const fp_t tef = ctx.Y_k(idx) + (
                (ctx.lambdas(n_temps - 1) / ctx.lambdas(idx))
                * (ctx.temps(idx) / ctx.temps(n_temps - 1))
                * (std::pow(ctx.temps(idx) / temperature, alpha_k_m1) - 1.0) / alpha_k_m1
            );
            const fp_t tef_adj = (
                tef
                + ctx.lambdas(n_temps - 1) * dt_sub / ctx.temps(n_temps - 1)
                * (nh_tot * ne) / (nh_tot + ne) * (eos.gamma - 1.0_fp) / k_B
            );
            while ((idx > 0) && (tef_adj > ctx.Y_k(idx))) {
                idx -= 1;
            }

            fp_t new_temperature = ctx.temps(idx) * std::pow(
                (
                    1.0_fp - (1.0_fp - ctx.alpha_k(idx))
                    * (ctx.lambdas(idx) / ctx.lambdas(n_temps - 1))
                    * (ctx.temps(n_temps - 1) / ctx.temps(idx))
                    * (tef_adj - ctx.Y_k(idx))
                ),
                1.0_fp / (1.0_fp - ctx.alpha_k(idx))
            );
            new_temperature = std::max(new_temperature, ctx.min_temperature);
            const fp_t delta_temp = new_temperature - temperature;
            const fp_t delta_e = 1.0_fp / (eos.gamma - 1.0_fp) * (nh_tot + ne) * k_B * delta_temp;
            result(i) = delta_e / dt_sub;
        }
    );
    Kokkos::fence();
    return result.createHostCopy()(0);
}

}

#else
#endif