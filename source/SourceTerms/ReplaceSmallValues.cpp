#include "ReplaceSmallValues.hpp"
#include "../MosscapConfig.hpp"
#include "../SourceTerms.hpp"

namespace Mosscap {

template <typename FTraits>
void replace_small_values_kernel(const Simulation& sim, const ReplaceSmallValuesContext& small_vals) {
    using Cons = typename FTraits::cons;
    constexpr int NumDim = FTraits::is_mhd ? 3 : FTraits::num_dim;

    const auto& Q = sim.state.Q;
    const auto& S = sim.sources.S;
    const auto& sz = sim.state.sz;
    const fp_t dt_sub = sim.dt_sub;
    const auto& eos = sim.eos;
    const fp_t mu0 = sim.state.mu0;

    dex_parallel_for(
        "Replace small values",
        FlatLoop<3>(sz.zc, sz.yc, sz.xc),
        KOKKOS_LAMBDA (int k, int j, int i) {
            QtyView q(Q, CellIndex{.i=i, .j=j, .k=k});
            fp_t rho = q(I(Cons::Rho));
            fp_t mom_x = q(I(Cons::MomX));
            fp_t mom_y = (NumDim > 1) ? q(I(Cons::MomY)) : 0.0_fp;
            fp_t mom_z = (NumDim > 1) ? q(I(Cons::MomZ)) : 0.0_fp;
            if (q(I(Cons::Rho)) < small_vals.density_floor) {
                // Restore the value via the source term (modulo flux divergence and other sources)
                S(I(Cons::Rho), k, j, i) += (small_vals.density_floor - q(I(Cons::Rho))) / dt_sub;
                rho = small_vals.density_floor;
                if (small_vals.zero_momentum) {
                    S(I(Cons::MomX), k, j, i) += -q(I(Cons::MomX)) / dt_sub;
                    mom_x = 0.0_fp;
                    if constexpr (NumDim > 1) {
                        S(I(Cons::MomY), k, j, i) += -q(I(Cons::MomY)) / dt_sub;
                        mom_y = 0.0_fp;
                    }
                    if constexpr (NumDim > 2) {
                        S(I(Cons::MomZ), k, j, i) += -q(I(Cons::MomZ)) / dt_sub;
                        mom_z = 0.0_fp;
                    }
                }
            }
            fp_t mom2_sum = square(q(I(Cons::MomX)));
            if constexpr (NumDim > 1) {
                mom2_sum += square(q(I(Cons::MomY)));
            }
            if constexpr (NumDim > 2) {
                mom2_sum += square(q(I(Cons::MomZ)));
            }
            JasUse(mu0);
            fp_t e_mag = 0.0_fp;
            if constexpr (FTraits::is_mhd) {
                e_mag = square(q(I(Cons::Bx))) + square(q(I(Cons::By))) + square(q(I(Cons::Bz)));
                e_mag /= (2.0_fp * mu0);
            }
            const fp_t e_floor = small_vals.pressure_floor / (eos.gamma - 1.0_fp) + (
                q(I(Cons::Rho)) * q(I(Cons::IonE))
                + 0.5_fp * mom2_sum / q(I(Cons::Rho))
                + e_mag
            );

            if (q(I(Cons::Ene)) < e_floor) {
                const fp_t e_target = small_vals.pressure_floor / (eos.gamma - 1.0_fp) + (
                    rho * q(I(Cons::IonE))
                    + 0.5_fp * (square(mom_x) + square(mom_y) + square(mom_z)) / rho
                    + e_mag
                );
                S(I(Cons::Ene), k, j, i) += (e_target - q(I(Cons::Ene))) / dt_sub;
            }
        }
    );
    Kokkos::fence();

}

void setup_replace_small_values(Simulation& sim, YAML::Node& config) {
    auto ctx = std::make_shared<ReplaceSmallValuesContext>(
        ReplaceSmallValuesContext{
            .density_floor = get_or<fp_t>(config, "sources.replace_small_values.density_floor", 0.0_fp),
            .pressure_floor = get_or<fp_t>(config, "sources.replace_small_values.pressure_floor", 0.0_fp),
            .zero_momentum = get_or<bool>(config, "sources.replace_small_values.zero_momentum", true)
        }
    );

    if (source_term_index(sim, "replace_small_values") != sim.compute_source_terms.size()) {
        throw std::runtime_error("Source \"replace_small_values\" already registered.");
    }

    sim.compute_source_terms.push_back(SourceTerm{
        .name = "replace_small_values",
        .fn = invoke_fluid_traits(
            sim.num_dim,
            sim.fluid_type,
            [=]<typename FTraits>(FTraits) -> std::function<void(const Simulation&)> {
                return [=] (const Simulation& sim) {
                    return replace_small_values_kernel<FTraits>(sim, *ctx);
                };
            }
        ),
        .get_context = [=]() { return ctx.get(); }
    });
}

}