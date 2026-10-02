// NOTE(claude): Work in progress -- not yet tested.
#include "Friction.hpp"
#include "../Simulation.hpp"
#include "../MosscapConfig.hpp"
#include "../SourceTerms.hpp"

namespace Mosscap {

template <typename FTraits>
void friction_kernel(const Simulation& sim, const FrictionVals& fric) {
    using Cons = typename FTraits::cons;

    const auto& Q = sim.state.Q;
    const auto& S = sim.sources.S;
    const auto& sz = sim.state.sz;
    const fp_t gamma = fric.gamma;

    dex_parallel_for(
        "Apply friction",
        FlatLoop<3>(sz.zc, sz.yc, sz.xc),
        KOKKOS_LAMBDA (int k, int j, int i) {
            constexpr i32 NumDim = FTraits::num_dim;
            const fp_t rho = Q(I(Cons::Rho), k, j, i);
            const fp_t mx = Q(I(Cons::MomX), k, j, i);
            S(I(Cons::MomX), k, j, i) -= gamma * mx;
            fp_t mom2 = mx * mx;
            if constexpr (NumDim > 1) {
                const fp_t my = Q(I(Cons::MomY), k, j, i);
                S(I(Cons::MomY), k, j, i) -= gamma * my;
                mom2 += my * my;
            }
            if constexpr (NumDim > 2) {
                const fp_t mz = Q(I(Cons::MomZ), k, j, i);
                S(I(Cons::MomZ), k, j, i) -= gamma * mz;
                mom2 += mz * mz;
            }
            // Remove the associated bulk kinetic energy (leaves internal energy
            // untouched rather than converting it to heat).
            S(I(Cons::Ene), k, j, i) -= gamma * mom2 / rho;
        }
    );
    Kokkos::fence();
}

void setup_friction(Simulation& sim, YAML::Node& config) {
    auto fric = std::make_shared<FrictionVals>(FrictionVals{
        .gamma = get_or<fp_t>(config, "sources.friction.gamma", 0.0_fp)
    });

    auto apply_friction = invoke_fluid_traits(
        sim.num_dim,
        sim.fluid_type,
        [=]<typename FTraits>(FTraits) -> std::function<void(const Simulation&)> {
            return [=] (const Simulation& sim) {
                return friction_kernel<FTraits>(sim, *fric);
            };
        }
    );

    if (source_term_index(sim, "friction") != sim.compute_source_terms.size()) {
        throw std::runtime_error("Source \"friction\" already registered.");
    }

    sim.compute_source_terms.push_back(SourceTerm{
        .name = "friction",
        .fn = apply_friction,
        .get_context = [=]() { return fric.get(); }
    });
}

}
