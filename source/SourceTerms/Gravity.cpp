#include "Gravity.hpp"
#include "../Simulation.hpp"
#include "../MosscapConfig.hpp"
#include "../SourceTerms.hpp"

namespace Mosscap {

template <typename FTraits>
void gravity_kernel(const Simulation& sim, const GravityVals& grav) {
    using Cons = typename FTraits::cons;

    const auto& Q = sim.state.Q;
    const auto& S = sim.sources.S;
    const auto& sz = sim.state.sz;

    dex_parallel_for(
        "Apply gravity",
        FlatLoop<3>(sz.zc, sz.yc, sz.xc),
        KOKKOS_LAMBDA (int k, int j, int i) {
            S(I(Cons::MomX), k, j, i) += Q(I(Cons::Rho), k, j, i) * grav.x;
            fp_t energy_update = Q(I(Cons::MomX), k, j, i) * grav.x;
            constexpr i32 NumDim = FTraits::num_dim;
            if constexpr (NumDim > 1) {
                S(I(Cons::MomY), k, j, i) += Q(I(Cons::Rho), k, j, i) * grav.y;
                energy_update += Q(I(Cons::MomY), k, j, i) * grav.y;
            }
            if constexpr (NumDim > 2) {
                S(I(Cons::MomZ), k, j, i) += Q(I(Cons::Rho), k, j, i) * grav.z;
                energy_update += Q(I(Cons::MomZ), k, j, i) * grav.z;
            }
            S(I(Cons::Ene), k, j, i) += energy_update;
        }
    );
    Kokkos::fence();
}

void setup_gravity(Simulation& sim, YAML::Node& config) {

    auto grav = std::make_shared<GravityVals>(GravityVals{
        .x = get_or<fp_t>(config, "sources.gravity.x", -1.0_fp),
        .y = get_or<fp_t>(config, "sources.gravity.y", 0.0_fp),
        .z = get_or<fp_t>(config, "sources.gravity.z", 0.0_fp)
    });

    auto apply_gravity = invoke_fluid_traits(
        sim.num_dim,
        sim.fluid_type,
        [=]<typename FTraits>(FTraits) -> std::function<void(const Simulation&)> {
            return [=] (const Simulation& sim) {
                return gravity_kernel<FTraits>(sim, *grav);
            };
        }
    );

    if (source_term_index(sim, "gravity") != sim.compute_source_terms.size()) {
        throw std::runtime_error("Source \"gravity\" already registered.");
    }

    sim.compute_source_terms.push_back(SourceTerm{
        .name = "gravity",
        .fn = apply_gravity,
        .get_context = [=]() { return grav.get(); }
    });
}

}