#include "Gravity.hpp"
#include "../Simulation.hpp"
#include "../MosscapConfig.hpp"
#include "../SourceTerms.hpp"

namespace Mosscap {

struct GravityVals {
    fp_t x;
    fp_t y;
    fp_t z;
};

template <int NumDim>
void gravity_kernel(const Simulation& sim, const GravityVals& grav) {
    using Cons = Cons<NumDim>;

    const auto& Q = sim.state.Q;
    const auto& S = sim.sources.S;
    const auto& sz = sim.state.sz;

    dex_parallel_for(
        "Apply gravity",
        FlatLoop<3>(sz.zc, sz.yc, sz.xc),
        KOKKOS_LAMBDA (int k, int j, int i) {
            S(I(Cons::MomX), k, j, i) += Q(I(Cons::Rho), k, j, i) * grav.x;
            fp_t energy_update = Q(I(Cons::MomX), k, j, i) * grav.x;
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

    GravityVals grav{
        .x = get_or<fp_t>(config, "sources.gravity.x", -1.0_fp),
        .y = get_or<fp_t>(config, "sources.gravity.y", 0.0_fp),
        .z = get_or<fp_t>(config, "sources.gravity.z", 0.0_fp)
    };

    auto apply_gravity = [=](const Simulation& sim) {
        const int num_dim = sim.num_dim;
        if (num_dim == 1) {
            gravity_kernel<1>(sim, grav);
        } else if (num_dim == 2) {
            gravity_kernel<2>(sim, grav);
        } else {
            gravity_kernel<3>(sim, grav);
        }
    };

    if (source_term_index(sim, "gravity") != sim.compute_source_terms.size()) {
        throw std::runtime_error("Source \"gravity\" already registered.");
    }

    sim.compute_source_terms.push_back(SourceTerm{
        .name = "gravity",
        .fn = apply_gravity
    });
}

}