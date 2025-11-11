#include "ProblemGenerator.hpp"
#include "../Hydro.hpp"
#include "../MosscapConfig.hpp"

namespace Mosscap {

template <int NumDim>
void shock_tube_impl(Simulation& sim, int axis) {
    using Fluid = FluidTraits<NumDim, FluidType::Mhd>;
    using Prim = Fluid::prim;
    constexpr int n_hydro = Fluid::num_vars;
    const auto& state = sim.state;
    const auto& eos = sim.eos;
    const auto& sz = state.sz;

    if (axis >= NumDim) {
        throw std::runtime_error(fmt::format("Cannot create a shock on axis {} in a {}d problem", axis, NumDim));
    }

    dex_parallel_for(
        FlatLoop<3>(sz.zc, sz.yc, sz.xc),
        KOKKOS_LAMBDA (int k, int j, int i) {
            vec3 p = state.get_pos(i, j, k);
            yakl::SArray<fp_t, 1, n_hydro> w(0.0_fp);
            if (p(axis) < 0.5_fp * state.get_axis_length(axis)) {
                w(I(Prim::Rho)) = 1.0_fp;
                w(I(Prim::Pres)) = 1.0_fp;
                w(I(Prim::Bx)) = 0.75_fp;
                w(I(Prim::By)) = 1.0_fp;
            } else {
                w(I(Prim::Rho)) = 0.125_fp;
                w(I(Prim::Pres)) = 0.1_fp;
                w(I(Prim::Bx)) = 0.75_fp;
                w(I(Prim::By)) = -1.0_fp;
            }
            CellIndex idx {
                .i = i,
                .j = j,
                .k = k
            };
            prim_to_cons<Fluid>(eos.gamma, state.mu0, w, QtyView(state.Q, idx));
        }
    );
}

MOSSCAP_NEW_PROBLEM(brio_wu_shock_tube) {
    MOSSCAP_PROBLEM_PREAMBLE(brio_wu_shock_tube);

    FluidTraitsRt traits(sim.num_dim, sim.fluid_type);
    if (!traits.is_mhd) {
        throw std::runtime_error("This is an MHD problem.");
    }
    sim.state.mu0 = 1.0_fp;
    sim.max_time = get_or<fp_t>(config, "timestep.max_time", 0.05_fp);
    int axis = get_or<int>(config, "problem.shock_axis", 0);

    sim.setup_ics = [=](Simulation& sim) {
        if (sim.num_dim == 1) {
            shock_tube_impl<1>(sim, axis);
        } else if (sim.num_dim == 2) {
            shock_tube_impl<2>(sim, axis);
        } else {
            shock_tube_impl<3>(sim, axis);
        }
    };
}

}