#include "ProblemGenerator.hpp"
#include "../Hydro.hpp"
#include "../MosscapConfig.hpp"

using namespace Mosscap;

template <int NumDim>
void double_shock_tube_impl(Simulation& sim, int axis) {
    using Prim = Prim<NumDim>;
    constexpr int n_hydro = N_HYDRO_VARS<NumDim>;
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
            yakl::SArray<fp_t, 1, n_hydro> w(FP(0.0));
            if (p(axis) < FP(0.1)) {
                w(I(Prim::Rho)) = FP(1.0);
                w(I(Prim::Pres)) = FP(1000.0);
            } else if (p(axis) > FP(0.9)) {
                w(I(Prim::Rho)) = FP(1.0);
                w(I(Prim::Pres)) = FP(100.0);
            } else {
                w(I(Prim::Rho)) = FP(1.0);
                w(I(Prim::Pres)) = FP(0.01);
            }
            CellIndex idx {
                .i = i,
                .j = j,
                .k = k
            };
            prim_to_cons<NumDim>(eos.gamma, w, QtyView(state.Q, idx));
        }
    );
}

MOSSCAP_NEW_PROBLEM(double_shock_tube) {
    MOSSCAP_PROBLEM_PREAMBLE(double_shock_tube);

    sim.max_time = get_or<fp_t>(config, "timestep.max_time", FP(0.038));
    int axis = get_or<int>(config, "problem.shock_axis", 0);
    if (sim.num_dim == 1) {
        double_shock_tube_impl<1>(sim, axis);
    } else if (sim.num_dim == 2) {
        double_shock_tube_impl<2>(sim, axis);
    } else {
        double_shock_tube_impl<3>(sim, axis);
    }
}