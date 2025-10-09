#include "ProblemGenerator.hpp"
#include "../Hydro.hpp"
#include "../MosscapConfig.hpp"

// NOTE(cmo): This is a 2d problem
static constexpr int num_dim = 2;

using namespace Mosscap;

// Case 3 of liska + wendroff

MOSSCAP_NEW_PROBLEM(quadrants_3) {
    MOSSCAP_PROBLEM_PREAMBLE(quadrants_3);

    if (sim.num_dim != num_dim) {
        throw std::runtime_error(fmt::format(
            "{} only handles {}d problems", PROBLEM_NAME, num_dim
        ));
    }
    using Prim = Prim<num_dim>;
    constexpr int n_hydro = N_HYDRO_VARS<num_dim>;
    const auto& state = sim.state;
    const auto& eos = sim.eos;
    const auto& sz = state.sz;

    dex_parallel_for(
        FlatLoop<3>(sz.zc, sz.yc, sz.xc),
        KOKKOS_LAMBDA (int k, int j, int i) {
            vec3 p = state.get_pos(i, j);
            yakl::SArray<fp_t, 1, n_hydro> w(FP(0.0));
            if (p(0) < FP(0.5) && p(1) >= FP(0.5)) {
                w(I(Prim::Pres)) = FP(0.3);
                w(I(Prim::Rho)) = FP(0.5323);
                w(I(Prim::Vx)) = FP(1.206);
                w(I(Prim::Vy)) = FP(0.0);
            } else if (p(0) < FP(0.5) && p(1) < FP(0.5)) {
                w(I(Prim::Pres)) = FP(0.029);
                w(I(Prim::Rho)) = FP(0.138);
                w(I(Prim::Vx)) = FP(1.206);
                w(I(Prim::Vy)) = FP(1.206);
            } else if (p(0) >= FP(0.5) && p(1) >= FP(0.5)) {
                w(I(Prim::Pres)) = FP(1.5);
                w(I(Prim::Rho)) = FP(1.5);
                w(I(Prim::Vx)) = FP(0.0);
                w(I(Prim::Vy)) = FP(0.0);
            } else if (p(0) >= FP(0.5) && p(1) < FP(0.5)) {
                w(I(Prim::Pres)) = FP(0.3);
                w(I(Prim::Rho)) = FP(0.5323);
                w(I(Prim::Vx)) = FP(0.0);
                w(I(Prim::Vy)) = FP(1.206);
            }
            CellIndex idx {
                .i = i,
                .j = j,
                .k = k
            };
            prim_to_cons<num_dim>(eos.gamma, w, QtyView(state.Q, idx));
        }
    );

    sim.max_time = get_or<fp_t>(config, "timestep.max_time", FP(0.3));
}