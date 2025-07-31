#include "ProblemGenerator.hpp"
#include "../Hydro.hpp"

// NOTE(cmo): This is a 2d problem
static constexpr int num_dim = 2;

MOSSCAP_NEW_PROBLEM(circular_explosion) {
    MOSSCAP_PROBLEM_PREAMBLE(circular_explosion);

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
            if (std::sqrt(square(p(0) - FP(0.5)) + square(p(1) - FP(0.5))) <  FP(0.25)) {
                w(I(Prim::Rho)) = FP(10.0);
                w(I(Prim::Vx)) = FP(0.0);
                w(I(Prim::Vy)) = FP(0.0);
                w(I(Prim::Pres)) = FP(10.0);
            } else {
                w(I(Prim::Rho)) = FP(0.125);
                w(I(Prim::Vx)) = FP(0.0);
                w(I(Prim::Vy)) = FP(0.0);
                w(I(Prim::Pres)) = FP(0.1);
            }
            CellIndex idx {
                .i = i,
                .j = j,
                .k = k
            };
            prim_to_cons<num_dim>(EosView(eos, idx), w, QtyView(state.Q, idx));
        }
    );
}
