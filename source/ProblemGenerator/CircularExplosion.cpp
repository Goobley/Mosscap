#include "ProblemGenerator.hpp"
#include "../Hydro.hpp"

// NOTE(cmo): This is a 2d problem
static constexpr int num_dim = 2;

namespace Mosscap {

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
            yakl::SArray<fp_t, 1, n_hydro> w(0.0_fp);
            if (std::sqrt(square(p(0) - 0.5_fp) + square(p(1) - 0.5_fp)) <  0.25_fp) {
                w(I(Prim::Rho)) = 10.0_fp;
                w(I(Prim::Vx)) = 0.0_fp;
                w(I(Prim::Vy)) = 0.0_fp;
                w(I(Prim::Pres)) = 10.0_fp;
            } else {
                w(I(Prim::Rho)) = 0.125_fp;
                w(I(Prim::Vx)) = 0.0_fp;
                w(I(Prim::Vy)) = 0.0_fp;
                w(I(Prim::Pres)) = 0.1_fp;
            }
            CellIndex idx {
                .i = i,
                .j = j,
                .k = k
            };
            prim_to_cons<num_dim>(eos.gamma, w, QtyView(state.Q, idx));
        }
    );
}

}