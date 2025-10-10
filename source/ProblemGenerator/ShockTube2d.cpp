#include "ProblemGenerator.hpp"
#include "../Hydro.hpp"
#include "../MosscapConfig.hpp"

// NOTE(cmo): This is a 2d problem
static constexpr int num_dim = 2;

namespace Mosscap {

MOSSCAP_NEW_PROBLEM(shock_tube_2d) {
    MOSSCAP_PROBLEM_PREAMBLE(shock_tube_2d);
    using Prim = Prim<num_dim>;
    constexpr int n_hydro = N_HYDRO_VARS<num_dim>;
    if (sim.num_dim != num_dim) {
        throw std::runtime_error(fmt::format(
            "{} only handles {}d problems", PROBLEM_NAME, num_dim
        ));
    }

    const auto& state = sim.state;
    const auto& sz = state.sz;
    const auto& eos = sim.eos;

    dex_parallel_for(
        FlatLoop<3>(sz.zc, sz.yc, sz.xc),
        KOKKOS_LAMBDA (int k, int j, int i) {
            vec3 p = state.get_pos(i);
            yakl::SArray<fp_t, 1, n_hydro> w(0.0_fp);
            if (p(0) < 0.5_fp) {
                w(I(Prim::Rho)) = 1.0_fp;
                w(I(Prim::Vx)) = 0.0_fp;
                w(I(Prim::Vy)) = 0.0_fp;
                w(I(Prim::Pres)) = 1.0_fp;
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
    sim.max_time = get_or<fp_t>(config, "timestep.max_time", 0.2_fp);
}

}