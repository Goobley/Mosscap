#include "ProblemGenerator.hpp"
#include "../Hydro.hpp"
#include "../MosscapConfig.hpp"
#include "../SourceTerms/Gravity.hpp"

// NOTE(cmo): This is a 2d problem
static constexpr int num_dim = 2;

MOSSCAP_NEW_PROBLEM(kelvin_helmholtz) {
    MOSSCAP_PROBLEM_PREAMBLE(kelvin_helmholtz);

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

    const u64 seed = 123456UL;
    const int axis = get_or<int>(config, "problem.axis", 1);
    std::string source_name = fmt::format("sources.gravity.{}", axis == 0 ? "x" : "y");
    set_if_not_present<fp_t>(config, source_name, FP(-0.1));
    const fp_t grav = get_or<fp_t>(config, source_name, FP(-0.1));
    int other_axis;
    if (axis == 0) {
        if ((sz.xc - 2 * sz.ng) / (sz.yc - 2 * sz.ng) != 3) {
            throw std::runtime_error("Aspect ratio should be 3 (x / y) == 3 for axis = 0");
        }
        other_axis = 1;
    } else {
        if ((sz.yc - 2 * sz.ng) / (sz.xc - 2 * sz.ng) != 3) {
            throw std::runtime_error("Aspect ratio should be 3 (y / x) == 3 for axis = 1");
        }
        other_axis = 0;
    }
    const int IV = axis == 0 ? Prim::Vx : Prim::Vy;
    const fp_t axis_length = state.get_axis_length(axis);
    const fp_t other_axis_length = state.get_axis_length(other_axis);
    dex_parallel_for(
        "Setup problem",
        FlatLoop<3>(sz.zc, sz.yc, sz.xc),
        KOKKOS_LAMBDA (int k, int j, int i) {
            vec3 p = state.get_pos(i, j);
            const fp_t pres_0 = FP(2.5);
            yakl::SArray<fp_t, 1, n_hydro> w(FP(0.0));

            if (p(axis) > FP(0.0)) {
                w(I(Prim::Rho)) = FP(2.0);
            } else {
                w(I(Prim::Rho)) = FP(1.0);
            }
            w(I(Prim::Pres)) = pres_0 + w(I(Prim::Rho)) * grav * p(axis);
            w(IV) = FP(0.01) * (FP(1.0) + std::cos(FP(2.0) * M_PI / other_axis_length * p(other_axis)))
                             * (FP(1.0) + std::cos(FP(2.0) * M_PI / axis_length * p(axis))) * FP(0.25);

            CellIndex idx {
                .i = i,
                .j = j,
                .k = k
            };
            prim_to_cons<num_dim>(eos.gamma, w, QtyView(state.Q, idx));
        }
    );

    setup_gravity(sim, config);
}