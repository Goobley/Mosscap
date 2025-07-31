#include "ProblemGenerator.hpp"
#include "../Hydro.hpp"
#include "../MosscapConfig.hpp"

// NOTE(cmo): This is only 1D and isn't really set up correctly!

template <int NumDim>
void linear_waves_impl(Simulation& sim, fp_t amp, fp_t vflow) {
    using Prim = Prim<NumDim>;
    constexpr int n_hydro = N_HYDRO_VARS<NumDim>;
    const auto& state = sim.state;
    const auto& eos = sim.eos;
    const auto& sz = state.sz;

    constexpr int axis = 0;
    const fp_t k_par = FP(2.0) * M_PI / ((sz.xc - 2 * sz.ng) * state.dx);

    dex_parallel_for(
        FlatLoop<3>(sz.zc, sz.yc, sz.xc),
        KOKKOS_LAMBDA (int k, int j, int i) {
            vec3 p = state.get_pos(i, j, k);
            const fp_t sx = std::sin(k_par * p(axis));
            yakl::SArray<fp_t, 1, n_hydro> w(FP(0.0));
            CellIndex idx {
                .i = i,
                .j = j,
                .k = k
            };
            auto ev = EosView(eos, idx);
            auto g = ev.get_gamma();
            w(I(Prim::Rho)) = FP(1.0) + amp * sx;
            w(I(Prim::Vx)) = FP(1.0) + amp * sx;
            w(I(Prim::Pres)) = FP(1.0) / g.Gamma;
            prim_to_cons<NumDim>(ev, w, QtyView(state.Q, idx));
        }
    );
}

MOSSCAP_NEW_PROBLEM(linear_waves) {
    MOSSCAP_PROBLEM_PREAMBLE(linear_waves);

    sim.max_time = get_or<fp_t>(config, "timestep.max_time", FP(0.038));

    auto& bd = sim.state.boundaries;
    auto set_boundary = [&](BoundaryType& out, const std::string& bdry) {
        std::string bdry_string = get_or<std::string>(config, fmt::format("boundary.{}", bdry), "periodic");
        out = find_associated_enum<BoundaryType>(BoundaryTypeName, NumBoundaryType, bdry_string);
    };

    fp_t vflow = get_or<fp_t>(config, "problem.v_flow", FP(0.0));
    fp_t amp = get_or<fp_t>(config, "problem.amplitude", FP(1.0));

    set_boundary(bd.xs, "xs");
    set_boundary(bd.xe, "xe");
    if (sim.num_dim > 1) {
        set_boundary(bd.ys, "ys");
        set_boundary(bd.ye, "ye");
    } else {
        set_boundary(bd.zs, "zs");
        set_boundary(bd.ze, "ze");
    }

    if (sim.num_dim == 1) {
        linear_waves_impl<1>(sim, amp, vflow);
    } else if (sim.num_dim == 2) {
        linear_waves_impl<2>(sim, amp, vflow);
    } else {
        linear_waves_impl<3>(sim, amp, vflow);
    }
}