#include "ProblemGenerator.hpp"
#include "../Hydro.hpp"
#include "../MosscapConfig.hpp"

namespace Mosscap {

constexpr int num_dim = 2;
template <FluidType fluid_type>
void initial_conditions(Simulation& sim) {
    using Fluid = FluidTraits<num_dim, fluid_type>;
    using Prim = Fluid::prim;
    constexpr int n_hydro = Fluid::num_vars;
    const auto& state = sim.state;
    const auto& eos = sim.eos;
    const auto& sz = state.sz;

    constexpr fp_t pi = 3.14159265358979312_fp;
    constexpr fp_t v0 = 1.0_fp;
    const fp_t B0 = 1.0_fp / std::sqrt(4.0_fp * pi);
    dex_parallel_for(
        FlatLoop<3>(sz.zc, sz.yc, sz.xc),
        KOKKOS_LAMBDA (int k, int j, int i) {
            vec3 p = state.get_pos(i, j, k);
            yakl::SArray<fp_t, 1, n_hydro> w(0.0_fp);
            w(I(Prim::Rho)) = 25.0_fp / (36.0_fp * pi);
            w(I(Prim::Vx)) = -v0 * std::sin(2.0_fp * pi * p(1));
            w(I(Prim::Vy)) = v0 * std::sin(2.0_fp * pi * p(0));
            w(I(Prim::Pres)) = 5.0_fp / (12.0_fp * pi);
            w(I(Prim::Bx)) = -B0 * std::sin(2.0_fp * pi * p(1));
            w(I(Prim::By)) = B0 * std::sin(4.0_fp * pi * p(0));
            CellIndex idx {
                .i = i,
                .j = j,
                .k = k
            };
            prim_to_cons<Fluid>(eos.gamma, state.mu0, w, QtyView(state.Q, idx));
        }
    );
}

MOSSCAP_NEW_PROBLEM(orszag_tang_vortex) {
    MOSSCAP_PROBLEM_PREAMBLE(orszag_tang_vortex);

    if (sim.num_dim != num_dim) {
        throw std::runtime_error(fmt::format(
            "{} only handles {}d problems", PROBLEM_NAME, num_dim
        ));
    }

    if (sim.fluid_type == FluidType::Hydro) {
        throw std::runtime_error("This is an MHD problem.");
    }
    sim.state.mu0 = 1.0_fp;

    sim.max_time = get_or<fp_t>(config, "timestep.max_time", 0.05_fp);
    if (sim.fluid_type == FluidType::Mhd) {
        initial_conditions<FluidType::Mhd>(sim);
    } else {
        initial_conditions<FluidType::GlmMhd>(sim);
    }
}

}