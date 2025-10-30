#include "ProblemGenerator.hpp"
#include "../Hydro.hpp"
#include "../MosscapConfig.hpp"

namespace Mosscap {

constexpr int num_dim = 2;
template <FluidType fluid_type>
static void initial_conditions(Simulation& sim) {
    using Fluid = FluidTraits<num_dim, fluid_type>;
    using Prim = Fluid::prim;
    constexpr int n_hydro = Fluid::num_vars;
    const auto& state = sim.state;
    const auto& eos = sim.eos;
    const auto& sz = state.sz;

    constexpr fp_t pi = 3.14159265358979312_fp;
    const fp_t B0 = 5.0_fp / std::sqrt(4.0_fp * pi);
    dex_parallel_for(
        FlatLoop<3>(sz.zc, sz.yc, sz.xc),
        KOKKOS_LAMBDA (int k, int j, int i) {
            vec3 p = state.get_pos(i, j, k);
            yakl::SArray<fp_t, 1, n_hydro> w(0.0_fp);
            w(I(Prim::Pres)) = 1.0_fp;
            w(I(Prim::Bx)) = B0;

            const fp_t r = std::sqrt(square(p(0) - 0.5_fp) + square(p(1) - 0.5_fp));
            constexpr fp_t r0 = 0.1_fp;
            constexpr fp_t r1 = 0.115_fp;
            if (r < r0) {
                w(I(Prim::Rho)) = 10.0_fp;
                w(I(Prim::Vx)) = -2.0_fp * (p(1) - 0.5_fp) / r0;
                w(I(Prim::Vy)) = 2.0_fp * (p(0) - 0.5_fp) / r0;
            } else if (r < r1) {
                const fp_t frac = (r - r0) / (r1 - r0);
                w(I(Prim::Rho)) = (1.0_fp - frac) * 10.0_fp + frac * 1.0_fp;
                w(I(Prim::Vx)) = (1.0_fp - frac) * -2.0_fp * (p(1) - 0.5_fp) / r0;
                w(I(Prim::Vy)) = (1.0_fp - frac) * 2.0_fp * (p(0) - 0.5_fp) / r0;
            } else {
                w(I(Prim::Rho)) = 1.0_fp;
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

MOSSCAP_NEW_PROBLEM(mhd_rotor) {
    MOSSCAP_PROBLEM_PREAMBLE(mhd_rotor);

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
    sim.setup_ics = [=](Simulation& sim) {
        if (sim.fluid_type == FluidType::Mhd) {
            initial_conditions<FluidType::Mhd>(sim);
        } else {
            initial_conditions<FluidType::GlmMhd>(sim);
        }
    };
}

}