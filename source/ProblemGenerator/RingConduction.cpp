#include "ProblemGenerator.hpp"
#include "../Hydro.hpp"
#include "../MosscapConfig.hpp"

// NOTE(cmo): This is a 2d problem
static constexpr int num_dim = 2;

namespace Mosscap {

using Fluid = FluidTraits<num_dim, FluidType::HyperTcOnly>;

MOSSCAP_NEW_PROBLEM(ring_conduction) {
    MOSSCAP_PROBLEM_PREAMBLE(ring_conduction);
    using Prim = Fluid::prim;
    constexpr int n_hydro = Fluid::num_vars;
    if (sim.num_dim != num_dim) {
        throw std::runtime_error(fmt::format(
            "{} only handles {}d problems", PROBLEM_NAME, num_dim
        ));
    }
    if (sim.fluid_type != FluidType::HyperTcOnly) {
        throw std::runtime_error(fmt::format(
            "{} only runs as HyperTcOnly", PROBLEM_NAME
        ));
    }

    sim.state.cond.hypertc_kappa = 0.01_fp;
    sim.state.cond.spitzer = false;
    sim.state.p_mass = 1.0_fp;
    sim.max_time = get_or<fp_t>(config, "timestep.max_time", 400.0_fp);

    constexpr fp_t pi = 3.14159265358979312_fp;
    constexpr fp_t k_B = 1.380649e-23_fp; // [J / K]
    constexpr fp_t inv_k_B = 1.0_fp / k_B;
    // NOTE(cmo): For no ionisation, and assuming a particle mass of 1.0 we have
    // T = P / (k_B rho), so for rho = 1.0 / k_B we have T = P.

    sim.setup_ics = [](Simulation& sim) {
        const auto& state = sim.state;
        const auto& sz = state.sz;
        const auto& eos = sim.eos;

        dex_parallel_for(
            FlatLoop<3>(sz.zc, sz.yc, sz.xc),
            KOKKOS_LAMBDA (int k, int j, int i) {
                vec3 p = state.get_pos(i, j, k);
                yakl::SArray<fp_t, 1, n_hydro> w(0.0_fp);
                w(I(Prim::Rho)) = inv_k_B;
                fp_t theta = std::atan2(p(1), p(0));
                if (theta < 0.0_fp) {
                    theta += 2.0_fp * pi;
                }
                w(I(Prim::Bx)) = std::cos(theta + 0.5_fp * pi);
                w(I(Prim::By)) = std::sin(theta + 0.5_fp * pi);

                const fp_t r = std::sqrt(square(p(0)) + square(p(1)));
                w(I(Prim::Pres)) = 10.0_fp;
                if (r > 0.5_fp && r < 0.7_fp && theta > (11.0_fp / 12.0_fp) * pi && theta < (13.0_fp / 12.0_fp) * pi) {
                    w(I(Prim::Pres)) = 12.0_fp;
                }
                CellIndex idx {
                    .i = i,
                    .j = j,
                    .k = k
                };
                prim_to_cons<Fluid>(eos.gamma, state.mu0, w, QtyView(state.Q, idx));
            }
        );
    };
}

}