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
    sim.max_time = get_or<fp_t>(config, "timestep.max_time", 400.0_fp);

    constexpr fp_t pi = 3.14159265358979312_fp;
    constexpr fp_t k_B = 1.380649e-23_fp; // [J / K]
    sim.state.p_mass = k_B;
    sim.state.mu0 = 1.0_fp;

    sim.setup_ics = [](Simulation& sim) {
        const auto& state = sim.state;
        const auto& sz = state.sz;
        const auto& eos = sim.eos;

        dex_parallel_for(
            FlatLoop<3>(sz.zc, sz.yc, sz.xc),
            KOKKOS_LAMBDA (int k, int j, int i) {
                vec3 p = state.get_pos(i, j, k);
                yakl::SArray<fp_t, 1, n_hydro> w(0.0_fp);
                w(I(Prim::Rho)) = 1.0_fp;
                fp_t theta = std::atan2(p(1), p(0));
                if (theta < 0.0_fp) {
                    theta += 2.0_fp * pi;
                }
                const fp_t r = std::sqrt(square(p(0)) + square(p(1)));
                w(I(Prim::Bx)) = std::cos(theta + 0.5_fp * pi) * 1e-5_fp / (r + 0.1_fp);
                w(I(Prim::By)) = std::sin(theta + 0.5_fp * pi) * 1e-5_fp / (r + 0.1_fp);

                fp_t temperature = 10.0_fp;
                if (r > 0.5_fp && r < 0.7_fp && theta > (11.0_fp / 12.0_fp) * pi && theta < (13.0_fp / 12.0_fp) * pi) {
                    temperature = 12.0_fp;
                }
                const fp_t n_baryon = w(I(Prim::Rho)) / (eos.avg_mass * state.p_mass);
                w(I(Prim::Pres)) = n_baryon * (1.0_fp + eos.y) * k_B * temperature;
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