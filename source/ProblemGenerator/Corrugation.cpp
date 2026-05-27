#include "ProblemGenerator.hpp"
#include "../MosscapConfig.hpp"

namespace Mosscap {

constexpr int num_dim = 2;
template <FluidType fluid_type>
void initial_conditions(Simulation& sim) {
    using Fluid = FluidTraits<num_dim, fluid_type>;
    using Prim = typename Fluid::prim;
    constexpr int n_hydro = Fluid::num_vars;
    const auto& state = sim.state;
    const auto& eos = sim.eos;
    const auto& sz = state.sz;

    const fp_t beta = 0.1_fp;
    const fp_t mach = 2.0_fp;
    const fp_t rho_l = (eos.gamma + 1.0_fp) * square(mach) / (2.0_fp + (eos.gamma - 1.0_fp) * square(mach));
    const fp_t vx_l = -mach / rho_l;
    const fp_t rpres = 1.0_fp + eos.gamma * square(mach) * (1.0_fp - 1.0_fp / rho_l);
    const fp_t pres_l = rpres / eos.gamma;

    const fp_t rho_r = 1.0_fp;
    const fp_t vx_r = -mach;
    const fp_t pres_r = 1.0_fp / eos.gamma;

    constexpr fp_t pi = 3.14159265358979312_fp;
    dex_parallel_for(
        FlatLoop<3>(sz.zc, sz.yc, sz.xc),
        KOKKOS_LAMBDA (int k, int j, int i) {
            vec3 p = state.get_pos(i, j, k);
            yakl::SArray<fp_t, 1, n_hydro> w(0.0_fp);
            if (p(0) < 0.0_fp) {
                w(I(Prim::Rho)) = rho_l;
                w(I(Prim::Vx)) = vx_l;
                w(I(Prim::Pres)) = pres_l;
            } else {
                w(I(Prim::Rho)) = rho_r;
                w(I(Prim::Vx)) = vx_r;
                w(I(Prim::Pres)) = pres_r;
            }
            w(I(Prim::Bx)) = std::sqrt(2.0_fp / (beta * eos.gamma));

            if ((p(0) > 1.0_fp) && (p(0) < 2.0_fp)) {
                w(I(Prim::Rho)) += -0.1_fp * (
                    0.5_fp * (2.0_fp * std::sin(2.0_fp * pi * p(1) - 0.5_fp * pi))
                    + 0.5_fp
                )
                * 0.5_fp * (
                    std::sin(2.0_fp * pi * (p(0) - 0.25_fp)) + 1.0_fp
                );
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


MOSSCAP_NEW_PROBLEM(corrugation) {
    MOSSCAP_PROBLEM_PREAMBLE(corrugation);

    if (sim.num_dim != num_dim) {
        throw std::runtime_error(fmt::format(
            "{} only handles {}d problems", PROBLEM_NAME, num_dim
        ));
    }

    if (sim.fluid_type == FluidType::Hydro || sim.fluid_type == FluidType::HyperTcOnly) {
        throw std::runtime_error("This is an MHD problem.");
    }
    sim.state.mu0 = 1.0_fp;

    sim.max_time = get_or<fp_t>(config, "timestep.max_time", 0.5_fp);
    sim.setup_ics = [](Simulation& sim) {
        if (is_instance(sim.fluid_type, FluidType::Mhd)) {
            initial_conditions<FluidType::Mhd>(sim);
        } else {
            initial_conditions<FluidType::GlmMhd>(sim);
        }
    };
}
}
