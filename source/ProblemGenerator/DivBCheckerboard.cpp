#include "ProblemGenerator.hpp"
#include "../Hydro.hpp"
#include "../MosscapConfig.hpp"

namespace Mosscap {

constexpr int num_dim = 2;
void initial_conditions_checkerboard(Simulation& sim) {
    using Fluid = FluidTraits<num_dim, FluidType::Mhd>;
    using Prim = Fluid::prim;
    constexpr int n_hydro = Fluid::num_vars;
    const auto& state = sim.state;
    const auto& eos = sim.eos;
    const auto& sz = state.sz;

    dex_parallel_for(
        FlatLoop<3>(sz.zc, sz.yc, sz.xc),
        KOKKOS_LAMBDA (int k, int j, int i) {
            yakl::SArray<fp_t, 1, n_hydro> w(0.0_fp);
            w(I(Prim::Rho)) = 1.0_fp;
            w(I(Prim::Pres)) = 1.0_fp;
            if (i % 3 == 0) {
                w(I(Prim::Bx)) = 1.0_fp;
            } else {
                w(I(Prim::Bx)) = 2.0_fp;
            }
            w(I(Prim::By)) = 1.0_fp;
            CellIndex idx {
                .i = i,
                .j = j,
                .k = k
            };
            prim_to_cons<Fluid>(eos.gamma, state.mu0, w, QtyView(state.Q, idx));
        }
    );
}

MOSSCAP_NEW_PROBLEM(divb_checkerboard) {
    MOSSCAP_PROBLEM_PREAMBLE(divb_checkerboard);

    if (sim.num_dim != num_dim) {
        throw std::runtime_error(fmt::format(
            "{} only handles {}d problems", PROBLEM_NAME, num_dim
        ));
    }

    FluidTraitsRt traits(sim.num_dim, sim.fluid_type);
    if (!traits.is_mhd) {
        throw std::runtime_error("This is an MHD problem.");
    }
    sim.state.mu0 = 1.0_fp;

    sim.max_time = get_or<fp_t>(config, "timestep.max_time", 0.00001_fp);

    sim.setup_ics = [](Simulation& sim) {
        initial_conditions_checkerboard(sim);
    };
}

}