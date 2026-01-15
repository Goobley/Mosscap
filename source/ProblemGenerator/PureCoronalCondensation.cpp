#include "ProblemGenerator.hpp"
#include "../Hydro.hpp"
#include "../MosscapConfig.hpp"
#include "../SourceTerms/Gravity.hpp"

// NOTE(cmo): This is a 2d problem
static constexpr int num_dim = 2;

namespace Mosscap {

template <typename Fluid>
static void initial_conditions(Simulation& sim, const YAML::Node& config) {
    using Prim = typename Fluid::prim;
    constexpr int n_hydro = Fluid::num_vars;
    typedef yakl::Array<f64, 1, yakl::memHost> F64Host;
    const auto& state = sim.state;
    const auto& sz = state.sz;
    const auto& eos = sim.eos;

    static constexpr f64 h_mass = 1.6737830080950003e-27;
    static constexpr f64 k_B = 1.380649e-23;
    const fp_t rho_0 = get_or<fp_t>(config, "problem.base_density", 5e-12_fp);
    const fp_t P_0 = get_or<fp_t>(config, "problem.base_pressure", 0.165_fp);
    const fp_t rho_b0 = get_or<fp_t>(config, "problem.blob_density", 5e-10_fp);

    const fp_t x0 = get_or<fp_t>(config, "problem.blob_x0", 0.0_fp);
    const fp_t z0 = get_or<fp_t>(config, "problem.blob_z0", 50e6_fp);
    const fp_t delta = get_or<fp_t>(config, "problem.blob_delta", 0.5e6_fp);
    const fp_t b0 = get_or<fp_t>(config, "problem.b0", 10e-4_fp); // 10 G

    // Coronal background temperature = P_0 / (n k_B) -- fully ionised
    const fp_t T_0 = P_0 / (2.0_fp * rho_0 / h_mass * k_B);
    fmt::println("Base coronal temperature {:.2e} K", T_0);
    const fp_t mean_mass = 1.0_fp;

    dex_parallel_for(
        FlatLoop<3>(sz.zc, sz.yc, sz.xc),
        KOKKOS_LAMBDA (int k, int j, int i) {
            yakl::SArray<fp_t, 1, n_hydro> w(0.0_fp);
            w(I(Prim::Vx)) = 0.0_fp;
            w(I(Prim::Vy)) = 0.0_fp;
            w(I(Prim::Rho)) = rho_0;
            w(I(Prim::Pres)) = P_0;

            vec3 p = state.get_pos(i, j, k);
            const fp_t gauss_factor = std::exp(-(square(p(0) - x0) + square(p(1) - z0)) / square(delta));
            w(I(Prim::Rho)) += rho_b0 * gauss_factor;

            JasUse(b0);
            if constexpr (Fluid::is_mhd) {
                w(I(Prim::Bx)) = b0;
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

MOSSCAP_NEW_PROBLEM(pure_coronal_condensation) {
    MOSSCAP_PROBLEM_PREAMBLE(pure_coronal_condensation);
    if (sim.num_dim != num_dim) {
        throw std::runtime_error(fmt::format(
            "{} only handles {}d problems", PROBLEM_NAME, num_dim
        ));
    }

    FluidTraitsRt traits(sim.num_dim, sim.fluid_type);
    sim.setup_ics = [=](Simulation& sim) {
        if (sim.fluid_type == FluidType::Hydro) {
            initial_conditions<FluidTraits<num_dim, FluidType::Hydro>>(sim, config);
        } else if (traits.is_mhd) {
            initial_conditions<FluidTraits<num_dim, FluidType::Mhd>>(sim, config);
        } else {
            throw std::runtime_error("Unknown fluid type");
        }
    };
}

}