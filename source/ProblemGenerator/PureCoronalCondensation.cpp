#include "ProblemGenerator.hpp"
#include "../Hydro.hpp"
#include "../MosscapConfig.hpp"
#include "../SourceTerms/Sponge.hpp"

// NOTE(cmo): This is a 2d problem
static constexpr int num_dim = 2;

namespace Mosscap {

static constexpr fp_t rho_0_d = 5e-12_fp;
static constexpr fp_t P_0_d = 0.165_fp;
static constexpr fp_t rho_b0_d = 5e-10_fp;
static constexpr fp_t b0_d = 10e-4_fp;
static constexpr fp_t x0_d = 0.0_fp;
static constexpr fp_t z0_d = 15e6_fp;
static constexpr fp_t delta_d = 0.5e6_fp;

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
    const fp_t rho_0 = get_or<fp_t>(config, "problem.base_density", rho_0_d);
    const fp_t P_0 = get_or<fp_t>(config, "problem.base_pressure", P_0_d);
    const fp_t rho_b0 = get_or<fp_t>(config, "problem.blob_density", rho_b0_d);

    const fp_t x0 = get_or<fp_t>(config, "problem.blob_x0", x0_d);
    const fp_t z0 = get_or<fp_t>(config, "problem.blob_z0", z0_d);
    const fp_t delta = get_or<fp_t>(config, "problem.blob_delta", delta_d);
    const fp_t b0 = get_or<fp_t>(config, "problem.b0", b0_d); // 10 G

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

template <typename FTraits>
static void setup_boundaries(Simulation& sim, const YAML::Node& config) {
    auto& bound = sim.state.boundaries;
    auto check_and_set_constant = [&](
            const BoundaryType boundary,
            const decltype(bound.xs_const)& arr,
            const std::string& bdry
    ) {
        if (boundary != BoundaryType::Constant) {
            return;
        }
        // NOTE(cmo): If it's not problem_supplied, then ignore
        const auto name = fmt::format("{}_const", bdry);
        if (config["boundary"][name].IsSequence()) {
            return;
        }

        const fp_t rho_0 = get_or<fp_t>(config, "problem.base_density", rho_0_d);
        const fp_t P_0 = get_or<fp_t>(config, "problem.base_pressure", P_0_d);
        const fp_t b0 = get_or<fp_t>(config, "problem.b0", b0_d); // 10 G

        yakl::SArray<fp_t, 1, FTraits::num_vars> w(0.0_fp);
        using Prim = FTraits::prim;
        w(I(Prim::Rho)) = rho_0;
        w(I(Prim::Pres)) = P_0;
        if (FTraits::is_mhd) {
            w(I(Prim::Bx)) = b0;
        }
        yakl::SArray<fp_t, 1, FTraits::num_vars> q(0.0_fp);
        prim_to_cons<FTraits>(sim.eos.gamma, sim.state.mu0, w, q);

        using Cons3 = Cons<3, FLUID_WITH_MAX_VARS>;
        using C = FTraits::cons;
        arr(I(C::Rho)) = q(I(Cons3::Rho));
        arr(I(C::MomX)) = q(I(Cons3::MomX));
        if constexpr (FTraits::is_mhd || FTraits::num_dim > 1) {
            arr(I(C::MomY)) = q(I(Cons3::MomY));
        }
        if constexpr (FTraits::is_mhd || FTraits::num_dim > 2) {
            arr(I(C::MomZ)) = q(I(Cons3::MomZ));
        }
        arr(I(C::Ene)) = q(I(Cons3::Ene));
        if constexpr (FTraits::is_mhd) {
            arr(I(C::Bx)) = q(I(Cons3::Bx));
            arr(I(C::By)) = q(I(Cons3::By));
            arr(I(C::Bz)) = q(I(Cons3::Bz));
            if constexpr (is_instance(FTraits::fluid_type, FluidType::GlmMhd)) {
                arr(I(C::Psi)) = q(I(Cons3::Psi));
            }
            if constexpr (FTraits::has_hypertc) {
                arr(I(C::HeatF)) = q(I(Cons3::HeatF));
            }
        }
    };
    check_and_set_constant(bound.xs, bound.xs_const, "xs");
    check_and_set_constant(bound.xe, bound.xe_const, "xe");
    check_and_set_constant(bound.ys, bound.ys_const, "ys");
    check_and_set_constant(bound.ye, bound.ye_const, "ye");
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

    invoke_fluid_traits(
        sim.num_dim,
        sim.fluid_type,
        [&]<typename FTraits>(FTraits) {
            setup_boundaries<FTraits>(sim, config);
        }
    );

    if (get_or<bool>(config, "problem.enable_sponge", false)) {
        setup_sponge(sim, config);
    }
}

}