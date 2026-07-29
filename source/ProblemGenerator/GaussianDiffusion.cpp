#include "ProblemGenerator.hpp"
#include "../Hydro.hpp"
#include "../MosscapConfig.hpp"
#include "../SourceTerms.hpp"
#include "../SourceTerms/ThermalConduction.hpp"

// NOTE(claude): Diffusion of a Gaussian temperature perturbation, for validating
// the classic (explicit/STS) and hyperbolic thermal conduction
// implementations against each other and against the analytic solution of
// the linear heat equation. Zero velocity, uniform density, with an optional
// uniform background field aligned with the direction the temperature
// varies in ("problem.axis"). With the field aligned this way, both the
// anisotropic classic flux and the (field-aligned-only) hyperbolic flux
// reduce exactly to the isotropic 1D heat equation along that axis,
// regardless of "sources.thermal_conduction.anisotropic", so both schemes
// can be checked against the same closed-form reference. Set
// "problem.radial: true" for a genuinely multi-D isotropic Gaussian instead
// (only directly comparable to an analytic solution for the classic scheme
// with anisotropic conduction disabled, or B == 0).

namespace Mosscap {

static constexpr fp_t k_B = 1.380649e-23_fp;

template <typename FTraits>
static void initial_conditions(Simulation& sim, const YAML::Node& config) {
    using Prim = typename FTraits::prim;
    constexpr int n_hydro = FTraits::num_vars;
    const auto& state = sim.state;
    const auto& sz = state.sz;
    const auto& eos = sim.eos;

    // NOTE(claude): Convenience mode for round-numbered, easy-to-reason-about
    // test configurations, matching RingConduction's "coronal" toggle.
    if (get_or<bool>(config, "problem.dimensionless", false)) {
        sim.state.mu0 = 1.0_fp;
        sim.state.p_mass = k_B;
        const fp_t kappa0 = get_or<fp_t>(config, "problem.kappa0", 0.01_fp);
        const bool spitzer = get_or<bool>(config, "problem.spitzer", false);
        if constexpr (FTraits::has_hypertc) {
            sim.state.cond.hypertc_kappa = kappa0;
            sim.state.cond.spitzer = spitzer;
        } else {
            int idx = source_term_index(sim, "thermal_conduction");
            if (idx >= sim.compute_source_terms.size()) {
                throw std::runtime_error("gaussian_diffusion: thermal conduction not registered.");
            }
            auto* ctx = (ThermalConductionContext*)sim.compute_source_terms[idx].get_context();
            ctx->kappa0 = kappa0;
            ctx->spitzer = spitzer;
        }
    }

    const bool radial = get_or<bool>(config, "problem.radial", false);
    const int axis = get_or<int>(config, "problem.axis", 0);
    if (axis < 0 || axis >= sim.num_dim) {
        throw std::runtime_error("gaussian_diffusion: problem.axis must be a valid axis for this simulation's dimensionality.");
    }

    const fp_t rho0 = get_or<fp_t>(config, "problem.base_density", 5e-12_fp);
    const fp_t T0 = get_or<fp_t>(config, "problem.base_temperature", 1.0e6_fp);
    const fp_t dT = get_or<fp_t>(config, "problem.delta_temperature", 0.5_fp * T0);
    const fp_t sigma0 = get_or<fp_t>(config, "problem.sigma0", 0.1_fp * state.get_axis_length(axis));
    const fp_t b0 = get_or<fp_t>(config, "problem.b0", 0.0_fp);

    vec3 center;
    center(0) = get_or<fp_t>(config, "problem.center_x", state.loc.x + 0.5_fp * state.get_axis_length(0));
    center(1) = get_or<fp_t>(config, "problem.center_y", state.loc.y + 0.5_fp * state.get_axis_length(1));
    center(2) = get_or<fp_t>(config, "problem.center_z", state.loc.z + 0.5_fp * state.get_axis_length(2));

    dex_parallel_for(
        "Gaussian diffusion ICs",
        FlatLoop<3>(sz.zc, sz.yc, sz.xc),
        KOKKOS_LAMBDA (int k, int j, int i) {
            yakl::SArray<fp_t, 1, n_hydro> w(0.0_fp);
            vec3 p = state.get_pos(i, j, k);
            JasUse(b0);

            fp_t r2 = square(p(axis) - center(axis));
            if (radial) {
                r2 = square(p(0) - center(0));
                if constexpr (FTraits::num_dim > 1) {
                    r2 += square(p(1) - center(1));
                }
                if constexpr (FTraits::num_dim > 2) {
                    r2 += square(p(2) - center(2));
                }
            }

            const fp_t temperature = T0 + dT * std::exp(-0.5_fp * r2 / square(sigma0));
            const fp_t nh_tot = rho0 / (eos.mass_per_h * state.p_mass);

            w(I(Prim::Rho)) = rho0;
            w(I(Prim::Pres)) = nh_tot * (eos.total_abund + eos.y) * k_B * temperature;

            if constexpr (FTraits::is_mhd) {
                vec3 B(0.0_fp);
                B(axis) = b0;
                w(I(Prim::Bx)) = B(0);
                w(I(Prim::By)) = B(1);
                w(I(Prim::Bz)) = B(2);
            }

            CellIndex idx{
                .i = i,
                .j = j,
                .k = k
            };
            prim_to_cons<FTraits>(eos.gamma, state.mu0, w, QtyView(state.Q, idx));
        }
    );
}

MOSSCAP_NEW_PROBLEM(gaussian_diffusion) {
    MOSSCAP_PROBLEM_PREAMBLE(gaussian_diffusion);

    if (!sim.eos.is_constant) {
        throw std::runtime_error(fmt::format(
            "{} requires eos.type: ideal (constant y) for a clean analytic reference.", PROBLEM_NAME
        ));
    }

    const bool flux_free = (sim.fluid_type == FluidType::HyperTcOnly) || get_or<bool>(config, "simulation.zero_hydro_flux", false);
    if (!flux_free) {
        throw std::runtime_error(fmt::format(
            "{} is a pure-conduction test: set simulation.zero_hydro_flux: true (or use fluid_type: "
            "hypertconly) to suppress advection of the initial pressure perturbation.",
            PROBLEM_NAME
        ));
    }

    FluidTraitsRt traits(sim.num_dim, sim.fluid_type);
    if (!traits.has_hypertc) {
        // NOTE(cmo): Hyperbolic conduction is wired up automatically by
        // setup_hyperbolic_tc whenever the fluid type carries HeatF; the
        // classic explicit/STS scheme needs registering explicitly.
        setup_thermal_conduction(sim, config);
    }

    sim.setup_ics = [=](Simulation& sim) {
        invoke_fluid_traits(
            sim.num_dim,
            sim.fluid_type,
            [&]<typename FTraits>(FTraits) {
                initial_conditions<FTraits>(sim, config);
            }
        );
    };
}

}
