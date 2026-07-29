#include "ProblemGenerator.hpp"
#include "../Hydro.hpp"
#include "../MosscapConfig.hpp"
#include "../SourceTerms.hpp"
#include "../SourceTerms/ThermalConduction.hpp"

// NOTE(cmo): This is a 2d problem
static constexpr int num_dim = 2;

namespace Mosscap {


template <typename Fluid>
static void initial_conditions(Simulation& sim, const YAML::Node& config) {
    constexpr fp_t pi = 3.14159265358979312_fp;
    constexpr fp_t k_B = 1.380649e-23_fp; // [J / K]
    constexpr int n_hydro = Fluid::num_vars;
    using Prim = Fluid::prim;

    bool dimensioned = get_or<bool>(config, "problem.coronal", false);

    fp_t mass_density = 1.0_fp;
    fp_t length_scale = 1.0_fp;
    fp_t inner_temperature = 12.0_fp;
    fp_t outer_temperature = 10.0_fp;
    if (!dimensioned) {
        sim.max_time = get_or<fp_t>(config, "timestep.max_time", 400.0_fp);
        sim.state.mu0 = 1.0_fp;
        sim.state.p_mass = k_B;
        const fp_t kappa0 = get_or<fp_t>(config, "problem.kappa0", 0.01_fp);
        const bool spitzer = get_or<bool>(config, "problem.spitzer", false);
        if constexpr (Fluid::has_hypertc) {
            sim.state.cond.hypertc_kappa = kappa0;
            sim.state.cond.spitzer = spitzer;
        } else {
            int idx = source_term_index(sim, "thermal_conduction");
            if (idx >= sim.compute_source_terms.size()) {
                throw std::runtime_error("Conduction not initialised!");
            }
            auto ctx = (ThermalConductionContext*)sim.compute_source_terms[idx].get_context();
            ctx->kappa0 = kappa0;
            ctx->spitzer = spitzer;
        }
    } else {
        mass_density = 5e-12_fp;
        length_scale = 1.0e7_fp;
        inner_temperature = 1e6_fp;
        outer_temperature = 1e4_fp;
        sim.max_time = get_or<fp_t>(config, "timestep.max_time", 200.0_fp * 3600.0_fp);
    }

    const auto& state = sim.state;
    const auto& sz = state.sz;
    const auto& eos = sim.eos;

    dex_parallel_for(
        FlatLoop<3>(sz.zc, sz.yc, sz.xc),
        KOKKOS_LAMBDA (int k, int j, int i) {
            vec3 p = state.get_pos(i, j, k);
            yakl::SArray<fp_t, 1, n_hydro> w(0.0_fp);
            w(I(Prim::Rho)) = mass_density;
            fp_t theta = std::atan2(p(1), p(0));
            if (theta < 0.0_fp) {
                theta += 2.0_fp * pi;
            }
            const fp_t r = std::sqrt(square(p(0)) + square(p(1)));
            // w(I(Prim::Bx)) = std::cos(theta + 0.5_fp * pi) * 1e-5_fp / (r / length_scale + 1.0_fp);
            // w(I(Prim::By)) = std::sin(theta + 0.5_fp * pi) * 1e-5_fp / (r / length_scale + 1.0_fp);
            // fp_t denom = std::sqrt(square(w(I(Prim::Bx))) + square(w(I(Prim::By))) + 1e-20_fp);
            // w(I(Prim::Bx)) /= denom;
            // w(I(Prim::By)) /= denom;
            w(I(Prim::Bx)) = std::cos(theta + 0.5_fp * pi) * 1e-8_fp;
            w(I(Prim::By)) = std::sin(theta + 0.5_fp * pi) * 1e-8_fp;

            fp_t temperature = outer_temperature;
            if (r > 0.5_fp * length_scale && r < 0.7_fp * length_scale && theta > (11.0_fp / 12.0_fp) * pi && theta < (13.0_fp / 12.0_fp) * pi) {
                temperature = inner_temperature;
            }
            const fp_t nh_tot = w(I(Prim::Rho)) / (eos.mass_per_h * state.p_mass);
            w(I(Prim::Pres)) = nh_tot * (eos.total_abund + eos.y) * k_B * temperature;
            CellIndex idx {
                .i = i,
                .j = j,
                .k = k
            };
            prim_to_cons<Fluid>(eos.gamma, state.mu0, w, QtyView(state.Q, idx));
        }
    );
    Kokkos::fence();
}

MOSSCAP_NEW_PROBLEM(ring_conduction) {
    MOSSCAP_PROBLEM_PREAMBLE(ring_conduction);
    if (sim.num_dim != num_dim) {
        throw std::runtime_error(fmt::format(
            "{} only handles {}d problems", PROBLEM_NAME, num_dim
        ));
    }
    if (
        sim.fluid_type != FluidType::HyperTcOnly
        && !(get_or<bool>(config, "sources.thermal_conduction.enable", false) && get_or<bool>(config, "simulation.zero_hydro_flux", false))
    ) {
        throw std::runtime_error(fmt::format(
            "{} only runs as HyperTcOnly or with thermal_conduction and zero_hydro_flux", PROBLEM_NAME
        ));
    }

    sim.setup_ics = [=](Simulation& sim) {
        invoke_fluid_traits(
            sim.num_dim,
            sim.fluid_type,
            [&]<typename FTraits>(FTraits) {
                return initial_conditions<FTraits>(sim, config);
            }
        );
    };

    if (sim.fluid_type != FluidType::HyperTcOnly) {
        setup_thermal_conduction(sim, config);
    }
}

}