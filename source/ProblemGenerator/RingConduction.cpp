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
        mass_density = get_or<fp_t>(config, "problem.ring_density", 5e-12_fp);
        length_scale = 1.0e7_fp;
        inner_temperature = get_or<fp_t>(config, "problem.t_in", 1e6_fp);
        outer_temperature = get_or<fp_t>(config, "problem.t_out", 1e4_fp);
        sim.max_time = get_or<fp_t>(config, "timestep.max_time", 200.0_fp * 3600.0_fp);
    }

    // NOTE(claude): Options for coupled (full MHD) validation of conduction against STS:
    // - pressure_balance: rho = rho_ring * T_in / T, so p is uniform and the T jumps are contacts.
    // - field: "uniform" (default; |B| = b0, the unbalanced-tension field used for conduction-only runs) or
    //   "potential": B_theta = b0 * (0.6 L) / r, current-free outside a uniform-current core of radius
    //   core_radius (B_theta proportional to r inside); the core's Lorentz force is balanced by a gas-pressure
    //   excess (B_c^2 / mu0) (1 - r^2 / r_c^2) carried as density at fixed T, so the state is in equilibrium at any beta.
    const bool pressure_balance = get_or<bool>(config, "problem.pressure_balance", false);
    const std::string field_type = get_or<std::string>(config, "problem.field", "uniform");
    const bool potential_field = field_type == "potential";
    const bool straight_field = field_type == "x"; // diagnostic: uniform B along x
    // field: angle -> uniform straight B at problem.field_angle degrees to x (exact equilibrium at any beta)
    const bool angled_field = field_type == "angle";
    const fp_t field_angle = get_or<fp_t>(config, "problem.field_angle", 30.0_fp) * pi / 180.0_fp;
    const fp_t b0 = get_or<fp_t>(config, "problem.b0", 1e-8_fp);
    const fp_t r_core = get_or<fp_t>(config, "problem.core_radius", 0.2_fp) * length_scale;
    const fp_t r_ring = 0.6_fp * length_scale;
    // NOTE(claude): edge_cells > 0 smooths the ring edges with tanh profiles of that width (in cells) in r and in arc
    // length, blending log T linearly (as the HyperTcWork test bed's CoupledRing).
    // problem.edge_width [m] fixes the physical width (resolution-independent IC, for convergence studies).
    const fp_t edge_width = get_or<fp_t>(config, "problem.edge_width",
                                         get_or<fp_t>(config, "problem.edge_cells", 0.0_fp) * sim.state.dx);

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
            fp_t b_mag = b0;
            fp_t core_dp = 0.0_fp;
            if (potential_field) {
                const fp_t b_core = b0 * r_ring / r_core;
                if (r >= r_core) {
                    b_mag = b0 * r_ring / r;
                } else {
                    b_mag = b_core * r / r_core;
                    core_dp = square(b_core) / state.mu0 * (1.0_fp - square(r / r_core));
                }
            }
            if (angled_field) {
                w(I(Prim::Bx)) = b0 * std::cos(field_angle);
                w(I(Prim::By)) = b0 * std::sin(field_angle);
            } else {
                w(I(Prim::Bx)) = straight_field ? b0 : std::cos(theta + 0.5_fp * pi) * b_mag;
                w(I(Prim::By)) = straight_field ? 0.0_fp : std::sin(theta + 0.5_fp * pi) * b_mag;
            }

            fp_t temperature = outer_temperature;
            if (edge_width > 0.0_fp) {
                auto sm = [&](fp_t d) { return 0.5_fp * (1.0_fp + std::tanh(d / edge_width)); };
                const fp_t arc = r * std::min(std::abs(theta - pi), pi);
                const fp_t frac = sm(r - 0.5_fp * length_scale) * sm(0.7_fp * length_scale - r) * sm(r * pi / 12.0_fp - arc);
                temperature = outer_temperature * std::pow(inner_temperature / outer_temperature, frac);
            } else if (r > 0.5_fp * length_scale && r < 0.7_fp * length_scale && theta > (11.0_fp / 12.0_fp) * pi && theta < (13.0_fp / 12.0_fp) * pi) {
                temperature = inner_temperature;
            }
            if (pressure_balance) {
                w(I(Prim::Rho)) = mass_density * inner_temperature / temperature;
            }
            const fp_t p_per_rho = (eos.total_abund + eos.y) * k_B * temperature / (eos.mass_per_h * state.p_mass);
            w(I(Prim::Pres)) = w(I(Prim::Rho)) * p_per_rho + core_dp;
            w(I(Prim::Rho)) = w(I(Prim::Pres)) / p_per_rho;
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
    // NOTE(claude): problem.full_mhd allows coupled runs (hydro fluxes on) with either HTC fluid types or STS.
    const bool full_mhd = get_or<bool>(config, "problem.full_mhd", false);
    if (
        !full_mhd
        && sim.fluid_type != FluidType::HyperTcOnly
        && !(get_or<bool>(config, "sources.thermal_conduction.enable", false) && get_or<bool>(config, "simulation.zero_hydro_flux", false))
    ) {
        throw std::runtime_error(fmt::format(
            "{} only runs as HyperTcOnly or with thermal_conduction and zero_hydro_flux (or set problem.full_mhd)", PROBLEM_NAME
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

    // NOTE(claude): ThermalConductionContext::enable is not checked by the kernel, so only register the source when
    // enabled (needed for conduction-free equilibrium checks; all previously valid configs had enable: true).
    if (!FluidTraitsRt(sim.num_dim, sim.fluid_type).has_hypertc && get_or<bool>(config, "sources.thermal_conduction.enable", false)) {
        setup_thermal_conduction(sim, config);
    }
}

}