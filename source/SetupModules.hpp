#if !defined(MOSSCAP_SETUP_MODULES_HPP)
#define MOSSCAP_SETUP_MODULES_HPP

#include "Simulation.hpp"
#include "DivBCleaning.hpp"
#include "HyperbolicThermalConduction.hpp"
#include "yaml-cpp/yaml.h"

namespace Mosscap {

struct RestartArgs {
    bool restart = false;
    i64 restart_from = -1;
};

void setup_grid(Simulation& sim, YAML::Node& config) {
    auto& sz = sim.state.sz;
    sz.ng = get_or<int>(config, "grid.num_ghost", 3);
    sz.xc = get_or<int>(config, "grid.x", 256) + 2 * sz.ng;
    sz.yc = 1;
    sz.zc = 1;

    if (sim.num_dim > 1) {
        sz.yc = get_or<int>(config, "grid.y", 256) + 2 * sz.ng;
    }
    if (sim.num_dim > 2) {
        sz.zc = get_or<int>(config, "grid.z", 256) + 2 * sz.ng;
    }

    f64 dx;
    if (config["grid"] && config["grid"]["dx"]) {
        f64 dx = get_or<f64>(config, "grid.dx", -1.0);
        if (dx < 0.0) {
            throw std::runtime_error("grid.dx cannot be negative.");
        }
    } else {
        f64 x_dim = get_or<f64>(config, "grid.x_dim", 1.0);
        dx = x_dim / (sz.xc - 2 * sz.ng);
    }
    auto& state = sim.state;
    state.dx = dx;
    state.loc.x = get_or<fp_t>(config, "grid.x_start", 0.0_fp);
    state.loc.y = get_or<fp_t>(config, "grid.y_start", 0.0_fp);
    state.loc.z = get_or<fp_t>(config, "grid.z_start", 0.0_fp);

    const std::string fluid_string = get_or<std::string>(config, "simulation.fluid_type", "hydro");
    FluidType fluid_type = find_associated_enum<FluidType>(FluidTypeName, NumFluidType, fluid_string);
    sim.fluid_type = fluid_type;

    const int n_hydro = get_num_hydro_vars(sim.num_dim, fluid_type);
    int n_extra = get_or<int>(config, "simulation.n_extra_fields", 0);
    if (sim.dex.interface_config.enable && sim.dex.interface_config.advect) {
        int dex_tracers = sim.dex.state.adata_host.energy.extent(0);
        if (sim.dex.state.config.conserve_charge) {
            dex_tracers += 1;
        }
        sim.dex.interface_config.field_start_idx = n_hydro + n_extra;
        n_extra += dex_tracers;
    }
    sim.state.num_tracers = n_extra;
    const int n_total = n_hydro + n_extra;
    state.Q = Fp4d("Q", n_total, sz.zc, sz.yc, sz.xc);
    // NOTE(cmo): Zero this in case the ICs don't
    state.Q = 0.0_fp;
    state.W = Fp4d("W", n_total, sz.zc, sz.yc, sz.xc);
    sim.recon_scratch.RR = Fp4d("RR", n_total, sz.zc, sz.yc, sz.xc);
    sim.recon_scratch.RL = Fp4d("RL", n_total, sz.zc, sz.yc, sz.xc);
    // NOTE(cmo): The density flux is just rescaled for the tracer fields
    sim.fluxes.Fx = Fp4d("Fx", n_total, sz.zc, sz.yc, sz.xc);
    if (sim.num_dim > 1) {
        sim.fluxes.Fy = Fp4d("Fy", n_total, sz.zc, sz.yc, sz.xc);
    }
    if (sim.num_dim > 2) {
        sim.fluxes.Fz = Fp4d("Fz", n_total, sz.zc, sz.yc, sz.xc);
    }
    sim.sources.S = Fp4d("S", n_hydro, sz.zc, sz.yc, sz.xc);
    sim.state.mu0 = get_or<fp_t>(config, "simulation.mu0", sim.state.mu0);
    sim.state.p_mass = get_or<fp_t>(config, "simulation.p_mass", sim.state.p_mass);

    sim.state.cond.hypertc_kappa = get_or<fp_t>(config, "simulation.hypertc_kappa", sim.state.cond.hypertc_kappa);
    sim.state.cond.limited = get_or<bool>(config, "simulation.hypertc_limit", false);
    sim.state.cond.spitzer = get_or<bool>(config, "simulation.hypertc_spitzer", true);
}

template <typename FTraits>
void load_constant_boundary(
    const yakl::SArray<fp_t, 1, N_HYDRO_VARS<3, FLUID_WITH_MAX_VARS>>& arr,
    const YAML::Node& node
) {
    using Cons3 = Cons<3, FLUID_WITH_MAX_VARS>;
    using C = FTraits::cons;
    arr(I(C::Rho)) = node[I(Cons3::Rho)].as<fp_t>();
    arr(I(C::MomX)) = node[I(Cons3::MomX)].as<fp_t>();
    if constexpr (FTraits::is_mhd || FTraits::num_dim > 1) {
        arr(I(C::MomY)) = node[I(Cons3::MomY)].as<fp_t>();
    }
    if constexpr (FTraits::is_mhd || FTraits::num_dim > 2) {
        arr(I(C::MomZ)) = node[I(Cons3::MomZ)].as<fp_t>();
    }
    arr(I(C::Ene)) = node[I(Cons3::Ene)].as<fp_t>();
    if constexpr (FTraits::is_mhd) {
        arr(I(C::Bx)) = node[I(Cons3::Bx)].as<fp_t>();
        arr(I(C::By)) = node[I(Cons3::By)].as<fp_t>();
        arr(I(C::Bz)) = node[I(Cons3::Bz)].as<fp_t>();
        if constexpr (is_instance(FTraits::fluid_type, FluidType::GlmMhd)) {
            arr(I(C::Psi)) = node[I(Cons3::Psi)].as<fp_t>();
        }
        if constexpr (FTraits::has_hypertc) {
            arr(I(C::HeatF)) = node[I(Cons3::HeatF)].as<fp_t>();
        }
    }
}

void setup_boundaries(Simulation& sim, YAML::Node& config) {
    auto& bound = sim.state.boundaries;

    auto set_boundary = [&](BoundaryType& out, const std::string& bdry) {
        std::string bdry_string = get_or<std::string>(config, fmt::format("boundary.{}", bdry), "wall");
        out = find_associated_enum<BoundaryType>(BoundaryTypeName, NumBoundaryType, bdry_string);
    };

    set_boundary(bound.xs, "xs");
    set_boundary(bound.xe, "xe");
    if (sim.num_dim > 1) {
        set_boundary(bound.ys, "ys");
        set_boundary(bound.ye, "ye");
    }
    if (sim.num_dim > 2) {
        set_boundary(bound.zs, "zs");
        set_boundary(bound.ze, "ze");
    }

    auto check_and_load_constant = [&](
        const BoundaryType boundary,
        const decltype(bound.xs_const)& arr,
        const std::string& bdry) {
            if (boundary != BoundaryType::Constant) {
                return;
            }
            const auto name = fmt::format("{}_const", bdry);
            if (config["boundary"][name]){
                if (config["boundary"][name].IsSequence()) {
                    invoke_fluid_traits(
                        sim.num_dim,
                        sim.fluid_type,
                        [&]<typename FTraits>(FTraits) {
                            load_constant_boundary<FTraits>(arr, config["boundary"][name]);
                        });
                } else {
                    std::string vals = get_or<std::string>(config, fmt::format("boundary.{}_const", bdry), "xxx");
                    if (vals == "xxx") {
                        throw std::runtime_error(fmt::format("Provide constant boundary values for {}_const, or set to problem_supplied and fill in your problem setup function", bdry));
                    } else if (vals == "problem_supplied") {
                        return;
                    }
                }
            }
    };
    check_and_load_constant(bound.xs, bound.xs_const, "xs");
    check_and_load_constant(bound.xe, bound.xe_const, "xe");
    if (sim.num_dim > 1) {
        check_and_load_constant(bound.ys, bound.ys_const, "ys");
        check_and_load_constant(bound.ye, bound.ye_const, "ye");
    }
    if (sim.num_dim > 2) {
        check_and_load_constant(bound.zs, bound.zs_const, "zs");
        check_and_load_constant(bound.ze, bound.ze_const, "ze");
    }
}

void setup_hydro_fns(Simulation& sim, YAML::Node& config) {
    auto& scheme = sim.scheme;

    std::string recon_str = get_or<std::string>(config, "scheme.reconstruction", "muscl");
    scheme.reconstruction = find_associated_enum<Reconstruction>(ReconstructionName, NumReconstructionType, recon_str);

    const int min_ghost = min_ghost_cells(scheme.reconstruction);
    if (sim.state.sz.ng < min_ghost) {
        throw std::runtime_error(fmt::format("Insufficient ghost cells for reconstruction scheme {}, need minimum of {}.", ReconstructionName[int(scheme.reconstruction)], min_ghost));
    }

    std::string sl_str = get_or<std::string>(config, "scheme.slope_limiter", "monotonizedcentral");
    scheme.slope_limit = find_associated_enum<SlopeLimiter>(SlopeLimiterName, NumSlopeLimiterType, sl_str);

    std::string rs_str = get_or<std::string>(config, "scheme.riemann_solver", "hll");
    scheme.riemann_solver = find_associated_enum<RiemannSolver>(RiemannSolverName, NumRiemannSolverType, rs_str);

    std::string cfl_str = get_or<std::string>(config, "scheme.cfl_scheme", "maxsum");
    scheme.cfl_scheme = find_associated_enum<CflScheme>(CflSchemeName, NumCflScheme, cfl_str);

    select_hydro_fns(sim);
}

void setup_timestepper(Simulation& sim, YAML::Node& config) {
    auto& scheme = sim.scheme;

    std::string ts_str = get_or<std::string>(config, "timestep.scheme", "ssprk3");
    scheme.time_stepper = find_associated_enum<TimeStepScheme>(TimeStepName, NumTimeStepType, ts_str);

    select_timestepper(sim);

    sim.max_cfl = get_or<fp_t>(config, "timestep.max_cfl", 0.8_fp);
    sim.max_time = get_or<fp_t>(config, "timestep.max_time", 1.0_fp);
}

void setup_eos(Simulation& sim, YAML::Node& config) {
    sim.eos.init(sim, config);
}

void setup_problem(Simulation& sim, YAML::Node& config) {
    std::string problem_name = get_or<std::string>(config, "problem.name", "circular_explosion");
    get_problem_generator().dispatch(problem_name, sim, config);
}

void setup_output(Simulation& sim, YAML::Node& config) {
    sim.out_cfg.problem_name = get_or<std::string>(config, "problem.name", "circular_explosion");
    sim.out_cfg.filename = get_or<std::string>(config, "output.name", fmt::format("output_{}", sim.out_cfg.problem_name));
    sim.out_cfg.single_file = get_or<bool>(config, "output.single_file", true);
    sim.out_cfg.delta = get_or<f64>(config, "output.delta_t", 0.1);

    sim.out_cfg.output_count = 0;
    sim.out_cfg.prev_output_time = -1.0;
    sim.out_cfg.variables.conserved = get_or<bool>(config, "output.variables.conserved", true);
    sim.out_cfg.variables.primitive = get_or<bool>(config, "output.variables.primitive", false);
    sim.out_cfg.variables.fluxes = get_or<bool>(config, "output.variables.fluxes", false);
    sim.out_cfg.variables.source = get_or<bool>(config, "output.variables.source", false);

    if (!sim.write_output) {
        sim.write_output = write_output;
    }
}

void setup_dex_config(Simulation& sim, YAML::Node& config, const std::string& config_path) {
    if (!get_or<bool>(config, "dex.enable", false)) {
        return;
    }

    if (sim.num_dim != 2) {
        throw std::runtime_error("Dex integration only supports 2D models!");
    }

    bool rad_loss = get_or<std::string>(config, "dex.rad_loss", "None") != "None";
    bool advect = get_or<bool>(config, "dex.advect", false);
    bool time_dependent = get_or<bool>(config, "dex.time_dep_update", false);
    fp_t theta = get_or<fp_t>(config, "dex.theta", 1.0_fp);

    sim.dex.interface_config.advect = advect;
    sim.dex.interface_config.rad_loss = rad_loss;
    sim.dex.interface_config.time_dependent_updates = time_dependent;
    sim.dex.interface_config.theta = theta;
    sim.dex.init_config(sim, config, config_path);
}

void setup_dex(Simulation& sim, YAML::Node& config, bool is_restart) {
    if (!sim.dex.interface_config.enable) {
        return;
    }

    sim.dex.init(sim, config);
    if (!is_restart) {
        bool initial_rad_eq = get_or<bool>(config, "dex.initial_rad_eq", false);
        sim.dex.lte_init_aux_fields(sim);
        sim.dex.iterate(
            DexConvergence {
                .convergence = 1e-3_fp,
                .max_iter = 300,
            },
            IterateArgs {
                .first_iter = true
            }
        );
        if (initial_rad_eq) {
            fp_t delta_t = 10.0_fp;
            fp_t current_time = 0.0_fp;
            for (int ii = 0; ii < 300; ++ii) {
                fp_t tcv = sim.dex.min_characteristic_cooling_time();
                if (tcv > 2e3) {
                    break;
                }
                fp_t mean_temp = sim.dex.mean_temperature();
                fmt::println("\n~~~~~~~~~~~~ Step {} ~~~~~~~~~~~~~~", ii);
                fmt::println("t_cv = {:e} s", tcv);
                fmt::println("mean temperature = {:e} K", mean_temp);
                delta_t = std::min(0.1 * tcv, 50.0);
                sim.dex.update_temperature_rad_eq(delta_t);
                // fp_t pop_tol = std::min(FP(9e-4), fp_t(FP(5e-2) / (FP(0.01) * tcv)));
                // fp_t pop_tol = std::min(FP(5e-4), fp_t(FP(1e-2) / (FP(0.01) * tcv)));
                fmt::println("delta_t: {}", delta_t);
                fmt::println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");

                sim.dex.iterate(
                    DexConvergence {
                        .convergence = 8e-4_fp,
                        .max_iter = 300,
                    },
                    IterateArgs {
                        .first_iter = false
                    }
                );

                current_time += delta_t;
            }
        }
        sim.dex.copy_nhtot_to_rho(sim);
        sim.dex.copy_pops_to_aux_fields(sim);
        sim.dex.copy_to_eos(sim);
    }
    // NOTE(cmo): The first step conserves pressure, but from then on, we simply
    // load nhtot from rho and conserve charge
    sim.dex.state.config.conserve_pressure = false;
}

Simulation setup_sim(YAML::Node& config, const std::string& config_path, const RestartArgs& restart) {
    // TODO(cmo): Do an early check if problem.name is "from_file", and have a separate path for that.
    const int num_dim = get_or<int>(config, "simulation.num_dim", 0);
    if (num_dim < 1 || num_dim > 3) {
        throw std::runtime_error(fmt::format("simulation.num_dim = {}, must be in range [1, 3]", num_dim));
    }
    Simulation sim{};
    sim.num_dim = num_dim;

    // NOTE(cmo): This loads the model atoms etc, needed to setup the tracer fields
    setup_dex_config(sim, config, config_path);
    setup_grid(sim, config);
    setup_boundaries(sim, config);
    setup_hydro_fns(sim, config);
    setup_timestepper(sim, config);
    setup_divb_cleaning(sim, config);
    setup_hyperbolic_tc(sim, config);
    setup_eos(sim, config);
    setup_output(sim, config);
    setup_problem(sim, config);

    if (restart.restart) {
        load_restart(sim, restart.restart_from);
        setup_dex(sim, config, restart.restart);
    } else {
        sim.setup_ics(sim);
        // NOTE(cmo): Load the rest of dex, and the starting atmosphere
        setup_dex(sim, config, restart.restart);
        if (sim.update_eos) {
            sim.update_eos(sim);
        }
        fill_bcs(sim);
        // Write the header + ICs
        // TODO(cmo): Don't do this on restart
        sim.write_output(sim);
    }

    return sim;
}

}

#else
#endif