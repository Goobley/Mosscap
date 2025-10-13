#if !defined(MOSSCAP_SETUP_MODULES_HPP)
#define MOSSCAP_SETUP_MODULES_HPP

#include "Simulation.hpp"
#include "yaml-cpp/yaml.h"

namespace Mosscap {

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

    const int n_hydro = get_num_hydro_vars(sim.num_dim);
    int n_extra = get_or<int>(config, "simulation.n_extra_fields", 0);
    if (sim.dex.interface_config.enable) {
        int dex_tracers = sim.dex.state.adata_host.energy.extent(0);
        if (sim.dex.state.config.conserve_charge) {
            dex_tracers += 1;
        }
        sim.dex.interface_config.field_start_idx = n_extra;
        n_extra += dex_tracers;
    }
    sim.state.num_tracers = n_extra;
    const int n_total = n_hydro + n_extra;
    state.Q = Fp4d("Q", n_total, sz.zc, sz.yc, sz.xc);
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

    // TODO(cmo): Check if any are constant, and load the values if so.
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
                    using Cons3 = Cons<3>;
                    auto node = config["boundary"][fmt::format("{}_const", bdry)];
                    if (sim.num_dim == 1) {
                        using Cons1 = Cons<1>;
                        arr(I(Cons1::Rho)) = node[I(Cons3::Rho)].as<fp_t>();
                        arr(I(Cons1::MomX)) = node[I(Cons3::MomX)].as<fp_t>();
                        arr(I(Cons1::Ene)) = node[I(Cons3::Ene)].as<fp_t>();
                    } else if (sim.num_dim == 2) {
                        using Cons2 = Cons<2>;
                        arr(I(Cons2::Rho)) = node[I(Cons3::Rho)].as<fp_t>();
                        arr(I(Cons2::MomX)) = node[I(Cons3::MomX)].as<fp_t>();
                        arr(I(Cons2::MomY)) = node[I(Cons3::MomY)].as<fp_t>();
                        arr(I(Cons2::Ene)) = node[I(Cons3::Ene)].as<fp_t>();
                    } else {
                        arr(I(Cons3::Rho)) = node[I(Cons3::Rho)].as<fp_t>();
                        arr(I(Cons3::MomX)) = node[I(Cons3::MomX)].as<fp_t>();
                        arr(I(Cons3::MomY)) = node[I(Cons3::MomY)].as<fp_t>();
                        arr(I(Cons3::MomZ)) = node[I(Cons3::MomZ)].as<fp_t>();
                        arr(I(Cons3::Ene)) = node[I(Cons3::Ene)].as<fp_t>();
                    }
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

void setup_dex_config(Simulation& sim, YAML::Node& config) {
    if (!get_or<bool>(config, "dex.enable", false)) {
        return;
    }

    if (sim.num_dim != 2) {
        throw std::runtime_error("Dex integration only supports 2D models!");
    }

    sim.dex.init_config(sim, config);
}

void setup_dex(Simulation& sim, YAML::Node& config) {
    if (!sim.dex.interface_config.enable) {
        return;
    }

    sim.dex.init(sim, config);
}

Simulation setup_sim(YAML::Node& config) {
    // TODO(cmo): Do an early check if problem.name is "from_file", and have a separate path for that.
    const int num_dim = get_or<int>(config, "simulation.num_dim", 0);
    if (num_dim < 1 || num_dim > 3) {
        throw std::runtime_error(fmt::format("simulation.num_dim = {}, must be in range [1, 3]", num_dim));
    }
    Simulation sim{};
    sim.num_dim = num_dim;

    // NOTE(cmo): This loads the model atoms etc, needed to setup the tracer fields
    setup_dex_config(sim, config);
    setup_grid(sim, config);
    setup_boundaries(sim, config);
    setup_hydro_fns(sim, config);
    setup_timestepper(sim, config);
    setup_eos(sim, config);
    setup_output(sim, config);
    setup_problem(sim, config);

    // Write the header + ICs
    if (sim.update_eos) {
        sim.update_eos(sim);
    }
    // NOTE(cmo): Load the rest of dex, and the starting atmosphere
    setup_dex(sim, config);
    fill_bcs(sim);
    // TODO(cmo): Don't do this on restart
    sim.write_output(sim);

    return sim;
}

}

#else
#endif