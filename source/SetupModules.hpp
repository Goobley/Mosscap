#if !defined(MOSSCAP_SETUP_MODULES_HPP)
#define MOSSCAP_SETUP_MODULES_HPP

#include "Simulation.hpp"
#include "yaml-cpp/yaml.h"

void setup_grid(Simulation& sim, const YAML::Node& config) {
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

    const int n_hydro = get_num_hydro_vars(sim.num_dim);
    state.Q = Fp4d("Q", n_hydro, sz.zc, sz.yc, sz.xc);
    state.W = Fp4d("W", n_hydro, sz.zc, sz.yc, sz.xc);
    sim.recon_scratch.RR = Fp4d("RR", n_hydro, sz.zc, sz.yc, sz.xc);
    sim.recon_scratch.RL = Fp4d("RL", n_hydro, sz.zc, sz.yc, sz.xc);
    sim.fluxes.Fx = Fp4d("Fx", n_hydro, sz.zc, sz.yc, sz.xc);
    if (sim.num_dim > 1) {
        sim.fluxes.Fy = Fp4d("Fy", n_hydro, sz.zc, sz.yc, sz.xc);
    }
    if (sim.num_dim > 2) {
        sim.fluxes.Fz = Fp4d("Fz", n_hydro, sz.zc, sz.yc, sz.xc);
    }
}

void setup_boundaries(Simulation& sim, const YAML::Node& config) {
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
    } else {
        set_boundary(bound.zs, "zs");
        set_boundary(bound.ze, "ze");
    }

    // TODO(cmo): Check if any are constant, and load the values if so.
}

void setup_hydro_fns(Simulation& sim, const YAML::Node& config) {
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

void setup_timestepper(Simulation& sim, const YAML::Node& config) {
    auto& scheme = sim.scheme;

    std::string ts_str = get_or<std::string>(config, "timestep.scheme", "ssprk3");
    scheme.time_stepper = find_associated_enum<TimeStepScheme>(TimeStepName, NumTimeStepType, ts_str);

    select_timestepper(sim);

    sim.max_cfl = get_or<fp_t>(config, "timestep.max_cfl", FP(0.8));
    sim.max_time = get_or<fp_t>(config, "timestep.max_time", FP(1.0));
}

void setup_eos(Simulation& sim, const YAML::Node& config) {
    sim.eos.init(sim, config);
}

void setup_problem(Simulation& sim, const YAML::Node& config) {
    std::string problem_name = get_or<std::string>(config, "problem.name", "circular_explosion");
    get_problem_generator().dispatch(problem_name, sim, config);
}

Simulation setup_sim(const YAML::Node& config) {
    // TODO(cmo): Do an early check if problem.name is "from_file", and have a separate path for that.
    const int num_dim = get_or<int>(config, "simulation.num_dim", 0);
    if (num_dim < 1 || num_dim > 3) {
        throw std::runtime_error(fmt::format("simulation.num_dim = {}, must be in range [1, 3]", num_dim));
    }
    Simulation sim{};
    sim.num_dim = num_dim;

    setup_grid(sim, config);
    setup_boundaries(sim, config);
    setup_hydro_fns(sim, config);
    setup_timestepper(sim, config);
    setup_eos(sim, config);
    setup_problem(sim, config);

    return sim;
}

#else
#endif