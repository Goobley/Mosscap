#if !defined(MOSSCAP_SIMULATION_HPP)
#define MOSSCAP_SIMULATION_HPP

#include "Types.hpp"
#include "State.hpp"
#include "TimeStepping.hpp"
#include "Reconstruct.hpp"
#include "Riemann.hpp"
#include "Eos.hpp"

struct NumericalSchemes {
    Reconstruction reconstruction;
    SlopeLimiter slope_limit;
    RiemannSolver riemann_solver;
    TimeStepScheme time_stepper;
};

struct FluxFns {
    std::function<void(const Simulation&)> recon_x;
    std::function<void(const Simulation&)> recon_y;
    std::function<void(const Simulation&)> recon_z;
    std::function<void(const Simulation&)> flux_x;
    std::function<void(const Simulation&)> flux_y;
    std::function<void(const Simulation&)> flux_z;
};

struct Simulation {
    int num_dim;
    i64 current_step;
    fp_t max_cfl;
    fp_t time;
    fp_t max_time;
    Eos eos;
    State state;
    ReconScratch recon_scratch;
    Fluxes fluxes;
    Sources sources;
    TimeStepperStorage ts_storage;
    std::function<fp_t(const Simulation&)> compute_dt;
    std::function<void(Simulation&, fp_t)> time_step;
    std::function<void(const Simulation&)> user_bc;
    FluxFns flux_fns;
    NumericalSchemes scheme;
};

#else
#endif