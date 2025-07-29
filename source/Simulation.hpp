#if !defined(MOSSCAP_SIMULATION_HPP)
#define MOSSCAP_SIMULATION_HPP

#include "Types.hpp"
#include "State.hpp"
#include "TimeStepping.hpp"
#include "Reconstruct.hpp"
#include "Riemann.hpp"
#include "Boundaries.hpp"

struct NumericalSchemes {
    Reconstruction reconstruction;
    SlopeLimiter slope_limit;
    RiemannSolver riemann_solver;
    TimeStepScheme time_stepper;
};

struct Simulation {
    i64 current_step;
    fp_t max_cfl;
    fp_t time;
    fp_t max_time;
    State state;
    ReconScratch recon_scratch;
    Fluxes fluxes;
    Sources sources;
    TimeStepperStorage ts_storage;
    std::function<fp_t(const State&)> compute_timestep;
    std::function<void(const State&)> time_integrate;
    std::function<void(const Simulation&)> compute_hydro_fluxes;
    void step();
    void write();
};

#else
#endif