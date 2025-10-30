#if !defined(MOSSCAP_SIMULATION_HPP)
#define MOSSCAP_SIMULATION_HPP

#include "Types.hpp"
#include "State.hpp"
#include "TimeStepping.hpp"
#include "Reconstruct.hpp"
#include "Riemann.hpp"
#include "Eos.hpp"
#include "Output.hpp"
#include "DexInterface.hpp"

#include <vector>

namespace Mosscap {

enum class CflScheme {
    MaxSum = 0,
    MinPerAxis,
};
constexpr const char* CflSchemeName[] = {
    "maxsum",
    "minperaxis",
};
constexpr int NumCflScheme = sizeof(CflSchemeName) / sizeof(CflSchemeName[0]);

struct NumericalSchemes {
    Reconstruction reconstruction;
    SlopeLimiter slope_limit;
    RiemannSolver riemann_solver;
    TimeStepScheme time_stepper;
    CflScheme cfl_scheme;
};

struct FluxFns {
    std::function<void(const Simulation&)> recon_x;
    std::function<void(const Simulation&)> recon_y;
    std::function<void(const Simulation&)> recon_z;
    std::function<void(const Simulation&)> flux_x;
    std::function<void(const Simulation&)> flux_y;
    std::function<void(const Simulation&)> flux_z;
};

struct SourceTerm {
    std::string name;
    std::function<void(const Simulation&)> fn;
};

struct Simulation {
    int num_dim;
    FluidType fluid_type;
    i64 current_step;
    fp_t max_cfl;
    f64 time;
    f64 max_time;
    fp_t dt = 0.0_fp;
    fp_t dt_sub = 0.0_fp;
    Eos eos;
    State state;
    ReconScratch recon_scratch;
    Fluxes fluxes;
    Sources sources;
    TimeStepperStorage ts_storage;
    std::function<void(Simulation&)> setup_ics;
    std::function<f64(const Simulation&)> compute_dt;
    std::function<void(const Simulation&)> clean_divb;
    std::function<void(const Simulation&)> update_eos;
    std::function<void(Simulation&, fp_t)> time_step;
    std::function<void(const Simulation&)> user_bc;
    std::vector<SourceTerm> compute_source_terms;
    std::function<bool(Simulation&)> write_output;
    FluxFns flux_fns;
    NumericalSchemes scheme;
    OutputConfig out_cfg;
    DexInterface dex;
};

}

#else
#endif