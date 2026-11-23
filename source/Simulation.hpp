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

/// Tag for how a source term interacts with TRAC. EnergySourceSink terms (e.g.
/// thin_losses/background heating), will be wrapped so they can be
/// transparently scaled. Conduction is tagged to occur after these and receive
/// the correct scaling. Everything else is ignored by TRAC.
enum class TracRole {
    None = 0,
    Conduction,
    EnergySourceSink
};

/// A TracRole with no default, so every SourceTerm has to state its role (omitting .trac_role doesn't compile).
struct RequiredTracRole {
    TracRole role;
    constexpr RequiredTracRole(TracRole r) : role(r) {}
};

/// TRAC state, uninitialised unless conduction.trac.enable.
struct TracFields {
    Fp3d net; /// H - R this stage [W m-3], deposited by EnergySourceSink terms, applied and zeroed by the trac term
    Fp3d kappa; /// kappa_eff [W m-1 K-1]
    Fp3d scale; /// kappa_sp / kappa_eff, applied to the thin losses and heating (and Dex's loss with include_dex)
    Fp3d dex_net; /// -L_dex [W m-3] of the previous Dex solve if trac.include_dex
    Fp4d source_scratch; /// [var, z, y, x] target of the wrapped EnergySourceSink terms in a step, zero between uses

    bool active() const {
        return kappa.initialized();
    }
};

/// A source term adds its rates into `out` when that is initialised (the term evaluated on its own, e.g. for
/// output), and otherwise into the shared accumulator sim.sources.S -- see source_target.
using SourceFn = std::function<void(const Simulation& sim, const Fp4d& out)>;

struct SourceTerm {
    std::string name;
    SourceFn fn;
    std::function<void*()> get_context = [](){ return nullptr; };
    RequiredTracRole trac_role;
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
    bool zero_hydro_flux = false;
    Eos eos;
    State state;
    ReconScratch recon_scratch;
    Fluxes fluxes;
    Sources sources;
    TracFields trac; /// TRAC conductivity and source scaling, see SourceTerms/Trac.hpp
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


/// The array a source term writes into: `out` if initialised (evaluation of that term alone), else sim.sources.S.
inline const Fp4d& source_target(const Simulation& sim, const Fp4d& out) {
    return out.initialized() ? out : sim.sources.S;
}

}

#else
#endif