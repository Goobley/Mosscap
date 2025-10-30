#if !defined(MOSSCAP_TIME_DEPENDENT_POPULATIONS)

#include "../DexRT/source/Config.hpp"
#include "../DexRT/source/Types.hpp"
#include "../DexRT/source/Constants.hpp"
#include "../DexRT/source/Utils.hpp"

struct KineticEqOptions {
    fp_t dt = FP(0.0);
    /// When computing relative change, ignore the change in populations with a
    /// starting fraction lower than this
    fp_t ignore_change_below_ntot_frac = FP(0.0);
    /// If only solving one atom, its index in adata. A negative number implies all atoms.
    int only_atom = -1;
};

struct TimeDepNrPostUpdateOptions {
    fp_t dt = FP(0.0);
    /// When computing relative change, ignore the change in populations with a
    /// starting fraction lower than this
    fp_t ignore_change_below_ntot_frac = FP(0.0);
    bool conserve_pressure = false;
    /// Set the total baryon abundance relative to H (e.g. 1.1 for a 10% He mix). Negative for default
    fp_t total_abund = FP(-1.0);
};

template <typename State>
fp_t time_dep_update(State& state, const Fp2d& prev_pops, const KineticEqOptions& args = KineticEqOptions());
template <typename State>
fp_t time_dep_nr_post_update(State& state, const Fp2d& prev_pops, const TimeDepNrPostUpdateOptions& args = TimeDepNrPostUpdateOptions());

#else
#endif