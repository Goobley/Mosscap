#if !defined(MOSSCAP_TIME_DEPENDENT_POPULATIONS)

#include "../DexRT/source/Config.hpp"
#include "../DexRT/source/Types.hpp"
#include "../DexRT/source/Constants.hpp"
#include "../DexRT/source/Utils.hpp"

struct KineticEqOptions {
    fp_t dt = FP(0.0);
    fp_t theta = FP(1.0); /// The fraction of the update to employ using the time-advanced rates. 1.0 => backwards euler, 0.0 forwards Euler
    bool initial_iter = false;
    const Fp2d& predicted_pops = Fp2d(); /// To be filled if theta < 1.0 and initial_iter = true, on exit contains Gamma . n
    /// When computing relative change, ignore the change in populations with a
    /// starting fraction lower than this
    fp_t ignore_change_below_ntot_frac = FP(0.0);
    /// If only solving one atom, its index in adata. A negative number implies all atoms.
    int only_atom = -1;
};

struct TimeDepNrPostUpdateOptions {
    fp_t dt = FP(0.0);
    fp_t theta = FP(1.0); /// The fraction of the update to employ using the time-advanced rates. 1.0 => backwards euler, 0.0 forwards Euler
    const Fp2d& predicted_pops = Fp2d(); /// Frozen Gamma_old * n_old array filled by time_dep_update.
    fp_t ignore_change_below_ntot_frac = FP(0.0);
};

template <typename State>
fp_t time_dep_update(State& state, const Fp2d& prev_pops, const KineticEqOptions& args = KineticEqOptions());
template <typename State>
fp_t time_dep_nr_post_update(State& state, const Fp2d& prev_pops, const TimeDepNrPostUpdateOptions& args = TimeDepNrPostUpdateOptions());

#else
#endif
