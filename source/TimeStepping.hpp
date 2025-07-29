#if !defined(MOSSCAP_TIME_STEPPING_HPP)
#define MOSSCAP_TIME_STEPPING_HPP

#include "State.hpp"

struct Simulation;

enum class TimeStepScheme {
    Rk2,
    SspRk3,
    SspRk4
};

struct TimeStepperStorage {
    std::vector<Fp4d> Q_old;
};

template <TimeStepScheme scheme>
struct TimeStepper {
    static bool init(Simulation& sim);
    static void time_step(Simulation& sim, fp_t dt);
};

// extern template struct TimeStepper<TimeStepScheme::Rk2>;
// extern template struct TimeStepper<TimeStepScheme::SspRk3>;

#else
#endif