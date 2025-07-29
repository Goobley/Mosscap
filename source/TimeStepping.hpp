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
    template <int NumDim>
    static void time_step(Simulation& sim, fp_t dt);
};

#else
#endif