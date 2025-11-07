#if !defined(MOSSCAP_HYPERBOLIC_THERMAL_CONDUCTION_HPP)
#define MOSSCAP_HYPERBOLIC_THERMAL_CONDUCTION_HPP

#include "State.hpp"

namespace YAML { class Node; }

namespace Mosscap {
    struct Simulation;

    void setup_hyperbolic_tc(Simulation& sim, YAML::Node& config);
}

#else
#endif