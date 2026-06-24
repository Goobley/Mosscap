#if !defined(MOSSCAP_REPLACE_SMALL_VALUES_HPP)
#define MOSSCAP_REPLACE_SMALL_VALUES_HPP
#include "../Simulation.hpp"

namespace YAML { class Node; };

namespace Mosscap {

struct ReplaceSmallValuesContext {
    fp_t density_floor = 0.0_fp;
    fp_t pressure_floor = 0.0_fp;
    bool zero_momentum = true;
    bool enable = false;
};

struct Simulation;
void setup_replace_small_values(Simulation& sim, YAML::Node& config);

}

#else
#endif