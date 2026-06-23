#if !defined(MOSSCAP_GRAVITY_HPP)
#define MOSSCAP_GRAVITY_HPP
#include "../Simulation.hpp"

namespace YAML { class Node; };

namespace Mosscap {

struct GravityVals {
    fp_t x;
    fp_t y;
    fp_t z;
};

struct Simulation;
void setup_gravity(Simulation& sim, YAML::Node& config);

}

#else
#endif