// NOTE(claude): Work in progress -- not yet tested.
#if !defined(MOSSCAP_FRICTION_HPP)
#define MOSSCAP_FRICTION_HPP
#include "../Simulation.hpp"

namespace YAML { class Node; };

namespace Mosscap {

struct FrictionVals {
    fp_t gamma; // damping rate [1/s]
};

struct Simulation;
// Velocity-damping ("magneto-frictional") relaxation source term:
//   d(rho v)/dt -= gamma * rho v ,  d(E)/dt -= gamma * rho v^2
// drains bulk kinetic energy so an approximate initial state settles onto the
// numerical equilibrium. Registered only when sources.friction.enable is true.
void setup_friction(Simulation& sim, YAML::Node& config);

}

#else
#endif
