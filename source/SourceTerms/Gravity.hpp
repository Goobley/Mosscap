#if !defined(MOSSCAP_GRAVITY_HPP)
#define MOSSCAP_GRAVITY_HPP

namespace YAML { class Node; };

namespace Mosscap {

struct Simulation;
void setup_gravity(Simulation& sim, YAML::Node& config);

}

#else
#endif