#if !defined(MOSSCAP_GRAVITY_HPP)
#define MOSSCAP_GRAVITY_HPP

struct Simulation;
namespace YAML { class Node; };
void setup_gravity(Simulation& sim, YAML::Node& config);

#else
#endif