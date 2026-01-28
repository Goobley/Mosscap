#if !defined(MOSSCAP_SPONGE_HPP)
#define MOSSCAP_SPONGE_HPP


namespace YAML { class Node; };

namespace Mosscap {

struct Simulation;
void setup_sponge(Simulation& sim, YAML::Node& config);

}

#else
#endif