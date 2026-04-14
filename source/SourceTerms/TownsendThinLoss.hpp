#if !defined(MOSSCAP_TOWNSEND_THIN_LOSS_HPP)
#define MOSSCAP_TOWNSEND_THIN_LOSS_HPP


namespace YAML { class Node; };

namespace Mosscap {

struct Simulation;
void setup_thin_loss(Simulation& sim, YAML::Node& config);

}

#else
#endif