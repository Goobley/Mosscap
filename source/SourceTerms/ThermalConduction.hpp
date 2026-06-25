#if !defined(MOSSCAP_THERMAL_CONDUCTION_HPP)
#define MOSSCAP_THERMAL_CONDUCTION_HPP
#include "../Simulation.hpp"

namespace YAML { class Node; };

namespace Mosscap {

struct ThermalConductionContext {
    bool enable = false;
    bool use_sts = true;
    bool saturate = true;
    bool spitzer = true;
    bool anisotropic = true;
    fp_t saturation_phi = 1.1_fp;
    fp_t kappa0 = 8e-12_fp;
};

void setup_thermal_conduction(Simulation& sim, YAML::Node& config);

}

#else
#endif