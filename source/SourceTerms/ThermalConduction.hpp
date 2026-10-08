#if !defined(MOSSCAP_THERMAL_CONDUCTION_HPP)
#define MOSSCAP_THERMAL_CONDUCTION_HPP
#include "../Simulation.hpp"

namespace YAML { class Node; };

namespace Mosscap {

enum class ConductionMethod {
    Explicit = 0,
    Sts,
    Hybrid2
};

struct ThermalConductionContext {
    bool enable = false;
    bool use_sts = true;
    ConductionMethod method = ConductionMethod::Sts;
    bool saturate = true;
    bool spitzer = true;
    bool anisotropic = true;
    fp_t saturation_phi = 1.1_fp;
    fp_t kappa0 = 8e-12_fp;
    // hybrid2 (sources.thermal_conduction.hybrid2.*), see Hybrid2Conduction.hpp
    fp_t h2_s_max = 10.0_fp; /// substep each RK stage when S = dt / dt_diff exceeds this (<= 0: never)
    fp_t h2_alpha = 4.0_fp; /// tau floor in units of the step
    fp_t h2_f_wave = 0.8_fp; /// conduction-wave Courant number pinned by tau
    fp_t h2_r0 = 1.5_fp; /// T_max / T_min below which the target is purely symmetric
    fp_t h2_r1 = 3.0_fp; /// T_max / T_min above which the target is purely the face flux
    bool h2_limit = true; /// lim_nodiss: lower bound (box minimum of T) on the applied face fluxes
    int h2_box_radius = 0; /// limiter box radius in cells (<= 0: per-step diffusion length)
};

void setup_thermal_conduction(Simulation& sim, YAML::Node& config);

}

#else
#endif