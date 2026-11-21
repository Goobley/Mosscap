#if !defined(MOSSCAP_THERMAL_CONDUCTION_HPP)
#define MOSSCAP_THERMAL_CONDUCTION_HPP
#include "../Simulation.hpp"

namespace YAML { class Node; };

namespace Mosscap {

enum class ConductionMethod {
    Explicit = 0,
    Sts,
    SemiImp
};

enum class SemiImpCross {
    SharmaSymmetric = 0,
    Mean
};

struct ThermalConductionContext {
    bool enable = false;
    bool use_sts = true;
    ConductionMethod method = ConductionMethod::Sts;
    bool anisotropic = true;
    // Copies of the shared parameters in State::cond (the `conduction` config block), see Conduction.
    bool saturate = false;
    bool spitzer = true;
    fp_t saturation_phi = 1.1_fp;
    fp_t kappa0 = 8e-12_fp;
    bool out_of_plane_field = true;
    Fp3d trac_kappa; /// sim.trac.kappa while TRAC is active (refreshed each call), else uninitialised

    // semi_implicit (sources.thermal_conduction.semi_implicit.*), see SemiImplicitConduction.hpp
    SemiImpCross semi_imp_cross = SemiImpCross::SharmaSymmetric; /// explicit part: Sharma & Hammett's limited symmetric residual | unlimited asymmetric mean
    fp_t semi_imp_s_max = 0.0_fp; /// substep each RK stage when S = dt_sub / dt_diff exceeds this (<= 0: never)
    i64 semi_imp_steps = 0; /// semi-implicit steps taken (alternates the sweep order)
};

void setup_thermal_conduction(Simulation& sim, YAML::Node& config);

}

#else
#endif