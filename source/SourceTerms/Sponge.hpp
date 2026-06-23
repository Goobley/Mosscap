#if !defined(MOSSCAP_SPONGE_HPP)
#define MOSSCAP_SPONGE_HPP
#include "../Simulation.hpp"

namespace YAML { class Node; };

namespace Mosscap {

struct SpongeParams {
    /// Amplitude on exp
    fp_t A;
    /// Decay param in exp
    fp_t B;
    /// damp for x <= xs
    fp_t xs;
    /// damp for x >= xe
    fp_t xe;
    /// damp for y <= ys
    fp_t ys;
    /// damp for y >= ye
    fp_t ye;
    /// damp for z <= zs
    fp_t zs;
    /// damp for z >= ze
    fp_t ze;
    /// Use values from edge of grid rather than constant
    bool use_edge_vals;
    /// Damp the GLM psi term to 0 when use_edge_vals is enabled
    bool damp_psi_to_zero;
    /// Ignore the GLM psi term (don't damp them)
    bool ignore_psi;
};

struct Simulation;
void setup_sponge(Simulation& sim, YAML::Node& config);

}

#else
#endif