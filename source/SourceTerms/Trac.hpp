#if !defined(MOSSCAP_TRAC_HPP)
#define MOSSCAP_TRAC_HPP
// Transition Region Adaptive Conduction (TRAC), local form (no field-line
// tracing), after Johnston et al (2021). Here, the kappa_lim form is adopted,
// rather than the one containing the mass flux.
// Per cell,
//     kappa_eff = max(kappa_sp, (ds / delta) sqrt(kappa_sp |H - R| / T)),   s = kappa_sp / kappa_eff,
// with ds the field-aligned grid spacing (in-plane field only; an invariant
// direction needs no resolution), R and H the optically thin losses and
// heating. Conduction uses kappa_eff, and R and H are scaled by s, so the local
// length scale due to these terms sqrt(kappa_eff T / (s |H - R|)) is ds / delta
// wherever the transition region would be under-resolved, while the conductive
// flux through it is preserved. Off (kappa_eff = kappa_sp, s = 1) below t_min.
//
// Every source term states its TracRole (see Simulation.hpp). setup_trac is
// called after problem initialisation, and reorders and wraps the source terms
// so that:
// 1. EnergySourceSink terms are accumulated
// 2. Trac runs -> kappa_eff (in sim.trac.kappa) and applies scaled losses.
// 3. Conduction uses kappa_eff
// The wrapping ensures that source terms are scaled as needed if requested for
// output without them being explicitly aware of TRAC.
// Caveat Emptor: This ignores DexRT losses -- TRAC doesn't make sense there anyway.

#include "../Simulation.hpp"

namespace YAML { class Node; };

namespace Mosscap {

/// If conduction.trac.enable: allocate sim.trac, wrap the EnergySourceSink terms, add the trac term and order the
/// TRAC terms. Call after all source terms are registered (i.e. after the problem setup).
void setup_trac(Simulation& sim, YAML::Node& config);

}

#else
#endif
