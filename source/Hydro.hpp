#if !defined(MOSSCAP_HYDRO_HPP)
#define MOSSCAP_HYDRO_HPP

#include "Types.hpp"
#include "Simulation.hpp"
#include "Reconstruct.hpp"
#include "Riemann.hpp"

namespace Mosscap {

void global_cons_to_prim(const Simulation& sim);

void compute_hydro_fluxes(const Simulation& sim);
void select_hydro_fns(Simulation& sim);

f64 compute_dt(const Simulation& sim);

}

#else
#endif