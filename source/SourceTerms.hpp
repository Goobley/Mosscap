#if !defined(MOSSCAP_SOURCE_TERMS_HPP)
#define MOSSCAP_SOURCE_TERMS_HPP

#include "Simulation.hpp"

inline void zero_source_terms(const Simulation& sim) {
    sim.sources.S = FP(0.0);
}

inline void compute_source_terms(const Simulation& sim) {
    for (const auto& fn : sim.compute_source_terms) {
        fn.fn(sim);
    }
}

inline int source_term_index(const Simulation& sim, std::string name) {
    for (int i = 0; i < sim.compute_source_terms.size(); ++i) {
        if (name == sim.compute_source_terms[i].name) {
            return i;
        }
    }
    return sim.compute_source_terms.size();
}

#else
#endif