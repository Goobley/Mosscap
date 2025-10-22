#if !defined(MOSSCAP_DIV_B_CLEANING_HPP)
#define MOSSCAP_DIV_B_CLEANING_HPP

#include "State.hpp"

namespace YAML { class Node; }

namespace Mosscap {
    struct Simulation;

    enum class DivBCleaningSchemes {
        Linde = 0
    };

    void clean_divb(const Simulation& sim);
    void setup_divb_cleaning(Simulation& sim, YAML::Node& config);
}

#else
#endif