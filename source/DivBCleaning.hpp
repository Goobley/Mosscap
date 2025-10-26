#if !defined(MOSSCAP_DIV_B_CLEANING_HPP)
#define MOSSCAP_DIV_B_CLEANING_HPP

#include "State.hpp"

namespace YAML { class Node; }

namespace Mosscap {
    struct Simulation;

    enum class DivBCleaningScheme {
        Linde = 0,
        Janhunen,
        LindeJanhunen,
        Powell8Wave,
        Projection
    };
    constexpr const char* DivBCleaningName[] = {
        "linde",
        "janhunen",
        "lindejanhunen",
        "powell8wave",
        "projection"
    };
    constexpr int NumDivBCleaningScheme = sizeof(DivBCleaningName) / sizeof(DivBCleaningName[0]);

    Fp3d compute_divb(const Simulation& sim, fp_t* max_divb);
    void clean_divb(const Simulation& sim);
    void setup_divb_cleaning(Simulation& sim, YAML::Node& config);
}

#else
#endif