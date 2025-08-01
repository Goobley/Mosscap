#if !defined(MOSSCAP_OUTPUT_HPP)
#define MOSSCAP_OUTPUT_HPP

#include "Types.hpp"
#include <string>

struct OutputOptions {
    bool conserved = true;
    bool primitive = false;
    bool fluxes = false;
    bool source = false;
};

struct OutputConfig {
    std::string filename;
    std::string problem_name;
    bool single_file;
    int output_count;
    f64 delta;
    f64 prev_output_time;
    OutputOptions variables;
};

struct Simulation;
bool write_output(Simulation& sim);

#else
#endif