#if !defined(MOSSCAP_OUTPUT_HPP)
#define MOSSCAP_OUTPUT_HPP

#include "Types.hpp"
#include <string>
#include <vector>

namespace yakl { class SimpleNetCDF; }

namespace Mosscap {

/// A registered source term to be re-evaluated at output time, and the slots of
/// its (conserved variable) source array to write out.
struct SourceTermOutput {
    std::string name; /// Name the term is registered under in Simulation::compute_source_terms
    std::vector<int> indices; /// Resolved I(Cons::X) indices
    std::vector<std::string> slot_names; /// Names of those slots, used to name the output variables
};

struct OutputOptions {
    bool conserved = true;
    bool primitive = false;
    bool fluxes = false;
    bool source = false;
    std::vector<SourceTermOutput> source_terms;
};

struct OutputConfig {
    std::string filename;
    std::string problem_name;
    bool single_file;
    int output_count;
    f64 delta;
    f64 prev_output_time;
    /// Number of consecutive snapshots to write each time the output interval elapses.
    int n_burst = 1;
    /// Snapshots still owed for the burst in progress.
    int burst_remaining = 0;
    OutputOptions variables;
};

struct Simulation;
bool write_output(Simulation& sim);
/// Write a snapshot if the output interval has elapsed, or if a burst is in
/// progress. Returns whether anything was written.
bool maybe_write_output(Simulation& sim);
bool load_restart(Simulation& sim, i64 restart_from);

}

#else
#endif