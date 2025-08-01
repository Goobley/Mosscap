#include "Config.hpp"
#include "Hydro.hpp"
#include "Boundaries.hpp"
#include <fmt/core.h>
#include "YAKL_netcdf.h"
#include "ProblemGenerator/ProblemGenerator.hpp"
#include "argparse/argparse.hpp"
#include "GitVersion.hpp"
#include "MosscapConfig.hpp"
#include "SetupModules.hpp"
#include "SourceTerms.hpp"

void write_output_inner(const Simulation& sim, int i, fp_t time) {
    global_cons_to_prim(sim);
    yakl::SimpleNetCDF nc;
    std::string name = fmt::format("out_{:06d}.nc", i);
    nc.create(name, yakl::NETCDF_MODE_REPLACE);

    nc.write(sim.state.Q, "Q", {"var", "z", "y", "x"});
    nc.write(sim.state.W, "W", {"var", "z", "y", "x"});
    nc.write(time, "time");
}

int main(int argc, char** argv) {
    argparse::ArgumentParser program("Mosscap", GIT_HASH);
    program
        .add_argument("--config")
        .default_value(std::string("mosscap.yaml"))
        .help("Path to config file")
        .metavar("FILE");
    // TODO(cmo): restart
    program.add_epilog("Simple accelerated n-dimensional hydro code");
    program.parse_known_args(argc, argv);

    std::string config_path(program.get<std::string>("--config"));
    YAML::Node config = YAML::LoadFile(config_path);

    Kokkos::initialize(argc, argv);
    yakl::init(
        yakl::InitConfig()
            .set_pool_size_mb(get_or<f64>(config, "system.mem_pool_gb", 2.0) * 1024)
    );
    {
        Simulation sim = setup_sim(config);

        fill_bcs(sim);

        while (sim.time < sim.max_time) {
            const f64 dt = compute_dt(sim);
            sim.time_step(sim, dt);
            if (sim.time >= sim.out_cfg.prev_output_time + sim.out_cfg.delta) {
                sim.write_output(sim);
                // TODO(cmo): This is printing very small dt due to step rounding... save dt natural too?
                fmt::println("t = {:.03f}, dt = {:.03e}, iter = {}", sim.time, dt, sim.current_step);
            }
        }
        sim.write_output(sim);
    }
    yakl::finalize();
    Kokkos::finalize();

    return 0;
}