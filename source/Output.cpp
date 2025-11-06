#include "Output.hpp"
#include "Simulation.hpp"
#include "YAKL_netcdf.h"
#include "GitVersion.hpp"
#include "DexInterface.hpp"
#include <fmt/core.h>
#include <filesystem>
#include <regex>

namespace Mosscap {

void ncwrap (int ierr, int line) {
    if (ierr != NC_NOERR) {
        printf("NetCDF Error writing attributes at Output.cpp:%d\n", line);
        printf("%s\n",nc_strerror(ierr));
        Kokkos::abort(nc_strerror(ierr));
    }
}

template <typename FTraits>
void write_cons_header(yakl::SimpleNetCDF& nc, const Simulation& sim) {
    int ncid = nc.file.ncid;
    using Cons = FTraits::cons;
    int irho = I(Cons::Rho);
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "irho", NC_INT, 1, &irho),
        __LINE__
    );
    int imx = I(Cons::MomX);
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "imx", NC_INT, 1, &imx),
        __LINE__
    );
    if constexpr (FTraits::is_mhd || FTraits::num_dim > 1) {
        int imy = I(Cons::MomY);
        ncwrap(
            nc_put_att_int(ncid, NC_GLOBAL, "imy", NC_INT, 1, &imy),
            __LINE__
        );
    }
    if constexpr (FTraits::is_mhd || FTraits::num_dim > 2) {
        int imz = I(Cons::MomZ);
        ncwrap(
            nc_put_att_int(ncid, NC_GLOBAL, "imz", NC_INT, 1, &imz),
            __LINE__
        );
    }
    int iene = I(Cons::Ene);
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "iene", NC_INT, 1, &iene),
        __LINE__
    );
    if constexpr (FTraits::is_mhd) {
        int ibx = I(Cons::Bx);
        ncwrap(
            nc_put_att_int(ncid, NC_GLOBAL, "ibx", NC_INT, 1, &ibx),
            __LINE__
        );
        int iby = I(Cons::By);
        ncwrap(
            nc_put_att_int(ncid, NC_GLOBAL, "iby", NC_INT, 1, &iby),
            __LINE__
        );
        int ibz = I(Cons::Bz);
        ncwrap(
            nc_put_att_int(ncid, NC_GLOBAL, "ibz", NC_INT, 1, &ibz),
            __LINE__
        );
        if constexpr (is_instance(FTraits::fluid_type, FluidType::GlmMhd)) {
            int ipsi = I(Cons::Psi);
            ncwrap(
                nc_put_att_int(ncid, NC_GLOBAL, "ipsi", NC_INT, 1, &ipsi),
                __LINE__
            );
        }
        if constexpr (FTraits::has_hypertc) {
            int iheat = I(Cons::HeatF);
            ncwrap(
                nc_put_att_int(ncid, NC_GLOBAL, "iheat", NC_INT, 1, &iheat),
                __LINE__
            );
        }
    }
}

template <typename FTraits>
void write_prim_header(yakl::SimpleNetCDF& nc, const Simulation& sim) {
    int ncid = nc.file.ncid;
    using Prim = FTraits::prim;
    int irho = I(Prim::Rho);
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "irho", NC_INT, 1, &irho),
        __LINE__
    );
    int ivx = I(Prim::Vx);
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "ivx", NC_INT, 1, &ivx),
        __LINE__
    );
    if constexpr (FTraits::is_mhd || FTraits::num_dim > 1) {
        int ivy = I(Prim::Vy);
        ncwrap(
            nc_put_att_int(ncid, NC_GLOBAL, "ivy", NC_INT, 1, &ivy),
            __LINE__
        );
    }
    if constexpr (FTraits::is_mhd || FTraits::num_dim > 2) {
        int ivz = I(Prim::Vz);
        ncwrap(
            nc_put_att_int(ncid, NC_GLOBAL, "ivz", NC_INT, 1, &ivz),
            __LINE__
        );
    }
    int ipre = I(Prim::Pres);
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "ipre", NC_INT, 1, &ipre),
        __LINE__
    );
    if constexpr (FTraits::is_mhd) {
        int ibx = I(Prim::Bx);
        ncwrap(
            nc_put_att_int(ncid, NC_GLOBAL, "ibx", NC_INT, 1, &ibx),
            __LINE__
        );
        int iby = I(Prim::By);
        ncwrap(
            nc_put_att_int(ncid, NC_GLOBAL, "iby", NC_INT, 1, &iby),
            __LINE__
        );
        int ibz = I(Prim::Bz);
        ncwrap(
            nc_put_att_int(ncid, NC_GLOBAL, "ibz", NC_INT, 1, &ibz),
            __LINE__
        );
        if constexpr (is_instance(FTraits::fluid_type, FluidType::GlmMhd)) {
            int ipsi = I(Prim::Psi);
            ncwrap(
                nc_put_att_int(ncid, NC_GLOBAL, "ipsi", NC_INT, 1, &ipsi),
                __LINE__
            );
        }
        if constexpr (FTraits::has_hypertc) {
            int iheat = I(Prim::HeatF);
            ncwrap(
                nc_put_att_int(ncid, NC_GLOBAL, "iheat", NC_INT, 1, &iheat),
                __LINE__
            );
        }
    }
}

void write_header(yakl::SimpleNetCDF& nc, const Simulation& sim) {
    const auto& cfg = sim.out_cfg;

    int ncid = nc.file.ncid;
    std::string program = "mosscap";
    ncwrap(
        nc_put_att_text(ncid, NC_GLOBAL, "hydro_program", program.size(), program.c_str()),
        __LINE__
    );
    std::string git_hash(GIT_HASH);
    ncwrap(
        nc_put_att_text(ncid, NC_GLOBAL, "hydro_git_hash", git_hash.size(), git_hash.c_str()),
        __LINE__
    );
    ncwrap(
        nc_put_att_text(ncid, NC_GLOBAL, "problem", cfg.problem_name.size(), cfg.problem_name.c_str()),
        __LINE__
    );
    int num_ghost = sim.state.sz.ng;
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "num_ghost", NC_INT, 1, &num_ghost),
        __LINE__
    );
    int single_file = cfg.single_file;
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "single_file", NC_INT, 1, &single_file),
        __LINE__
    );
    int conserved = cfg.variables.conserved;
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "write_conserved", NC_INT, 1, &conserved),
        __LINE__
    );
    int primitive = cfg.variables.primitive;
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "write_primitive", NC_INT, 1, &primitive),
        __LINE__
    );
    int fluxes = cfg.variables.fluxes;
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "write_fluxes", NC_INT, 1, &fluxes),
        __LINE__
    );
    int source = cfg.variables.source;
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "write_source_term", NC_INT, 1, &source),
        __LINE__
    );

    invoke_fluid_traits(sim.num_dim, sim.fluid_type, [&]<typename FTraits>(FTraits) {
        if (cfg.variables.conserved) {
            write_cons_header<FTraits>(nc, sim);
        }
        if (cfg.variables.primitive) {
            write_prim_header<FTraits>(nc, sim);
        }
    });
}

void write_axes(yakl::SimpleNetCDF& nc, const Simulation& sim) {
    const auto& state = sim.state;
    const auto& sz = state.sz;
    nc.createDim("x", sz.xc);
    nc.createDim("y", sz.yc);
    nc.createDim("z", sz.zc);

    Fp1dHost x_pos("x_pos", sz.xc);
    for (int i = 0; i < sz.xc; ++i) {
        x_pos(i) = state.get_pos(i)(0);
    }
    nc.write(x_pos, "x", {"x"});


    if (sim.num_dim > 1) {
        Fp1dHost y_pos("y_pos", sz.yc);
        for (int i = 0; i < sz.yc; ++i) {
            y_pos(i) = state.get_pos(0, i)(1);
        }
        nc.write(y_pos, "y", {"y"});
    }

    if (sim.num_dim > 2) {
        Fp1dHost z_pos("z_pos", sz.zc);
        for (int i = 0; i < sz.zc; ++i) {
            z_pos(i) = state.get_pos(0, 0, i)(2);
        }
        nc.write(z_pos, "z", {"z"});
    }

    const bool single_file = sim.out_cfg.single_file;
    if (!single_file) {
        yakl::Array<const f64, 1, yakl::memHost> time_arr("time", &sim.time, 1);
        nc.write(time_arr, "time", {"time"});
    } else {
        // unlimited dim
        nc.createDim("time");
    }
}

static std::string get_filename(Simulation& sim, i64 write_count = -1) {
    const auto& cfg = sim.out_cfg;
    if (write_count < 0) {
        write_count = cfg.output_count;
    }
    std::string filename;
    if (cfg.single_file) {
        filename = fmt::format("{}.nc", cfg.filename);
    } else {
        filename = fmt::format("{}_{:05d}.nc", cfg.filename, write_count);
    }

    return filename;
}

static std::regex get_filename_regex(Simulation& sim) {
    const auto& cfg = sim.out_cfg;
    if (cfg.single_file) {
        return std::regex(fmt::format("{}\\.nc", cfg.filename));
    }

    return std::regex(fmt::format("{}_(\\d{{5}})\\.nc", cfg.filename));
}

bool write_output(Simulation& sim) {
    auto& cfg = sim.out_cfg;
    const bool single_file = cfg.single_file;
    std::string filename = get_filename(sim);

    yakl::SimpleNetCDF nc;
    if (cfg.prev_output_time < 0.0 || !cfg.single_file) {
        nc.create(filename, yakl::NETCDF_MODE_REPLACE);
        write_header(nc, sim);
        write_axes(nc, sim);
    } else {
        nc.open(filename, yakl::NETCDF_MODE_WRITE);
    }

    const auto& eos = sim.eos;
    if (!single_file) {
        if (cfg.variables.conserved) {
            nc.write(sim.state.Q, "Q", {"var", "z", "y", "x"});
        }
        if (cfg.variables.primitive) {
            nc.write(sim.state.W, "W", {"var", "z", "y", "x"});
        }
        if (cfg.variables.fluxes) {
            nc.write(sim.fluxes.Fx, "Fx", {"var", "z", "y", "x"});
            if (sim.num_dim > 1) {
                nc.write(sim.fluxes.Fy, "Fy", {"var", "z", "y", "x"});
            }
            if (sim.num_dim > 2) {
                nc.write(sim.fluxes.Fz, "Fz", {"var", "z", "y", "x"});
            }
        }
        if (cfg.variables.source) {
            nc.write(sim.sources.S, "S", {"conserved_var", "z", "y", "x"});
        }
        if (!eos.is_constant) {
            nc.write(eos.y_space, "ion_frac", {"z", "y", "x"});
            if (eos.T_space.initialized()) {
                nc.write(eos.T_space, "T", {"z", "y", "x"});
            }
        }
    } else {
        std::string time_name("time");
        int time_idx = cfg.output_count;
        nc.write1(sim.time, time_name, time_idx, time_name);
        if (cfg.variables.conserved) {
            nc.write1(sim.state.Q, "Q", {"var", "z", "y", "x"}, time_idx, time_name);
        }
        if (cfg.variables.primitive) {
            nc.write1(sim.state.W, "W", {"var", "z", "y", "x"}, time_idx, time_name);
        }
        if (cfg.variables.fluxes) {
            nc.write1(sim.fluxes.Fx, "Fx", {"var", "z", "y", "x"}, time_idx, time_name);
            if (sim.num_dim > 1) {
                nc.write1(sim.fluxes.Fy, "Fy", {"var", "z", "y", "x"}, time_idx, time_name);
            }
            if (sim.num_dim > 2) {
                nc.write1(sim.fluxes.Fz, "Fz", {"var", "z", "y", "x"}, time_idx, time_name);
            }
        }
        if (cfg.variables.source) {
            nc.write1(sim.sources.S, "S", {"conserved_var", "z", "y", "x"}, time_idx, time_name);
        }
        if (!eos.is_constant) {
            nc.write1(eos.y_space, "ion_frac", {"z", "y", "x"}, time_idx, time_name);
            if (eos.T_space.initialized()) {
                nc.write1(eos.T_space, "T", {"z", "y", "x"}, time_idx, time_name);
            }
        }
    }

    if (sim.dex.interface_config.enable) {
        sim.dex.write_output(sim, nc);
    }

    cfg.prev_output_time = sim.time;
    cfg.output_count += 1;

    nc.close();

    return true;
}

bool load_restart(Simulation& sim, i64 restart_from) {
    namespace fs = std::filesystem;
    std::string filename;
    auto& cfg = sim.out_cfg;
    if (cfg.single_file || restart_from > -1) {
        filename = get_filename(sim, restart_from);
        if (!fs::exists(filename)) {
            throw std::runtime_error(fmt::format("File \"{}\" does not exist", filename));
        }
    } else {
        i64 max_file_seen = -1;
        auto regex = get_filename_regex(sim);

        for (const auto& dir_entry : fs::directory_iterator(".")) {
            const std::string file = dir_entry.path().filename();
            std::smatch regex_match;
            if (std::regex_match(file, regex_match, regex)) {
                if (regex_match.size() == 2) {
                    max_file_seen = std::max(max_file_seen, i64(std::stoll(regex_match[1].str())));
                }
            }
        }

        if (max_file_seen == -1) {
            throw std::runtime_error(fmt::format("Unable to find snapshosts matching \"{}\" in the current directory.", cfg.filename));
        }
        filename = get_filename(sim, max_file_seen);
        restart_from = max_file_seen;
    }

    yakl::SimpleNetCDF nc;
    nc.open(filename, yakl::NETCDF_MODE_READ);
    int ncid = nc.file.ncid;

    int wrote_conserved;
    ncwrap(
        nc_get_att_int(ncid, NC_GLOBAL, "write_conserved", &wrote_conserved),
        __LINE__
    );
    if (!wrote_conserved) {
        throw std::runtime_error("Can't restart unless conserved variables are written");
    }

    f64 current_time = 0.0_fp;
    if (cfg.single_file) {
        if (restart_from < 0) {
            restart_from = nc.getDimSize("time") - 1;
        }
        nc.read1(sim.state.Q, "Q", restart_from, "time");
        auto& eos = sim.eos;
        if (!eos.is_constant) {
            nc.read1(eos.y_space, "ion_frac", restart_from, "time");
            if (eos.T_space.initialized()) {
                nc.read1(eos.T_space, "T", restart_from, "time");
            }
        }
        yakl::Array<f64, 1, yakl::memHost> time;
        nc.read(time, "time");
        current_time = time(restart_from);
    } else {
        nc.read(sim.state.Q, "Q");
        auto& eos = sim.eos;
        if (!eos.is_constant) {
            nc.read(eos.y_space, "ion_frac");
            if (eos.T_space.initialized()) {
                nc.read(eos.T_space, "T");
            }
        }
        nc.read(current_time, "time");
    }
    cfg.output_count = restart_from + 1;
    cfg.prev_output_time = current_time;
    sim.time = current_time;

    return true;
}

}
