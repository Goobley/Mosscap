#include "Output.hpp"
#include "Simulation.hpp"
#include "YAKL_netcdf.h"
#include "GitVersion.hpp"
#include <fmt/core.h>

void ncwrap (int ierr, int line) {
    if (ierr != NC_NOERR) {
        printf("NetCDF Error writing attributes at Output.cpp:%d\n", line);
        printf("%s\n",nc_strerror(ierr));
        Kokkos::abort(nc_strerror(ierr));
    }
}

template <int NumDim>
void write_cons_header(yakl::SimpleNetCDF& nc, const Simulation& sim) {
    int ncid = nc.file.ncid;
    using Cons = Cons<NumDim>;
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
    if constexpr (NumDim > 1) {
        int imy = I(Cons::MomY);
        ncwrap(
            nc_put_att_int(ncid, NC_GLOBAL, "imy", NC_INT, 1, &imy),
            __LINE__
        );
    }
    if constexpr (NumDim > 2) {
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
}

template <int NumDim>
void write_prim_header(yakl::SimpleNetCDF& nc, const Simulation& sim) {
    int ncid = nc.file.ncid;
    using Prim = Prim<NumDim>;
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
    if constexpr (NumDim > 1) {
        int ivy = I(Prim::Vy);
        ncwrap(
            nc_put_att_int(ncid, NC_GLOBAL, "ivy", NC_INT, 1, &ivy),
            __LINE__
        );
    }
    if constexpr (NumDim > 2) {
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
}

void write_header(yakl::SimpleNetCDF& nc, const Simulation& sim) {
    const auto& cfg = sim.out_cfg;

    int ncid = nc.file.ncid;
    std::string program = "mosscap";
    ncwrap(
        nc_put_att_text(ncid, NC_GLOBAL, "program", program.size(), program.c_str()),
        __LINE__
    );
    std::string git_hash(GIT_HASH);
    ncwrap(
        nc_put_att_text(ncid, NC_GLOBAL, "git_hash", git_hash.size(), git_hash.c_str()),
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

    if (sim.num_dim == 1) {
        if (cfg.variables.conserved) {
            write_cons_header<1>(nc, sim);
        }
        if (cfg.variables.primitive) {
            write_prim_header<1>(nc, sim);
        }
    } else if (sim.num_dim == 2) {
        if (cfg.variables.conserved) {
            write_cons_header<2>(nc, sim);
        }
        if (cfg.variables.primitive) {
            write_prim_header<2>(nc, sim);
        }
    } else {
        if (cfg.variables.conserved) {
            write_cons_header<3>(nc, sim);
        }
        if (cfg.variables.primitive) {
            write_prim_header<3>(nc, sim);
        }
    }
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

bool write_output(Simulation& sim) {
    std::string filename;
    auto& cfg = sim.out_cfg;
    const bool single_file = cfg.single_file;
    if (single_file) {
        filename = fmt::format("{}.nc", cfg.filename);
    } else {
        filename = fmt::format("{}_{:05d}.nc", cfg.filename, cfg.output_count);
    }

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
            nc.write(sim.fluxes.Fy, "Fy", {"var", "z", "y", "x"});
            nc.write(sim.fluxes.Fz, "Fz", {"var", "z", "y", "x"});
        }
        if (cfg.variables.source) {
            nc.write(sim.sources.S, "S", {"var", "z", "y", "x"});
        }
        if (!eos.is_constant) {
            nc.write(eos.y_space, "ion_frac", {"z", "y", "x"});
            if (eos.T_space.initialized()) {
                nc.write(eos.T_space, "T", {"z", "y", "x"});
            }
        }
    } else {
        std::string time_name("time");
        int time_idx = nc.getDimSize(time_name);
        nc.write1(sim.time, time_name, time_idx, time_name);
        if (cfg.variables.conserved) {
            nc.write1(sim.state.Q, "Q", {"var", "z", "y", "x"}, time_idx, time_name);
        }
        if (cfg.variables.primitive) {
            nc.write1(sim.state.W, "W", {"var", "z", "y", "x"}, time_idx, time_name);
        }
        if (cfg.variables.fluxes) {
            nc.write1(sim.fluxes.Fx, "Fx", {"var", "z", "y", "x"}, time_idx, time_name);
            nc.write1(sim.fluxes.Fy, "Fy", {"var", "z", "y", "x"}, time_idx, time_name);
            nc.write1(sim.fluxes.Fz, "Fz", {"var", "z", "y", "x"}, time_idx, time_name);
        }
        if (cfg.variables.source) {
            nc.write1(sim.sources.S, "S", {"var", "z", "y", "x"}, time_idx, time_name);
        }
        if (!eos.is_constant) {
            nc.write1(eos.y_space, "ion_frac", {"z", "y", "x"}, time_idx, time_name);
            if (eos.T_space.initialized()) {
                nc.write1(eos.T_space, "T", {"z", "y", "x"}, time_idx, time_name);
            }
        }
    }

    cfg.prev_output_time = sim.time;
    cfg.output_count += 1;

    nc.close();

    return true;
}
