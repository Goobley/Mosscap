#include "YAKL.h"
#include "Config.hpp"
#include "Hydro.hpp"
#include <fmt/core.h>
#include "YAKL_netcdf.h"

void set_intial_conditions(const Simulation& sim) {
    const auto& state = sim.state;
    const int N = state.sz.xc;

    dex_parallel_for(
        FlatLoop<3>(state.sz.zc, state.sz.yc, state.sz.xc),
        KOKKOS_LAMBDA (int k, int j, int i) {
            const fp_t px = i * state.dx - FP(0.5) * state.dx;
            const fp_t py = j * state.dx - FP(0.5) * state.dx;
            yakl::SArray<fp_t, 1, N_HYDRO_VARS> w;
            if (std::sqrt(square(px - FP(0.5)) + square(py - FP(0.5))) <  FP(0.25)) {
                w(I(Prim::Rho)) = FP(10.0);
                w(I(Prim::Vx)) = FP(0.0);
                w(I(Prim::Vy)) = FP(0.0);
                w(I(Prim::Pres)) = FP(10.0);
            } else {
                w(I(Prim::Rho)) = FP(0.125);
                w(I(Prim::Vx)) = FP(0.0);
                w(I(Prim::Vy)) = FP(0.0);
                w(I(Prim::Pres)) = FP(0.1);
            }
            CellIndex idx {
                .i = i,
                .j = j,
                .k = k
            };
            prim_to_cons(w, QtyView(state.Q, idx));
        }
    );
    Kokkos::fence();
}

void write_output(const Simulation& sim, int i) {
    yakl::SimpleNetCDF nc;
    std::string name = fmt::format("out_{:04d}.nc", i);
    nc.create(name, yakl::NETCDF_MODE_REPLACE);

    nc.write(sim.state.Q, "Q", {"var", "z", "y", "x"});
    nc.write(sim.state.W, "W", {"var", "z", "y", "x"});
    nc.write(sim.scratch.Fx, "Fx", {"var", "z", "y", "x"});
    nc.write(sim.scratch.Fy, "Fy", {"var", "z", "y", "x"});
    nc.write(sim.scratch.RR, "RR", {"var", "z", "y", "x"});
    nc.write(sim.scratch.RL, "RL", {"var", "z", "y", "x"});
}

int main(int argc, const char** argv) {
    Kokkos::initialize();
    yakl::init();
    {
        State state;
        state.boundaries.xs = BoundaryType::Wall;
        state.boundaries.xe = BoundaryType::Wall;
        state.boundaries.ys = BoundaryType::Wall;
        state.boundaries.ye = BoundaryType::Wall;

        const int N = 256;
        state.dx = FP(1.0) / (N - 2);
        state.sz = GridSize{
            .xc = N,
            .yc = N,
            .zc = 1,
            .ng = 1
        };
        state.Q = Fp4d("Q", N_HYDRO_VARS, state.sz.zc, state.sz.yc, state.sz.xc);
        state.Q_old = Fp4d("Q_old", N_HYDRO_VARS, state.sz.zc, state.sz.yc, state.sz.xc);
        state.W = Fp4d("W", N_HYDRO_VARS, state.sz.zc, state.sz.yc, state.sz.xc);

        Simulation sim {
            .current_step = 0,
            .max_cfl = FP(0.8),
            .state = state,
            .scratch = ScratchSpace {
                .RR = Fp4d("RR", N_HYDRO_VARS, state.sz.zc, state.sz.yc, state.sz.xc),
                .RL = Fp4d("RL", N_HYDRO_VARS, state.sz.zc, state.sz.yc, state.sz.xc),
                .Fx = Fp4d("Fx", N_HYDRO_VARS, state.sz.zc, state.sz.yc, state.sz.xc),
                .Fy = Fp4d("Fy", N_HYDRO_VARS, state.sz.zc, state.sz.yc, state.sz.xc),
                .Fz = Fp4d("Fz", N_HYDRO_VARS, state.sz.zc, state.sz.yc, state.sz.xc)
            }
        };

        set_intial_conditions(sim);
        fill_bcs(sim.state);

        for (int i = 0; i < 10; ++i) {
            write_output(sim, i);
            const fp_t dt = compute_dt(sim);
            fmt::println("dt: {}", dt);
            sim.state.Q_old = sim.state.Q.createDeviceCopy();
            for (int rk = 0; rk < 2; ++rk) {
                calc_hydro_fluxes(sim);
                step_Q(sim, rk, dt);
                fill_bcs(sim.state);
            }
        }
        write_output(sim, 100);
    }
    yakl::finalize();
    Kokkos::finalize();

    return 0;
}