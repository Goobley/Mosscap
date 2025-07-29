#include "Config.hpp"
#include "Hydro.hpp"
#include <fmt/core.h>
#include "YAKL_netcdf.h"

// static constexpr int problem = 0; // Explosion (10x density gradient in Sod)
static constexpr int problem = 1; // Sod 2d
// cases from liska + wendroff
// static constexpr int problem = 2; // quadrants (case 3)

fp_t set_intial_conditions(State& state) {
    fp_t max_time = FP(0.1);
    if constexpr (problem == 0) {
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
            .ng = 3
        };
        state.Q = Fp4d("Q", N_HYDRO_VARS, state.sz.zc, state.sz.yc, state.sz.xc);
        state.W = Fp4d("W", N_HYDRO_VARS, state.sz.zc, state.sz.yc, state.sz.xc);

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
        max_time = FP(0.2);
    } else if constexpr (problem == 1) {
        state.boundaries.xs = BoundaryType::Wall;
        state.boundaries.xe = BoundaryType::Wall;
        state.boundaries.ys = BoundaryType::Wall;
        state.boundaries.ye = BoundaryType::Wall;

        const int Nx = 1024;
        const int Ny = 1024;
        const int Ng = 3;
        state.dx = FP(1.0) / (Nx - 2 * Ng);
        state.sz = GridSize{
            .xc = Nx,
            .yc = Ny,
            .zc = 1,
            .ng = Ng
        };
        state.Q = Fp4d("Q", N_HYDRO_VARS, state.sz.zc, state.sz.yc, state.sz.xc);
        state.W = Fp4d("W", N_HYDRO_VARS, state.sz.zc, state.sz.yc, state.sz.xc);
        dex_parallel_for(
            FlatLoop<3>(state.sz.zc, state.sz.yc, state.sz.xc),
            KOKKOS_LAMBDA (int k, int j, int i) {
                const fp_t px = i * state.dx - (Ng - 1 + FP(0.5)) * state.dx;
                yakl::SArray<fp_t, 1, N_HYDRO_VARS> w;
                if (px < FP(0.5)) {
                    w(I(Prim::Rho)) = FP(1.0);
                    w(I(Prim::Vx)) = FP(0.0);
                    w(I(Prim::Vy)) = FP(0.0);
                    w(I(Prim::Pres)) = FP(1.0);
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
        max_time = FP(0.2);
    } else if constexpr (problem == 2) {
        state.boundaries.xs = BoundaryType::Symmetric;
        state.boundaries.xe = BoundaryType::Symmetric;
        state.boundaries.ys = BoundaryType::Symmetric;
        state.boundaries.ye = BoundaryType::Symmetric;

        const int Nx = 1024;
        const int Ny = 1024;
        const int Ng = 3;
        state.dx = FP(1.0) / (Nx - 2 * Ng);
        state.sz = GridSize{
            .xc = Nx,
            .yc = Ny,
            .zc = 1,
            .ng = Ng
        };
        state.Q = Fp4d("Q", N_HYDRO_VARS, state.sz.zc, state.sz.yc, state.sz.xc);
        state.W = Fp4d("W", N_HYDRO_VARS, state.sz.zc, state.sz.yc, state.sz.xc);
        dex_parallel_for(
            FlatLoop<3>(state.sz.zc, state.sz.yc, state.sz.xc),
            KOKKOS_LAMBDA (int k, int j, int i) {
                const fp_t px = i * state.dx - (Ng - 1 + FP(0.5)) * state.dx;
                const fp_t py = j * state.dx - (Ng - 1 + FP(0.5)) * state.dx;
                yakl::SArray<fp_t, 1, N_HYDRO_VARS> w;
                if (px < FP(0.5) && py >= FP(0.5)) {
                    w(I(Prim::Pres)) = FP(0.3);
                    w(I(Prim::Rho)) = FP(0.5323);
                    w(I(Prim::Vx)) = FP(1.206);
                    w(I(Prim::Vy)) = FP(0.0);
                } else if (px < FP(0.5) && py < FP(0.5)) {
                    w(I(Prim::Pres)) = FP(0.029);
                    w(I(Prim::Rho)) = FP(0.138);
                    w(I(Prim::Vx)) = FP(1.206);
                    w(I(Prim::Vy)) = FP(1.206);
                } else if (px >= FP(0.5) && py >= FP(0.5)) {
                    w(I(Prim::Pres)) = FP(1.5);
                    w(I(Prim::Rho)) = FP(1.5);
                    w(I(Prim::Vx)) = FP(0.0);
                    w(I(Prim::Vy)) = FP(0.0);
                } else if (px >= FP(0.5) && py < FP(0.5)) {
                    w(I(Prim::Pres)) = FP(0.3);
                    w(I(Prim::Rho)) = FP(0.5323);
                    w(I(Prim::Vx)) = FP(0.0);
                    w(I(Prim::Vy)) = FP(1.206);
                }
                CellIndex idx {
                    .i = i,
                    .j = j,
                    .k = k
                };
                prim_to_cons(w, QtyView(state.Q, idx));
            }
        );
        max_time = FP(0.3);
    }
    Kokkos::fence();
    return max_time;
}

void write_output(const Simulation& sim, int i, fp_t time) {
    global_cons_to_prim(sim);
    yakl::SimpleNetCDF nc;
    std::string name = fmt::format("out_{:06d}.nc", i);
    nc.create(name, yakl::NETCDF_MODE_REPLACE);

    nc.write(sim.state.Q, "Q", {"var", "z", "y", "x"});
    nc.write(sim.state.W, "W", {"var", "z", "y", "x"});
    nc.write(time, "time");
}

int main(int argc, const char** argv) {
    Kokkos::initialize();
    yakl::init();
    {
        State state;
        fp_t max_time = set_intial_conditions(state);

        Simulation sim {
            .current_step = 0,
            .max_cfl = FP(0.99),
            .time = FP(0.0),
            .max_time = max_time,
            .state = state,
            .recon_scratch = ReconScratch {
                .RR = Fp4d("RR", N_HYDRO_VARS, state.sz.zc, state.sz.yc, state.sz.xc),
                .RL = Fp4d("RL", N_HYDRO_VARS, state.sz.zc, state.sz.yc, state.sz.xc)
            },
            .fluxes = Fluxes {
                .Fx = Fp4d("Fx", N_HYDRO_VARS, state.sz.zc, state.sz.yc, state.sz.xc),
                .Fy = Fp4d("Fy", N_HYDRO_VARS, state.sz.zc, state.sz.yc, state.sz.xc),
                .Fz = Fp4d("Fz", N_HYDRO_VARS, state.sz.zc, state.sz.yc, state.sz.xc)
            }
        };
        constexpr TimeStepScheme time_step_scheme = TimeStepScheme::SspRk3;
        TimeStepper<time_step_scheme>::init(sim);

        fill_bcs(sim.state);

        int i = 0;

        while (sim.time < sim.max_time) {
            if (i % 10 == 0) {
                // write_output(sim, i, sim.time);
            }
            const fp_t dt = compute_dt(sim);
            fmt::println("dt: {}", dt);
            TimeStepper<time_step_scheme>::time_step(sim, dt);
            i += 1;
        }
        write_output(sim, i, sim.time);
        fmt::println("{} iterations", i);
    }
    yakl::finalize();
    Kokkos::finalize();

    return 0;
}