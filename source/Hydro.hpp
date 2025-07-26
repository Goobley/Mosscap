#if !defined(MOSSCAP_HYDRO_HPP)
#define MOSSCAP_HYDRO_HPP

#include "Types.hpp"
#include "Simulation.hpp"
#include "Reconstruct.hpp"
#include "Riemann.hpp"


void calc_hydro_fluxes(const Simulation& sim) {
    constexpr Reconstruction recon = Reconstruction::Ppm;
    constexpr SlopeLimiter slope_limiter = SlopeLimiter::MonotonizedCentral;
    constexpr RiemannSolver rsolver = RiemannSolver::Hllc;

    const auto& state = sim.state;
    const auto& scratch = sim.scratch;
    const auto& sz = sim.state.sz;
    const int nx = sz.xc - 2 * sz.ng;
    const int ny = std::max(sz.yc - 2 * sz.ng, 1);
    const int nz = std::max(sz.zc - 2 * sz.ng, 1);

    dex_parallel_for(
        "Q -> W",
        FlatLoop<3>(sz.zc, sz.yc, sz.xc),
        KOKKOS_LAMBDA (int k, int j, int i) {
            CellIndex idx {
                .i = i,
                .j = j,
                .k = k
            };
            auto WView = QtyView(state.W, idx);
            auto QView = QtyView(state.Q, idx);
            cons_to_prim(QView, WView);
        }
    );
    Kokkos::fence();

    dex_parallel_for(
        "recon x",
        FlatLoop<3>(nz, ny, nx + 2),
        KOKKOS_LAMBDA (int ki, int ji, int ii) {
            CellIndex idx {
                .i = ii + (sz.ng - 1),
                .j = ny == 1 ? ji : ji + sz.ng,
                .k = nz == 1 ? ki : ki + sz.ng
            };
            constexpr int Axis = 0;
            for (int var = 0; var < state.W.extent(0); ++var) {
                reconstruct<recon, slope_limiter, Axis>(state.W, var, idx, scratch.RL(var, idx.k, idx.j, idx.i), scratch.RR(var, idx.k, idx.j, idx.i));
            }
        }
    );
    Kokkos::fence();
    dex_parallel_for(
        "flux x",
        FlatLoop<3>(nz, ny, nx + 1),
        KOKKOS_LAMBDA (int ki, int ji, int ii) {
            CellIndex idx {
                .i = ii + sz.ng,
                .j = ny == 1 ? ji : ji + sz.ng,
                .k = nz == 1 ? ki : ki + sz.ng
            };
            CellIndex idxm {
                .i = ii + sz.ng - 1,
                .j = ny == 1 ? ji : ji + sz.ng,
                .k = nz == 1 ? ki : ki + sz.ng
            };
            // NOTE(cmo): Left and right relative to the interface, taking the reconstructions from the left/right edges of the cells
            QtyView rL_view(scratch.RR, idxm);
            QtyView rR_view(scratch.RL, idx);
            QtyView flux_view(scratch.Fx, idx);
            constexpr int Axis = 0;
            RiemannFlux<rsolver, Axis>(rL_view, rR_view, flux_view);
        }
    );
    Kokkos::fence();

    if constexpr (NUM_DIM > 1) {
        dex_parallel_for(
            "recon y",
            FlatLoop<3>(nz, ny + 2, nx),
            KOKKOS_LAMBDA (int ki, int ji, int ii) {
                CellIndex idx {
                    .i = ii + sz.ng,
                    .j = ji + (sz.ng - 1),
                    .k = nz == 1 ? ki : ki + sz.ng
                };
                constexpr int Axis = 1;
                for (int var = 0; var < state.W.extent(0); ++var) {
                    reconstruct<recon, slope_limiter, Axis>(state.W, var, idx, scratch.RL(var, idx.k, idx.j, idx.i), scratch.RR(var, idx.k, idx.j, idx.i));
                }
            }
        );
        Kokkos::fence();
        dex_parallel_for(
            "flux y",
            FlatLoop<3>(nz, ny + 1, nx),
            KOKKOS_LAMBDA (int ki, int ji, int ii) {
                CellIndex idx {
                    .i = ii + sz.ng,
                    .j = ji + sz.ng,
                    .k = nz == 1 ? ki : ki + sz.ng
                };
                CellIndex idxm {
                    .i = ii + sz.ng,
                    .j = ji + sz.ng - 1,
                    .k = nz == 1 ? ki : ki + sz.ng
                };
                constexpr int Axis = 1;
                // NOTE(cmo): Left and right relative to the interface, taking the reconstructions from the left/right edges of the cells
                QtyView rL_view(scratch.RR, idxm);
                QtyView rR_view(scratch.RL, idx);
                QtyView flux_view(scratch.Fy, idx);
                RiemannFlux<rsolver, Axis>(rL_view, rR_view, flux_view);
            }
        );
        Kokkos::fence();
    }

    if constexpr (NUM_DIM > 2) {
        static_assert(NUM_DIM < 3, "3D Fluxes not done");
    }
}

void step_Q(const Simulation& sim, int rk_step, fp_t dt) {
    const auto& state = sim.state;
    const auto& Q = state.Q;
    const auto& scratch = sim.scratch;
    const auto& sz = sim.state.sz;
    int nx = sz.xc - 2 * sz.ng;
    int ny = std::max(sz.yc - 2 * sz.ng, 1);
    int nz = std::max(sz.zc - 2 * sz.ng, 1);

    if (rk_step == 0) {
        dex_parallel_for(
            "RK2 step 0",
            FlatLoop<3>(nz, ny, nx),
            KOKKOS_LAMBDA (int ki, int ji, int ii) {
                const int k = nz == 1 ? ki : ki + sz.ng;
                const int j = ny == 1 ? ji : ji + sz.ng;
                const int i = ii + sz.ng;
                for (int var = 0; var < state.Q.extent(0); ++var) {
                    fp_t q_update = FP(0.0);
                    q_update += scratch.Fx(var, k, j, i) - scratch.Fx(var, k, j, i+1);
                    if constexpr (NUM_DIM > 1) {
                        q_update += scratch.Fy(var, k, j, i) - scratch.Fy(var, k, j+1, i);
                    }
                    if constexpr (NUM_DIM > 2) {
                        q_update += scratch.Fz(var, k, j, i) - scratch.Fz(var, k+1, j, i);
                    }
                    q_update *= dt / state.dx;
                    Q(var, k, j, i) += q_update;
                }
            }
        );
    } else if (rk_step == 1) {
        dex_parallel_for(
            "RK2 step 1",
            FlatLoop<3>(nz, ny, nx),
            KOKKOS_LAMBDA (int ki, int ji, int ii) {
                const int k = nz == 1 ? ki : ki + sz.ng;
                const int j = ny == 1 ? ji : ji + sz.ng;
                const int i = ii + sz.ng;
                for (int var = 0; var < state.Q.extent(0); ++var) {
                    fp_t q_update = FP(0.0);
                    q_update += scratch.Fx(var, k, j, i) - scratch.Fx(var, k, j, i+1);
                    if constexpr (NUM_DIM > 1) {
                        q_update += scratch.Fy(var, k, j, i) - scratch.Fy(var, k, j+1, i);
                    }
                    if constexpr (NUM_DIM > 2) {
                        q_update += scratch.Fz(var, k, j, i) - scratch.Fz(var, k+1, j, i);
                    }
                    q_update *= dt / state.dx;
                    Q(var, k, j, i) = FP(0.5) * (Q(var, k, j, i) + state.Q_old(var, k, j, i)) + q_update;
                }
            }
        );
    }
    Kokkos::fence();
}

fp_t compute_dt(const Simulation& sim) {
    // NOTE(cmo): This is bad and not 3D, sue me
    auto dt_max_h = Fp1dHost("dt_max", 1);
    dt_max_h(0) = 1e5;
    auto dt_max = dt_max_h.createDeviceCopy();

    const auto& state = sim.state;
    dex_parallel_for(
        "Terrible CFL Loop",
        FlatLoop<3>(state.sz.zc, state.sz.yc, state.sz.xc),
        KOKKOS_LAMBDA (int k, int j, int i) {
            yakl::SArray<fp_t, 1, N_HYDRO_VARS> w;
            CellIndex idx {
                .i = i,
                .j = j,
                .k = k
            };
            cons_to_prim(QtyView(state.Q, idx), w);
            const fp_t cs = std::sqrt(Gamma * w(I(Prim::Pres)) / w(I(Prim::Rho)));
            fp_t vel2 = square(w(I(Prim::Vx)));
            if (NUM_DIM > 1) {
                vel2 += square(w(I(Prim::Vy)));
            }
            const fp_t vel = std::sqrt(vel2);
            const fp_t dt_local = FP(0.5) * state.dx / (cs + vel);
            Kokkos::atomic_min(&dt_max(0), dt_local);
        }
    );

    dt_max_h = dt_max.createHostCopy();
    return dt_max_h(0) * sim.max_cfl;
}

#else
#endif