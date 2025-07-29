
#include "Hydro.hpp"
#include <map>
#include <tuple>

void global_cons_to_prim(const Simulation& sim) {
    const auto& state = sim.state;
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
}

template <int NumDim, Reconstruction reconstruction, SlopeLimiter slope_limiter, RiemannSolver rsolver>
void compute_hydro_fluxes_impl(const Simulation& sim) {
    const auto& state = sim.state;
    const auto& recon = sim.recon_scratch;
    const auto& flux = sim.fluxes;
    const auto& sz = sim.state.sz;
    const int nx = sz.xc - 2 * sz.ng;
    const int ny = std::max(sz.yc - 2 * sz.ng, 1);
    const int nz = std::max(sz.zc - 2 * sz.ng, 1);

    global_cons_to_prim(sim);

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
                reconstruct<reconstruction, slope_limiter, Axis>(state.W, var, idx, recon.RL(var, idx.k, idx.j, idx.i), recon.RR(var, idx.k, idx.j, idx.i));
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
            QtyView rL_view(recon.RR, idxm);
            QtyView rR_view(recon.RL, idx);
            QtyView flux_view(flux.Fx, idx);
            constexpr int Axis = 0;
            riemann_flux<rsolver, Axis>(rL_view, rR_view, flux_view);
        }
    );
    Kokkos::fence();

    if constexpr (NumDim > 1) {
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
                    reconstruct<reconstruction, slope_limiter, Axis>(state.W, var, idx, recon.RL(var, idx.k, idx.j, idx.i), recon.RR(var, idx.k, idx.j, idx.i));
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
                QtyView rL_view(recon.RR, idxm);
                QtyView rR_view(recon.RL, idx);
                QtyView flux_view(flux.Fy, idx);
                riemann_flux<rsolver, Axis>(rL_view, rR_view, flux_view);
            }
        );
        Kokkos::fence();
    }

    if constexpr (NumDim > 2) {
        dex_parallel_for(
            "recon z",
            FlatLoop<3>(nz + 2, ny, nx),
            KOKKOS_LAMBDA (int ki, int ji, int ii) {
                CellIndex idx {
                    .i = ii + sz.ng,
                    .j = ji + sz.ng,
                    .k = ki + (sz.ng - 1)
                };
                constexpr int Axis = 2;
                for (int var = 0; var < state.W.extent(0); ++var) {
                    reconstruct<reconstruction, slope_limiter, Axis>(state.W, var, idx, recon.RL(var, idx.k, idx.j, idx.i), recon.RR(var, idx.k, idx.j, idx.i));
                }
            }
        );
        Kokkos::fence();
        dex_parallel_for(
            "flux z",
            FlatLoop<3>(nz + 1, ny, nx),
            KOKKOS_LAMBDA (int ki, int ji, int ii) {
                CellIndex idx {
                    .i = ii + sz.ng,
                    .j = ji + sz.ng,
                    .k = ki + sz.ng
                };
                CellIndex idxm {
                    .i = ii + sz.ng,
                    .j = ji + sz.ng,
                    .k = ki + sz.ng - 1
                };
                constexpr int Axis = 2;
                // NOTE(cmo): Left and right relative to the interface, taking the reconstructions from the left/right edges of the cells
                QtyView rL_view(recon.RR, idxm);
                QtyView rR_view(recon.RL, idx);
                QtyView flux_view(flux.Fy, idx);
                riemann_flux<rsolver, Axis>(rL_view, rR_view, flux_view);
            }
        );
        Kokkos::fence();
    }
}


template <int NumDim, Reconstruction recon, SlopeLimiter sl, RiemannSolver rs>
std::pair<
    std::tuple<int, Reconstruction, SlopeLimiter, RiemannSolver>,
    std::function<void(const Simulation&)>
>
make_flux_impl() {
    return {
        std::make_tuple(NumDim, recon, sl, rs),
        compute_hydro_fluxes_impl<NumDim, recon, sl, rs>
    };
}


void select_hydro_fns(const NumericalSchemes& schemes, Simulation& sim) {
    // NOTE(cmo): yes map is bad, but it doesn't matter
    std::map<
        std::tuple<int, Reconstruction, SlopeLimiter, RiemannSolver>,
        std::function<void(const Simulation&)>
    > available = {
        make_flux_impl<1, Reconstruction::Fog, SlopeLimiter::VanLeer, RiemannSolver::Rusanov>(),
        make_flux_impl<1, Reconstruction::Fog, SlopeLimiter::VanLeer, RiemannSolver::Hll>(),
        make_flux_impl<1, Reconstruction::Fog, SlopeLimiter::VanLeer, RiemannSolver::Hllc>(),
        make_flux_impl<1, Reconstruction::Fog, SlopeLimiter::MonotonizedCentral, RiemannSolver::Rusanov>(),
        make_flux_impl<1, Reconstruction::Fog, SlopeLimiter::MonotonizedCentral, RiemannSolver::Hll>(),
        make_flux_impl<1, Reconstruction::Fog, SlopeLimiter::MonotonizedCentral, RiemannSolver::Hllc>(),
        make_flux_impl<1, Reconstruction::Fog, SlopeLimiter::Minmod, RiemannSolver::Rusanov>(),
        make_flux_impl<1, Reconstruction::Fog, SlopeLimiter::Minmod, RiemannSolver::Hll>(),
        make_flux_impl<1, Reconstruction::Fog, SlopeLimiter::Minmod, RiemannSolver::Hllc>(),
        make_flux_impl<1, Reconstruction::Muscl, SlopeLimiter::VanLeer, RiemannSolver::Rusanov>(),
        make_flux_impl<1, Reconstruction::Muscl, SlopeLimiter::VanLeer, RiemannSolver::Hll>(),
        make_flux_impl<1, Reconstruction::Muscl, SlopeLimiter::VanLeer, RiemannSolver::Hllc>(),
        make_flux_impl<1, Reconstruction::Muscl, SlopeLimiter::MonotonizedCentral, RiemannSolver::Rusanov>(),
        make_flux_impl<1, Reconstruction::Muscl, SlopeLimiter::MonotonizedCentral, RiemannSolver::Hll>(),
        make_flux_impl<1, Reconstruction::Muscl, SlopeLimiter::MonotonizedCentral, RiemannSolver::Hllc>(),
        make_flux_impl<1, Reconstruction::Muscl, SlopeLimiter::Minmod, RiemannSolver::Rusanov>(),
        make_flux_impl<1, Reconstruction::Muscl, SlopeLimiter::Minmod, RiemannSolver::Hll>(),
        make_flux_impl<1, Reconstruction::Muscl, SlopeLimiter::Minmod, RiemannSolver::Hllc>(),
        make_flux_impl<1, Reconstruction::Ppm, SlopeLimiter::VanLeer, RiemannSolver::Rusanov>(),
        make_flux_impl<1, Reconstruction::Ppm, SlopeLimiter::VanLeer, RiemannSolver::Hll>(),
        make_flux_impl<1, Reconstruction::Ppm, SlopeLimiter::VanLeer, RiemannSolver::Hllc>(),
        make_flux_impl<1, Reconstruction::Ppm, SlopeLimiter::MonotonizedCentral, RiemannSolver::Rusanov>(),
        make_flux_impl<1, Reconstruction::Ppm, SlopeLimiter::MonotonizedCentral, RiemannSolver::Hll>(),
        make_flux_impl<1, Reconstruction::Ppm, SlopeLimiter::MonotonizedCentral, RiemannSolver::Hllc>(),
        make_flux_impl<1, Reconstruction::Ppm, SlopeLimiter::Minmod, RiemannSolver::Rusanov>(),
        make_flux_impl<1, Reconstruction::Ppm, SlopeLimiter::Minmod, RiemannSolver::Hll>(),
        make_flux_impl<1, Reconstruction::Ppm, SlopeLimiter::Minmod, RiemannSolver::Hllc>(),
        make_flux_impl<1, Reconstruction::Weno5Z, SlopeLimiter::VanLeer, RiemannSolver::Rusanov>(),
        make_flux_impl<1, Reconstruction::Weno5Z, SlopeLimiter::VanLeer, RiemannSolver::Hll>(),
        make_flux_impl<1, Reconstruction::Weno5Z, SlopeLimiter::VanLeer, RiemannSolver::Hllc>(),
        make_flux_impl<1, Reconstruction::Weno5Z, SlopeLimiter::MonotonizedCentral, RiemannSolver::Rusanov>(),
        make_flux_impl<1, Reconstruction::Weno5Z, SlopeLimiter::MonotonizedCentral, RiemannSolver::Hll>(),
        make_flux_impl<1, Reconstruction::Weno5Z, SlopeLimiter::MonotonizedCentral, RiemannSolver::Hllc>(),
        make_flux_impl<1, Reconstruction::Weno5Z, SlopeLimiter::Minmod, RiemannSolver::Rusanov>(),
        make_flux_impl<1, Reconstruction::Weno5Z, SlopeLimiter::Minmod, RiemannSolver::Hll>(),
        make_flux_impl<1, Reconstruction::Weno5Z, SlopeLimiter::Minmod, RiemannSolver::Hllc>(),
        make_flux_impl<2, Reconstruction::Fog, SlopeLimiter::VanLeer, RiemannSolver::Rusanov>(),
        make_flux_impl<2, Reconstruction::Fog, SlopeLimiter::VanLeer, RiemannSolver::Hll>(),
        make_flux_impl<2, Reconstruction::Fog, SlopeLimiter::VanLeer, RiemannSolver::Hllc>(),
        make_flux_impl<2, Reconstruction::Fog, SlopeLimiter::MonotonizedCentral, RiemannSolver::Rusanov>(),
        make_flux_impl<2, Reconstruction::Fog, SlopeLimiter::MonotonizedCentral, RiemannSolver::Hll>(),
        make_flux_impl<2, Reconstruction::Fog, SlopeLimiter::MonotonizedCentral, RiemannSolver::Hllc>(),
        make_flux_impl<2, Reconstruction::Fog, SlopeLimiter::Minmod, RiemannSolver::Rusanov>(),
        make_flux_impl<2, Reconstruction::Fog, SlopeLimiter::Minmod, RiemannSolver::Hll>(),
        make_flux_impl<2, Reconstruction::Fog, SlopeLimiter::Minmod, RiemannSolver::Hllc>(),
        make_flux_impl<2, Reconstruction::Muscl, SlopeLimiter::VanLeer, RiemannSolver::Rusanov>(),
        make_flux_impl<2, Reconstruction::Muscl, SlopeLimiter::VanLeer, RiemannSolver::Hll>(),
        make_flux_impl<2, Reconstruction::Muscl, SlopeLimiter::VanLeer, RiemannSolver::Hllc>(),
        make_flux_impl<2, Reconstruction::Muscl, SlopeLimiter::MonotonizedCentral, RiemannSolver::Rusanov>(),
        make_flux_impl<2, Reconstruction::Muscl, SlopeLimiter::MonotonizedCentral, RiemannSolver::Hll>(),
        make_flux_impl<2, Reconstruction::Muscl, SlopeLimiter::MonotonizedCentral, RiemannSolver::Hllc>(),
        make_flux_impl<2, Reconstruction::Muscl, SlopeLimiter::Minmod, RiemannSolver::Rusanov>(),
        make_flux_impl<2, Reconstruction::Muscl, SlopeLimiter::Minmod, RiemannSolver::Hll>(),
        make_flux_impl<2, Reconstruction::Muscl, SlopeLimiter::Minmod, RiemannSolver::Hllc>(),
        make_flux_impl<2, Reconstruction::Ppm, SlopeLimiter::VanLeer, RiemannSolver::Rusanov>(),
        make_flux_impl<2, Reconstruction::Ppm, SlopeLimiter::VanLeer, RiemannSolver::Hll>(),
        make_flux_impl<2, Reconstruction::Ppm, SlopeLimiter::VanLeer, RiemannSolver::Hllc>(),
        make_flux_impl<2, Reconstruction::Ppm, SlopeLimiter::MonotonizedCentral, RiemannSolver::Rusanov>(),
        make_flux_impl<2, Reconstruction::Ppm, SlopeLimiter::MonotonizedCentral, RiemannSolver::Hll>(),
        make_flux_impl<2, Reconstruction::Ppm, SlopeLimiter::MonotonizedCentral, RiemannSolver::Hllc>(),
        make_flux_impl<2, Reconstruction::Ppm, SlopeLimiter::Minmod, RiemannSolver::Rusanov>(),
        make_flux_impl<2, Reconstruction::Ppm, SlopeLimiter::Minmod, RiemannSolver::Hll>(),
        make_flux_impl<2, Reconstruction::Ppm, SlopeLimiter::Minmod, RiemannSolver::Hllc>(),
        make_flux_impl<2, Reconstruction::Weno5Z, SlopeLimiter::VanLeer, RiemannSolver::Rusanov>(),
        make_flux_impl<2, Reconstruction::Weno5Z, SlopeLimiter::VanLeer, RiemannSolver::Hll>(),
        make_flux_impl<2, Reconstruction::Weno5Z, SlopeLimiter::VanLeer, RiemannSolver::Hllc>(),
        make_flux_impl<2, Reconstruction::Weno5Z, SlopeLimiter::MonotonizedCentral, RiemannSolver::Rusanov>(),
        make_flux_impl<2, Reconstruction::Weno5Z, SlopeLimiter::MonotonizedCentral, RiemannSolver::Hll>(),
        make_flux_impl<2, Reconstruction::Weno5Z, SlopeLimiter::MonotonizedCentral, RiemannSolver::Hllc>(),
        make_flux_impl<2, Reconstruction::Weno5Z, SlopeLimiter::Minmod, RiemannSolver::Rusanov>(),
        make_flux_impl<2, Reconstruction::Weno5Z, SlopeLimiter::Minmod, RiemannSolver::Hll>(),
        make_flux_impl<2, Reconstruction::Weno5Z, SlopeLimiter::Minmod, RiemannSolver::Hllc>(),
        make_flux_impl<3, Reconstruction::Fog, SlopeLimiter::VanLeer, RiemannSolver::Rusanov>(),
        make_flux_impl<3, Reconstruction::Fog, SlopeLimiter::VanLeer, RiemannSolver::Hll>(),
        make_flux_impl<3, Reconstruction::Fog, SlopeLimiter::VanLeer, RiemannSolver::Hllc>(),
        make_flux_impl<3, Reconstruction::Fog, SlopeLimiter::MonotonizedCentral, RiemannSolver::Rusanov>(),
        make_flux_impl<3, Reconstruction::Fog, SlopeLimiter::MonotonizedCentral, RiemannSolver::Hll>(),
        make_flux_impl<3, Reconstruction::Fog, SlopeLimiter::MonotonizedCentral, RiemannSolver::Hllc>(),
        make_flux_impl<3, Reconstruction::Fog, SlopeLimiter::Minmod, RiemannSolver::Rusanov>(),
        make_flux_impl<3, Reconstruction::Fog, SlopeLimiter::Minmod, RiemannSolver::Hll>(),
        make_flux_impl<3, Reconstruction::Fog, SlopeLimiter::Minmod, RiemannSolver::Hllc>(),
        make_flux_impl<3, Reconstruction::Muscl, SlopeLimiter::VanLeer, RiemannSolver::Rusanov>(),
        make_flux_impl<3, Reconstruction::Muscl, SlopeLimiter::VanLeer, RiemannSolver::Hll>(),
        make_flux_impl<3, Reconstruction::Muscl, SlopeLimiter::VanLeer, RiemannSolver::Hllc>(),
        make_flux_impl<3, Reconstruction::Muscl, SlopeLimiter::MonotonizedCentral, RiemannSolver::Rusanov>(),
        make_flux_impl<3, Reconstruction::Muscl, SlopeLimiter::MonotonizedCentral, RiemannSolver::Hll>(),
        make_flux_impl<3, Reconstruction::Muscl, SlopeLimiter::MonotonizedCentral, RiemannSolver::Hllc>(),
        make_flux_impl<3, Reconstruction::Muscl, SlopeLimiter::Minmod, RiemannSolver::Rusanov>(),
        make_flux_impl<3, Reconstruction::Muscl, SlopeLimiter::Minmod, RiemannSolver::Hll>(),
        make_flux_impl<3, Reconstruction::Muscl, SlopeLimiter::Minmod, RiemannSolver::Hllc>(),
        make_flux_impl<3, Reconstruction::Ppm, SlopeLimiter::VanLeer, RiemannSolver::Rusanov>(),
        make_flux_impl<3, Reconstruction::Ppm, SlopeLimiter::VanLeer, RiemannSolver::Hll>(),
        make_flux_impl<3, Reconstruction::Ppm, SlopeLimiter::VanLeer, RiemannSolver::Hllc>(),
        make_flux_impl<3, Reconstruction::Ppm, SlopeLimiter::MonotonizedCentral, RiemannSolver::Rusanov>(),
        make_flux_impl<3, Reconstruction::Ppm, SlopeLimiter::MonotonizedCentral, RiemannSolver::Hll>(),
        make_flux_impl<3, Reconstruction::Ppm, SlopeLimiter::MonotonizedCentral, RiemannSolver::Hllc>(),
        make_flux_impl<3, Reconstruction::Ppm, SlopeLimiter::Minmod, RiemannSolver::Rusanov>(),
        make_flux_impl<3, Reconstruction::Ppm, SlopeLimiter::Minmod, RiemannSolver::Hll>(),
        make_flux_impl<3, Reconstruction::Ppm, SlopeLimiter::Minmod, RiemannSolver::Hllc>(),
        make_flux_impl<3, Reconstruction::Weno5Z, SlopeLimiter::VanLeer, RiemannSolver::Rusanov>(),
        make_flux_impl<3, Reconstruction::Weno5Z, SlopeLimiter::VanLeer, RiemannSolver::Hll>(),
        make_flux_impl<3, Reconstruction::Weno5Z, SlopeLimiter::VanLeer, RiemannSolver::Hllc>(),
        make_flux_impl<3, Reconstruction::Weno5Z, SlopeLimiter::MonotonizedCentral, RiemannSolver::Rusanov>(),
        make_flux_impl<3, Reconstruction::Weno5Z, SlopeLimiter::MonotonizedCentral, RiemannSolver::Hll>(),
        make_flux_impl<3, Reconstruction::Weno5Z, SlopeLimiter::MonotonizedCentral, RiemannSolver::Hllc>(),
        make_flux_impl<3, Reconstruction::Weno5Z, SlopeLimiter::Minmod, RiemannSolver::Rusanov>(),
        make_flux_impl<3, Reconstruction::Weno5Z, SlopeLimiter::Minmod, RiemannSolver::Hll>(),
        make_flux_impl<3, Reconstruction::Weno5Z, SlopeLimiter::Minmod, RiemannSolver::Hllc>()
    };

    sim.compute_hydro_fluxes = available[
        std::make_tuple(NUM_DIM, schemes.reconstruction, schemes.slope_limit, schemes.riemann_solver)
    ];
}

void compute_hydro_fluxes(const Simulation& sim) {
    sim.compute_hydro_fluxes(sim);
}

template <int NumDim, typename WType>
KOKKOS_INLINE_FUNCTION void dt_reducer(const WType& w, const fp_t dx, fp_t& running_dt) {
    const fp_t cs = std::sqrt(Gamma * w(I(Prim::Pres)) / w(I(Prim::Rho)));
    fp_t vel2 = square(w(I(Prim::Vx)));
    if (NumDim > 1) {
        vel2 += square(w(I(Prim::Vy)));
    }
    if (NumDim > 2) {
        vel2 += square(w(I(Prim::Vz)));
    }
    const fp_t vel = std::sqrt(vel2);
    const fp_t dt_local = dx / (cs + vel);
    running_dt = std::min(dt_local, running_dt);
}

fp_t compute_dt(const Simulation& sim) {
    const auto& state = sim.state;

    fp_t dt_max = FP(1e5);
    dex_parallel_reduce(
        "CFL reduction",
        FlatLoop<3>(state.sz.zc, state.sz.yc, state.sz.xc),
        KOKKOS_LAMBDA (int k, int j, int i, fp_t& running_dt) {
            yakl::SArray<fp_t, 1, N_HYDRO_VARS> w;
            CellIndex idx {
                .i = i,
                .j = j,
                .k = k
            };
            cons_to_prim(QtyView(state.Q, idx), w);
            dt_reducer<NUM_DIM>(w, state.dx, running_dt);
        },
        Kokkos::Min<fp_t>(dt_max)
    );

    fp_t dt = dt_max * sim.max_cfl;
    if (sim.time + dt >= sim.max_time) {
        dt = sim.max_time - sim.time + FP(1e-8);
    }
    return dt;
}
