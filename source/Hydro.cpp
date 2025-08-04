
#include "Hydro.hpp"
#include <map>
#include <tuple>

template <int NumDim>
void global_cons_to_prim_impl(const Simulation& sim) {
    const auto& state = sim.state;
    const auto& eos = sim.eos;
    const auto& sz = sim.state.sz;

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
            cons_to_prim<NumDim>(EosView(eos, idx), QView, WView);
        }
    );
    Kokkos::fence();
}

void global_cons_to_prim(const Simulation& sim) {
    switch (sim.num_dim) {
        case 1: {
            global_cons_to_prim_impl<1>(sim);
        } break;
        case 2: {
            global_cons_to_prim_impl<2>(sim);
        } break;
        case 3: {
            global_cons_to_prim_impl<3>(sim);
        } break;
        default:
            KOKKOS_ASSERT(false && ("Weird num dim"));
    }
}

template<Reconstruction reconstruction, SlopeLimiter slope_limiter, int Axis>
void compute_recon_impl(const Simulation& sim) {
    static_assert(Axis < 3, "What are you doing?");
    const auto& state = sim.state;
    const auto& recon = sim.recon_scratch;
    const auto& eos = sim.eos;
    const auto& sz = sim.state.sz;
    int nx = sz.xc - 2 * sz.ng;
    int ny = std::max(sz.yc - 2 * sz.ng, 1);
    int nz = std::max(sz.zc - 2 * sz.ng, 1);
    int xs = sz.ng;
    int ys = ny == 1 ? 0 : sz.ng;
    int zs = nz == 1 ? 0 : sz.ng;

    if constexpr (Axis == 0) {
        nx += 2;
        xs -= 1;
    } else if constexpr (Axis == 1) {
        ny += 2;
        ys -= 1;
    } else {
        nz += 2;
        zs -= 1;
    }

    static constexpr const char* kernel_name[3] = {"Reconstruct x", "Reconstruct y", "Reconstruct z"};
    dex_parallel_for(
        kernel_name[Axis],
        FlatLoop<3>(nz, ny, nx),
        KOKKOS_LAMBDA (int ki, int ji, int ii) {
            CellIndex idx {
                .i = ii + xs,
                .j = ji + ys,
                .k = ki + zs
            };
            for (int var = 0; var < state.W.extent(0); ++var) {
                reconstruct<reconstruction, slope_limiter, Axis>(
                    state.W,
                    var,
                    idx,
                    recon.RL(var, idx.k, idx.j, idx.i),
                    recon.RR(var, idx.k, idx.j, idx.i)
                );
            }
            if (!eos.is_constant) {
                Fp4d GammaE(
                    "GammaE 4D",
                    eos.gamma_e_space.data(),
                    1,
                    eos.gamma_e_space.extent(0),
                    eos.gamma_e_space.extent(1),
                    eos.gamma_e_space.extent(2)
                );
                reconstruct<reconstruction, slope_limiter, Axis>(
                    GammaE,
                    0,
                    idx,
                    eos.gamma_e_space_L(idx.k, idx.j, idx.i),
                    eos.gamma_e_space_R(idx.k, idx.j, idx.i)
                );
            }
        }
    );
    Kokkos::fence();
}

template <RiemannSolver rsolver, int Axis, int NumDim>
void compute_flux_impl(const Simulation& sim) {
    static_assert(Axis < NumDim, "What are you doing?");
    const auto& recon = sim.recon_scratch;
    const auto& fluxes = sim.fluxes;
    const auto& sz = sim.state.sz;
    const auto& eos = sim.eos;
    int nx = sz.xc - 2 * sz.ng;
    int ny = std::max(sz.yc - 2 * sz.ng, 1);
    int nz = std::max(sz.zc - 2 * sz.ng, 1);
    int xs = sz.ng;
    int ys = ny == 1 ? 0 : sz.ng;
    int zs = nz == 1 ? 0 : sz.ng;

    Fp4d flux;
    if constexpr (Axis == 0) {
        nx += 1;
        flux = fluxes.Fx;
    } else if constexpr (Axis == 1) {
        ny += 1;
        flux = fluxes.Fy;
    } else {
        nz += 1;
        flux = fluxes.Fz;
    }
    static constexpr const char* kernel_name[3] = {"Flux x", "Flux y", "Flux z"};
    dex_parallel_for(
        kernel_name[Axis],
        FlatLoop<3>(nz, ny, nx),
        KOKKOS_LAMBDA (int ki, int ji, int ii) {
            CellIndex idx {
                .i = ii + xs,
                .j = ji + ys,
                .k = ki + zs
            };
            CellIndex idxm(idx);
            idxm.along<Axis>() -= 1;
            // NOTE(cmo): Left and right relative to the interface, taking the reconstructions from the left/right edges of the cells
            QtyView rL_view(recon.RR, idxm);
            QtyView rR_view(recon.RL, idx);
            QtyView flux_view(flux, idx);
            EosView eos_L(eos, idxm, ReconstructionEdge::Right);
            EosView eos_R(eos, idx, ReconstructionEdge::Left);
            // EosView eos_L(eos, idxm, ReconstructionEdge::Centred);
            // EosView eos_R(eos, idx, ReconstructionEdge::Centred);
            riemann_flux<rsolver, Axis, NumDim>(
                ReconstructedEos{
                    .L = eos_L,
                    .R = eos_R
                },
                rL_view,
                rR_view,
                flux_view
            );
        }
    );
    Kokkos::fence();
}

template <int Axis, Reconstruction recon, SlopeLimiter sl = SlopeLimiter::MonotonizedCentral>
std::pair<
    std::tuple<int, ReconstructionScheme>,
    std::function<void(const Simulation&)>
>
make_recon_impl() {
    return {
        std::make_tuple(Axis, ReconstructionScheme{.reconstruction=recon, .slope_limiter=sl}),
        compute_recon_impl<recon, sl, Axis>
    };
}

template <RiemannSolver rs, int Axis, int NumDim>
std::pair<
    std::tuple<RiemannSolver, int, int>,
    std::function<void(const Simulation&)>
>
make_flux_impl() {
    return {
        std::make_tuple(rs, Axis, NumDim),
        compute_flux_impl<rs, Axis, NumDim>
    };
}

template <int NumDim, typename WType>
KOKKOS_INLINE_FUNCTION void dt_reducer(const EosView& eos, const WType& w, const fp_t dx, fp_t& running_dt) {
    using Prim = Prim<NumDim>;
    const fp_t cs = sound_speed<NumDim>(eos, w);
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

template <int NumDim>
f64 compute_dt_impl(const Simulation& sim) {
    const auto& state = sim.state;
    const auto& eos = sim.eos;

    fp_t dt_max = FP(1e5);
    dex_parallel_reduce(
        "CFL reduction",
        FlatLoop<3>(state.sz.zc, state.sz.yc, state.sz.xc),
        KOKKOS_LAMBDA (int k, int j, int i, fp_t& running_dt) {
            yakl::SArray<fp_t, 1, N_HYDRO_VARS<NumDim>> w;
            CellIndex idx {
                .i = i,
                .j = j,
                .k = k
            };
            EosView eosv(eos, idx);
            cons_to_prim<NumDim>(eosv, QtyView(state.Q, idx), w);
            dt_reducer<NumDim>(eosv, w, state.dx, running_dt);
        },
        Kokkos::Min<fp_t>(dt_max)
    );
    Kokkos::fence();

    f64 dt = dt_max * sim.max_cfl;
    const auto& out = sim.out_cfg;
    f64 next_write_time = std::min(sim.max_time, out.prev_output_time + out.delta);
    if (sim.time + dt >= next_write_time) {
        dt = next_write_time - sim.time;
        while (sim.time + dt < next_write_time) {
            // NOTE(cmo): Ensure time + dt carries us until at least next_write_time
            dt = std::nextafter(dt, std::numeric_limits<f64>::max());
        }
    }
    return dt;
}

void select_hydro_fns(Simulation& sim) {
    const auto& schemes = sim.scheme;
    // NOTE(cmo): yes map is bad, but it doesn't matter
    std::map<
        std::tuple<int, ReconstructionScheme>,
        std::function<void(const Simulation&)>
    > recon_fns = {
        make_recon_impl<0, Reconstruction::Fog>(),
        make_recon_impl<0, Reconstruction::Muscl, SlopeLimiter::VanLeer>(),
        make_recon_impl<0, Reconstruction::Muscl, SlopeLimiter::MonotonizedCentral>(),
        make_recon_impl<0, Reconstruction::Muscl, SlopeLimiter::Minmod>(),
        make_recon_impl<0, Reconstruction::Ppm, SlopeLimiter::VanLeer>(),
        make_recon_impl<0, Reconstruction::Ppm, SlopeLimiter::MonotonizedCentral>(),
        make_recon_impl<0, Reconstruction::Ppm, SlopeLimiter::Minmod>(),
        make_recon_impl<0, Reconstruction::Weno5Z>(),
        make_recon_impl<1, Reconstruction::Fog>(),
        make_recon_impl<1, Reconstruction::Muscl, SlopeLimiter::VanLeer>(),
        make_recon_impl<1, Reconstruction::Muscl, SlopeLimiter::MonotonizedCentral>(),
        make_recon_impl<1, Reconstruction::Muscl, SlopeLimiter::Minmod>(),
        make_recon_impl<1, Reconstruction::Ppm, SlopeLimiter::VanLeer>(),
        make_recon_impl<1, Reconstruction::Ppm, SlopeLimiter::MonotonizedCentral>(),
        make_recon_impl<1, Reconstruction::Ppm, SlopeLimiter::Minmod>(),
        make_recon_impl<1, Reconstruction::Weno5Z>(),
        make_recon_impl<2, Reconstruction::Fog>(),
        make_recon_impl<2, Reconstruction::Muscl, SlopeLimiter::VanLeer>(),
        make_recon_impl<2, Reconstruction::Muscl, SlopeLimiter::MonotonizedCentral>(),
        make_recon_impl<2, Reconstruction::Muscl, SlopeLimiter::Minmod>(),
        make_recon_impl<2, Reconstruction::Ppm, SlopeLimiter::VanLeer>(),
        make_recon_impl<2, Reconstruction::Ppm, SlopeLimiter::MonotonizedCentral>(),
        make_recon_impl<2, Reconstruction::Ppm, SlopeLimiter::Minmod>(),
        make_recon_impl<2, Reconstruction::Weno5Z>(),
    };
    ReconstructionScheme recon{.reconstruction=schemes.reconstruction, .slope_limiter=schemes.slope_limit};
    sim.flux_fns.recon_x = recon_fns[std::make_tuple(0, recon)];
    if (sim.num_dim > 1) {
        sim.flux_fns.recon_y = recon_fns[std::make_tuple(1, recon)];
    }
    if (sim.num_dim > 2) {
        sim.flux_fns.recon_z = recon_fns[std::make_tuple(2, recon)];
    }

    std::map<
        std::tuple<RiemannSolver, int, int>,
        std::function<void(const Simulation&)>
    > flux_fns = {
        //             solver                axis num_dim
        make_flux_impl<RiemannSolver::Rusanov, 0, 1>(),
        make_flux_impl<RiemannSolver::Hll,     0, 1>(),
        make_flux_impl<RiemannSolver::Hllc,    0, 1>(),
        make_flux_impl<RiemannSolver::Rusanov, 0, 2>(),
        make_flux_impl<RiemannSolver::Hll,     0, 2>(),
        make_flux_impl<RiemannSolver::Hllc,    0, 2>(),
        make_flux_impl<RiemannSolver::Rusanov, 0, 3>(),
        make_flux_impl<RiemannSolver::Hll,     0, 3>(),
        make_flux_impl<RiemannSolver::Hllc,    0, 3>(),
        make_flux_impl<RiemannSolver::Rusanov, 1, 2>(),
        make_flux_impl<RiemannSolver::Hll,     1, 2>(),
        make_flux_impl<RiemannSolver::Hllc,    1, 2>(),
        make_flux_impl<RiemannSolver::Rusanov, 1, 3>(),
        make_flux_impl<RiemannSolver::Hll,     1, 3>(),
        make_flux_impl<RiemannSolver::Hllc,    1, 3>(),
        make_flux_impl<RiemannSolver::Rusanov, 2, 3>(),
        make_flux_impl<RiemannSolver::Hll,     2, 3>(),
        make_flux_impl<RiemannSolver::Hllc,    2, 3>()
    };
    sim.flux_fns.flux_x = flux_fns[std::make_tuple(schemes.riemann_solver, 0, sim.num_dim)];
    if (sim.num_dim > 1) {
        sim.flux_fns.flux_y = flux_fns[std::make_tuple(schemes.riemann_solver, 1, sim.num_dim)];
    }
    if (sim.num_dim > 2) {
        sim.flux_fns.flux_z = flux_fns[std::make_tuple(schemes.riemann_solver, 2, sim.num_dim)];
    }

    std::vector<std::function<fp_t(const Simulation&)>> cfl_fns = {
        compute_dt_impl<1>,
        compute_dt_impl<2>,
        compute_dt_impl<3>
    };
    sim.compute_dt = cfl_fns.at(sim.num_dim - 1);
}

void compute_hydro_fluxes(const Simulation& sim) {
    if (sim.update_eos) {
        sim.update_eos(sim);
    }
    global_cons_to_prim(sim);
    sim.flux_fns.recon_x(sim);
    sim.flux_fns.flux_x(sim);

    if (sim.num_dim > 1) {
        sim.flux_fns.recon_y(sim);
        sim.flux_fns.flux_y(sim);
    }

    if (sim.num_dim > 2) {
        sim.flux_fns.recon_z(sim);
        sim.flux_fns.flux_z(sim);
    }
}

f64 compute_dt(const Simulation& sim) {
    return sim.compute_dt(sim);
}