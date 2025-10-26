
#include "Hydro.hpp"
#include <map>
#include <tuple>

namespace Mosscap {

void update_glm_ch(Simulation& sim) {
    // Max wave propagation speed for divB cleaning
    sim.state.glm_ch = sim.max_cfl * sim.state.dx / sim.dt;
    fmt::println("{} * {} / {} = {}", sim.max_cfl, sim.state.dx, sim.dt, sim.state.glm_ch);
}

template <typename FTraits>
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
            cons_to_prim<FTraits>(eos.gamma, state.mu0, QView, WView);

            // NOTE(cmo): For each tracer field, scale to mass density
            using Cons = FTraits::cons;
            constexpr i32 n_hydro = FTraits::num_vars;
            const fp_t inv_rho = 1.0_fp / QView(I(Cons::Rho));
            for (int i = n_hydro; i < n_hydro + state.num_tracers; ++i) {
                WView(i) = QView(i) * inv_rho;
            }
        }
    );
    Kokkos::fence();
}

template <int NumDim>
void global_cons_to_prim_fluid_dispatch(const Simulation& sim) {
    switch (sim.fluid_type) {
        case FluidType::Hydro: {
            global_cons_to_prim_impl<FluidTraits<NumDim, FluidType::Hydro>>(sim);
        } break;
        case FluidType::Mhd: {
            global_cons_to_prim_impl<FluidTraits<NumDim, FluidType::Mhd>>(sim);
        } break;
        case FluidType::GlmMhd: {
            global_cons_to_prim_impl<FluidTraits<NumDim, FluidType::GlmMhd>>(sim);
        } break;
        default: {
            KOKKOS_ASSERT(false && "Unknown fluid type");
        }
    }
}

void global_cons_to_prim(const Simulation& sim) {
    switch (sim.num_dim) {
        case 1: {
            global_cons_to_prim_fluid_dispatch<1>(sim);
        } break;
        case 2: {
            global_cons_to_prim_fluid_dispatch<2>(sim);
        } break;
        case 3: {
            global_cons_to_prim_fluid_dispatch<3>(sim);
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
        }
    );
    Kokkos::fence();
}

template <RiemannSolver rsolver, int Axis, typename FTraits>
void compute_flux_impl(const Simulation& sim) {
    constexpr int NumDim = FTraits::num_dim;
    static_assert(Axis < NumDim, "What are you doing?");
    const auto& recon = sim.recon_scratch;
    const auto& fluxes = sim.fluxes;
    const auto& sz = sim.state.sz;
    const auto& eos = sim.eos;
    const fp_t glm_ch = sim.state.glm_ch;
    const fp_t mu0 = sim.state.mu0;
    int n_tracer = sim.state.num_tracers;
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
            riemann_flux<rsolver, Axis, FTraits>(
                eos,
                mu0,
                glm_ch,
                rL_view,
                rR_view,
                flux_view
            );

            using Cons = FTraits::cons;
            constexpr i32 n_hydro = FTraits::num_vars;
            for (int v = n_hydro; v < n_hydro + n_tracer; ++v) {
                // NOTE(cmo): Upwind at the interface
                if (flux_view(I(Cons::Rho)) >= 0.0_fp) {
                    flux_view(v) = flux_view(I(Cons::Rho)) * rL_view(v);
                } else {
                    flux_view(v) = flux_view(I(Cons::Rho)) * rR_view(v);
                }
            }
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

template <RiemannSolver rs, int Axis, typename FTraits>
std::pair<
    //         solver        axis  fluid_type num_dim
    std::tuple<RiemannSolver, int, FluidType, int>,
    std::function<void(const Simulation&)>
>
make_flux_impl() {
    return {
        std::make_tuple(rs, Axis, FTraits::fluid_type, FTraits::num_dim),
        compute_flux_impl<rs, Axis, FTraits>
    };
}

// template <typename FTraits, typename WType>
// KOKKOS_INLINE_FUNCTION void glm_ch_reducer(const Eos& eos, const fp_t mu0, const WType& w, const fp_t dx, fp_t& max_ch) {
//     static_assert(FTraits::fluid_type == FluidType::GlmMhd, "Only needed by GLM MHD");

//     using Prim = FTraits::prim;
//     constexpr i32 NumDim = FTraits::num_dim;

//     fp_t cf = fast_wave_speed<FTraits, 0>(eos.gamma, mu0, w);
//     fp_t vel = std::abs(w(I(Prim::Vx)));
//     // NOTE(cmo): Absolute values of the outermost edges of the Riemann fan
//     max_ch = std::max(max_ch, std::abs(vel - cf));
//     max_ch = std::max(max_ch, std::abs(vel + cf));
//     if constexpr (NumDim > 1) {
//         vel = std::abs(w(I(Prim::Vy)));
//         cf = fast_wave_speed<FTraits, 1>(eos.gamma, mu0, w);
//         max_ch = std::max(max_ch, std::abs(vel - cf));
//         max_ch = std::max(max_ch, std::abs(vel + cf));
//     }
//     if constexpr (NumDim > 2) {
//         vel = std::abs(w(I(Prim::Vz)));
//         cf = fast_wave_speed<FTraits, 2>(eos.gamma, mu0, w);
//         max_ch = std::max(max_ch, std::abs(vel - cf));
//         max_ch = std::max(max_ch, std::abs(vel + cf));
//     }
// }

template <typename FTraits, typename WType>
KOKKOS_INLINE_FUNCTION void dt_reducer(const Eos& eos, const fp_t mu0, const WType& w, const fp_t dx, fp_t& running_dt) {
    using Prim = FTraits::prim;
    constexpr i32 NumDim = FTraits::num_dim;
    // fp_t cs = fast_wave_speed<FTraits, 0>(eos.gamma, mu0, w);
    // fp_t vel2 = square(w(I(Prim::Vx)));
    // if (NumDim > 1) {
    //     vel2 += square(w(I(Prim::Vy)));
    //     if constexpr (FTraits::is_mhd) {
    //         cs = std::max(cs, fast_wave_speed<FTraits, 1>(eos.gamma, mu0, w));
    //     }
    // }
    // if (NumDim > 2) {
    //     vel2 += square(w(I(Prim::Vz)));
    //     if constexpr (FTraits::is_mhd) {
    //         cs = std::max(cs, fast_wave_speed<FTraits, 2>(eos.gamma, mu0, w));
    //     }
    // }
    // const fp_t vel = std::sqrt(vel2);
    // const fp_t dt_local = dx / (cs + vel);
    fp_t cs = fast_wave_speed<FTraits, 0>(eos.gamma, mu0, w);
    fp_t vel = std::abs(w(I(Prim::Vx)));
    // fp_t dt_local = dx / (cs + vel);
    fp_t inv_dt = (cs + vel) / dx;
    if constexpr (NumDim > 1) {
        vel = std::abs(w(I(Prim::Vy)));
        if constexpr (FTraits::is_mhd) {
            cs = fast_wave_speed<FTraits, 1>(eos.gamma, mu0, w);
        }
        // dt_local = std::min(dt_local, dx / (cs + vel));
        inv_dt += (cs + vel) / dx;
    }
    if constexpr (NumDim > 2) {
        vel = std::abs(w(I(Prim::Vz)));
        if constexpr (FTraits::is_mhd) {
            cs = fast_wave_speed<FTraits, 2>(eos.gamma, mu0, w);
        }
        // dt_local = std::min(dt_local, dx / (cs + vel));
        inv_dt += (cs + vel) / dx;
    }
    // running_dt = std::min(dt_local, running_dt);
    running_dt = std::min(1.0_fp / inv_dt, running_dt);
}

template <typename FTraits>
f64 compute_dt_impl(const Simulation& sim) {
    const auto& state = sim.state;
    const auto& eos = sim.eos;

    fp_t dt_max = 1e5_fp;
    dex_parallel_reduce(
        "CFL reduction",
        FlatLoop<3>(state.sz.zc, state.sz.yc, state.sz.xc),
        KOKKOS_LAMBDA (int k, int j, int i, fp_t& running_dt) {
            yakl::SArray<fp_t, 1, FTraits::num_vars> w;
            CellIndex idx {
                .i = i,
                .j = j,
                .k = k
            };
            cons_to_prim<FTraits>(eos.gamma, state.mu0, QtyView(state.Q, idx), w);
            dt_reducer<FTraits>(eos, state.mu0, w, state.dx, running_dt);
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
        std::tuple<RiemannSolver, int, FluidType, int>,
        std::function<void(const Simulation&)>
    > flux_fns = {
        //             solver                axis          num_dim   fluid_type
        make_flux_impl<RiemannSolver::Rusanov, 0, FluidTraits<1, FluidType::Hydro>>(),
        make_flux_impl<RiemannSolver::Hll,     0, FluidTraits<1, FluidType::Hydro>>(),
        make_flux_impl<RiemannSolver::Hllc,    0, FluidTraits<1, FluidType::Hydro>>(),
        make_flux_impl<RiemannSolver::Rusanov, 0, FluidTraits<2, FluidType::Hydro>>(),
        make_flux_impl<RiemannSolver::Hll,     0, FluidTraits<2, FluidType::Hydro>>(),
        make_flux_impl<RiemannSolver::Hllc,    0, FluidTraits<2, FluidType::Hydro>>(),
        make_flux_impl<RiemannSolver::Rusanov, 0, FluidTraits<3, FluidType::Hydro>>(),
        make_flux_impl<RiemannSolver::Hll,     0, FluidTraits<3, FluidType::Hydro>>(),
        make_flux_impl<RiemannSolver::Hllc,    0, FluidTraits<3, FluidType::Hydro>>(),
        make_flux_impl<RiemannSolver::Rusanov, 1, FluidTraits<2, FluidType::Hydro>>(),
        make_flux_impl<RiemannSolver::Hll,     1, FluidTraits<2, FluidType::Hydro>>(),
        make_flux_impl<RiemannSolver::Hllc,    1, FluidTraits<2, FluidType::Hydro>>(),
        make_flux_impl<RiemannSolver::Rusanov, 1, FluidTraits<3, FluidType::Hydro>>(),
        make_flux_impl<RiemannSolver::Hll,     1, FluidTraits<3, FluidType::Hydro>>(),
        make_flux_impl<RiemannSolver::Hllc,    1, FluidTraits<3, FluidType::Hydro>>(),
        make_flux_impl<RiemannSolver::Rusanov, 2, FluidTraits<3, FluidType::Hydro>>(),
        make_flux_impl<RiemannSolver::Hll,     2, FluidTraits<3, FluidType::Hydro>>(),
        make_flux_impl<RiemannSolver::Hllc,    2, FluidTraits<3, FluidType::Hydro>>(),

        make_flux_impl<RiemannSolver::Rusanov, 0, FluidTraits<1, FluidType::Mhd>>(),
        make_flux_impl<RiemannSolver::Hll,     0, FluidTraits<1, FluidType::Mhd>>(),
        make_flux_impl<RiemannSolver::Rusanov, 0, FluidTraits<2, FluidType::Mhd>>(),
        make_flux_impl<RiemannSolver::Hll,     0, FluidTraits<2, FluidType::Mhd>>(),
        make_flux_impl<RiemannSolver::Rusanov, 0, FluidTraits<3, FluidType::Mhd>>(),
        make_flux_impl<RiemannSolver::Hll,     0, FluidTraits<3, FluidType::Mhd>>(),
        make_flux_impl<RiemannSolver::Rusanov, 1, FluidTraits<2, FluidType::Mhd>>(),
        make_flux_impl<RiemannSolver::Hll,     1, FluidTraits<2, FluidType::Mhd>>(),
        make_flux_impl<RiemannSolver::Rusanov, 1, FluidTraits<3, FluidType::Mhd>>(),
        make_flux_impl<RiemannSolver::Hll,     1, FluidTraits<3, FluidType::Mhd>>(),
        make_flux_impl<RiemannSolver::Rusanov, 2, FluidTraits<3, FluidType::Mhd>>(),
        make_flux_impl<RiemannSolver::Hll,     2, FluidTraits<3, FluidType::Mhd>>(),

        make_flux_impl<RiemannSolver::Rusanov, 0, FluidTraits<1, FluidType::GlmMhd>>(),
        make_flux_impl<RiemannSolver::Hll,     0, FluidTraits<1, FluidType::GlmMhd>>(),
        make_flux_impl<RiemannSolver::Rusanov, 0, FluidTraits<2, FluidType::GlmMhd>>(),
        make_flux_impl<RiemannSolver::Hll,     0, FluidTraits<2, FluidType::GlmMhd>>(),
        make_flux_impl<RiemannSolver::Rusanov, 0, FluidTraits<3, FluidType::GlmMhd>>(),
        make_flux_impl<RiemannSolver::Hll,     0, FluidTraits<3, FluidType::GlmMhd>>(),
        make_flux_impl<RiemannSolver::Rusanov, 1, FluidTraits<2, FluidType::GlmMhd>>(),
        make_flux_impl<RiemannSolver::Hll,     1, FluidTraits<2, FluidType::GlmMhd>>(),
        make_flux_impl<RiemannSolver::Rusanov, 1, FluidTraits<3, FluidType::GlmMhd>>(),
        make_flux_impl<RiemannSolver::Hll,     1, FluidTraits<3, FluidType::GlmMhd>>(),
        make_flux_impl<RiemannSolver::Rusanov, 2, FluidTraits<3, FluidType::GlmMhd>>(),
        make_flux_impl<RiemannSolver::Hll,     2, FluidTraits<3, FluidType::GlmMhd>>()
    };
    sim.flux_fns.flux_x = flux_fns[std::make_tuple(schemes.riemann_solver, 0, sim.fluid_type, sim.num_dim)];
    if (sim.num_dim > 1) {
        sim.flux_fns.flux_y = flux_fns[std::make_tuple(schemes.riemann_solver, 1, sim.fluid_type, sim.num_dim)];
    }
    if (sim.num_dim > 2) {
        sim.flux_fns.flux_z = flux_fns[std::make_tuple(schemes.riemann_solver, 2, sim.fluid_type, sim.num_dim)];
    }

    std::map<std::pair<int, FluidType>, std::function<fp_t(const Simulation&)>> cfl_fns = {
        {{1, FluidType::Hydro}, compute_dt_impl<FluidTraits<1, FluidType::Hydro>>},
        {{2, FluidType::Hydro}, compute_dt_impl<FluidTraits<2, FluidType::Hydro>>},
        {{3, FluidType::Hydro}, compute_dt_impl<FluidTraits<3, FluidType::Hydro>>},
        {{1, FluidType::Mhd}, compute_dt_impl<FluidTraits<1, FluidType::Mhd>>},
        {{2, FluidType::Mhd}, compute_dt_impl<FluidTraits<2, FluidType::Mhd>>},
        {{3, FluidType::Mhd}, compute_dt_impl<FluidTraits<3, FluidType::Mhd>>},
        {{1, FluidType::GlmMhd}, compute_dt_impl<FluidTraits<1, FluidType::GlmMhd>>},
        {{2, FluidType::GlmMhd}, compute_dt_impl<FluidTraits<2, FluidType::GlmMhd>>},
        {{3, FluidType::GlmMhd}, compute_dt_impl<FluidTraits<3, FluidType::GlmMhd>>}
    };
    sim.compute_dt = cfl_fns[std::make_pair(sim.num_dim, sim.fluid_type)];
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

}