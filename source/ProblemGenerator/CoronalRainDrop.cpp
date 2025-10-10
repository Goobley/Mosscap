#include "ProblemGenerator.hpp"
#include "../Hydro.hpp"
#include "../MosscapConfig.hpp"
#include "../SourceTerms/Gravity.hpp"

// NOTE(cmo): This is a 2d problem
static constexpr int num_dim = 2;

namespace Mosscap {

struct BcParams {
    fp_t g_y;
};

template <int Axis, int NumDim>
static void fill_one_bc_hse(const Simulation& sim, const BcParams& driver) {
    static_assert(Axis < 3, "What are you doing?");
    const auto& state = sim.state;
    const auto& sz = state.sz;
    const auto& bdry = state.boundaries;
    const int ng = state.sz.ng;
    const auto& eos = sim.eos;
    const fp_t time = sim.time;

    constexpr const char* kernel_name[3] = {"Fill BCs x", "Fill BCs y", "Fill BCs z"};
    int dims[3] = {sz.xc, sz.yc, sz.zc};
    int launch_dims[3] = {sz.xc, sz.yc, sz.zc};
    launch_dims[Axis] = 1;

    fp_t dt_sub = sim.dt_sub;

    dex_parallel_for(
        kernel_name[Axis],
        FlatLoop<3>(launch_dims[2], launch_dims[1], launch_dims[0]),
        KOKKOS_LAMBDA (int ki, int ji, int ii) {
            using Cons = Cons<NumDim>;
            constexpr int IM = Momentum<Axis, NumDim>();
            int coord[3] = {ii, ji, ki};
            for (int a = ng - 1; a > -1; --a) {
                coord[Axis] = a;
                const int pencil_idx = coord[Axis];
                int cflip = (2 * ng - 1) - coord[Axis];
                int cedge = ng;
                if (pencil_idx >= ng) {
                    coord[Axis] = (dims[Axis] - 1) - (pencil_idx - ng);
                    cflip = (dims[Axis] - 1) - (2 * ng - 1) + (pencil_idx - ng);
                    cedge = (dims[Axis] - 1) - ng;
                }

                CellIndex idx{
                    .i = coord[0],
                    .j = coord[1],
                    .k = coord[2]
                };
                CellIndex i_prev(idx);
                // NOTE(cmo): we are integrating downwards
                i_prev.along<Axis>() += 1;
                CellIndex i_edge(idx);
                i_edge.along<Axis>() = cedge;

                auto Q_view = QtyView(state.Q, idx);
                auto Q_edge = QtyView(state.Q, i_edge);
                auto Q_prev = QtyView(state.Q, i_prev);

                BoundaryType start_bound, end_bound;
                JasUse(bdry);
                if constexpr (Axis == 0) {
                    start_bound = bdry.xs;
                    end_bound = bdry.xe;
                } else if constexpr (Axis == 1) {
                    start_bound = bdry.ys;
                    end_bound = bdry.ye;
                } else {
                    start_bound = bdry.zs;
                    end_bound = bdry.ze;
                }
                BoundaryType bound = (coord[Axis] < ng) ? start_bound : end_bound;

                if (bound == BoundaryType::UserFn) {
                    using Prim = Prim<NumDim>;
                    yakl::SArray<fp_t, 1, N_HYDRO_VARS<NumDim>> w;
                    cons_to_prim<NumDim>(eos.gamma, Q_prev, w);
                    // NOTE(cmo): The following is hardcoded to 1D for now
                    fp_t p = w(I(Prim::Pres)) - 0.5_fp * (Q_view(I(Cons::Rho)) + Q_prev(I(Cons::Rho))) * driver.g_y * state.dx;
                    // add that contribution to rho and eint
                    // flip or set momentum to 0

                    // Assume all change in pressure from rho
                    Q_view(I(Cons::Rho)) = p / w(I(Prim::Pres)) * w(I(Prim::Rho));
                    Q_view(IM) = 0.0_fp;
                    // Diode condition
                    // if (Q_edge(IM) < 0.0_fp) {
                    //     Q_view(IM) = Q_edge(IM) / Q_edge(I(Cons::Rho)) * Q_view(I(Cons::Rho));
                    // }
                    // TODO(cmo): This isn't technically correct in the 2D case as there could be x-momentum too
                    Q_view(I(Cons::Ene)) = p / (eos.gamma - 1.0_fp)+ square(Q_view(IM)) / Q_view(I(Cons::Rho));

                    // const fp_t prev_mom2 = square(Q_view(IM));
                    // Q_view(IM) = -(Q_flip(IM) / Q_flip(I(Cons::Rho))) * Q_view(I(Cons::Rho));
                    // Q_view(IM) -= driver.g_y * dt_sub * Q_view(I(Cons::Rho));
                    // const fp_t new_mom2 = square(Q_view(IM));
                    // Q_view(I(Cons::Ene)) += (new_mom2 - prev_mom2) / Q_view(I(Cons::Rho));
                }
            }
        }
    );
    dex_parallel_for(
        kernel_name[Axis],
        FlatLoop<3>(launch_dims[2], launch_dims[1], launch_dims[0]),
        KOKKOS_LAMBDA (int ki, int ji, int ii) {
            using Cons = Cons<NumDim>;
            constexpr int IM = Momentum<Axis, NumDim>();
            int coord[3] = {ii, ji, ki};
            for (int a = 2 * sz.ng - 1; a > sz.ng - 1; --a) {
                coord[Axis] = a;
                const int pencil_idx = coord[Axis];
                int cflip = (2 * ng - 1) - coord[Axis];
                int cedge = ng;
                if (pencil_idx >= ng) {
                    coord[Axis] = (dims[Axis] - 1) - (pencil_idx - ng);
                    cflip = (dims[Axis] - 1) - (2 * ng - 1) + (pencil_idx - ng);
                    cedge = (dims[Axis] - 1) - ng;
                }

                CellIndex idx{
                    .i = coord[0],
                    .j = coord[1],
                    .k = coord[2]
                };
                CellIndex i_prev(idx);
                // NOTE(cmo): we are integrating upwards
                i_prev.along<Axis>() -= 1;
                CellIndex i_edge(idx);
                i_edge.along<Axis>() = cedge;

                auto Q_view = QtyView(state.Q, idx);
                auto Q_edge = QtyView(state.Q, i_edge);
                auto Q_prev = QtyView(state.Q, i_prev);

                BoundaryType start_bound, end_bound;
                JasUse(bdry);
                if constexpr (Axis == 0) {
                    start_bound = bdry.xs;
                    end_bound = bdry.xe;
                } else if constexpr (Axis == 1) {
                    start_bound = bdry.ys;
                    end_bound = bdry.ye;
                } else {
                    start_bound = bdry.zs;
                    end_bound = bdry.ze;
                }
                BoundaryType bound = (coord[Axis] < ng) ? start_bound : end_bound;

                if (bound == BoundaryType::UserFn) {
                    using Prim = Prim<NumDim>;
                    yakl::SArray<fp_t, 1, N_HYDRO_VARS<NumDim>> w;
                    cons_to_prim<NumDim>(eos.gamma, Q_prev, w);
                    // NOTE(cmo): The following is hardcoded to 1D for now
                    fp_t p = w(I(Prim::Pres)) + 0.5_fp * (Q_view(I(Cons::Rho)) + Q_prev(I(Cons::Rho))) * driver.g_y * state.dx;
                    // const fp_t dP_dz = h_mass * gravity;
                    // add that contribution to rho and eint
                    // flip or set momentum to 0

                    // Assume all change in pressure from rho
                    Q_view(I(Cons::Rho)) = p / w(I(Prim::Pres)) * w(I(Prim::Rho));
                    Q_view(IM) = 0.0_fp;
                    // Diode condition
                    // if (Q_edge(IM) > 0.0_fp) {
                    //     Q_view(IM) = Q_edge(IM) / Q_edge(I(Cons::Rho)) * Q_view(I(Cons::Rho));
                    // }
                    Q_view(I(Cons::Ene)) = p / (eos.gamma - 1.0_fp) + square(Q_view(IM)) / Q_view(I(Cons::Rho));
                    // for (int var = 0; var < state.Q.extent(0); ++var) {
                    //     Q_view(var) = Q_edge(var);
                    // }

                    // const fp_t prev_mom2 = square(Q_view(IM));
                    // Q_view(IM) = -(Q_flip(IM) / Q_flip(I(Cons::Rho))) * Q_view(I(Cons::Rho));
                    // Q_view(IM) -= driver.g_y * dt_sub * Q_view(I(Cons::Rho));
                    // const fp_t new_mom2 = square(Q_view(IM));
                    // Q_view(I(Cons::Ene)) += (new_mom2 - prev_mom2) / Q_view(I(Cons::Rho));
                }
            }
        }
    );
    Kokkos::fence();
}

MOSSCAP_NEW_PROBLEM(coronal_rain_drop_2d) {
    MOSSCAP_PROBLEM_PREAMBLE(coronal_rain_drop_2d);
    using Prim = Prim<num_dim>;
    constexpr int n_hydro = N_HYDRO_VARS<num_dim>;
    if (sim.num_dim != num_dim) {
        throw std::runtime_error(fmt::format(
            "{} only handles {}d problems", PROBLEM_NAME, num_dim
        ));
    }

    typedef yakl::Array<f64, 1, yakl::memHost> F64Host;
    const auto& state = sim.state;
    const auto& sz = state.sz;
    const auto& eos = sim.eos;

    static constexpr f64 h_mass = 1.6737830080950003e-27;
    static constexpr f64 k_B = 1.380649e-23;
    const fp_t rho_0 = 5e-12_fp;
    const fp_t P_0 = 0.165_fp;
    const fp_t T_0 = 2e6_fp;
    const fp_t g = get_or<fp_t>(config, "sources.gravity.y", -274.0_fp);
    const fp_t mean_mass = 1.0_fp;
    const fp_t H = k_B * T_0 / (mean_mass * h_mass * -g);
    // const fp_t rho_b0 = 10.0_fp * rho_0;
    const fp_t rho_b0 = 5e-10_fp;

    constexpr fp_t x0 = 0.0_fp;
    constexpr fp_t z0 = 50e6_fp;
    constexpr fp_t delta = 0.5e6_fp;

    F64Host rho("rho", sz.yc);
    F64Host pressure("pressure", sz.yc);
    rho(sz.ng) = rho_0;
    pressure(sz.ng) = P_0;
    const f64 dz = state.dx;
    for (int i = sz.ng + 1; i < sz.yc; ++i) {
        const f64 dP_dz_base = rho(i - 1) * g;
        const f64 P_half = pressure(i - 1) + dP_dz_base * 0.5 * dz;
        const f64 T_half = T_0;
        // NOTE(cmo): Assuming fully ionised background
        const f64 rho_half = 0.5_fp * P_half / (k_B * T_half) * (mean_mass * h_mass);

        const f64 dP_dz_mid = rho_half * g;
        pressure(i) = pressure(i - 1) + dP_dz_mid * dz;
        rho(i) = 0.5_fp * pressure(i) / (k_B * T_0) * (mean_mass * h_mass);
        // try to refine guess for FV scheme
        int iter = 0;
        for (iter = 0; iter < 100; ++iter) {
            const fp_t old_pressure = pressure(i);
            // https://iopscience.iop.org/article/10.1086/342754/fulltext/
            // Eq 40 + 41
            if (i == sz.ng + 1) {
                pressure(i) = pressure(i - 1) + 0.5 * g * dz * (rho(i) + rho(i - 1));
            } else {
                pressure(i) = pressure(i - 1) + 1.0/12.0 * g * dz * (5 * rho(i) + 8 * rho(i - 1) - rho(i-2));
            }
            if (std::abs(1.0 - pressure(i) / old_pressure) < 1e-5) {
                break;
            }
            rho(i) = 0.5_fp * pressure(i) / (k_B * T_0) * (mean_mass * h_mass);
        }
        if (iter == 100) {
            fmt::println("No converge: {}", i);
        }
    }
    const auto rho_z = rho.createDeviceCopy();
    const auto p_z = pressure.createDeviceCopy();

    dex_parallel_for(
        FlatLoop<3>(sz.zc, sz.yc, sz.xc),
        KOKKOS_LAMBDA (int k, int j, int i) {
            yakl::SArray<fp_t, 1, n_hydro> w(0.0_fp);
            w(I(Prim::Vx)) = 0.0_fp;
            w(I(Prim::Vy)) = 0.0_fp;
            const int jj = std::max(j, sz.ng);
            w(I(Prim::Rho)) = rho_z(jj);
            w(I(Prim::Pres)) = p_z(jj);

            vec3 p = state.get_pos(i, j, k);
            const fp_t gauss_factor = std::exp(-(square(p(0) - x0) + square(p(1) - z0)) / square(delta));
            w(I(Prim::Rho)) += rho_b0 * gauss_factor;

            CellIndex idx {
                .i = i,
                .j = j,
                .k = k
            };
            prim_to_cons<num_dim>(eos.gamma, w, QtyView(state.Q, idx));
        }
    );
    sim.max_time = get_or<fp_t>(config, "timestep.max_time", 400.0_fp);
    setup_gravity(sim, config);
    BcParams bc_params {
        .g_y = g
    };
    sim.user_bc = [=](const Simulation& sim) {
        fill_one_bc_hse<1, num_dim>(sim, bc_params);
    };
}

}