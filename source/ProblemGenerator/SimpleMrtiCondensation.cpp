#include "ProblemGenerator.hpp"
#include "../Hydro.hpp"
#include "../MosscapConfig.hpp"
#include "../SourceTerms/Gravity.hpp"
#include "../SourceTerms/Sponge.hpp"

// NOTE(cmo): This is a 2d problem
static constexpr int num_dim = 2;

namespace Mosscap {

struct BcParams {
    fp_t g_y;
};

template <int Axis, typename FTraits>
static void fill_one_bc_hse(const Simulation& sim, const BcParams& driver) {
    static_assert(Axis < 3, "What are you doing?");
    const auto& state = sim.state;
    const auto& sz = state.sz;
    const auto& bdry = state.boundaries;
    const int ng = state.sz.ng;
    const auto& eos = sim.eos;

    constexpr fp_t damping_factor = 0.0_fp;
    constexpr const char* kernel_name[3] = {"Fill BCs x", "Fill BCs y", "Fill BCs z"};
    int dims[3] = {sz.xc, sz.yc, sz.zc};
    int launch_dims[3] = {sz.xc, sz.yc, sz.zc};
    launch_dims[Axis] = 1;

    dex_parallel_for(
        kernel_name[Axis],
        FlatLoop<3>(launch_dims[2], launch_dims[1], launch_dims[0]),
        KOKKOS_LAMBDA (int ki, int ji, int ii) {
            using Cons = typename FTraits::cons;
            constexpr int IM = Momentum<Axis, FTraits>();
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
                    using Prim = typename FTraits::prim;
                    yakl::SArray<fp_t, 1, FTraits::num_vars> w;
                    cons_to_prim<FTraits>(eos.gamma, state.mu0, Q_prev, w);
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
                    Q_view(I(Cons::Ene)) = p / (eos.gamma - 1.0_fp) + square(Q_view(IM)) / Q_view(I(Cons::Rho));
                    JasUse(damping_factor);
                    if constexpr (FTraits::is_mhd) {
                        fp_t damping = std::exp(-damping_factor * std::abs(i_edge.j - idx.j));
                        Q_view(I(Cons::Bx)) = damping * Q_edge(I(Cons::Bx));
                        Q_view(I(Cons::By)) = Q_edge(I(Cons::By));
                        Q_view(I(Cons::Bz)) = damping * Q_edge(I(Cons::Bz));
                        Q_view(I(Cons::Ene)) += (square(Q_view(I(Cons::Bx))) + square(Q_view(I(Cons::By))) + square(Q_view(I(Cons::Bz)))) / (2.0_fp * state.mu0);
                    }

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
            using Cons = typename FTraits::cons;
            constexpr int IM = Momentum<Axis, FTraits>();
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
                    using Prim = typename FTraits::prim;
                    yakl::SArray<fp_t, 1, FTraits::num_vars> w;
                    cons_to_prim<FTraits>(eos.gamma, state.mu0, Q_prev, w);
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
                    JasUse(damping_factor);
                    if constexpr (FTraits::is_mhd) {
                        fp_t damping = std::exp(-damping_factor * std::abs(i_edge.j - idx.j));
                        Q_view(I(Cons::Bx)) = damping * Q_edge(I(Cons::Bx));
                        Q_view(I(Cons::By)) = Q_edge(I(Cons::By));
                        Q_view(I(Cons::Bz)) = damping * Q_edge(I(Cons::Bz));
                        Q_view(I(Cons::Ene)) += (square(Q_view(I(Cons::Bx))) + square(Q_view(I(Cons::By))) + square(Q_view(I(Cons::Bz)))) / (2.0_fp * state.mu0);
                    }
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

template <typename Fluid>
static void initial_conditions(Simulation& sim, const YAML::Node& config) {
    using Prim = typename Fluid::prim;
    constexpr int n_hydro = Fluid::num_vars;
    typedef yakl::Array<f64, 1, yakl::memHost> F64Host;
    const auto& state = sim.state;
    const auto& sz = state.sz;
    const auto& eos = sim.eos;

    static constexpr f64 h_mass = 1.6737830080950003e-27;
    static constexpr f64 k_B = 1.380649e-23;
    const fp_t T_0 = get_or<fp_t>(config, "problem.base_temperature", 1.2e6_fp);
    const fp_t P_0 = get_or<fp_t>(config, "problem.base_pressure", 0.07_fp);
    const fp_t g = get_or<fp_t>(config, "sources.gravity.y", -274.0_fp);
    const fp_t T_blob = get_or<fp_t>(config, "problem.blob_temperature", 10e3_fp);

    const fp_t x0 = get_or<fp_t>(config, "problem.blob_x0", 0.0_fp);
    const fp_t y0 = get_or<fp_t>(config, "problem.blob_y0", 10e6_fp);
    const fp_t blob_width_x = get_or<fp_t>(config, "problem.blob_width_x", 3e6_fp);
    const fp_t blob_width_y = get_or<fp_t>(config, "problem.blob_width_y", 3e6_fp);
    const fp_t tr_width = get_or<fp_t>(config, "problem.tr_width", 2e5_fp);

    const fp_t bx0 = get_or<fp_t>(config, "problem.bx0", 0.0);
    const fp_t by0 = get_or<fp_t>(config, "problem.by0", 0.0);
    const fp_t bz0 = get_or<fp_t>(config, "problem.bz0", 0.0);
    const fp_t pert_scale = get_or<fp_t>(config, "problem.vel_pert_scale", 10.0);
    const u64 seed = get_or<u64>(config, "problem.seed", 1234567UL);

    const bool col_by_col = get_or<bool>(config, "problem.column_by_column_hse", false);
    const fp_t bz_blob_multiplier = get_or<fp_t>(config, "problem.bz_blob_multiplier", 1.0);

    // Coronal background density = P_0 / (2 * k_B T_0) * h_mass -- fully ionised
    const fp_t rho_0 = P_0 / (2.0_fp * k_B * T_0) * h_mass;
    fmt::println("Base coronal density {:.2e} kg/m3", rho_0);
    const fp_t mean_mass = 1.0_fp;
    const fp_t H = k_B * T_0 / (mean_mass * h_mass * -g);
    const f64 dy = state.dx;

    F64Host rho("rho", sz.yc);
    F64Host pressure("pressure", sz.yc);
    rho(sz.ng) = rho_0;
    pressure(sz.ng) = P_0;
    for (int i = sz.ng + 1; i < sz.yc; ++i) {
        const f64 dP_dy_base = rho(i - 1) * g;
        const f64 P_half = pressure(i - 1) + dP_dy_base * 0.5 * dy;
        const f64 T_half = T_0;
        // NOTE(cmo): Assuming fully ionised background
        const f64 rho_half = 0.5_fp * P_half / (k_B * T_half) * (mean_mass * h_mass);

        const f64 dP_dy_mid = rho_half * g;
        pressure(i) = pressure(i - 1) + dP_dy_mid * dy;
        rho(i) = 0.5_fp * pressure(i) / (k_B * T_0) * (mean_mass * h_mass);
        // try to refine guess for FV scheme
        int iter = 0;
        for (iter = 0; iter < 100; ++iter) {
            const fp_t old_pressure = pressure(i);
            // https://iopscience.iop.org/article/10.1086/342754/fulltext/
            // Eq 40 + 41
            if (i == sz.ng + 1) {
                pressure(i) = pressure(i - 1) + 0.5 * g * dy * (rho(i) + rho(i - 1));
            } else {
                pressure(i) = pressure(i - 1) + 1.0/12.0 * g * dy * (5 * rho(i) + 8 * rho(i - 1) - rho(i-2));
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
        FlatLoop<2>(sz.zc, sz.xc),
        KOKKOS_LAMBDA (int k, int i) {
            if (col_by_col) {
                yakl::SArray<fp_t, 1, n_hydro> w(0.0_fp);
                for (int j = 0; j < sz.ng + 1; ++j) {
                    w(I(Prim::Rho)) = rho_0;
                    w(I(Prim::Pres)) = P_0;

                    JasUse(bx0, by0, bz0);
                    if constexpr (Fluid::is_mhd) {
                        w(I(Prim::Bx)) = bx0;
                        w(I(Prim::By)) = by0;
                        w(I(Prim::Bz)) = bz0;
                    }
                    CellIndex idx {
                        .i = i,
                        .j = j,
                        .k = k
                    };
                    prim_to_cons<Fluid>(eos.gamma, state.mu0, w, QtyView(state.Q, idx));
                }
                for (int j = sz.ng + 1; j < sz.yc + 2 * sz.ng; ++j) {
                    w(I(Prim::Vx)) = 0.0_fp;
                    w(I(Prim::Vy)) = 0.0_fp;
                    if constexpr (Fluid::is_mhd) {
                        w(I(Prim::Bx)) = bx0;
                        w(I(Prim::By)) = by0;
                        w(I(Prim::Bz)) = bz0;
                    }

                    const fp_t prev_pres = w(I(Prim::Pres));
                    const fp_t prev_rho = w(I(Prim::Rho));
                    const fp_t dP_dy_base = prev_rho * g;

                    vec3 p = state.get_pos(i, j, k);
                    vec3 p_half(p);
                    p_half(1) -= 0.5 * dy;

                    const fp_t pres_half = prev_pres + dP_dy_base * 0.5 * dy;
                    fp_t temp_half = T_0;

                    // NOTE(cmo): Use the product of tanhs in x and y to produce a
                    // rectangular-ish shape. This is ~1 inside the condensation, and 0
                    // outside
                    const fp_t fn_half_x = 0.5 * (
                        std::tanh((p_half(0) - x0 + 0.5 * blob_width_x) / tr_width)
                        - std::tanh((p_half(0) - x0 - 0.5 * blob_width_x) / tr_width)
                    );
                    const fp_t fn_half_y = 0.5 * (
                        std::tanh((p_half(1) - y0 + 0.5 * blob_width_y) / tr_width)
                        - std::tanh((p_half(1) - y0 - 0.5 * blob_width_y) / tr_width)
                    );
                    fp_t prod = fn_half_x * fn_half_y;
                    if (prod > 1e-6_fp) {
                        temp_half = T_blob + (T_0 - T_blob) * (1.0 - prod);
                    }
                    const fp_t rho_half = pres_half / (2.0_fp * k_B * temp_half) * h_mass;
                    const fp_t dP_dy_mid = rho_half * g;

                    w(I(Prim::Pres)) = prev_pres + dP_dy_mid * dy;
                    fp_t temp_full = T_0;

                    const fp_t fn_x = 0.5 * (
                        std::tanh((p(0) - x0 + 0.5 * blob_width_x) / tr_width)
                        - std::tanh((p(0) - x0 - 0.5 * blob_width_x) / tr_width)
                    );
                    const fp_t fn_y = 0.5 * (
                        std::tanh((p(1) - y0 + 0.5 * blob_width_y) / tr_width)
                        - std::tanh((p(1) - y0 - 0.5 * blob_width_y) / tr_width)
                    );
                    prod = fn_x * fn_y;
                    if (prod > 1e-6_fp) {
                        temp_full = T_blob + (T_0 - T_blob) * (1.0 - prod);
                        if (prod > 0.1 && prod < 0.9) {
                            yakl::Random rng(seed + k * sz.yc * sz.xc + j * sz.xc + i);
                            w(I(Prim::Vy)) = pert_scale * (rng.genFP<fp_t>() - 0.5_fp);
                        }
                        w(I(Prim::Bz)) *= std::max(bz_blob_multiplier * prod, 1.0_fp);
                    }
                    w(I(Prim::Rho)) = w(I(Prim::Pres)) / (2.0_fp * k_B * temp_full) * h_mass;

                    CellIndex idx {
                        .i = i,
                        .j = j,
                        .k = k
                    };
                    prim_to_cons<Fluid>(eos.gamma, state.mu0, w, QtyView(state.Q, idx));
                }
            } else {
                yakl::SArray<fp_t, 1, n_hydro> w(0.0_fp);
                for (int j = 0; j < sz.yc; ++j) {
                    int jj = std::max(j, sz.ng);
                    w(I(Prim::Rho)) = rho_z(jj);
                    w(I(Prim::Vx)) = 0.0_fp;
                    w(I(Prim::Vy)) = 0.0_fp;
                    if constexpr (Fluid::is_mhd) {
                        w(I(Prim::Bx)) = bx0;
                        w(I(Prim::By)) = by0;
                        w(I(Prim::Bz)) = bz0;
                    }
                    w(I(Prim::Pres)) = p_z(jj);

                    vec3 p = state.get_pos(i, j, k);
                    const fp_t fn_x = 0.5 * (
                        std::tanh((p(0) - x0 + 0.5 * blob_width_x) / tr_width)
                        - std::tanh((p(0) - x0 - 0.5 * blob_width_x) / tr_width)
                    );
                    const fp_t fn_y = 0.5 * (
                        std::tanh((p(1) - y0 + 0.5 * blob_width_y) / tr_width)
                        - std::tanh((p(1) - y0 - 0.5 * blob_width_y) / tr_width)
                    );
                    fp_t prod = fn_x * fn_y;
                    if (prod > 1e-6_fp) {
                        fp_t temp_full = T_blob + (T_0 - T_blob) * (1.0 - prod);
                        w(I(Prim::Rho)) = w(I(Prim::Pres)) / (2.0_fp * k_B * temp_full) * h_mass;
                        if (prod > 0.1 && prod < 0.9) {
                            yakl::Random rng(seed + k * sz.yc * sz.xc + j * sz.xc + i);
                            w(I(Prim::Vy)) = pert_scale * (rng.genFP<fp_t>() - 0.5_fp);
                        }
                        w(I(Prim::Bz)) *= std::max(bz_blob_multiplier * prod, 1.0_fp);
                    }
                    CellIndex idx {
                        .i = i,
                        .j = j,
                        .k = k
                    };
                    prim_to_cons<Fluid>(eos.gamma, state.mu0, w, QtyView(state.Q, idx));
                }
            }
        }
    );
}

MOSSCAP_NEW_PROBLEM(simple_mrti_condensation) {
    MOSSCAP_PROBLEM_PREAMBLE(simple_mrti_condensation);
    if (sim.num_dim != num_dim) {
        throw std::runtime_error(fmt::format(
            "{} only handles {}d problems", PROBLEM_NAME, num_dim
        ));
    }

    FluidTraitsRt traits(sim.num_dim, sim.fluid_type);
    sim.setup_ics = [=](Simulation& sim) {
        if (sim.fluid_type == FluidType::Hydro) {
            initial_conditions<FluidTraits<num_dim, FluidType::Hydro>>(sim, config);
        } else if (traits.is_mhd) {
            initial_conditions<FluidTraits<num_dim, FluidType::Mhd>>(sim, config);
        } else {
            throw std::runtime_error("Unknown fluid type");
        }
    };

    setup_gravity(sim, config);
    const fp_t g = get_or<fp_t>(config, "sources.gravity.y", -274.0_fp);
    BcParams bc_params {
        .g_y = g
    };

    if (get_or<bool>(config, "problem.enable_sponge", false)) {
        setup_sponge(sim, config);
    }

    sim.user_bc = invoke_fluid_traits(
        sim.num_dim,
        sim.fluid_type,
        [=] <typename FTraits> (FTraits) -> std::function<void(const Simulation&)> {
            return [=] (const Simulation& sim) {
                fill_one_bc_hse<1, FTraits>(sim, bc_params);
            };
        }
    );
}

}