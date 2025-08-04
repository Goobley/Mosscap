#include "ProblemGenerator.hpp"
#include "../Hydro.hpp"
#include "../MosscapConfig.hpp"
#include "YAKL_netcdf.h"
#include "../SourceTerms/Gravity.hpp"
#include "../TabulatedLteH.hpp"

// NOTE(cmo): This is a 1d problem
static constexpr int num_dim = 1;

// We assume a fully H atmosphere with LTE ionisation

struct OurWaveDriver {
    fp_t start_time;
    fp_t period;
    fp_t amplitude;
};

template <int Axis, int NumDim>
static void fill_one_bc_hse(const Simulation& sim, const OurWaveDriver& driver) {
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
    // launch_dims[Axis] = 2 * ng;
    launch_dims[Axis] = 1;

    constexpr fp_t g_x = -FP(274.0);
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
                    auto ev = EosView(eos, i_edge);
                    yakl::SArray<fp_t, 1, N_HYDRO_VARS<NumDim>> w;
                    cons_to_prim<NumDim>(ev, Q_prev, w);
                    auto g = ev.get_gamma_e();
                    // NOTE(cmo): The following is hardcoded to 1D for now
                    fp_t p = w(I(Prim::Pres)) - FP(0.5) * (Q_view(I(Cons::Rho)) + Q_prev(I(Cons::Rho))) * g_x * state.dx;
                    // const fp_t dP_dz = h_mass * gravity;
                    // add that contribution to rho and eint
                    // flip or set momentum to 0

                    // Assume all change in pressure from rho
                    Q_view(I(Cons::Rho)) = p / w(I(Prim::Pres)) * w(I(Prim::Rho));
                    Q_view(IM) = FP(0.0);
                    if (time >= driver.start_time) {
                        const fp_t cs = sound_speed<num_dim>(EosView(eos, idx), w);
                        const fp_t vbase = driver.amplitude * cs * std::sin((FP(2.0) * M_PI) / driver.period * time);
                        Q_view(IM) = Q_view(I(Cons::Rho)) * vbase;
                    }
                    // Diode condition
                    // if (Q_edge(IM) > FP(0.0)) {
                    //     Q_view(IM) = Q_edge(IM) / Q_edge(I(Cons::Rho)) * Q_view(I(Cons::Rho));
                    // }
                    Q_view(I(Cons::Ene)) = p / g.gamma_e_m1 + square(Q_view(IM)) / Q_view(I(Cons::Rho));
                    // for (int var = 0; var < state.Q.extent(0); ++var) {
                    //     Q_view(var) = Q_edge(var);
                    // }

                    // const fp_t prev_mom2 = square(Q_view(IM));
                    // Q_view(IM) = -(Q_flip(IM) / Q_flip(I(Cons::Rho))) * Q_view(I(Cons::Rho));
                    // Q_view(IM) -= FP(1.001) * g_x * dt_sub * Q_view(I(Cons::Rho));
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
                    auto ev = EosView(eos, i_edge);
                    yakl::SArray<fp_t, 1, N_HYDRO_VARS<NumDim>> w;
                    cons_to_prim<NumDim>(ev, Q_prev, w);
                    auto g = ev.get_gamma_e();
                    // NOTE(cmo): The following is hardcoded to 1D for now
                    fp_t p = w(I(Prim::Pres)) + FP(0.5) * (Q_view(I(Cons::Rho)) + Q_prev(I(Cons::Rho))) * g_x * state.dx;
                    // const fp_t dP_dz = h_mass * gravity;
                    // add that contribution to rho and eint
                    // flip or set momentum to 0

                    // Assume all change in pressure from rho
                    Q_view(I(Cons::Rho)) = p / w(I(Prim::Pres)) * w(I(Prim::Rho));
                    Q_view(IM) = FP(0.0);
                    // Diode condition
                    if (Q_edge(IM) > FP(0.0)) {
                        Q_view(IM) = Q_edge(IM) / Q_edge(I(Cons::Rho)) * Q_view(I(Cons::Rho));
                    }
                    Q_view(I(Cons::Ene)) = p / g.gamma_e_m1 + square(Q_view(IM)) / Q_view(I(Cons::Rho));
                    // for (int var = 0; var < state.Q.extent(0); ++var) {
                    //     Q_view(var) = Q_edge(var);
                    // }

                    // const fp_t prev_mom2 = square(Q_view(IM));
                    // Q_view(IM) = -(Q_flip(IM) / Q_flip(I(Cons::Rho))) * Q_view(I(Cons::Rho));
                    // Q_view(IM) -= FP(1.001) * g_x * dt_sub * Q_view(I(Cons::Rho));
                    // const fp_t new_mom2 = square(Q_view(IM));
                    // Q_view(I(Cons::Ene)) += (new_mom2 - prev_mom2) / Q_view(I(Cons::Rho));
                }
            }
        }
    );
    Kokkos::fence();
}

// Interp functions from dex
/** Upper bound on YAKL arrays, returns index rather than iterator.
 *
*/
template <typename T, int mem_space=yakl::memDevice>
KOKKOS_INLINE_FUNCTION int upper_bound(const yakl::Array<const T, 1, mem_space>& x, T value) {
    int count = x.extent(0);
    int step;
    const T* first = &x(0);
    const T* it;

    while (count > 0)
    {
        it = first;
        step = count / 2;
        it += step;

        if (!(value < *it)) {
            // target in right sub-array
            first = ++it;
            count -= step + 1;
        } else {
            // target in left sub-array
            count = step;
        }
    }
    return first - &x(0);
}

/** Linearly interpolate a sample (at alpha) from array y on grid x. Assumes x is positive sorted.
 * Clamps on ends.
*/
template <typename T=fp_t, int mem_space=yakl::memDevice>
KOKKOS_INLINE_FUNCTION T interp(
    T alpha,
    const yakl::Array<T const, 1, mem_space>& x,
    const yakl::Array<T const, 1, mem_space>& y
) {
    if (alpha <= x(0)) {
        return y(0);
    } else if (alpha >= x(x.extent(0)-1)) {
        return y(y.extent(0)-1);
    }

    // NOTE(cmo): We know from the previous checks that idxp is in [1,
    // x.extent(0)-1] because our sample is guaranteed inside the grid. This is
    // the upper bound of the linear range.
    int idxp = upper_bound(x, alpha);
    int idx = idxp - 1;

    T t = (x(idxp) - alpha) / (x(idxp) - x(idx));
    return t * y(idx) + (FP(1.0) - t) * y(idxp);
}

template <typename T=fp_t, int mem_space=yakl::memDevice>
KOKKOS_INLINE_FUNCTION T interp(
    T alpha,
    const yakl::Array<T, 1, mem_space>& x,
    const yakl::Array<T, 1, mem_space>& y
) {
    // NOTE(cmo): The optimiser should eat this up
    // Narrator: It did not (because of the mutex in OpenMP land)
    yakl::Array<T const, 1, mem_space> xx(x);
    yakl::Array<T const, 1, mem_space> yy(y);
    return interp(alpha, xx, yy);
}

static KOKKOS_INLINE_FUNCTION f64 saha_rhs_H(f64 T) {
    // NOTE(cmo): From sympy
    return 2.4146830395719654e+21*std::pow(T, 1.5)*std::exp(-157763.42386247337/T);
}

static KOKKOS_INLINE_FUNCTION f64 y_from_nhtot(f64 nhtot, f64 T) {
    f64 X = saha_rhs_H(T);
    return 0.5 * (-X + std::sqrt(square(X) + 4 * nhtot * X)) / nhtot;
}

static KOKKOS_INLINE_FUNCTION f64 ne_from_ntot(f64 ntot, f64 T) {
    f64 X = saha_rhs_H(T);
    return 0.5 * (-2 * X + std::sqrt(square(2 * X) + 4 * ntot * X));
}

MOSSCAP_NEW_PROBLEM(solar_1d) {
    MOSSCAP_PROBLEM_PREAMBLE(solar_1d);

    std::string data_path = get_or<std::string>(config, "problem.data_path", "fal.nc");
    // fp_t solar_g = get_or<fp_t>(config, "sources.gravity.x", -274);
    fp_t solar_g = -274.0;
    // TODO(cmo): Set if not present?

    typedef yakl::Array<f64, 1, yakl::memHost> F64Host ;
    F64Host z;
    F64Host temperature_profile;
    yakl::SimpleNetCDF nc;
    nc.open(data_path, yakl::NETCDF_MODE_READ);
    nc.read(temperature_profile, "temperature");
    nc.read(z, "z");
    f64 base_pressure, base_nh;
    nc.read(base_pressure, "base_pressure");
    nc.read(base_nh, "base_nhtot");

    const bool ideal = sim.eos.is_constant;

    const auto& state = sim.state;
    const auto& eos = sim.eos;
    const auto& sz = sim.state.sz;
    F64Host temperature("temp", sz.xc);
    F64Host pressure("pressure", sz.xc);
    F64Host nhtot("nhtot", sz.xc);
    F64Host y("y", sz.xc);

    static constexpr f64 h_mass = 1.6737830080950003e-27;
    static constexpr f64 k_B = 1.380649e-23;
    static constexpr f64 chi_H = 2.178710282685096e-18; // [J]
    // Set up base values and interpolate run of temperature
    for (int i = 0; i < sz.xc; ++i) {
        temperature(i) = interp(state.get_pos(i)(0) + z(0), z, temperature_profile);
    }
    fmt::println("Max temperature {:.3e}", temperature(sz.xc - sz.ng - 1));
    nhtot(sz.ng) = base_nh;
    y(sz.ng) = y_from_nhtot(nhtot(sz.ng), temperature(sz.ng));
    if (ideal) {
        y(sz.ng) = FP(0.0);
    }
    pressure(sz.ng) = base_nh * (1.0 + y(sz.ng)) * k_B * temperature(sz.ng);
    fmt::println("P: {}, y: {}, nhtot {:e}", pressure(sz.ng), y(sz.ng), nhtot(sz.ng));

    const f64 dz = state.dx;
    for (int i = sz.ng + 1; i < sz.xc; ++i) {
        const f64 dP_dz_base = h_mass * nhtot(i-1) * solar_g;
        const f64 P_half = pressure(i - 1) + dP_dz_base * 0.5 * dz;
        const f64 T_half = 0.5 * (temperature(i) + temperature(i-1));
        const f64 ntot_half = P_half / (k_B * T_half);
        const f64 ne_half = ideal ? FP(0.0) : ne_from_ntot(ntot_half, T_half);
        const f64 nhtot_half = ntot_half - ne_half;

        const f64 dP_dz_mid = h_mass * nhtot_half * solar_g;
        pressure(i) = pressure(i - 1) + dP_dz_mid * dz;
        const f64 ntotal = pressure(i) / (k_B * temperature(i));
        const f64 ne = ideal ? FP(0.0) : ne_from_ntot(ntotal, temperature(i));
        nhtot(i) = ntotal - ne;
        // try to refine guess
        int iter = 0;
        for (iter = 0; iter < 100; ++iter) {
            const fp_t old_pressure = pressure(i);
            // https://iopscience.iop.org/article/10.1086/342754/fulltext/
            // Eq 40 + 41
            if (i == sz.ng + 1) {
                pressure(i) = pressure(i - 1) + 0.5 * solar_g * dz * (nhtot(i) + nhtot(i - 1)) * h_mass;
            } else {
                pressure(i) = pressure(i - 1) + 1.0/12.0 * solar_g * dz * (5 * nhtot(i) + 8 * nhtot(i - 1) - nhtot(i-2)) * h_mass;
            }
            if (std::abs(1.0 - pressure(i) / old_pressure) < 1e-7) {
                break;
            }
            const f64 ntotal = pressure(i) / (k_B * temperature(i));
            const f64 ne = ideal ? FP(0.0) : ne_from_ntot(ntotal, temperature(i));
            nhtot(i) = ntotal - ne;
        }
        if (iter == 100) {
            fmt::println("No converge: {}", iter);
        }
        y(i) = ne / nhtot(i);
    }

    F64Host rho("rho", sz.xc);
    F64Host eint("eint", sz.xc);
    for (int i = sz.ng; i < sz.xc; ++i) {
        rho(i) = nhtot(i) * h_mass;
        eint(i) = 1.0 / (eos.Gamma - 1.0) * pressure(i) + nhtot(i) * y(i) * chi_H;
    }

    {
    // NOTE(cmo): Needs to be self-consistent with the table
        auto rho_d = rho.createDeviceCopy();
        auto eint_d = eint.createDeviceCopy();
        auto y_d = y.createDeviceCopy();
        auto pressure_d = pressure.createDeviceCopy();
        const auto& Q = state.Q;
        const auto& eos = sim.eos;
        TabulatedLteH lte_h;
        lte_h.init(get_or<std::string>(config, "eos.table_path", "mosscap_lte_h_tables.nc"));
        using Cons = Cons<num_dim>;

        dex_parallel_for(
            "Setup Q",
            FlatLoop<3>(sz.zc, sz.yc, sz.xc),
            KOKKOS_LAMBDA (int k, int j, int i) {
                int ii = std::max(i, sz.ng);
                fp_t eint = eint_d(ii);
                const fp_t rho = rho_d(ii);
                fp_t starting_pressure;

                if (!ideal) {
                    int iter = 0;
                    const int max_iter = 1000;
                    for (iter = 0; iter < max_iter; ++iter) {
                        fp_t log_eint = std::log10(eint);
                        fp_t log_rho = std::log10(rho);
                        fp_t log_eint_rho = log_eint - log_rho;
                        auto s = lte_h.sample(log_eint_rho, log_rho);
                        fp_t T = std::pow(FP(10.0), s.log_T);
                        fp_t pressure = rho * (FP(1.0) + s.y) * (k_B / h_mass) * T;
                        if (iter == 0) {
                            starting_pressure = pressure;
                        }

                        fp_t pressure_err = pressure_d(ii) - pressure;
                        eint += FP(0.25) * pressure_err / (eos.Gamma - FP(1.0));

                        if (std::abs(pressure_err) / pressure < 1e-6) {
                            break;
                        }
                    }
                    if (ii % 20 == 0) {
                        printf("i: %d, num_iter %d %e->%e\n", ii, iter, starting_pressure, pressure_d(ii));
                    }
                    if (iter == max_iter) {
                        printf("Equalisation didn't converge (%d)\n", ii);
                    }
                }

                Q(I(Cons::Rho), k, j, i) = rho;
                Q(I(Cons::Ene), k, j, i) = eint;
                Q(I(Cons::MomX), k, j, i) = FP(0.0);
            }
        );
        Kokkos::fence();
    }

    setup_gravity(sim, config);
    OurWaveDriver driver{
        .start_time = get_or<fp_t>(config, "problem.drive_start", FP(0.0)),
        .period = get_or<fp_t>(config, "problem.period", FP(300.0)),
        .amplitude = get_or<fp_t>(config, "problem.amplitude", FP(0.01))
    };
    sim.user_bc = [=](const Simulation& sim){
        fill_one_bc_hse<0, num_dim>(sim, driver);
    };


}