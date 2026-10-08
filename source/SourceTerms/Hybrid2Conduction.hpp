#if !defined(MOSSCAP_HYBRID2_CONDUCTION_HPP)
#define MOSSCAP_HYBRID2_CONDUCTION_HPP
// hybrid2 hyperbolic (Cattaneo) thermal conduction, a method of the thermal_conduction source term
// (sources.thermal_conduction.method: hybrid2). Reference: HyperTcWork/standalone/hybrid2_conduction.py, which
// documents the scheme; HyperTcWork/writeup/hybrid2.pdf describes it.
//
//     de/dt = -div q,        dq_f/dt = (F_f - q_f) / tau_f
//
// q_f (State::heat_face, (num_dim, z, y, x)) is the energy flux through each cell's *lower* face along each axis.
// It is auxiliary RK state: the time steppers combine it like Q, with the rate in Sources::S_face.
// F_f blends the symmetric (Guenter) corner flux and the asymmetric face flux with a linear transverse mean,
// weighted by the local T contrast; each part is saturated on its own. tau_f pins the conduction-wave Courant
// number at f_wave.
//
// Coupling (unsplit, inside each RK stage, like STS): if S = dt / dt_diff <= s_max, this is one more right-hand
// side per stage. Otherwise each stage sub-integrates (e, q) over its dt_sub in ceil(dt_sub / dt_diff / s_max)
// SSP-RK3 substeps, and returns the increments as rates. (The substeps need an integrator that is stable for the
// conduction wave, whose eigenvalues are nearly imaginary at large S; forward Euler and SSP-RK2 are not.)
// NOTE(claude): Unsplit coupling was chosen for like-for-like comparison against STS. In the Python tests, Lie
// splitting (one conduction sub-integration after the hydro step) was equally accurate at half the conduction work,
// so it need not stay this way.
//
// Only the components of B along the simulated axes enter b (as the explicit/STS flux). At field nulls b = 0.
//
// Configuration (sources.thermal_conduction, with enable / anisotropic / saturate / saturation_phi / kappa0 /
// spitzer shared with the STS method):
//     method: hybrid2
//     hybrid2:
//       s_max: 10          # substep each RK stage when S = dt / dt_diff exceeds this (<= 0: never)
//       limit: true        # lim_nodiss applied-flux lower bound
//       box_radius: 0      # limiter box radius in cells (<= 0: per-step diffusion length)
//       alpha: 4           # tau floor in units of the step
//       f_wave: 0.8        # conduction-wave Courant number pinned by tau
//       r0: 1.5, r1: 3     # T_max / T_min thresholds of the symmetric / face-flux blend
// Needs fluid_type mhd or glmmhd. State::heat_face is written to the output and read on restart.
#include "ThermalConduction.hpp"
#include "ConductionCommon.hpp"
#include <limits>

namespace Mosscap {

constexpr fp_t H2_TINY = std::numeric_limits<fp_t>::min();
constexpr fp_t H2_HUGE = std::numeric_limits<fp_t>::max();
/// At a field null the averaged B is a rounding-level remainder of the cells' fields and its direction is noise:
/// where |mean B| <= H2_NULL_FRAC * mean |B|, b is set to zero (no conduction through that corner / face).
constexpr fp_t H2_NULL_FRAC = 1e-10_fp;

/// a(c[2], c[1], c[0]): c[0] = x (i), c[1] = y (j), c[2] = z (k).
KOKKOS_INLINE_FUNCTION fp_t& h2_at(const Fp3d& a, const int* c) {
    return a(c[2], c[1], c[0]);
}

/// Cell quantities, evaluated inline from W wherever needed (cheaper than storing them: the kernels are memory bound).
struct H2Cell {
    fp_t p;
    fp_t rho;
    fp_t T; // temperature
    fp_t K; // conductivity
    fp_t chi; // diffusivity K / (rho c_v), rho c_v = e / T
    fp_t qsat; // saturated flux 5 phi sqrt(p / rho) p
};

struct H2Params {
    fp_t m_p;
    fp_t kappa0;
    bool spitzer;
    bool saturate;
    fp_t phi;
};

template <typename FTraits>
KOKKOS_INLINE_FUNCTION fp_t h2_temperature(const Fp4d& W, const Eos& eos, fp_t m_p, const int* c) {
    using Prim = typename FTraits::prim;
    return temperature_si(
        W(I(Prim::Pres), c[2], c[1], c[0]),
        W(I(Prim::Rho), c[2], c[1], c[0]) / (eos.mass_per_h * m_p),
        eos.total_abund,
        ion_frac(eos, CellIndex{.i=c[0], .j=c[1], .k=c[2]})
    );
}

template <typename FTraits>
KOKKOS_INLINE_FUNCTION H2Cell h2_cell(const Fp4d& W, const Eos& eos, const H2Params& par, const int* c) {
    using Prim = typename FTraits::prim;
    H2Cell r;
    r.p = W(I(Prim::Pres), c[2], c[1], c[0]);
    r.rho = W(I(Prim::Rho), c[2], c[1], c[0]);
    r.T = h2_temperature<FTraits>(W, eos, par.m_p, c);
    r.K = par.kappa0;
    if (par.spitzer) {
        r.K *= square(r.T) * std::sqrt(r.T);
    }
    r.chi = r.K * r.T * (eos.gamma - 1.0_fp) / r.p;
    r.qsat = 5.0_fp * par.phi * std::sqrt(std::max(r.p, 0.0_fp) / r.rho) * r.p;
    return r;
}

/// Components of B along the simulated axes (as the explicit/STS flux); returns |B|.
template <typename FTraits>
KOKKOS_INLINE_FUNCTION fp_t h2_field(const Fp4d& W, const int* c, fp_t* B) {
    using Prim = typename FTraits::prim;
    constexpr int ND = FTraits::num_dim;
    B[0] = W(I(Prim::Bx), c[2], c[1], c[0]);
    B[1] = 0.0_fp;
    B[2] = 0.0_fp;
    if constexpr (ND > 1) {
        B[1] = W(I(Prim::By), c[2], c[1], c[0]);
    }
    if constexpr (ND > 2) {
        B[2] = W(I(Prim::Bz), c[2], c[1], c[0]);
    }
    return std::sqrt(square(B[0]) + square(B[1]) + square(B[2]));
}

inline H2Params h2_params(const Simulation& sim, const ThermalConductionContext& ctx) {
    return H2Params{
        .m_p = sim.state.p_mass,
        .kappa0 = ctx.kappa0,
        .spitzer = ctx.spitzer,
        .saturate = ctx.saturate,
        .phi = ctx.saturation_phi
    };
}

/// Interior extents (1 on inactive axes) and the first index on each axis (0 on inactive axes).
struct H2Grid {
    int ng;
    int n[3];
    int lo[3];
    bool periodic[3];
};

inline H2Grid h2_grid(const State& state, int num_dim) {
    const auto& sz = state.sz;
    const auto& bdry = state.boundaries;
    H2Grid g;
    g.ng = sz.ng;
    g.n[0] = sz.xc - 2 * sz.ng;
    g.n[1] = std::max(sz.yc - 2 * sz.ng, 1);
    g.n[2] = std::max(sz.zc - 2 * sz.ng, 1);
    for (int a = 0; a < 3; ++a) {
        g.lo[a] = (a < num_dim) ? sz.ng : 0;
    }
    g.periodic[0] = bdry.xs == BoundaryType::Periodic;
    g.periodic[1] = bdry.ys == BoundaryType::Periodic;
    g.periodic[2] = bdry.zs == BoundaryType::Periodic;
    return g;
}

struct Hybrid2Scratch {
    Fp4d cell; // [0] T, [1] K, [2] chi, [3] qsat of every cell (ghosts included), from W
    Fp4d qcb; // corner flux times b, per axis (corner (k, j, i) is the lower corner of cell (k, j, i))
    Fp3d wc; // corner blend weight
    Fp3d t_lo; // limiter: box minimum of T
    Fp3d t_tmp; // limiter: ping-pong buffer for the separable box minimum
    Fp3d r; // limiter: per-cell outflow factor (ghosts included)
};

inline Hybrid2Scratch allocate_hybrid2_scratch(const GridSize& sz, int num_dim, bool limit) {
    Hybrid2Scratch s;
    s.cell = Fp4d("h2_cell", 4, sz.zc, sz.yc, sz.xc);
    s.qcb = Fp4d("h2_qcb", num_dim, sz.zc, sz.yc, sz.xc);
    s.wc = Fp3d("h2_wc", sz.zc, sz.yc, sz.xc);
    if (limit) {
        s.t_lo = Fp3d("h2_t_lo", sz.zc, sz.yc, sz.xc);
        s.t_tmp = Fp3d("h2_t_tmp", sz.zc, sz.yc, sz.xc);
        s.r = Fp3d("h2_r", sz.zc, sz.yc, sz.xc);
    }
    return s;
}

/// Store T, K, chi and q_sat of every cell (ghosts included). (Storing them once is cheaper than evaluating them
/// inline in every stencil: the conductivity needs a pow and the temperature a division.)
template <typename FTraits>
void hybrid2_cell_quantities(const Simulation& sim, const ThermalConductionContext& ctx, const Hybrid2Scratch& s) {
    JasUnpack(sim, state, eos);
    JasUnpack(state, W, sz);
    const auto& cell = s.cell;
    const H2Params par = h2_params(sim, ctx);
    dex_parallel_for(
        "hybrid2 cell quantities",
        FlatLoop<3>(sz.zc, sz.yc, sz.xc),
        KOKKOS_LAMBDA (int k, int j, int i) {
            const int c[3] = {i, j, k};
            const H2Cell q = h2_cell<FTraits>(W, eos, par, c);
            cell(0, k, j, i) = q.T;
            cell(1, k, j, i) = q.K;
            cell(2, k, j, i) = q.chi;
            cell(3, k, j, i) = q.qsat;
        }
    );
}

/// Largest interior diffusivity (from the stored cell quantities) -> dt_diff = 0.8 dx^2 / (2 chi_max), the
/// yardstick for S.
inline fp_t hybrid2_dt_diff(const State& state, int num_dim, const Fp4d& cell) {
    const H2Grid g = h2_grid(state, num_dim);
    fp_t chi_max = 0.0_fp;
    dex_parallel_reduce(
        "hybrid2 chi max",
        FlatLoop<3>(g.n[2], g.n[1], g.n[0]),
        KOKKOS_LAMBDA (int ki, int ji, int ii, fp_t& running) {
            running = std::max(running, cell(2, ki + g.lo[2], ji + g.lo[1], ii + g.lo[0]));
        },
        Kokkos::Max<fp_t>(chi_max)
    );
    Kokkos::fence();
    return 0.8_fp * square(state.dx) / (2.0_fp * std::max(chi_max, H2_TINY));
}

/// Corner fluxes q_c b_c (symmetric operator) and blend weights, for the lower corner of every cell from the first
/// interior index to one past the last on each axis.
template <typename FTraits>
void hybrid2_corners(const Simulation& sim, const ThermalConductionContext& ctx, const Hybrid2Scratch& s) {
    constexpr int ND = FTraits::num_dim;
    constexpr int NC = 1 << ND;
    JasUnpack(sim, state, eos);
    JasUnpack(state, W, dx);
    JasUnpack(s, qcb, wc);
    const auto& cq = s.cell;
    const H2Params par = h2_params(sim, ctx);
    const H2Grid g = h2_grid(state, ND);
    const fp_t inv_dx = 1.0_fp / dx;
    const fp_t ln_r0 = std::log(ctx.h2_r0);
    const fp_t ln_r1 = std::log(ctx.h2_r1);
    dex_parallel_for(
        "hybrid2 corners",
        FlatLoop<3>(ND > 2 ? g.n[2] + 1 : 1, ND > 1 ? g.n[1] + 1 : 1, g.n[0] + 1),
        KOKKOS_LAMBDA (int kc, int jc, int ic) {
            const int c[3] = {ic + g.lo[0], jc + g.lo[1], kc + g.lo[2]};
            fp_t tmax = -H2_HUGE;
            fp_t tmin = H2_HUGE;
            fp_t B[3] = {0.0_fp, 0.0_fp, 0.0_fp};
            fp_t grad[3] = {0.0_fp, 0.0_fp, 0.0_fp};
            fp_t T_cell[NC];
            fp_t kmean = 0.0_fp;
            fp_t smean = 0.0_fp;
            fp_t b_abs = 0.0_fp;
            for (int m = 0; m < NC; ++m) {
                int cell[3] = {c[0], c[1], c[2]};
                for (int a = 0; a < ND; ++a) {
                    cell[a] -= (m >> a) & 1;
                }
                const fp_t t = cq(0, cell[2], cell[1], cell[0]);
                T_cell[m] = t;
                tmax = std::max(tmax, t);
                tmin = std::min(tmin, t);
                kmean += cq(1, cell[2], cell[1], cell[0]);
                smean += cq(3, cell[2], cell[1], cell[0]);
                fp_t Bc[3];
                b_abs += h2_field<FTraits>(W, cell, Bc);
                for (int a = 0; a < 3; ++a) {
                    B[a] += Bc[a];
                }
            }
            // gradient along a: the 2^(ND-1) cell pairs differing along a (bit a of m: 0 = upper cell)
            for (int a = 0; a < ND; ++a) {
                for (int m = 0; m < NC; ++m) {
                    if (((m >> a) & 1) == 0) {
                        grad[a] += T_cell[m] - T_cell[m | (1 << a)];
                    }
                }
                grad[a] *= inv_dx / fp_t(NC / 2);
            }
            kmean /= fp_t(NC);
            smean /= fp_t(NC);
            fp_t b_norm = std::sqrt(square(B[0]) + square(B[1]) + square(B[2]));
            if (b_norm <= H2_NULL_FRAC * b_abs) {
                B[0] = B[1] = B[2] = 0.0_fp;
                b_norm = 1.0_fp;
            }
            b_norm = std::max(b_norm, H2_TINY);
            fp_t b_dot_grad = 0.0_fp;
            for (int a = 0; a < ND; ++a) {
                b_dot_grad += B[a] / b_norm * grad[a];
            }
            fp_t qc = -kmean * b_dot_grad;
            if (par.saturate) {
                qc *= smean / (smean + std::abs(qc));
            }
            for (int a = 0; a < ND; ++a) {
                qcb(a, c[2], c[1], c[0]) = qc * B[a] / b_norm;
            }
            const fp_t lr = std::log(tmax / std::max(tmin, H2_TINY));
            h2_at(wc, c) = std::min(std::max((ln_r1 - lr) / (ln_r1 - ln_r0), 0.0_fp), 1.0_fp);
        }
    );
}

/// Fused: face targets and dq/dt for every axis, and (without the limiter) de/dt, in one pass over the cells whose
/// lower faces are interior faces. Writes out_q(q_base + a) and out_e(e_idx) (added to when accumulate).
template <typename FTraits>
void hybrid2_faces(
    const Simulation& sim, const ThermalConductionContext& ctx, const Fp4d& q, fp_t h, const Hybrid2Scratch& s,
    const Fp4d& out_e, int e_idx, const Fp4d& out_q, int q_base, bool accumulate, bool compute_de
) {
    constexpr int ND = FTraits::num_dim;
    constexpr int NC = 1 << ND;
    JasUnpack(sim, state, eos);
    JasUnpack(state, W, dx);
    JasUnpack(s, qcb, wc);
    const auto& cq = s.cell;
    const H2Params par = h2_params(sim, ctx);
    const H2Grid g = h2_grid(state, ND);
    const fp_t inv_dx = 1.0_fp / dx;
    const fp_t alpha_h = ctx.h2_alpha * h;
    const fp_t c_wave = ctx.h2_f_wave * dx / h;
    const fp_t acc = accumulate ? 1.0_fp : 0.0_fp;
    dex_parallel_for(
        "hybrid2 faces",
        FlatLoop<3>(ND > 2 ? g.n[2] + 1 : 1, ND > 1 ? g.n[1] + 1 : 1, g.n[0] + 1),
        KOKKOS_LAMBDA (int kf, int jf, int iff) {
            const int c[3] = {iff + g.lo[0], jf + g.lo[1], kf + g.lo[2]};
            bool interior_cell = true;
            for (int a = 0; a < ND; ++a) {
                interior_cell = interior_cell && (c[a] < g.lo[a] + g.n[a]);
            }
            using Prim = typename FTraits::prim;
            auto stored = [&] (const int* x) {
                H2Cell r;
                r.p = W(I(Prim::Pres), x[2], x[1], x[0]);
                r.rho = W(I(Prim::Rho), x[2], x[1], x[0]);
                r.T = cq(0, x[2], x[1], x[0]);
                r.K = cq(1, x[2], x[1], x[0]);
                r.chi = cq(2, x[2], x[1], x[0]);
                r.qsat = cq(3, x[2], x[1], x[0]);
                return r;
            };
            const H2Cell hc = stored(c);
            for (int ax = 0; ax < ND; ++ax) {
                bool valid = true;
                for (int b = 0; b < ND; ++b) {
                    if (b != ax) {
                        valid = valid && (c[b] < g.lo[b] + g.n[b]);
                    }
                }
                if (!valid) {
                    continue;
                }
                int cm[3] = {c[0], c[1], c[2]};
                cm[ax] -= 1;
                const H2Cell hm = stored(cm);

                // Symmetric part: mean of the face's 2^(ND-1) corners (offsets 0 / +1 along the other axes).
                fp_t f_sym = 0.0_fp;
                fp_t w = 0.0_fp;
                for (int m = 0; m < NC; ++m) {
                    if ((m >> ax) & 1) {
                        continue;
                    }
                    int corner[3] = {c[0], c[1], c[2]};
                    for (int b = 0; b < ND; ++b) {
                        if (b != ax) {
                            corner[b] += (m >> b) & 1;
                        }
                    }
                    f_sym += qcb(ax, corner[2], corner[1], corner[0]);
                    w += h2_at(wc, corner);
                }
                f_sym /= fp_t(NC / 2);
                w /= fp_t(NC / 2);

                // Asymmetric part: compact normal gradient, linear mean of the transverse differences.
                fp_t Bp[3];
                fp_t Bm[3];
                const fp_t b_abs = 0.5_fp * (h2_field<FTraits>(W, c, Bp) + h2_field<FTraits>(W, cm, Bm));
                fp_t B[3] = {0.5_fp * (Bp[0] + Bm[0]), 0.5_fp * (Bp[1] + Bm[1]), 0.5_fp * (Bp[2] + Bm[2])};
                fp_t b_norm = std::sqrt(square(B[0]) + square(B[1]) + square(B[2]));
                if (b_norm <= H2_NULL_FRAC * b_abs) {
                    B[0] = B[1] = B[2] = 0.0_fp;
                    b_norm = 1.0_fp;
                }
                b_norm = std::max(b_norm, 1e-20_fp);
                fp_t b_dot_grad = B[ax] * (hc.T - hm.T) * inv_dx;
                for (int b = 0; b < ND; ++b) {
                    if (b == ax) {
                        continue;
                    }
                    int up[3] = {c[0], c[1], c[2]};
                    int dn[3] = {c[0], c[1], c[2]};
                    int mup[3] = {cm[0], cm[1], cm[2]};
                    int mdn[3] = {cm[0], cm[1], cm[2]};
                    up[b] += 1;
                    dn[b] -= 1;
                    mup[b] += 1;
                    mdn[b] -= 1;
                    const fp_t tr = 0.25_fp * (
                        (cq(0, up[2], up[1], up[0]) - cq(0, dn[2], dn[1], dn[0]))
                        + (cq(0, mup[2], mup[1], mup[0]) - cq(0, mdn[2], mdn[1], mdn[0]))
                    ) * inv_dx;
                    b_dot_grad += B[b] * tr;
                }
                b_dot_grad /= b_norm;
                const fp_t kappa_f = 0.5_fp * (hc.K + hm.K);
                fp_t f_asym = -kappa_f * b_dot_grad * B[ax] / b_norm;
                fp_t sat_fac = 1.0_fp;
                if (par.saturate) {
                    const fp_t p_up = (f_asym > 0.0_fp) ? hm.p : (f_asym < 0.0_fp) ? hc.p : 0.5_fp * (hc.p + hm.p);
                    const fp_t rho_f = 0.5_fp * (hc.rho + hm.rho);
                    const fp_t s_flux = 5.0_fp * par.phi * std::sqrt(std::max(p_up, 0.0_fp) / rho_f) * p_up;
                    sat_fac = s_flux / (s_flux + std::abs(kappa_f * b_dot_grad));
                    f_asym *= sat_fac;
                }

                const fp_t target = w * f_sym + (1.0_fp - w) * f_asym;
                const fp_t chi_f = 0.5_fp * (hc.chi + hm.chi);
                fp_t tau = std::max(alpha_h, chi_f / square(c_wave));
                if (par.saturate) {
                    // saturated conductivity in tau, floored at alpha h
                    tau = std::max(alpha_h, tau * sat_fac);
                }
                fp_t& dst = out_q(q_base + ax, c[2], c[1], c[0]);
                dst = acc * dst + (target - q(ax, c[2], c[1], c[0])) / tau;
            }
            if (compute_de && interior_cell) {
                fp_t div = 0.0_fp;
                for (int a = 0; a < ND; ++a) {
                    int up[3] = {c[0], c[1], c[2]};
                    up[a] += 1;
                    div += q(a, up[2], up[1], up[0]) - q(a, c[2], c[1], c[0]);
                }
                fp_t& dst = out_e(e_idx, c[2], c[1], c[0]);
                dst = acc * dst - div * inv_dx;
            }
        }
    );
}

/// Ghost fill of a cell scalar along one axis: periodic copy, otherwise the edge cell.
template <typename FTraits>
void hybrid2_fill_scalar(const State& state, const Fp3d& u, int ax) {
    const auto& sz = state.sz;
    const H2Grid g = h2_grid(state, FTraits::num_dim);
    const int ng = sz.ng;
    dex_parallel_for(
        "hybrid2 scalar ghosts",
        FlatLoop<3>(sz.zc, sz.yc, sz.xc),
        KOKKOS_LAMBDA (int k, int j, int i) {
            int c[3] = {i, j, k};
            if (c[ax] >= ng && c[ax] < ng + g.n[ax]) {
                return;
            }
            int src[3] = {i, j, k};
            if (g.periodic[ax]) {
                src[ax] += (c[ax] < ng) ? g.n[ax] : -g.n[ax];
            } else {
                src[ax] = (c[ax] < ng) ? ng : ng + g.n[ax] - 1;
            }
            h2_at(u, c) = h2_at(u, src);
        }
    );
}

/// lim_nodiss: per-cell outflow factor r = min(1, B / D) such that no interior cell falls below the minimum T over
/// the (2R+1)^ND box around it in a forward-Euler stage: B = e - (rho c_v) T_lo = e (1 - T_lo / T), D = h * outflow
/// / dx. R = box_radius, or the per-step diffusion length ceil(sqrt(2 chi_max h) / dx). The box minimum is
/// separable: per axis, passes of radius <= ng (the first reading T straight from W), with a ghost fill between
/// passes when R > ng (beyond the ghosts of a non-periodic boundary the edge value is repeated). r is written on
/// ghosts too: the periodic image, 1 elsewhere (inflow from outside is not limited).
template <typename FTraits>
void hybrid2_limit_factor(const Simulation& sim, const ThermalConductionContext& ctx, const Fp4d& q, fp_t h,
                          Hybrid2Scratch& s) {
    using Prim = typename FTraits::prim;
    constexpr int ND = FTraits::num_dim;
    JasUnpack(sim, state, eos);
    JasUnpack(state, W, sz, dx);
    const H2Params par = h2_params(sim, ctx);
    const H2Grid g = h2_grid(state, ND);
    const int ng = g.ng;
    const auto& cq = s.cell;

    int R = ctx.h2_box_radius;
    if (R <= 0) {
        const fp_t chi_max = 0.8_fp * square(dx) / (2.0_fp * hybrid2_dt_diff(state, ND, s.cell));
        R = std::max(1, int(std::ceil(std::sqrt(2.0_fp * chi_max * h) / dx)));
    }

    Fp3d cur = s.t_lo;
    Fp3d nxt = s.t_tmp;
    bool from_W = true;
    for (int ax = 0; ax < ND; ++ax) {
        int remaining = R;
        while (remaining > 0) {
            const int rad = std::min(remaining, ng);
            const Fp3d src = cur;
            const Fp3d dst = nxt;
            const bool read_W = from_W;
            dex_parallel_for(
                "hybrid2 box min pass",
                FlatLoop<3>(sz.zc, sz.yc, sz.xc),
                KOKKOS_LAMBDA (int k, int j, int i) {
                    const int c[3] = {i, j, k};
                    if (c[ax] < ng || c[ax] >= ng + g.n[ax]) {
                        return;
                    }
                    // the first pass reads T from W, so needs all other-axis indices valid in W (ghosts filled)
                    fp_t m = read_W ? cq(0, k, j, i) : h2_at(src, c);
                    for (int o = 1; o <= rad; ++o) {
                        int a[3] = {i, j, k};
                        int b[3] = {i, j, k};
                        a[ax] += o;
                        b[ax] -= o;
                        const fp_t ta = read_W ? cq(0, a[2], a[1], a[0]) : h2_at(src, a);
                        const fp_t tb = read_W ? cq(0, b[2], b[1], b[0]) : h2_at(src, b);
                        m = std::min(m, std::min(ta, tb));
                    }
                    h2_at(dst, c) = m;
                }
            );
            from_W = false;
            remaining -= rad;
            std::swap(cur, nxt);
            if (remaining > 0) {
                hybrid2_fill_scalar<FTraits>(state, cur, ax);
            }
        }
    }
    const Fp3d t_lo = cur;
    const auto& r = s.r;

    dex_parallel_for(
        "hybrid2 limiter factor",
        FlatLoop<3>(sz.zc, sz.yc, sz.xc),
        KOKKOS_LAMBDA (int k, int j, int i) {
            int c[3] = {i, j, k};
            for (int a = 0; a < ND; ++a) {
                if (c[a] < ng || c[a] >= ng + g.n[a]) {
                    if (!g.periodic[a]) {
                        r(k, j, i) = 1.0_fp;
                        return;
                    }
                    c[a] += (c[a] < ng) ? g.n[a] : -g.n[a];
                    if (c[a] < ng || c[a] >= ng + g.n[a]) {
                        r(k, j, i) = 1.0_fp; // deeper than one period of ghosts: unused
                        return;
                    }
                }
            }
            const fp_t T = cq(0, c[2], c[1], c[0]);
            const fp_t e = W(I(Prim::Pres), c[2], c[1], c[0]) / (eos.gamma - 1.0_fp);
            const fp_t budget = std::max(e * (1.0_fp - h2_at(t_lo, c) / T), 0.0_fp);
            fp_t out = 0.0_fp;
            for (int a = 0; a < ND; ++a) {
                int up[3] = {c[0], c[1], c[2]};
                up[a] += 1;
                out += std::max(-q(a, c[2], c[1], c[0]), 0.0_fp) + std::max(q(a, up[2], up[1], up[0]), 0.0_fp);
            }
            const fp_t D = h * out / dx;
            r(k, j, i) = (D > budget) ? budget / D : 1.0_fp;
        }
    );
}

/// de/dt from the limited fluxes: each face scaled by the factor r of the cell it drains (the cell below the face if
/// q > 0, else the cell above); q itself is untouched.
template <typename FTraits>
void hybrid2_limited_energy(const Simulation& sim, const Fp4d& q, const Hybrid2Scratch& s, const Fp4d& out_e,
                            int e_idx, bool accumulate) {
    constexpr int ND = FTraits::num_dim;
    JasUnpack(sim, state);
    const H2Grid g = h2_grid(state, ND);
    const fp_t inv_dx = 1.0_fp / state.dx;
    const fp_t acc = accumulate ? 1.0_fp : 0.0_fp;
    const auto& r = s.r;
    dex_parallel_for(
        "hybrid2 limited energy rate",
        FlatLoop<3>(g.n[2], g.n[1], g.n[0]),
        KOKKOS_LAMBDA (int ki, int ji, int ii) {
            const int c[3] = {ii + g.lo[0], ji + g.lo[1], ki + g.lo[2]};
            fp_t div = 0.0_fp;
            for (int a = 0; a < ND; ++a) {
                int up[3] = {c[0], c[1], c[2]};
                int lo[3] = {c[0], c[1], c[2]};
                up[a] += 1;
                lo[a] -= 1;
                fp_t q_lo = q(a, c[2], c[1], c[0]);
                fp_t q_up = q(a, up[2], up[1], up[0]);
                q_lo *= (q_lo > 0.0_fp) ? h2_at(r, lo) : h2_at(r, c);
                q_up *= (q_up > 0.0_fp) ? h2_at(r, c) : h2_at(r, up);
                div += q_up - q_lo;
            }
            fp_t& dst = out_e(e_idx, c[2], c[1], c[0]);
            dst = acc * dst - div * inv_dx;
        }
    );
}

/// One evaluation of the right-hand side into (out_e, out_q). Needs hybrid2_cell_quantities for the same W.
template <typename FTraits>
void hybrid2_rhs(const Simulation& sim, const ThermalConductionContext& ctx, const Fp4d& q, fp_t h, Hybrid2Scratch& s,
                 const Fp4d& out_e, int e_idx, const Fp4d& out_q, int q_base, bool accumulate) {
    hybrid2_corners<FTraits>(sim, ctx, s);
    hybrid2_faces<FTraits>(sim, ctx, q, h, s, out_e, e_idx, out_q, q_base, accumulate, !ctx.h2_limit);
    if (ctx.h2_limit) {
        hybrid2_limit_factor<FTraits>(sim, ctx, q, h, s);
        hybrid2_limited_energy<FTraits>(sim, q, s, out_e, e_idx, accumulate);
    }
}

template <typename FTraits>
void hybrid2_conduction_kernel(const Simulation& sim, const ThermalConductionContext& ctx) {
    using Prim = typename FTraits::prim;
    using Cons = typename FTraits::cons;
    constexpr int ND = FTraits::num_dim;
    if constexpr (!FTraits::is_mhd || FTraits::has_hypertc) {
        throw std::runtime_error("hybrid2 conduction needs an MHD fluid without HeatF (mhd / glmmhd)");
    } else {
        JasUnpack(sim, state, eos, sources, dt, dt_sub);
        JasUnpack(state, W, sz, heat_face);
        const auto& S = sources.S;
        const auto& S_face = sources.S_face;
        const H2Grid g = h2_grid(state, ND);
        const int ng = g.ng;

        Hybrid2Scratch s = allocate_hybrid2_scratch(sz, ND, ctx.h2_limit);

        // S of the whole step, from the stage state.
        hybrid2_cell_quantities<FTraits>(sim, ctx, s);
        const fp_t dt_diff = hybrid2_dt_diff(state, ND, s.cell);
        const bool substep = ctx.h2_s_max > 0.0_fp && dt / dt_diff > ctx.h2_s_max;

        if (!substep) {
            // One more right-hand side in this stage, straight into the sources: the outer RK integrates it.
            hybrid2_rhs<FTraits>(sim, ctx, heat_face, dt, s, S, I(Cons::Ene), S_face, 0, true);
            Kokkos::fence();
            return;
        }

        // Sub-integrate (e, q) over this stage's dt_sub in n SSP-RK3 substeps, with rho, v, B frozen.
        const int n_sub = std::max(1, int(std::ceil(dt_sub / dt_diff / ctx.h2_s_max - 1e-12_fp)));
        const fp_t h = dt_sub / fp_t(n_sub);
        if (n_sub > 150) {
            fmt::println("hybrid2: {} conduction substeps in one stage (S = {:.1f})", n_sub, dt_sub / dt_diff);
        }
        Fp3d e0("h2_e0", sz.zc, sz.yc, sz.xc); // start of the stage, ghosts included (to restore W)
        Fp3d e_n("h2_e_n", sz.zc, sz.yc, sz.xc); // start of the substep
        Fp4d q("h2_q", ND, sz.zc, sz.yc, sz.xc);
        Fp4d q_n("h2_q_n", ND, sz.zc, sz.yc, sz.xc);
        Fp4d rates("h2_rates", ND + 1, sz.zc, sz.yc, sz.xc); // [0] = de/dt, [1 + a] = dq_a/dt
        dex_parallel_for(
            "hybrid2 substep init",
            FlatLoop<3>(sz.zc, sz.yc, sz.xc),
            KOKKOS_LAMBDA (int k, int j, int i) {
                e0(k, j, i) = W(I(Prim::Pres), k, j, i) / (eos.gamma - 1.0_fp);
                for (int a = 0; a < ND; ++a) {
                    q(a, k, j, i) = heat_face(a, k, j, i);
                }
            }
        );

        // Shu-Osher SSP-RK3: u = c_n u_n + c_u (u + h L(u)); e lives in W's pressure between stages.
        constexpr fp_t c_n[3] = {0.0_fp, 0.75_fp, 1.0_fp / 3.0_fp};
        constexpr fp_t c_u[3] = {1.0_fp, 0.25_fp, 2.0_fp / 3.0_fp};
        const fp_t gm1 = eos.gamma - 1.0_fp;
        for (int sub = 0; sub < n_sub; ++sub) {
            dex_parallel_for(
                "hybrid2 substep save",
                FlatLoop<3>(sz.zc, sz.yc, sz.xc),
                KOKKOS_LAMBDA (int k, int j, int i) {
                    e_n(k, j, i) = W(I(Prim::Pres), k, j, i) / gm1;
                    for (int a = 0; a < ND; ++a) {
                        q_n(a, k, j, i) = q(a, k, j, i);
                    }
                }
            );
            for (int stage = 0; stage < 3; ++stage) {
                if (sub > 0 || stage > 0) { // the stage's starting state is already in scratch for the first one
                    hybrid2_cell_quantities<FTraits>(sim, ctx, s);
                }
                hybrid2_rhs<FTraits>(sim, ctx, q, h, s, rates, 0, rates, 1, false);
                const fp_t cn = c_n[stage];
                const fp_t cu = c_u[stage];
                // fused update of the interior cells and faces; periodic upper boundary faces copy their twin
                dex_parallel_for(
                    "hybrid2 substep stage",
                    FlatLoop<3>(ND > 2 ? g.n[2] + 1 : 1, ND > 1 ? g.n[1] + 1 : 1, g.n[0] + 1),
                    KOKKOS_LAMBDA (int kf, int jf, int iff) {
                        const int c[3] = {iff + g.lo[0], jf + g.lo[1], kf + g.lo[2]};
                        bool interior_cell = true;
                        for (int a = 0; a < ND; ++a) {
                            interior_cell = interior_cell && (c[a] < g.lo[a] + g.n[a]);
                        }
                        if (interior_cell) {
                            const fp_t e = W(I(Prim::Pres), c[2], c[1], c[0]) / gm1;
                            const fp_t e_new = cn * e_n(c[2], c[1], c[0]) + cu * (e + h * rates(0, c[2], c[1], c[0]));
                            W(I(Prim::Pres), c[2], c[1], c[0]) = e_new * gm1;
                        }
                        for (int ax = 0; ax < ND; ++ax) {
                            bool valid = true;
                            for (int b = 0; b < ND; ++b) {
                                if (b != ax) {
                                    valid = valid && (c[b] < g.lo[b] + g.n[b]);
                                }
                            }
                            if (!valid || (g.periodic[ax] && c[ax] == ng + g.n[ax])) {
                                continue;
                            }
                            fp_t& qa = q(ax, c[2], c[1], c[0]);
                            qa = cn * q_n(ax, c[2], c[1], c[0]) + cu * (qa + h * rates(1 + ax, c[2], c[1], c[0]));
                            if (g.periodic[ax] && c[ax] == ng) {
                                int twin[3] = {c[0], c[1], c[2]};
                                twin[ax] += g.n[ax];
                                q(ax, twin[2], twin[1], twin[0]) = qa;
                            }
                        }
                    }
                );
                fill_sts_pressure_ghosts<FTraits>(sim);
            }
        }

        const fp_t inv_dt_sub = 1.0_fp / dt_sub;
        dex_parallel_for(
            "hybrid2 substep increments",
            FlatLoop<3>(sz.zc, sz.yc, sz.xc),
            KOKKOS_LAMBDA (int k, int j, int i) {
                const int c[3] = {i, j, k};
                bool interior = true;
                for (int a = 0; a < ND; ++a) {
                    interior = interior && (c[a] >= ng && c[a] < ng + g.n[a]);
                }
                if (interior) {
                    S(I(Cons::Ene), k, j, i) += (W(I(Prim::Pres), k, j, i) / gm1 - e0(k, j, i)) * inv_dt_sub;
                }
                for (int a = 0; a < ND; ++a) {
                    S_face(a, k, j, i) += (q(a, k, j, i) - heat_face(a, k, j, i)) * inv_dt_sub;
                }
                // restore the stage pressure for the other source terms
                W(I(Prim::Pres), k, j, i) = e0(k, j, i) * gm1;
            }
        );
        Kokkos::fence();
    }
}

/// On periodic axes the upper boundary face is the lower boundary face: copy it so the two stay identical.
template <typename FTraits>
void hybrid2_sync_periodic_faces(const State& state, const Fp4d& q) {
    constexpr int ND = FTraits::num_dim;
    const auto& sz = state.sz;
    const H2Grid g = h2_grid(state, ND);
    const int ng = g.ng;
    dex_parallel_for(
        "hybrid2 periodic faces",
        FlatLoop<4>(ND, sz.zc, sz.yc, sz.xc),
        KOKKOS_LAMBDA (int ax, int k, int j, int i) {
            int c[3] = {i, j, k};
            if (!g.periodic[ax] || c[ax] != ng) {
                return;
            }
            c[ax] += g.n[ax];
            q(ax, c[2], c[1], c[0]) = q(ax, k, j, i);
        }
    );
    Kokkos::fence();
}

}

#else
#endif
