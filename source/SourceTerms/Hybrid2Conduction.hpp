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
// Only the components of B along the simulated axes enter b (as the explicit/STS flux).
#include "ThermalConduction.hpp"
#include "ConductionCommon.hpp"
#include <limits>

namespace Mosscap {

struct Hybrid2Scratch {
    Fp3d T; // temperature
    Fp3d K; // conductivity
    Fp3d chi; // diffusivity K / (rho c_v)
    Fp3d qsat; // saturated flux 5 phi sqrt(p / rho) p
    Fp4d qcb; // corner flux times b, per axis (corner (k, j, i) is the lower corner of cell (k, j, i))
    Fp3d wc; // corner blend weight
    Fp4d de_dq; // [0] = de/dt (cells), [1 + a] = dq_a/dt (lower faces)
};

inline Hybrid2Scratch allocate_hybrid2_scratch(const GridSize& sz, int num_dim) {
    Hybrid2Scratch s;
    s.T = Fp3d("h2_T", sz.zc, sz.yc, sz.xc);
    s.K = Fp3d("h2_K", sz.zc, sz.yc, sz.xc);
    s.chi = Fp3d("h2_chi", sz.zc, sz.yc, sz.xc);
    s.qsat = Fp3d("h2_qsat", sz.zc, sz.yc, sz.xc);
    s.qcb = Fp4d("h2_qcb", num_dim, sz.zc, sz.yc, sz.xc);
    s.wc = Fp3d("h2_wc", sz.zc, sz.yc, sz.xc);
    s.de_dq = Fp4d("h2_rates", num_dim + 1, sz.zc, sz.yc, sz.xc);
    s.de_dq = 0.0_fp; // slots outside the interior cells / faces are never written
    return s;
}

constexpr fp_t H2_TINY = std::numeric_limits<fp_t>::min();
constexpr fp_t H2_HUGE = std::numeric_limits<fp_t>::max();

/// a(c[2], c[1], c[0]): c[0] = x (i), c[1] = y (j), c[2] = z (k).
KOKKOS_INLINE_FUNCTION fp_t& h2_at(const Fp3d& a, const int* c) {
    return a(c[2], c[1], c[0]);
}

/// Temperature, conductivity, diffusivity and saturated flux of every cell (ghosts included) from W.
template <typename FTraits>
void hybrid2_cell_quantities(const Simulation& sim, const ThermalConductionContext& ctx, const Hybrid2Scratch& s) {
    using Prim = typename FTraits::prim;
    JasUnpack(sim, state, eos);
    JasUnpack(state, W, sz);
    JasUnpack(s, T, K, chi, qsat);
    const fp_t m_p = state.p_mass;
    const fp_t phi = ctx.saturation_phi;
    const fp_t kappa0 = ctx.kappa0;
    const bool spitzer = ctx.spitzer;
    dex_parallel_for(
        "hybrid2 cell quantities",
        FlatLoop<3>(sz.zc, sz.yc, sz.xc),
        KOKKOS_LAMBDA (int k, int j, int i) {
            const fp_t p = W(I(Prim::Pres), k, j, i);
            const fp_t rho = W(I(Prim::Rho), k, j, i);
            const fp_t temp = temperature_si(
                p, rho / (eos.mass_per_h * m_p), eos.total_abund, ion_frac(eos, CellIndex{.i=i, .j=j, .k=k})
            );
            fp_t kappa = kappa0;
            if (spitzer) {
                kappa *= square(temp) * std::sqrt(temp);
            }
            T(k, j, i) = temp;
            K(k, j, i) = kappa;
            // rho c_v = e / T, e = p / (gamma - 1)
            chi(k, j, i) = kappa * temp * (eos.gamma - 1.0_fp) / p;
            qsat(k, j, i) = 5.0_fp * phi * std::sqrt(std::max(p, 0.0_fp) / rho) * p;
        }
    );
    Kokkos::fence();
}

/// Right-hand side of hybrid2 for the state in W (pressure, density, B; ghosts filled) and the face fluxes q, with
/// h the step that sets tau. Needs hybrid2_cell_quantities for the same W. Writes de/dt (interior cells) and dq/dt
/// (interior faces) into scratch.de_dq.
template <typename FTraits>
void hybrid2_rhs(const Simulation& sim, const ThermalConductionContext& ctx, const Fp4d& q, fp_t h, const Hybrid2Scratch& s) {
    using Prim = typename FTraits::prim;
    constexpr int ND = FTraits::num_dim;
    constexpr int NC = 1 << ND;
    JasUnpack(sim, state);
    JasUnpack(state, W, sz, dx);
    const int ng = sz.ng;
    const int nx = sz.xc - 2 * ng;
    const int ny = std::max(sz.yc - 2 * ng, 1);
    const int nz = std::max(sz.zc - 2 * ng, 1);
    JasUnpack(s, T, K, chi, qsat, qcb, wc, de_dq);
    const bool saturate = ctx.saturate;
    const fp_t phi = ctx.saturation_phi;
    const fp_t inv_dx = 1.0_fp / dx;
    const fp_t ln_r0 = std::log(ctx.h2_r0);
    const fp_t ln_r1 = std::log(ctx.h2_r1);
    const fp_t alpha_h = ctx.h2_alpha * h;
    const fp_t c_wave = ctx.h2_f_wave * dx / h;

    // Corners: the lower corner of every cell from the first interior index to one past the last, on each axis.
    dex_parallel_for(
        "hybrid2 corners",
        FlatLoop<3>(ND > 2 ? nz + 1 : 1, ND > 1 ? ny + 1 : 1, nx + 1),
        KOKKOS_LAMBDA (int kc, int jc, int ic) {
            const int c[3] = {ic + ng, ND > 1 ? jc + ng : 0, ND > 2 ? kc + ng : 0};
            fp_t tmax = -H2_HUGE;
            fp_t tmin = H2_HUGE;
            fp_t B[3] = {0.0_fp, 0.0_fp, 0.0_fp};
            fp_t grad[3] = {0.0_fp, 0.0_fp, 0.0_fp};
            fp_t kmean = 0.0_fp;
            fp_t smean = 0.0_fp;
            for (int m = 0; m < NC; ++m) {
                int cell[3] = {c[0], c[1], c[2]};
                for (int a = 0; a < ND; ++a) {
                    cell[a] -= (m >> a) & 1;
                }
                const fp_t t = h2_at(T, cell);
                tmax = std::max(tmax, t);
                tmin = std::min(tmin, t);
                kmean += h2_at(K, cell);
                smean += h2_at(qsat, cell);
                B[0] += W(I(Prim::Bx), cell[2], cell[1], cell[0]);
                if constexpr (ND > 1) {
                    B[1] += W(I(Prim::By), cell[2], cell[1], cell[0]);
                }
                if constexpr (ND > 2) {
                    B[2] += W(I(Prim::Bz), cell[2], cell[1], cell[0]);
                }
                for (int a = 0; a < ND; ++a) {
                    if (((m >> a) & 1) == 0) {
                        int lower[3] = {cell[0], cell[1], cell[2]};
                        lower[a] -= 1;
                        grad[a] += t - h2_at(T, lower);
                    }
                }
            }
            kmean /= fp_t(NC);
            smean /= fp_t(NC);
            const fp_t b_norm = std::max(std::sqrt(square(B[0]) + square(B[1]) + square(B[2])), H2_TINY);
            fp_t b_dot_grad = 0.0_fp;
            for (int a = 0; a < ND; ++a) {
                grad[a] *= inv_dx / fp_t(NC / 2);
                b_dot_grad += B[a] / b_norm * grad[a];
            }
            fp_t qc = -kmean * b_dot_grad;
            if (saturate) {
                qc *= smean / (smean + std::abs(qc));
            }
            for (int a = 0; a < ND; ++a) {
                qcb(a, c[2], c[1], c[0]) = qc * B[a] / b_norm;
            }
            const fp_t lr = std::log(tmax / std::max(tmin, H2_TINY));
            h2_at(wc, c) = std::min(std::max((ln_r1 - lr) / (ln_r1 - ln_r0), 0.0_fp), 1.0_fp);
        }
    );
    Kokkos::fence();

    // Faces along each axis: the lower faces of interior cells plus the upper boundary face.
    for (int ax = 0; ax < ND; ++ax) {
        int ext[3] = {nx, ny, nz};
        ext[ax] += 1;
        dex_parallel_for(
            "hybrid2 face target",
            FlatLoop<3>(ext[2], ext[1], ext[0]),
            KOKKOS_LAMBDA (int kf, int jf, int iff) {
                const int c[3] = {iff + ng, ND > 1 ? jf + ng : 0, ND > 2 ? kf + ng : 0};
                int cm[3] = {c[0], c[1], c[2]};
                cm[ax] -= 1;

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
                fp_t B[3] = {0.0_fp, 0.0_fp, 0.0_fp};
                B[0] = 0.5_fp * (W(I(Prim::Bx), c[2], c[1], c[0]) + W(I(Prim::Bx), cm[2], cm[1], cm[0]));
                if constexpr (ND > 1) {
                    B[1] = 0.5_fp * (W(I(Prim::By), c[2], c[1], c[0]) + W(I(Prim::By), cm[2], cm[1], cm[0]));
                }
                if constexpr (ND > 2) {
                    B[2] = 0.5_fp * (W(I(Prim::Bz), c[2], c[1], c[0]) + W(I(Prim::Bz), cm[2], cm[1], cm[0]));
                }
                const fp_t b_norm = std::max(std::sqrt(square(B[0]) + square(B[1]) + square(B[2])), 1e-20_fp);
                fp_t b_dot_grad = B[ax] * (h2_at(T, c) - h2_at(T, cm)) * inv_dx;
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
                        (h2_at(T, up) - h2_at(T, dn)) + (h2_at(T, mup) - h2_at(T, mdn))
                    ) * inv_dx;
                    b_dot_grad += B[b] * tr;
                }
                b_dot_grad /= b_norm;
                const fp_t kappa_f = 0.5_fp * (h2_at(K, c) + h2_at(K, cm));
                fp_t f_asym = -kappa_f * b_dot_grad * B[ax] / b_norm;
                fp_t sat_fac = 1.0_fp;
                if (saturate) {
                    const fp_t pc = W(I(Prim::Pres), c[2], c[1], c[0]);
                    const fp_t pm = W(I(Prim::Pres), cm[2], cm[1], cm[0]);
                    const fp_t p_up = (f_asym > 0.0_fp) ? pm : (f_asym < 0.0_fp) ? pc : 0.5_fp * (pc + pm);
                    const fp_t rho_f = 0.5_fp * (W(I(Prim::Rho), c[2], c[1], c[0]) + W(I(Prim::Rho), cm[2], cm[1], cm[0]));
                    const fp_t s_flux = 5.0_fp * phi * std::sqrt(std::max(p_up, 0.0_fp) / rho_f) * p_up;
                    sat_fac = s_flux / (s_flux + std::abs(kappa_f * b_dot_grad));
                    f_asym *= sat_fac;
                }

                const fp_t target = w * f_sym + (1.0_fp - w) * f_asym;
                const fp_t chi_f = 0.5_fp * (h2_at(chi, c) + h2_at(chi, cm));
                fp_t tau = std::max(alpha_h, chi_f / square(c_wave));
                if (saturate) {
                    // saturated conductivity in tau, floored at alpha h
                    tau = std::max(alpha_h, tau * sat_fac);
                }
                de_dq(1 + ax, c[2], c[1], c[0]) = (target - q(ax, c[2], c[1], c[0])) / tau;
            }
        );
    }
    Kokkos::fence();

    dex_parallel_for(
        "hybrid2 energy rate",
        FlatLoop<3>(nz, ny, nx),
        KOKKOS_LAMBDA (int ki, int ji, int ii) {
            const int k = nz == 1 ? ki : ki + ng;
            const int j = ny == 1 ? ji : ji + ng;
            const int i = ii + ng;
            fp_t div = q(0, k, j, i + 1) - q(0, k, j, i);
            if constexpr (ND > 1) {
                div += q(1, k, j + 1, i) - q(1, k, j, i);
            }
            if constexpr (ND > 2) {
                div += q(2, k + 1, j, i) - q(2, k, j, i);
            }
            de_dq(0, k, j, i) = -div * inv_dx;
        }
    );
    Kokkos::fence();
}

/// On periodic axes the upper boundary face is the lower boundary face: copy it so the two stay identical.
template <typename FTraits>
void hybrid2_sync_periodic_faces(const State& state, const Fp4d& q) {
    constexpr int ND = FTraits::num_dim;
    const auto& sz = state.sz;
    const auto& bdry = state.boundaries;
    const int ng = sz.ng;
    const int n[3] = {sz.xc - 2 * ng, std::max(sz.yc - 2 * ng, 1), std::max(sz.zc - 2 * ng, 1)};
    const bool periodic[3] = {
        bdry.xs == BoundaryType::Periodic,
        bdry.ys == BoundaryType::Periodic,
        bdry.zs == BoundaryType::Periodic
    };
    for (int ax = 0; ax < ND; ++ax) {
        if (!periodic[ax]) {
            continue;
        }
        dex_parallel_for(
            "hybrid2 periodic faces",
            FlatLoop<3>(sz.zc, sz.yc, sz.xc),
            KOKKOS_LAMBDA (int k, int j, int i) {
                int c[3] = {i, j, k};
                if (c[ax] != ng) {
                    return;
                }
                int u[3] = {i, j, k};
                u[ax] = ng + n[ax];
                q(ax, u[2], u[1], u[0]) = q(ax, k, j, i);
            }
        );
    }
    Kokkos::fence();
}

/// Largest interior diffusivity -> dt_diff = 0.8 dx^2 / (2 chi_max), the yardstick for S.
inline fp_t hybrid2_dt_diff(const State& state, const Fp3d& chi) {
    const auto& sz = state.sz;
    const int ng = sz.ng;
    const int nx = sz.xc - 2 * ng;
    const int ny = std::max(sz.yc - 2 * ng, 1);
    const int nz = std::max(sz.zc - 2 * ng, 1);
    fp_t chi_max = 0.0_fp;
    dex_parallel_reduce(
        "hybrid2 chi max",
        FlatLoop<3>(nz, ny, nx),
        KOKKOS_LAMBDA (int ki, int ji, int ii, fp_t& running) {
            const int k = nz == 1 ? ki : ki + ng;
            const int j = ny == 1 ? ji : ji + ng;
            const int i = ii + ng;
            running = std::max(running, chi(k, j, i));
        },
        Kokkos::Max<fp_t>(chi_max)
    );
    Kokkos::fence();
    return 0.8_fp * square(state.dx) / (2.0_fp * std::max(chi_max, H2_TINY));
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
        const int ng = sz.ng;
        const int nx = sz.xc - 2 * ng;
        const int ny = std::max(sz.yc - 2 * ng, 1);
        const int nz = std::max(sz.zc - 2 * ng, 1);

        Hybrid2Scratch s = allocate_hybrid2_scratch(sz, ND);
        const auto& de_dq = s.de_dq;

        // S of the whole step, from the stage state.
        hybrid2_cell_quantities<FTraits>(sim, ctx, s);
        const fp_t dt_diff = hybrid2_dt_diff(state, s.chi);
        const bool substep = ctx.h2_s_max > 0.0_fp && dt / dt_diff > ctx.h2_s_max;

        if (!substep) {
            hybrid2_rhs<FTraits>(sim, ctx, heat_face, dt, s);
            // One more right-hand side in this stage: the outer RK integrates it.
            dex_parallel_for(
                "hybrid2 accumulate rates",
                FlatLoop<3>(sz.zc, sz.yc, sz.xc),
                KOKKOS_LAMBDA (int k, int j, int i) {
                    S(I(Cons::Ene), k, j, i) += de_dq(0, k, j, i);
                    for (int a = 0; a < ND; ++a) {
                        S_face(a, k, j, i) += de_dq(1 + a, k, j, i);
                    }
                }
            );
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
        Fp3d e("h2_e", sz.zc, sz.yc, sz.xc); // current
        Fp3d e_n("h2_e_n", sz.zc, sz.yc, sz.xc); // start of the substep
        Fp4d q("h2_q", ND, sz.zc, sz.yc, sz.xc);
        Fp4d q_n("h2_q_n", ND, sz.zc, sz.yc, sz.xc);
        dex_parallel_for(
            "hybrid2 substep init",
            FlatLoop<3>(sz.zc, sz.yc, sz.xc),
            KOKKOS_LAMBDA (int k, int j, int i) {
                const fp_t ei = W(I(Prim::Pres), k, j, i) / (eos.gamma - 1.0_fp);
                e0(k, j, i) = ei;
                e(k, j, i) = ei;
                for (int a = 0; a < ND; ++a) {
                    q(a, k, j, i) = heat_face(a, k, j, i);
                }
            }
        );
        Kokkos::fence();

        // Shu-Osher SSP-RK3: u = c_n u_n + c_u (u + h L(u)).
        constexpr fp_t c_n[3] = {0.0_fp, 0.75_fp, 1.0_fp / 3.0_fp};
        constexpr fp_t c_u[3] = {1.0_fp, 0.25_fp, 2.0_fp / 3.0_fp};
        for (int sub = 0; sub < n_sub; ++sub) {
            e.deep_copy_to(e_n);
            q.deep_copy_to(q_n);
            Kokkos::fence();
            for (int stage = 0; stage < 3; ++stage) {
                if (sub > 0 || stage > 0) {
                    hybrid2_cell_quantities<FTraits>(sim, ctx, s);
                }
                hybrid2_rhs<FTraits>(sim, ctx, q, h, s);
                const fp_t cn = c_n[stage];
                const fp_t cu = c_u[stage];
                dex_parallel_for(
                    "hybrid2 substep stage",
                    FlatLoop<3>(nz, ny, nx),
                    KOKKOS_LAMBDA (int ki, int ji, int ii) {
                        const int k = nz == 1 ? ki : ki + ng;
                        const int j = ny == 1 ? ji : ji + ng;
                        const int i = ii + ng;
                        e(k, j, i) = cn * e_n(k, j, i) + cu * (e(k, j, i) + h * de_dq(0, k, j, i));
                        W(I(Prim::Pres), k, j, i) = e(k, j, i) * (eos.gamma - 1.0_fp);
                    }
                );
                for (int ax = 0; ax < ND; ++ax) {
                    int ext[3] = {nx, ny, nz};
                    ext[ax] += 1;
                    dex_parallel_for(
                        "hybrid2 substep stage faces",
                        FlatLoop<3>(ext[2], ext[1], ext[0]),
                        KOKKOS_LAMBDA (int kf, int jf, int iff) {
                            const int k = nz == 1 ? kf : kf + ng;
                            const int j = ny == 1 ? jf : jf + ng;
                            const int i = iff + ng;
                            q(ax, k, j, i) = cn * q_n(ax, k, j, i) + cu * (q(ax, k, j, i) + h * de_dq(1 + ax, k, j, i));
                        }
                    );
                }
                Kokkos::fence();
                hybrid2_sync_periodic_faces<FTraits>(state, q);
                fill_sts_pressure_ghosts<FTraits>(sim);
            }
        }

        const fp_t inv_dt_sub = 1.0_fp / dt_sub;
        dex_parallel_for(
            "hybrid2 substep increments",
            FlatLoop<3>(sz.zc, sz.yc, sz.xc),
            KOKKOS_LAMBDA (int k, int j, int i) {
                const bool interior = (
                    i >= ng && i < ng + nx
                    && (ND < 2 || (j >= ng && j < ng + ny))
                    && (ND < 3 || (k >= ng && k < ng + nz))
                );
                if (interior) {
                    S(I(Cons::Ene), k, j, i) += (e(k, j, i) - e0(k, j, i)) * inv_dt_sub;
                }
                for (int a = 0; a < ND; ++a) {
                    S_face(a, k, j, i) += (q(a, k, j, i) - heat_face(a, k, j, i)) * inv_dt_sub;
                }
                // restore the stage pressure for the other source terms
                W(I(Prim::Pres), k, j, i) = e0(k, j, i) * (eos.gamma - 1.0_fp);
            }
        );
        Kokkos::fence();
    }
}

}

#else
#endif
