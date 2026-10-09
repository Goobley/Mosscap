#if !defined(MOSSCAP_YE_CONDUCTION_HPP)
#define MOSSCAP_YE_CONDUCTION_HPP
// Ye et al. (2020) directionally split, semi-implicit field-aligned conduction, a method of the thermal_conduction
// source term (sources.thermal_conduction.method: ye). Reference: HyperTcWork/standalone/ye_conduction.py (same
// order of operations), from HyperTcWork/htc2d/ye.py.
//
// One Ye step of length h is a line sweep along each axis in turn (x first on even steps, y first on odd ones). In
// the sweep along axis a every face along a carries
//
//     F = -A (T*_R - T*_L) / dx + G
//
// with T* the temperature after the sweep (implicit: one tridiagonal solve per line, cyclic on periodic axes) and G
// explicit, from the temperature before the sweep. Each face flux is single valued, so energy is conserved to
// round-off. The conductivity is frozen for the step at T~ = max(2 T^n - T^(n-1), T_floor) (q = 2; T^n on the first
// step of a stage, or with q = 1). cross:
//   amrvac  amrvac's slope-limited symmetric operator: corners take the mean of their cells' unit vectors b (not
//           renormalised) and K(T~); kaf = mean of the face's 2 corner K x saturation factor;
//           A = kaf (bnN^2 + bnS^2) / 2;
//           G = -kaf [(bnN^2 (gN - g) + bnS^2 (gS - g)) / 2 + bn_f (btN + btS) / 2 mcl(mcl_L, mcl_R) / dx],
//           g the face normal gradient, gN / gS the corner normal gradients clipped to [0.75 g, g / 0.75].
//   mean    the asymmetric face flux, transverse difference the unlimited mean of four: face b from the 2 cells'
//           unit vectors, renormalised; Kf = K(face-mean T~) x saturation factor; A = Kf bn^2,
//           G = -Kf bn bt mean(4 transverse differences) / dx.
// Saturation: per face q_sat / (q_sat + |F_ex|), F_ex the explicit face flux from T~ with the mean transverse
// difference (upwind pressure, face-mean density), fixed for the step.
// Boundaries: periodic -> cyclic lines; constant -> the ghost T enters the implicit flux; otherwise no implicit flux
// through the boundary face. G on a boundary face uses the ghosts, refreshed by fill_sts_pressure_ghosts.
//
// Coupling (unsplit, inside each RK stage, like STS / hybrid2): each stage advances e over its dt_sub in
// n = ceil(dt_sub / dt_diff / s_max) Ye steps (one if s_max <= 0) with rho and B frozen, and returns the increment
// as a rate. First order in time.
// NOTE(claude): The sweep order alternates over all Ye steps taken (ThermalConductionContext::ye_steps), which is
// not saved on restart.
//
// Configuration (sources.thermal_conduction, with enable / anisotropic / saturate / saturation_phi / kappa0 /
// spitzer shared with the STS method):
//     method: ye
//     ye:
//       cross: amrvac      # amrvac | mean
//       s_max: 0           # substep each RK stage when S = dt_sub / dt_diff exceeds this (<= 0: never)
//       q: 2               # extrapolation order of the frozen conductivity (1 | 2)
// 2D, mhd or glmmhd only. A domain must hold whole lines (no MPI decomposition of the hydro grid).
#include "ThermalConduction.hpp"
#include "ConductionCommon.hpp"
#include "Hybrid2Conduction.hpp"

namespace Mosscap {

constexpr fp_t YE_ALPHA = 0.75_fp; /// amrvac's clipping of the corner normal gradients
constexpr fp_t YE_T_FLOOR = 1e-20_fp;

enum class YeBoundary : int {
    Periodic = 0,
    Fixed,
    ZeroFlux
};

inline YeBoundary ye_boundary(BoundaryType b) {
    if (b == BoundaryType::Periodic) {
        return YeBoundary::Periodic;
    }
    if (b == BoundaryType::Constant) {
        return YeBoundary::Fixed;
    }
    return YeBoundary::ZeroFlux;
}

/// amrvac slope_limiter MC: 2 sign(a) max(0, min(|a|, sign(a) b, (a + b) / 4 sign(a))).
KOKKOS_INLINE_FUNCTION fp_t ye_mcl(fp_t a, fp_t b) {
    const fp_t s = fp_t((a > 0.0_fp) - (a < 0.0_fp));
    return 2.0_fp * s * std::max(0.0_fp, std::min(std::min(std::abs(a), s * b), 0.25_fp * (a + b) * s));
}

KOKKOS_INLINE_FUNCTION fp_t ye_kappa(const H2Params& par, fp_t T) {
    fp_t K = par.kappa0;
    if (par.spitzer) {
        K *= square(T) * std::sqrt(T);
    }
    return K;
}

struct YeScratch {
    Fp4d cell; // [0] T (current), [1] T~, [2] C = e / T, [3] K(T~), [4] T at the start of the previous step
    Fp4d b; // unit vectors of B along the simulated axes (0 at |B| = 0)
    Fp3d e0; // e at the start of the stage, ghosts included (to restore W)
    Fp3d A; // implicit coefficient, lower face of each cell along the sweep axis
    Fp3d G; // explicit flux, same faces
    Fp3d cp; // Thomas: modified upper diagonal
    Fp3d dp; // Thomas: modified right-hand side, then the solution
    Fp3d dz; // cyclic lines: the Sherman-Morrison correction
};

inline YeScratch allocate_ye_scratch(const GridSize& sz, int num_dim) {
    YeScratch s;
    s.cell = Fp4d("ye_cell", 5, sz.zc, sz.yc, sz.xc);
    s.b = Fp4d("ye_b", num_dim, sz.zc, sz.yc, sz.xc);
    s.e0 = Fp3d("ye_e0", sz.zc, sz.yc, sz.xc);
    s.A = Fp3d("ye_A", sz.zc, sz.yc, sz.xc);
    s.G = Fp3d("ye_G", sz.zc, sz.yc, sz.xc);
    s.cp = Fp3d("ye_cp", sz.zc, sz.yc, sz.xc);
    s.dp = Fp3d("ye_dp", sz.zc, sz.yc, sz.xc);
    s.dz = Fp3d("ye_dz", sz.zc, sz.yc, sz.xc);
    return s;
}

/// Stage start: e0, C, b and T (all cells, ghosts included); returns dt_diff = 0.8 dx^2 / (2 chi_max) (interior).
template <typename FTraits>
fp_t ye_stage_init(const Simulation& sim, const ThermalConductionContext& ctx, const YeScratch& s) {
    using Prim = typename FTraits::prim;
    constexpr int ND = FTraits::num_dim;
    JasUnpack(sim, state, eos);
    JasUnpack(state, W, sz, dx);
    JasUnpack(s, cell, b, e0);
    const H2Params par = h2_params(sim, ctx);
    const H2Grid g = h2_grid(state, ND);
    const fp_t gm1 = eos.gamma - 1.0_fp;
    dex_parallel_for(
        "ye stage init",
        FlatLoop<3>(sz.zc, sz.yc, sz.xc),
        KOKKOS_LAMBDA (int k, int j, int i) {
            const int c[3] = {i, j, k};
            const fp_t T = h2_temperature<FTraits>(W, eos, par.m_p, c);
            const fp_t e = W(I(Prim::Pres), k, j, i) / gm1;
            e0(k, j, i) = e;
            cell(0, k, j, i) = T;
            cell(2, k, j, i) = e / T;
            fp_t B[3];
            const fp_t nb = h2_field<FTraits>(W, c, B);
            for (int a = 0; a < ND; ++a) {
                b(a, k, j, i) = (nb > 0.0_fp) ? B[a] / nb : 0.0_fp;
            }
        }
    );
    fp_t chi_max = 0.0_fp;
    dex_parallel_reduce(
        "ye chi max",
        FlatLoop<3>(g.n[2], g.n[1], g.n[0]),
        KOKKOS_LAMBDA (int ki, int ji, int ii, fp_t& running) {
            const int k = ki + g.lo[2];
            const int j = ji + g.lo[1];
            const int i = ii + g.lo[0];
            const fp_t C = cell(2, k, j, i);
            running = std::max(running, ye_kappa(par, cell(0, k, j, i)) / C);
        },
        Kokkos::Max<fp_t>(chi_max)
    );
    Kokkos::fence();
    return 0.8_fp * square(dx) / (2.0_fp * std::max(chi_max, H2_TINY));
}

/// Start of a Ye step: T from W, T~ (q = 1 on the first step of the stage), K(T~), and T^n kept for the next step.
template <typename FTraits>
void ye_step_cells(const Simulation& sim, const ThermalConductionContext& ctx, const YeScratch& s, bool first) {
    JasUnpack(sim, state, eos);
    JasUnpack(state, W, sz);
    const auto& cell = s.cell;
    const H2Params par = h2_params(sim, ctx);
    const bool extrapolate = !first && ctx.ye_q >= 2;
    dex_parallel_for(
        "ye step cells",
        FlatLoop<3>(sz.zc, sz.yc, sz.xc),
        KOKKOS_LAMBDA (int k, int j, int i) {
            const int c[3] = {i, j, k};
            const fp_t T = h2_temperature<FTraits>(W, eos, par.m_p, c);
            fp_t Tt = extrapolate ? 2.0_fp * T - cell(4, k, j, i) : T;
            Tt = std::max(Tt, YE_T_FLOOR);
            cell(0, k, j, i) = T;
            cell(1, k, j, i) = Tt;
            cell(3, k, j, i) = ye_kappa(par, Tt);
            cell(4, k, j, i) = T;
        }
    );
}

/// T from W (all cells), between the sweeps of a step.
template <typename FTraits>
void ye_refresh_T(const Simulation& sim, const ThermalConductionContext& ctx, const YeScratch& s) {
    JasUnpack(sim, state, eos);
    JasUnpack(state, W, sz);
    const auto& cell = s.cell;
    const fp_t m_p = state.p_mass;
    dex_parallel_for(
        "ye refresh T",
        FlatLoop<3>(sz.zc, sz.yc, sz.xc),
        KOKKOS_LAMBDA (int k, int j, int i) {
            const int c[3] = {i, j, k};
            cell(0, k, j, i) = h2_temperature<FTraits>(W, eos, m_p, c);
        }
    );
}

/// A and G on the lower face along Ax of every cell from the first interior index to one past the last along Ax
/// (interior along the other axis).
template <typename FTraits, int Ax>
void ye_faces(const Simulation& sim, const ThermalConductionContext& ctx, const YeScratch& s) {
    using Prim = typename FTraits::prim;
    constexpr int ND = FTraits::num_dim;
    constexpr int At = 1 - Ax; // transverse axis (2D)
    JasUnpack(sim, state, eos);
    JasUnpack(state, W, dx);
    JasUnpack(s, cell, b, A, G);
    const H2Params par = h2_params(sim, ctx);
    const H2Grid g = h2_grid(state, ND);
    const fp_t inv_dx = 1.0_fp / dx;
    const fp_t gm1 = eos.gamma - 1.0_fp;
    const bool amrvac = ctx.ye_cross == YeCross::Amrvac;
    int dims[3] = {g.n[0], g.n[1], g.n[2]};
    dims[Ax] += 1;
    dex_parallel_for(
        "ye faces",
        FlatLoop<3>(dims[2], dims[1], dims[0]),
        KOKKOS_LAMBDA (int kf, int jf, int iff) {
            const int R[3] = {iff + g.lo[0], jf + g.lo[1], kf + g.lo[2]};
            int L[3] = {R[0], R[1], R[2]};
            L[Ax] -= 1;
            int Ru[3] = {R[0], R[1], R[2]};
            int Rd[3] = {R[0], R[1], R[2]};
            int Lu[3] = {L[0], L[1], L[2]};
            int Ld[3] = {L[0], L[1], L[2]};
            Ru[At] += 1;
            Rd[At] -= 1;
            Lu[At] += 1;
            Ld[At] -= 1;
            auto at = [&](int q, const int* c) -> fp_t { return cell(q, c[2], c[1], c[0]); };
            auto bv = [&](int a, const int* c) -> fp_t { return b(a, c[2], c[1], c[0]); };

            // saturation factor from T~ (explicit face flux, mean transverse difference)
            fp_t sat = 1.0_fp;
            if (par.saturate) {
                const fp_t tR = at(1, R);
                const fp_t tL = at(1, L);
                const fp_t dTn = (tR - tL) * inv_dx;
                const fp_t dTt = 0.25_fp * ((at(1, Ru) - tR) + (tR - at(1, Rd)) + (at(1, Lu) - tL) + (tL - at(1, Ld))) * inv_dx;
                const fp_t kf = 0.5_fp * (at(3, R) + at(3, L));
                fp_t BR[3], BL[3];
                h2_field<FTraits>(W, R, BR);
                h2_field<FTraits>(W, L, BL);
                const fp_t bn = 0.5_fp * (BR[Ax] + BL[Ax]);
                const fp_t bt = 0.5_fp * (BR[At] + BL[At]);
                const fp_t Bmag = std::max(std::sqrt(bn * bn + bt * bt), 1e-20_fp);
                const fp_t bdg = (bn * dTn + bt * dTt) / Bmag;
                const fp_t F = -kf * bdg * bn / Bmag;
                const fp_t pR = gm1 * at(2, R) * tR;
                const fp_t pL = gm1 * at(2, L) * tL;
                const fp_t pup = (F > 0.0_fp) ? pL : ((F < 0.0_fp) ? pR : 0.5_fp * (pL + pR));
                const fp_t rf = 0.5_fp * (W(I(Prim::Rho), R[2], R[1], R[0]) + W(I(Prim::Rho), L[2], L[1], L[0]));
                const fp_t qs = 5.0_fp * par.phi * std::sqrt(pup / rf) * pup;
                sat = qs / (qs + std::abs(kf * bdg));
            }

            const fp_t T_R = at(0, R);
            const fp_t T_L = at(0, L);
            fp_t A_f, G_f;
            if (amrvac) {
                // corners N (rows t, t + 1) and S (rows t - 1, t): means over their 4 cells
                const fp_t BnN = 0.25_fp * (bv(Ax, Lu) + bv(Ax, L) + bv(Ax, Ru) + bv(Ax, R));
                const fp_t BnS = 0.25_fp * (bv(Ax, L) + bv(Ax, Ld) + bv(Ax, R) + bv(Ax, Rd));
                const fp_t BtN = 0.25_fp * (bv(At, Lu) + bv(At, L) + bv(At, Ru) + bv(At, R));
                const fp_t BtS = 0.25_fp * (bv(At, L) + bv(At, Ld) + bv(At, R) + bv(At, Rd));
                const fp_t KcN = 0.25_fp * (at(3, Lu) + at(3, L) + at(3, Ru) + at(3, R));
                const fp_t KcS = 0.25_fp * (at(3, L) + at(3, Ld) + at(3, R) + at(3, Rd));
                const fp_t kaf = 0.5_fp * (KcN + KcS) * sat;
                const fp_t gu = (at(0, Ru) - at(0, Lu)) * inv_dx;
                const fp_t gc = (T_R - T_L) * inv_dx;
                const fp_t gd = (at(0, Rd) - at(0, Ld)) * inv_dx;
                const fp_t qN = 0.5_fp * (gu + gc);
                const fp_t qS = 0.5_fp * (gc + gd);
                const fp_t lo = std::min(YE_ALPHA * gc, gc / YE_ALPHA);
                const fp_t hi = std::max(YE_ALPHA * gc, gc / YE_ALPHA);
                const fp_t gN = std::min(std::max(qN, lo), hi);
                const fp_t gS = std::min(std::max(qS, lo), hi);
                A_f = kaf * 0.5_fp * (BnN * BnN + BnS * BnS);
                const fp_t cl = ye_mcl(at(0, Lu) - T_L, T_L - at(0, Ld));
                const fp_t cr = ye_mcl(at(0, Ru) - T_R, T_R - at(0, Rd));
                const fp_t tg = ye_mcl(cl, cr) * inv_dx;
                const fp_t coef = 0.5_fp * (bv(Ax, L) + bv(Ax, R)) * 0.5_fp * (BtN + BtS);
                G_f = -(kaf * 0.5_fp * (BnN * BnN * (gN - gc) + BnS * BnS * (gS - gc)) + kaf * coef * tg);
            } else {
                const fp_t Tf = 0.5_fp * (at(1, L) + at(1, R));
                fp_t bnf = 0.5_fp * (bv(Ax, L) + bv(Ax, R));
                fp_t btf = 0.5_fp * (bv(At, L) + bv(At, R));
                fp_t m = std::sqrt(bnf * bnf + btf * btf);
                m = (m > 0.0_fp) ? m : 1.0_fp;
                bnf /= m;
                btf /= m;
                const fp_t Kf = ye_kappa(par, std::max(Tf, YE_T_FLOOR)) * sat;
                const fp_t d = 0.25_fp * ((at(0, Ru) - T_R) + (T_R - at(0, Rd)) + (at(0, Lu) - T_L) + (T_L - at(0, Ld)));
                A_f = Kf * bnf * bnf;
                G_f = -Kf * bnf * btf * d * inv_dx;
            }
            h2_at(A, R) = A_f;
            h2_at(G, R) = G_f;
        }
    );
}

/// One implicit sweep along Ax: a tridiagonal (cyclic on periodic axes) solve per line; writes p = (gamma - 1) C T*
/// on the interior cells. Needs ye_faces for the same state.
template <typename FTraits, int Ax>
void ye_solve(const Simulation& sim, const YeScratch& s, fp_t h) {
    using Prim = typename FTraits::prim;
    constexpr int ND = FTraits::num_dim;
    JasUnpack(sim, state, eos);
    JasUnpack(state, W, dx);
    JasUnpack(s, cell, A, G, cp, dp, dz);
    const H2Grid g = h2_grid(state, ND);
    const auto& bdry = state.boundaries;
    const YeBoundary lo_kind = ye_boundary(Ax == 0 ? bdry.xs : bdry.ys);
    const YeBoundary hi_kind = ye_boundary(Ax == 0 ? bdry.xe : bdry.ye);
    const bool periodic = lo_kind == YeBoundary::Periodic;
    const fp_t c = h / square(dx);
    const fp_t h_dx = h / dx;
    const fp_t gm1 = eos.gamma - 1.0_fp;
    const int n = g.n[Ax];
    int dims[3] = {g.n[0], g.n[1], g.n[2]};
    dims[Ax] = 1;
    dex_parallel_for(
        "ye line solve",
        FlatLoop<3>(dims[2], dims[1], dims[0]),
        KOKKOS_LAMBDA (int kl, int jl, int il) {
            const int base[3] = {il + g.lo[0], jl + g.lo[1], kl + g.lo[2]};
            auto idx = [&](int m, int* c3) {
                c3[0] = base[0];
                c3[1] = base[1];
                c3[2] = base[2];
                c3[Ax] = g.lo[Ax] + m;
            };
            // face m is the lower face of cell m along the line (m = 0 .. n)
            auto Af = [&](int m) -> fp_t {
                int c3[3];
                idx(m, c3);
                return h2_at(A, c3);
            };
            // row m of the system: lower / diagonal / upper coefficients and the right-hand side
            auto row = [&](int m, fp_t& lo, fp_t& di, fp_t& up, fp_t& rhs) {
                int c3[3];
                idx(m, c3);
                const fp_t C = cell(2, c3[2], c3[1], c3[0]);
                const fp_t T = cell(0, c3[2], c3[1], c3[0]);
                fp_t AL = Af(m);
                fp_t AR = Af(m + 1);
                lo = -c * AL;
                up = -c * AR;
                int c4[3];
                idx(m + 1, c4);
                rhs = C * T + h_dx * (G(c3[2], c3[1], c3[0]) - G(c4[2], c4[1], c4[0]));
                if (periodic) {
                    if (m == n - 1) {
                        AR = Af(0);
                    }
                    di = C + c * (AL + AR);
                    return;
                }
                const fp_t bl = AL;
                const fp_t bu = AR;
                if (m == 0) {
                    AL = 0.0_fp;
                }
                if (m == n - 1) {
                    AR = 0.0_fp;
                }
                di = C + c * (AL + AR);
                if (m == 0 && lo_kind == YeBoundary::Fixed) {
                    int gh[3];
                    idx(-1, gh);
                    di += c * bl;
                    rhs += c * bl * cell(0, gh[2], gh[1], gh[0]);
                }
                if (m == n - 1 && hi_kind == YeBoundary::Fixed) {
                    int gh[3];
                    idx(n, gh);
                    di += c * bu;
                    rhs += c * bu * cell(0, gh[2], gh[1], gh[0]);
                }
            };

            // forward elimination; on a cyclic line, the diagonal is modified (Sherman-Morrison) and a second
            // right-hand side u = (gam, 0, ..., 0, corner) is eliminated alongside
            const fp_t corner = -c * Af(0);
            fp_t lo, di, up, rhs;
            row(0, lo, di, up, rhs);
            const fp_t gam = -di;
            int c3[3];
            {
                fp_t d0 = periodic ? di - gam : di;
                if (n == 1 && periodic) {
                    d0 -= corner * corner / gam;
                }
                idx(0, c3);
                h2_at(cp, c3) = up / d0;
                h2_at(dp, c3) = rhs / d0;
                if (periodic) {
                    h2_at(dz, c3) = gam / d0;
                }
            }
            fp_t cp_prev = h2_at(cp, c3);
            fp_t dp_prev = h2_at(dp, c3);
            fp_t dz_prev = periodic ? h2_at(dz, c3) : 0.0_fp;
            for (int m = 1; m < n; ++m) {
                row(m, lo, di, up, rhs);
                if (periodic && m == n - 1) {
                    di -= corner * corner / gam;
                }
                const fp_t piv = di - lo * cp_prev;
                idx(m, c3);
                cp_prev = up / piv;
                dp_prev = (rhs - lo * dp_prev) / piv;
                h2_at(cp, c3) = cp_prev;
                h2_at(dp, c3) = dp_prev;
                if (periodic) {
                    const fp_t u = (m == n - 1) ? corner : 0.0_fp;
                    dz_prev = (u - lo * dz_prev) / piv;
                    h2_at(dz, c3) = dz_prev;
                }
            }
            // back substitution (in place)
            for (int m = n - 2; m >= 0; --m) {
                int c4[3];
                idx(m, c3);
                idx(m + 1, c4);
                h2_at(dp, c3) = h2_at(dp, c3) - h2_at(cp, c3) * h2_at(dp, c4);
                if (periodic) {
                    h2_at(dz, c3) = h2_at(dz, c3) - h2_at(cp, c3) * h2_at(dz, c4);
                }
            }
            fp_t fact = 0.0_fp;
            if (periodic) {
                int cl3[3];
                idx(0, c3);
                idx(n - 1, cl3);
                fact = (h2_at(dp, c3) + corner * h2_at(dp, cl3) / gam)
                    / (1.0_fp + h2_at(dz, c3) + corner * h2_at(dz, cl3) / gam);
            }
            for (int m = 0; m < n; ++m) {
                idx(m, c3);
                fp_t T = h2_at(dp, c3);
                if (periodic) {
                    T -= fact * h2_at(dz, c3);
                }
                W(I(Prim::Pres), c3[2], c3[1], c3[0]) = gm1 * cell(2, c3[2], c3[1], c3[0]) * T;
            }
        }
    );
}

template <typename FTraits, int Ax>
void ye_sweep(const Simulation& sim, const ThermalConductionContext& ctx, const YeScratch& s, fp_t h) {
    ye_faces<FTraits, Ax>(sim, ctx, s);
    ye_solve<FTraits, Ax>(sim, s, h);
    Kokkos::fence();
    fill_sts_pressure_ghosts<FTraits>(sim);
}

template <typename FTraits>
void ye_conduction_kernel(const Simulation& sim, const ThermalConductionContext& ctx) {
    using Prim = typename FTraits::prim;
    using Cons = typename FTraits::cons;
    constexpr int ND = FTraits::num_dim;
    if constexpr (!FTraits::is_mhd || FTraits::has_hypertc || ND != 2) {
        throw std::runtime_error("ye conduction needs a 2D MHD fluid without HeatF (mhd / glmmhd)");
    } else {
        JasUnpack(sim, state, eos, sources, dt_sub);
        JasUnpack(state, W, sz);
        const auto& S = sources.S;
        const H2Grid g = h2_grid(state, ND);
        const int ng = g.ng;
        const fp_t gm1 = eos.gamma - 1.0_fp;

        YeScratch s = allocate_ye_scratch(sz, ND);
        const fp_t dt_diff = ye_stage_init<FTraits>(sim, ctx, s);
        int n_sub = 1;
        if (ctx.ye_s_max > 0.0_fp) {
            n_sub = std::max(1, int(std::ceil(dt_sub / dt_diff / ctx.ye_s_max - 1e-12_fp)));
        }
        const fp_t h = dt_sub / fp_t(n_sub);
        for (int sub = 0; sub < n_sub; ++sub) {
            ye_step_cells<FTraits>(sim, ctx, s, sub == 0);
            const bool x_first = (ctx.ye_steps % 2) == 0;
            if (x_first) {
                ye_sweep<FTraits, 0>(sim, ctx, s, h);
                ye_refresh_T<FTraits>(sim, ctx, s);
                ye_sweep<FTraits, 1>(sim, ctx, s, h);
            } else {
                ye_sweep<FTraits, 1>(sim, ctx, s, h);
                ye_refresh_T<FTraits>(sim, ctx, s);
                ye_sweep<FTraits, 0>(sim, ctx, s, h);
            }
            ctx.ye_steps += 1;
        }

        const fp_t inv_dt_sub = 1.0_fp / dt_sub;
        const auto& e0 = s.e0;
        dex_parallel_for(
            "ye increments",
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
                // restore the stage pressure for the other source terms
                W(I(Prim::Pres), k, j, i) = e0(k, j, i) * gm1;
            }
        );
        Kokkos::fence();
    }
}

}

#else
#endif
