#if !defined(MOSSCAP_SEMI_IMPLICIT_CONDUCTION_HPP)
#define MOSSCAP_SEMI_IMPLICIT_CONDUCTION_HPP
// Directionally split, semi-implicit field-aligned conduction (Ye et al. 2020).
// Each step sweeps each direction in turn. The face flux along the sweep axis
// is F = -A dT*/dx + G, where the normal component is evaluated implicitly
// (tridiagonal, possibly periodic), and the transverse term G is explicit using
// either the unlimited mean, or the Sharma & Hammett (2007) symmetric limited
// form.

// Conductivity for the solve is frozen at its initial value based on the
// temperature. As this method is semi-implicit, it can run at much greater than
// the explicit diffusive limit, but will start to lose accuracy. Some of this can be
// regained by subcycling this process directly, triggered by setting s_max
// (default 1 step per MHD substep). Sweep order alternates starting axis, so
// the solve count is written back to the conduction context.

// Currently only implemented in 2D.

#include "ThermalConduction.hpp"
#include "ConductionCommon.hpp"

namespace Mosscap {

constexpr fp_t SEMI_IMP_ALPHA = 0.75_fp; /// clipping of the corner normal gradients
constexpr fp_t SEMI_IMP_T_FLOOR = 1e-20_fp;

enum class SemiImpBoundary : int {
    Periodic = 0,
    Fixed,
    ZeroFlux
};

inline SemiImpBoundary semi_imp_boundary(BoundaryType b) {
    if (b == BoundaryType::Periodic) {
        return SemiImpBoundary::Periodic;
    }
    if (b == BoundaryType::Constant) {
        return SemiImpBoundary::Fixed;
    }
    return SemiImpBoundary::ZeroFlux;
}

/// Slots of SemiImpScratch::cell
enum class SemiImpCell : int {
    T = 0, /// current temperature
    TStep, /// T~: temperature at the start of the semi-implicit step (floored), held for both sweeps
    C, /// e / T
    K, /// conductivity K(T~)
    NumSlots
};

struct SemiImpScratch {
    Fp4d cell; // per-cell quantities, indexed by SemiImpCell
    Fp4d b; // unit vector of B, (x, y, z), from conduction_field (0 at |B| = 0)
    Fp3d e0; // e at the start of the stage, ghosts included (to restore W)
    Fp3d A; // implicit coefficient, lower face of each cell along the sweep axis
    Fp3d G; // explicit flux, same faces
    Fp3d cp; // Thomas: modified upper diagonal
    Fp3d dp; // Thomas: modified right-hand side, then the solution
    Fp3d dz; // cyclic lines: the Sherman-Morrison correction
};

inline SemiImpScratch allocate_semi_imp_scratch(const GridSize& sz) {
    SemiImpScratch s;
    s.cell = Fp4d("semi_imp_cell", I(SemiImpCell::NumSlots), sz.zc, sz.yc, sz.xc);
    s.b = Fp4d("semi_imp_b", 3, sz.zc, sz.yc, sz.xc);
    s.e0 = Fp3d("semi_imp_e0", sz.zc, sz.yc, sz.xc);
    s.A = Fp3d("semi_imp_A", sz.zc, sz.yc, sz.xc);
    s.G = Fp3d("semi_imp_G", sz.zc, sz.yc, sz.xc);
    s.cp = Fp3d("semi_imp_cp", sz.zc, sz.yc, sz.xc);
    s.dp = Fp3d("semi_imp_dp", sz.zc, sz.yc, sz.xc);
    s.dz = Fp3d("semi_imp_dz", sz.zc, sz.yc, sz.xc);
    return s;
}

/// Stage start: e0, C, b and T (all cells, ghosts included).
template <typename FTraits>
void semi_imp_stage_init(const Simulation& sim, const ThermalConductionContext& ctx, const SemiImpScratch& s) {
    using Prim = typename FTraits::prim;
    JasUnpack(sim, state, eos);
    JasUnpack(state, W, sz);
    JasUnpack(s, cell, b, e0);
    const fp_t m_p = state.p_mass;
    const fp_t gm1 = eos.gamma - 1.0_fp;
    dex_parallel_for(
        "semi-imp stage init",
        FlatLoop<3>(sz.zc, sz.yc, sz.xc),
        KOKKOS_LAMBDA (int k, int j, int i) {
            const QtyView w(W, CellIndex{.i = i, .j = j, .k = k});
            const fp_t T = cell_temperature<FTraits>(m_p, eos, w);
            const fp_t e = w(I(Prim::Pres)) / gm1;
            e0(k, j, i) = e;
            cell(I(SemiImpCell::T), k, j, i) = T;
            cell(I(SemiImpCell::C), k, j, i) = e / T;
            const vec3 B = conduction_field<FTraits>(ctx, w);
            const fp_t nb = std::sqrt(square(B(0)) + square(B(1)) + square(B(2)));
            for (int a = 0; a < 3; ++a) {
                b(a, k, j, i) = (nb > 0.0_fp) ? B(a) / nb : 0.0_fp;
            }
        }
    );
    Kokkos::fence();
}

/// Start of a semi-implicit step: T from W, and the step temperature T~ (T floored) with K(T~), held for both sweeps.
template <typename FTraits>
void semi_imp_step_cells(const Simulation& sim, const ThermalConductionContext& ctx, const SemiImpScratch& s) {
    JasUnpack(sim, state, eos);
    JasUnpack(state, W, sz);
    const auto& cell = s.cell;
    const fp_t m_p = state.p_mass;
    dex_parallel_for(
        "semi-imp step cells",
        FlatLoop<3>(sz.zc, sz.yc, sz.xc),
        KOKKOS_LAMBDA (int k, int j, int i) {
            const fp_t T = cell_temperature<FTraits>(m_p, eos, QtyView(W, CellIndex{.i = i, .j = j, .k = k}));
            const fp_t Tt = std::max(T, fp_t(SEMI_IMP_T_FLOOR));
            cell(I(SemiImpCell::T), k, j, i) = T;
            cell(I(SemiImpCell::TStep), k, j, i) = Tt;
            cell(I(SemiImpCell::K), k, j, i) = cell_kappa(ctx, Tt, CellIndex{.i = i, .j = j, .k = k});
        }
    );
}

/// T from W (all cells), between the sweeps of a step.
template <typename FTraits>
void semi_imp_refresh_T(const Simulation& sim, const SemiImpScratch& s) {
    JasUnpack(sim, state, eos);
    JasUnpack(state, W, sz);
    const auto& cell = s.cell;
    const fp_t m_p = state.p_mass;
    dex_parallel_for(
        "semi-imp refresh T",
        FlatLoop<3>(sz.zc, sz.yc, sz.xc),
        KOKKOS_LAMBDA (int k, int j, int i) {
            cell(I(SemiImpCell::T), k, j, i) = cell_temperature<FTraits>(m_p, eos, QtyView(W, CellIndex{.i = i, .j = j, .k = k}));
        }
    );
}

/// A and G on the lower face along Ax of every cell from the first interior index to one past the last along Ax
/// (interior along the other axis).
template <typename FTraits, int Ax>
void semi_imp_faces(const Simulation& sim, const ThermalConductionContext& ctx, const SemiImpScratch& s) {
    using Prim = typename FTraits::prim;
    constexpr int At = 1 - Ax; // transverse axis (2D)
    JasUnpack(sim, state, eos);
    JasUnpack(state, W, sz, dx);
    JasUnpack(s, cell, b, A, G);
    const fp_t inv_dx = 1.0_fp / dx;
    const fp_t gm1 = eos.gamma - 1.0_fp;
    const bool sharma = ctx.semi_imp_cross == SemiImpCross::SharmaSymmetric;
    const CellIndex start = interior_start(sz);
    CellIndex dims = interior_extent(sz);
    dims.along<Ax>() += 1;
    dex_parallel_for(
        "semi-imp faces",
        FlatLoop<3>(dims.k, dims.j, dims.i),
        KOKKOS_LAMBDA (int kf, int jf, int iff) {
            const CellIndex R{.i = iff + start.i, .j = jf + start.j, .k = kf + start.k};
            const CellIndex L = shift_along<Ax>(R, -1);
            // cell quantities (cR, ...) and unit vectors (bR, ...) of the face's cells and their transverse
            // neighbours (u: +1, d: -1 along At)
            const QtyView cR(cell, R);
            const QtyView cL(cell, L);
            const QtyView cRu = shift_along<At>(cR, 1);
            const QtyView cRd = shift_along<At>(cR, -1);
            const QtyView cLu = shift_along<At>(cL, 1);
            const QtyView cLd = shift_along<At>(cL, -1);

            const QtyView bR(b, R);
            const QtyView bL(b, L);
            const QtyView bRu = shift_along<At>(bR, 1);
            const QtyView bRd = shift_along<At>(bR, -1);
            const QtyView bLu = shift_along<At>(bL, 1);
            const QtyView bLd = shift_along<At>(bL, -1);

            // saturation factor from T~ (explicit face flux, mean transverse difference)
            fp_t sat_fac = 1.0_fp;
            if (ctx.saturate) {
                const QtyView wR(W, R);
                const QtyView wL(W, L);
                const fp_t T_step_R = cR(I(SemiImpCell::TStep));
                const fp_t T_step_L = cL(I(SemiImpCell::TStep));

                // normal gradient, and the transverse gradient as the mean of the four one-sided differences
                const fp_t dTdn = (T_step_R - T_step_L) * inv_dx;
                const fp_t dTdt_mean = 0.25_fp * (
                    (cRu(I(SemiImpCell::TStep)) - T_step_R) + (T_step_R - cRd(I(SemiImpCell::TStep)))
                    + (cLu(I(SemiImpCell::TStep)) - T_step_L) + (T_step_L - cLd(I(SemiImpCell::TStep)))
                ) * inv_dx;
                const fp_t K_face = 0.5_fp * (cR(I(SemiImpCell::K)) + cL(I(SemiImpCell::K)));

                // face-mean field (not unit): normal, transverse, out of plane
                const vec3 B_R = conduction_field<FTraits>(ctx, wR);
                const vec3 B_L = conduction_field<FTraits>(ctx, wL);
                const fp_t B_n = 0.5_fp * (B_R(Ax) + B_L(Ax));
                const fp_t B_t = 0.5_fp * (B_R(At) + B_L(At));
                const fp_t B_z = 0.5_fp * (B_R(2) + B_L(2)); // 0 unless out_of_plane_field

                const fp_t B_norm = std::max(std::sqrt(B_n * B_n + B_t * B_t + B_z * B_z), 1e-20_fp);
                const fp_t b_dot_gradT = (B_n * dTdn + B_t * dTdt_mean) / B_norm;
                const fp_t full_flux = -K_face * b_dot_gradT * B_n / B_norm;
                // upwind the saturated flux (Mignone+ 2012), as the STS method
                const fp_t p_R = gm1 * cR(I(SemiImpCell::C)) * T_step_R;
                const fp_t p_L = gm1 * cL(I(SemiImpCell::C)) * T_step_L;
                const fp_t upwind_pressure = (full_flux > 0.0_fp) ? p_L : ((full_flux < 0.0_fp) ? p_R : 0.5_fp * (p_L + p_R));
                const fp_t mean_rho = 0.5_fp * (wR(I(Prim::Rho)) + wL(I(Prim::Rho)));
                const fp_t sat_flux = 5.0_fp * ctx.saturation_phi * std::sqrt(upwind_pressure / mean_rho) * upwind_pressure;
                sat_fac = sat_flux / (sat_flux + std::abs(K_face * b_dot_gradT));
            }

            const fp_t T_R = cR(I(SemiImpCell::T));
            const fp_t T_L = cL(I(SemiImpCell::T));
            fp_t A_f, G_f;
            if (sharma) {
                // corners N (rows t, t + 1) and S (rows t - 1, t): unit-vector components and K as means over
                // their 4 cells
                const fp_t b_n_N = 0.25_fp * (bLu(Ax) + bL(Ax) + bRu(Ax) + bR(Ax));
                const fp_t b_n_S = 0.25_fp * (bL(Ax) + bLd(Ax) + bR(Ax) + bRd(Ax));
                const fp_t b_t_N = 0.25_fp * (bLu(At) + bL(At) + bRu(At) + bR(At));
                const fp_t b_t_S = 0.25_fp * (bL(At) + bLd(At) + bR(At) + bRd(At));
                const fp_t K_N = 0.25_fp * (cLu(I(SemiImpCell::K)) + cL(I(SemiImpCell::K)) + cRu(I(SemiImpCell::K)) + cR(I(SemiImpCell::K)));
                const fp_t K_S = 0.25_fp * (cL(I(SemiImpCell::K)) + cLd(I(SemiImpCell::K)) + cR(I(SemiImpCell::K)) + cRd(I(SemiImpCell::K)));
                const fp_t K_sat = 0.5_fp * (K_N + K_S) * sat_fac;

                // normal gradients in the rows above, through and below the face; the corner normal gradients
                // average adjacent rows and are clipped to [SEMI_IMP_ALPHA, 1 / SEMI_IMP_ALPHA] x the face's (amrvac)
                const fp_t dTdn_u = (cRu(I(SemiImpCell::T)) - cLu(I(SemiImpCell::T))) * inv_dx;
                const fp_t dTdn = (T_R - T_L) * inv_dx;
                const fp_t dTdn_d = (cRd(I(SemiImpCell::T)) - cLd(I(SemiImpCell::T))) * inv_dx;
                const fp_t dTdn_N_raw = 0.5_fp * (dTdn_u + dTdn);
                const fp_t dTdn_S_raw = 0.5_fp * (dTdn + dTdn_d);
                const fp_t clip_lo = std::min(SEMI_IMP_ALPHA * dTdn, dTdn / SEMI_IMP_ALPHA);
                const fp_t clip_hi = std::max(SEMI_IMP_ALPHA * dTdn, dTdn / SEMI_IMP_ALPHA);
                const fp_t dTdn_N = std::min(std::max(dTdn_N_raw, clip_lo), clip_hi);
                const fp_t dTdn_S = std::min(std::max(dTdn_S_raw, clip_lo), clip_hi);
                A_f = K_sat * 0.5_fp * (b_n_N * b_n_N + b_n_S * b_n_S);

                // transverse gradient: Sharma & Hammett symmetric slope limiter (MC) over the two cells' limited
                // transverse differences
                const fp_t dT_t_L = mc(cLu(I(SemiImpCell::T)) - T_L, T_L - cLd(I(SemiImpCell::T)));
                const fp_t dT_t_R = mc(cRu(I(SemiImpCell::T)) - T_R, T_R - cRd(I(SemiImpCell::T)));
                const fp_t dTdt = mc(dT_t_L, dT_t_R) * inv_dx;
                const fp_t b_n_b_t = 0.5_fp * (bL(Ax) + bR(Ax)) * 0.5_fp * (b_t_N + b_t_S);
                // explicit part: the corner normal gradients' departure from the implicit face gradient, plus the
                // cross term
                G_f = -(K_sat * 0.5_fp * (b_n_N * b_n_N * (dTdn_N - dTdn) + b_n_S * b_n_S * (dTdn_S - dTdn)) + K_sat * b_n_b_t * dTdt);
            } else {
                // face unit vector from the 2 cells' (renormalised), K at the face-mean T~
                const fp_t T_step_face = 0.5_fp * (cL(I(SemiImpCell::TStep)) + cR(I(SemiImpCell::TStep)));
                fp_t b_n = 0.5_fp * (bL(Ax) + bR(Ax));
                fp_t b_t = 0.5_fp * (bL(At) + bR(At));
                const fp_t b_z = 0.5_fp * (bL(2) + bR(2)); // 0 unless out_of_plane_field
                fp_t b_norm = std::sqrt(b_n * b_n + b_t * b_t + b_z * b_z);
                b_norm = (b_norm > 0.0_fp) ? b_norm : 1.0_fp;
                b_n /= b_norm;
                b_t /= b_norm;
                // With TRAC, kappa_eff is a cell quantity, so average for the
                // face quantity. Otherwise evaluate at the face-mean T~ (Ye's
                // form).
                const fp_t K_face = ctx.trac_kappa.initialized()
                    ? face_kappa(cL(I(SemiImpCell::K)), cR(I(SemiImpCell::K)))
                    : spitzer_kappa(ctx.kappa0, ctx.spitzer, std::max(T_step_face, fp_t(SEMI_IMP_T_FLOOR)));
                const fp_t K_sat = K_face * sat_fac;
                // unlimited mean of the four one-sided transverse differences
                const fp_t dT_t = 0.25_fp * (
                    (cRu(I(SemiImpCell::T)) - T_R) + (T_R - cRd(I(SemiImpCell::T)))
                    + (cLu(I(SemiImpCell::T)) - T_L) + (T_L - cLd(I(SemiImpCell::T)))
                );
                A_f = K_sat * b_n * b_n;
                G_f = -K_sat * b_n * b_t * dT_t * inv_dx;
            }
            A(R.k, R.j, R.i) = A_f;
            G(R.k, R.j, R.i) = G_f;
        }
    );
}

/// One implicit sweep along Ax: a tridiagonal (cyclic on periodic axes) solve per line; writes p = (gamma - 1) C T*
/// on the interior cells. Needs semi_imp_faces for the same state.
template <typename FTraits, int Ax>
void semi_imp_solve(const Simulation& sim, const SemiImpScratch& s, fp_t h) {
    using Prim = typename FTraits::prim;
    JasUnpack(sim, state, eos);
    JasUnpack(state, W, sz, dx);
    JasUnpack(s, cell, A, G, cp, dp, dz);
    const auto& bdry = state.boundaries;
    const SemiImpBoundary lo_kind = semi_imp_boundary(Ax == 0 ? bdry.xs : bdry.ys);
    const SemiImpBoundary hi_kind = semi_imp_boundary(Ax == 0 ? bdry.xe : bdry.ye);
    const bool periodic = lo_kind == SemiImpBoundary::Periodic;
    const fp_t c = h / square(dx);
    const fp_t h_dx = h / dx;
    const fp_t gm1 = eos.gamma - 1.0_fp;
    const CellIndex start = interior_start(sz);
    CellIndex dims = interior_extent(sz);
    const int n = dims.along<Ax>();
    // One GPU thread per line/tridiagonal system. Could be improved with teams, probably
    dims.along<Ax>() = 1;
    dex_parallel_for(
        "semi-imp line solve",
        FlatLoop<3>(dims.k, dims.j, dims.i),
        KOKKOS_LAMBDA (int kl, int jl, int il) {
            // cell m of the line is shift_along<Ax>(base, m); face m is its lower face (m = 0 .. n)
            const CellIndex base{.i = il + start.i, .j = jl + start.j, .k = kl + start.k};
            auto Af = [&](int m) -> fp_t {
                const CellIndex cm = shift_along<Ax>(base, m);
                return A(cm.k, cm.j, cm.i);
            };
            // row m of the system: lower / diagonal / upper coefficients and the right-hand side
            auto row = [&](int m, fp_t& lo, fp_t& di, fp_t& up, fp_t& rhs) {
                const CellIndex cm = shift_along<Ax>(base, m);
                const CellIndex cm1 = shift_along<Ax>(base, m + 1);
                const fp_t C = cell(I(SemiImpCell::C), cm.k, cm.j, cm.i);
                const fp_t T = cell(I(SemiImpCell::T), cm.k, cm.j, cm.i);
                fp_t AL = Af(m);
                fp_t AR = Af(m + 1);
                lo = -c * AL;
                up = -c * AR;
                rhs = C * T + h_dx * (G(cm.k, cm.j, cm.i) - G(cm1.k, cm1.j, cm1.i));
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
                if (m == 0 && lo_kind == SemiImpBoundary::Fixed) {
                    const CellIndex gh = shift_along<Ax>(base, -1);
                    di += c * bl;
                    rhs += c * bl * cell(I(SemiImpCell::T), gh.k, gh.j, gh.i);
                }
                if (m == n - 1 && hi_kind == SemiImpBoundary::Fixed) {
                    const CellIndex gh = shift_along<Ax>(base, n);
                    di += c * bu;
                    rhs += c * bu * cell(I(SemiImpCell::T), gh.k, gh.j, gh.i);
                }
            };

            // forward elimination; on a cyclic line, the diagonal is modified (Sherman-Morrison) and a second
            // right-hand side u = (gam, 0, ..., 0, corner) is eliminated alongside
            const fp_t corner = -c * Af(0);
            fp_t lo, di, up, rhs;
            row(0, lo, di, up, rhs);
            const fp_t gam = -di;
            {
                fp_t d0 = periodic ? di - gam : di;
                if (n == 1 && periodic) {
                    d0 -= corner * corner / gam;
                }
                cp(base.k, base.j, base.i) = up / d0;
                dp(base.k, base.j, base.i) = rhs / d0;
                if (periodic) {
                    dz(base.k, base.j, base.i) = gam / d0;
                }
            }
            fp_t cp_prev = cp(base.k, base.j, base.i);
            fp_t dp_prev = dp(base.k, base.j, base.i);
            fp_t dz_prev = periodic ? dz(base.k, base.j, base.i) : 0.0_fp;
            for (int m = 1; m < n; ++m) {
                row(m, lo, di, up, rhs);
                if (periodic && m == n - 1) {
                    di -= corner * corner / gam;
                }
                const fp_t piv = di - lo * cp_prev;
                const CellIndex cm = shift_along<Ax>(base, m);
                cp_prev = up / piv;
                dp_prev = (rhs - lo * dp_prev) / piv;
                cp(cm.k, cm.j, cm.i) = cp_prev;
                dp(cm.k, cm.j, cm.i) = dp_prev;
                if (periodic) {
                    const fp_t u = (m == n - 1) ? corner : 0.0_fp;
                    dz_prev = (u - lo * dz_prev) / piv;
                    dz(cm.k, cm.j, cm.i) = dz_prev;
                }
            }
            // back substitution (in place)
            for (int m = n - 2; m >= 0; --m) {
                const CellIndex cm = shift_along<Ax>(base, m);
                const CellIndex cm1 = shift_along<Ax>(base, m + 1);
                dp(cm.k, cm.j, cm.i) = dp(cm.k, cm.j, cm.i) - cp(cm.k, cm.j, cm.i) * dp(cm1.k, cm1.j, cm1.i);
                if (periodic) {
                    dz(cm.k, cm.j, cm.i) = dz(cm.k, cm.j, cm.i) - cp(cm.k, cm.j, cm.i) * dz(cm1.k, cm1.j, cm1.i);
                }
            }
            fp_t fact = 0.0_fp;
            if (periodic) {
                const CellIndex last = shift_along<Ax>(base, n - 1);
                fact = (dp(base.k, base.j, base.i) + corner * dp(last.k, last.j, last.i) / gam)
                    / (1.0_fp + dz(base.k, base.j, base.i) + corner * dz(last.k, last.j, last.i) / gam);
            }
            for (int m = 0; m < n; ++m) {
                const CellIndex cm = shift_along<Ax>(base, m);
                fp_t T = dp(cm.k, cm.j, cm.i);
                if (periodic) {
                    T -= fact * dz(cm.k, cm.j, cm.i);
                }
                W(I(Prim::Pres), cm.k, cm.j, cm.i) = gm1 * cell(I(SemiImpCell::C), cm.k, cm.j, cm.i) * T;
            }
        }
    );
}

template <typename FTraits, int Ax>
void semi_imp_sweep(const Simulation& sim, const ThermalConductionContext& ctx, const SemiImpScratch& s, fp_t h) {
    semi_imp_faces<FTraits, Ax>(sim, ctx, s);
    semi_imp_solve<FTraits, Ax>(sim, s, h);
    Kokkos::fence();
    fill_sts_pressure_ghosts<FTraits>(sim);
}

template <typename FTraits>
void semi_imp_conduction_kernel(const Simulation& sim, ThermalConductionContext& ctx, const Fp4d& S) {
    using Prim = typename FTraits::prim;
    using Cons = typename FTraits::cons;
    if constexpr (!FTraits::is_mhd || FTraits::has_hypertc || FTraits::num_dim != 2) {
        throw std::runtime_error("semi_implicit conduction needs a 2D MHD fluid without HeatF (mhd / glmmhd)");
    } else {
        JasUnpack(sim, state, eos, dt_sub);
        JasUnpack(state, W, sz);
        const fp_t gm1 = eos.gamma - 1.0_fp;

        SemiImpScratch s = allocate_semi_imp_scratch(sz);
        semi_imp_stage_init<FTraits>(sim, ctx, s);
        int n_sub = 1;
        if (ctx.semi_imp_s_max > 0.0_fp) {
            // NOTE(cmo): S = dt_sub / dt_diff is measured against the explicit (STS) conduction limit
            const fp_t dt_diff = estimate_thermal_conduction_timestep<FTraits>(sim, ctx);
            n_sub = std::max(1, int(std::ceil(dt_sub / dt_diff / ctx.semi_imp_s_max)));
        }
        const fp_t h = dt_sub / fp_t(n_sub);
        for (int sub = 0; sub < n_sub; ++sub) {
            semi_imp_step_cells<FTraits>(sim, ctx, s);
            const bool x_first = (ctx.semi_imp_steps % 2) == 0;
            if (x_first) {
                semi_imp_sweep<FTraits, 0>(sim, ctx, s, h);
                semi_imp_refresh_T<FTraits>(sim, s);
                semi_imp_sweep<FTraits, 1>(sim, ctx, s, h);
            } else {
                semi_imp_sweep<FTraits, 1>(sim, ctx, s, h);
                semi_imp_refresh_T<FTraits>(sim, s);
                semi_imp_sweep<FTraits, 0>(sim, ctx, s, h);
            }
            ctx.semi_imp_steps += 1;
        }

        const fp_t inv_dt_sub = 1.0_fp / dt_sub;
        const auto& e0 = s.e0;
        const CellIndex n = interior_extent(sz);
        const CellIndex start = interior_start(sz);
        dex_parallel_for(
            "semi-imp increments",
            FlatLoop<3>(sz.zc, sz.yc, sz.xc),
            KOKKOS_LAMBDA (int k, int j, int i) {
                const bool interior = (
                    i >= start.i && i < start.i + n.i
                    && j >= start.j && j < start.j + n.j
                    && k >= start.k && k < start.k + n.k
                );
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
