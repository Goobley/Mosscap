#include "ThermalConduction.hpp"
#include "../Simulation.hpp"
#include "../MosscapConfig.hpp"
#include "../SourceTerms.hpp"


namespace Mosscap {

// NOTE(cmo): This roughly follows athenapk, but I've tried to make the flow a little clearer
// Thus, it essentially follows PLUTO (Mignone+ 2012)

KOKKOS_INLINE_FUNCTION fp_t mc(const fp_t a, const fp_t b) {
    // phi(r) = max(0, min(2r, 0.5 * (1 + r), 2)). The term in the min is
    // multiplied through by up and sign terms implement the monotonicity
    // (and the factor of 2 that needs to multiply the second term)
    return (copysign(1.0_fp, a) + copysign(1.0_fp, b)) * std::min(std::abs(a), std::min(0.25_fp * std::abs(a + b), std::abs(b)));
}

KOKKOS_INLINE_FUNCTION fp_t mc4(const fp_t a, const fp_t b, const fp_t c, const fp_t d) {
    // 4 way limiter for transverse gradients in conduction
    return mc(mc(a, b), mc(c, d));
}

template <int Axis>
KOKKOS_INLINE_FUNCTION CellIndex shift_along(const CellIndex& from, int how_much) {
    CellIndex result(from);
    result.along<Axis>() += how_much;
    return result;
}

template <int Axis>
KOKKOS_INLINE_FUNCTION QtyView shift_along(const QtyView& from, int how_much) {
    CellIndex result(from.idx);
    result.along<Axis>() += how_much;
    return QtyView(from.q, result);
}

KOKKOS_INLINE_FUNCTION fp_t ion_frac(const Eos& eos, const CellIndex& idx) {
    fp_t y = eos.y;
    if (!eos.is_constant) {
        y = eos.y_space(idx.k, idx.j, idx.i);
    }
    return y;
}

template <typename FTraits, int Axis>
KOKKOS_INLINE_FUNCTION fp_t backwards_temperature_diff(const fp_t m_p, const Eos& eos, const Fp4d& W, const CellIndex& from) {
    using Prim = typename FTraits::prim;

    QtyView w_i(W, from);
    const fp_t Ti = temperature_si(
        w_i(I(Prim::Pres)),
        w_i(I(Prim::Rho)) / (eos.mass_per_h * m_p),
        eos.total_abund,
        ion_frac(eos, w_i.idx)
    );
    auto w_im1 = shift_along<Axis>(w_i, -1);
    const fp_t Tim1 = temperature_si(
        w_im1(I(Prim::Pres)),
        w_im1(I(Prim::Rho)) / (eos.mass_per_h * m_p),
        eos.total_abund,
        ion_frac(eos, w_im1.idx)
    );

    return Ti - Tim1;
}

template <typename FTraits, int Axis>
KOKKOS_INLINE_FUNCTION fp_t centred_temperature_diff(const fp_t m_p, const Eos& eos, const Fp4d& W, const CellIndex& around) {
    using Prim = typename FTraits::prim;

    CellIndex from = shift_along<Axis>(around, 1);
    QtyView w_ip1(W, from);
    const fp_t Tip1 = temperature_si(
        w_ip1(I(Prim::Pres)),
        w_ip1(I(Prim::Rho)) / (eos.mass_per_h * m_p),
        eos.total_abund,
        ion_frac(eos, w_ip1.idx)
    );
    auto w_im1 = shift_along<Axis>(w_ip1, -2);
    const fp_t Tim1 = temperature_si(
        w_im1(I(Prim::Pres)),
        w_im1(I(Prim::Rho)) / (eos.mass_per_h * m_p),
        eos.total_abund,
        ion_frac(eos, w_im1.idx)
    );

    return 0.5_fp * (Tip1 - Tim1);
}

template <typename FTraits>
KOKKOS_INLINE_FUNCTION fp_t compute_kappa(const fp_t m_p, const Eos& eos, const ThermalConductionContext& ctx, const QtyView& cell) {
    fp_t kappa = ctx.kappa0;
    if (ctx.spitzer) {
        using Prim = typename FTraits::prim;
        const fp_t temperature = temperature_si(
            cell(I(Prim::Pres)),
            cell(I(Prim::Rho)) / (eos.mass_per_h * m_p),
            eos.total_abund,
            ion_frac(eos, cell.idx)
        );
        kappa *= std::pow(temperature, 2.5_fp);
    }
    return kappa;
}

// TODO(cmo): This is only technically correct for the first evaluation, as the
// temperature is always computed from W, rather than Y. We could fix this by
// shuffling the array contents back and forth (if a bit inefficient).

template <typename FTraits, int Axis>
void explicit_thermal_flux_for_axis(const Simulation& sim, const ThermalConductionContext& ctx, const Fp3d& flux) {
    // NOTE(cmo): Overwrites the contents of flux, but not any guard cells.
    // Additionally, if reusing flux, it needs to be of size nz+1, ny+1, nx+1
    static_assert(Axis < 3, "Conductive flux only defined for 3 axes");
    static_assert(Axis < FTraits::num_dim, "Conductive flux axis cannot be larger than number of axes in problem");
    JasUnpack(sim, state, eos);
    JasUnpack(state, W, sz, dx);
    const fp_t m_p = state.p_mass;

    int nx = sz.xc - 2 * sz.ng;
    int ny = std::max(sz.yc - 2 * sz.ng, 1);
    int nz = std::max(sz.zc - 2 * sz.ng, 1);
    int dims[3] = {nx, ny, nz};
    dims[Axis] += 1;

    dex_parallel_for(
        "Conduction flux",
        FlatLoop<3>(dims[2], dims[1], dims[0]),
        KOKKOS_LAMBDA (int ki, int ji, int ii) {
            using Cons = typename FTraits::cons;
            using Prim = typename FTraits::prim;
            const int k = nz == 1 ? ki : ki + sz.ng;
            const int j = ny == 1 ? ji : ji + sz.ng;
            const int i = ii + sz.ng;
            constexpr int Ax2 = (Axis + 1) % 3;
            constexpr int Ax3 = (Axis + 2) % 3;
            vec3 dTdax(0.0_fp);

            CellIndex idx{
                .i=i,
                .j=j,
                .k=k
            };
            CellIndex idxm1 = shift_along<Axis>(idx, -1);
            QtyView w_i(W, idx);
            QtyView w_im1(W, idxm1);

            dTdax(Axis) = backwards_temperature_diff<FTraits, Axis>(m_p, eos, W, idx) / dx;
            if constexpr (Ax2 < FTraits::num_dim) {
                dTdax(Ax2) = mc4(
                    backwards_temperature_diff<FTraits, Ax2>(m_p, eos, W, shift_along<Ax2>(idx, 1)),
                    backwards_temperature_diff<FTraits, Ax2>(m_p, eos, W, idx),
                    backwards_temperature_diff<FTraits, Ax2>(m_p, eos, W, shift_along<Ax2>(idxm1, 1)),
                    backwards_temperature_diff<FTraits, Ax2>(m_p, eos, W, idxm1)
                ) / dx;
            }
            if constexpr (Ax3 < FTraits::num_dim) {
                dTdax(Ax3) = mc4(
                    backwards_temperature_diff<FTraits, Ax3>(m_p, eos, W, shift_along<Ax3>(idx, 1)),
                    backwards_temperature_diff<FTraits, Ax3>(m_p, eos, W, idx),
                    backwards_temperature_diff<FTraits, Ax3>(m_p, eos, W, shift_along<Ax3>(idxm1, 1)),
                    backwards_temperature_diff<FTraits, Ax3>(m_p, eos, W, idxm1)
                ) / dx;
            }

            const fp_t gradT_norm = std::sqrt(square(dTdax(0)) + square(dTdax(1)) + square(dTdax(2)));
            const fp_t kappa = 0.5_fp * (
                compute_kappa<FTraits>(m_p, eos, ctx, w_i) + compute_kappa<FTraits>(m_p, eos, ctx, w_im1)
            );

            fp_t full_flux = 0.0_fp;
            fp_t full_flux_norm = 0.0_fp;
            if (ctx.anisotropic) {
                vec3 B(0.0_fp);
                B(0) = 0.5_fp * (w_i(I(Prim::Bx)) + w_im1(I(Prim::Bx)));
                if constexpr (FTraits::num_dim > 1) {
                    B(1) = 0.5_fp * (w_i(I(Prim::By)) + w_im1(I(Prim::By)));
                }
                if constexpr (FTraits::num_dim > 2) {
                    B(2) = 0.5_fp * (w_i(I(Prim::Bz)) + w_im1(I(Prim::Bz)));
                }
                const fp_t B_norm = std::max(
                    std::sqrt(square(B(0)) + square(B(1)) + square(B(2))),
                    1e-20_fp
                );
                const fp_t b_ax = B(Axis) / B_norm;
                const fp_t b_dot_gradT = (
                    B(0) * dTdax(0) + B(1) * dTdax(1) + B(2) * dTdax(2)
                ) / B_norm;
                full_flux = -kappa * b_dot_gradT * b_ax;
                full_flux_norm = std::abs(kappa * b_dot_gradT);
            } else {
                full_flux = -kappa * dTdax(Axis);
                full_flux_norm = kappa * gradT_norm;
            }

            fp_t sat_fac = 1.0_fp;
            if (ctx.saturate) {
                // NOTE(cmo): upwind the limited flux, as per Mignone+ 2012, with
                // averaging for the case of Spitzer flux = 0
                fp_t mean_rho = 0.5_fp * (w_i(I(Prim::Rho)) + w_im1(I(Prim::Rho)));
                fp_t upwind_pressure;
                if (full_flux > 0.0_fp) {
                    upwind_pressure = w_im1(I(Prim::Pres));
                } else if (full_flux < 0.0_fp) {
                    upwind_pressure = w_i(I(Prim::Pres));
                } else {
                    upwind_pressure = 0.5_fp * (w_i(I(Prim::Pres)) + w_im1(I(Prim::Pres)));
                }
                // NOTE(cmo): Cowie & McKee 1977 form
                const fp_t sat_flux = 5.0_fp * ctx.saturation_phi * std::sqrt(upwind_pressure / mean_rho) * upwind_pressure;
                sat_fac = sat_flux / (sat_flux + full_flux_norm);
            }
            flux(k, j, i) = sat_fac * full_flux;
        }
    );
}

template <typename FTraits>
void explicit_thermal_cond(const Simulation& sim, const ThermalConductionContext& ctx, const Fp3d& flux_div) {
    // Adds the divergence of the explicit thermal flux to the flux_div array. N.B. does not zero flux_div.

    JasUnpack(sim, state);
    JasUnpack(state, sz, dx);
    Fp3d flux(
        "conduction_flux",
        sz.zc + 1,
        sz.yc + 1,
        sz.xc + 1
    );

    int nx = sz.xc - 2 * sz.ng;
    int ny = std::max(sz.yc - 2 * sz.ng, 1);
    int nz = std::max(sz.zc - 2 * sz.ng, 1);
    const fp_t inv_dx = 1.0_fp / dx;

    explicit_thermal_flux_for_axis<FTraits, 0>(sim, ctx, flux);
    Kokkos::fence();
    // NOTE(cmo): We could compute the flux directly into flux div, if we zero'd
    // it first, and then used atomic ops, but that's extra effort!
    dex_parallel_for(
        "Accumulate thermal flux div",
        FlatLoop<3>(nz, ny, nx),
        KOKKOS_LAMBDA (int ki, int ji, int ii) {
            const int k = nz == 1 ? ki : ki + sz.ng;
            const int j = ny == 1 ? ji : ji + sz.ng;
            const int i = ii + sz.ng;

            flux_div(k, j, i) += inv_dx * (flux(k, j, i) - flux(k, j, i+1));
        }
    );
    Kokkos::fence();


    if constexpr (FTraits::num_dim > 1) {
        explicit_thermal_flux_for_axis<FTraits, 1>(sim, ctx, flux);
        Kokkos::fence();
        dex_parallel_for(
            "Accumulate thermal flux div",
            FlatLoop<3>(nz, ny, nx),
            KOKKOS_LAMBDA (int ki, int ji, int ii) {
                const int k = nz == 1 ? ki : ki + sz.ng;
                const int j = ny == 1 ? ji : ji + sz.ng;
                const int i = ii + sz.ng;

                flux_div(k, j, i) += inv_dx * (flux(k, j, i) - flux(k, j+1, i));
            }
        );
        Kokkos::fence();
    }

    if constexpr (FTraits::num_dim > 2) {
        explicit_thermal_flux_for_axis<FTraits, 2>(sim, ctx, flux);
        Kokkos::fence();
        dex_parallel_for(
            "Accumulate thermal flux div",
            FlatLoop<3>(nz, ny, nx),
            KOKKOS_LAMBDA (int ki, int ji, int ii) {
                const int k = nz == 1 ? ki : ki + sz.ng;
                const int j = ny == 1 ? ji : ji + sz.ng;
                const int i = ii + sz.ng;

                flux_div(k, j, i) += inv_dx * (flux(k, j, i) - flux(k+1, j, i));
            }
        );
        Kokkos::fence();
    }
}

template <typename FTraits>
fp_t estimate_thermal_conduction_timestep(const Simulation& sim, const ThermalConductionContext& ctx) {
    JasUnpack(sim, state, eos);
    JasUnpack(state, W, sz, dx);
    const fp_t m_p = state.p_mass;
    int nx = sz.xc - 2 * sz.ng + 1;
    int ny = std::max(sz.yc - 2 * sz.ng + 1, 1);
    int nz = std::max(sz.zc - 2 * sz.ng + 1, 1);

    const fp_t cfl_fac = sim.max_cfl * 0.5_fp / fp_t(FTraits::num_dim);

    fp_t dt_max = 1e5_fp;
    dex_parallel_reduce(
        "Conductive dt",
        FlatLoop<3>(nz, ny, nx),
        KOKKOS_LAMBDA (int ki, int ji, int ii, fp_t& running_dt) {
            using Cons = typename FTraits::cons;
            using Prim = typename FTraits::prim;
            const int k = nz == 1 ? ki : ki + sz.ng;
            const int j = ny == 1 ? ji : ji + sz.ng;
            const int i = ii + sz.ng;
            CellIndex idx{
                .i=i,
                .j=j,
                .k=k
            };
            QtyView w_i(W, idx);
            const fp_t kappa = compute_kappa<FTraits>(m_p, eos, ctx, w_i);
            const fp_t Ti = temperature_si(
                w_i(I(Prim::Pres)),
                w_i(I(Prim::Rho)) / (eos.mass_per_h * m_p),
                eos.total_abund,
                ion_frac(eos, w_i.idx)
            );
            // e_int = rho cv T = P / (gamma - 1)
            const fp_t rho_cv = w_i(I(Prim::Pres)) / ((eos.gamma - 1.0_fp) * Ti);

            if (!ctx.anisotropic) {
                running_dt = std::min(running_dt, rho_cv * square(dx) / kappa);
            }

            if constexpr (!FTraits::is_mhd) {
                return;
            }

            const fp_t dTdx = centred_temperature_diff<FTraits, 0>(m_p, eos, W, idx) / dx;
            const fp_t dTdy = (FTraits::num_dim > 1) ? centred_temperature_diff<FTraits, 1>(m_p, eos, W, idx) / dx : 0.0_fp;
            const fp_t dTdz = (FTraits::num_dim > 2) ? centred_temperature_diff<FTraits, 2>(m_p, eos, W, idx) / dx : 0.0_fp;
            const fp_t gradT_norm = std::sqrt(square(dTdx) + square(dTdy) + square(dTdz));

            if (gradT_norm == 0.0_fp) {
                return;
            }

            const fp_t Bx = w_i(I(Prim::Bx));
            const fp_t By = w_i(I(Prim::By));
            const fp_t Bz = w_i(I(Prim::Bz));
            const fp_t B_norm = std::sqrt(square(Bx) + square(By) + square(Bz));

            if (B_norm == 0.0_fp) {
                return;
            }
            fp_t full_flux = kappa * gradT_norm;
            fp_t sat_fac = 1.0_fp;
            if (ctx.saturate) {
                // NOTE(cmo): Cowie & McKee 1977 form
                const fp_t sat_flux = 5.0_fp * ctx.saturation_phi * std::sqrt(w_i(Prim::Pres) / w_i(Prim::Rho)) * w_i(I(Prim::Pres));
                sat_fac = sat_flux / (sat_flux + full_flux);
            }

            const fp_t cos_theta = std::abs(Bx * dTdx + By * dTdy + Bz * dTdz) / (B_norm * gradT_norm);
            running_dt = std::min(
                running_dt,
                rho_cv * square(dx) / (kappa * std::abs(Bx) / B_norm * cos_theta + 1e-20_fp)
            );
            if constexpr (FTraits::num_dim > 1) {
                running_dt = std::min(
                    running_dt,
                    rho_cv * square(dx) / (kappa * std::abs(By) / B_norm * cos_theta + 1e-20_fp)
                );
            }
            if constexpr (FTraits::num_dim > 2) {
                running_dt = std::min(
                    running_dt,
                    rho_cv * square(dx) / (kappa * std::abs(Bz) / B_norm * cos_theta + 1e-20_fp)
                );
            }
        },
        Kokkos::Min<fp_t>(dt_max)
    );
    Kokkos::fence();

    return cfl_fac * dt_max;
}

template <typename FTraits, typename W>
KOKKOS_INLINE_FUNCTION fp_t prim_to_eint(const fp_t gamma, const W& w) {
    using Prim = FTraits::prim;

    return w(I(Prim::Pres)) / (gamma - 1.0_fp);
}

template <typename FTraits>
void thermal_conduction_kernel(const Simulation& sim, const ThermalConductionContext& ctx) {
    JasUnpack(sim, state, eos, sources, dt_sub);
    JasUnpack(state, W, sz);
    using Cons = FTraits::cons;
    using Prim = FTraits::prim;
    if constexpr (FTraits::has_hypertc) {
        throw std::runtime_error("Cannot use classic thermal conduction on a fluid with hypertc");
    }

    if (!ctx.use_sts) {
        return explicit_thermal_cond<FTraits>(
            sim,
            ctx,
            Fp3d(
                "Energy source",
                &sources.S(I(Cons::Ene), 0, 0, 0),
                sources.S.extent(1),
                sources.S.extent(2),
                sources.S.extent(3)
            )
        );
    }

    const fp_t dt_para = estimate_thermal_conduction_timestep<FTraits>(sim, ctx);
    int n_stages = int(std::ceil(
        0.5_fp * (std::sqrt(9.0_fp + 16.0_fp * (dt_sub / dt_para)) - 1.0_fp)
    ));
    if (n_stages % 2 == 0) {
        n_stages += 1;
    }

    constexpr bool verbose = false;
    if (verbose) {
        fmt::println("STS Stages: {}, dt {}, dt_para {}", n_stages, dt_sub, dt_para);
    }

    if (n_stages <= 1) {
        return explicit_thermal_cond<FTraits>(
            sim,
            ctx,
            Fp3d(
                "Energy source",
                &sources.S(I(Cons::Ene), 0, 0, 0),
                sources.S.extent(1),
                sources.S.extent(2),
                sources.S.extent(3)
            )
        );
    }

    if (n_stages > 150) {
        fmt::println("More than 150 STS substeps!");
    }

    // NOTE(cmo): Constantly reallocating these small arrays isn't ideal,
    // especially as they may cause the linear allocator to move the big ones
    // like Y. Allocate those first to try and minimise this impact
    Fp1dHost ah("a", n_stages+1);
    Fp1dHost bh("b", n_stages+1);
    Fp1dHost mu_tilde_h("mu_tilde", n_stages+1);
    Fp1dHost muh("mu", n_stages+1);
    Fp1dHost nuh("nu", n_stages+1);
    Fp1dHost gamma_tilde_h("gamma_tilde", n_stages+1);

    for (int j = 0; j < n_stages + 1; ++j) {
        if (j < 3) {
            bh(j) = 1.0_fp / 3.0_fp;
        } else {
            bh(j) = (square(j) + j - 2.0_fp) / (2.0_fp * j * (j + 1.0_fp));
        }
        ah(j) = 1.0_fp - bh(j);
    }
    const fp_t omega_1 = 4.0_fp / (square(n_stages) + n_stages - 2.0_fp);
    mu_tilde_h(1) = omega_1 / 3.0_fp;

    // The first 1/2 entries of many of these arrays aren't used
    for (int j = 2; j < n_stages + 1; ++j) {
        const fp_t fac = (2.0_fp * j - 1.0_fp) / j;
        mu_tilde_h(j) = fac * omega_1 * (bh(j) / bh(j-1));
        muh(j) = fac * (bh(j) / bh(j-1));
        nuh(j) = (1.0_fp - j) / fp_t(j) * (bh(j) / bh(j-2));
        gamma_tilde_h(j) = -ah(j-1) * mu_tilde_h(j);
    }

    Fp4d Y("STS_Y", 4, sz.zc, sz.yc, sz.xc);
    Fp3d LcY0("STS_LcY0", sz.zc, sz.yc, sz.xc);
    Fp3d flux_div("STS_flux_div", sz.zc, sz.yc, sz.xc);

    Fp1d a = ah.createDeviceCopy();
    Fp1d b = bh.createDeviceCopy();
    Fp1d mu_tilde = mu_tilde_h.createDeviceCopy();
    Fp1d mu = muh.createDeviceCopy();
    Fp1d nu = nuh.createDeviceCopy();
    Fp1d gamma_tilde = gamma_tilde_h.createDeviceCopy();

    // NOTE(cmo): We rely on Y(0) to hold the original e_int
    dex_parallel_for(
        "Initialise STS Arrays",
        FlatLoop<3>(sz.zc, sz.yc, sz.xc),
        KOKKOS_LAMBDA (int k, int j, int i) {
            flux_div(k, j, i) = 0.0_fp;
            LcY0(k, j, i) = 0.0_fp;
            const fp_t e_int = prim_to_eint<FTraits>(
                eos.gamma,
                QtyView(W, CellIndex{.i=i, .j=j, .k=k})
            );
            for (int x = 0; x < Y.extent(0); ++x) {
                Y(x, k, j, i) = e_int;
            }
        }
    );
    Kokkos::fence();

    int nx = sz.xc - 2 * sz.ng;
    int ny = std::max(sz.yc - 2 * sz.ng, 1);
    int nz = std::max(sz.zc - 2 * sz.ng, 1);

    // NOTE(cmo): First STS stage
    explicit_thermal_cond<FTraits>(sim, ctx, flux_div);
    dex_parallel_for(
        "STS Step 1",
        FlatLoop<3>(nz, ny, nx),
        KOKKOS_LAMBDA (int ki, int ji, int ii) {
            const int k = nz == 1 ? ki : ki + sz.ng;
            const int j = ny == 1 ? ji : ji + sz.ng;
            const int i = ii + sz.ng;

            LcY0(k, j, i) = flux_div(k, j, i);
            const fp_t c0 = mu_tilde(1) * dt_sub * LcY0(k, j, i);
            Y(2, k, j, i) = Y(0, k, j, i) + c0;
            flux_div(k, j, i) = 0.0_fp;

            // NOTE(cmo): Store the pressure due to the updated e_int
            W(I(Prim::Pres), k, j, i) = Y(2, k, j, i) * (eos.gamma - 1.0_fp);
        }
    );
    Kokkos::fence();

    // NOTE(cmo): Remaining STS stages
    for (int sj = 2; sj < n_stages + 1; ++sj) {
        explicit_thermal_cond<FTraits>(sim, ctx, flux_div);
        dex_parallel_for(
            "STS Later Step",
            FlatLoop<3>(nz, ny, nx),
            KOKKOS_LAMBDA (int ki, int ji, int ii) {
                const int k = nz == 1 ? ki : ki + sz.ng;
                const int j = ny == 1 ? ji : ji + sz.ng;
                const int i = ii + sz.ng;

                const fp_t c0 = gamma_tilde(sj) * dt_sub * LcY0(k, j, i);
                const fp_t LcYj1 = flux_div(k, j, i);
                const fp_t c1 = mu_tilde(sj) * dt_sub * LcYj1;
                Y(3, k, j, i) = (
                    mu(sj) * Y(2, k, j, i)
                    + nu(sj) * Y(1, k, j, i)
                    + (1.0_fp - mu(sj) - nu(sj)) * Y(0, k, j, i)
                    + c1 + c0
                );
                flux_div(k, j, i) = 0.0_fp;

                if (sj < n_stages) {
                    // NOTE(cmo): Shuffle the terms down for the next loop
                    Y(1, k, j, i) = Y(2, k, j, i);
                    Y(2, k, j, i) = Y(3, k, j, i);
                    Y(3, k, j, i) = 0.0_fp;

                    // NOTE(cmo): Store the pressure due to the updated e_int
                    W(I(Prim::Pres), k, j, i) = Y(2, k, j, i) * (eos.gamma - 1.0_fp);
                }
            }
        );
        Kokkos::fence();
    }

    dex_parallel_for(
        "STS deltaE",
        FlatLoop<3>(nz, ny, nx),
        KOKKOS_LAMBDA (int ki, int ji, int ii) {
            const int k = nz == 1 ? ki : ki + sz.ng;
            const int j = ny == 1 ? ji : ji + sz.ng;
            const int i = ii + sz.ng;

            const fp_t e_int = Y(0, k, j, i);
            const fp_t delta_E = (Y(3, k, j, i) - e_int);
            sources.S(I(Cons::Ene), k, j, i) += delta_E / dt_sub;

            // NOTE(cmo): Restore the pressure in W for other source terms to use.
            W(I(Prim::Pres), k, j, i) = e_int * (eos.gamma - 1.0_fp);
        }
    );
    Kokkos::fence();
}

void setup_thermal_conduction(Simulation& sim, YAML::Node& config) {
    auto ctx = std::make_shared<ThermalConductionContext>(ThermalConductionContext{
        .enable = get_or<bool>(config, "sources.thermal_conduction.enable", false),
        .use_sts = get_or<bool>(config, "sources.thermal_conduction.use_sts", true),
        .saturate = get_or<bool>(config, "sources.thermal_conduction.saturate", false),
        .spitzer = get_or<bool>(config, "sources.thermal_conduction.spitzer", true),
        .anisotropic = get_or<bool>(config, "sources.thermal_conduction.anisotropic", is_instance(sim.fluid_type, FluidType::Mhd)),
        .saturation_phi = get_or<fp_t>(config, "sources.thermal_conduction.saturation_phi", 1.1_fp),
        .kappa0 = get_or<fp_t>(config, "sources.thermal_conduction.kappa0", 8e-12_fp),
    });

    if (source_term_index(sim, "thermal_conduction") != sim.compute_source_terms.size()) {
        throw std::runtime_error("Source \"thermal_conduction\" already registered.");
    }

    sim.compute_source_terms.push_back(SourceTerm{
        .name = "thermal_conduction",
        .fn = invoke_fluid_traits(
            sim.num_dim,
            sim.fluid_type,
            [=]<typename FTraits>(FTraits) -> std::function<void(const Simulation&)> {
                return [=] (const Simulation& sim) {
                    return thermal_conduction_kernel<FTraits>(sim, *ctx);
                };
            }
        ),
        .get_context = [=]() { return ctx.get(); }
    });
}

}