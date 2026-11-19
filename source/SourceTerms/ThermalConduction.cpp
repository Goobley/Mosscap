#include "ThermalConduction.hpp"
#include "../Simulation.hpp"
#include "../MosscapConfig.hpp"
#include "../SourceTerms.hpp"
#include "ConductionCommon.hpp"
#include "SemiImplicitConduction.hpp"


namespace Mosscap {

// NOTE(cmo): This roughly follows athenapk, but I've tried to make the flow a little clearer
// Thus, it essentially follows PLUTO (Mignone+ 2012)

template <typename FTraits, int Axis>
KOKKOS_INLINE_FUNCTION fp_t backwards_temperature_diff(const fp_t m_p, const Eos& eos, const Fp4d& W, const CellIndex& from) {
    QtyView w_i(W, from);
    const fp_t Ti = cell_temperature<FTraits>(m_p, eos, w_i);
    const fp_t Tim1 = cell_temperature<FTraits>(m_p, eos, shift_along<Axis>(w_i, -1));
    return Ti - Tim1;
}

template <typename FTraits>
KOKKOS_INLINE_FUNCTION fp_t compute_kappa(const fp_t m_p, const Eos& eos, const ThermalConductionContext& ctx, const QtyView& cell) {
    if (!ctx.spitzer) {
        return ctx.kappa0;
    }
    return conductivity(ctx, cell_temperature<FTraits>(m_p, eos, cell));
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
                const vec3 B_i = conduction_field<FTraits>(ctx, w_i);
                const vec3 B_im1 = conduction_field<FTraits>(ctx, w_im1);
                vec3 B;
                for (int a = 0; a < 3; ++a) {
                    B(a) = 0.5_fp * (B_i(a) + B_im1(a));
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
            const fp_t Ti = cell_temperature<FTraits>(m_p, eos, w_i);
            // e_int = rho cv T = P / (gamma - 1)
            const fp_t rho_cv = w_i(I(Prim::Pres)) / ((eos.gamma - 1.0_fp) * Ti);

            if (!ctx.anisotropic) {
                running_dt = std::min(running_dt, rho_cv * square(dx) / kappa);
            }

            if constexpr (!FTraits::is_mhd) {
                return;
            }

            const vec3 B = conduction_field<FTraits>(ctx, w_i);
            const fp_t B_norm = std::sqrt(square(B(0)) + square(B(1)) + square(B(2)));

            if (B_norm == 0.0_fp) {
                return;
            }
            // NOTE(cmo): Do this for the limit per axis (different to athenapk)
            running_dt = std::min(
                running_dt,
                rho_cv * square(dx) / (kappa * std::abs(B(0)) / B_norm + 1e-20_fp)
            );
            if constexpr (FTraits::num_dim > 1) {
                running_dt = std::min(
                    running_dt,
                    rho_cv * square(dx) / (kappa * std::abs(B(1)) / B_norm + 1e-20_fp)
                );
            }
            if constexpr (FTraits::num_dim > 2) {
                running_dt = std::min(
                    running_dt,
                    rho_cv * square(dx) / (kappa * std::abs(B(2)) / B_norm + 1e-20_fp)
                );
            }
        },
        Kokkos::Min<fp_t>(dt_max)
    );
    Kokkos::fence();

    return cfl_fac * dt_max;
}

template <typename FTraits>
void thermal_conduction_kernel(const Simulation& sim, ThermalConductionContext& ctx) {
    JasUnpack(sim, state, eos, sources, dt_sub);
    JasUnpack(state, W, sz);
    using Cons = FTraits::cons;
    using Prim = FTraits::prim;
    if constexpr (FTraits::has_hypertc) {
        throw std::runtime_error("Cannot use classic thermal conduction on a fluid with hypertc");
    }
    if (ctx.method == ConductionMethod::SemiImp) {
        return semi_imp_conduction_kernel<FTraits>(sim, ctx);
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
    fill_sts_pressure_ghosts<FTraits>(sim);

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
        if (sj < n_stages) {
            fill_sts_pressure_ghosts<FTraits>(sim);
        }
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
        }
    );
    // NOTE(cmo): Restore the pressure in W for other source terms to use.
    // NOTE(claude): Ghosts included, as the stages overwrite them too.
    dex_parallel_for(
        "STS restore pressure",
        FlatLoop<3>(sz.zc, sz.yc, sz.xc),
        KOKKOS_LAMBDA (int k, int j, int i) {
            W(I(Prim::Pres), k, j, i) = Y(0, k, j, i) * (eos.gamma - 1.0_fp);
        }
    );
    Kokkos::fence();
}

void setup_thermal_conduction(Simulation& sim, YAML::Node& config) {
    auto ctx = std::make_shared<ThermalConductionContext>(ThermalConductionContext{
        .enable = get_or<bool>(config, "sources.thermal_conduction.enable", false),
        .use_sts = get_or<bool>(config, "sources.thermal_conduction.use_sts", true),
        .anisotropic = get_or<bool>(config, "sources.thermal_conduction.anisotropic", is_instance(sim.fluid_type, FluidType::Mhd)),
    });
    const auto& cond = sim.state.cond;
    ctx->saturate = cond.saturate;
    ctx->spitzer = cond.spitzer;
    ctx->saturation_phi = cond.saturation_phi;
    ctx->kappa0 = cond.kappa0;
    ctx->out_of_plane_field = cond.out_of_plane_field;
    ctx->kappa_floor = cond.kappa_floor;

    // method: sts | explicit | semi_implicit (default from use_sts, for older configs)
    const std::string method = get_or<std::string>(
        config,
        "sources.thermal_conduction.method",
        ctx->use_sts ? "sts" : "explicit"
    );

    if (method == "sts") {
        ctx->method = ConductionMethod::Sts;
    } else if (method == "explicit") {
        ctx->method = ConductionMethod::Explicit;
        ctx->use_sts = false;
    } else if (method == "ye") {
        throw std::runtime_error("thermal_conduction.method: ye has been renamed to semi_implicit (with its options under sources.thermal_conduction.semi_implicit)");
    } else if (method == "semi_implicit") {
        ctx->method = ConductionMethod::SemiImp;
        const std::string cross = get_or<std::string>(
            config,
            "sources.thermal_conduction.semi_implicit.cross",
            "sharma_symmetric"
        );
        if (cross == "sharma_symmetric") {
            ctx->semi_imp_cross = SemiImpCross::SharmaSymmetric;
        } else if (cross == "mean") {
            ctx->semi_imp_cross = SemiImpCross::Mean;
        } else {
            throw std::runtime_error(fmt::format("Unknown thermal_conduction.semi_implicit.cross \"{}\" (sharma_symmetric | mean)", cross));
        }

        ctx->semi_imp_s_max = get_or<fp_t>(config, "sources.thermal_conduction.semi_implicit.s_max", ctx->semi_imp_s_max);

        const FluidTraitsRt traits(sim.num_dim, sim.fluid_type);
        if (!traits.is_mhd || traits.has_hypertc || sim.num_dim != 2) {
            throw std::runtime_error("thermal_conduction.method: semi_implicit needs a 2D fluid_type mhd or glmmhd");
        }

        if (!ctx->anisotropic) {
            throw std::runtime_error("thermal_conduction.method: semi_implicit is field-aligned only (anisotropic: true)");
        }
    } else {
        throw std::runtime_error(fmt::format("Unknown thermal_conduction.method \"{}\" (sts | explicit | semi_implicit)", method));
    }

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