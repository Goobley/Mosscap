#include "ProblemGenerator.hpp"
#include "../Hydro.hpp"
#include "../MosscapConfig.hpp"
#include "../SourceTerms.hpp"
#include "../SourceTerms/TownsendThinLoss.hpp"
#include "../SourceTerms/ReplaceSmallValues.hpp"
#include "../SourceTerms/ThermalConduction.hpp"
#include "../AnalyticLteH.hpp"

// NOTE(claude): This is a 2d problem
static constexpr int num_dim = 2;

namespace Mosscap {

// Loop top with spatially varying FIP bias (cf. Benavitz et al. 2025, ApJ 992, 4).
//
// A horizontal field (x) threads a uniform temperature corona at rest. A strand
// (band in y) holds evaporated material at density contrast `density_contrast`
// relative to the background, with the extra gas pressure balanced by reducing
// Bx inside the strand (Bx depends only on y, so div B = 0). Along the strand a
// core of fractionated plasma (fip_core) sits between evaporated photospheric
// material (fip_flank); the background carries fip_background.
//
// Heating is fixed in time, varying only in y:
//     H = [(1 - s) H_bg + s n_H n_e Λ(T_0, f_ref(y))] (ρ / ρ_0(y))^a
// where H_bg balances the background, ρ_0 is the initial density, and f_ref(y)
// blends from fip_background to heating.fip_reference in the strand.
// s = 0 heats uniformly at the background rate (the strand cools as a whole),
// s = 1 puts the strand in equilibrium at f = fip_reference, so only cells with
// f != fip_reference are out of balance.
//
// Requires an MHD fluid, sources.thin_loss_fip, and a tracer
// (simulation.n_extra_fields) for the FIP bias. Supports the ideal (fixed
// ionisation) and analytic LTE H equations of state.
//
// problem:
//   name: fip_looptop_2d
//   temperature: 1e6               # [K]
//   background_nh: 1e15            # total hydrogen number density [m-3]
//   b0: 2e-3                       # background Bx [T]
//   density_contrast: 10.0         # strand / background
//   strand_y0: 0.0                 # [m]
//   strand_width: 2e6              # [m]
//   strand_tr_width: 2e5           # [m]
//   core_x0: 0.0                   # [m]
//   core_width: 4e6                # [m]
//   core_tr_width: 2e5             # [m]
//   fip_core: 4.0
//   fip_flank: 1.0
//   fip_background: 4.0
//   heating:
//     strand_balance: 0.0          # s
//     fip_reference: <fip_flank>   # f_ref in the strand
//     density_exponent: 0.0        # a

struct FipLooptopParams {
    fp_t T0;
    fp_t nh_bg;
    fp_t b0;
    fp_t density_contrast;
    fp_t strand_y0;
    fp_t strand_width;
    fp_t strand_tr_width;
    fp_t core_x0;
    fp_t core_width;
    fp_t core_tr_width;
    fp_t fip_core;
    fp_t fip_flank;
    fp_t fip_bg;
    fp_t strand_balance;
    fp_t fip_ref;
    fp_t density_exponent;

    /// 1 inside the strand, 0 in the background
    KOKKOS_INLINE_FUNCTION fp_t strand_mask(fp_t y) const {
        return 0.5_fp * (
            std::tanh((y - strand_y0 + 0.5_fp * strand_width) / strand_tr_width)
            - std::tanh((y - strand_y0 - 0.5_fp * strand_width) / strand_tr_width)
        );
    }

    /// 1 in the fractionated core, 0 in the flanks
    KOKKOS_INLINE_FUNCTION fp_t core_mask(fp_t x) const {
        return 0.5_fp * (
            std::tanh((x - core_x0 + 0.5_fp * core_width) / core_tr_width)
            - std::tanh((x - core_x0 - 0.5_fp * core_width) / core_tr_width)
        );
    }

    KOKKOS_INLINE_FUNCTION fp_t nh(fp_t y) const {
        return nh_bg * (1.0_fp + (density_contrast - 1.0_fp) * strand_mask(y));
    }

    KOKKOS_INLINE_FUNCTION fp_t fip_bias(fp_t x, fp_t y) const {
        const fp_t f_strand = fip_flank + core_mask(x) * (fip_core - fip_flank);
        return fip_bg + strand_mask(y) * (f_strand - fip_bg);
    }

    KOKKOS_INLINE_FUNCTION fp_t heating_fip_bias(fp_t y) const {
        return fip_bg + strand_mask(y) * (fip_ref - fip_bg);
    }
};

static FipLooptopParams load_params(const YAML::Node& config) {
    FipLooptopParams p;
    p.T0 = get_or<fp_t>(config, "problem.temperature", 1e6_fp);
    p.nh_bg = get_or<fp_t>(config, "problem.background_nh", 1e15_fp);
    p.b0 = get_or<fp_t>(config, "problem.b0", 2e-3_fp);
    p.density_contrast = get_or<fp_t>(config, "problem.density_contrast", 10.0_fp);
    p.strand_y0 = get_or<fp_t>(config, "problem.strand_y0", 0.0_fp);
    p.strand_width = get_or<fp_t>(config, "problem.strand_width", 2e6_fp);
    p.strand_tr_width = get_or<fp_t>(config, "problem.strand_tr_width", 2e5_fp);
    p.core_x0 = get_or<fp_t>(config, "problem.core_x0", 0.0_fp);
    p.core_width = get_or<fp_t>(config, "problem.core_width", 4e6_fp);
    p.core_tr_width = get_or<fp_t>(config, "problem.core_tr_width", 2e5_fp);
    p.fip_core = get_or<fp_t>(config, "problem.fip_core", 4.0_fp);
    p.fip_flank = get_or<fp_t>(config, "problem.fip_flank", 1.0_fp);
    p.fip_bg = get_or<fp_t>(config, "problem.fip_background", 4.0_fp);
    p.strand_balance = get_or<fp_t>(config, "problem.heating.strand_balance", 0.0_fp);
    p.fip_ref = get_or<fp_t>(config, "problem.heating.fip_reference", p.fip_flank);
    p.density_exponent = get_or<fp_t>(config, "problem.heating.density_exponent", 0.0_fp);

    if (p.density_contrast <= 0.0_fp || p.T0 <= 0.0_fp || p.nh_bg <= 0.0_fp) {
        throw std::runtime_error("fip_looptop_2d: temperature, background_nh and density_contrast must be positive.");
    }
    if (p.strand_balance < 0.0_fp || p.strand_balance > 1.0_fp) {
        throw std::runtime_error("fip_looptop_2d: heating.strand_balance must lie in [0, 1].");
    }
    return p;
}

/// Ionisation fraction at (nh, T) consistent with the EOS in use.
struct IonFrac {
    bool lte;
    fp_t y_const;

    KOKKOS_INLINE_FUNCTION fp_t operator()(fp_t nh, fp_t T) const {
        if (lte) {
            return y_from_nhtot(nh, T);
        }
        return y_const;
    }
};

static IonFrac get_ion_frac(const Simulation& sim) {
    const auto& eos = sim.eos;
    if (eos.type == EosType::AnalyticLteH) {
        return IonFrac{.lte = true, .y_const = 0.0_fp};
    }
    if (eos.type == EosType::Ideal) {
        return IonFrac{.lte = false, .y_const = eos.y};
    }
    throw std::runtime_error("fip_looptop_2d only supports the ideal and analyticlteh equations of state.");
}

template <typename Fluid>
static void initial_conditions(Simulation& sim, const YAML::Node& config, const FipLooptopParams& params, int tracer_idx) {
    using Prim = typename Fluid::prim;
    using Cons = typename Fluid::cons;
    constexpr int n_hydro = Fluid::num_vars;
    constexpr fp_t k_B = ConstantsF64::k_B;
    constexpr fp_t m_u = ConstantsF64::u;
    constexpr fp_t chi_H = 2.178710282685096e-18_fp; // [J]
    const auto& state = sim.state;
    const auto& sz = state.sz;
    const auto& eos = sim.eos;
    const fp_t mu0 = state.mu0;

    const IonFrac ion_frac = get_ion_frac(sim);
    AnalyticLteH lte_eos;
    lte_eos.init(get_or<bool>(config, "eos.include_ionisation_energy", false));

    const fp_t y_bg = ion_frac(params.nh_bg, params.T0);
    const fp_t p_bg = (eos.total_abund + y_bg) * params.nh_bg * k_B * params.T0;
    const fp_t b0 = params.b0;
    // NOTE(claude): Total pressure balance across the field needs Bx^2 > 0 everywhere.
    {
        const fp_t nh_max = params.nh(params.strand_y0);
        const fp_t y_max = ion_frac(nh_max, params.T0);
        const fp_t p_max = (eos.total_abund + y_max) * nh_max * k_B * params.T0;
        const fp_t bx2_min = square(b0) - 2.0_fp * mu0 * (p_max - p_bg);
        if (bx2_min <= 0.0_fp) {
            throw std::runtime_error(fmt::format(
                "fip_looptop_2d: b0 = {:e} T is too weak to confine the strand (needs > {:e} T).",
                b0,
                std::sqrt(2.0_fp * mu0 * (p_max - p_bg))
            ));
        }
        fmt::println(
            "fip_looptop_2d: background p = {:.3e} Pa, beta = {:.3e}; strand core Bx = {:.3e} T",
            p_bg,
            2.0_fp * mu0 * p_bg / square(b0),
            std::sqrt(bx2_min)
        );
    }

    const fp_t mass_per_h = eos.mass_per_h;
    const fp_t total_abund = eos.total_abund;
    const bool has_ion_e = eos.has_ion_e;
    dex_parallel_for(
        "fip_looptop_2d ICs",
        FlatLoop<3>(sz.zc, sz.yc, sz.xc),
        KOKKOS_LAMBDA (int k, int j, int i) {
            const vec3 pos = state.get_pos(i, j, k);
            const fp_t T = params.T0;
            const fp_t nh = params.nh(pos(1));
            const fp_t y = ion_frac(nh, T);

            yakl::SArray<fp_t, 1, n_hydro> w(0.0_fp);
            w(I(Prim::Rho)) = nh * mass_per_h * m_u;
            w(I(Prim::Pres)) = (total_abund + y) * nh * k_B * T;
            w(I(Prim::Bx)) = std::sqrt(square(b0) - 2.0_fp * mu0 * (w(I(Prim::Pres)) - p_bg));
            if (ion_frac.lte) {
                w(I(Prim::IonE)) = lte_eos.ionisation_energy(eos, w(I(Prim::Rho)), y, T) / w(I(Prim::Rho));
            } else if (has_ion_e) {
                w(I(Prim::IonE)) = y * chi_H / (m_u * mass_per_h);
            }

            CellIndex idx{.i = i, .j = j, .k = k};
            auto q = QtyView(state.Q, idx);
            prim_to_cons<Fluid>(eos.gamma, mu0, w, q);
            q(tracer_idx) = q(I(Cons::Rho)) * params.fip_bias(pos(0), pos(1));
        }
    );
    Kokkos::fence();
}

struct FipLooptopHeating {
    /// Heating rate at the initial density [W m-3], indexed by j
    Fp1d heating;
    /// Initial density, indexed by j
    Fp1d rho0;
    fp_t density_exponent;
};

static FipLooptopHeating compute_heating(const Simulation& sim, const FipLooptopParams& params, const FipThinLossContext& loss_ctx) {
    constexpr fp_t m_u = ConstantsF64::u;
    const auto& state = sim.state;
    const auto& sz = state.sz;
    const IonFrac ion_frac = get_ion_frac(sim);
    const fp_t mass_per_h = sim.eos.mass_per_h;

    FipLooptopHeating result{
        .heating = Fp1d("fip_looptop_heating", sz.yc),
        .rho0 = Fp1d("fip_looptop_rho0", sz.yc),
        .density_exponent = params.density_exponent
    };
    JasUnpack(result, heating, rho0);
    dex_parallel_for(
        "fip_looptop_2d heating profile",
        FlatLoop<1>(sz.yc),
        KOKKOS_LAMBDA (int j) {
            const fp_t y_pos = state.get_pos(sz.ng, j, 0)(1);
            const fp_t T = params.T0;
            auto loss = [&](fp_t nh, fp_t fip) {
                const FipTownsendCurve curve{.ctx = loss_ctx, .fip_bias = fip};
                return nh * ion_frac(nh, T) * nh * townsend_lambda(curve, T);
            };
            const fp_t nh = params.nh(y_pos);
            const fp_t h_bg = loss(params.nh_bg, params.fip_bg);
            const fp_t h_eq = loss(nh, params.heating_fip_bias(y_pos));
            heating(j) = (1.0_fp - params.strand_balance) * h_bg + params.strand_balance * h_eq;
            rho0(j) = nh * mass_per_h * m_u;
        }
    );
    Kokkos::fence();
    return result;
}

template <typename FTraits>
static void heating_kernel(const Simulation& sim, const FipLooptopHeating& heat) {
    using Cons = typename FTraits::cons;
    const auto& state = sim.state;
    const auto& sz = state.sz;
    const auto& Q = state.Q;
    const auto& S = sim.sources.S;
    JasUnpack(heat, heating, rho0, density_exponent);
    const bool scale_with_density = density_exponent != 0.0_fp;

    dex_parallel_for(
        "fip_looptop_2d heating",
        FlatLoop<3>(sz.zc, sz.yc, sz.xc),
        KOKKOS_LAMBDA (int k, int j, int i) {
            fp_t h = heating(j);
            if (scale_with_density) {
                h *= std::pow(Q(I(Cons::Rho), k, j, i) / rho0(j), density_exponent);
            }
            S(I(Cons::Ene), k, j, i) += h;
        }
    );
    Kokkos::fence();
}

MOSSCAP_NEW_PROBLEM(fip_looptop_2d) {
    MOSSCAP_PROBLEM_PREAMBLE(fip_looptop_2d);
    if (sim.num_dim != num_dim) {
        throw std::runtime_error(fmt::format(
            "{} only handles {}d problems", PROBLEM_NAME, num_dim
        ));
    }
    FluidTraitsRt traits(sim.num_dim, sim.fluid_type);
    if (!traits.is_mhd) {
        throw std::runtime_error(fmt::format(
            "{} requires an MHD fluid: the strand is confined by magnetic pressure.", PROBLEM_NAME
        ));
    }
    // NOTE(claude): Fail early on an unsupported EOS
    get_ion_frac(sim);

    const FipLooptopParams params = load_params(config);

    setup_thin_loss_fip(sim, config);
    const int loss_idx = source_term_index(sim, "thin_loss_fip");
    if (loss_idx == sim.compute_source_terms.size()) {
        throw std::runtime_error(fmt::format("{} requires sources.thin_loss_fip.enable.", PROBLEM_NAME));
    }
    const auto* loss_ctx = (FipThinLossContext*)sim.compute_source_terms[loss_idx].get_context();
    const int tracer_idx = loss_ctx->tracer_idx;

    sim.setup_ics = [=](Simulation& sim) {
        invoke_fluid_traits(
            sim.num_dim,
            sim.fluid_type,
            [&]<typename FTraits>(FTraits) {
                initial_conditions<FTraits>(sim, config, params, tracer_idx);
            }
        );
    };

    const FipLooptopHeating heat = compute_heating(sim, params, *loss_ctx);
    sim.compute_source_terms.push_back(SourceTerm{
        .name = "background_heating",
        .fn = [=](const Simulation& sim) {
            invoke_fluid_traits(
                sim.num_dim,
                sim.fluid_type,
                [&]<typename FTraits>(FTraits) {
                    heating_kernel<FTraits>(sim, heat);
                }
            );
        }
    });

    if (get_or<bool>(config, "sources.thermal_conduction.enable", false)) {
        setup_thermal_conduction(sim, config);
    }
    setup_replace_small_values(sim, config);
}

}
