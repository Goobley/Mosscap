#if !defined(MOSSCAP_TOWNSEND_THIN_LOSS_HPP)
#define MOSSCAP_TOWNSEND_THIN_LOSS_HPP
#include "../Simulation.hpp"


namespace YAML { class Node; };

namespace Mosscap {

// Townsend (2009) exact integration of optically thin losses over a
// piecewise power-law cooling curve.
// To support traditional and FIP varying curves, the machinery works on `Curve`
// objects which provide as methods:
//   int n_temps()                  number of temperature nodes (N + 1)
//   fp_t temp(int k)               node temperature [K]
//   fp_t lambda(int k)             cooling function at node k
//   fp_t alpha(int k)              power-law index of bin k (k < N)
//
// The scheme only ever needs differences of the temporal evolution function
// (TEF) Y, so it tracks the residual Y_target - Y(T_idx) while walking down the
// bins rather than absolute values of Y. Hence no reference temperature is
// needed, and the Λ_ref / T_ref normalisation of Townsend (2009) is dropped as
// it cancels (in exchange for the TEF becoming dimensionalised). Cost scales
// with the number of bins crossed during the step.

/// Magnitude of alpha_k below which to employ the alpha_k = 1 solution
constexpr fp_t townsend_alpha_one_tol = 1e-6_fp;

/// TEF accumulated within bin k between T_k and temperature, i.e.
/// (T_k / Λ_k) ((T_k / T)^(α_k - 1) - 1) / (α_k - 1). This is Y(T) - Y(T_k),
/// and is <= 0 for T >= T_k.
template <typename Curve>
KOKKOS_INLINE_FUNCTION fp_t townsend_bin_tef(const Curve& curve, int k, fp_t temperature) {
    const fp_t scale = curve.temp(k) / curve.lambda(k);
    const fp_t alpha_m1 = curve.alpha(k) - 1.0_fp;
    const fp_t ratio = curve.temp(k) / temperature;
    if (std::abs(alpha_m1) < townsend_alpha_one_tol) {
        return scale * std::log(ratio);
    }
    return scale * (std::pow(ratio, alpha_m1) - 1.0_fp) / alpha_m1;
}

/// Inverse of townsend_bin_tef: temperature in bin k for a TEF offset dY = Y - Y_k.
template <typename Curve>
KOKKOS_INLINE_FUNCTION fp_t townsend_bin_inverse_tef(const Curve& curve, int k, fp_t dY) {
    const fp_t inv_scale = curve.lambda(k) / curve.temp(k);
    const fp_t one_m_alpha = 1.0_fp - curve.alpha(k);
    if (std::abs(one_m_alpha) < townsend_alpha_one_tol) {
        return curve.temp(k) * std::exp(-inv_scale * dY);
    }
    return curve.temp(k) * std::pow(
        1.0_fp - one_m_alpha * inv_scale * dY,
        1.0_fp / one_m_alpha
    );
}

/// Rate of change of internal energy density [J m-3 s-1] from integrating the
/// optically thin losses over dt with the Townsend scheme.
template <typename Curve>
KOKKOS_INLINE_FUNCTION fp_t townsend_energy_rate(
    const Curve& curve,
    fp_t temperature,
    fp_t nh_tot,
    fp_t ne,
    fp_t gamma,
    fp_t dt,
    fp_t min_temperature
) {
    constexpr fp_t k_B = ConstantsF64::k_B;
    if (temperature < min_temperature) {
        return 0.0_fp;
    }
    const int N = curve.n_temps() - 1;
    const int n_bins = N;

    // Find temperature bin
    int idx = 0;
    while ((idx < n_bins - 1) && (curve.temp(idx + 1) < temperature)) {
        idx += 1;
    }

    // Residual of the target TEF relative to the TEF at the lower edge of bin
    // idx, i.e. idx serves as the temperature reference.
    fp_t dY = (
        townsend_bin_tef(curve, idx, temperature)
        + dt * (nh_tot * ne) / (nh_tot + ne) * (gamma - 1.0_fp) / k_B
    );

    // Walk down while the target lies below T_idx, re-basing the residual on
    // each lower node: Y(T_{idx+1}) - Y(T_idx) = townsend_bin_tef(idx, T_{idx+1}).
    // i.e. effectively moving the temperature reference as we go.
    while ((idx > 0) && (dY > 0.0_fp)) {
        idx -= 1;
        dY += townsend_bin_tef(curve, idx, curve.temp(idx + 1));
    }

    fp_t new_temperature = townsend_bin_inverse_tef(curve, idx, dY);
    new_temperature = std::max(new_temperature, min_temperature);
    const fp_t delta_temp = new_temperature - temperature;
    const fp_t delta_e = 1.0_fp / (gamma - 1.0_fp) * (nh_tot + ne) * k_B * delta_temp;
    return delta_e / dt;
}

/// A cooling curve with precomputed nodes and power-law indices.
template <typename Arr>
struct TownsendCurve {
    Arr temps;
    Arr lambdas;
    Arr alpha_k;

    KOKKOS_INLINE_FUNCTION int n_temps() const { return temps.extent(0); }
    KOKKOS_INLINE_FUNCTION fp_t temp(int k) const { return temps(k); }
    KOKKOS_INLINE_FUNCTION fp_t lambda(int k) const { return lambdas(k); }
    KOKKOS_INLINE_FUNCTION fp_t alpha(int k) const { return alpha_k(k); }
};

struct ThinLossContext {
    TownsendCurve<Fp1d> curve;
    fp_t min_temperature;
};

struct FipThinLossContext {
    Fp1d temps;
    /// Contribution of high-FIP elements (base curve minus the low-FIP curve)
    Fp1d lambdas_high_fip;
    /// Contribution of low-FIP elements at FIP bias 1, scaled by the tracer
    Fp1d lambdas_low_fip;
    /// 1 / ln(T_{k+1} / T_k)
    Fp1d inv_log_dtemp;
    /// Range the tracer is clamped to against over/undershoots from advection
    fp_t min_fip_bias;
    fp_t max_fip_bias;
    fp_t min_temperature;
    /// Index of the FIP bias tracer in Q (stored as rho * fip_bias)
    i32 tracer_idx;

    template <typename QType>
    KOKKOS_INLINE_FUNCTION fp_t fip_bias(const QType& q, int rho_idx) const {
        const fp_t fip = q(tracer_idx) / q(rho_idx);
        return std::min(std::max(fip, min_fip_bias), max_fip_bias);
    }
};

/// Cooling curve for a given FIP bias f: Λ = Λ_high + f Λ_low at each node,
/// treated as a piecewise power law between nodes.
struct FipTownsendCurve {
    const FipThinLossContext& ctx;
    fp_t fip_bias;

    KOKKOS_INLINE_FUNCTION int n_temps() const { return ctx.temps.extent(0); }
    KOKKOS_INLINE_FUNCTION fp_t temp(int k) const { return ctx.temps(k); }
    KOKKOS_INLINE_FUNCTION fp_t lambda(int k) const {
        return ctx.lambdas_high_fip(k) + fip_bias * ctx.lambdas_low_fip(k);
    }
    KOKKOS_INLINE_FUNCTION fp_t alpha(int k) const {
        return std::log(lambda(k + 1) / lambda(k)) * ctx.inv_log_dtemp(k);
    }
};

void setup_thin_loss(Simulation& sim, YAML::Node& config);
void setup_thin_loss_fip(Simulation& sim, YAML::Node& config);

template <typename FTraits, typename QType>
fp_t thin_loss_single_val(
    const Simulation& sim,
    const ThinLossContext& ctx,
    const QType& q,
    const fp_t ion_frac=1.0_fp
) {
    using Prim = typename FTraits::prim;
    constexpr int n_hydro = FTraits::num_vars;
    constexpr fp_t m_p = ConstantsF64::u;

    JasUnpack(sim, state, eos, dt_sub);
    JasUnpack(state, mu0);

    Fp1d result("thin loss result", 1);
    result = 0.0_fp;
    Kokkos::fence();
    dex_parallel_for(
        "Compute thin loss",
        FlatLoop<1>(1),
        KOKKOS_LAMBDA (int i) {
            yakl::SArray<fp_t, 1, n_hydro> w;
            cons_to_prim<FTraits>(eos.gamma, mu0, q, w);

            const fp_t nh_tot = w(I(Prim::Rho)) / (eos.mass_per_h * m_p);
            fp_t y = eos.y;
            if (!eos.is_constant) {
                y = ion_frac;
            }
            auto temperature = temperature_si(w(I(Prim::Pres)), nh_tot, eos.total_abund, y);
            fp_t ne = y * nh_tot;
            result(i) = townsend_energy_rate(
                ctx.curve,
                temperature,
                nh_tot,
                ne,
                eos.gamma,
                dt_sub,
                ctx.min_temperature
            );
        }
    );
    Kokkos::fence();
    return result.createHostCopy()(0);
}

}

#else
#endif
