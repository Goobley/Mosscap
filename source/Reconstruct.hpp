#if !defined(MOSSCAP_RECONSTRUCT_HPP)
#define MOSSCAP_RECONSTRUCT_HPP

#include "Types.hpp"

namespace Mosscap {

enum class Reconstruction {
    Fog = 0, // first order Godunov
    Muscl, // piecewise linear method
    Ppm, // piecewise parabolic method
    Weno5Z // 5th order weno
};
constexpr const char* ReconstructionName[] = {
    "fog",
    "muscl",
    "ppm",
    "weno5z"
};
constexpr int NumReconstructionType = sizeof(ReconstructionName) / sizeof(ReconstructionName[0]);

enum class SlopeLimiter {
    VanLeer = 0,
    MonotonizedCentral,
    Minmod
};
constexpr const char* SlopeLimiterName[] = {
    "vanleer",
    "monotonizedcentral",
    "minmod"
};
constexpr int NumSlopeLimiterType = sizeof(SlopeLimiterName) / sizeof(SlopeLimiterName[0]);

constexpr bool is_limiter_agnostic(Reconstruction recon) {
    switch (recon) {
        case Reconstruction::Fog: {
            return true;
        } break;
        case Reconstruction::Weno5Z: {
            return true;
        } break;
        default:
            return false;
    }
}

constexpr int min_ghost_cells(Reconstruction recon) {
    switch (recon) {
        case Reconstruction::Fog: {
            return 1;
        } break;
        case Reconstruction::Muscl: {
            return 2;
        } break;
        case Reconstruction::Ppm:
        case Reconstruction::Weno5Z: {
            return 3;
        }
    }
    return 1;
}

struct ReconstructionScheme {
    Reconstruction reconstruction;
    SlopeLimiter slope_limiter;

    inline friend bool operator<(const ReconstructionScheme& self, const ReconstructionScheme& other) {
        if (self.reconstruction != other.reconstruction) {
            return self.reconstruction < other.reconstruction;
        }
        if (is_limiter_agnostic(self.reconstruction)) {
            return false;
        } else {
            return self.slope_limiter < other.slope_limiter;
        }
    }

    inline friend bool operator==(const ReconstructionScheme& self, const ReconstructionScheme& other) {
        if (self.reconstruction != other.reconstruction) {
            return false;
        }
        if (is_limiter_agnostic(self.reconstruction)) {
            return true;
        } else {
            return self.slope_limiter == other.slope_limiter;
        }
    }
};

struct ReconScratch {
    Fp4d RR; /// Cell right-hand reconstruction [w, k, j, i]
    Fp4d RL; /// Cell left-hand reconstruction [w, k, j, i]
};

template <int Order>
struct Stencil {
    constexpr static int order = Order;
    yakl::SArray<fp_t, 1, 2 * Order + 1> w;

    template <int Axis>
    KOKKOS_INLINE_FUNCTION void fill(const Fp4d& W, const int var, const CellIndex& idx) {
        if constexpr (Axis == 0) {
            for (int o = -Order; o <= Order; ++o) {
                int i = idx.i + o;
                w(o + Order) = W(var, idx.k, idx.j, i);
            }
        } else if constexpr (Axis == 1) {
            for (int o = -Order; o <= Order; ++o) {
                int j = idx.j + o;
                w(o + Order) = W(var, idx.k, j, idx.i);
            }
        } else if constexpr (Axis == 2) {
            for (int o = -Order; o <= Order; ++o) {
                int k = idx.k + o;
                w(o + Order) = W(var, k, idx.j, idx.i);
            }
        }
    }

    KOKKOS_INLINE_FUNCTION fp_t at(const int i) const {
        return w(i + Order);
    }
};

/**
 * Compute the flux limited slope using a MUSCL-like scheme. i.e. phi(r)
 * (u_{i+1} - u_i) with um = (u_i - u_{i-1}), up = (u_{i+1} - u_i), and r =
 * um/up. phi(r) is the chosen flux/slope-limiter scheme.
 */
template <SlopeLimiter scheme>
KOKKOS_INLINE_FUNCTION fp_t slope_limiter(const fp_t um, const fp_t up) {
    using std::copysign;
    if constexpr (scheme == SlopeLimiter::VanLeer) {
        // phi(r) = 2r / (1 + r) but monotonic
        return (um * up > 0.0_fp) ? 2.0_fp * um * up / (um + up) : 0.0_fp;
    } else if constexpr (scheme == SlopeLimiter::MonotonizedCentral) {
        // phi(r) = max(0, min(2r, 0.5 * (1 + r), 2)). The term in the min is
        // multiplied through by up and sign terms implement the monotonicity
        // (and the factor of 2 that needs to multiply the second term)
        return (copysign(1.0_fp, um) + copysign(1.0_fp, up)) * std::min(std::abs(um), std::min(0.25_fp * std::abs(um + up), std::abs(up)));
    }  else if constexpr (scheme == SlopeLimiter::Minmod) {
        // minmod(a, b) = whichever of a or b has the small magnitude, or 0 if a
        // * b < 0. The copysign implements the a * b < 0 term.
        return 0.5_fp * (copysign(1.0_fp, um) + copysign(1.0_fp, up)) * std::min(std::abs(um), std::abs(up));
    }

    return 0.0_fp;
}

template <Reconstruction recon, SlopeLimiter sl, int Axis, std::enable_if_t<recon == Reconstruction::Fog, int> = 0>
KOKKOS_INLINE_FUNCTION void reconstruct(const Fp4d& W, const int var, const CellIndex& idx, fp_t& wL, fp_t& wR) {
    wL = W(var, idx.k, idx.j, idx.i);
    wR = W(var, idx.k, idx.j, idx.i);
}

template <Reconstruction recon, SlopeLimiter sl, int Axis, std::enable_if_t<recon == Reconstruction::Muscl, int> = 0>
KOKKOS_INLINE_FUNCTION void reconstruct(const Fp4d& W, const int var, const CellIndex& idx, fp_t& wL, fp_t& wR) {
    Stencil<1> s;
    s.fill<Axis>(W, var, idx);

    const fp_t dwL = s.at(0) - s.at(-1);
    const fp_t dwR = s.at(1) - s.at(0);
    const fp_t delta = slope_limiter<sl>(dwL, dwR);

    wL = s.at(0) - 0.5_fp * delta;
    wR = s.at(0) + 0.5_fp * delta;
}

template <Reconstruction recon, SlopeLimiter sl, int Axis, std::enable_if_t<recon == Reconstruction::Ppm, int> = 0>
KOKKOS_INLINE_FUNCTION void reconstruct(const Fp4d& W, const int var, const CellIndex& idx, fp_t& wL, fp_t& wR) {
    Stencil<2> s;
    s.fill<Axis>(W, var, idx);

    constexpr bool do_clamp = false;

    auto limited_slope = [](const Stencil<2>& s, const int idx) {
        return slope_limiter<sl>(s.at(idx) - s.at(idx - 1), s.at(idx + 1) - s.at(idx));
    };

    const fp_t dw_m = limited_slope(s, -1);
    const fp_t dw_0 = limited_slope(s, 0);
    const fp_t dw_p = limited_slope(s, 1);

    // NOTE(cmo): Cubic reconstruction
    wL = 0.5_fp * (s.at(-1) + s.at(0)) - (1.0_fp / 6.0_fp) * (dw_0 - dw_m);
    wR = 0.5_fp * (s.at(0) + s.at(1)) - (1.0_fp / 6.0_fp) * (dw_p - dw_0);

    if constexpr (do_clamp) {
        // Following Castro: make sure in between adjacent centred values --
        // this is likely unnecessary here as we only use the local slopes (no
        // second order gradients when computing the limited terms)
        fp_t lower = std::min(s.at(0), s.at(-1));
        fp_t upper = std::max(s.at(0), s.at(-1));
        wL = (wL < lower) ? lower : ((wL > upper) ? upper : wL);

        lower = std::min(s.at(0), s.at(1));
        upper = std::max(s.at(0), s.at(1));
        wR = (wR < lower) ? lower : ((wR > upper) ? upper : wR);
    }

    // NOTE(cmo): Enforce monotonicity -- following McCorquodale & Collela
    // (2011) for the latter cases without flattening (based on Collela & Sekora
    // 2008) as written in Castro (Almgren+ 2010)
    if ((wR - s.at(0)) * (s.at(0) - wL) <= 0.0_fp) {
        wL = s.at(0);
        wR = s.at(0);
    }
    if (std::abs(wR - s.at(0)) >= 2.0_fp * std::abs(wL - s.at(0))) {
        wR = 3.0_fp * s.at(0) - 2.0_fp * wL;

    }
    if (std::abs(wL - s.at(0)) >= 2.0_fp * std::abs(wR - s.at(0))) {
        wL = 3.0_fp * s.at(0) - 2.0_fp * wR;
    }
}

template <Reconstruction recon, SlopeLimiter sl, int Axis, std::enable_if_t<recon == Reconstruction::Weno5Z, int> = 0>
KOKKOS_INLINE_FUNCTION void reconstruct(const Fp4d& W, const int var, const CellIndex& idx, fp_t& wL, fp_t& wR) {
    // Borges R., Carmona M., Costa B., Don W.S. , "An improved weighted essentially
    // non-oscillatory scheme for hyperbolic conservation laws" , JCP, 227, 3191 (2008)
    Stencil<2> s;
    s.fill<Axis>(W, var, idx);

    constexpr fp_t beta_coeff[2] = {13.0_fp / 12.0_fp, 0.25_fp};
    const fp_t beta0 = (
        beta_coeff[0] * square(s.at(-2) - 2.0_fp * s.at(-1) + s.at(0))
        + beta_coeff[1] * square(s.at(-2) - 4.0_fp * s.at(-1) + 3.0_fp * s.at(0))
    );
    const fp_t beta1 = (
        beta_coeff[0] * square(s.at(-1) - 2.0_fp * s.at(0) + s.at(1))
        + beta_coeff[1] * square(s.at(-1) - s.at(1))
    );
    const fp_t beta2 = (
        beta_coeff[0] * square(s.at(0) - 2.0_fp * s.at(1) + s.at(2))
        + beta_coeff[1] * square(3.0_fp * s.at(0) - 4.0_fp * s.at(1) + s.at(2))
    );

#ifdef MOSSCAP_SINGLE_PRECISION
    constexpr fp_t eps = 1e-15_fp;
#else
    constexpr fp_t eps = 1e-36_fp;
#endif
    // WENO-Z+: Acker et al. 2016
    const fp_t tau5 = std::abs(beta0 - beta2);

    const fp_t indicator0 = tau5 / (beta0 + eps);
    const fp_t indicator1 = tau5 / (beta1 + eps);
    const fp_t indicator2 = tau5 / (beta2 + eps);

    // evaluate wL * 6 (division included later): eno, then weight
    // These coefficients are flipped relative to the original Jiang + Shu paper
    fp_t eno0 = (2.0_fp * s.at(0) + 5.0_fp * s.at(-1) - s.at(-2));
    fp_t eno1 = (-s.at(1) + 5.0_fp * s.at(0) + 2.0_fp * s.at(-1));
    fp_t eno2 = (2.0_fp * s.at(2) - 7.0_fp * s.at(1) + 11.0_fp * s.at(0));

    fp_t alpha0 = 3.0_fp * (1.0_fp + square(indicator0));
    fp_t alpha1 = 6.0_fp * (1.0_fp + square(indicator1));
    fp_t alpha2 = 1.0_fp * (1.0_fp + square(indicator2));
    fp_t denom = 6.0_fp * (alpha0 + alpha1 + alpha2);

    wL = (eno0 * alpha0 + eno1 * alpha1 + eno2 * alpha2) / denom;

    // evaluate wR equivalently (indicators flip)
    eno0 = (2.0_fp * s.at(0) + 5.0_fp * s.at(1) - s.at(2));
    eno1 = (-s.at(-1) + 5.0_fp * s.at(0) + 2.0_fp * s.at(1));
    eno2 = (2.0_fp * s.at(-2) - 7.0_fp * s.at(-1) + 11.0_fp * s.at(0));

    alpha0 = 3.0_fp * (1.0_fp + square(indicator2));
    alpha1 = 6.0_fp * (1.0_fp + square(indicator1));
    alpha2 = 1.0_fp * (1.0_fp + square(indicator0));
    denom = 6.0_fp * (alpha0 + alpha1 + alpha2);

    wR = (eno0 * alpha0 + eno1 * alpha1 + eno2 * alpha2) / denom;
}

}

#else
#endif