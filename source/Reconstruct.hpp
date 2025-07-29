#if !defined(MOSSCAP_RECONSTRUCT_HPP)
#define MOSSCAP_RECONSTRUCT_HPP

#include "Types.hpp"

enum class Reconstruction {
    Fog, // first order Godunov
    Muscl, // piecewise linear method
    Ppm, // piecewise parabolic method
    Weno5Z // 5th order weno
};

enum class SlopeLimiter {
    VanLeer,
    MonotonizedCentral,
    Minmod
};

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
    Fp4d RR; /// Left-hand reconstruction [w, k, j, i]
    Fp4d RL; /// Right-hand reconstruction [w, k, j, i]
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

template <SlopeLimiter scheme>
KOKKOS_INLINE_FUNCTION fp_t slope_limiter(const fp_t a, const fp_t b) {
    using std::copysign;
    if constexpr (scheme == SlopeLimiter::VanLeer) {
        return (a * b > FP(0.0)) ? FP(2.0) * a * b / (a + b) : FP(0.0);
    } else if constexpr (scheme == SlopeLimiter::MonotonizedCentral) {
        return (copysign(FP(1.0), a) + copysign(FP(1.0), b)) * std::min(std::abs(a), std::min(FP(0.25) * std::abs(a + b), std::abs(b)));
    }  else if constexpr (scheme == SlopeLimiter::Minmod) {
        return FP(0.5) * (copysign(FP(1.0), a) + copysign(FP(1.0), b)) * std::min(std::abs(a), std::abs(b));
    }

    return FP(0.0);
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

    wL = s.at(0) - FP(0.5) * delta;
    wR = s.at(0) + FP(0.5) * delta;
}

template <Reconstruction recon, SlopeLimiter sl, int Axis, std::enable_if_t<recon == Reconstruction::Ppm, int> = 0>
KOKKOS_INLINE_FUNCTION void reconstruct(const Fp4d& W, const int var, const CellIndex& idx, fp_t& wL, fp_t& wR) {
    Stencil<2> s;
    s.fill<Axis>(W, var, idx);

    auto limited_slope = [](const Stencil<2>& s, const int idx) {
        return slope_limiter<sl>(s.at(idx + 1) - s.at(idx), s.at(idx) - s.at(idx - 1));
    };

    const fp_t dw_m = limited_slope(s, -1);
    const fp_t dw_0 = limited_slope(s, 0);
    const fp_t dw_p = limited_slope(s, 1);

    // NOTE(cmo): Cubic reconstruction
    wL = FP(0.5) * (s.at(-1) + s.at(0)) - (FP(1.0) / FP(6.0)) * (dw_0 - dw_m);
    wR = FP(0.5) * (s.at(0) + s.at(1)) - (FP(1.0) / FP(6.0)) * (dw_p - dw_0);

    // NOTE(cmo): Enforce monotonicity
    if ((wR - s.at(0)) * (s.at(0) - wL) <= FP(0.0)) {
        wL = s.at(0);
        wR = s.at(0);
    }

    if (-square(wR - wL) > FP(6.0) * (wR - wL) * (s.at(0) - FP(0.5) * (wL + wR))) {
        wR = FP(3.0) * s.at(0) - FP(2.0) * wL;
    }
    if (square(wR - wL) < FP(6.0) * (wR - wL) * (s.at(0) - FP(0.5) * (wL + wR))) {
        wL = FP(3.0) * s.at(0) - FP(2.0) * wR;
    }
}

template <Reconstruction recon, SlopeLimiter sl, int Axis, std::enable_if_t<recon == Reconstruction::Weno5Z, int> = 0>
KOKKOS_INLINE_FUNCTION void reconstruct(const Fp4d& W, const int var, const CellIndex& idx, fp_t& wL, fp_t& wR) {
    // Borges R., Carmona M., Costa B., Don W.S. , "An improved weighted essentially
    // non-oscillatory scheme for hyperbolic conservation laws" , JCP, 227, 3191 (2008)
    // Implementation following athenapk, but correcting beta2
    Stencil<2> s;
    s.fill<Axis>(W, var, idx);

    constexpr fp_t beta_coeff[2] = {FP(13.0) / FP(12.0), FP(0.25)};
    const fp_t beta0 = (
        beta_coeff[0] * square(s.at(-2) - FP(2.0) * s.at(-1) + s.at(0))
        + beta_coeff[1] * square(s.at(-2) - FP(4.0) * s.at(-1) + FP(3.0) * s.at(0))
    );
    const fp_t beta1 = (
        beta_coeff[0] * square(s.at(-1) - FP(2.0) * s.at(0) + s.at(1))
        + beta_coeff[1] * square(s.at(-1) - s.at(1))
    );
    const fp_t beta2 = (
        beta_coeff[0] * square(s.at(0) - FP(2.0) * s.at(1) + s.at(2))
        + beta_coeff[1] * square(s.at(0) - FP(4.0) * s.at(1) + FP(3.0) * s.at(2))
    );

#ifdef MOSSCAP_SINGLE_PRECISION
    constexpr fp_t eps = FP(1e-12);
#else
    constexpr fp_t eps = FP(1e-20);
#endif
    // WENO-Z+: Acker et al. 2016
    const fp_t tau5 = std::abs(beta0 - beta2);

    const fp_t indicator0 = tau5 / (beta0 + eps);
    const fp_t indicator1 = tau5 / (beta1 + eps);
    const fp_t indicator2 = tau5 / (beta2 + eps);

    // evaluate wL * 6 (division included later): eno, then weight
    fp_t f0 = (FP(2.0) * s.at(2) - FP(7.0) * s.at(1) + FP(11.0) * s.at(0));
    fp_t f1 = (-s.at(1) + FP(5.0) * s.at(0) + FP(2.0) * s.at(-1));
    fp_t f2 = (FP(2.0) * s.at(0) + FP(5.0) * s.at(-1) - s.at(-2));

    fp_t alpha0 = FP(0.1) * (FP(1.0) + square(indicator2));
    fp_t alpha1 = FP(0.6) * (FP(1.0) + square(indicator1));
    fp_t alpha2 = FP(0.3) * (FP(1.0) + square(indicator0));
    fp_t denom = FP(6.0) * (alpha0 + alpha1 + alpha2);

    wL = (f0 * alpha0 + f1 * alpha1 + f2 * alpha2) / denom;

    // evaluate wR equivalently (indicators flip)
    f0 = (FP(2.0) * s.at(-2) - FP(7.0) * s.at(-1) + FP(11.0) * s.at(0));
    f1 = (-s.at(-1) + FP(5.0) * s.at(0) + FP(2.0) * s.at(1));
    f2 = (FP(2.0) * s.at(0) + FP(5.0) * s.at(1) - s.at(2));

    alpha0 = FP(0.1) * (FP(1.0) + square(indicator0));
    alpha1 = FP(0.6) * (FP(1.0) + square(indicator1));
    alpha2 = FP(0.3) * (FP(1.0) + square(indicator2));
    denom = FP(6.0) * (alpha0 + alpha1 + alpha2);

    wR = (f0 * alpha0 + f1 * alpha1 + f2 * alpha2) / denom;
}


#else
#endif