#if !defined(MOSSCAP_RECONSTRUCT_HPP)
#define MOSSCAP_RECONSTRUCT_HPP

#include "Types.hpp"

enum class Reconstruction {
    Fog, // first order Godunov
    Muscl, // piecewise linear method
    Ppm // piecewise parabolic method
};

enum class SlopeLimiter {
    VanLeer,
    MonotonizedCentral,
    Minmod
};

template <SlopeLimiter scheme>
KOKKOS_INLINE_FUNCTION fp_t slope_limiter(const fp_t a, const fp_t b) {
    using Kokkos::copysign;
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

#else
#endif