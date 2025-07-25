#if !defined(MOSSCAP_RECONSTRUCT_HPP)
#define MOSSCAP_RECONSTRUCT_HPP

#include "Types.hpp"

enum class Reconstruction {
    Fog, // first order Godunov
    Muscl, // piecewise linear method
    PPM // piecewise parabolic method
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
    if (idx.j == 64 && idx.i == 128) {
        printf("Writing var: %d, value: %f\n", var, W(var, idx.k, idx.j, idx.i));
    }
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
            for (int i = idx.i - Order; i <= idx.i + Order; ++i) {
                w(i) = W(var, idx.k, idx.j, i);
            }
        } else if constexpr (Axis == 1) {
            for (int j = idx.j - Order; j <= idx.j + Order; ++j) {
                w(j) = W(var, idx.k, j, idx.i);
            }
        } else if constexpr (Axis == 2) {
            for (int k = idx.k - Order; k <= idx.k + Order; ++k) {
                w(k) = W(var, k, idx.j, idx.i);
            }
        }
    }

    KOKKOS_INLINE_FUNCTION fp_t at(const int i) const {
        return w(i + order);
    }
};

template <Reconstruction recon, SlopeLimiter sl, int Axis, std::enable_if_t<recon == Reconstruction::Muscl, int> = 0>
KOKKOS_INLINE_FUNCTION void reconstruct(const Fp4d& W, const int var, const CellIndex& idx, fp_t& wL, fp_t& wR) {
    Stencil<2> s;
    s.fill<Axis>(W, var, idx);

    const fp_t dwL = s.at(0) - s.at(-1);
    const fp_t dwR = s.at(1) - s.at(0);
    const fp_t delta = slope_limiter<sl>(dwL, dwR);

    wL = s.at(0) - FP(0.5) * delta;
    wR = s.at(0) + FP(0.5) * delta;
}

#else
#endif