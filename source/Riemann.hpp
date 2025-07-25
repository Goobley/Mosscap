#if !defined(MOSSCAP_RIEMANN_HPP)
#define MOSSCAP_RIEMANN_HPP

#include "Types.hpp"
#include "State.hpp"
#include "Eos.hpp"

enum class RiemannSolver {
    Llf,
    Hll,
    Hllc
};

template <RiemannSolver rs, int Axis, std::enable_if_t<rs == RiemannSolver::Llf, int> = 0>
KOKKOS_INLINE_FUNCTION void RiemannFlux(const QtyView& wL, const QtyView& wR, const QtyView& flux) {
    constexpr int IV = Velocity<Axis>();
    const fp_t vL = wL(IV);
    const fp_t vR = wR(IV);

    yakl::SArray<fp_t, 1, N_HYDRO_VARS> qL;
    prim_to_cons(wL, qL);
    yakl::SArray<fp_t, 1, N_HYDRO_VARS> qR;
    prim_to_cons(wR, qR);

    yakl::SArray<fp_t, 1, N_HYDRO_VARS> fL;
    prim_to_flux<Axis>(wL, fL);
    yakl::SArray<fp_t, 1, N_HYDRO_VARS> fR;
    prim_to_flux<Axis>(wR, fR);

    const fp_t csL = std::sqrt(Gamma * wL(I(Prim::Pres)) / wL(I(Prim::Rho)));
    const fp_t csR = std::sqrt(Gamma * wR(I(Prim::Pres)) / wR(I(Prim::Rho)));
    const fp_t max_c = FP(0.5) * (csL + std::abs(vL) + csR + std::abs(vR));

    #pragma unroll
    for (int i = 0; i < N_HYDRO_VARS; ++i) {
        flux(i) = FP(0.5) * (fL(i) + fR(i) - max_c * (qR(i) - qL(i)));
    }
}

#else
#endif