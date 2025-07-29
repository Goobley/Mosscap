#if !defined(MOSSCAP_RIEMANN_HPP)
#define MOSSCAP_RIEMANN_HPP

#include "Types.hpp"
#include "State.hpp"
#include "Eos.hpp"

enum class RiemannSolver {
    Rusanov,
    Hll,
    Hllc
};

template <RiemannSolver rs, int Axis, int NumDim, std::enable_if_t<rs == RiemannSolver::Rusanov, int> = 0>
KOKKOS_INLINE_FUNCTION void riemann_flux(const QtyView& wL, const QtyView& wR, const QtyView& flux) {
    using Prim = Prim<NumDim>;
    using Cons = Cons<NumDim>;
    constexpr int IV = Velocity<Axis, NumDim>();
    const fp_t vL = wL(IV);
    const fp_t vR = wR(IV);

    yakl::SArray<fp_t, 1, N_HYDRO_VARS> qL, qR, fL, fR;
    prim_to_cons<NumDim>(wL, qL);
    prim_to_cons<NumDim>(wR, qR);

    prim_to_flux<Axis, NumDim>(wL, fL);
    prim_to_flux<Axis, NumDim>(wR, fR);

    const fp_t csL = std::sqrt(Gamma * wL(I(Prim::Pres)) / wL(I(Prim::Rho)));
    const fp_t csR = std::sqrt(Gamma * wR(I(Prim::Pres)) / wR(I(Prim::Rho)));
    const fp_t max_c = FP(0.5) * (csL + std::abs(vL) + csR + std::abs(vR));

    #pragma unroll
    for (int i = 0; i < N_HYDRO_VARS; ++i) {
        flux(i) = FP(0.5) * (fL(i) + fR(i) - max_c * (qR(i) - qL(i)));
    }
}

template <RiemannSolver rs, int Axis, int NumDim, std::enable_if_t<rs == RiemannSolver::Hll, int> = 0>
KOKKOS_INLINE_FUNCTION void riemann_flux(const QtyView& wL, const QtyView& wR, const QtyView& flux) {
    using Prim = Prim<NumDim>;
    using Cons = Cons<NumDim>;
    constexpr int IV = Velocity<Axis, NumDim>();

    const fp_t csL = std::sqrt(Gamma * wL(I(Prim::Pres)) / wL(I(Prim::Rho)));
    const fp_t csR = std::sqrt(Gamma * wR(I(Prim::Pres)) / wR(I(Prim::Rho)));

    constexpr fp_t tiny = FP(1e-7);
    const fp_t sL = std::min(-tiny, std::min(wL(IV) - csL, wR(IV) - csR));
    const fp_t sR = std::max(-tiny, std::max(wL(IV) + csL, wL(IV) + csR));
    const fp_t sM = FP(1.0) / (sR - sL);

    yakl::SArray<fp_t, 1, N_HYDRO_VARS> qL, qR, fL, fR;
    prim_to_cons<NumDim>(wL, qL);
    prim_to_cons<NumDim>(wR, qR);
    prim_to_flux<Axis, NumDim>(wL, fL);
    prim_to_flux<Axis, NumDim>(wR, fR);

    #pragma unroll
    for (int i = 0; i < N_HYDRO_VARS; ++i) {
        flux(i) = sM * (sR * fL(i) - sL * fR(i) + sR * sL * (qR(i) - qL(i)));
    }
}

template <RiemannSolver rs, int Axis, int NumDim, std::enable_if_t<rs == RiemannSolver::Hllc, int> = 0>
KOKKOS_INLINE_FUNCTION void riemann_flux(const QtyView& wL, const QtyView& wR, const QtyView& flux) {
    using Prim = Prim<NumDim>;
    using Cons = Cons<NumDim>;
    constexpr int IV = Velocity<Axis, NumDim>();
    constexpr int IM = Momentum<Axis, NumDim>();

    const fp_t csL = std::sqrt(Gamma * wL(I(Prim::Pres)) / wL(I(Prim::Rho)));
    const fp_t csR = std::sqrt(Gamma * wR(I(Prim::Pres)) / wR(I(Prim::Rho)));

    constexpr fp_t tiny = FP(1e-7);
    // const fp_t sL = std::min(-tiny, std::min(wL(IV) - csL, wR(IV) - csR));
    // const fp_t sR = std::max(-tiny, std::max(wL(IV) + csL, wL(IV) + csR));
    // const fp_t sM = FP(1.0) / (sR - sL);

    yakl::SArray<fp_t, 1, N_HYDRO_VARS> qL, qR, fL, fR;
    prim_to_cons<NumDim>(wL, qL);
    prim_to_cons<NumDim>(wR, qR);

    // following athena impl
    // TODO(cmo): Go back to the original paper and try to refactor in terms of hll flux

    const fp_t pL = wL(I(Prim::Pres));
    const fp_t pR = wR(I(Prim::Pres));
    const fp_t rho_avg = FP(0.5) * (wL(I(Prim::Rho)) + wR(I(Prim::Rho)));
    const fp_t cs_avg = FP(0.5) * (csL + csR);
    const fp_t p_mid = FP(0.5) * (pL + pR + (wL(IV) - wR(IV)) * rho_avg * cs_avg);

    // compute L, R sound speed
    const fp_t cLstar = (p_mid <= wL(I(Prim::Pres)))
                            ? FP(1.0)
                            : std::sqrt(FP(1.0) + (Gamma + FP(1.0)) / (FP(2.0) * Gamma) * (p_mid / wL(I(Prim::Pres)) - FP(1.0)));
    const fp_t cRstar = (p_mid <= wR(I(Prim::Pres)))
                            ? FP(1.0)
                            : std::sqrt(FP(1.0) + (Gamma + FP(1.0)) / (FP(2.0) * Gamma) * (p_mid / wR(I(Prim::Pres)) - FP(1.0)));

    // compute min/max wave speeds based on L/R
    const fp_t aL = wL(IV) - csL * cLstar;
    const fp_t aR = wR(IV) + csR * cRstar;

    const fp_t bM = std::min(-tiny, aL);
    const fp_t bP = std::max(tiny, aR);

    // compute contact wave speed and pressure
    fp_t vxL = wL(IV) - aL;
    fp_t vxR = wR(IV) - aR;
    const fp_t tL = wL(I(Prim::Pres)) + vxL * qL(IM);
    const fp_t tR = wR(I(Prim::Pres)) + vxR * qR(IM);
    const fp_t mL = wL(I(Prim::Rho)) * vxL;
    const fp_t mR = -(wR(I(Prim::Rho)) * vxR);

    // determine contact wave speed
    const fp_t aM = (tL - tR) / (mL + mR);
    // pressure at contact surface
    const fp_t cp = std::max((mL * tR + mR * tL) / (mL + mR), FP(0.0));

    // compute L/R fluxes along clamped line bm, bp
    vxL = wL(IV) - bM;
    vxR = wR(IV) - bP;

    const fp_t mass_flux_L = wL(I(Prim::Rho)) * vxL;
    const fp_t mass_flux_R = wR(I(Prim::Rho)) * vxR;
    fL(I(Cons::Rho)) = mass_flux_L;
    fR(I(Cons::Rho)) = mass_flux_R;

    fL(I(Cons::MomX)) = mass_flux_L * wL(I(Prim::Vx));
    fR(I(Cons::MomX)) = mass_flux_R * wR(I(Prim::Vx));

    if (NUM_DIM > 1) {
        fL(I(Cons::MomY)) = mass_flux_L * wL(I(Prim::Vy));
        fR(I(Cons::MomY)) = mass_flux_R * wR(I(Prim::Vy));
    }
    if (NUM_DIM > 2) {
        fL(I(Cons::MomZ)) = mass_flux_L * wL(I(Prim::Vz));
        fR(I(Cons::MomZ)) = mass_flux_R * wR(I(Prim::Vz));
    }
    fL(IM) += wL(I(Prim::Pres));
    fR(IM) += wR(I(Prim::Pres));

    fL(I(Cons::Ene)) = qL(I(Cons::Ene)) * vxL + wL(I(Prim::Pres)) * wL(IV);
    fR(I(Cons::Ene)) = qR(I(Cons::Ene)) * vxR + wR(I(Prim::Pres)) * wR(IV);

    // compute flux weights
    fp_t sL, sR, sM;
    if (aM >= FP(0.0)) {
        sL = aM / (aM - bM);
        sR = FP(0.0);
        sM = -bM / (aM - bM);
    } else {
        sL = FP(0.0);
        sR = -aM / (bP - aM);
        sM = bP / (bP - aM);
    }

    #pragma unroll
    for (int var = 0; var < N_HYDRO_VARS; ++var) {
        flux(var) = sL * fL(var) + sR * fR(var);
    }
    flux(IM) += sM * cp;
    flux(I(Cons::Ene)) += sM * cp * aM;
}

#else
#endif