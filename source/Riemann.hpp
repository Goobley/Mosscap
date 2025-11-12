#if !defined(MOSSCAP_RIEMANN_HPP)
#define MOSSCAP_RIEMANN_HPP

#include "Types.hpp"
#include "State.hpp"
#include "Eos.hpp"
#include "GlmMhd.hpp"

namespace Mosscap {

enum class RiemannSolver {
    Rusanov = 0,
    Hll,
    Hllc
};
constexpr const char* RiemannSolverName[] = {
    "rusanov",
    "hll",
    "hllc"
};
constexpr int NumRiemannSolverType = sizeof(RiemannSolverName) / sizeof(RiemannSolverName[0]);

template <typename FTraits>
KOKKOS_INLINE_FUNCTION void fix_hypertconly_flux(const QtyView& flux) {
    if constexpr (FTraits::fluid_type != FluidType::HyperTcOnly) {
        return;
    }

    constexpr int n_hydro = FTraits::num_vars;
    const fp_t ene = flux(FTraits::cons::Ene);
    const fp_t heat_f = flux(FTraits::cons::HeatF);
    #pragma unroll
    for (int i = 0; i < n_hydro; ++i) {
        flux(i) = 0.0_fp;
    }
    flux(FTraits::cons::Ene) = ene;
    if (HYPERTC_IN_FLUX_VECTOR) {
        flux(FTraits::cons::HeatF) = heat_f;
    }
}

template <RiemannSolver rs, int Axis, typename FTraits, std::enable_if_t<rs == RiemannSolver::Rusanov, int> = 0>
KOKKOS_INLINE_FUNCTION void riemann_flux(const Eos& eos, const fp_t mu0, const fp_t c_h, const QtyView& wL, const QtyView& wR, const QtyView& flux) {
    using Prim = typename FTraits::prim;
    using Cons = typename FTraits::cons;
    constexpr int n_hydro = FTraits::num_vars;
    constexpr int IV = Velocity<Axis, FTraits>();
    glm_set_mid_state<Axis, FTraits>(c_h, wL, wR);
    const fp_t vL = wL(IV);
    const fp_t vR = wR(IV);

    yakl::SArray<fp_t, 1, n_hydro> qL, qR, fL, fR;
    prim_to_cons<FTraits>(eos.gamma, mu0, wL, qL);
    prim_to_cons<FTraits>(eos.gamma, mu0, wR, qR);

    prim_to_flux<FTraits, Axis>(eos.gamma, mu0, c_h, wL, fL);
    prim_to_flux<FTraits, Axis>(eos.gamma, mu0, c_h, wR, fR);

    const fp_t cfL = fast_wave_speed<FTraits, Axis>(eos.gamma, mu0, wL);
    const fp_t cfR = fast_wave_speed<FTraits, Axis>(eos.gamma, mu0, wR);
    const fp_t max_c = 0.5_fp * (cfL + std::abs(vL) + cfR + std::abs(vR));

    #pragma unroll
    for (int i = 0; i < n_hydro; ++i) {
        flux(i) = 0.5_fp * (fL(i) + fR(i) - max_c * (qR(i) - qL(i)));
    }
    fix_hypertconly_flux<FTraits>(flux);
}

template <RiemannSolver rs, int Axis, typename FTraits, std::enable_if_t<rs == RiemannSolver::Hll, int> = 0>
KOKKOS_INLINE_FUNCTION void riemann_flux(const Eos& eos, const fp_t mu0, const fp_t c_h, const QtyView& wL, const QtyView& wR, const QtyView& flux) {
    using Prim = typename FTraits::prim;
    using Cons = typename FTraits::cons;
    constexpr int n_hydro = FTraits::num_vars;
    constexpr int IV = Velocity<Axis, FTraits>();

    glm_set_mid_state<Axis, FTraits>(c_h, wL, wR);

    const fp_t cfL = fast_wave_speed<FTraits, Axis>(eos.gamma, mu0, wL);
    const fp_t cfR = fast_wave_speed<FTraits, Axis>(eos.gamma, mu0, wR);

    // NOTE(cmo): We use this tiny value to kill the contribution from one side
    // and select the two edges of the fan when needed.
    constexpr fp_t tiny = 1e-30_fp;
    // Davis wave speed estimates
    // const fp_t sL = std::min(-tiny, std::min(wL(IV) - cfL, wR(IV) - cfR));
    // const fp_t sR = std::max(tiny, std::max(wL(IV) + cfL, wR(IV) + cfR));
    // const fp_t sL = std::min(wL(IV) - cfL, wR(IV) - cfR);
    // const fp_t sR = std::max(wL(IV) + cfL, wR(IV) + cfR);

    // 10.52 of  Riemann Solvers and Numerical Methods for Fluid Dynamics, Toro (Einfeldt description)
    const fp_t sqrt_rho_L = std::sqrt(wL(I(Prim::Rho)));
    const fp_t sqrt_rho_R = std::sqrt(wR(I(Prim::Rho)));
    const fp_t ubar = (sqrt_rho_L * wL(IV) + sqrt_rho_R * wR(IV)) / (sqrt_rho_L + sqrt_rho_R);
    const fp_t eta2 = 0.5_fp * (sqrt_rho_L * sqrt_rho_R) / square(sqrt_rho_L + sqrt_rho_R);
    const fp_t dbar2 = (sqrt_rho_L * square(cfL)  + sqrt_rho_R * square(cfR)) / (sqrt_rho_L + sqrt_rho_R) + eta2 * square(wR(IV) - wL(IV));
    const fp_t dbar = std::sqrt(dbar2);
    const fp_t sL = std::min(-tiny, ubar - dbar);
    const fp_t sR = std::max(tiny, ubar + dbar);
    // fp_t sL = ubar - dbar;
    // fp_t sR = ubar + dbar;
    const fp_t sM = 1.0_fp / (sR - sL);

    yakl::SArray<fp_t, 1, n_hydro> qL, qR, fL, fR;
    prim_to_cons<FTraits>(eos.gamma, mu0, wL, qL);
    prim_to_cons<FTraits>(eos.gamma, mu0, wR, qR);
    prim_to_flux<FTraits, Axis>(eos.gamma, mu0, c_h, wL, fL);
    prim_to_flux<FTraits, Axis>(eos.gamma, mu0, c_h, wR, fR);

    // if (sR <= 0.0_fp) {
    //     #pragma unroll
    //     for (int i = 0; i < n_hydro; ++i) {
    //         flux(i) = fR(i);
    //     }
    //     return;
    // } else if (sL >= 0.0_fp) {
    //     #pragma unroll
    //     for (int i = 0; i < n_hydro; ++i) {
    //         flux(i) = fL(i);
    //     }
    //     return;
    // }

    #pragma unroll
    for (int i = 0; i < n_hydro; ++i) {
        flux(i) = sM * (sR * fL(i) - sL * fR(i) + sR * sL * (qR(i) - qL(i)));
    }
    fix_hypertconly_flux<FTraits>(flux);
}

template <RiemannSolver rs, int Axis, typename FTraits, std::enable_if_t<rs == RiemannSolver::Hllc, int> = 0>
KOKKOS_INLINE_FUNCTION void riemann_flux(const Eos& eos, const fp_t mu0, const fp_t c_h, const QtyView& wL, const QtyView& wR, const QtyView& flux) {
    // NOTE(cmo): Not currently valid for MHD. At all.
    using Prim = typename FTraits::prim;
    using Cons = typename FTraits::cons;
    constexpr int NumDim = FTraits::num_dim;
    constexpr int n_hydro = FTraits::num_vars;
    constexpr int IV = Velocity<Axis, FTraits>();
    constexpr int IM = Momentum<Axis, FTraits>();

    glm_set_mid_state<Axis, FTraits>(c_h, wL, wR);

    const fp_t csL = sound_speed<FTraits>(eos.gamma, wL);
    const fp_t csR = sound_speed<FTraits>(eos.gamma, wR);

    constexpr fp_t tiny = 1e-20_fp;
    // const fp_t sL = std::min(-tiny, std::min(wL(IV) - csL, wR(IV) - csR));
    // const fp_t sR = std::max(-tiny, std::max(wL(IV) + csL, wL(IV) + csR));
    // const fp_t sM = 1.0_fp / (sR - sL);

    yakl::SArray<fp_t, 1, n_hydro> qL, qR, fL, fR;
    prim_to_cons<FTraits>(eos.gamma, mu0, wL, qL);
    prim_to_cons<FTraits>(eos.gamma, mu0, wR, qR);

    // following athena impl
    // TODO(cmo): Go back to the original paper and try to refactor in terms of hll flux

    const fp_t pL = wL(I(Prim::Pres));
    const fp_t pR = wR(I(Prim::Pres));
    const fp_t rho_avg = 0.5_fp * (wL(I(Prim::Rho)) + wR(I(Prim::Rho)));
    const fp_t cs_avg = 0.5_fp * (csL + csR);
    const fp_t p_mid = 0.5_fp * (pL + pR + (wL(IV) - wR(IV)) * rho_avg * cs_avg);

    // compute L, R sound speed
    const fp_t cLstar = (p_mid <= wL(I(Prim::Pres)))
                            ? 1.0_fp
                            : std::sqrt(1.0_fp + (eos.gamma + 1.0_fp) / (2.0_fp * eos.gamma) * (p_mid / wL(I(Prim::Pres)) - 1.0_fp));
    const fp_t cRstar = (p_mid <= wR(I(Prim::Pres)))
                            ? 1.0_fp
                            : std::sqrt(1.0_fp + (eos.gamma + 1.0_fp) / (2.0_fp * eos.gamma) * (p_mid / wR(I(Prim::Pres)) - 1.0_fp));

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
    const fp_t cp = std::max((mL * tR + mR * tL) / (mL + mR), 0.0_fp);

    // compute L/R fluxes along clamped line bm, bp
    vxL = wL(IV) - bM;
    vxR = wR(IV) - bP;

    const fp_t mass_flux_L = wL(I(Prim::Rho)) * vxL;
    const fp_t mass_flux_R = wR(I(Prim::Rho)) * vxR;
    fL(I(Cons::Rho)) = mass_flux_L;
    fR(I(Cons::Rho)) = mass_flux_R;

    fL(I(Cons::MomX)) = mass_flux_L * wL(I(Prim::Vx));
    fR(I(Cons::MomX)) = mass_flux_R * wR(I(Prim::Vx));

    if (NumDim > 1) {
        fL(I(Cons::MomY)) = mass_flux_L * wL(I(Prim::Vy));
        fR(I(Cons::MomY)) = mass_flux_R * wR(I(Prim::Vy));
    }
    if (NumDim > 2) {
        fL(I(Cons::MomZ)) = mass_flux_L * wL(I(Prim::Vz));
        fR(I(Cons::MomZ)) = mass_flux_R * wR(I(Prim::Vz));
    }
    fL(IM) += wL(I(Prim::Pres));
    fR(IM) += wR(I(Prim::Pres));

    fL(I(Cons::Ene)) = qL(I(Cons::Ene)) * vxL + wL(I(Prim::Pres)) * wL(IV);
    fR(I(Cons::Ene)) = qR(I(Cons::Ene)) * vxR + wR(I(Prim::Pres)) * wR(IV);

    // compute flux weights
    fp_t sL, sR, sM;
    if (aM >= 0.0_fp) {
        sL = aM / (aM - bM);
        sR = 0.0_fp;
        sM = -bM / (aM - bM);
    } else {
        sL = 0.0_fp;
        sR = -aM / (bP - aM);
        sM = bP / (bP - aM);
    }

    #pragma unroll
    for (int var = 0; var < n_hydro; ++var) {
        flux(var) = sL * fL(var) + sR * fR(var);
    }
    flux(IM) += sM * cp;
    flux(I(Cons::Ene)) += sM * cp * aM;
}

}

#else
#endif