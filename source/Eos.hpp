#if !defined(MOSSCAP_EOS_HPP)
#define MOSSCAP_EOS_HPP

#include "State.hpp"


template <typename QType, typename WType>
KOKKOS_INLINE_FUNCTION void cons_to_prim(const QType& q, const WType& w) {
    w(I(Prim::Rho)) = q(I(Cons::Rho));
    w(I(Prim::Vx)) = q(I(Cons::MomX)) / q(I(Cons::Rho));
    fp_t v2_sum = square(w(I(Prim::Vx)));
    if constexpr (NUM_DIM > 1) {
        w(I(Prim::Vy)) = q(I(Cons::MomY)) / q(I(Cons::Rho));
        v2_sum += square(w(I(Prim::Vy)));
    }
    if constexpr (NUM_DIM > 2) {
        w(I(Prim::Vz)) = q(I(Cons::MomZ)) / q(I(Cons::Rho));
        v2_sum += square(w(I(Prim::Vz)));
    }
    const fp_t e_kin = FP(0.5) * q(I(Cons::Rho)) * v2_sum;
    // TODO(cmo): EOS
    w(I(Prim::Pres)) = GammaM1 * ((q(I(Cons::Ene)) - e_kin));
}

template <typename WType, typename QType>
KOKKOS_INLINE_FUNCTION void prim_to_cons(const WType& w, const QType& q) {
    q(I(Cons::Rho)) = w(I(Prim::Rho));
    q(I(Cons::MomX)) = w(I(Prim::Rho)) * w(I(Prim::Vx));
    fp_t v2_sum = square(w(I(Prim::Vx)));
    if constexpr (NUM_DIM > 1) {
        q(I(Cons::MomY)) = w(I(Prim::Rho)) * w(I(Prim::Vy));
        v2_sum += square(w(I(Prim::Vy)));
    }
    if constexpr (NUM_DIM > 2) {
        q(I(Cons::MomZ)) = w(I(Prim::Rho)) * w(I(Prim::Vz));
        v2_sum += square(w(I(Prim::Vz)));
    }
    const fp_t e_kin = FP(0.5) * w(I(Prim::Rho)) * v2_sum;
    // TODO(cmo): EOS
    const fp_t e_int = w(I(Prim::Pres)) / GammaM1;
    q(I(Cons::Ene)) = e_int + e_kin;
}

template <int Axis, typename WType, typename FType>
KOKKOS_INLINE_FUNCTION void prim_to_flux(const WType& w, const FType& f) {
    constexpr int IV1 = Velocity<Axis>();
    // constexpr int IV2 = Velocity<(Axis + 1) % 3>();
    // constexpr int IV3 = Velocity<(Axis + 2) % 3>();
    constexpr int IM1 = Momentum<Axis>();
    // constexpr int IM2 = Momentum<(Axis + 1) % 3>();
    // constexpr int IM3 = Momentum<(Axis + 2) % 3>();

    const fp_t mass_flux = w(I(Prim::Rho)) * w(IV1);
    fp_t e_kin = FP(0.0);
    f(I(Cons::Rho)) = mass_flux;
    f(I(Cons::MomX)) = mass_flux * w(I(Prim::Vx));
    e_kin += square(w(I(Prim::Vx)));
    if constexpr (NUM_DIM > 1) {
        f(I(Cons::MomY)) = mass_flux * w(I(Prim::Vy));
        e_kin += square(w(I(Prim::Vy)));
    }
    if constexpr (NUM_DIM > 2) {
        f(I(Cons::MomZ)) = mass_flux * w(I(Prim::Vz));
        e_kin += square(w(I(Prim::Vz)));
    }
    e_kin *= FP(0.5) * w(I(Prim::Rho));

    f(IM1) += w(I(Prim::Pres));

    const fp_t e_tot = w(I(Prim::Pres)) / GammaM1 + e_kin;
    f(I(Cons::Ene)) = (e_tot + w(I(Prim::Pres))) * w(IV1);
}


#else
#endif