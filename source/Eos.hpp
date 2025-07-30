#if !defined(MOSSCAP_EOS_HPP)
#define MOSSCAP_EOS_HPP

#include "State.hpp"

struct Gammas {
    fp_t Gamma;
    fp_t GammaM1;
};

enum class ReconstructionEdge {
    Centred,
    Left,
    Right
};

struct Eos {
    bool is_constant;
    fp_t Gamma;
    fp_t GammaM1;
    Fp3d gamma_space;
    // Reconstruction
    Fp3d gamma_space_R;
    Fp3d gamma_space_L;

    bool init_ideal(fp_t gamma) {
        is_constant = true;
        Gamma = gamma;
        GammaM1 = gamma - FP(1.0);
        return true;
    }

    KOKKOS_INLINE_FUNCTION Gammas get_gamma(const CellIndex& idx, const ReconstructionEdge& edge) const {
        if (is_constant) {
            return Gammas {
                .Gamma = Gamma,
                .GammaM1 = GammaM1
            };
        }
        const Fp3d* gamma_arrs[3] = {&gamma_space, &gamma_space_L, &gamma_space_R};
        const fp_t g = (*gamma_arrs[int(edge)])(idx.k, idx.j, idx.i);
        return Gammas {
            .Gamma = g,
            .GammaM1 = g - FP(1.0)
        };
    }
};

struct EosView {
    const Eos& eos;
    CellIndex idx;
    ReconstructionEdge edge;
    KOKKOS_INLINE_FUNCTION EosView(
        const Eos& eos_,
        CellIndex idx_,
        ReconstructionEdge = ReconstructionEdge::Centred
    ) : eos(eos_), idx(idx_)
    {}

    KOKKOS_INLINE_FUNCTION Gammas get_gamma() const {
        return eos.get_gamma(idx, edge);
    }
};

template <int NumDim, typename QType, typename WType>
KOKKOS_INLINE_FUNCTION void cons_to_prim(const EosView& eos, const QType& q, const WType& w) {
    using Prim = Prim<NumDim>;
    using Cons = Cons<NumDim>;
    w(I(Prim::Rho)) = q(I(Cons::Rho));
    w(I(Prim::Vx)) = q(I(Cons::MomX)) / q(I(Cons::Rho));
    fp_t v2_sum = square(w(I(Prim::Vx)));
    if constexpr (NumDim > 1) {
        w(I(Prim::Vy)) = q(I(Cons::MomY)) / q(I(Cons::Rho));
        v2_sum += square(w(I(Prim::Vy)));
    }
    if constexpr (NumDim > 2) {
        w(I(Prim::Vz)) = q(I(Cons::MomZ)) / q(I(Cons::Rho));
        v2_sum += square(w(I(Prim::Vz)));
    }
    const fp_t e_kin = FP(0.5) * q(I(Cons::Rho)) * v2_sum;
    const auto g = eos.get_gamma();
    w(I(Prim::Pres)) = g.GammaM1 * ((q(I(Cons::Ene)) - e_kin));
}

template <int NumDim, typename WType, typename QType>
KOKKOS_INLINE_FUNCTION void prim_to_cons(const EosView& eos, const WType& w, const QType& q) {
    using Prim = Prim<NumDim>;
    using Cons = Cons<NumDim>;
    q(I(Cons::Rho)) = w(I(Prim::Rho));
    q(I(Cons::MomX)) = w(I(Prim::Rho)) * w(I(Prim::Vx));
    fp_t v2_sum = square(w(I(Prim::Vx)));
    if constexpr (NumDim > 1) {
        q(I(Cons::MomY)) = w(I(Prim::Rho)) * w(I(Prim::Vy));
        v2_sum += square(w(I(Prim::Vy)));
    }
    if constexpr (NumDim > 2) {
        q(I(Cons::MomZ)) = w(I(Prim::Rho)) * w(I(Prim::Vz));
        v2_sum += square(w(I(Prim::Vz)));
    }
    const fp_t e_kin = FP(0.5) * w(I(Prim::Rho)) * v2_sum;
    const auto g = eos.get_gamma();
    const fp_t e_int = w(I(Prim::Pres)) / g.GammaM1;
    q(I(Cons::Ene)) = e_int + e_kin;
}

template <int Axis, int NumDim, typename WType, typename FType>
KOKKOS_INLINE_FUNCTION void prim_to_flux(const EosView& eos, const WType& w, const FType& f) {
    constexpr int IV1 = Velocity<Axis, NumDim>();
    // constexpr int IV2 = Velocity<(Axis + 1) % 3>();
    // constexpr int IV3 = Velocity<(Axis + 2) % 3>();
    constexpr int IM1 = Momentum<Axis, NumDim>();
    // constexpr int IM2 = Momentum<(Axis + 1) % 3>();
    // constexpr int IM3 = Momentum<(Axis + 2) % 3>();

    using Prim = Prim<NumDim>;
    using Cons = Cons<NumDim>;
    const fp_t mass_flux = w(I(Prim::Rho)) * w(IV1);
    fp_t e_kin = FP(0.0);
    f(I(Cons::Rho)) = mass_flux;
    f(I(Cons::MomX)) = mass_flux * w(I(Prim::Vx));
    e_kin += square(w(I(Prim::Vx)));
    if constexpr (NumDim > 1) {
        f(I(Cons::MomY)) = mass_flux * w(I(Prim::Vy));
        e_kin += square(w(I(Prim::Vy)));
    }
    if constexpr (NumDim > 2) {
        f(I(Cons::MomZ)) = mass_flux * w(I(Prim::Vz));
        e_kin += square(w(I(Prim::Vz)));
    }
    e_kin *= FP(0.5) * w(I(Prim::Rho));

    f(IM1) += w(I(Prim::Pres));

    const auto g = eos.get_gamma();
    const fp_t e_tot = w(I(Prim::Pres)) / g.GammaM1 + e_kin;
    f(I(Cons::Ene)) = (e_tot + w(I(Prim::Pres))) * w(IV1);
}


#else
#endif