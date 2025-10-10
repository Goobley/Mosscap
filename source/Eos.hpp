#if !defined(MOSSCAP_EOS_HPP)
#define MOSSCAP_EOS_HPP

#include "State.hpp"

namespace YAML { class Node; };

namespace Mosscap {

enum class ReconstructionEdge {
    Centred,
    Left,
    Right
};

enum class EosType {
    Ideal = 0,
    AnalyticLteH,
    TabulatedLteH,
    DexrtEos,
};
constexpr const char* EosTypeName[] = {"ideal", "analyticlteh", "tabulatedlteh"};
constexpr int NumEosType = sizeof(EosTypeName) / sizeof(EosTypeName[0]);

struct Simulation;

struct Eos {
    bool is_constant;
    fp_t gamma;
    fp_t y; // ion_frac
    fp_t avg_mass;
    Fp3d y_space;
    Fp3d T_space;

    bool init(Simulation& sim, const YAML::Node& config);

    inline bool init_ideal(fp_t gamma_, fp_t ion_frac, Simulation& sim) {
        is_constant = true;
        y = ion_frac;
        gamma = gamma_;
        return true;
    }

    bool init_analytic_lte_h(fp_t gamma, Simulation& sim, bool include_ionisation_energy);
    bool init_tabulated_lte_h(fp_t gamma, Simulation& sim, const std::string& table_path);
    bool init_dexrt(fp_t gamma, Simulation& sim);
};

KOKKOS_INLINE_FUNCTION fp_t temperature_si(fp_t pressure, fp_t n_baryon, fp_t y = 1.0_fp) {
    constexpr fp_t k_B = 1.380649e-23_fp; // [J / K]
    return pressure / (n_baryon * (1.0_fp + y) * k_B);
}

template <int NumDim, typename WType>
KOKKOS_INLINE_FUNCTION fp_t sound_speed(const fp_t& gamma, const WType& w) {
    // NOTE(cmo): This uses the "standard" gamma = c_P / c_V
    using Prim = Prim<NumDim>;
    return std::sqrt(gamma * w(I(Prim::Pres)) / w(I(Prim::Rho)));
}

template <int NumDim, typename QType, typename WType>
KOKKOS_INLINE_FUNCTION void cons_to_prim(const fp_t gamma, const QType& q, const WType& w) {
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
    const fp_t e_kin = 0.5_fp * q(I(Cons::Rho)) * v2_sum;
    // NOTE(cmo): Will probably need to bring EosView back in some capacity to handle ionisation energy
    w(I(Prim::Pres)) = (gamma - 1.0_fp) * ((q(I(Cons::Ene)) - e_kin));
}

template <int NumDim, typename WType, typename QType>
KOKKOS_INLINE_FUNCTION void prim_to_cons(const fp_t gamma, const WType& w, const QType& q) {
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
    const fp_t e_kin = 0.5_fp * w(I(Prim::Rho)) * v2_sum;
    const fp_t e_int = w(I(Prim::Pres)) / (gamma - 1.0_fp);
    q(I(Cons::Ene)) = e_int + e_kin;
}

template <int Axis, int NumDim, typename WType, typename FType>
KOKKOS_INLINE_FUNCTION void prim_to_flux(const fp_t gamma, const WType& w, const FType& f) {
    constexpr int IV1 = Velocity<Axis, NumDim>();
    // constexpr int IV2 = Velocity<(Axis + 1) % 3>();
    // constexpr int IV3 = Velocity<(Axis + 2) % 3>();
    constexpr int IM1 = Momentum<Axis, NumDim>();
    // constexpr int IM2 = Momentum<(Axis + 1) % 3>();
    // constexpr int IM3 = Momentum<(Axis + 2) % 3>();

    using Prim = Prim<NumDim>;
    using Cons = Cons<NumDim>;
    const fp_t mass_flux = w(I(Prim::Rho)) * w(IV1);
    fp_t e_kin = 0.0_fp;
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
    e_kin *= 0.5_fp * w(I(Prim::Rho));

    f(IM1) += w(I(Prim::Pres));

    const fp_t e_tot = w(I(Prim::Pres)) / (gamma - 1.0_fp) + e_kin;
    f(I(Cons::Ene)) = (e_tot + w(I(Prim::Pres))) * w(IV1);
}

}

#else
#endif