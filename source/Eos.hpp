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
    DexPressure,
};
constexpr const char* EosTypeName[] = {"ideal", "analyticlteh", "tabulatedlteh", "dexpressure"};
constexpr int NumEosType = sizeof(EosTypeName) / sizeof(EosTypeName[0]);

struct Simulation;

struct Eos {
    bool is_constant;
    fp_t gamma;
    fp_t y; // ion_frac
    fp_t avg_mass = 1.0_fp;
    fp_t total_abund = 1.0_fp;
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

template <typename FTraits, typename WType>
KOKKOS_INLINE_FUNCTION fp_t sound_speed(const fp_t gamma, const WType& w) {
    // NOTE(cmo): This uses the "standard" gamma = c_P / c_V
    using Prim = typename FTraits::prim;
    return std::sqrt(gamma * w(I(Prim::Pres)) / w(I(Prim::Rho)));
}

template <typename FTraits, typename WType>
KOKKOS_INLINE_FUNCTION fp_t total_pressure(const fp_t mu0, const WType& w) {
    using Prim = typename FTraits::prim;
    fp_t pres = w(I(Prim::Pres));
    if constexpr (FTraits::is_mhd) {
        pres += 0.5_fp / mu0 * (square(w(I(Prim::Bx))) + square(w(I(Prim::By))) + square(w(I(Prim::Bz))));
    }
    return pres;
}

template <typename FTraits, int Axis, typename WType>
KOKKOS_INLINE_FUNCTION fp_t fast_wave_speed(const fp_t gamma, const fp_t mu0, const WType& w) {
    using Prim = typename FTraits::prim;
    const fp_t irho = 1.0_fp / w(I(Prim::Rho));
    const fp_t cs2 = gamma * w(I(Prim::Pres)) * irho;

    fp_t cfast2 = cs2;
    if constexpr (FTraits::is_mhd) {
        constexpr int IB1 = MagneticField<Axis, FTraits>();
        const fp_t irhomu = irho / mu0;
        const fp_t ca2 = (square(w(I(Prim::Bx))) + square(w(I(Prim::By))) + square(w(I(Prim::Bz)))) * irhomu;
        const fp_t c_axis2 = square(w(IB1)) * irhomu;
        // NOTE(cmo): This is the positive-definite form used in Athena++. It
        // expands correctly, but looks a bit weird.
        cfast2 = 0.5_fp * ((cs2 + ca2) + std::sqrt(square(ca2 - cs2) + 4.0_fp * cs2 * (ca2 - c_axis2)));
    }
    return std::sqrt(cfast2);
}

template <typename FTraits, typename QType, typename WType>
KOKKOS_INLINE_FUNCTION void cons_to_prim(const fp_t gamma, const fp_t mu0, const QType& q, const WType& w) {
    using Prim = typename FTraits::prim;
    using Cons = typename FTraits::cons;
    constexpr int NumDim = FTraits::is_mhd ? 3 : FTraits::num_dim;

    w(I(Prim::Rho)) = q(I(Cons::Rho));
    w(I(Prim::Vx)) = q(I(Cons::MomX)) / q(I(Cons::Rho));
    w(I(Prim::IonE)) = q(I(Cons::IonE));
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
    const fp_t e_ion = w(I(Prim::Rho)) * w(I(Prim::IonE));
    fp_t e_mag = 0.0_fp;
    if constexpr (FTraits::is_mhd) {
        w(I(Prim::Bx)) = q(I(Cons::Bx));
        w(I(Prim::By)) = q(I(Cons::By));
        w(I(Prim::Bz)) = q(I(Cons::Bz));
        e_mag = square(w(I(Prim::Bx))) + square(w(I(Prim::By))) + square(w(I(Prim::Bz)));
        e_mag /= (2.0_fp * mu0);
        if constexpr (is_instance(FTraits::fluid_type, FluidType::GlmMhd)) {
            w(I(Prim::Psi)) = q(I(Cons::Psi));
        }
        if constexpr (FTraits::has_hypertc) {
            w(I(Prim::HeatF)) = q(I(Cons::HeatF));
        }
    }
    w(I(Prim::Pres)) = (gamma - 1.0_fp) * (q(I(Cons::Ene)) - e_kin - e_mag - e_ion);
}

template <typename FTraits, typename WType, typename QType>
KOKKOS_INLINE_FUNCTION void prim_to_cons(const fp_t gamma, const fp_t mu0, const WType& w, const QType& q) {
    using Prim = typename FTraits::prim;
    using Cons = typename FTraits::cons;
    constexpr int NumDim = FTraits::is_mhd ? 3 : FTraits::num_dim;

    q(I(Cons::Rho)) = w(I(Prim::Rho));
    q(I(Cons::MomX)) = w(I(Prim::Rho)) * w(I(Prim::Vx));
    q(I(Cons::IonE)) = w(I(Prim::IonE));
    fp_t v2_sum = square(w(I(Prim::Vx)));
    if constexpr (NumDim > 1) {
        q(I(Cons::MomY)) = w(I(Prim::Rho)) * w(I(Prim::Vy));
        v2_sum += square(w(I(Prim::Vy)));
    }
    if constexpr (NumDim > 2) {
        q(I(Cons::MomZ)) = w(I(Prim::Rho)) * w(I(Prim::Vz));
        v2_sum += square(w(I(Prim::Vz)));
    }
    const fp_t e_ion = w(I(Prim::Rho)) * w(I(Prim::IonE));
    fp_t e_mag = 0.0_fp;
    if constexpr (FTraits::is_mhd) {
        q(I(Cons::Bx)) = w(I(Prim::Bx));
        q(I(Cons::By)) = w(I(Prim::By));
        q(I(Cons::Bz)) = w(I(Prim::Bz));
        e_mag = square(w(I(Prim::Bx))) + square(w(I(Prim::By))) + square(w(I(Prim::Bz)));
        e_mag /= (2.0_fp * mu0);
        if constexpr (is_instance(FTraits::fluid_type, FluidType::GlmMhd)) {
            q(I(Cons::Psi)) = w(I(Prim::Psi));
        }
        if constexpr (FTraits::has_hypertc) {
            q(I(Cons::HeatF)) = w(I(Prim::HeatF));
        }
    }
    const fp_t e_kin = 0.5_fp * w(I(Prim::Rho)) * v2_sum;
    const fp_t e_int = w(I(Prim::Pres)) / (gamma - 1.0_fp);
    q(I(Cons::Ene)) = e_int + e_kin + e_mag + e_ion;
}

constexpr bool HYPERTC_IN_FLUX_VECTOR = false;
template <typename FTraits, int Axis, typename WType, typename FType>
KOKKOS_INLINE_FUNCTION void prim_to_flux(const fp_t gamma, const fp_t mu0, const fp_t c_h, const WType& w, const FType& f) {
    using Prim = typename FTraits::prim;
    using Cons = typename FTraits::cons;
    constexpr int NumDim = FTraits::is_mhd ? 3 : FTraits::num_dim;
    constexpr int IV1 = Velocity<Axis, FTraits>();
    constexpr int IV2 = Velocity<(Axis + 1) % 3, FTraits>();
    constexpr int IV3 = Velocity<(Axis + 2) % 3, FTraits>();
    constexpr int IM1 = Momentum<Axis, FTraits>();
    constexpr int IB1 = MagneticField<Axis, FTraits>();
    constexpr int IB2 = MagneticField<(Axis + 1) % 3, FTraits>();
    constexpr int IB3 = MagneticField<(Axis + 2) % 3, FTraits>();

    if constexpr (FTraits::fluid_type == FluidType::HyperTcOnly) {
        for (int i = 0; i < FTraits::num_vars; ++i) {
            f(i) = 0.0_fp;
        }
        if constexpr (HYPERTC_IN_FLUX_VECTOR) {
            fp_t b2_in_plane = square(w(I(Prim::Bx)));
            if constexpr (FTraits::num_dim > 1) {
                b2_in_plane += square(w(I(Prim::By)));
            }
            if constexpr (FTraits::num_dim > 2)  {
                b2_in_plane += square(w(I(Prim::Bz)));
            }
            f(I(Cons::Ene)) = w(I(Prim::HeatF)) * w(IB1) / std::max(std::sqrt(b2_in_plane), 1e-60_fp);
        }
        return;
    }

    const fp_t mass_flux = w(I(Prim::Rho)) * w(IV1);
    fp_t e_kin = 0.0_fp;
    f(I(Cons::Rho)) = mass_flux;
    f(I(Cons::MomX)) = mass_flux * w(I(Prim::Vx));
    f(I(Cons::IonE)) = 0.0_fp;
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

    const fp_t inv_mu0 = 1.0_fp / mu0;
    fp_t p_tot = w(I(Prim::Pres));
    fp_t vdotB = 0.0_fp;
    fp_t b2 = 0.0_fp;
    fp_t b2_in_plane  = 0.0_fp;
    fp_t e_mag = 0.0_fp;
    if constexpr (FTraits::is_mhd) {
        b2 = square(w(I(Prim::Bx))) + square(w(I(Prim::By))) + square(w(I(Prim::Bz)));
        b2_in_plane = square(w(I(Prim::Bx)));
        if constexpr (FTraits::num_dim > 1) {
            b2_in_plane += square(w(I(Prim::By)));
        }
        if constexpr (FTraits::num_dim > 2)  {
            b2_in_plane += square(w(I(Prim::Bz)));
        }
        e_mag = b2 * 0.5_fp * inv_mu0;
        p_tot += e_mag;
        vdotB = w(I(Prim::Vx)) * w(I(Prim::Bx)) + w(I(Prim::Vy)) * w(I(Prim::By)) + w(I(Prim::Vz)) * w(I(Prim::Bz));
        f(I(Cons::MomX)) -= (w(IB1) * w(I(Prim::Bx))) * inv_mu0;
        f(I(Cons::MomY)) -= (w(IB1) * w(I(Prim::By))) * inv_mu0;
        f(I(Cons::MomZ)) -= (w(IB1) * w(I(Prim::Bz))) * inv_mu0;

        // Induction
        f(IB1) = 0.0_fp;
        f(IB2) = w(IV1) * w(IB2) - w(IV2) * w(IB1);
        f(IB3) = w(IV1) * w(IB3) - w(IV3) * w(IB1);

        if constexpr (is_instance(FTraits::fluid_type, FluidType::GlmMhd)) {
            f(IB1) = w(I(Prim::Psi));
            f(I(Prim::Psi)) = square(c_h) * w(IB1);
        }
    }

    f(IM1) += p_tot;
    const fp_t e_ion = w(I(Prim::Rho)) * w(I(Prim::IonE));
    const fp_t e_tot = w(I(Prim::Pres)) / (gamma - 1.0_fp) + e_kin + e_mag + e_ion;
    f(I(Cons::Ene)) = (e_tot + p_tot) * w(IV1);

    if constexpr (FTraits::is_mhd) {
        f(I(Cons::Ene)) -= (vdotB * w(IB1)) * inv_mu0;
        if constexpr (FTraits::has_hypertc) {
            if constexpr (HYPERTC_IN_FLUX_VECTOR) {
                f(I(Cons::Ene)) += w(I(Prim::HeatF)) * w(IB1) / std::max(std::sqrt(b2_in_plane), 1e-60_fp);
            }
            f(I(Cons::HeatF)) = 0.0_fp;
        }
    }
}

}

#else
#endif