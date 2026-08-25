#if !defined(MOSSCAP_ANALYTIC_LTE_H_HPP)
#define MOSSCAP_ANALYTIC_LTE_H_HPP

#include "Types.hpp"
#include "Simulation.hpp"

namespace Mosscap {

template <typename T>
KOKKOS_INLINE_FUNCTION T saha_rhs_H(T temp) {
    // NOTE(cmo): From sympy
    return T(2.4146830395719654e+21)*std::pow(temp, T(1.5))*std::exp(T(-157763.42386247337)/temp);
}

template <typename T>
KOKKOS_INLINE_FUNCTION T y_from_nhtot(T nhtot, T temp) {
    T X = saha_rhs_H(temp);
    constexpr bool use_old_saha_root = false;
    if constexpr (use_old_saha_root) {
        return T(0.5) * (-X + std::sqrt(square(X) + T(4) * nhtot * X)) / nhtot;
    } else {
        // The positive root of y^2 n_H / (1-y) = X, rationalised and
        // simplified to avoid subtractive cancellation when X >> n_H.
        const T sqrt_X = std::sqrt(X);
        return T(2) * sqrt_X / (sqrt_X + std::sqrt(X + T(4) * nhtot));
    }
}

template <typename T>
KOKKOS_INLINE_FUNCTION T y_from_ntot(T ntot, T temp) {
    T X = saha_rhs_H(temp);
    T ne = T(0.5) * (-2 * X + std::sqrt(square(2 * X) + 4 * ntot * X));
    return ne / (ntot - ne);
}

// NOTE(cmo): The actual solver equations derived with sympy are only really
// valid for total_abund = 1 (i.e. pure H), but when refactoring we added the
// terms throughout here, for when those are swapped out.
struct AnalyticLteH {
    static constexpr fp_t h_mass = 1.6737830080950003e-27_fp;
    static constexpr fp_t k_B = 1.380649e-23_fp;
    static constexpr fp_t chi_H = 2.178710282685096e-18_fp; // [J]

    bool include_ionisation_e;

    bool init(bool include_ionisation_e_=false) {
        include_ionisation_e = include_ionisation_e_;
        return true;
    }

    KOKKOS_INLINE_FUNCTION fp_t internal_energy(const Eos& eos, fp_t rho, fp_t y, fp_t T) const {
        const fp_t inv_mass_per_h = 1.0_fp / eos.mass_per_h;
        fp_t eint  = rho * (k_B / h_mass) * inv_mass_per_h * (eos.total_abund + y) * T;
        eint /= (eos.gamma - 1.0_fp);
        if (include_ionisation_e) {
            eint += y * rho * (chi_H / h_mass) * inv_mass_per_h;
        }
        return eint;
    }

    KOKKOS_INLINE_FUNCTION fp_t ionisation_energy(const Eos& eos, fp_t rho, fp_t y, fp_t T) const {
        // NOTE(claude): No abundance term -- the y * chi_H term is already per
        // hydrogen atom.
        const fp_t inv_mass_per_h = 1.0_fp / eos.mass_per_h;
        fp_t e_ion = 0.0_fp;
        if (include_ionisation_e) {
            e_ion += y * rho * (chi_H / h_mass) * inv_mass_per_h;
        }
        return e_ion;
    }

    template <typename FTraits>
    inline void update_eos(const Simulation& sim) const {
        const auto& state = sim.state;
        const auto& eos = sim.eos;
        const auto& sz = state.sz;
        const auto& Q = state.Q;
        const bool ionisation_e = include_ionisation_e;
        const fp_t inv_mass_per_h = 1.0_fp / eos.mass_per_h;
        const fp_t total_abund = eos.total_abund;
        const fp_t mu0 = state.mu0;
        using Cons = typename FTraits::cons;

        // NOTE(cmo): Scheme similar to lare
        dex_parallel_for(
            "Update analytic EOS (bisection)",
            FlatLoop<3>(sz.zc, sz.yc, sz.xc),
            KOKKOS_LAMBDA (int k, int j, int i) {
                const fp_t rho = Q(I(Cons::Rho), k, j, i);
                fp_t mom2_sum = square(Q(I(Cons::MomX), k, j, i));
                if constexpr (FTraits::is_mhd || FTraits::num_dim > 1) {
                    mom2_sum += square(Q(I(Cons::MomY), k, j, i));
                }
                if constexpr (FTraits::is_mhd || FTraits::num_dim > 2) {
                    mom2_sum += square(Q(I(Cons::MomZ), k, j, i));
                }
                const fp_t e_kin = 0.5_fp * mom2_sum / rho;
                fp_t e_mag = 0.0_fp;
                JasUse(mu0);
                if constexpr (FTraits::is_mhd) {
                    e_mag = (
                        square(Q(I(Cons::Bx), k, j, i))
                        + square(Q(I(Cons::By), k, j, i))
                        + square(Q(I(Cons::Bz), k, j, i))
                    ) / (2.0_fp * mu0);
                }
                const fp_t eint = Q(I(Cons::Ene), k, j, i) - e_kin - e_mag;

                const fp_t e_to_T = (eos.gamma - 1.0_fp) / (rho * (k_B / h_mass) * inv_mass_per_h);
                auto temp_from_y = [&](fp_t y) {
                    return e_to_T / (total_abund + y) * (eint - ionisation_e * y * rho * (chi_H / h_mass) * inv_mass_per_h);
                };

                const fp_t rho_to_nhtot = (1.0_fp / h_mass) * inv_mass_per_h;
                constexpr fp_t min_temperature = 100.0_fp;
                fp_t temp_bounds[2] = {
                    // 0.5_fp * e_to_T * (eint - rho * (chi_H / h_mass)), // Fully ionised
                    // e_to_T * eint // no ionisation
                    std::max(temp_from_y(1.0_fp), min_temperature),
                    temp_from_y(0.0_fp)
                };

                if (temp_bounds[0] > temp_bounds[1]) {
                    printf("%e > %e!!!\n", temp_bounds[0], temp_bounds[1]);
                    Kokkos::abort("Temperature bounds flipped!");
                }

                // TODO(cmo): We could start from our previous temperature?
                fp_t temp_step = temp_bounds[1] - temp_bounds[0];
                fp_t temp = temp_bounds[0];
                fp_t y;
                fp_t test_temp;
                fp_t temp_err;

                int iter;
                for (iter = 0; iter < 100; ++iter) {
                    temp_step *= 0.5_fp;
                    test_temp = temp + temp_step;
                    y = y_from_nhtot(rho * rho_to_nhtot, test_temp);
                    temp_err = test_temp - temp_from_y(y);
                    if (temp_err <= 0.0_fp) {
                        temp = test_temp; // upper half of range
                    }
                    constexpr fp_t temp_err_bound = 1e-1_fp;
                    if (std::abs(temp_step) < temp_err_bound || std::abs(temp_err) < temp_err_bound) {
                        break;
                    }
                }
                const fp_t pressure = rho * (total_abund + y) * (k_B / h_mass) * inv_mass_per_h * temp;

                eos.y_space(k, j, i) = y;
                eos.T_space(k, j, i) = temp;
                Q(I(Cons::IonE), k, j, i) = ionisation_e ? y * (chi_H / h_mass) * inv_mass_per_h : 0.0_fp;
                // eos.gamma_e_space(k, j, i) = 1.0_fp + pressure / eint;
            }
        );
        Kokkos::fence();

    }



};

}

#else
#endif
