#if !defined(MOSSCAP_TABULATE_LTE_H_HPP)
#define MOSSCAP_TABULATE_LTE_H_HPP

#include "Simulation.hpp"
#include "YAKL_netcdf.h"

struct TabulatedLteH {
    struct LogSpace {
        fp_t start;
        fp_t step;
        i32 num;
    };

    struct Sample {
        fp_t y;
        fp_t log_T;
    };

    LogSpace eint_rho_grid;
    LogSpace rho_grid;

    Fp2d y_table;
    Fp2d log_T_table;
    static constexpr fp_t h_mass = FP(1.6737830080950003e-27);
    static constexpr fp_t k_B = FP(1.380649e-23);
    static constexpr fp_t chi_H = FP(2.178710282685096e-18); // [J]

    inline bool init(const std::string& table_path) {
        yakl::SimpleNetCDF nc;
        nc.open(table_path, yakl::NETCDF_MODE_READ);

        f64 start, step;
        i64 num;
        nc.read(start, "log_eint_rho_start");
        nc.read(step, "log_eint_rho_step");
        nc.read(num, "n_eint_rho");

        eint_rho_grid = LogSpace {
            .start = fp_t(start),
            .step = fp_t(step),
            .num = i32(num)
        };

        nc.read(start, "log_rho_start");
        nc.read(step, "log_rho_step");
        nc.read(num, "n_rho");
        rho_grid = LogSpace {
            .start = fp_t(start),
            .step = fp_t(step),
            .num = i32(num)
        };

        nc.read(y_table, "y_table");
        nc.read(log_T_table, "log_T_table");
        return true;
    }

    KOKKOS_INLINE_FUNCTION Sample sample(fp_t log_eint_rho, fp_t log_rho) const {
        const fp_t frac_eint_rho = (log_eint_rho - eint_rho_grid.start) / eint_rho_grid.step;
        int ix = int(std::floor(frac_eint_rho));
        int ixp;
        fp_t tx, txp;

        const i32 max_x = eint_rho_grid.num - 1;
        const i32 max_y = rho_grid.num - 1;

        if (frac_eint_rho < FP(0.0) || frac_eint_rho >= max_x) {
            ix = std::min(std::max(ix, 0), max_x);
            ixp = ix;
            tx = FP(1.0);
            txp = FP(0.0);
        } else {
            ixp = ix + 1;
            txp = frac_eint_rho - ix;
            tx = FP(1.0) - txp;
        }
        if (ix < 0 || ixp > eint_rho_grid.num) {
            printf("Sadness x: %d, %d (%d) %e\n", ix, ixp, eint_rho_grid.num, log_eint_rho);
        }

        const fp_t frac_rho = (log_rho - rho_grid.start) / rho_grid.step;
        int iy = int(std::floor(frac_rho));
        int iyp;
        fp_t ty, typ;

        if (frac_rho < FP(0.0) || frac_rho >= max_y) {
            iy = std::min(std::max(iy, 0), max_y);
            iyp = iy;
            ty = FP(1.0);
            typ = FP(0.0);
        } else {
            iyp = iy + 1;
            typ = frac_rho - iy;
            ty = FP(1.0) - typ;
        }
        if (iy < 0 || iyp > eint_rho_grid.num) {
            printf("Sadness y: %d, %d (%d): %e\n", iy, iyp, rho_grid.num, log_rho);
        }

        const fp_t result_y = (
            ty * (tx * y_table(iy, ix) + txp * y_table(iy, ixp)) +
            typ * (tx * y_table(iyp, ix) + txp * y_table(iyp, ixp))
        );
        const fp_t result_log_T = (
            ty * (tx * log_T_table(iy, ix) + txp * log_T_table(iy, ixp)) +
            typ * (tx * log_T_table(iyp, ix) + txp * log_T_table(iyp, ixp))
        );
        return Sample {
            .y = result_y,
            .log_T = result_log_T
        };
    }

    template <int NumDim>
    inline void update_eos(const Simulation& sim) const {
        const auto& state = sim.state;
        const auto& eos = sim.eos;
        const auto& sz = state.sz;
        const auto& Q = state.Q;

        using Cons = Cons<NumDim>;

        dex_parallel_for(
            "Update tabulated EOS",
            FlatLoop<3>(sz.zc, sz.yc, sz.xc),
            KOKKOS_CLASS_LAMBDA (int k, int j, int i) {
                fp_t mom2_sum = square(Q(I(Cons::MomX), k, j, i));
                if constexpr (NumDim > 1) {
                    mom2_sum += square(Q(I(Cons::MomY), k, j, i));
                }
                if constexpr (NumDim > 2) {
                    mom2_sum += square(Q(I(Cons::MomZ), k, j, i));
                }
                const fp_t rho = Q(I(Cons::Rho), k, j, i);
                const fp_t e_kin = FP(0.5) * mom2_sum / rho;
                const fp_t eint = Q(I(Cons::Ene), k, j, i) - e_kin;
                if (eint < FP(0.0)) {
                    Kokkos::abort("eint negative");
                }

                const fp_t log_rho = std::log10(rho);
                const fp_t log_eint_rho = std::log10(eint) - log_rho;

                Sample s = sample(log_eint_rho, log_rho);
                const fp_t prev_y = eos.y_space(k, j, i);
                eos.y_space(k, j, i) = s.y;
                const fp_t delta_eint = (prev_y < FP(0.0)) ? FP(0.0) : rho * (chi_H / h_mass) * (s.y - prev_y);
                const fp_t T = std::pow(FP(10.0), s.log_T);
                eos.T_space(k, j, i) = T;

                // P = nh (1 + y) kB T
                const fp_t pressure = rho * (k_B / h_mass) * (FP(1.0) + s.y) * T;
                const fp_t new_eint = pressure / (eos.Gamma - FP(1.0)) + rho * (chi_H / h_mass) * s.y;

                eos.gamma_e_space(k, j, i) = FP(1.0) + pressure / (new_eint);
                // Q(I(Cons::Ene), k, j, i) = new_eint + e_kin;
                Q(I(Cons::Ene), k, j, i) += delta_eint;
            }
        );
        Kokkos::fence();
    }
};

#else
#endif