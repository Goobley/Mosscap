#if !defined(MOSSCAP_CMA_HPP)
#define MOSSCAP_CMA_HPP

#include "Types.hpp"
#include "State.hpp"

namespace Mosscap {

    // def tracer_cma_flattening(n_tracer_cc, n_tracer_l, n_tracer_r):
    // # NOTE(cmo): Method 2
    // S_L_plus = np.sum(np.maximum(0.0, n_tracer_l - n_tracer_cc), axis=0)
    // S_L_minus = np.sum(np.maximum(0.0, n_tracer_cc - n_tracer_l), axis=0)
    // S_R_plus = np.sum(np.maximum(0.0, n_tracer_r - n_tracer_cc), axis=0)
    // S_R_minus = np.sum(np.maximum(0.0, n_tracer_cc - n_tracer_r), axis=0)

    // delta_i_min_L = np.minimum(S_L_plus, S_L_minus)
    // delta_i_min_R = np.minimum(S_R_plus, S_R_minus)
    // delta_i_max_L = np.maximum(S_L_plus, S_L_minus)
    // delta_i_max_R = np.maximum(S_R_plus, S_R_minus)
    // s_L = 0.5 * np.abs(
    //     np.sign(n_tracer_r - n_tracer_l) - np.sign(S_L_plus - S_L_minus)
    // )
    // s_R = 0.5 * np.abs(
    //     np.sign(n_tracer_r - n_tracer_l) + np.sign(S_R_plus - S_R_minus)
    // )
    // beta = 0.25
    // w_L = s_L * np.maximum(0.0, np.minimum(1.0, beta * (delta_i_max_L - delta_i_min_L) / (delta_i_min_L + 1e-20)))
    // w_R = s_R * np.maximum(0.0, np.minimum(1.0, beta * (delta_i_max_R - delta_i_min_R) / (delta_i_min_R + 1e-20)))

    // n_tracer_l[...] = w_L * n_tracer_cc + (1.0 - w_L) * n_tracer_l
    // n_tracer_r[...] = w_R * n_tracer_cc + (1.0 - w_R) * n_tracer_r

KOKKOS_INLINE_FUNCTION void tracer_cma_flatten(
    const QtyView& cc,
    const QtyView& l,
    const QtyView& r,
    const CmaParams& cma
) {
    for (int fluid = 0; fluid < cma.fluid_start_idx.extent(0); ++fluid) {
        // NOTE(cmo): Flattening method 2 of the CMA paper
        fp_t S_L_plus = 0.0_fp;
        fp_t S_L_minus = 0.0_fp;
        fp_t S_R_plus = 0.0_fp;
        fp_t S_R_minus = 0.0_fp;
        for (int i = cma.fluid_start_idx(fluid); i < cma.fluid_end_idx(fluid); ++i) {
            S_L_plus += std::max(0.0_fp, l(i) - cc(i));
            S_L_minus += std::max(0.0_fp, cc(i) - l(i));
            S_R_plus += std::max(0.0_fp, r(i) - cc(i));
            S_R_minus += std::max(0.0_fp, cc(i) - r(i));
        }

        auto sign = [](fp_t x) {
            if (x == 0.0_fp) {
                return 0.0_fp;
            }
            return std::copysign(1.0_fp, x);
        };

        const fp_t delta_i_min_L = std::min(S_L_plus, S_L_minus);
        const fp_t delta_i_min_R = std::min(S_R_plus, S_R_minus);
        const fp_t delta_i_max_L = std::max(S_L_plus, S_L_minus);
        const fp_t delta_i_max_R = std::max(S_R_plus, S_R_minus);
        for (int i = cma.fluid_start_idx(fluid); i < cma.fluid_end_idx(fluid); ++i) {
            const fp_t s_L = 0.5_fp * std::abs(
                sign(r(i) - l(i)) - sign(S_L_plus - S_L_minus)
            );
            const fp_t s_R = 0.5_fp * std::abs(
                sign(r(i) - l(i)) + sign(S_R_plus - S_R_minus)
            );
            constexpr fp_t beta = 0.25_fp;
            const fp_t w_L = std::max(
                0.0_fp,
                std::min(
                    1.0_fp,
                    beta * (delta_i_max_L - delta_i_min_L) / (delta_i_min_L + 1e-80_fp)
                )
            );
            const fp_t w_R = std::max(
                0.0_fp,
                std::min(
                    1.0_fp,
                    beta * (delta_i_max_R - delta_i_min_R) / (delta_i_min_R + 1e-80_fp)
                )
            );
            l(i) = w_L * cc(i) + (1.0_fp - w_L) * l(i);
            r(i) = w_R * cc(i) + (1.0_fp - w_R) * r(i);
        }
    }
}

KOKKOS_INLINE_FUNCTION void tracer_cma_normalise(
    const QtyView& recon,
    const CmaParams& cma
) {
    for (int fluid = 0; fluid < cma.fluid_start_idx.extent(0); ++fluid) {
        fp_t fluid_sum = 0.0_fp;
        for (int i = cma.fluid_start_idx(fluid); i < cma.fluid_end_idx(fluid); ++i) {
            fluid_sum += recon(i);
        }
        const fp_t ratio = cma.fluid_inv_sum(fluid) / fluid_sum;
        for (int i = cma.fluid_start_idx(fluid); i < cma.fluid_end_idx(fluid); ++i) {
            recon(i) *= ratio;
        }
    }
}

}

#else
#endif