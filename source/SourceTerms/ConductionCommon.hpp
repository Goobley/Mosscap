#if !defined(MOSSCAP_CONDUCTION_COMMON_HPP)
#define MOSSCAP_CONDUCTION_COMMON_HPP
// Helpers shared by the parabolic (explicit/STS) and hybrid2 conduction source terms.
#include "../Simulation.hpp"

namespace Mosscap {

KOKKOS_INLINE_FUNCTION fp_t ion_frac(const Eos& eos, const CellIndex& idx) {
    fp_t y = eos.y;
    if (!eos.is_constant) {
        y = eos.y_space(idx.k, idx.j, idx.i);
    }
    return y;
}

template <typename FTraits, typename W>
KOKKOS_INLINE_FUNCTION fp_t prim_to_eint(const fp_t gamma, const W& w) {
    using Prim = FTraits::prim;

    return w(I(Prim::Pres)) / (gamma - 1.0_fp);
}

/// Refresh the ghost-cell pressure in W from the current STS stage so the
/// boundary faces see the stage temperature (cf. temperature_bcs in Lare2d).
/// Ghosts follow the hydro BC type, except: Constant keeps its start-of-step
/// value (fixed temperature), and UserFn mirrors the temperature (insulating),
/// so user code is never run inside the STS loop.
template <int Axis, typename FTraits>
void fill_sts_pressure_ghosts_axis(const Simulation& sim) {
    static_assert(Axis < 3, "What are you doing?");
    JasUnpack(sim, state, eos);
    JasUnpack(state, W, sz);
    const auto& bdry = state.boundaries;
    const int ng = sz.ng;
    const fp_t m_p = state.p_mass;
    int dims[3] = {sz.xc, sz.yc, sz.zc};
    int launch_dims[3] = {sz.xc, sz.yc, sz.zc};
    launch_dims[Axis] = 2 * ng;

    dex_parallel_for(
        "STS pressure ghosts",
        FlatLoop<3>(launch_dims[2], launch_dims[1], launch_dims[0]),
        KOKKOS_LAMBDA (int ki, int ji, int ii) {
            using Prim = typename FTraits::prim;
            constexpr int IP = I(Prim::Pres);
            int coord[3] = {ii, ji, ki};
            const int pencil_idx = coord[Axis];
            const bool start = (pencil_idx < ng);
            int cflip = (2 * ng - 1) - pencil_idx;
            int cedge = ng;
            if (!start) {
                coord[Axis] = (dims[Axis] - 1) - (pencil_idx - ng);
                cflip = (dims[Axis] - 1) - (2 * ng - 1) + (pencil_idx - ng);
                cedge = (dims[Axis] - 1) - ng;
            }
            CellIndex idx{.i = coord[0], .j = coord[1], .k = coord[2]};
            CellIndex i_flip(idx);
            i_flip.along<Axis>() = cflip;
            CellIndex i_edge(idx);
            i_edge.along<Axis>() = cedge;
            CellIndex i_periodic(idx);
            i_periodic.along<Axis>() += (start ? 1 : -1) * (dims[Axis] - 2 * ng);

            BoundaryType bound;
            JasUse(bdry);
            if constexpr (Axis == 0) {
                bound = start ? bdry.xs : bdry.xe;
            } else if constexpr (Axis == 1) {
                bound = start ? bdry.ys : bdry.ye;
            } else {
                bound = start ? bdry.zs : bdry.ze;
            }

            QtyView w(W, idx);
            if (bound == BoundaryType::Periodic) {
                w(IP) = QtyView(W, i_periodic)(IP);
            } else if (
                bound == BoundaryType::Wall
                || bound == BoundaryType::Symmetric
                || bound == BoundaryType::SymmetricOutflowDiode
            ) {
                w(IP) = QtyView(W, i_flip)(IP);
            } else if (bound == BoundaryType::ZeroGrad) {
                w(IP) = QtyView(W, i_edge)(IP);
            } else if (bound == BoundaryType::UserFn) {
                // NOTE(claude): Mirror T rather than p, as a user BC may have
                // left a different density in the ghosts.
                QtyView w_flip(W, i_flip);
                const fp_t temperature = temperature_si(
                    w_flip(IP),
                    w_flip(I(Prim::Rho)) / (eos.mass_per_h * m_p),
                    eos.total_abund,
                    ion_frac(eos, i_flip)
                );
                const fp_t nh_tot = w(I(Prim::Rho)) / (eos.mass_per_h * m_p);
                w(IP) = temperature * nh_tot * (eos.total_abund + ion_frac(eos, idx)) * ConstantsF64::k_B;
            }
        }
    );
    Kokkos::fence();
}

template <typename FTraits>
void fill_sts_pressure_ghosts(const Simulation& sim) {
    fill_sts_pressure_ghosts_axis<0, FTraits>(sim);
    if constexpr (FTraits::num_dim > 1) {
        fill_sts_pressure_ghosts_axis<1, FTraits>(sim);
    }
    if constexpr (FTraits::num_dim > 2) {
        fill_sts_pressure_ghosts_axis<2, FTraits>(sim);
    }
}

}

#else
#endif
