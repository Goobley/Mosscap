#if !defined(MOSSCAP_CONDUCTION_COMMON_HPP)
#define MOSSCAP_CONDUCTION_COMMON_HPP
// Helpers shared by the parabolic (explicit/STS) and semi-implicit conduction source terms.
#include "../Simulation.hpp"
#include "ThermalConduction.hpp"

namespace Mosscap {

KOKKOS_INLINE_FUNCTION fp_t ion_frac(const Eos& eos, const CellIndex& idx) {
    fp_t y = eos.y;
    if (!eos.is_constant) {
        y = eos.y_space(idx.k, idx.j, idx.i);
    }
    return y;
}

template <int Axis>
KOKKOS_INLINE_FUNCTION CellIndex shift_along(const CellIndex& from, int how_much) {
    CellIndex result(from);
    result.along<Axis>() += how_much;
    return result;
}

template <int Axis>
KOKKOS_INLINE_FUNCTION QtyView shift_along(const QtyView& from, int how_much) {
    return QtyView(from.q, shift_along<Axis>(from.idx, how_much));
}

/// Number of interior cells along each axis (1 on unsimulated axes).
inline CellIndex interior_extent(const GridSize& sz) {
    return CellIndex{
        .i = sz.xc - 2 * sz.ng,
        .j = std::max(sz.yc - 2 * sz.ng, 1),
        .k = std::max(sz.zc - 2 * sz.ng, 1)
    };
}

/// First interior cell (index 0 on unsimulated axes).
inline CellIndex interior_start(const GridSize& sz) {
    return CellIndex{
        .i = sz.ng,
        .j = (sz.yc == 1) ? 0 : sz.ng,
        .k = (sz.zc == 1) ? 0 : sz.ng
    };
}

KOKKOS_INLINE_FUNCTION fp_t mc(const fp_t a, const fp_t b) {
    // phi(r) = max(0, min(2r, 0.5 * (1 + r), 2)). The term in the min is
    // multiplied through by up and sign terms implement the monotonicity
    // (and the factor of 2 that needs to multiply the second term)
    return (copysign(1.0_fp, a) + copysign(1.0_fp, b)) * std::min(std::abs(a), std::min(0.25_fp * std::abs(a + b), std::abs(b)));
}

KOKKOS_INLINE_FUNCTION fp_t mc4(const fp_t a, const fp_t b, const fp_t c, const fp_t d) {
    // 4 way limiter for transverse gradients in conduction
    return mc(mc(a, b), mc(c, d));
}

template <typename FTraits>
KOKKOS_INLINE_FUNCTION fp_t cell_temperature(const fp_t m_p, const Eos& eos, const QtyView& w) {
    using Prim = typename FTraits::prim;
    return temperature_si(
        w(I(Prim::Pres)),
        w(I(Prim::Rho)) / (eos.mass_per_h * m_p),
        eos.total_abund,
        ion_frac(eos, w.idx)
    );
}

/// The material conductivity at temperature T.
KOKKOS_INLINE_FUNCTION fp_t spitzer_kappa(const fp_t kappa0, const bool spitzer, const fp_t T) {
    return spitzer ? kappa0 * (square(T) * std::sqrt(T)) : kappa0;
}

/// The conductivity of a cell: TRAC's stored kappa_eff while TRAC is active, otherwise spitzer_kappa(T_cell).
KOKKOS_INLINE_FUNCTION fp_t cell_kappa(const ThermalConductionContext& ctx, const fp_t T_cell, const CellIndex& idx) {
    if (ctx.trac_kappa.initialized()) {
        return ctx.trac_kappa(idx.k, idx.j, idx.i);
    }
    return spitzer_kappa(ctx.kappa0, ctx.spitzer, T_cell);
}

/// cell_kappa for the cell w
template <typename FTraits>
KOKKOS_INLINE_FUNCTION fp_t cell_kappa_from_prim(
    const fp_t m_p,
    const Eos& eos,
    const ThermalConductionContext& ctx,
    const QtyView& w
) {
    if (!ctx.spitzer && !ctx.trac_kappa.initialized()) {
        return ctx.kappa0;
    }
    return cell_kappa(ctx, cell_temperature<FTraits>(m_p, eos, w), w.idx);
}

/// The face conductivity from the two adjacent cells' conductivities. Currently arithmetic mean.
KOKKOS_INLINE_FUNCTION fp_t face_kappa(const fp_t kappa_a, const fp_t kappa_b) {
    return 0.5_fp * (kappa_a + kappa_b);
}

/// The magnetic field that sets the conduction direction
template <typename FTraits>
KOKKOS_INLINE_FUNCTION vec3 conduction_field(const ThermalConductionContext& ctx, const QtyView& w) {
    using Prim = typename FTraits::prim;
    vec3 B(0.0_fp);
    B(0) = w(I(Prim::Bx));
    if (FTraits::num_dim > 1 || ctx.out_of_plane_field) {
        B(1) = w(I(Prim::By));
    }
    if (FTraits::num_dim > 2 || ctx.out_of_plane_field) {
        B(2) = w(I(Prim::Bz));
    }
    return B;
}

/// The explicit conduction time step limit (defined in ThermalConduction.cpp).
template <typename FTraits>
fp_t estimate_thermal_conduction_timestep(const Simulation& sim, const ThermalConductionContext& ctx);

template <typename FTraits, typename W>
KOKKOS_INLINE_FUNCTION fp_t prim_to_eint(const fp_t gamma, const W& w) {
    using Prim = FTraits::prim;

    return w(I(Prim::Pres)) / (gamma - 1.0_fp);
}

/// Refresh the ghost-cell pressure in W from the current STS stage so the
/// boundary faces see the stage temperature.  Ghosts follow the hydro BC type,
/// except: Constant keeps its start-of-step value (fixed temperature), and
/// UserFn mirrors the temperature (insulating), so user code is never run
/// inside the STS loop. This is a pain, but would require quite a lot of
/// rearchitecting to change.
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
                const fp_t temperature = cell_temperature<FTraits>(m_p, eos, QtyView(W, i_flip));
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
