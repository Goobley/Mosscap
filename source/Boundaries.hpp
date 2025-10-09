#if !defined(MOSSCAP_BOUNDARIES_HPP)
#define MOSSCAP_BOUNDARIES_HPP

#include "Types.hpp"
#include "State.hpp"
#include "Simulation.hpp"
#include <fmt/core.h>

namespace Mosscap {

template <int Axis, int NumDim>
inline void fill_one_bc_impl(const State& state) {
    static_assert(Axis < 3, "What are you doing?");
    const auto& sz = state.sz;
    const auto& bdry = state.boundaries;
    const int ng = state.sz.ng;

    constexpr const char* kernel_name[3] = {"Fill BCs x", "Fill BCs y", "Fill BCs z"};
    int dims[3] = {sz.xc, sz.yc, sz.zc};
    int launch_dims[3] = {sz.xc, sz.yc, sz.zc};
    launch_dims[Axis] = 2 * ng;

    dex_parallel_for(
        kernel_name[Axis],
        FlatLoop<3>(launch_dims[2], launch_dims[1], launch_dims[0]),
        KOKKOS_LAMBDA (int ki, int ji, int ii) {
            using Cons = Cons<NumDim>;
            constexpr int IM = Momentum<Axis, NumDim>();
            int coord[3] = {ii, ji, ki};
            const int pencil_idx = coord[Axis];
            int cflip = (2 * ng - 1) - coord[Axis];
            int cedge = ng;
            if (pencil_idx >= ng) {
                coord[Axis] = (dims[Axis] - 1) - (pencil_idx - ng);
                cflip = (dims[Axis] - 1) - (2 * ng - 1) + (pencil_idx - ng);
                cedge = (dims[Axis] - 1) - ng;
            }

            CellIndex idx{
                .i = coord[0],
                .j = coord[1],
                .k = coord[2]
            };
            CellIndex i_flip{
                .i = coord[0],
                .j = coord[1],
                .k = coord[2]
            };
            i_flip.along<Axis>() = cflip;
            CellIndex i_periodic{
                .i = coord[0],
                .j = coord[1],
                .k = coord[2]
            };
            i_periodic.along<Axis>() += (pencil_idx < ng ? 1 : -1) * (dims[Axis] - 2 * ng);
            CellIndex i_edge(idx);
            i_edge.along<Axis>() = cedge;

            auto Q_view = QtyView(state.Q, idx);
            auto Q_flip = QtyView(state.Q, i_flip);
            auto Q_periodic = QtyView(state.Q, i_periodic);
            auto Q_edge = QtyView(state.Q, i_edge);

            const bool start = (coord[Axis] < ng);
            BoundaryType start_bound, end_bound;
            JasUse(bdry);
            decltype(bdry.xs_const) const_vals;
            if constexpr (Axis == 0) {
                start_bound = bdry.xs;
                end_bound = bdry.xe;
                const_vals = (start) ? bdry.xs_const : bdry.xe_const;
            } else if constexpr (Axis == 1) {
                start_bound = bdry.ys;
                end_bound = bdry.ye;
                const_vals = (start) ? bdry.ys_const : bdry.ye_const;
            } else {
                start_bound = bdry.zs;
                end_bound = bdry.ze;
                const_vals = (start) ? bdry.zs_const : bdry.ze_const;
            }
            BoundaryType bound = (start) ? start_bound : end_bound;

            if (bound == BoundaryType::Wall) {
                for (int var = 0; var < state.Q.extent(0); ++var) {
                    Q_view(var) = Q_flip(var);
                }
                Q_view(IM) = -Q_view(IM);
            } else if (bound == BoundaryType::Periodic) {
                for (int var = 0; var < state.Q.extent(0); ++var) {
                    Q_view(var) = Q_periodic(var);
                }
            } else if (bound == BoundaryType::Symmetric) {
                for (int var = 0; var < state.Q.extent(0); ++var) {
                    Q_view(var) = Q_flip(var);
                }
            } else if (bound == BoundaryType::SymmetricOutflowDiode) {
                for (int var = 0; var < state.Q.extent(0); ++var) {
                    Q_view(var) = Q_flip(var);
                }
                const fp_t mom_pre = Q_view(IM);
                if (start) {
                    Q_view(IM) = std::min(mom_pre, FP(0.0));
                } else {
                    Q_view(IM) = std::max(mom_pre, FP(0.0));
                }
                Q_view(I(Cons::Ene)) -= square(Q_view(IM) - mom_pre) / Q_view(I(Cons::Rho));
            } else if (bound == BoundaryType::ZeroGrad) {
                for (int var = 0; var < state.Q.extent(0); ++var) {
                    Q_view(var) = Q_edge(var);
                }
            } else if (bound == BoundaryType::Constant) {
                for (int var = 0; var < state.Q.extent(0); ++var) {
                    Q_view(var) = const_vals(var);
                }
            }
        }
    );
    Kokkos::fence();
}

template <int NumDim>
inline void fill_bcs_impl(const State& state) {
    fill_one_bc_impl<0, NumDim>(state);
    if constexpr (NumDim > 1) {
        fill_one_bc_impl<1, NumDim>(state);
    }
    if constexpr (NumDim > 2) {
        fill_one_bc_impl<2, NumDim>(state);
    }
}

inline void fill_bcs(const Simulation& sim) {
    switch (sim.num_dim) {
        case 1: {
            fill_bcs_impl<1>(sim.state);
        } break;
        case 2: {
            fill_bcs_impl<2>(sim.state);
        } break;
        case 3: {
            fill_bcs_impl<3>(sim.state);
        } break;
        default:
            KOKKOS_ASSERT(false && "Weird num dim");
    }
    if (sim.user_bc) {
        sim.user_bc(sim);
    }
}

}

#else
#endif