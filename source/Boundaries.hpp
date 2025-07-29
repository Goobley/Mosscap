#if !defined(MOSSCAP_BOUNDARIES_HPP)
#define MOSSCAP_BOUNDARIES_HPP

#include "Types.hpp"
#include "State.hpp"
#include "Simulation.hpp"
#include <fmt/core.h>

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
            constexpr int IM = Momentum<Axis, NumDim>();
            int coord[3] = {ii, ji, ki};
            const int pencil_idx = coord[Axis];
            int cflip = (2 * ng - 1) - coord[Axis];
            if (pencil_idx >= ng) {
                coord[Axis] = (dims[Axis] - 1) - (pencil_idx - ng);
                cflip = (dims[Axis] - 1) - (2 * ng - 1) + (pencil_idx - ng);
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

            auto Q_view = QtyView(state.Q, idx);
            auto Q_flip = QtyView(state.Q, i_flip);
            auto Q_periodic = QtyView(state.Q, i_periodic);

            BoundaryType start_bound, end_bound;
            JasUse(bdry);
            if constexpr (Axis == 0) {
                start_bound = bdry.xs;
                end_bound = bdry.xe;
            } else if constexpr (Axis == 1) {
                start_bound = bdry.ys;
                end_bound = bdry.ye;
            } else {
                start_bound = bdry.zs;
                end_bound = bdry.ze;
            }
            BoundaryType bound = (coord[Axis] < ng) ? start_bound : end_bound;

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
}

#else
#endif