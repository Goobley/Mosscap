#if !defined(MOSSCAP_BOUNDARIES_HPP)
#define MOSSCAP_BOUNDARIES_HPP

#include "Types.hpp"
#include "State.hpp"
#include <fmt/core.h>

inline void fill_bcs(const State& state) {
    const auto& sz = state.sz;
    const auto& bdry = state.boundaries;
    const int ng = state.sz.ng;
    dex_parallel_for(
        "Fill BCs x",
        FlatLoop<3>(sz.zc, sz.yc, 2 * ng),
        KOKKOS_LAMBDA (int k, int j, int ig) {
            int i = ig;
            int iflip = (2 * ng - 1) - i;
            if (ig >= ng) {
                i = (sz.xc - 1) - (ig - ng);
                iflip = (sz.xc - 1) - (2 * ng - 1) -  (ig - ng);
            }

            if (ig < ng) {
                if (bdry.xs == BoundaryType::Wall) {
                    for (int var = 0; var < state.Q.extent(0); ++var) {
                        state.Q(var, k, j, i) = state.Q(var, k, j, iflip);
                    }
                    state.Q(I(Cons::MomX), k, j, i) = -state.Q(I(Cons::MomX), k, j, i);
                } else if (bdry.xs == BoundaryType::Periodic) {
                    for (int var = 0; var < state.Q.extent(0); ++var) {
                        state.Q(var, k, j, i) = state.Q(var, k, j, sz.xc - ng - i);
                    }
                } else if (bdry.xs == BoundaryType::Symmetric) {
                    for (int var = 0; var < state.Q.extent(0); ++var) {
                        state.Q(var, k, j, i) = state.Q(var, k, j, iflip);
                    }
                }
            } else {
                if (bdry.xe == BoundaryType::Wall) {
                    for (int var = 0; var < state.Q.extent(0); ++var) {
                        state.Q(var, k, j, i) = state.Q(var, k, j, iflip);
                    }
                    state.Q(I(Cons::MomX), k, j, i) = -state.Q(I(Cons::MomX), k, j, i);
                } else if (bdry.xe == BoundaryType::Periodic) {
                    for (int var = 0; var < state.Q.extent(0); ++var) {
                        state.Q(var, k, j, i) = state.Q(var, k, j, ng + (ig - ng));
                    }
                } else if (bdry.xe == BoundaryType::Symmetric) {
                    for (int var = 0; var < state.Q.extent(0); ++var) {
                        state.Q(var, k, j, i) = state.Q(var, k, j, iflip);
                    }
                }
            }
        }
    );
    Kokkos::fence();
    if constexpr (NUM_DIM > 1)  {
        dex_parallel_for(
            "Fill BCs y",
            FlatLoop<3>(sz.zc, 2 * ng, sz.xc),
            KOKKOS_LAMBDA (int k, int jg, int i) {
                int j = jg;
                int jflip = (2 * ng - 1) - j;
                if (jg >= ng) {
                    j = (sz.yc - 1) - (jg - ng);
                    jflip = (sz.yc - 1) - (2 * ng - 1) -  (jg - ng);
                }

                if (jg < ng) {
                    if (bdry.ys == BoundaryType::Wall) {
                        for (int var = 0; var < state.Q.extent(0); ++var) {
                            state.Q(var, k, j, i) = state.Q(var, k, jflip, i);
                        }
                        state.Q(I(Cons::MomY), k, j, i) = -state.Q(I(Cons::MomY), k, j, i);
                    } else if (bdry.ys == BoundaryType::Periodic) {
                        for (int var = 0; var < state.Q.extent(0); ++var) {
                            state.Q(var, k, j, i) = state.Q(var, k, sz.yc - ng - j, i);
                        }
                    } else if (bdry.ys == BoundaryType::Symmetric) {
                        for (int var = 0; var < state.Q.extent(0); ++var) {
                            state.Q(var, k, j, i) = state.Q(var, k, jflip, i);
                        }
                    }
                } else {
                    if (bdry.ye == BoundaryType::Wall) {
                        for (int var = 0; var < state.Q.extent(0); ++var) {
                            state.Q(var, k, j, i) = state.Q(var, k, jflip, i);
                        }
                        state.Q(I(Cons::MomY), k, j, i) = -state.Q(I(Cons::MomY), k, j, i);
                    } else if (bdry.ye == BoundaryType::Periodic) {
                        for (int var = 0; var < state.Q.extent(0); ++var) {
                            state.Q(var, k, j, i) = state.Q(var, k, ng + (jg - ng), i);
                        }
                    } else if (bdry.ye == BoundaryType::Symmetric) {
                        for (int var = 0; var < state.Q.extent(0); ++var) {
                            state.Q(var, k, j, i) = state.Q(var, k, jflip, i);
                        }
                    }
                }
            }
        );
        Kokkos::fence();
    }
    static_assert(NUM_DIM < 3, "Only done 2D");

}

#else
#endif