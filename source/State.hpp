#if !defined(MOSSCAP_STATE_HPP)
#define MOSSCAP_STATE_HPP

#include <functional>

#include "Types.hpp"

template <int NumDim = 1>
struct Prim {
    static constexpr i32 Rho = 0;
    static constexpr i32 Vx = 1;
    static constexpr i32 Vy = NumDim > 1 ? 2 : 1024;
    static constexpr i32 Vz = NumDim > 2 ? 3 : 1024;
    static constexpr i32 Pres = 1 + NumDim;
};

template <int NumDim = 1>
struct Cons {
    static constexpr i32 Rho = 0;
    static constexpr i32 MomX = 1;
    static constexpr i32 MomY = NumDim > 1 ? 2 : 1024;
    static constexpr i32 MomZ = NumDim > 2 ? 3 : 1024;
    static constexpr i32 Ene = 1 + NumDim;
};

template <int NumDim>
constexpr int N_HYDRO_VARS = 2 + NumDim;

constexpr int get_num_hydro_vars(int num_dim)  {
    return 2 + num_dim;
}

template <typename E>
constexpr int I(E e) {
    return static_cast<int>(e);
}

template <int Axis, int NumDim>
constexpr int Velocity() {
    if constexpr (Axis == 0) {
        return I(Prim<NumDim>::Vx);
    } else if constexpr (Axis == 1) {
        return I(Prim<NumDim>::Vy);
    } else if constexpr (Axis == 2) {
        return I(Prim<NumDim>::Vz);
    }
}

template <int Axis, int NumDim>
constexpr int Momentum() {
    if constexpr (Axis == 0) {
        return I(Cons<NumDim>::MomX);
    } else if constexpr (Axis == 1) {
        return I(Cons<NumDim>::MomY);
    } else if constexpr (Axis == 2) {
        return I(Cons<NumDim>::MomZ);
    }
}

enum class BoundaryType : i32 {
    Wall = 0,
    Periodic,
    Symmetric,
    Constant,
    UserFn
};
constexpr const char* BoundaryTypeName[] = {
    "wall",
    "periodic",
    "symmetric",
    "constant",
    "user_fn"
};
constexpr int NumBoundaryType = sizeof(BoundaryTypeName) / sizeof(BoundaryTypeName[0]);

struct Boundaries {
    BoundaryType xs;
    BoundaryType xe;
    BoundaryType ys;
    BoundaryType ye;
    BoundaryType zs;
    BoundaryType ze;

    /// Storage for constant boundaries -- may be longer than actual content due
    /// to dimensionality, make sure to loop over the correct number!
    yakl::SArray<fp_t, 1, N_HYDRO_VARS<3>> xs_const;
    yakl::SArray<fp_t, 1, N_HYDRO_VARS<3>> xe_const;
    yakl::SArray<fp_t, 1, N_HYDRO_VARS<3>> ys_const;
    yakl::SArray<fp_t, 1, N_HYDRO_VARS<3>> ye_const;
    yakl::SArray<fp_t, 1, N_HYDRO_VARS<3>> zs_const;
    yakl::SArray<fp_t, 1, N_HYDRO_VARS<3>> ze_const;
};


struct GridSize {
    i32 xc = 0; /// x-cells including ghosts
    i32 yc = 0; /// y-cells including ghosts
    i32 zc = 0; /// z-cells including ghosts
    i32 ng = 0; /// num ghost cells (same on both ends of all axes)
};

struct State {
    GridSize sz; /// Grid dimensions + number of ghosts
    fp_t dx; /// Spatial grid step (constant)
    Boundaries boundaries; /// Boundary handling specifications
    Fp4d Q; // Conserved State
    Fp4d W; /// Primitive State

    KOKKOS_INLINE_FUNCTION vec3 get_pos(int i, int j=0, int k=0) const {
        vec3 result;
        const fp_t ghost_offset = -(sz.ng - FP(0.5)) * dx;
        result(0) = i * dx + ghost_offset;
        if (sz.yc > 1) {
            result(1) = j * dx + ghost_offset;
        }
        if (sz.zc > 1) {
            result(2) = k * dx + ghost_offset;
        }
        return result;
    }
};

struct Fluxes {
    Fp4d Fx; /// x flux
    Fp4d Fy; /// y flux
    Fp4d Fz; /// z flux
};

struct Sources {
    Fp4d Sx; /// x source
    Fp4d Sy; /// y source
    Fp4d Sz; /// z source
};

#else
#endif
