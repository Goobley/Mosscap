#if !defined(MOSSCAP_STATE_HPP)
#define MOSSCAP_STATE_HPP

#include <functional>

#include "Types.hpp"

constexpr fp_t Gamma = FP(1.4); // Ratio of specific heats -- monatomic
constexpr fp_t GammaM1 = Gamma - FP(1.0);

enum class BoundaryType : i32 {
    Wall,
    Periodic,
    Symmetric
};

struct Boundaries {
    BoundaryType xs;
    BoundaryType xe;
    BoundaryType ys;
    BoundaryType ye;
    BoundaryType zs;
    BoundaryType ze;
};

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

// enum class Prim : i32 {
//     Rho = 0,
//     Vx = 1,
//     Vy = 2,
//     Vz = (NUM_DIM > 2) ? 3 : -200,
//     Pres = 3 + int(NUM_DIM > 2)
// };

// enum class Cons : i32 {
//     Rho = 0,
//     MomX = 1,
//     MomY = 2,
//     MomZ = (NUM_DIM > 2) ? 3 : -200,
//     Ene = 3 + int(NUM_DIM > 2)
// };
template <int NumDim>
constexpr int N_HYDRO_VARS = 2 + NumDim;

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
