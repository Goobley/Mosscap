#if !defined(MOSSCAP_STATE_HPP)
#define MOSSCAP_STATE_HPP

#include <functional>

#include "Types.hpp"

constexpr i32 NUM_DIM = 2;
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

enum class Prim : i32 {
    Rho = 0,
    Vx = 1,
    Vy = 2,
    Vz = (NUM_DIM > 2) ? 3 : -200,
    Pres = 3 + int(NUM_DIM > 2)
};

enum class Cons : i32 {
    Rho = 0,
    MomX = 1,
    MomY = 2,
    MomZ = (NUM_DIM > 2) ? 3 : -200,
    Ene = 3 + int(NUM_DIM > 2)
};
constexpr int N_HYDRO_VARS = 2 + NUM_DIM;

template <typename E>
constexpr int I(E e) {
    return static_cast<int>(e);
}

template <int Axis>
constexpr int Velocity() {
    if constexpr (Axis == 0) {
        return I(Prim::Vx);
    } else if constexpr (Axis == 1) {
        return I(Prim::Vy);
    } else if constexpr (Axis == 2) {
        return I(Prim::Vz);
    }
}

template <int Axis>
constexpr int Momentum() {
    if constexpr (Axis == 0) {
        return I(Cons::MomX);
    } else if constexpr (Axis == 1) {
        return I(Cons::MomY);
    } else if constexpr (Axis == 2) {
        return I(Cons::MomZ);
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
    Fp4d Q_old; // TODO(cmo): Have a vector for different schemes - or move burden to time-integrator
};

#else
#endif
