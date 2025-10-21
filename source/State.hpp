#if !defined(MOSSCAP_STATE_HPP)
#define MOSSCAP_STATE_HPP

#include <functional>

#include "Types.hpp"

namespace Mosscap {

enum class FluidType {
    Hydro = 0,
    Mhd,
};
constexpr const char* FluidTypeName[] = {
    "hydro",
    "mhd",
};
constexpr int NumFluidType = sizeof(FluidTypeName) / sizeof(FluidTypeName[0]);


template <int NumDim = 1, FluidType Fluid = FluidType::Hydro>
struct Prim {
    static constexpr i32 Rho = 0;
    static constexpr i32 Vx = 1;
    static constexpr i32 Vy = (Fluid != FluidType::Hydro || NumDim > 1) ? 2 : 1024;
    static constexpr i32 Vz = (Fluid != FluidType::Hydro || NumDim > 2) ? 3 : 1024;
    static constexpr i32 Pres = 1 + NumDim + 2 * (Fluid != FluidType::Hydro);
    static constexpr i32 Bx = (Fluid == FluidType::Hydro) ? 2048 : 5;
    static constexpr i32 By = (Fluid == FluidType::Hydro) ? 2048 : 6;
    static constexpr i32 Bz = (Fluid == FluidType::Hydro) ? 2048 : 7;
};

template <int NumDim = 1, FluidType Fluid = FluidType::Hydro>
struct Cons {
    static constexpr i32 Rho = 0;
    static constexpr i32 MomX = 1;
    static constexpr i32 MomY = (Fluid != FluidType::Hydro || NumDim > 1) ? 2 : 1024;
    static constexpr i32 MomZ = (Fluid != FluidType::Hydro || NumDim > 2) ? 3 : 1024;
    static constexpr i32 Ene = 1 + NumDim + 2 * (Fluid != FluidType::Hydro);
    static constexpr i32 Bx = (Fluid == FluidType::Hydro) ? 2048 : 5;
    static constexpr i32 By = (Fluid == FluidType::Hydro) ? 2048 : 6;
    static constexpr i32 Bz = (Fluid == FluidType::Hydro) ? 2048 : 7;
};

template <int NumDim, FluidType Fluid = FluidType::Hydro>
constexpr int N_HYDRO_VARS = (Fluid == FluidType::Hydro) ? 2 + NumDim : 8;

constexpr int get_num_hydro_vars(int num_dim, FluidType fluid = FluidType::Hydro)  {
    return (fluid == FluidType::Hydro) ? 2 + num_dim : 8;
}

template <int NumDim, FluidType fluid>
struct FluidTraits {
    static constexpr bool is_mhd = (fluid != FluidType::Hydro);
    static constexpr int num_dim = NumDim;
    static constexpr int num_vars = N_HYDRO_VARS<NumDim, fluid>;
    static constexpr FluidType fluid_type = fluid;
    typedef Prim<NumDim, fluid> prim;
    typedef Cons<NumDim, fluid> cons;
};

template <typename E>
constexpr int I(E e) {
    return static_cast<int>(e);
}

/// @tparam f FluidTraits
/// @tparam Axis int
template <int Axis, typename f>
constexpr int Velocity() {
    if constexpr (Axis == 0) {
        return I(f::prim::Vx);
    } else if constexpr (Axis == 1) {
        return I(f::prim::Vy);
    } else if constexpr (Axis == 2) {
        return I(f::prim::Vz);
    }
}

template <int Axis, typename f>
constexpr int Momentum() {
    if constexpr (Axis == 0) {
        return I(f::cons::MomX);
    } else if constexpr (Axis == 1) {
        return I(f::cons::MomY);
    } else if constexpr (Axis == 2) {
        return I(f::cons::MomZ);
    }
}

template <int Axis, typename f>
constexpr int MagneticField() {
    if constexpr (Axis == 0) {
        return I(f::cons::Bx);
    } else if constexpr (Axis == 1) {
        return I(f::cons::By);
    } else if constexpr (Axis == 2) {
        return I(f::cons::Bz);
    }
}

enum class BoundaryType : i32 {
    Wall = 0,
    Periodic,
    Symmetric,
    SymmetricOutflowDiode,
    ZeroGrad,
    Constant,
    UserFn
};
constexpr const char* BoundaryTypeName[] = {
    "wall",
    "periodic",
    "symmetric",
    "symmetricoutflowdiode",
    "zerograd",
    "constant",
    "userfn"
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
    yakl::SArray<fp_t, 1, N_HYDRO_VARS<3, FluidType::Mhd>> xs_const;
    yakl::SArray<fp_t, 1, N_HYDRO_VARS<3, FluidType::Mhd>> xe_const;
    yakl::SArray<fp_t, 1, N_HYDRO_VARS<3, FluidType::Mhd>> ys_const;
    yakl::SArray<fp_t, 1, N_HYDRO_VARS<3, FluidType::Mhd>> ye_const;
    yakl::SArray<fp_t, 1, N_HYDRO_VARS<3, FluidType::Mhd>> zs_const;
    yakl::SArray<fp_t, 1, N_HYDRO_VARS<3, FluidType::Mhd>> ze_const;
};


struct GridSize {
    i32 xc = 0; /// x-cells including ghosts
    i32 yc = 0; /// y-cells including ghosts
    i32 zc = 0; /// z-cells including ghosts
    i32 ng = 0; /// num ghost cells (same on both ends of all axes)
};

struct GridLoc {
    fp_t x;
    fp_t y;
    fp_t z;
};

struct State {
    GridSize sz; /// Grid dimensions + number of ghosts
    fp_t mu0 = 4.0e-7_fp * 3.14159265358979312_fp; /// Value of mu0 used in model
    fp_t dx; /// Spatial grid step (constant)
    GridLoc loc; /// Logical grid position (bottom left corner of cell 0, 0, 0)
    Boundaries boundaries; /// Boundary handling specifications
    Fp4d Q; // Conserved State
    Fp4d W; /// Primitive State
    i32 num_tracers;

    KOKKOS_INLINE_FUNCTION vec3 get_pos(int i, int j=0, int k=0) const {
        vec3 result;
        const fp_t ghost_offset = -(sz.ng - 0.5_fp) * dx;
        result(0) = i * dx + ghost_offset + loc.x;
        if (sz.yc > 1) {
            result(1) = j * dx + ghost_offset + loc.y;
        }
        if (sz.zc > 1) {
            result(2) = k * dx + ghost_offset + loc.z;
        }
        return result;
    }

    KOKKOS_INLINE_FUNCTION fp_t get_axis_length(int axis) const {
        if (axis == 0) {
            return (sz.xc - 2 * sz.ng) * dx;
        }
        if (axis == 1) {
            if (sz.yc == 1) {
                return 1.0_fp;
            }
            return (sz.yc - 2 * sz.ng) * dx;
        }

        if (sz.zc == 1) {
            return 1.0_fp;
        }
        return (sz.zc - 2 * sz.ng) * dx;
    }
};

struct Fluxes {
    Fp4d Fx; /// x flux
    Fp4d Fy; /// y flux
    Fp4d Fz; /// z flux
};

struct Sources {
    Fp4d S;
};

}

#else
#endif
