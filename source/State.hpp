#if !defined(MOSSCAP_STATE_HPP)
#define MOSSCAP_STATE_HPP

#include <functional>

#include "Types.hpp"

namespace Mosscap {

enum class FluidType {
    Hydro = 0,
    Mhd,
    MhdHyperTc,
    HyperTcOnly,
    GlmMhd,
    GlmMhdHyperTc
};
constexpr const char* FluidTypeName[] = {
    "hydro",
    "mhd",
    "mhdhypertc",
    "hypertconly",
    "glmmhd",
    "glmmhdhypertc"
};
constexpr int NumFluidType = sizeof(FluidTypeName) / sizeof(FluidTypeName[0]);

template <int NumDim, FluidType Fluid>
struct Prim {
    static constexpr i32 Rho = 0;
    static constexpr i32 Vx = 1;
    static constexpr i32 Vy = (Fluid != FluidType::Hydro || NumDim > 1) ? 2 : 1024;
    static constexpr i32 Vz = (Fluid != FluidType::Hydro || NumDim > 2) ? 3 : 1024;
    static constexpr i32 Pres = (Fluid == FluidType::Hydro) ? 1 + NumDim : 4;
    static constexpr i32 IonE = (Fluid == FluidType::Hydro) ? 2 + NumDim : 5; // NOTE(cmo): holds the specific ion e (per unit mass)
    static constexpr i32 Bx = (Fluid == FluidType::Hydro) ? 2048 : 6;
    static constexpr i32 By = (Fluid == FluidType::Hydro) ? 2048 : 7;
    static constexpr i32 Bz = (Fluid == FluidType::Hydro) ? 2048 : 8;
    static constexpr i32 Psi = (Fluid != FluidType::GlmMhd && Fluid != FluidType::GlmMhdHyperTc) ? 3072 : 9;
    static constexpr i32 HeatF = (Fluid == FluidType::MhdHyperTc || Fluid == FluidType::HyperTcOnly) ? 9 : (Fluid == FluidType::GlmMhdHyperTc) ? 10 : 4096;
};

template <int NumDim, FluidType Fluid>
struct Cons {
    static constexpr i32 Rho = 0;
    static constexpr i32 MomX = 1;
    static constexpr i32 MomY = (Fluid != FluidType::Hydro || NumDim > 1) ? 2 : 1024;
    static constexpr i32 MomZ = (Fluid != FluidType::Hydro || NumDim > 2) ? 3 : 1024;
    static constexpr i32 Ene = (Fluid == FluidType::Hydro) ? 1 + NumDim : 4;
    static constexpr i32 IonE = (Fluid == FluidType::Hydro) ? 2 + NumDim : 5;
    static constexpr i32 Bx = (Fluid == FluidType::Hydro) ? 2048 : 6;
    static constexpr i32 By = (Fluid == FluidType::Hydro) ? 2048 : 7;
    static constexpr i32 Bz = (Fluid == FluidType::Hydro) ? 2048 : 8;
    static constexpr i32 Psi = (Fluid != FluidType::GlmMhd && Fluid != FluidType::GlmMhdHyperTc) ? 3072 : 9;
    static constexpr i32 HeatF = (Fluid == FluidType::MhdHyperTc || Fluid == FluidType::HyperTcOnly) ? 9 : (Fluid == FluidType::GlmMhdHyperTc) ? 10 : 4096;
};

constexpr FluidType FLUID_WITH_MAX_VARS = FluidType::GlmMhdHyperTc;

constexpr int get_num_hydro_vars(int num_dim, FluidType fluid = FluidType::Hydro)  {
    int num_vars = 11;
    switch (fluid) {
        case FluidType::Hydro: {
            num_vars = 3 + num_dim;
        } break;
        case FluidType::Mhd: {
            num_vars = 9;
        } break;
        case FluidType::MhdHyperTc: {
            num_vars = 10;
        } break;
        case FluidType::HyperTcOnly: {
            num_vars = 10;
        } break;
        case FluidType::GlmMhd: {
            num_vars = 10;
        } break;
        case FluidType::GlmMhdHyperTc: {
            num_vars = 11;
        } break;
    }
    return num_vars;
}

constexpr bool is_instance(FluidType to_check, FluidType base) {
    if (to_check == base) {
        return true;
    } else if (base == FluidType::Mhd) {
        return to_check != FluidType::Hydro;
    } else if (base == FluidType::GlmMhd) {
        return to_check == FluidType::GlmMhd || to_check == FluidType::GlmMhdHyperTc;
    }

    return false;
}

template <int NumDim, FluidType Fluid>
constexpr int N_HYDRO_VARS = get_num_hydro_vars(NumDim, Fluid);

template <int NumDim, FluidType fluid>
struct FluidTraits {
    static constexpr bool is_mhd = (fluid != FluidType::Hydro);
    static constexpr bool has_hypertc = (fluid == FluidType::HyperTcOnly || fluid == FluidType::MhdHyperTc || fluid == FluidType::GlmMhdHyperTc);
    static constexpr int num_dim = NumDim;
    static constexpr int num_vars = N_HYDRO_VARS<NumDim, fluid>;
    static constexpr FluidType fluid_type = fluid;
    typedef Prim<NumDim, fluid> prim;
    typedef Cons<NumDim, fluid> cons;
};

template <
    typename Lambda,
    typename ...Args,
    std::enable_if_t<
        std::is_void_v<typename std::invoke_result_t<Lambda, FluidTraits<2, FluidType::Hydro>, Args...>>,
        int
    > = 0
>
auto invoke_fluid_traits(
    int num_dim,
    FluidType fluid_type,
    Lambda&& fn,
    Args&&... args)
-> void {
    auto invoke_dim = [...args = std::forward<Args>(args), fn, num_dim]<FluidType FType>(FluidType fluid_type) {
        switch (num_dim) {
            case 1: {
                fn(FluidTraits<1, FType>{}, std::forward<Args>(args)...);
            } break;
            case 2: {
                fn(FluidTraits<2, FType>{}, std::forward<Args>(args)...);
            } break;
            case 3: {
                fn(FluidTraits<3, FType>{}, std::forward<Args>(args)...);
            } break;
        }
    };

    switch (fluid_type) {
        case FluidType::Hydro: {
            invoke_dim.template operator()<FluidType::Hydro>(FluidType::Hydro);
        } break;
        case FluidType::Mhd: {
            invoke_dim.template operator()<FluidType::Mhd>(FluidType::Mhd);
        } break;
        case FluidType::MhdHyperTc: {
            invoke_dim.template operator()<FluidType::MhdHyperTc>(FluidType::MhdHyperTc);
        } break;
        case FluidType::HyperTcOnly: {
            invoke_dim.template operator()<FluidType::HyperTcOnly>(FluidType::HyperTcOnly);
        } break;
        case FluidType::GlmMhd: {
            invoke_dim.template operator()<FluidType::GlmMhd>(FluidType::GlmMhd);
        } break;
        case FluidType::GlmMhdHyperTc: {
            invoke_dim.template operator()<FluidType::GlmMhdHyperTc>(FluidType::GlmMhdHyperTc);
        } break;
    }
}

template <
    typename Lambda,
    typename ...Args,
    std::enable_if_t<
        !std::is_void_v<typename std::invoke_result_t<Lambda, FluidTraits<2, FluidType::Hydro>, Args...>>,
        int
    > = 0
>
auto invoke_fluid_traits(
    int num_dim,
    FluidType fluid_type,
    Lambda&& fn,
    Args&&... args)
-> decltype(fn(FluidTraits<2, FluidType::Hydro>{}, std::forward<Args>(args)...)) {
    typedef typename std::invoke_result_t<Lambda, FluidTraits<2, FluidType::Hydro>, Args...> Result;

    auto invoke_dim = [...args = std::forward<Args>(args), fn, num_dim]<FluidType FType>(FluidType fluid_type) {
        Result result;
        switch (num_dim) {
            case 1: {
                result = fn(FluidTraits<1, FType>{}, std::forward<Args>(args)...);
            } break;
            case 2: {
                result = fn(FluidTraits<2, FType>{}, std::forward<Args>(args)...);
            } break;
            case 3: {
                result = fn(FluidTraits<3, FType>{}, std::forward<Args>(args)...);
            } break;
        }
        return result;
    };

    Result result;
    switch (fluid_type) {
        case FluidType::Hydro: {
            result = invoke_dim.template operator()<FluidType::Hydro>(FluidType::Hydro);
        } break;
        case FluidType::Mhd: {
            result = invoke_dim.template operator()<FluidType::Mhd>(FluidType::Mhd);
        } break;
        case FluidType::MhdHyperTc: {
            result = invoke_dim.template operator()<FluidType::MhdHyperTc>(FluidType::MhdHyperTc);
        } break;
        case FluidType::HyperTcOnly: {
            result = invoke_dim.template operator()<FluidType::HyperTcOnly>(FluidType::HyperTcOnly);
        } break;
        case FluidType::GlmMhd: {
            result = invoke_dim.template operator()<FluidType::GlmMhd>(FluidType::GlmMhd);
        } break;
        case FluidType::GlmMhdHyperTc: {
            result = invoke_dim.template operator()<FluidType::GlmMhdHyperTc>(FluidType::GlmMhdHyperTc);
        } break;
    }
    return result;
}

template <
    typename ...FnArgs,
    typename Lambda
>
auto select_fluid_traits(
    int num_dim,
    FluidType fluid_type,
    Lambda&& fn)
-> std::function<typename std::invoke_result_t<Lambda, FluidTraits<2, FluidType::Hydro>, FnArgs...>(FnArgs...)> {
    typedef typename std::invoke_result_t<Lambda, FluidTraits<2, FluidType::Hydro>, FnArgs...> FnResult;
    typedef typename std::function<FnResult(FnArgs...)> Result;

    auto select_dim = [fn, num_dim]<FluidType FType>(FluidType fluid_type) {
        Result result;
        switch (num_dim) {
            case 1: {
                result = [fn](FnArgs... args) {
                    return fn(FluidTraits<1, FType>{}, std::forward<FnArgs>(args)...);
                };
            } break;
            case 2: {
                result = [fn](FnArgs... args) {
                    return fn(FluidTraits<2, FType>{}, std::forward<FnArgs>(args)...);
                };
            } break;
            case 3: {
                result = [fn](FnArgs... args) {
                    return fn(FluidTraits<3, FType>{}, std::forward<FnArgs>(args)...);
                };
            } break;
        }
        return result;
    };

    // NOTE(cmo): I think we can actually use a multilambda and std::apply to tidy this
    Result result;
    switch (fluid_type) {
        case FluidType::Hydro: {
            result = select_dim.template operator()<FluidType::Hydro>(FluidType::Hydro);
        } break;
        case FluidType::Mhd: {
            result = select_dim.template operator()<FluidType::Mhd>(FluidType::Mhd);
        } break;
        case FluidType::MhdHyperTc: {
            result = select_dim.template operator()<FluidType::MhdHyperTc>(FluidType::MhdHyperTc);
        } break;
        case FluidType::HyperTcOnly: {
            result = select_dim.template operator()<FluidType::HyperTcOnly>(FluidType::HyperTcOnly);
        } break;
        case FluidType::GlmMhd: {
            result = select_dim.template operator()<FluidType::GlmMhd>(FluidType::GlmMhd);
        } break;
        case FluidType::GlmMhdHyperTc: {
            result = select_dim.template operator()<FluidType::GlmMhdHyperTc>(FluidType::GlmMhdHyperTc);
        } break;
    }
    return result;
}

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
        return I(f::prim::Bx);
    } else if constexpr (Axis == 1) {
        return I(f::prim::By);
    } else if constexpr (Axis == 2) {
        return I(f::prim::Bz);
    }
}

struct PrimRt {
    i32 Rho;
    i32 Vx;
    i32 Vy;
    i32 Vz;
    i32 Pres;
    i32 Bx;
    i32 By;
    i32 Bz;
    i32 Psi;
    i32 HeatF;
};

struct ConsRt {
    i32 Rho;
    i32 MomX;
    i32 MomY;
    i32 MomZ;
    i32 Ene;
    i32 Bx;
    i32 By;
    i32 Bz;
    i32 Psi;
    i32 HeatF;
};

/// Runtime access to traits
struct FluidTraitsRt {
    bool is_mhd;
    bool has_hypertc;
    int num_dim;
    int num_vars;
    FluidType fluid_type;
    PrimRt prim;
    ConsRt cons;

    FluidTraitsRt () {};
    FluidTraitsRt(int num_dim_, FluidType type) :
        is_mhd(type != FluidType::Hydro),
        has_hypertc(type == FluidType::HyperTcOnly || type == FluidType::MhdHyperTc || type == FluidType::GlmMhdHyperTc),
        num_dim(num_dim_),
        num_vars(get_num_hydro_vars(num_dim_, type)),
        fluid_type(type)
    {
        invoke_fluid_traits(num_dim, fluid_type, [this]<typename FTraits>(FTraits) {
            using Prim = typename FTraits::prim;
            using Cons = typename FTraits::cons;

            prim.Rho = Prim::Rho;
            prim.Vx = Prim::Vx;
            prim.Vy = Prim::Vy;
            prim.Vz = Prim::Vz;
            prim.Pres = Prim::Pres;
            prim.Bx = Prim::Bx;
            prim.By = Prim::By;
            prim.Bz = Prim::Bz;
            prim.Psi = Prim::Psi;
            prim.HeatF = Prim::HeatF;

            cons.Rho = Cons::Rho;
            cons.MomX = Cons::MomX;
            cons.MomY = Cons::MomY;
            cons.MomZ = Cons::MomZ;
            cons.Ene = Cons::Ene;
            cons.Bx = Cons::Bx;
            cons.By = Cons::By;
            cons.Bz = Cons::Bz;
            cons.Psi = Cons::Psi;
            cons.HeatF = Cons::HeatF;
        });
    };
};

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
    yakl::SArray<fp_t, 1, N_HYDRO_VARS<3, FLUID_WITH_MAX_VARS>> xs_const;
    yakl::SArray<fp_t, 1, N_HYDRO_VARS<3, FLUID_WITH_MAX_VARS>> xe_const;
    yakl::SArray<fp_t, 1, N_HYDRO_VARS<3, FLUID_WITH_MAX_VARS>> ys_const;
    yakl::SArray<fp_t, 1, N_HYDRO_VARS<3, FLUID_WITH_MAX_VARS>> ye_const;
    yakl::SArray<fp_t, 1, N_HYDRO_VARS<3, FLUID_WITH_MAX_VARS>> zs_const;
    yakl::SArray<fp_t, 1, N_HYDRO_VARS<3, FLUID_WITH_MAX_VARS>> ze_const;
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

struct Conduction {
    fp_t hypertc_kappa = 8e-12; /// Hyperbolic conduction coefficient [W m-1 K-7/2]. Default value is Spitzer like (i.e. (8e-7 erg cm-1 s-1 K-7/2)).
    bool spitzer = false; /// Whether kappa scales with temperature**5/2
    bool limited = false; /// Whether to limit conduction to the saturation limit.
};

struct State {
    GridSize sz; /// Grid dimensions + number of ghosts
    fp_t glm_ch; /// Hyperbolic wave speed in GLM MHD
    fp_t mu0 = 4.0e-7_fp * 3.14159265358979312_fp; /// Value of mu0 used in model
    fp_t p_mass = 1.6737830080950003e-27_fp; /// Base particle mass, combined with eos mean mass [kg]
    Conduction cond;
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
