#include "ProblemGenerator.hpp"
#include "../Hydro.hpp"
#include "../MosscapConfig.hpp"
#include "../SourceTerms/Sponge.hpp"
#include "../SourceTerms/TownsendThinLoss.hpp"

// NOTE(cmo): This is a 2d problem
static constexpr int num_dim = 2;

namespace Mosscap {

static constexpr fp_t T_0_d = 1.2e6_fp;
static constexpr fp_t P_0_d = 0.07_fp;
static constexpr fp_t T_blob_d = 10e3_fp;
static constexpr fp_t x0_d = 0.0_fp;
static constexpr fp_t y0_d = 10e6_fp;
static constexpr fp_t delta_d = 0.5e6_fp;
static constexpr fp_t width_d = 2e6_fp;
static constexpr fp_t tr_width_d = 2e5_fp;

static constexpr const char * blob_types[] = {
    "gaussian",
    "tanh"
};
enum class BlobType {
    Gaussian,
    Tanh
};

template <typename Fluid>
static void initial_conditions(Simulation& sim, const YAML::Node& config) {
    using Prim = typename Fluid::prim;
    constexpr int n_hydro = Fluid::num_vars;
    typedef yakl::Array<f64, 1, yakl::memHost> F64Host;
    const auto& state = sim.state;
    const auto& sz = state.sz;
    const auto& eos = sim.eos;

    static constexpr f64 h_mass = 1.6737830080950003e-27;
    static constexpr f64 k_B = 1.380649e-23;

    const std::string blob_type_s = get_or<std::string>(config, "problem.blob_type", "gaussian");
    BlobType blob_type = find_associated_enum<BlobType>(blob_types, sizeof(blob_types)/sizeof(blob_types[0]), blob_type_s);
    const fp_t T_0 = get_or<fp_t>(config, "problem.base_temperature", T_0_d);
    const fp_t P_0 = get_or<fp_t>(config, "problem.base_pressure", P_0_d);
    const fp_t T_blob = get_or<fp_t>(config, "problem.blob_temperature", T_blob_d);

    const fp_t x0 = get_or<fp_t>(config, "problem.blob_x0", x0_d);
    const fp_t y0 = get_or<fp_t>(config, "problem.blob_y0", y0_d);
    const fp_t blob_width_x = get_or<fp_t>(config, "problem.blob_width_x", width_d);
    const fp_t blob_width_y = get_or<fp_t>(config, "problem.blob_width_y", width_d);
    const fp_t tr_width = get_or<fp_t>(config, "problem.tr_width", tr_width_d);
    const fp_t delta = get_or<fp_t>(config, "problem.delta", delta_d);

    const fp_t bx0 = get_or<fp_t>(config, "problem.bx0", 0.0);
    const fp_t by0 = get_or<fp_t>(config, "problem.by0", 0.0);
    const fp_t bz0 = get_or<fp_t>(config, "problem.bz0", 0.0);

    // Coronal background density = P_0 / (2 * k_B T_0) * h_mass -- fully ionised
    const fp_t rho_0 = P_0 / (2.0_fp * k_B * T_0) * h_mass;
    fmt::println("Base coronal density {:.2e} kg/m3", rho_0);
    const fp_t mean_mass = 1.0_fp;

    dex_parallel_for(
        FlatLoop<3>(sz.zc, sz.yc, sz.xc),
        KOKKOS_LAMBDA (int k, int j, int i) {
            yakl::SArray<fp_t, 1, n_hydro> w(0.0_fp);
            w(I(Prim::Vx)) = 0.0_fp;
            w(I(Prim::Vy)) = 0.0_fp;
            w(I(Prim::Rho)) = rho_0;
            w(I(Prim::Pres)) = P_0;

            vec3 p = state.get_pos(i, j, k);

            // NOTE(cmo): 1.0 at peak blob density, 0.0 for corona
            fp_t blob_scale = 0.0_fp;
            if (blob_type == BlobType::Gaussian) {
                blob_scale = std::exp(-(square(p(0) - x0) + square(p(1) - y0)) / square(delta));
            } else if (blob_type == BlobType::Tanh) {
                const fp_t fn_x = 0.5 * (
                    std::tanh((p(0) - x0 + 0.5 * blob_width_x) / tr_width)
                    - std::tanh((p(0) - x0 - 0.5 * blob_width_x) / tr_width)
                );
                const fp_t fn_y = 0.5 * (
                    std::tanh((p(1) - y0 + 0.5 * blob_width_y) / tr_width)
                    - std::tanh((p(1) - y0 - 0.5 * blob_width_y) / tr_width)
                );
                blob_scale = fn_x * fn_y;
            }

            fp_t temp_full = T_0;
            if (blob_scale > 1e-8_fp) {
                temp_full = T_blob + (T_0 - T_blob) * (1.0 - blob_scale);
            }
            w(I(Prim::Rho)) = w(I(Prim::Pres)) / (2.0_fp * k_B * temp_full) * h_mass;

            JasUse(bx0, by0, bz0);
            if constexpr (Fluid::is_mhd) {
                w(I(Prim::Bx)) = bx0;
                w(I(Prim::By)) = by0;
                w(I(Prim::Bz)) = bz0;
            }

            CellIndex idx {
                .i = i,
                .j = j,
                .k = k
            };
            prim_to_cons<Fluid>(eos.gamma, state.mu0, w, QtyView(state.Q, idx));
        }
    );
}

template <typename FTraits>
static void setup_boundaries(Simulation& sim, const YAML::Node& config) {
    auto& bound = sim.state.boundaries;
    auto check_and_set_constant = [&](
            const BoundaryType boundary,
            const decltype(bound.xs_const)& arr,
            const std::string& bdry
    ) {
        if (boundary != BoundaryType::Constant) {
            return;
        }
        // NOTE(cmo): If it's not problem_supplied, then ignore
        const auto name = fmt::format("{}_const", bdry);
        if (config["boundary"][name].IsSequence()) {
            return;
        }

        const fp_t T_0 = get_or<fp_t>(config, "problem.base_temperature", T_0_d);
        const fp_t P_0 = get_or<fp_t>(config, "problem.base_pressure", P_0_d);

        const fp_t bx0 = get_or<fp_t>(config, "problem.bx0", 0.0);
        const fp_t by0 = get_or<fp_t>(config, "problem.by0", 0.0);
        const fp_t bz0 = get_or<fp_t>(config, "problem.bz0", 0.0);

        static constexpr f64 h_mass = 1.6737830080950003e-27;
        static constexpr f64 k_B = 1.380649e-23;
        const fp_t rho_0 = P_0 / (2.0_fp * k_B * T_0) * h_mass;

        yakl::SArray<fp_t, 1, FTraits::num_vars> w(0.0_fp);
        using Prim = FTraits::prim;
        w(I(Prim::Rho)) = rho_0;
        w(I(Prim::Pres)) = P_0;
        if (FTraits::is_mhd) {
            w(I(Prim::Bx)) = bx0;
            w(I(Prim::By)) = by0;
            w(I(Prim::Bz)) = bz0;
        }
        yakl::SArray<fp_t, 1, FTraits::num_vars> q(0.0_fp);
        prim_to_cons<FTraits>(sim.eos.gamma, sim.state.mu0, w, q);

        using Cons3 = Cons<3, FLUID_WITH_MAX_VARS>;
        using C = FTraits::cons;
        arr(I(C::Rho)) = q(I(Cons3::Rho));
        arr(I(C::MomX)) = q(I(Cons3::MomX));
        if constexpr (FTraits::is_mhd || FTraits::num_dim > 1) {
            arr(I(C::MomY)) = q(I(Cons3::MomY));
        }
        if constexpr (FTraits::is_mhd || FTraits::num_dim > 2) {
            arr(I(C::MomZ)) = q(I(Cons3::MomZ));
        }
        arr(I(C::Ene)) = q(I(Cons3::Ene));
        if constexpr (FTraits::is_mhd) {
            arr(I(C::Bx)) = q(I(Cons3::Bx));
            arr(I(C::By)) = q(I(Cons3::By));
            arr(I(C::Bz)) = q(I(Cons3::Bz));
            if constexpr (is_instance(FTraits::fluid_type, FluidType::GlmMhd)) {
                arr(I(C::Psi)) = q(I(Cons3::Psi));
            }
            if constexpr (FTraits::has_hypertc) {
                arr(I(C::HeatF)) = q(I(Cons3::HeatF));
            }
        }
    };
    check_and_set_constant(bound.xs, bound.xs_const, "xs");
    check_and_set_constant(bound.xe, bound.xe_const, "xe");
    check_and_set_constant(bound.ys, bound.ys_const, "ys");
    check_and_set_constant(bound.ye, bound.ye_const, "ye");
}

MOSSCAP_NEW_PROBLEM(pure_coronal_condensation) {
    MOSSCAP_PROBLEM_PREAMBLE(pure_coronal_condensation);
    if (sim.num_dim != num_dim) {
        throw std::runtime_error(fmt::format(
            "{} only handles {}d problems", PROBLEM_NAME, num_dim
        ));
    }

    FluidTraitsRt traits(sim.num_dim, sim.fluid_type);
    sim.setup_ics = [=](Simulation& sim) {
        if (sim.fluid_type == FluidType::Hydro) {
            initial_conditions<FluidTraits<num_dim, FluidType::Hydro>>(sim, config);
        } else if (traits.is_mhd) {
            initial_conditions<FluidTraits<num_dim, FluidType::Mhd>>(sim, config);
        } else {
            throw std::runtime_error("Unknown fluid type");
        }
    };

    invoke_fluid_traits(
        sim.num_dim,
        sim.fluid_type,
        [&]<typename FTraits>(FTraits) {
            setup_boundaries<FTraits>(sim, config);
        }
    );

    if (get_or<bool>(config, "problem.enable_sponge", false)) {
        setup_sponge(sim, config);
    }

    setup_thin_loss(sim, config);
}

}