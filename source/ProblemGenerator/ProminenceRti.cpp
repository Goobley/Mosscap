// NOTE(claude): Work in progress -- not yet tested.
#include "ProblemGenerator.hpp"
#include "../Hydro.hpp"
#include "../MosscapConfig.hpp"
#include "../SourceTerms/Gravity.hpp"
#include "../SourceTerms/Friction.hpp"

// NOTE(claude): This is a 2d problem
static constexpr int num_dim = 2;

namespace Mosscap {

// Released dense prominence column -> Rayleigh-Taylor plumes.
//
// A finite-width (column_width), finite-height (column_height) cool dense
// prominence column sits in a hot, hydrostatic corona with gravity down (-y).
// The gas pressure is the ambient coronal hydrostatic profile, uniform in x, so
// there is no horizontal pressure imbalance -- but inside the column the gas is
// ~100x denser (cool) at that same pressure, so it is negatively buoyant: it
// descends and its dense-over-light underside goes Rayleigh-Taylor unstable,
// shedding plumes (the in-plane, cross-section analogue of Hillier's plumes).
// Corona surrounds the column on all sides; boundaries are non-periodic (wall in
// x, constant corona in y).
//
// A weak uniform horizontal field bx0 sets the short-wavelength cutoff
//   lambda_c = 4 pi bx0^2 / (mu0 rho_dense g)
// so plumes form at scales >~ max(lambda_c, interface thickness). A uniform
// out-of-plane guide field bz_guide can be added (force-free, passive).

static constexpr fp_t T_cor_d = 1.0e6_fp;
static constexpr fp_t T_prom_d = 1.0e4_fp;
static constexpr fp_t rho_cor_d = 2.0e-11_fp;   // coronal density at the column height
static constexpr fp_t width_d = 1.5e6_fp;
static constexpr fp_t height_d = 1.2e6_fp;
static constexpr fp_t delta_d = 0.12e6_fp;      // envelope / interface thickness
static constexpr fp_t bx0_d = 1.0e-4_fp;        // weak horizontal field (sets lambda_c)
static constexpr f64 h_mass = 1.6737830080950003e-27;
static constexpr f64 k_B = 1.380649e-23;

struct RtiParams {
    fp_t T_cor, T_prom, rho_cor;
    fp_t x0, y0, width, height, delta;
    fp_t bx0, bz_guide, seed_amp, cs_dense;
    fp_t g_abs, mu0;
    u64 seed;
};

static RtiParams read_params(Simulation& sim, const YAML::Node& config) {
    const auto& state = sim.state;
    const auto& sz = state.sz;
    RtiParams P;
    P.T_cor = get_or<fp_t>(config, "problem.coronal_temperature", T_cor_d);
    P.T_prom = get_or<fp_t>(config, "problem.prominence_temperature", T_prom_d);
    P.rho_cor = get_or<fp_t>(config, "problem.coronal_density", rho_cor_d);
    P.width = get_or<fp_t>(config, "problem.column_width", width_d);
    P.height = get_or<fp_t>(config, "problem.column_height", height_d);
    P.delta = get_or<fp_t>(config, "problem.interface_thickness", delta_d);
    P.bx0 = get_or<fp_t>(config, "problem.bx0", bx0_d);
    P.bz_guide = get_or<fp_t>(config, "problem.bz_guide", 0.0_fp);
    P.seed_amp = get_or<fp_t>(config, "problem.seed_amplitude", 0.01_fp);
    P.seed = get_or<u64>(config, "problem.seed", 12345UL);
    P.g_abs = std::abs(get_or<fp_t>(config, "sources.gravity.y", -274.0_fp));
    P.mu0 = state.mu0;
    P.x0 = get_or<fp_t>(config, "problem.column_x0", 0.0_fp);
    const int ny = sz.yc - 2 * sz.ng;
    const fp_t y_height = ny * state.dx;
    P.y0 = get_or<fp_t>(config, "problem.column_y0", state.loc.y + 0.7_fp * y_height);
    return P;
}

// Coronal hydrostatic pressure at height y (isothermal T_cor), referenced to the
// column height y0 where rho = rho_cor.
KOKKOS_INLINE_FUNCTION fp_t coronal_pressure(const RtiParams& P, fp_t y) {
    const fp_t H_c = 2.0_fp * k_B * P.T_cor / (h_mass * P.g_abs);
    const fp_t P0 = 2.0_fp * P.rho_cor * k_B * P.T_cor / h_mass;
    return P0 * std::exp(-(y - P.y0) / H_c);
}

template <typename Prim, int n_hydro, bool is_mhd>
KOKKOS_INLINE_FUNCTION
void state_at(const RtiParams& P, fp_t X, fp_t Y, yakl::SArray<fp_t, 1, n_hydro>& w) {
    // Smooth top-hat column envelope (1 inside the column, 0 in the corona)
    const fp_t Ex = 0.5_fp * (std::tanh((X - P.x0 + 0.5_fp * P.width) / P.delta)
                            - std::tanh((X - P.x0 - 0.5_fp * P.width) / P.delta));
    const fp_t Ey = 0.5_fp * (std::tanh((Y - P.y0 + 0.5_fp * P.height) / P.delta)
                            - std::tanh((Y - P.y0 - 0.5_fp * P.height) / P.delta));
    const fp_t C = Ex * Ey;
    const fp_t T = P.T_cor + (P.T_prom - P.T_cor) * C;   // cool inside, hot outside
    const fp_t pres = coronal_pressure(P, Y);            // ambient, uniform in x
    const fp_t rho = pres * h_mass / (2.0_fp * k_B * T); // dense where cool

    w(I(Prim::Rho)) = rho;
    w(I(Prim::Pres)) = pres;
    if constexpr (is_mhd) {
        w(I(Prim::Bx)) = P.bx0;
        w(I(Prim::By)) = 0.0_fp;
        w(I(Prim::Bz)) = P.bz_guide;
    }
}

template <typename Fluid>
static void initial_conditions(Simulation& sim, const YAML::Node& config) {
    using Prim = typename Fluid::prim;
    constexpr int n_hydro = Fluid::num_vars;
    constexpr bool is_mhd = Fluid::is_mhd;
    const auto& state = sim.state;
    const auto& sz = state.sz;
    const auto& eos = sim.eos;

    RtiParams P = read_params(sim, config);
    const fp_t P0 = 2.0_fp * P.rho_cor * k_B * P.T_cor / h_mass;
    const fp_t rho_dense = P0 * h_mass / (2.0_fp * k_B * P.T_prom);
    P.cs_dense = std::sqrt(eos.gamma * P0 / rho_dense);
    const fp_t lambda_c = 4.0_fp * M_PI * P.bx0 * P.bx0 / (P.mu0 * rho_dense * P.g_abs);
    fmt::println(
        "Prominence RTI (released column): x0={:.2e} y0={:.2e} m, W={:.2e} H={:.2e} m",
        P.x0, P.y0, P.width, P.height
    );
    fmt::println(
        "                rho_dense={:.3e} rho_cor={:.3e} kg/m3, bx0={:.2e} T -> lambda_c={:.3e} m ({:.3f} Mm), c_s={:.2e} m/s",
        rho_dense, P.rho_cor, P.bx0, lambda_c, lambda_c / 1e6_fp, P.cs_dense
    );

    dex_parallel_for(
        FlatLoop<3>(sz.zc, sz.yc, sz.xc),
        KOKKOS_LAMBDA (int k, int j, int i) {
            yakl::SArray<fp_t, 1, n_hydro> w(0.0_fp);
            vec3 p = state.get_pos(i, j, k);
            state_at<Prim, n_hydro, is_mhd>(P, p(0), p(1), w);

            // Small broadband velocity seed inside the column to break symmetry
            const fp_t Ex = 0.5_fp * (std::tanh((p(0) - P.x0 + 0.5_fp * P.width) / P.delta)
                                    - std::tanh((p(0) - P.x0 - 0.5_fp * P.width) / P.delta));
            const fp_t Ey = 0.5_fp * (std::tanh((p(1) - P.y0 + 0.5_fp * P.height) / P.delta)
                                    - std::tanh((p(1) - P.y0 - 0.5_fp * P.height) / P.delta));
            const fp_t C = Ex * Ey;
            if (C > 1e-3_fp) {
                yakl::Random rng(P.seed + u64(k) * sz.yc * sz.xc + u64(j) * sz.xc + u64(i));
                w(I(Prim::Vy)) = P.seed_amp * P.cs_dense * C * (rng.genFP<fp_t>() - 0.5_fp) * 2.0_fp;
            }

            CellIndex idx { .i = i, .j = j, .k = k };
            prim_to_cons<Fluid>(eos.gamma, state.mu0, w, QtyView(state.Q, idx));
        }
    );
}

template <typename FTraits>
static void setup_boundaries(Simulation& sim, const YAML::Node& config) {
    // Constant y-faces: ambient corona (the column is well inside the domain, so
    // the faces are x-uniform corona). x-faces are wall (see the yaml).
    auto& bound = sim.state.boundaries;
    const auto& state = sim.state;
    const auto& sz = state.sz;
    const RtiParams P = read_params(sim, config);
    const fp_t y_lo = state.loc.y + (sz.ng - sz.ng + 0.5_fp) * state.dx;
    const int ny = sz.yc - 2 * sz.ng;
    const fp_t y_hi = state.loc.y + (ny - 0.5_fp) * state.dx;

    using Prim = typename FTraits::prim;
    constexpr bool is_mhd = FTraits::is_mhd;
    auto fill_face = [&](const decltype(bound.ys_const)& arr, fp_t y_face) {
        yakl::SArray<fp_t, 1, FTraits::num_vars> w(0.0_fp);
        // Far from the column x=x0 +- large: envelope ~0 -> pure corona. Evaluate
        // well outside the column in x to be safe.
        state_at<Prim, FTraits::num_vars, is_mhd>(P, P.x0 + 10.0_fp * P.width, y_face, w);
        yakl::SArray<fp_t, 1, FTraits::num_vars> q(0.0_fp);
        prim_to_cons<FTraits>(sim.eos.gamma, sim.state.mu0, w, q);

        using Cons3 = Cons<3, FLUID_WITH_MAX_VARS>;
        using C = typename FTraits::cons;
        arr(I(C::Rho)) = q(I(Cons3::Rho));
        arr(I(C::MomX)) = q(I(Cons3::MomX));
        if constexpr (FTraits::is_mhd || FTraits::num_dim > 1) arr(I(C::MomY)) = q(I(Cons3::MomY));
        if constexpr (FTraits::is_mhd || FTraits::num_dim > 2) arr(I(C::MomZ)) = q(I(Cons3::MomZ));
        arr(I(C::Ene)) = q(I(Cons3::Ene));
        arr(I(C::IonE)) = q(I(Cons3::IonE));
        if constexpr (FTraits::is_mhd) {
            arr(I(C::Bx)) = q(I(Cons3::Bx));
            arr(I(C::By)) = q(I(Cons3::By));
            arr(I(C::Bz)) = q(I(Cons3::Bz));
            if constexpr (is_instance(FTraits::fluid_type, FluidType::GlmMhd)) arr(I(C::Psi)) = q(I(Cons3::Psi));
            if constexpr (FTraits::has_hypertc) arr(I(C::HeatF)) = q(I(Cons3::HeatF));
        }
    };
    fill_face(bound.ys_const, y_lo);
    fill_face(bound.ye_const, y_hi);
}

MOSSCAP_NEW_PROBLEM(prominence_rti) {
    MOSSCAP_PROBLEM_PREAMBLE(prominence_rti);
    if (sim.num_dim != num_dim) {
        throw std::runtime_error(fmt::format("{} only handles {}d problems", PROBLEM_NAME, num_dim));
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
        sim.num_dim, sim.fluid_type,
        [&]<typename FTraits>(FTraits) { setup_boundaries<FTraits>(sim, config); }
    );

    setup_gravity(sim, config);
    if (get_or<bool>(config, "sources.friction.enable", false)) {
        setup_friction(sim, config);
    }
}

}
