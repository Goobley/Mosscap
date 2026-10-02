// NOTE(claude): Work in progress -- not yet tested.
#include "ProblemGenerator.hpp"
#include "../Hydro.hpp"
#include "../MosscapConfig.hpp"
#include "../SourceTerms/Gravity.hpp"
#include "../SourceTerms/Friction.hpp"

// NOTE(cmo): This is a 2d problem
static constexpr int num_dim = 2;

namespace Mosscap {

// Vertically-localised Kippenhahn-Schluter prominence: a cool, dense prominence
// body cradled in a magnetic dip, embedded in a hot hydrostatic corona (above,
// below and to either side). This is a principled localisation of the KS model
// (Hillier et al. 2012, ApJ 746 120): in the loaded region it reduces to KS,
// away from it the field straightens into a horizontal coronal field.
//
// The in-plane field comes from a flux function A(x,y) = A_z, so B = curl(A z),
// which is divergence-free by construction. A localised flux concentration on top
// of a uniform horizontal field sags the field lines into a genuine magnetic DIP
// (a bowl) at (0, y0) -- so descending mass stretches the field and feels a
// restoring tension (a stable trap), unlike an envelope that merely weakens the
// support below. G is a 2D Gaussian centred on the prominence:
//   G(x,y) = exp(-(x/wx)^2/2 - ((y-y0)/wy)^2/2)
//   A(x,y) = bx0*y + Phi*G,   Phi = by_inf*wx
//   Bx =  dA/dy = bx0 - Phi*(y-y0)/wy^2 * G
//   By = -dA/dx = Phi*x/wx^2   * G
// Far from the prominence G->0, so Bx->bx0 and By->0 (uniform, current-free
// corona) -- essential for a quiet relaxation. Mass is loaded into the dip and
// the friction source term settles it onto the numerical equilibrium.
//
// Plasma: isothermal hot corona in hydrostatic equilibrium in y, plus a
// prominence loaded into the dip. The density enhancement rho0*sech^2(x/w)*h(y)
// is what the local dip tension can support (rho0 = bx0*by_inf/(mu0*w*|g|)); the
// gas-pressure bump keeps horizontal (gas + magnetic) pressure balance. The
// prominence then emerges cool and dense (T = P/rho) at ~coronal pressure.
//
// This initial state is only approximately static; run it with the friction
// source term (sources.friction.enable) to relax onto the numerical equilibrium,
// then restart with friction off to confirm it holds.

static constexpr fp_t T_cor_d = 1.0e6_fp;
static constexpr fp_t rho_cor0_d = 2.0e-11_fp;
static constexpr fp_t bx0_d = 5.0e-4_fp;
static constexpr fp_t by_inf_d = 1.0e-3_fp;
static constexpr fp_t w_d = 0.3e6_fp;
static constexpr fp_t prom_height_d = 1.0e6_fp;
static constexpr f64 h_mass = 1.6737830080950003e-27;
static constexpr f64 k_B = 1.380649e-23;

struct KsParams {
    fp_t T_cor;
    fp_t rho_cor0;   // coronal density at reference height y0
    fp_t bx0;
    fp_t by_inf;
    fp_t w;          // sheet width (x)
    fp_t y0;         // prominence centre height
    fp_t prom_height; // vertical half-width of the prominence/dip envelope
    fp_t g_abs;      // |gravity_y|
    fp_t mu0;
    fp_t P_cor0;     // coronal pressure at y0
    fp_t H_c;        // coronal pressure scale height
    fp_t rho0;       // marginal support density at the dip centre
    fp_t mass_load;  // fraction of marginal support actually loaded (<1 is safe)
};

static KsParams read_ks_params(Simulation& sim, const YAML::Node& config) {
    const auto& state = sim.state;
    const auto& sz = state.sz;

    KsParams P;
    P.T_cor = get_or<fp_t>(config, "problem.coronal_temperature", T_cor_d);
    P.rho_cor0 = get_or<fp_t>(config, "problem.coronal_density", rho_cor0_d);
    P.bx0 = get_or<fp_t>(config, "problem.bx0", bx0_d);
    P.by_inf = get_or<fp_t>(config, "problem.by_inf", by_inf_d);
    P.w = get_or<fp_t>(config, "problem.dip_width", w_d);
    P.prom_height = get_or<fp_t>(config, "problem.prom_height", prom_height_d);
    const fp_t g = get_or<fp_t>(config, "sources.gravity.y", -274.0_fp);
    P.g_abs = std::abs(g);
    P.mu0 = state.mu0;

    // Default the prominence height to the domain centre if not given.
    const fp_t y_start = get_or<fp_t>(config, "grid.y_start", 0.0_fp);
    const int ny = sz.yc - 2 * sz.ng;
    const fp_t y_height = ny * state.dx;
    P.y0 = get_or<fp_t>(config, "problem.y0", y_start + 0.5_fp * y_height);

    P.mass_load = get_or<fp_t>(config, "problem.mass_load", 0.8_fp);
    P.P_cor0 = 2.0_fp * P.rho_cor0 * k_B * P.T_cor / h_mass;
    P.H_c = 2.0_fp * k_B * P.T_cor / (h_mass * P.g_abs);
    P.rho0 = P.bx0 * P.by_inf / (P.mu0 * P.w * P.g_abs);
    return P;
}

// KOKKOS-callable evaluation of the analytic primitive state at a position.
template <typename Prim, int n_hydro, bool is_mhd>
KOKKOS_INLINE_FUNCTION
void ks_state_at(const KsParams& P, fp_t X, fp_t Y, yakl::SArray<fp_t, 1, n_hydro>& w) {
    const fp_t wx = P.w;
    const fp_t wy = P.prom_height;
    const fp_t xn = X / wx;
    const fp_t yn = (Y - P.y0) / wy;
    // 2D Gaussian flux concentration and the resulting dip field
    const fp_t G = std::exp(-0.5_fp * (xn * xn + yn * yn));
    const fp_t Phi = P.by_inf * wx;
    const fp_t bx = P.bx0 - Phi * (Y - P.y0) / (wy * wy) * G;
    const fp_t by = Phi * X / (wx * wx) * G;

    // Coronal hydrostatic background (isothermal)
    const fp_t P_c = P.P_cor0 * std::exp(-(Y - P.y0) / P.H_c);
    const fp_t rho_c = P_c * h_mass / (2.0_fp * k_B * P.T_cor);

    // Horizontal (gas + magnetic) pressure balance: matches coronal P_c far from
    // the prominence where By->0, with a gas-pressure deficit where By is strong.
    fp_t pres = P_c - by * by / (2.0_fp * P.mu0);
    if (pres < 0.05_fp * P_c) {
        pres = 0.05_fp * P_c; // floor; relaxation reconciles the remainder
    }
    // Cool dense mass loaded into the dip (fraction mass_load of marginal support)
    const fp_t rho = rho_c + P.mass_load * P.rho0 * (G * G);

    w(I(Prim::Rho)) = rho;
    w(I(Prim::Pres)) = pres;
    w(I(Prim::Vx)) = 0.0_fp;
    w(I(Prim::Vy)) = 0.0_fp;
    if constexpr (is_mhd) {
        w(I(Prim::Bx)) = bx;
        w(I(Prim::By)) = by;
        w(I(Prim::Bz)) = 0.0_fp;
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

    const KsParams P = read_ks_params(sim, config);

    fmt::println(
        "Kippenhahn-Schluter (localised): y0={:.3e} m, H_c={:.3e} m, rho0(prom)={:.3e} kg/m3",
        P.y0, P.H_c, P.rho0
    );
    fmt::println(
        "                     P_cor0={:.3e} Pa, rho_cor0={:.3e} kg/m3, T_cor={:.3e} K",
        P.P_cor0, P.rho_cor0, P.T_cor
    );

    dex_parallel_for(
        FlatLoop<3>(sz.zc, sz.yc, sz.xc),
        KOKKOS_LAMBDA (int k, int j, int i) {
            yakl::SArray<fp_t, 1, n_hydro> w(0.0_fp);
            vec3 p = state.get_pos(i, j, k);
            ks_state_at<Prim, n_hydro, is_mhd>(P, p(0), p(1), w);

            CellIndex idx { .i = i, .j = j, .k = k };
            prim_to_cons<Fluid>(eos.gamma, state.mu0, w, QtyView(state.Q, idx));
        }
    );
}

template <typename FTraits>
static void setup_boundaries(Simulation& sim, const YAML::Node& config) {
    // Fill the top/bottom (y) constant boundaries with the coronal background at
    // the domain edges (the prominence is localised near y0, so the y-faces are
    // pure, x-uniform corona). The x-faces use zerograd (see the yaml).
    auto& bound = sim.state.boundaries;
    const auto& state = sim.state;
    const auto& sz = state.sz;
    const KsParams P = read_ks_params(sim, config);

    const fp_t y_start = get_or<fp_t>(config, "grid.y_start", 0.0_fp);
    const int ny = sz.yc - 2 * sz.ng;
    const fp_t y_lo = y_start;
    const fp_t y_hi = y_start + ny * state.dx;

    using Prim = typename FTraits::prim;
    constexpr bool is_mhd = FTraits::is_mhd;
    auto fill_face = [&](const decltype(bound.ys_const)& arr, fp_t y_face) {
        yakl::SArray<fp_t, 1, FTraits::num_vars> w(0.0_fp);
        // At the y-faces env~0: pure corona (Bx=bx0, By=0). Evaluate at x=0.
        ks_state_at<Prim, FTraits::num_vars, is_mhd>(P, 0.0_fp, y_face, w);
        yakl::SArray<fp_t, 1, FTraits::num_vars> q(0.0_fp);
        prim_to_cons<FTraits>(sim.eos.gamma, sim.state.mu0, w, q);

        using Cons3 = Cons<3, FLUID_WITH_MAX_VARS>;
        using C = typename FTraits::cons;
        arr(I(C::Rho)) = q(I(Cons3::Rho));
        arr(I(C::MomX)) = q(I(Cons3::MomX));
        if constexpr (FTraits::is_mhd || FTraits::num_dim > 1) {
            arr(I(C::MomY)) = q(I(Cons3::MomY));
        }
        if constexpr (FTraits::is_mhd || FTraits::num_dim > 2) {
            arr(I(C::MomZ)) = q(I(Cons3::MomZ));
        }
        arr(I(C::Ene)) = q(I(Cons3::Ene));
        arr(I(C::IonE)) = q(I(Cons3::IonE));
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
    fill_face(bound.ys_const, y_lo);
    fill_face(bound.ye_const, y_hi);
}

MOSSCAP_NEW_PROBLEM(kippenhahn_schluter) {
    MOSSCAP_PROBLEM_PREAMBLE(kippenhahn_schluter);
    if (sim.num_dim != num_dim) {
        throw std::runtime_error(fmt::format(
            "{} only handles {}d problems", PROBLEM_NAME, num_dim
        ));
    }

    FluidTraitsRt traits(sim.num_dim, sim.fluid_type);
    sim.setup_ics = [=](Simulation& sim) {
        if (traits.is_mhd) {
            initial_conditions<FluidTraits<num_dim, FluidType::Mhd>>(sim, config);
        } else {
            throw std::runtime_error("kippenhahn_schluter requires an MHD fluid type (needs a guide field).");
        }
    };

    invoke_fluid_traits(
        sim.num_dim,
        sim.fluid_type,
        [&]<typename FTraits>(FTraits) {
            setup_boundaries<FTraits>(sim, config);
        }
    );

    setup_gravity(sim, config);
    if (get_or<bool>(config, "sources.friction.enable", false)) {
        setup_friction(sim, config);
    }
}

}
