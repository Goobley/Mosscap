#include "ProblemGenerator.hpp"
#include "../Hydro.hpp"
#include "../MosscapConfig.hpp"
#include "../SourceTerms.hpp"
#include "../SourceTerms/Gravity.hpp"
#include "../SourceTerms/TownsendThinLoss.hpp"
#include "../SourceTerms/ReplaceSmallValues.hpp"
#include "../SourceTerms/ThermalConduction.hpp"
#include "../SourceTerms/Friction.hpp"
#include "../AnalyticLteH.hpp"

// NOTE(claude): This is a 2d problem
static constexpr int num_dim = 2;

namespace Mosscap {

// A finite prominence body starting in magnetohydrostatic and thermal equilibrium (2.5D, fully
// ionised, x along the field, y vertical). The state is computed offline by
// ProminenceBody/mhs/gs_equilibrium.py (Grad-Shafranov with gravity, each field line in energy
// balance) and written with ghosts by ProminenceBody/mhs/write_ic.py; this problem only loads it,
// applies the matching heating law, and holds the x walls at it. Use periodic y boundaries.
// Gravity is optional (sources.gravity): without it, write the IC with gs_equilibrium.py --gravity 0
// (straight field, isobaric lines).
//
// EOS: ideal, or analyticlteh (LTE H). Under analyticlteh T comes from the EOS's y_space, the coronal
// y in the heating law is 1, and the IC must carry the ionisation state the equilibrium was built with
// (write_ic.py --ion-energy writes rho y chi_H / m_H into Ene); ProminenceBody/mhs/lte/ iterates the
// equilibrium against the EOS's y.
// Or TracerEos (Dex): the IC (e.g. an analyticlteh hold) may lack the Dex tracers, which setup_dex fills; y_space
// starts from the IC's IonE (y = IonE m_H / chi_H), so the LTE tracers and Dex's first solve see its ionisation.
//
// Heating, by problem.heating:
//   tanh (default), a function of the local state only, so it moves with the plasma:
//     H = y n_H(p, t_corona)^2 Λ(t_corona) tanh((T - t_core) / t_core)   (T > t_core, else 0)
//   with n_H(p, T) = p / ((A + y) k_B T). Isothermal lines at t_corona balance exactly at any pressure.
//   Set sources.thin_loss.min_temperature = t_core.
//   constant: H0 = y n_H(heating_pressure, t_corona)^2 Λ(t_corona) everywhere, balancing the corona at
//   problem.heating_pressure [Pa]. No cutoff: the cool core sits where the losses meet H0 (set
//   sources.thin_loss.min_temperature to the EOS floor); t_core is unused.
// Under conduction.trac the heating is scaled with the thin losses.
//
// x walls (boundary.xs/xe: constant), by problem.x_boundary:
//   closed: line-tied and reflecting. rho mirrored and T held at the loaded state (a heat bath, no
//     mass crosses the faces), normal B held, tangential B held plus the edge cell's deviation from
//     the loaded state, psi = 0, v reflected (about the wall velocity, with a driver).
//   held: as closed, but rho and p held at the loaded state.
//   open: characteristic, after Lare3d's openboundary.f90 (Warwick-Plasma/Lare3d). The loaded ghost
//     state is the far field. Each characteristic leaving the box takes the edge cell's state, each
//     entering takes the far field, and the boundary state follows from the linear Riemann invariants
//     with the far-field impedance. Two limits, chosen by the far-field angle: field mostly normal
//     (|B_n| >= |B| / sqrt(2)): sound along B (+-c_s), Alfven (+-c_A carrying v_t, B_t) and entropy;
//     otherwise field taken as tangential: fast (+-c_f) and entropy. Lare picks the normal limit
//     only when B_t is exactly zero, so the angle test is ours. All ghost layers take the boundary
//     state, with v set from the same invariants (Lare evolves its boundary velocities separately).
//     psi = 0.
//   open_tbath: open, but T held at the far field (rho from the boundary pressure), keeping the heat bath.
//   open_char: full characteristic, for any field angle (e.g. with a guide field along the spine). The
//     edge cell's deviation from the far field is split into all 7 waves along the normal (Roe & Balsara
//     primitive eigenvectors at the far state); outgoing ones are kept, incoming ones and the entropy wave
//     dropped (T held at the far field, the heat bath). B_n as in open.
//   open_transverse: closed for rho, p and v_n (heat bath, no mass flux, sound reflected), open for the
//     Alfven characteristics carrying v_t and B_t, with B_n held.
// pulse adds a velocity perturbation to the loaded state at the start: a Gaussian (boundary tests), or a
// standing-mode sine along x (slow modes of the thread).
// diag_interval prints max |v|, min T, the largest change in T from the loaded state, the mass
// below 1e5 K and the kinetic energy in the box.
//
// Requires an MHD fluid, the ideal or analyticlteh EOS, sources.thin_loss and sources.thermal_conduction.
//
// problem:
//   name: prominence_mhs_2d
//   ic_path: mhs_20km.nc          # required, Q including ghosts on this grid
//   t_core: 1e4                   # [K]
//   t_corona: 1e6                 # [K]
//   heating: tanh                 # tanh | constant
//   heating_pressure: 0.03        # [Pa] constant: the pressure at which H0 balances the corona
//   diag_interval: 10.0           # [s], <= 0 disables
//   x_boundary: closed            # closed | held | open | open_tbath | open_transverse | open_char
//   ic_closure: none              # none | fixed_pt: hold each loaded cell's p and T and let the EOS closure set y,
//                                 #   hence rho, IonE and Ene (analyticlteh with ionisation energy; the LTE stand-in
//                                 #   for a static Dex solve). The loaded y is read from IonE. Also applied to the
//                                 #   reference state the walls hold.
//   pulse:
//     velocity: [0.0, 0.0, 0.0]   # [m s-1], zero disables
//     x0: 0.0                     # [m]
//     y0: 12.5e6                  # [m]
//     wx: 3e5                     # [m]
//     wy: 3e5                     # [m]
//     cold_temperature: 0.0       # [K], > 0 also weights by 0.5 (1 - tanh(ln(T / this) / 0.2)), a kick on the cold body
//     profile: gaussian           # gaussian | sine. sine: x profile sin(n pi (x - x0) / L) on |x - x0| <= L / 2 (wx unused)
//     length: 2e6                 # [m], L for sine
//     harmonic: 1                 # n for sine; odd n compresses the thread symmetrically about x0
//   driver:                       # closed walls only: the wall moves, v_wall(t) = velocity sin(2 pi (t - t_start) / period)
//     velocity: [0.0, 0.0, 0.0]   # [m s-1], zero disables. x is a piston (slow waves), y a moving footpoint (transverse)
//     period: 180.0               # [s]
//     cycles: 1.0                 # number of periods, then the wall stops; <= 0 drives forever
//     t_start: <IC time>          # [s]
//     side: xs                    # xs | xe | both (xe mirrors the x component, so both walls push inward together)
//     y0: 12.5e6                  # [m], centre of an optional Gaussian window in y
//     wy: 0.0                     # [m], <= 0 drives the whole wall

/// The coefficient mhs_temperature takes: m / ((A + y) k_B) for the ideal EOS, so that T = p c / rho; otherwise
/// the mass per H, m = mass_per_h p_mass (simulation.p_mass, as the thin loss, conduction and AnalyticLteH use).
static fp_t ideal_temperature_coeff(const Simulation& sim) {
    const auto& eos = sim.eos;
    const fp_t m_per_h = eos.mass_per_h * sim.state.p_mass;
    if (!eos.is_constant) {
        return m_per_h;
    }
    return m_per_h / ((eos.total_abund + eos.y) * ConstantsF64::k_B);
}

/// Temperature of a cell [K]: p c / rho for the ideal EOS, otherwise with the EOS's ionisation y_space (an LTE
/// closure now, Dex later) and n_H = rho / c.
KOKKOS_INLINE_FUNCTION fp_t mhs_temperature(const Eos& eos, fp_t c, fp_t pres, fp_t rho, int k, int j, int i) {
    if (eos.is_constant) {
        return pres * c / rho;
    }
    return temperature_si(pres, rho / c, eos.total_abund, eos.y_space(k, j, i));
}

enum class PulseProfile { Gaussian, Sine };

enum class XBoundary { Closed, Held, Open, OpenTBath, OpenTransverse, OpenChar };

struct ProminenceMhsParams {
    std::string ic_path;
    fp_t t_core;
    fp_t t_corona;
    fp_t diag_interval;
    XBoundary x_boundary;
    vec3 pulse_velocity;
    fp_t pulse_x0;
    fp_t pulse_y0;
    fp_t pulse_wx;
    fp_t pulse_wy;
    fp_t pulse_cold_temperature;
    PulseProfile pulse_profile;
    fp_t pulse_length;
    int pulse_harmonic;
    vec3 driver_velocity;
    fp_t driver_period;
    fp_t driver_cycles;
    fp_t driver_t_start; // NOTE(claude): < 0 until set, then the IC time if not given.
    bool driver_xs;
    bool driver_xe;
    fp_t driver_y0;
    fp_t driver_wy;
    bool ic_closure_fixed_pt;
    bool heating_constant;
    fp_t heating_pressure;
};

static bool read_vec3(const YAML::Node& config, const char* key, const char* name, vec3& out) {
    for (int d = 0; d < 3; ++d) {
        out(d) = 0.0_fp;
    }
    if (!(config["problem"][key] && config["problem"][key]["velocity"])) {
        return false;
    }
    const auto node = config["problem"][key]["velocity"];
    if (!node.IsSequence() || node.size() != 3) {
        throw std::runtime_error(fmt::format("prominence_mhs_2d: problem.{}.velocity must be [vx, vy, vz].", name));
    }
    for (int d = 0; d < 3; ++d) {
        out(d) = node[d].as<fp_t>();
    }
    return true;
}

static ProminenceMhsParams load_params(const YAML::Node& config) {
    ProminenceMhsParams p;
    p.ic_path = get_or<std::string>(config, "problem.ic_path", "");
    if (p.ic_path.empty()) {
        throw std::runtime_error("prominence_mhs_2d: problem.ic_path is required.");
    }
    p.t_core = get_or<fp_t>(config, "problem.t_core", 1e4_fp);
    p.t_corona = get_or<fp_t>(config, "problem.t_corona", 1e6_fp);
    p.diag_interval = get_or<fp_t>(config, "problem.diag_interval", 10.0_fp);
    const std::string closure = get_or<std::string>(config, "problem.ic_closure", "none");
    if (closure != "none" && closure != "fixed_pt") {
        throw std::runtime_error(fmt::format("prominence_mhs_2d: problem.ic_closure \"{}\" is not none | fixed_pt.", closure));
    }
    p.ic_closure_fixed_pt = closure == "fixed_pt";
    const std::string heating = get_or<std::string>(config, "problem.heating", "tanh");
    if (heating != "tanh" && heating != "constant") {
        throw std::runtime_error(fmt::format("prominence_mhs_2d: problem.heating \"{}\" is not tanh | constant.", heating));
    }
    p.heating_constant = heating == "constant";
    p.heating_pressure = get_or<fp_t>(config, "problem.heating_pressure", -1.0_fp);
    if (p.heating_constant && p.heating_pressure <= 0.0_fp) {
        throw std::runtime_error("prominence_mhs_2d: problem.heating constant needs problem.heating_pressure [Pa].");
    }
    const std::string xb = get_or<std::string>(config, "problem.x_boundary", "closed");
    if (xb == "closed") {
        p.x_boundary = XBoundary::Closed;
    } else if (xb == "held") {
        p.x_boundary = XBoundary::Held;
    } else if (xb == "open") {
        p.x_boundary = XBoundary::Open;
    } else if (xb == "open_tbath") {
        p.x_boundary = XBoundary::OpenTBath;
    } else if (xb == "open_transverse") {
        p.x_boundary = XBoundary::OpenTransverse;
    } else if (xb == "open_char") {
        p.x_boundary = XBoundary::OpenChar;
    } else {
        throw std::runtime_error("prominence_mhs_2d: problem.x_boundary must be closed, held, open, open_tbath, open_transverse or open_char.");
    }
    read_vec3(config, "pulse", "pulse", p.pulse_velocity);
    p.pulse_x0 = get_or<fp_t>(config, "problem.pulse.x0", 0.0_fp);
    p.pulse_y0 = get_or<fp_t>(config, "problem.pulse.y0", 12.5e6_fp);
    p.pulse_wx = get_or<fp_t>(config, "problem.pulse.wx", 3e5_fp);
    p.pulse_wy = get_or<fp_t>(config, "problem.pulse.wy", 3e5_fp);
    p.pulse_cold_temperature = get_or<fp_t>(config, "problem.pulse.cold_temperature", 0.0_fp);
    if (p.pulse_wx <= 0.0_fp || p.pulse_wy <= 0.0_fp) {
        throw std::runtime_error("prominence_mhs_2d: problem.pulse.{wx, wy} must be positive.");
    }
    const std::string profile = get_or<std::string>(config, "problem.pulse.profile", "gaussian");
    if (profile == "gaussian") {
        p.pulse_profile = PulseProfile::Gaussian;
    } else if (profile == "sine") {
        p.pulse_profile = PulseProfile::Sine;
    } else {
        throw std::runtime_error("prominence_mhs_2d: problem.pulse.profile must be gaussian or sine.");
    }
    p.pulse_length = get_or<fp_t>(config, "problem.pulse.length", 2e6_fp);
    p.pulse_harmonic = get_or<int>(config, "problem.pulse.harmonic", 1);
    if (p.pulse_length <= 0.0_fp || p.pulse_harmonic < 1) {
        throw std::runtime_error("prominence_mhs_2d: problem.pulse.length must be positive and problem.pulse.harmonic >= 1.");
    }
    if (read_vec3(config, "driver", "driver", p.driver_velocity) && p.x_boundary != XBoundary::Closed) {
        throw std::runtime_error("prominence_mhs_2d: problem.driver needs x_boundary: closed.");
    }
    p.driver_period = get_or<fp_t>(config, "problem.driver.period", 180.0_fp);
    p.driver_cycles = get_or<fp_t>(config, "problem.driver.cycles", 1.0_fp);
    p.driver_t_start = get_or<fp_t>(config, "problem.driver.t_start", -1.0_fp);
    const std::string side = get_or<std::string>(config, "problem.driver.side", "xs");
    if (side != "xs" && side != "xe" && side != "both") {
        throw std::runtime_error("prominence_mhs_2d: problem.driver.side must be xs, xe or both.");
    }
    p.driver_xs = side != "xe";
    p.driver_xe = side != "xs";
    p.driver_y0 = get_or<fp_t>(config, "problem.driver.y0", 12.5e6_fp);
    p.driver_wy = get_or<fp_t>(config, "problem.driver.wy", 0.0_fp);
    if (p.driver_period <= 0.0_fp) {
        throw std::runtime_error("prominence_mhs_2d: problem.driver.period must be positive.");
    }
    return p;
}

/// y_space = IonE m_H / chi_H (clamped to [0, 1]) everywhere, for TracerEos: it starts at y = 1, and the LTE tracers
/// and Dex's first solve take the ionisation from it.
template <typename FTraits>
static void set_y_from_ion_e(const Simulation& sim) {
    using Cons = typename FTraits::cons;
    const auto& sz = sim.state.sz;
    const auto& Q = sim.state.Q;
    const auto& y_space = sim.eos.y_space;
    const fp_t m_per_h = sim.eos.mass_per_h * sim.state.p_mass;
    constexpr fp_t chi_H = AnalyticLteH::chi_H;
    dex_parallel_for(
        "prominence_mhs_2d y from IonE",
        FlatLoop<3>(sz.zc, sz.yc, sz.xc),
        KOKKOS_LAMBDA (int k, int j, int i) {
            const fp_t y = Q(I(Cons::IonE), k, j, i) * m_per_h / chi_H;
            y_space(k, j, i) = std::min(std::max(y, 0.0_fp), 1.0_fp);
        }
    );
    Kokkos::fence();
}

/// The loaded state, kept as the boundary and diagnostic reference.
static Fp4d load_reference(const Simulation& sim, const std::string& path, f64& time) {
    const auto& sz = sim.state.sz;
    Fp4d Q0;
    yakl::SimpleNetCDF nc;
    nc.open(path, yakl::NETCDF_MODE_READ);
    nc.read(Q0, "Q");
    nc.read(time, "time");
    // NOTE(claude): Fewer variables than the state is allowed: the Dex tracers that follow the hydro ones are filled
    // by setup_dex.
    if (Q0.extent(0) > sim.state.Q.extent(0) || Q0.extent(2) != sz.yc || Q0.extent(3) != sz.xc) {
        throw std::runtime_error(fmt::format(
            "prominence_mhs_2d: {} has shape ({}, {}, {}), expected (<= {}, {}, {}) including ghosts.",
            path, Q0.extent(0), Q0.extent(2), Q0.extent(3), sim.state.Q.extent(0), sz.yc, sz.xc
        ));
    }
    return Q0;
}

template <typename FTraits>
static void heating_kernel(const Simulation& sim, fp_t h_coeff, fp_t t_core, fp_t h_const, const Fp4d& S) {
    using Prim = typename FTraits::prim;
    using Cons = typename FTraits::cons;
    constexpr int n_hydro = FTraits::num_vars;
    const auto& state = sim.state;
    const auto& sz = state.sz;
    const auto& eos = sim.eos;
    const fp_t mu0 = state.mu0;
    const fp_t c = ideal_temperature_coeff(sim);
    dex_parallel_for(
        "prominence_mhs_2d heating",
        FlatLoop<3>(sz.zc, sz.yc, sz.xc),
        KOKKOS_LAMBDA (int k, int j, int i) {
            if (h_const > 0.0_fp) {
                S(I(Cons::Ene), k, j, i) += h_const;
                return;
            }
            yakl::SArray<fp_t, 1, n_hydro> w;
            cons_to_prim<FTraits>(eos.gamma, mu0, QtyView(state.Q, CellIndex{.i = i, .j = j, .k = k}), w);
            const fp_t pres = w(I(Prim::Pres));
            const fp_t temp = mhs_temperature(eos, c, pres, w(I(Prim::Rho)), k, j, i);
            if (temp <= t_core) {
                return;
            }
            S(I(Cons::Ene), k, j, i) += std::tanh((temp - t_core) / t_core) * h_coeff * square(pres);
        }
    );
    Kokkos::fence();
}

template <typename FTraits>
static void print_diagnostics(const Simulation& sim, const Fp4d& Q0) {
    using Prim = typename FTraits::prim;
    constexpr int n_hydro = FTraits::num_vars;
    const auto& state = sim.state;
    const auto& sz = state.sz;
    const auto& eos = sim.eos;
    const fp_t mu0 = state.mu0;
    const fp_t dx = state.dx;
    const fp_t c = ideal_temperature_coeff(sim);
    const int ng = sz.ng;
    const int nx_inner = sz.xc - 2 * ng;

    auto prim = KOKKOS_LAMBDA (const Fp4d& Q, int j, int i, yakl::SArray<fp_t, 1, n_hydro>& w) {
        cons_to_prim<FTraits>(eos.gamma, mu0, QtyView(Q, CellIndex{.i = i, .j = j, .k = 0}), w);
    };
    fp_t v2_max = 0.0_fp;
    dex_parallel_reduce(
        "prominence_mhs_2d diag v",
        FlatLoop<2>(sz.yc - 2 * ng, nx_inner),
        KOKKOS_LAMBDA (int ji, int ii, fp_t& running) {
            yakl::SArray<fp_t, 1, n_hydro> w;
            prim(state.Q, ji + ng, ii + ng, w);
            running = std::max(running, square(w(I(Prim::Vx))) + square(w(I(Prim::Vy))) + square(w(I(Prim::Vz))));
        },
        Kokkos::Max<fp_t>(v2_max)
    );
    const fp_t v2_peak = v2_max;
    fp_t v_cell = 0.0_fp;
    dex_parallel_reduce(
        "prominence_mhs_2d diag v location",
        FlatLoop<2>(sz.yc - 2 * ng, nx_inner),
        KOKKOS_LAMBDA (int ji, int ii, fp_t& running) {
            yakl::SArray<fp_t, 1, n_hydro> w;
            prim(state.Q, ji + ng, ii + ng, w);
            if (square(w(I(Prim::Vx))) + square(w(I(Prim::Vy))) + square(w(I(Prim::Vz))) >= v2_peak) {
                running = std::max(running, fp_t(ji * nx_inner + ii));
            }
        },
        Kokkos::Max<fp_t>(v_cell)
    );
    const vec3 v_pos = state.get_pos(int(v_cell) % nx_inner + ng, int(v_cell) / nx_inner + ng, 0);
    fp_t t_min = 0.0_fp;
    dex_parallel_reduce(
        "prominence_mhs_2d diag T min",
        FlatLoop<2>(sz.yc - 2 * ng, nx_inner),
        KOKKOS_LAMBDA (int ji, int ii, fp_t& running) {
            yakl::SArray<fp_t, 1, n_hydro> w;
            prim(state.Q, ji + ng, ii + ng, w);
            running = std::min(running, mhs_temperature(eos, c, w(I(Prim::Pres)), w(I(Prim::Rho)), 0, ji + ng, ii + ng));
        },
        Kokkos::Min<fp_t>(t_min)
    );
    fp_t dt_max = 0.0_fp;
    dex_parallel_reduce(
        "prominence_mhs_2d diag dT",
        FlatLoop<2>(sz.yc - 2 * ng, nx_inner),
        KOKKOS_LAMBDA (int ji, int ii, fp_t& running) {
            yakl::SArray<fp_t, 1, n_hydro> w;
            yakl::SArray<fp_t, 1, n_hydro> w0;
            prim(state.Q, ji + ng, ii + ng, w);
            prim(Q0, ji + ng, ii + ng, w0);
            const fp_t t = w(I(Prim::Pres)) / w(I(Prim::Rho));
            const fp_t t0 = w0(I(Prim::Pres)) / w0(I(Prim::Rho));
            running = std::max(running, std::abs(t - t0) / t0);
        },
        Kokkos::Max<fp_t>(dt_max)
    );
    fp_t mass = 0.0_fp;
    dex_parallel_reduce(
        "prominence_mhs_2d diag mass",
        FlatLoop<2>(sz.yc - 2 * ng, nx_inner),
        KOKKOS_LAMBDA (int ji, int ii, fp_t& running) {
            yakl::SArray<fp_t, 1, n_hydro> w;
            prim(state.Q, ji + ng, ii + ng, w);
            if (mhs_temperature(eos, c, w(I(Prim::Pres)), w(I(Prim::Rho)), 0, ji + ng, ii + ng) < 1e5_fp) {
                running += w(I(Prim::Rho)) * square(dx);
            }
        },
        Kokkos::Sum<fp_t>(mass)
    );
    fp_t e_kin = 0.0_fp;
    dex_parallel_reduce(
        "prominence_mhs_2d diag kinetic energy",
        FlatLoop<2>(sz.yc - 2 * ng, nx_inner),
        KOKKOS_LAMBDA (int ji, int ii, fp_t& running) {
            yakl::SArray<fp_t, 1, n_hydro> w;
            prim(state.Q, ji + ng, ii + ng, w);
            running += 0.5_fp * w(I(Prim::Rho)) * (square(w(I(Prim::Vx))) + square(w(I(Prim::Vy))) + square(w(I(Prim::Vz)))) * square(dx);
        },
        Kokkos::Sum<fp_t>(e_kin)
    );
    fmt::println(
        "prominence_mhs_2d diag: t = {:.3f} s, dt = {:.3e} s, max|v| = {:.4f} km/s at ({:.3f}, {:.3f}) Mm, min T = {:.3e} K, max|dT/T0| = {:.4e}, mass(T < 1e5 K) = {:.5e} kg/m, E_kin = {:.5e} J/m",
        sim.time, sim.dt, std::sqrt(v2_max) / 1e3, v_pos(0) / 1e6, v_pos(1) / 1e6, t_min, dt_max, mass, e_kin
    );
}

/// A state in the frame of the boundary: n outward normal, t1, t2 tangential.
struct NormalState {
    fp_t rho;
    fp_t p;
    fp_t un;
    fp_t ut1;
    fp_t ut2;
    fp_t bn;
    fp_t bt1;
    fp_t bt2;
};

/// Boundary state from the edge cell (in) and the far field (far), after Lare3d's open_bcs_alfven
/// and open_bcs_fast, with mu0 restored. Characteristics with speed >= 0 leave the box and take in,
/// the others take far.
KOKKOS_INLINE_FUNCTION NormalState open_bc_state(const NormalState& in, const NormalState& far, fp_t gamma, fp_t mu0) {
    constexpr fp_t tiny = 1e-30_fp;
    NormalState bc;
    bc.bn = 0.5_fp * (in.bn + far.bn);
    const fp_t c0far = std::sqrt(gamma * far.p / far.rho);
    const fp_t c0 = std::sqrt(gamma * in.p / in.rho);
    auto pick = [&](fp_t speed) -> const NormalState& { return speed >= 0.0_fp ? in : far; };
    const fp_t b2far = square(far.bn) + square(far.bt1) + square(far.bt2);
    if (square(far.bn) >= 0.5_fp * b2far) {
        // Field mostly normal: sound along B, Alfven, entropy.
        const fp_t cxfar = std::abs(far.bn) / std::sqrt(mu0 * far.rho);
        const fp_t cx = std::abs(in.bn) / std::sqrt(mu0 * in.rho);
        const NormalState& s1 = pick(in.un + c0);
        const NormalState& s2 = pick(in.un - c0);
        const NormalState& s3 = pick(in.un + cx);
        const NormalState& s4 = pick(in.un - cx);
        const NormalState& s5 = pick(in.un);
        const fp_t za = far.rho * cxfar;
        // Transverse momentum flux lambda = -B_n B_t / mu0; lambda +- rho c_A u_t ride the -+ Alfven waves.
        auto alfven = [&](fp_t l3, fp_t l4, fp_t u3, fp_t u4, fp_t& lam, fp_t& u) {
            lam = 0.5_fp * (l3 + l4 + za * (u3 - u4));
            u = 0.5_fp * (u3 + u4) + (l3 - l4) / (2.0_fp * za);
        };
        const fp_t bn_safe = std::abs(bc.bn) > tiny ? bc.bn : tiny;
        fp_t lam;
        alfven(-s3.bn * s3.bt1 / mu0, -s4.bn * s4.bt1 / mu0, s3.ut1, s4.ut1, lam, bc.ut1);
        bc.bt1 = -mu0 * lam / bn_safe;
        alfven(-s3.bn * s3.bt2 / mu0, -s4.bn * s4.bt2 / mu0, s3.ut2, s4.ut2, lam, bc.ut2);
        bc.bt2 = -mu0 * lam / bn_safe;
        // Normal stress P = p + (B_t^2 - B_n^2) / 2 mu0; P +- rho c_s u_n ride the -+ sound waves.
        auto stress = [&](const NormalState& s) {
            return s.p + 0.5_fp * (square(s.bt1) + square(s.bt2) - square(s.bn)) / mu0;
        };
        const fp_t zs = far.rho * c0far;
        const fp_t p1 = stress(s1);
        const fp_t p2 = stress(s2);
        const fp_t ptot = 0.5_fp * (p1 + p2 + zs * (s1.un - s2.un));
        bc.un = 0.5_fp * (s1.un + s2.un) + (p1 - p2) / (2.0_fp * zs);
        bc.p = std::max(ptot - 0.5_fp * (square(bc.bt1) + square(bc.bt2) - square(bc.bn)) / mu0, tiny);
        bc.rho = std::max(s5.rho + (bc.p - s5.p) / square(c0far), tiny);
    } else {
        // Field mostly tangential: fast, entropy. The transverse velocity is advected.
        const fp_t ctfar2 = (square(far.bt1) + square(far.bt2)) / (mu0 * far.rho);
        const fp_t cffar = std::sqrt(square(c0far) + ctfar2);
        const fp_t cf = std::sqrt(square(c0) + (square(in.bt1) + square(in.bt2)) / (mu0 * in.rho));
        const NormalState& s1 = pick(in.un + cf);
        const NormalState& s2 = pick(in.un - cf);
        const NormalState& s3 = pick(in.un);
        auto ptot = [&](const NormalState& s) { return s.p + 0.5_fp * (square(s.bt1) + square(s.bt2)) / mu0; };
        auto pmag = [&](const NormalState& s) { return 0.5_fp * (square(s.bt1) + square(s.bt2)) / mu0; };
        const fp_t zf = far.rho * cffar;
        bc.bt1 = 0.5_fp * (s1.bt1 + s2.bt1);
        bc.bt2 = 0.5_fp * (s1.bt2 + s2.bt2);
        bc.un = 0.5_fp * (s1.un + s2.un) + (ptot(s1) - ptot(s2)) / (2.0_fp * zf);
        bc.p = std::max(0.5_fp * (ptot(s1) + ptot(s2) + zf * (s1.un - s2.un)) - 0.5_fp * (pmag(s1) + pmag(s2)), tiny);
        bc.rho = std::max(s3.rho + (bc.p - s3.p) / square(c0far), tiny);
        bc.ut1 = s3.ut1;
        bc.ut2 = s3.ut2;
    }
    return bc;
}

/// Characteristic boundary state, linearised about the far field: the edge cell's deviation from the far
/// field is split into the 7 waves along the normal (Roe & Balsara 1996 primitive eigenvectors, at the far
/// state), the outgoing ones are kept and the incoming ones (and the entropy wave, so T stays at the far
/// field, a heat bath) are dropped. B_n is the mean of edge and far, as in Lare3d.
KOKKOS_INLINE_FUNCTION NormalState open_char_state(const NormalState& in, const NormalState& far, fp_t gamma, fp_t mu0) {
    constexpr int n = 7;
    const fp_t sm = std::sqrt(mu0);
    const fp_t rho = far.rho;
    const fp_t sr = std::sqrt(rho);
    const fp_t bn = far.bn / sm;
    const fp_t by = far.bt1 / sm;
    const fp_t bz = far.bt2 / sm;
    const fp_t a2 = gamma * far.p / rho;
    const fp_t a = std::sqrt(a2);
    const fp_t bx2 = square(bn) / rho;
    const fp_t b2 = bx2 + (square(by) + square(bz)) / rho;
    const fp_t disc = std::sqrt(std::max(square(a2 + b2) - 4.0_fp * a2 * bx2, 0.0_fp));
    const fp_t cf2 = 0.5_fp * (a2 + b2 + disc);
    const fp_t cs2 = std::max(0.5_fp * (a2 + b2 - disc), 0.0_fp);
    const fp_t cf = std::sqrt(cf2);
    const fp_t cs = std::sqrt(cs2);
    const fp_t ca = std::sqrt(bx2);
    fp_t af = 1.0_fp;
    fp_t as = 0.0_fp;
    if (cf2 - cs2 > 1e-12_fp * cf2) {
        af = std::sqrt(std::max(a2 - cs2, 0.0_fp) / (cf2 - cs2));
        as = std::sqrt(std::max(cf2 - a2, 0.0_fp) / (cf2 - cs2));
    }
    const fp_t bt = std::sqrt(square(by) + square(bz));
    fp_t beta_y = 1.0_fp / std::sqrt(2.0_fp);
    fp_t beta_z = beta_y;
    if (bt > 1e-12_fp * std::abs(bn)) {
        beta_y = by / bt;
        beta_z = bz / bt;
    }
    const fp_t sg = bn >= 0.0_fp ? 1.0_fp : -1.0_fp;
    // NOTE(claude): Columns -f, -A, -s, 0, +s, +A, +f. Rows (rho, u_n, u_t1, u_t2, B_t1, B_t2, p) with
    // B / sqrt(mu0), each row then scaled by d so the system is O(1).
    fp_t R[n][n];
    const fp_t lam[n] = {far.un - cf, far.un - ca, far.un - cs, far.un, far.un + cs, far.un + ca, far.un + cf};
    for (int k = 0; k < 2; ++k) {
        const int col = k == 0 ? 0 : 6;
        const fp_t s = k == 0 ? -1.0_fp : 1.0_fp;
        R[0][col] = rho * af;
        R[1][col] = s * af * cf;
        R[2][col] = -s * as * cs * beta_y * sg;
        R[3][col] = -s * as * cs * beta_z * sg;
        R[4][col] = as * sr * a * beta_y;
        R[5][col] = as * sr * a * beta_z;
        R[6][col] = rho * a2 * af;
    }
    for (int k = 0; k < 2; ++k) {
        const int col = k == 0 ? 1 : 5;
        const fp_t s = k == 0 ? -1.0_fp : 1.0_fp;
        R[0][col] = 0.0_fp;
        R[1][col] = 0.0_fp;
        R[2][col] = -beta_z;
        R[3][col] = beta_y;
        R[4][col] = s * sg * sr * beta_z;
        R[5][col] = -s * sg * sr * beta_y;
        R[6][col] = 0.0_fp;
    }
    for (int k = 0; k < 2; ++k) {
        const int col = k == 0 ? 2 : 4;
        const fp_t s = k == 0 ? -1.0_fp : 1.0_fp;
        R[0][col] = rho * as;
        R[1][col] = s * as * cs;
        R[2][col] = s * af * cf * beta_y * sg;
        R[3][col] = s * af * cf * beta_z * sg;
        R[4][col] = -af * sr * a * beta_y;
        R[5][col] = -af * sr * a * beta_z;
        R[6][col] = rho * a2 * as;
    }
    for (int r = 0; r < n; ++r) {
        R[r][3] = r == 0 ? 1.0_fp : 0.0_fp;
    }
    const fp_t d[n] = {1.0_fp / rho, 1.0_fp / cf, 1.0_fp / cf, 1.0_fp / cf, 1.0_fp / (sr * cf), 1.0_fp / (sr * cf), 1.0_fp / (rho * cf2)};
    const fp_t dw[n] = {
        in.rho - far.rho, in.un - far.un, in.ut1 - far.ut1, in.ut2 - far.ut2,
        (in.bt1 - far.bt1) / sm, (in.bt2 - far.bt2) / sm, in.p - far.p
    };
    // Solve (D R) alpha = D dw, Gaussian elimination with partial pivoting.
    fp_t M[n][n + 1];
    for (int r = 0; r < n; ++r) {
        for (int col = 0; col < n; ++col) {
            M[r][col] = d[r] * R[r][col];
        }
        M[r][n] = d[r] * dw[r];
    }
    for (int col = 0; col < n; ++col) {
        int piv = col;
        for (int r = col + 1; r < n; ++r) {
            if (std::abs(M[r][col]) > std::abs(M[piv][col])) {
                piv = r;
            }
        }
        for (int c2 = 0; c2 <= n; ++c2) {
            const fp_t tmp = M[col][c2];
            M[col][c2] = M[piv][c2];
            M[piv][c2] = tmp;
        }
        const fp_t diag = M[col][col];
        for (int r = col + 1; r < n; ++r) {
            const fp_t f = M[r][col] / diag;
            for (int c2 = col; c2 <= n; ++c2) {
                M[r][c2] -= f * M[col][c2];
            }
        }
    }
    fp_t alpha[n];
    for (int r = n - 1; r >= 0; --r) {
        fp_t acc = M[r][n];
        for (int col = r + 1; col < n; ++col) {
            acc -= M[r][col] * alpha[col];
        }
        alpha[r] = acc / M[r][r];
    }
    fp_t w[n] = {far.rho, far.un, far.ut1, far.ut2, far.bt1 / sm, far.bt2 / sm, far.p};
    for (int k = 0; k < n; ++k) {
        if (k == 3 || lam[k] <= 0.0_fp) {
            continue;
        }
        for (int r = 0; r < n; ++r) {
            w[r] += alpha[k] * R[r][k];
        }
    }
    NormalState bc;
    bc.p = std::max(w[6], 1e-30_fp);
    bc.rho = bc.p * far.rho / far.p;
    bc.un = w[1];
    bc.ut1 = w[2];
    bc.ut2 = w[3];
    bc.bt1 = w[4] * sm;
    bc.bt2 = w[5] * sm;
    bc.bn = 0.5_fp * (in.bn + far.bn);
    return bc;
}

/// x walls, see the problem description. y is expected to be periodic.
template <typename FTraits>
static void fill_bcs(const Simulation& sim, const Fp4d& Q0, const ProminenceMhsParams& params) {
    using Prim = typename FTraits::prim;
    const XBoundary mode = params.x_boundary;
    // NOTE(claude): Wall velocity for the driver (zero outside its window), from the time at the start of
    // the step. Ghost v = 2 v_wall - v_mirror puts the face velocity at v_wall.
    fp_t phase = 0.0_fp;
    const f64 t_drive = (sim.time - params.driver_t_start) / params.driver_period;
    if (t_drive >= 0.0 && (params.driver_cycles <= 0.0_fp || t_drive <= params.driver_cycles)) {
        phase = fp_t(std::sin(2.0 * M_PI * t_drive));
    }
    const vec3 v_drive = params.driver_velocity;
    const bool drive_xs = params.driver_xs;
    const bool drive_xe = params.driver_xe;
    const fp_t drive_y0 = params.driver_y0;
    const fp_t drive_wy = params.driver_wy;
    constexpr int n_hydro = FTraits::num_vars;
    const auto& state = sim.state;
    const auto& sz = state.sz;
    const auto& bdry = state.boundaries;
    const auto& eos = sim.eos;
    const auto& Q = state.Q;
    const int ng = sz.ng;
    const int n_vars = Q.extent(0);
    const fp_t mu0 = state.mu0;

    dex_parallel_for(
        "prominence_mhs_2d x BCs",
        FlatLoop<3>(sz.zc, sz.yc, 2 * ng),
        KOKKOS_LAMBDA (int k, int j, int ii) {
            const bool end = ii >= ng;
            const BoundaryType bound = end ? bdry.xe : bdry.xs;
            if (bound != BoundaryType::UserFn && bound != BoundaryType::Constant) {
                return;
            }
            const int i = end ? sz.xc - 1 - (ii - ng) : ii;
            const int i_flip = end ? sz.xc - 2 * ng + (ii - ng) : 2 * ng - 1 - ii;
            const int i_edge = end ? sz.xc - ng - 1 : ng;
            yakl::SArray<fp_t, 1, n_hydro> w;
            yakl::SArray<fp_t, 1, n_hydro> w_ref;
            yakl::SArray<fp_t, 1, n_hydro> w_edge;
            yakl::SArray<fp_t, 1, n_hydro> w_edge_ref;
            cons_to_prim<FTraits>(eos.gamma, mu0, QtyView(Q, CellIndex{.i = i_flip, .j = j, .k = k}), w);
            cons_to_prim<FTraits>(eos.gamma, mu0, QtyView(Q0, CellIndex{.i = i, .j = j, .k = k}), w_ref);
            cons_to_prim<FTraits>(eos.gamma, mu0, QtyView(Q, CellIndex{.i = i_edge, .j = j, .k = k}), w_edge);
            cons_to_prim<FTraits>(eos.gamma, mu0, QtyView(Q0, CellIndex{.i = i_edge, .j = j, .k = k}), w_edge_ref);
            if (mode == XBoundary::Open || mode == XBoundary::OpenTBath || mode == XBoundary::OpenTransverse || mode == XBoundary::OpenChar) {
                // NOTE(claude): Far field is the loaded state in the first ghost layer; every layer gets the
                // boundary state. sgn turns x into the outward normal.
                const int i_far = end ? sz.xc - ng : ng - 1;
                yakl::SArray<fp_t, 1, n_hydro> w_far;
                cons_to_prim<FTraits>(eos.gamma, mu0, QtyView(Q0, CellIndex{.i = i_far, .j = j, .k = k}), w_far);
                const fp_t sgn = end ? 1.0_fp : -1.0_fp;
                auto to_normal = [&](const yakl::SArray<fp_t, 1, n_hydro>& v) {
                    return NormalState{
                        .rho = v(I(Prim::Rho)), .p = v(I(Prim::Pres)),
                        .un = sgn * v(I(Prim::Vx)), .ut1 = v(I(Prim::Vy)), .ut2 = v(I(Prim::Vz)),
                        .bn = sgn * v(I(Prim::Bx)), .bt1 = v(I(Prim::By)), .bt2 = v(I(Prim::Bz))
                    };
                };
                const NormalState bc = (mode == XBoundary::OpenChar)
                    ? open_char_state(to_normal(w_edge), to_normal(w_far), eos.gamma, mu0)
                    : open_bc_state(to_normal(w_edge), to_normal(w_far), eos.gamma, mu0);
                if (mode == XBoundary::OpenTransverse) {
                    w(I(Prim::Pres)) = w(I(Prim::Rho)) * w_ref(I(Prim::Pres)) / w_ref(I(Prim::Rho));
                    w(I(Prim::Vx)) = -w(I(Prim::Vx));
                    w(I(Prim::Bx)) = w_ref(I(Prim::Bx));
                } else {
                    w = w_edge;
                    w(I(Prim::Pres)) = bc.p;
                    w(I(Prim::Rho)) = (mode == XBoundary::OpenTBath) ? bc.p * w_far(I(Prim::Rho)) / w_far(I(Prim::Pres)) : bc.rho;
                    w(I(Prim::Vx)) = sgn * bc.un;
                    w(I(Prim::Bx)) = sgn * bc.bn;
                }
                w(I(Prim::Vy)) = bc.ut1;
                w(I(Prim::Vz)) = bc.ut2;
                w(I(Prim::By)) = bc.bt1;
                w(I(Prim::Bz)) = bc.bt2;
            } else {
                if (mode == XBoundary::Closed) {
                    w(I(Prim::Pres)) = w(I(Prim::Rho)) * w_ref(I(Prim::Pres)) / w_ref(I(Prim::Rho));
                } else {
                    w(I(Prim::Rho)) = w_ref(I(Prim::Rho));
                    w(I(Prim::Pres)) = w_ref(I(Prim::Pres));
                }
                fp_t drive = 0.0_fp;
                if (phase != 0.0_fp && (end ? drive_xe : drive_xs)) {
                    drive = phase;
                    if (drive_wy > 0.0_fp) {
                        drive *= std::exp(-square((state.get_pos(i, j, k)(1) - drive_y0) / drive_wy));
                    }
                }
                const fp_t x_sign = end ? -1.0_fp : 1.0_fp;
                w(I(Prim::Vx)) = 2.0_fp * drive * x_sign * v_drive(0) - w(I(Prim::Vx));
                w(I(Prim::Vy)) = 2.0_fp * drive * v_drive(1) - w(I(Prim::Vy));
                w(I(Prim::Vz)) = 2.0_fp * drive * v_drive(2) - w(I(Prim::Vz));
                w(I(Prim::Bx)) = w_ref(I(Prim::Bx));
                w(I(Prim::By)) = w_ref(I(Prim::By)) + (w_edge(I(Prim::By)) - w_edge_ref(I(Prim::By)));
                w(I(Prim::Bz)) = w_ref(I(Prim::Bz)) + (w_edge(I(Prim::Bz)) - w_edge_ref(I(Prim::Bz)));
            }
            if constexpr (is_instance(FTraits::fluid_type, FluidType::GlmMhd)) {
                w(I(Prim::Psi)) = 0.0_fp;
            }
            if constexpr (FTraits::has_hypertc) {
                w(I(Prim::HeatF)) = w_edge(I(Prim::HeatF));
            }
            prim_to_cons<FTraits>(eos.gamma, mu0, w, QtyView(Q, CellIndex{.i = i, .j = j, .k = k}));
            for (int v = n_hydro; v < n_vars; ++v) {
                Q(v, k, j, i) = Q(v, k, j, i_edge) * w(I(Prim::Rho)) / w_edge(I(Prim::Rho));
            }
        }
    );
    Kokkos::fence();
}

/// problem.ic_closure: fixed_pt. Each cell keeps its loaded thermal pressure and temperature (T from the loaded y,
/// read back from IonE, in the EOS's n_H = rho / m_H convention); the LTE closure then sets y at that (p, T), and
/// rho, IonE and Ene follow, with v and B unchanged. A static Dex solve with conserve_pressure plays this role later.
template <typename FTraits>
static void apply_ic_closure(const Simulation& sim, const Fp4d& Q) {
    using Prim = typename FTraits::prim;
    using Cons = typename FTraits::cons;
    constexpr int n_hydro = FTraits::num_vars;
    const auto& eos = sim.eos;
    const fp_t mu0 = sim.state.mu0;
    const auto& sz = sim.state.sz;
    const fp_t m_h = sim.state.p_mass;
    constexpr fp_t k_B = AnalyticLteH::k_B;
    constexpr fp_t chi_H = AnalyticLteH::chi_H;
    const fp_t m_per_h = eos.mass_per_h * m_h;
    dex_parallel_for(
        "prominence_mhs_2d IC closure (fixed p, T)",
        FlatLoop<3>(sz.zc, sz.yc, sz.xc),
        KOKKOS_LAMBDA (int k, int j, int i) {
            yakl::SArray<fp_t, 1, n_hydro> w;
            const auto q = QtyView(Q, CellIndex{.i = i, .j = j, .k = k});
            cons_to_prim<FTraits>(eos.gamma, mu0, q, w);
            const fp_t pres = w(I(Prim::Pres));
            const fp_t y0 = w(I(Prim::IonE)) * m_per_h / chi_H;
            const fp_t temp = pres / (w(I(Prim::Rho)) / m_per_h * (1.0_fp + y0) * k_B);
            const fp_t y = y_from_ntot(pres / (k_B * temp), temp);
            w(I(Prim::Rho)) = pres / ((1.0_fp + y) * k_B * temp) * m_per_h;
            w(I(Prim::IonE)) = y * chi_H / m_per_h;
            prim_to_cons<FTraits>(eos.gamma, mu0, w, q);
        }
    );
    Kokkos::fence();
}

/// Gaussian velocity perturbation added to the loaded state, optionally only on the cold body.
template <typename FTraits>
static void add_pulse(Simulation& sim, const ProminenceMhsParams& params) {
    using Prim = typename FTraits::prim;
    using Cons = typename FTraits::cons;
    constexpr int n_hydro = FTraits::num_vars;
    const auto& eos = sim.eos;
    const fp_t mu0 = sim.state.mu0;
    const fp_t c = ideal_temperature_coeff(sim);
    const fp_t t_cold = params.pulse_cold_temperature;
    const auto& state = sim.state;
    const auto& sz = state.sz;
    const auto& Q = state.Q;
    const vec3 v0 = params.pulse_velocity;
    const fp_t x0 = params.pulse_x0;
    const fp_t y0 = params.pulse_y0;
    const fp_t wx = params.pulse_wx;
    const fp_t wy = params.pulse_wy;
    const bool sine = params.pulse_profile == PulseProfile::Sine;
    const fp_t length = params.pulse_length;
    const fp_t k_mode = params.pulse_harmonic * fp_t(M_PI) / length;
    if (t_cold > 0.0_fp && !eos.is_constant) {
        // NOTE(claude): setup_ics runs before the first EOS update, so y_space is not yet set.
        sim.update_eos(sim);
    }
    dex_parallel_for(
        "prominence_mhs_2d pulse",
        FlatLoop<3>(sz.zc, sz.yc, sz.xc),
        KOKKOS_LAMBDA (int k, int j, int i) {
            const vec3 pos = state.get_pos(i, j, k);
            fp_t g = std::exp(-square((pos(1) - y0) / wy));
            if (sine) {
                // NOTE(claude): sin(n pi (x - x0) / L) on |x - x0| <= L / 2: odd about x0 for odd n, so the
                // thread is compressed or expanded symmetrically (the slow mode, not the bulk shift).
                const fp_t dx0 = pos(0) - x0;
                g *= (std::abs(dx0) <= 0.5_fp * length) ? std::sin(k_mode * dx0) : 0.0_fp;
            } else {
                g *= std::exp(-square((pos(0) - x0) / wx));
            }
            if (t_cold > 0.0_fp) {
                yakl::SArray<fp_t, 1, n_hydro> w;
                cons_to_prim<FTraits>(eos.gamma, mu0, QtyView(Q, CellIndex{.i = i, .j = j, .k = k}), w);
                const fp_t temp = mhs_temperature(eos, c, w(I(Prim::Pres)), w(I(Prim::Rho)), k, j, i);
                g *= 0.5_fp * (1.0_fp - std::tanh(std::log(temp / t_cold) / 0.2_fp));
            }
            // NOTE(claude): Adds to the loaded velocity; the energy change is rho (v . dv + dv^2 / 2).
            const fp_t rho = Q(I(Cons::Rho), k, j, i);
            const fp_t dv0 = g * v0(0);
            const fp_t dv1 = g * v0(1);
            const fp_t dv2 = g * v0(2);
            Q(I(Cons::Ene), k, j, i) += Q(I(Cons::MomX), k, j, i) * dv0 + Q(I(Cons::MomY), k, j, i) * dv1
                + Q(I(Cons::MomZ), k, j, i) * dv2 + 0.5_fp * rho * (square(dv0) + square(dv1) + square(dv2));
            Q(I(Cons::MomX), k, j, i) += rho * dv0;
            Q(I(Cons::MomY), k, j, i) += rho * dv1;
            Q(I(Cons::MomZ), k, j, i) += rho * dv2;
        }
    );
    Kokkos::fence();
}

MOSSCAP_NEW_PROBLEM(prominence_mhs_2d) {
    MOSSCAP_PROBLEM_PREAMBLE(prominence_mhs_2d);
    if (sim.num_dim != num_dim) {
        throw std::runtime_error(fmt::format("{} only handles {}d problems", PROBLEM_NAME, num_dim));
    }
    FluidTraitsRt traits(sim.num_dim, sim.fluid_type);
    if (!traits.is_mhd) {
        throw std::runtime_error(fmt::format("{} requires an MHD fluid.", PROBLEM_NAME));
    }
    if (sim.eos.type != EosType::Ideal && sim.eos.type != EosType::AnalyticLteH && sim.eos.type != EosType::TracerEos) {
        throw std::runtime_error(fmt::format(
            "{} supports the ideal, analyticlteh and TracerEos equations of state.", PROBLEM_NAME
        ));
    }
    if (!get_or<bool>(config, "sources.thermal_conduction.enable", false)) {
        throw std::runtime_error(fmt::format("{} requires sources.thermal_conduction.enable.", PROBLEM_NAME));
    }
    ProminenceMhsParams params = load_params(config);
    f64 ic_time = 0.0;
    const Fp4d Q0 = load_reference(sim, params.ic_path, ic_time);
    if (params.ic_closure_fixed_pt) {
        if (sim.eos.type != EosType::AnalyticLteH || !sim.eos.has_ion_e || sim.eos.total_abund != 1.0_fp) {
            throw std::runtime_error(fmt::format(
                "{}: ic_closure fixed_pt needs eos analyticlteh with include_ionisation_energy (pure H).", PROBLEM_NAME
            ));
        }
        invoke_fluid_traits(sim.num_dim, sim.fluid_type, [&]<typename FTraits>(FTraits) {
            apply_ic_closure<FTraits>(sim, Q0);
        });
        fmt::println("{}: IC closure at fixed p, T (LTE H)", PROBLEM_NAME);
    }
    if (params.driver_t_start < 0.0_fp) {
        params.driver_t_start = ic_time;
    }
    if (square(params.driver_velocity(0)) + square(params.driver_velocity(1)) + square(params.driver_velocity(2)) > 0.0_fp) {
        fmt::println("prominence_mhs_2d: wall driver v = ({}, {}, {}) m/s, period {} s, {} cycles from t = {} s",
            params.driver_velocity(0), params.driver_velocity(1), params.driver_velocity(2), params.driver_period,
            params.driver_cycles, params.driver_t_start);
    }

    if (has_key(config, "sources.gravity")) {
        setup_gravity(sim, config);
    }
    setup_thin_loss(sim, config);
    const int loss_idx = source_term_index(sim, "thin_loss");
    if (loss_idx == sim.compute_source_terms.size()) {
        throw std::runtime_error(fmt::format("{} requires sources.thin_loss.enable.", PROBLEM_NAME));
    }
    const auto* loss_ctx = (ThinLossContext*)sim.compute_source_terms[loss_idx].get_context();
    const TownsendCurve<Fp1dHost> curve{
        .temps = loss_ctx->curve.temps.createHostCopy(),
        .lambdas = loss_ctx->curve.lambdas.createHostCopy(),
        .alpha_k = loss_ctx->curve.alpha_k.createHostCopy()
    };
    // NOTE(claude): H = y n_H(p, t_corona)^2 Λ(t_corona) ... = h_coeff p^2 ..., with the coronal y: the ideal
    // EOS's, else fully ionised (LTE H at t_corona).
    const fp_t y_corona = sim.eos.is_constant ? sim.eos.y : 1.0_fp;
    const f64 n_per_p = 1.0 / ((sim.eos.total_abund + y_corona) * ConstantsF64::k_B * params.t_corona);
    const fp_t h_coeff = y_corona * square(n_per_p) * townsend_lambda(curve, params.t_corona);
    const fp_t h_const = params.heating_constant ? h_coeff * square(params.heating_pressure) : 0.0_fp;
    if (params.heating_constant) {
        fmt::println("prominence_mhs_2d: heating H0 = {:.4e} W/m3 (corona balanced at p = {} Pa); IC {} at t = {}", h_const, params.heating_pressure, params.ic_path, ic_time);
    } else {
        fmt::println("prominence_mhs_2d: heating H = {:.4e} p^2 tanh((T - t_core) / t_core); IC {} at t = {}", h_coeff, params.ic_path, ic_time);
    }

    sim.setup_ics = [=](Simulation& sim) {
        const auto& sz = sim.state.sz;
        JasUnpack(sim.state, Q);
        sim.out_cfg.prev_output_time = ic_time;
        sim.time = ic_time;
        const int n_vars = Q0.extent(0);
        dex_parallel_for(
            "prominence_mhs_2d IC",
            FlatLoop<3>(sz.zc, sz.yc, sz.xc),
            KOKKOS_LAMBDA (int k, int j, int i) {
                for (int v = 0; v < n_vars; ++v) {
                    Q(v, k, j, i) = Q0(v, k, j, i);
                }
            }
        );
        Kokkos::fence();
        if (sim.eos.type == EosType::TracerEos) {
            invoke_fluid_traits(sim.num_dim, sim.fluid_type, [&]<typename FTraits>(FTraits) {
                set_y_from_ion_e<FTraits>(sim);
            });
        }
        if (square(params.pulse_velocity(0)) + square(params.pulse_velocity(1)) + square(params.pulse_velocity(2)) == 0.0_fp) {
            return;
        }
        invoke_fluid_traits(
            sim.num_dim,
            sim.fluid_type,
            [&]<typename FTraits>(FTraits) {
                add_pulse<FTraits>(sim, params);
            }
        );
    };

    sim.compute_source_terms.push_back(SourceTerm{
        .name = "mhs_heating",
        .fn = [=](const Simulation& sim, const Fp4d& out) {
            invoke_fluid_traits(
                sim.num_dim,
                sim.fluid_type,
                [&]<typename FTraits>(FTraits) {
                    heating_kernel<FTraits>(sim, h_coeff, params.t_core, h_const, source_target(sim, out));
                }
            );
        },
        .trac_role = TracRole::EnergySourceSink
    });
    if (params.diag_interval > 0.0_fp) {
        // NOTE(claude): Registered as a source term as it's the only per-step hook; it adds nothing to S.
        auto next_diag = std::make_shared<f64>(-1.0);
        sim.compute_source_terms.push_back(SourceTerm{
            .name = "mhs_diagnostics",
            .fn = [=](const Simulation& sim, const Fp4d& out) {
                if (out.initialized()) {
                    return; // nothing to evaluate alone, and printing would advance the diagnostic clock
                }
                if (*next_diag < 0.0) {
                    *next_diag = sim.time;
                }
                if (sim.time < *next_diag) {
                    return;
                }
                *next_diag = sim.time + params.diag_interval;
                invoke_fluid_traits(
                    sim.num_dim,
                    sim.fluid_type,
                    [&]<typename FTraits>(FTraits) {
                        print_diagnostics<FTraits>(sim, Q0);
                    }
                );
            },
            .trac_role = TracRole::None
        });
    }
    if (get_or<bool>(config, "sources.friction.enable", false)) {
        setup_friction(sim, config);
    }
    setup_thermal_conduction(sim, config);
    setup_replace_small_values(sim, config);

    sim.user_bc = [=](const Simulation& sim) {
        invoke_fluid_traits(
            sim.num_dim,
            sim.fluid_type,
            [&]<typename FTraits>(FTraits) {
                fill_bcs<FTraits>(sim, Q0, params);
            }
        );
    };
}

}
