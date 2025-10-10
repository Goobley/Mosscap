#include "ProblemGenerator.hpp"
#include "../Hydro.hpp"
#include "../MosscapConfig.hpp"
#include "../SourceTerms/Gravity.hpp"

// NOTE(cmo): This is a 2d problem
static constexpr int num_dim = 2;

using namespace Mosscap;

struct RadLossCoeffs {
    fp_t T_peak;
    fp_t tau_rad;
};

template <int NumDim>
static void rad_loss_kernel(const Simulation& sim, const RadLossCoeffs& rl) {
    using Cons = Cons<NumDim>;
    using Prim = Prim<NumDim>;

    const auto& eos = sim.eos;
    const auto& W = sim.state.W;
    const auto& S = sim.sources.S;
    const auto& sz = sim.state.sz;
    const fp_t inv_T_peak = 1.0_fp / rl.T_peak;
    const fp_t inv_timescale = 1.0_fp / rl.tau_rad;

    dex_parallel_for(
        "Apply rad loss",
        FlatLoop<3>(sz.zc, sz.yc, sz.xc),
        KOKKOS_LAMBDA (int k, int j, int i) {

            const CellIndex idx{
                .i=i,
                .j=j,
                .k=k
            };
            const auto& w = QtyView(W, idx);

            const fp_t temperature = eos.gamma * w(I(Prim::Pres)) / w(I(Prim::Rho));
            // fp_t loss = (1.0_fp - square(std::tanh(std::log10(temperature * inv_T_peak) * (0.5_fp * M_PI) * 5.0_fp))) / rl.tau_rad;
            // loss *= square(w(I(Prim::Rho)));
            fp_t loss = 1.0_fp / square(std::cosh(std::log10(temperature * inv_T_peak)) * (1.0_fp / 0.04_fp * M_PIf)) * inv_timescale;
            S(I(Cons::Ene), k, j, i) += -loss;
        }
    );
    Kokkos::fence();

}

MOSSCAP_NEW_PROBLEM(hillier_snow_kelvin_helmholtz) {
    MOSSCAP_PROBLEM_PREAMBLE(hillier_snow_kelvin_helmholtz);

    if (sim.num_dim != num_dim) {
        throw std::runtime_error(fmt::format(
            "{} only handles {}d problems", PROBLEM_NAME, num_dim
        ));
    }
    using Prim = Prim<num_dim>;
    constexpr int n_hydro = N_HYDRO_VARS<num_dim>;
    const auto& state = sim.state;
    const auto& eos = sim.eos;
    const auto& sz = state.sz;

    const u64 seed = 123456UL;
    const int axis = 0;
    // Modes:
    // 0 - Hillier + Snow periodic
    // 1 -  1 blob
    const int mode = get_or<int>(config, "problem.mode", 0);
    const fp_t blob_radius = get_or<fp_t>(config, "problem.blob_radius", 0.4_fp);
    const fp_t streaming_vel = get_or<fp_t>(config, "problem.streaming_vel", 0.1_fp);
    const fp_t random_scale = get_or<fp_t>(config, "problem.random_scale", 0.01_fp);
    const fp_t rho_h = 100.0_fp;
    const fp_t rho_l = 1.0_fp;
    const fp_t pressure = 1.0_fp / eos.gamma;

    dex_parallel_for(
        "Setup problem",
        FlatLoop<3>(sz.zc, sz.yc, sz.xc),
        KOKKOS_LAMBDA (int k, int j, int i) {
            vec3 p = state.get_pos(i, j);
            yakl::SArray<fp_t, 1, n_hydro> w(0.0_fp);

            w(I(Prim::Pres)) = pressure;
            if (mode == 0) {
                if (p(axis) > 0.0_fp) {
                    w(I(Prim::Rho)) = rho_h;
                } else {
                    w(I(Prim::Rho)) = rho_l;
                    w(I(Prim::Vy)) = streaming_vel;
                }
            } else if (mode == 1) {
                if (square(p(0) + 0.5_fp) + square(p(1) - 1.0_fp) < square(blob_radius)) {
                    w(I(Prim::Rho)) = rho_h;
                } else {
                    w(I(Prim::Rho)) = rho_l;
                    w(I(Prim::Vy)) = streaming_vel;
                }

            }
            yakl::Random rng(seed + k * sz.yc * sz.xc + j * sz.xc + i);
            w(I(Prim::Vx)) = random_scale * (rng.genFP<fp_t>() - 0.5_fp);
            CellIndex idx {
                .i = i,
                .j = j,
                .k = k
            };
            prim_to_cons<num_dim>(eos.gamma, w, QtyView(state.Q, idx));
        }
    );

    if (sim.state.boundaries.ys == BoundaryType::Constant && !config["boundary"]["ys"].IsSequence()) {
        const fp_t boundary_inflow_vel = get_or<fp_t>(config, "problem.boundary_inflow_vel", 1.0_fp);
        yakl::SArray<fp_t, 1, n_hydro> w(0.0_fp);
        w(I(Prim::Rho)) = rho_l;
        w(I(Prim::Vy)) = boundary_inflow_vel;
        w(I(Prim::Pres)) = pressure;
        // NOTE(cmo): This could blow up with EOSs that are spatially dependent
        // and not fully configured at this point.
        CellIndex idx{
            .i=0,
            .j=0,
            .k=0
        };
        prim_to_cons<num_dim>(eos.gamma, w, state.boundaries.ys_const);
    }

    bool enable_rad_loss = get_or<bool>(config, "problem.enable_rad_loss", false);
    if (enable_rad_loss) {
        RadLossCoeffs rad_loss{
            .T_peak = get_or<fp_t>(config, "sources.rad_loss.T_peak", 0.15_fp),
            .tau_rad = get_or<fp_t>(config, "sources.rad_loss.tau_rad", 1e2_fp)
        };

        // TODO(cmo): Setup sources
        sim.compute_source_terms.push_back(SourceTerm{
            .name = "thin_rad_loss",
            .fn = [=](const Simulation& sim) {
                const int num_dim = sim.num_dim;
                if (num_dim == 1) {
                    rad_loss_kernel<1>(sim, rad_loss);
                } else if (num_dim == 2) {
                    rad_loss_kernel<2>(sim, rad_loss);
                } else {
                    rad_loss_kernel<3>(sim, rad_loss);
                }
            }
        });
    }
    bool enable_gravity = get_or<bool>(config, "problem.enable_gravity", false);
    if (enable_gravity) {
        setup_gravity(sim, config);
    }
}