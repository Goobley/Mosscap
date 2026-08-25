#include "ProblemGenerator.hpp"
#include "../Hydro.hpp"
#include "../MosscapConfig.hpp"
#include "../SourceTerms/TownsendThinLoss.hpp"
#include "../SourceTerms/ReplaceSmallValues.hpp"
#include "../SourceTerms.hpp"
#include "../AnalyticLteH.hpp"

// NOTE(cmo): This is a 2d problem
static constexpr int num_dim = 2;

namespace Mosscap {

static constexpr f64 h_mass = ConstantsF64::u;
static constexpr f64 chi_H = 2.178710282685096e-18; // [J]

struct BackgroundParams {
    fp_t T0;
    fp_t lambda_T0;
    fp_t heating_coeff;
    yakl::SArray<fp_t, 1, N_HYDRO_VARS<3, FLUID_WITH_MAX_VARS>> background;
};

template <typename FTraits>
static void background_heating_kernel(const Simulation& sim, const BackgroundParams& bg) {
    JasUnpack(sim, state, sources);
    JasUnpack(state, sz);
    const auto& S = sources.S;
    int nx = sz.xc - 2 * sz.ng;
    int ny = std::max(sz.yc - 2 * sz.ng, 1);
    int nz = std::max(sz.zc - 2 * sz.ng, 1);

    int source_idx = source_term_index(sim, "thin_loss");
    auto loss_ctx = (ThinLossContext*)sim.compute_source_terms[source_idx].get_context();
    const fp_t H = -thin_loss_single_val<FTraits>(sim, *loss_ctx, bg.background);

    JasUnpack(bg, heating_coeff);
    dex_parallel_for(
        FlatLoop<3>(nz, ny, nx),
        KOKKOS_LAMBDA (int ki, int ji, int ii) {
            const int k = nz == 1 ? ki : ki + sz.ng;
            const int j = ny == 1 ? ji : ji + sz.ng;
            const int i = ii + sz.ng;
            using Cons = typename FTraits::cons;

            S(I(Cons::Ene), k, j, i) += heating_coeff * H;
        }
    );

    Kokkos::fence();
}

template <typename Fluid>
static BackgroundParams get_background_params(Simulation& sim, const YAML::Node& config) {
    using Cons = typename Fluid::cons;
    using Prim = typename Fluid::prim;
    constexpr int n_hydro = Fluid::num_vars;

    JasUnpack(sim, state, eos);
    BackgroundParams bg;

    const std::string input_path = get_or<std::string>(config, "problem.ic_path", "slow_mode_ti.nc");
    yakl::SimpleNetCDF nc;
    nc.open(input_path, yakl::NETCDF_MODE_READ);

    fp_t rho0, T0, lambda_T0, bx0, by0, bz0;
    nc.read(rho0, "rho0");
    nc.read(T0, "T0");
    nc.read(lambda_T0, "lambda_T0");
    nc.read(bx0, "bx0");
    nc.read(by0, "by0");
    nc.read(bz0, "bz0");

    yakl::SArray<fp_t, 1, n_hydro> w(0.0_fp);
    w(I(Prim::Rho)) = rho0;
    w(I(Prim::Vx)) = 0.0_fp;
    w(I(Prim::Vy)) = 0.0_fp;
    w(I(Prim::Bx)) = bx0;
    w(I(Prim::By)) = by0;
    w(I(Prim::Bz)) = bz0;
    w(I(Prim::Pres)) = rho0 / (ConstantsF64::u * eos.mass_per_h) * 2.0_fp * ConstantsF64::k_B * T0;
    prim_to_cons<Fluid>(eos.gamma, state.mu0, w, bg.background);

    bg.heating_coeff = get_or<fp_t>(config, "problem.background_heating_coeff", 1.0_fp);
    return bg;
}

template <typename Fluid>
static void initial_conditions(Simulation& sim, const YAML::Node& config) {
    using Prim = typename Fluid::prim;
    using Cons = typename Fluid::cons;
    constexpr int n_hydro = Fluid::num_vars;
    JasUnpack(sim, state, eos);
    JasUnpack(state, Q, sz);

    std::string restart_path = get_or<std::string>(config, "problem.restart_path", "xOxOplaceholderxOxO.nc");
    Fp4d Q0;
    yakl::SimpleNetCDF restart;
    restart.open(restart_path, yakl::NETCDF_MODE_READ);
    restart.read(Q0, "Q");
    f64 current_time = 0.0_fp;
    restart.read(current_time, "time");

    if (Q0.extent(0) != n_hydro) {
        throw std::runtime_error(fmt::format("Inconsistency between number of variables for fluid ({}) and in restart file ({})", Fluid::num_vars, Q0.extent(0)));
    }

    sim.out_cfg.prev_output_time = current_time;
    sim.time = current_time;
    const bool has_ion_e = eos.has_ion_e;
    const fp_t fully_ionised_specific_energy = chi_H / (h_mass * eos.mass_per_h);

    dex_parallel_for(
        FlatLoop<3>(sz.zc, sz.yc, sz.xc),
        KOKKOS_LAMBDA (int k, int j, int i) {
            for (int v = 0; v < Q0.extent(0); ++v) {
                state.Q(v, k, j, i) = Q0(v, k, j, i);
            }

            if (has_ion_e) {
                // The FI restart is initially fully ionised. Add its reservoir
                // only when the selected EOS evolves ionisation energy.
                Q(I(Cons::IonE), k, j, i) = fully_ionised_specific_energy;
                Q(I(Cons::Ene), k, j, i) += Q(I(Cons::Rho), k, j, i) * Q(I(Cons::IonE), k, j, i);
            } else {
                Q(I(Cons::IonE), k, j, i) = 0.0_fp;
            }
        }
    );
}

MOSSCAP_NEW_PROBLEM(pure_coronal_condensation_mhd_restart) {
    MOSSCAP_PROBLEM_PREAMBLE(pure_coronal_condensation_mhd_restart);
    if (sim.num_dim != num_dim) {
        throw std::runtime_error(fmt::format(
            "{} only handles {}d problems", PROBLEM_NAME, num_dim
        ));
    }

    if (!is_instance(sim.fluid_type, FluidType::GlmMhd)) {
        throw std::runtime_error("This restart expects a GLM based input and fluid");
    }

    FluidTraitsRt traits(sim.num_dim, sim.fluid_type);
    sim.setup_ics = [=](Simulation& sim) {
        invoke_fluid_traits(
            sim.num_dim,
            sim.fluid_type,
            [&] <typename FTraits> (FTraits) {
                initial_conditions<FTraits>(sim, config);
            }
        );
    };

    BackgroundParams background = invoke_fluid_traits(
        sim.num_dim,
        sim.fluid_type,
        [&]<typename FTraits>(FTraits) {
            return get_background_params<FTraits>(sim, config);
        }
    );

    setup_thin_loss(sim, config);
    sim.compute_source_terms.push_back(SourceTerm{
        .name = "background_heating",
        .fn = [=](const Simulation& sim) {
            invoke_fluid_traits(
                sim.num_dim,
                sim.fluid_type,
                [&]<typename FTraits>(FTraits) {
                    background_heating_kernel<FTraits>(sim, background);
                }
            );
        }
    });
    setup_replace_small_values(sim, config);
}

}
