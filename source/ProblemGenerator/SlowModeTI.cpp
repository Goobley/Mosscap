#include "ProblemGenerator.hpp"
#include "../Hydro.hpp"
#include "../MosscapConfig.hpp"
#include "../SourceTerms.hpp"
#include "../SourceTerms/Sponge.hpp"
#include "../SourceTerms/TownsendThinLoss.hpp"
#include "../AnalyticLteH.hpp"

// NOTE(cmo): This is a 2d problem
static constexpr int num_dim = 2;

namespace Mosscap {


struct BackgroundParams {
    fp_t T0;
    fp_t lambda_T0;
    fp_t heating_coeff;
    bool use_precomputed = false;
    yakl::SArray<fp_t, 1, N_HYDRO_VARS<3, FLUID_WITH_MAX_VARS>> background;
};

template <typename FTraits>
void background_heating_kernel(const Simulation& sim, const BackgroundParams& bg) {
    constexpr fp_t unit_numberdens = 1e15_fp;
    constexpr fp_t unit_rho = unit_numberdens * ConstantsF64::u;

    JasUnpack(sim, state, sources);
    JasUnpack(state, sz);
    const auto& S = sources.S;
    int nx = sz.xc - 2 * sz.ng;
    int ny = std::max(sz.yc - 2 * sz.ng, 1);
    int nz = std::max(sz.zc - 2 * sz.ng, 1);

    fp_t H = 0.0_fp;
    if (!bg.use_precomputed) {
        int source_idx = source_term_index(sim, "thin_loss");
        auto loss_ctx = (ThinLossContext*)sim.compute_source_terms[source_idx].get_context();
        H = -thin_loss_single_val<FTraits>(sim, *loss_ctx, bg.background);
    } else {
        H = square(bg.background(0) / unit_rho * unit_numberdens) * bg.lambda_T0;
    }

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
    bool analytic_pert = get_or<bool>(config, "problem.analytic_pert", false);
    if (analytic_pert) {
        const fp_t unit_length = state.get_axis_length(0);
        constexpr fp_t unit_temperature = 1e6_fp;
        constexpr fp_t unit_numberdens = 1e15_fp;
        constexpr fp_t unit_rho = unit_numberdens * ConstantsF64::u;
        constexpr fp_t unit_pres = 2.0 * unit_numberdens * ConstantsF64::k_B * unit_temperature;
        const fp_t unit_vel = std::sqrt(unit_pres / unit_rho);
        const fp_t unit_B = std::sqrt(state.mu0 * unit_pres);

        const fp_t rho0 = 1.0_fp * unit_rho;
        const fp_t p0 = 1.0_fp * unit_pres;
        const fp_t Bx0 = 0.0_fp;
        const fp_t By0 = 10e-4_fp;
        const fp_t vx0 = 1.0e5_fp;
        const fp_t vy0 = 1.0e5_fp;

        bg.T0 = unit_temperature;
        bg.lambda_T0 = 0.0_fp;
        yakl::SArray<fp_t, 1, n_hydro> w(0.0_fp);
        w(I(Prim::Rho)) = rho0;
        w(I(Prim::Vx)) = vx0;
        w(I(Prim::Vy)) = vy0;
        w(I(Prim::Bx)) = Bx0;
        w(I(Prim::By)) = By0;
        w(I(Prim::Pres)) = p0;
        prim_to_cons<Fluid>(eos.gamma, state.mu0, w, bg.background);
    } else {
        constexpr fp_t unit_numberdens = 1e15_fp;
        constexpr fp_t unit_rho = unit_numberdens * ConstantsF64::u;

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
        w(I(Prim::Pres)) = rho0 * unit_numberdens / unit_rho * 2.0_fp * ConstantsF64::k_B * T0;
        prim_to_cons<Fluid>(eos.gamma, state.mu0, w, bg.background);
    }

    bg.heating_coeff = get_or<fp_t>(config, "problem.background_heating_coeff", 1.0_fp);
    bg.use_precomputed = get_or<bool>(
        config,
        "problem.use_precomputed_background_heating",
        !analytic_pert
    );
    return bg;
}

template <typename Fluid>
static void initial_conditions(Simulation& sim, const YAML::Node& config) {
    using Cons = typename Fluid::cons;
    const auto& state = sim.state;
    const auto& sz = state.sz;
    const auto& eos = sim.eos;

    if (get_or<bool>(config, "problem.analytic_pert", false)) {
        const fp_t unit_length = state.get_axis_length(0);
        constexpr fp_t unit_temperature = 1e6_fp;
        constexpr fp_t unit_numberdens = 1e15_fp;
        constexpr fp_t unit_rho = unit_numberdens * ConstantsF64::u;
        constexpr fp_t unit_pres = 2.0 * unit_numberdens * ConstantsF64::k_B * unit_temperature;
        const fp_t unit_vel = std::sqrt(unit_pres / unit_rho);
        const fp_t unit_B = std::sqrt(state.mu0 * unit_pres);

        const fp_t rho0 = 1.0_fp * unit_rho;
        const fp_t p0 = 1.0_fp * unit_pres;
        const fp_t Bx0 = 0.0_fp;
        const fp_t By0 = 10e-4_fp;
        const fp_t vx0 = 1.0e5_fp;
        const fp_t vy0 = 1.0e5_fp;
        const fp_t amp = 0.1_fp;

        const fp_t angle = ConstantsF64::pi * 0.25_fp;
        const fp_t kx = 2.0_fp * ConstantsF64::pi / unit_length;
        const fp_t ky = kx / std::tan(angle);
        const fp_t k = std::sqrt(square(kx) + square(ky));

        const fp_t cs2 = eos.gamma * p0 / rho0;
        const fp_t va2 = square(By0) / (rho0 * state.mu0);
        const fp_t ws = k * std::sqrt(
            0.5_fp * (cs2 + va2) - 0.5_fp * std::sqrt(
                square(va2 + cs2) - 4.0_fp * (square(ky) / square(k)) * va2 * cs2
            )
        );
        const fp_t alphas = 1.0_fp - (square(k) * va2) / (square(ws));
        fmt::println("rho0 {}, p0 {}", rho0, p0);
        fmt::println("angle {}, kx {} ky {} k {}", angle, kx, ky, k);
        fmt::println("cs2 {}, va2 {} ws {} alphas {}", cs2, va2, ws, alphas);

        int nx = sz.xc - 2 * sz.ng;
        int ny = std::max(sz.yc - 2 * sz.ng, 1);
        int nz = std::max(sz.zc - 2 * sz.ng, 1);

        JasUnpack(state, Q);
        dex_parallel_for(
            FlatLoop<3>(nz, ny, nx),
            KOKKOS_LAMBDA (int ki, int ji, int ii) {
                const int k = nz == 1 ? ki : ki + sz.ng;
                const int j = ny == 1 ? ji : ji + sz.ng;
                const int i = ii + sz.ng;

                constexpr int n_hydro = Fluid::num_vars;
                using Prim = Fluid::prim;
                yakl::SArray<fp_t, 1, n_hydro> w(0.0_fp);
                w(I(Prim::Rho)) = rho0;
                const vec3 pos = state.get_pos(i, j, k);
                const fp_t vx_pert = amp * std::sin(kx * pos(0) + ky * pos(1));
                w(I(Prim::Vx)) = vx0 + vx_pert;
                w(I(Prim::Vy)) = vy0 + alphas * (ky / kx) * vx_pert;
                w(I(Prim::Bx)) = Bx0;
                w(I(Prim::By)) = By0 + (kx * std::sqrt(va2) / ws) * vx_pert * state.mu0;
                w(I(Prim::Pres)) = p0 + (alphas * ws / (kx * std::sqrt(cs2))) * vx_pert;
                CellIndex idx{
                    .i=i,
                    .j=j,
                    .k=k
                };
                prim_to_cons<Fluid>(eos.gamma, state.mu0, w, QtyView(Q, idx));
            }
        );
    } else {
        const std::string input_path = get_or<std::string>(config, "problem.ic_path", "slow_mode_ti.nc");
        yakl::SimpleNetCDF nc;
        nc.open(input_path, yakl::NETCDF_MODE_READ);

        Fp2d rho, momx, momy, momz, e_tot, bx, by, bz;
        nc.read(rho, "rho");
        nc.read(momx, "momx");
        nc.read(momy, "momy");
        nc.read(momz, "momz");
        nc.read(e_tot, "e_tot");
        nc.read(bx, "bx");
        nc.read(by, "by");
        nc.read(bz, "bz");

        int nx = sz.xc - 2 * sz.ng;
        int ny = std::max(sz.yc - 2 * sz.ng, 1);
        int nz = std::max(sz.zc - 2 * sz.ng, 1);

        if (ny != rho.extent(0) || nx != rho.extent(1)) {
            throw std::runtime_error(fmt::format("Mismatch between allocated size and array size in file ([{}, {}] vs [{}, {}])", ny, nx, rho.extent(0), rho.extent(1)));
        }

        JasUnpack(state, Q);
        dex_parallel_for(
            FlatLoop<3>(nz, ny, nx),
            KOKKOS_LAMBDA (int ki, int ji, int ii) {
                const int k = nz == 1 ? ki : ki + sz.ng;
                const int j = ny == 1 ? ji : ji + sz.ng;
                const int i = ii + sz.ng;

                Q(I(Cons::Rho), k, j, i) = rho(ji, ii);
                Q(I(Cons::MomX), k, j, i) = momx(ji, ii);
                Q(I(Cons::MomY), k, j, i) = momy(ji, ii);
                Q(I(Cons::MomZ), k, j, i) = momz(ji, ii);
                Q(I(Cons::Ene), k, j, i) = e_tot(ji, ii);
                Q(I(Cons::Bx), k, j, i) = bx(ji, ii);
                Q(I(Cons::By), k, j, i) = by(ji, ii);
                Q(I(Cons::Bz), k, j, i) = bz(ji, ii);
            }
        );
    }
    Kokkos::fence();
}

MOSSCAP_NEW_PROBLEM(slow_mode_ti) {
    MOSSCAP_PROBLEM_PREAMBLE(slow_mode_ti);
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
            throw std::runtime_error("Unsupported fluid type");
        }
    };

    BackgroundParams background = invoke_fluid_traits(
        sim.num_dim,
        sim.fluid_type,
        [&]<typename FTraits>(FTraits) {
            return get_background_params<FTraits>(sim, config);
        }
    );

    if (get_or<bool>(config, "problem.enable_thin_loss", true)) {
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
    }
}

}