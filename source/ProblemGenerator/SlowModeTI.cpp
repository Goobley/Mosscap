#include "ProblemGenerator.hpp"
#include "../Hydro.hpp"
#include "../MosscapConfig.hpp"
#include "../SourceTerms/Sponge.hpp"
#include "../SourceTerms/TownsendThinLoss.hpp"
#include "../AnalyticLteH.hpp"

// NOTE(cmo): This is a 2d problem
static constexpr int num_dim = 2;

namespace Mosscap {


struct BackgroundParams {
    fp_t rho0;
    fp_t T0;
    fp_t lambda_T0;
    fp_t bx0;
    fp_t by0;
    fp_t bz0;
};

template <typename FTraits>
void background_heating_kernel(const Simulation& sim, const BackgroundParams& bg) {
    constexpr fp_t unit_numberdens = 1e15_fp;
    const fp_t H = square(bg.rho0 * unit_numberdens) * bg.lambda_T0;

    JasUnpack(sim, state, sources);
    JasUnpack(state, sz);
    const auto& S = sources.S;
    int nx = sz.xc - 2 * sz.ng;
    int ny = std::max(sz.yc - 2 * sz.ng, 1);
    int nz = std::max(sz.zc - 2 * sz.ng, 1);

    dex_parallel_for(
        FlatLoop<3>(nz, ny, nx),
        KOKKOS_LAMBDA (int ki, int ji, int ii) {
            const int k = nz == 1 ? ki : ki + sz.ng;
            const int j = ny == 1 ? ji : ji + sz.ng;
            const int i = ii + sz.ng;
            using Cons = typename FTraits::cons;

            S(I(Cons::Ene), k, j, i) += H;
        }
    );
    Kokkos::fence();
}

template <typename Fluid>
static void initial_conditions(Simulation& sim, const YAML::Node& config) {
    using Cons = typename Fluid::cons;
    constexpr int n_hydro = Fluid::num_vars;
    const auto& state = sim.state;
    const auto& sz = state.sz;

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
    BackgroundParams background {
        .rho0 = rho0,
        .T0 = T0,
        .lambda_T0 = lambda_T0,
        .bx0 = bx0,
        .by0 = by0,
        .bz0 = bz0
    };

    if (get_or<bool>(config, "problem.enable_thin_loss", false)) {
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