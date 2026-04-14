#include "TownsendThinLoss.hpp"
#include "../Simulation.hpp"
#include "../MosscapConfig.hpp"
#include "../SourceTerms.hpp"

namespace Mosscap {

struct CoolingTable {
    i64 count;
    const f64* log_temp;
    const f64* log_lambda;
};

struct ThinLossContext {
    Fp1d temps;
    Fp1d lambdas;
    Fp1d Y_k;
    Fp1d alpha_k;
    fp_t min_temperature;
};

static constexpr f64 logt_simple[] = { 2.0, 4.45, 4.477, 5.0, 5.7, 6.0, 7.0, 8.0 };
static constexpr f64 loglambda_simple[] = {-39.5, -39.4, -35.819, -34.25, -34.6, -34.75, -35.25, -35.75 };
static constexpr f64 logt_DM[] = {
    2.0, 2.1, 2.2, 2.3, 2.4,
    2.5, 2.6, 2.7, 2.8, 2.9,
    3.0, 3.1, 3.2, 3.3, 3.4,
    3.5, 3.6, 3.7, 3.8, 3.9,
    4.0, 4.1, 4.2, 4.3, 4.4,
    4.5, 4.6, 4.7, 4.8, 4.9,
    5.0, 5.1, 5.2, 5.3, 5.4,
    5.5, 5.6, 5.7, 5.8, 5.9,
    6.0, 6.1, 6.2, 6.3, 6.4,
    6.5, 6.6, 6.7, 6.8, 6.9,
    7.0, 7.1, 7.2, 7.3, 7.4,
    7.5, 7.6, 7.7, 7.8, 7.9,
    8.0, 8.1, 8.2, 8.3, 8.4,
    8.5, 8.6, 8.7, 8.8, 8.9,
    9.0
};
static constexpr f64 loglambda_DM[] = {
    -39.523, -39.398, -39.301, -39.222, -39.097,
    -39.011, -38.936, -38.866, -38.807, -38.754,
    -38.708, -38.667, -38.63 , -38.595, -38.564,
    -38.534, -38.506, -38.479, -38.453, -38.429,
    -38.407, -36.019, -34.762, -34.742, -34.754,
    -34.73 , -34.523, -34.455, -34.314, -34.229,
    -34.163, -34.126, -34.092, -34.06 , -34.175,
    -34.28 , -34.39 , -34.547, -34.762, -35.05 ,
    -35.271, -35.521, -35.646, -35.66 , -35.676,
    -35.688, -35.69 , -35.662, -35.635, -35.609,
    -35.616, -35.646, -35.697, -35.74 , -35.788,
    -35.815, -35.785, -35.754, -35.728, -35.703,
    -35.68 , -35.63 , -35.58 , -35.53 , -35.48 ,
    -35.43 , -35.38 , -35.33 , -35.28 , -35.23 ,
    -35.18
};

static std::map<std::string, CoolingTable>&
get_cooling_tables() {
    static std::map<std::string, CoolingTable> tables = {
        {"simple", CoolingTable {
            .count = sizeof(logt_simple) / sizeof(logt_simple[0]),
            .log_temp = logt_simple,
            .log_lambda = loglambda_simple
        }},
        {"dm", CoolingTable {
            .count = sizeof(logt_DM) / sizeof(logt_DM[0]),
            .log_temp = logt_DM,
            .log_lambda = loglambda_DM
        }}
    };
    return tables;
}

template <typename FTraits>
void thin_loss_kernel(const Simulation& sim, const ThinLossContext& ctx) {
    using Cons = typename FTraits::cons;
    using Prim = typename FTraits::prim;
    constexpr int n_hydro = FTraits::num_vars;
    constexpr fp_t m_p = ConstantsF64::u;
    constexpr fp_t k_B = ConstantsF64::k_B;

    JasUnpack(sim, state, eos, dt_sub);
    JasUnpack(state, Q, sz, mu0);
    const auto& S = sim.sources.S;
    const int n_temps = ctx.temps.extent(0);
    const int n_bins = ctx.alpha_k.extent(0);

    dex_parallel_for(
        "Compute thin loss",
        FlatLoop<3>(sz.zc, sz.yc, sz.xc),
        KOKKOS_LAMBDA (int k, int j, int i) {
            yakl::SArray<fp_t, 1, n_hydro> w;
            CellIndex cell_idx{.i = i, .j = j, .k = k};
            const auto q = QtyView(Q, cell_idx);
            cons_to_prim<FTraits>(eos.gamma, mu0, q, w);

            // const bool do_print = (j == 256 && i == 256);
            const bool do_print = false;

            const fp_t nh_tot = w(I(Prim::Rho)) / (eos.avg_mass * m_p);
            fp_t y = eos.y;
            if (!eos.is_constant) {
                y = eos.y_space(cell_idx.k, cell_idx.j, cell_idx.i);
            }
            auto temperature = temperature_si(w(I(Prim::Pres)), nh_tot, y);
            fp_t ne = y * nh_tot;
            if (temperature < ctx.min_temperature) {
                return;
            }

            // Find temperature bin
            int idx = 0;
            while ((idx < n_bins - 1) && (ctx.temps(idx + 1) < temperature)) {
                idx += 1;
            }

            if (do_print) {
                printf("temperature: %f, idx: %d\n", temperature, idx);
            }

            const fp_t alpha_k_m1 = ctx.alpha_k(idx) - 1.0_fp;
            const fp_t tef = ctx.Y_k(idx) + (
                (ctx.lambdas(n_temps - 1) / ctx.lambdas(idx))
                * (ctx.temps(idx) / ctx.temps(n_temps - 1))
                * (std::pow(ctx.temps(idx) / temperature, alpha_k_m1) - 1.0) / alpha_k_m1
            );
            const fp_t tef_adj = (
                tef
                + ctx.lambdas(n_temps - 1) * dt_sub / ctx.temps(n_temps - 1)
                * (nh_tot * ne) / (nh_tot + ne) * (eos.gamma - 1.0_fp) / k_B
            );
            if (do_print) {
                printf("alpha_k_m1: %e, tef: %e, tef_adj: %e\n", alpha_k_m1, tef, tef_adj);
            }

            while ((idx > 0) && (tef_adj > ctx.Y_k(idx))) {
                idx -= 1;
            }
            if (do_print) {
                printf("Migrated idx: %d\n", idx);
            }

            fp_t new_temperature = ctx.temps(idx) * std::pow(
                (
                    1.0_fp - (1.0_fp - ctx.alpha_k(idx))
                    * (ctx.lambdas(idx) / ctx.lambdas(n_temps - 1))
                    * (ctx.temps(n_temps - 1) / ctx.temps(idx))
                    * (tef_adj - ctx.Y_k(idx))
                ),
                1.0_fp / (1.0_fp - ctx.alpha_k(idx))
            );
            new_temperature = std::max(new_temperature, ctx.min_temperature);
            const fp_t delta_temp = new_temperature - temperature;
            const fp_t delta_e = 1.0_fp / (eos.gamma - 1.0_fp) * (nh_tot + ne) * k_B * delta_temp;
            if (do_print) {
                printf("T' %e, dT: %e, dE %e\n", new_temperature, delta_temp, delta_e);
            }

            S(I(Cons::Ene), cell_idx.k, cell_idx.j, cell_idx.i) += delta_e / dt_sub;
        }
    );
    Kokkos::fence();
}

void setup_thin_loss(Simulation& sim, YAML::Node& config) {
    const bool enable = get_or<bool>(config, "sources.thin_loss.enable", false);
    if (!enable) {
        return;
    }

    std::string curve = get_or<std::string>(config, "sources.thin_loss.curve", "DM");
    std::transform(
        curve.begin(),
        curve.end(),
        curve.begin(),
        [](char c) { return std::tolower(c); }
    );

    const fp_t min_temperature = get_or<fp_t>(config, "sources.thin_loss.min_temperature", 5e2_fp);

    auto tables = get_cooling_tables();
    auto iter = tables.find(curve);
    if (iter == tables.end()) {
        throw std::runtime_error(fmt::format("No known thin loss table: {}", curve));
    }
    auto table = iter->second;

    i64 n_temps = table.count;
    i64 n_bins = n_temps - 1;

    auto temps = Fp1dHost("thin_temp_bin_edges", n_temps);
    auto lambdas = Fp1dHost("thin_lambda", n_temps);
    auto Y_k = Fp1dHost("thin_tef_bins", n_bins);
    auto alpha_k = Fp1dHost("thin_alpha", n_bins);

    for (int i = 0; i < n_temps; ++i) {
        temps(i) = std::pow(10.0_fp, table.log_temp[i]);
        lambdas(i) = std::pow(10.0_fp, table.log_lambda[i]);
    }
    for (int i = 0; i < n_bins; ++i) {
        alpha_k(i) = (table.log_lambda[i + 1] - table.log_lambda[i]) / (table.log_temp[i + 1] - table.log_temp[i]);
        if (alpha_k(i) == 1.0) {
            throw std::runtime_error("Special alpha=1 case for Townsend cooling curve not implemented");
        }
    }
    Y_k(n_bins - 1) = 0.0_fp;
    for (int i = n_bins - 2; i >= 0; --i) {
        const fp_t alpha_k_m1 = alpha_k(i) - 1.0_fp;
        const fp_t step = (
            (lambdas(n_bins) / lambdas(i)) *
            (temps(i) / temps(n_bins)) *
            (std::pow(temps(i) / temps(i+1), alpha_k_m1) - 1.0) / alpha_k_m1
        );
        Y_k(i) = Y_k(i+1) - step;
    }

    ThinLossContext ctx {
        .temps = temps.createDeviceCopy(),
        .lambdas = lambdas.createDeviceCopy(),
        .Y_k = Y_k.createDeviceCopy(),
        .alpha_k = alpha_k.createDeviceCopy(),
        .min_temperature = min_temperature
    };

    auto apply_thin_loss = invoke_fluid_traits(
        sim.num_dim,
        sim.fluid_type,
        [=]<typename FTraits>(FTraits) -> std::function<void(const Simulation&)> {
            return [=] (const Simulation& sim) {
                return thin_loss_kernel<FTraits>(sim, ctx);
            };
        }
    );

    if (source_term_index(sim, "thin_loss") != sim.compute_source_terms.size()) {
        throw std::runtime_error("Source \"thin_loss\" already registered.");
    }

    sim.compute_source_terms.push_back(SourceTerm{
        .name = "thin_loss",
        .fn = apply_thin_loss
    });
}

}