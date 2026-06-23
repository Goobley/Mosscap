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

// static constexpr f64 logt_DM_lowT[] = {
//     1.00, 1.04, 1.08, 1.12, 1.16, 1.20,
//     1.24, 1.28, 1.32, 1.36, 1.40,
//     1.44, 1.48, 1.52, 1.56, 1.60,
//     1.64, 1.68, 1.72, 1.76, 1.80,
//     1.84, 1.88, 1.92, 1.96, 2.00,
//     2.04, 2.08, 2.12, 2.16, 2.20,
//     2.24, 2.28, 2.32, 2.36, 2.40,
//     2.44, 2.48, 2.52, 2.56, 2.60,
//     2.64, 2.68, 2.72, 2.76, 2.80,
//     2.84, 2.88, 2.92, 2.96, 3.00,
//     3.04, 3.08, 3.12, 3.16, 3.20,
//     3.24, 3.28, 3.32, 3.36, 3.40,
//     3.44, 3.48, 3.52, 3.56, 3.60,
//     3.64, 3.68, 3.72, 3.76, 3.80,
//     3.84, 3.88, 3.92, 3.96, 4.00
// };

// static constexpr f64 loglambda_DM_lowT[] = {
//     -43.0377, -42.7062, -42.4055, -42.1331, -41.8864, -41.6631,
//     -41.4614, -41.2791, -41.1146, -40.9662, -40.833 , -40.7129,
//     -40.6052, -40.5088, -40.4225, -40.3454, -40.2767, -40.2153,
//     -40.1605, -40.1111, -40.0664, -40.0251, -39.9863, -39.9488,
//     -39.9119, -39.8742, -39.8353, -39.7948, -39.7523, -39.708 ,
//     -39.6619, -39.6146, -39.5666, -39.5183, -39.4702, -39.4229,
//     -39.3765, -39.3317, -39.2886, -39.2473, -39.2078, -39.1704,
//     -39.1348, -39.1012, -39.0692, -39.0389, -39.0101, -38.9825,
//     -38.9566, -38.9318, -38.9083, -38.8857, -38.8645, -38.8447,
//     -38.8259, -38.8085, -38.7926, -38.7778, -38.7642, -38.752 ,
//     -38.7409, -38.731 , -38.7222, -38.7142, -38.7071, -38.7005,
//     -38.6942, -38.6878, -38.6811, -38.6733, -38.6641, -38.6525,
//     -38.6325, -38.608 , -38.5367, -38.4806
// };


static constexpr f64 logt_colgan[] = {
    4.06460772, 4.14229559, 4.21995109, 4.29760733, 4.37527944, 4.45293587,
    4.53060946, 4.60826923, 4.68592974, 4.76359269, 4.79704583, 4.83049243,
    4.86394114, 4.89738514, 4.93083701, 4.96428321, 4.99773141, 5.03116600,
    5.06460772, 5.17574368, 5.28683805, 5.39795738, 5.50906805, 5.62017771,
    5.73129054, 5.84240328, 5.95351325, 6.06460772, 6.17574368, 6.28683805,
    6.39795738, 6.50906805, 6.62017771, 6.73129054, 6.84240328, 6.95351325,
    7.06460772, 7.17574368, 7.28683805, 7.39795738, 7.50906805, 7.62017771,
    7.73129054, 7.84240328, 7.95351325, 8.06460772, 8.17574368, 8.28683805,
    8.39795738, 8.50906805, 8.62017771, 8.73129054, 8.84240328, 8.95351325,
    9.06460772
};

static constexpr f64 loglambda_colgan[] = {
    -35.18883401, -34.78629635, -34.60383554, -34.68480662,
    -34.7644463 , -34.67935529, -34.54217864, -34.37958284,
    -34.25171892, -34.17584161, -34.15783402, -34.14491111,
    -34.13526945, -34.12837453, -34.12485189, -34.12438898,
    -34.12641785, -34.12802448, -34.1254776 , -34.08964778,
    -34.0881236 , -34.19542445, -34.34582346, -34.34839251,
    -34.31700703, -34.29072156, -34.28900309, -34.34104468,
    -34.43122351, -34.6244827 , -34.86694036, -35.02897478,
    -35.08050874, -35.06057061, -35.01973295, -35.00000434,
    -35.05161149, -35.22175466, -35.41451671, -35.52581288,
    -35.56913516, -35.57485721, -35.56150512, -35.53968863,
    -35.5149035 , -35.48895932, -35.46071057, -35.42908363,
    -35.39358639, -35.35456791, -35.31261375, -35.26827428,
    -35.22203698, -35.17422996, -35.12514145
};

static constexpr f64 logt_colgan_DM[] = {
    1.00, 1.04, 1.08, 1.12, 1.16, 1.20,
    1.24, 1.28, 1.32, 1.36, 1.40,
    1.44, 1.48, 1.52, 1.56, 1.60,
    1.64, 1.68, 1.72, 1.76, 1.80,
    1.84, 1.88, 1.92, 1.96, 2.00,
    2.04, 2.08, 2.12, 2.16, 2.20,
    2.24, 2.28, 2.32, 2.36, 2.40,
    2.44, 2.48, 2.52, 2.56, 2.60,
    2.64, 2.68, 2.72, 2.76, 2.80,
    2.84, 2.88, 2.92, 2.96, 3.00,
    3.04, 3.08, 3.12, 3.16, 3.20,
    3.24, 3.28, 3.32, 3.36, 3.40,
    3.44, 3.48, 3.52, 3.56, 3.60,
    3.64, 3.68, 3.72, 3.76, 3.80,
    3.84, 3.88, 3.92, 3.96, 4.00,
    4.06460772, 4.14229559, 4.21995109, 4.29760733, 4.37527944, 4.45293587,
    4.53060946, 4.60826923, 4.68592974, 4.76359269, 4.79704583, 4.83049243,
    4.86394114, 4.89738514, 4.93083701, 4.96428321, 4.99773141, 5.03116600,
    5.06460772, 5.17574368, 5.28683805, 5.39795738, 5.50906805, 5.62017771,
    5.73129054, 5.84240328, 5.95351325, 6.06460772, 6.17574368, 6.28683805,
    6.39795738, 6.50906805, 6.62017771, 6.73129054, 6.84240328, 6.95351325,
    7.06460772, 7.17574368, 7.28683805, 7.39795738, 7.50906805, 7.62017771,
    7.73129054, 7.84240328, 7.95351325, 8.06460772, 8.17574368, 8.28683805,
    8.39795738, 8.50906805, 8.62017771, 8.73129054, 8.84240328, 8.95351325,
    9.06460772
};

static constexpr f64 loglambda_colgan_DM[] = {
    -43.0377, -42.7062, -42.4055, -42.1331, -41.8864, -41.6631,
    -41.4614, -41.2791, -41.1146, -40.9662, -40.833 , -40.7129,
    -40.6052, -40.5088, -40.4225, -40.3454, -40.2767, -40.2153,
    -40.1605, -40.1111, -40.0664, -40.0251, -39.9863, -39.9488,
    -39.9119, -39.8742, -39.8353, -39.7948, -39.7523, -39.708 ,
    -39.6619, -39.6146, -39.5666, -39.5183, -39.4702, -39.4229,
    -39.3765, -39.3317, -39.2886, -39.2473, -39.2078, -39.1704,
    -39.1348, -39.1012, -39.0692, -39.0389, -39.0101, -38.9825,
    -38.9566, -38.9318, -38.9083, -38.8857, -38.8645, -38.8447,
    -38.8259, -38.8085, -38.7926, -38.7778, -38.7642, -38.752 ,
    -38.7409, -38.731 , -38.7222, -38.7142, -38.7071, -38.7005,
    -38.6942, -38.6878, -38.6811, -38.6733, -38.6641, -38.6525,
    -38.6325, -38.608 , -38.5367, -38.4806,
    -35.18883401, -34.78629635, -34.60383554, -34.68480662,
    -34.7644463 , -34.67935529, -34.54217864, -34.37958284,
    -34.25171892, -34.17584161, -34.15783402, -34.14491111,
    -34.13526945, -34.12837453, -34.12485189, -34.12438898,
    -34.12641785, -34.12802448, -34.1254776 , -34.08964778,
    -34.0881236 , -34.19542445, -34.34582346, -34.34839251,
    -34.31700703, -34.29072156, -34.28900309, -34.34104468,
    -34.43122351, -34.6244827 , -34.86694036, -35.02897478,
    -35.08050874, -35.06057061, -35.01973295, -35.00000434,
    -35.05161149, -35.22175466, -35.41451671, -35.52581288,
    -35.56913516, -35.57485721, -35.56150512, -35.53968863,
    -35.5149035 , -35.48895932, -35.46071057, -35.42908363,
    -35.39358639, -35.35456791, -35.31261375, -35.26827428,
    -35.22203698, -35.17422996, -35.12514145
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
        }},
        {"colgan", CoolingTable {
            .count = sizeof(logt_colgan) / sizeof(logt_colgan[0]),
            .log_temp = logt_colgan,
            .log_lambda = loglambda_colgan
        }},
        {"colgan_dm", CoolingTable {
            .count = sizeof(logt_colgan_DM) / sizeof(logt_colgan_DM[0]),
            .log_temp = logt_colgan_DM,
            .log_lambda = loglambda_colgan_DM
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

    auto ctx = std::make_shared<ThinLossContext>(ThinLossContext{
        .temps = temps.createDeviceCopy(),
        .lambdas = lambdas.createDeviceCopy(),
        .Y_k = Y_k.createDeviceCopy(),
        .alpha_k = alpha_k.createDeviceCopy(),
        .min_temperature = min_temperature
    });


    auto apply_thin_loss = invoke_fluid_traits(
        sim.num_dim,
        sim.fluid_type,
        [=]<typename FTraits>(FTraits) -> std::function<void(const Simulation&)> {
            return [=] (const Simulation& sim) {
                return thin_loss_kernel<FTraits>(sim, *ctx);
            };
        }
    );

    if (source_term_index(sim, "thin_loss") != sim.compute_source_terms.size()) {
        throw std::runtime_error("Source \"thin_loss\" already registered.");
    }

    sim.compute_source_terms.push_back(SourceTerm{
        .name = "thin_loss",
        .fn = apply_thin_loss,
        .get_context = [=]() { return ctx.get(); }
    });
}

}