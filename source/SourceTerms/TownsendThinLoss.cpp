#include "TownsendThinLoss.hpp"
#include "../Simulation.hpp"
#include "../MosscapConfig.hpp"
#include "../SourceTerms.hpp"

#include <type_traits>
#include <vector>
#include <algorithm>

namespace Mosscap {

struct CoolingTable {
    i64 count;
    const f64* log_temp;
    const f64* log_lambda;
};


static constexpr f64 logt_simple[] = { 2.0, 4.45, 4.477, 5.0, 5.7, 6.0, 7.0, 8.0 };
static constexpr f64 loglambda_simple[] = {-39.5, -39.4, -35.819, -34.25, -34.6, -34.75, -35.25, -35.75 };
// Dalgarno & McCray (1972) curve from log T = 4.0 (dm = dm_coronal + dm_tail).
static constexpr f64 logt_dm_coronal[] = {
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
static constexpr f64 loglambda_dm_coronal[] = {
    -38.407, -36.019, -34.762, -34.742, -34.754,
    -34.73, -34.523, -34.455, -34.314, -34.229,
    -34.163, -34.126, -34.092, -34.06, -34.175,
    -34.28, -34.39, -34.547, -34.762, -35.05,
    -35.271, -35.521, -35.646, -35.66, -35.676,
    -35.688, -35.69, -35.662, -35.635, -35.609,
    -35.616, -35.646, -35.697, -35.74, -35.788,
    -35.815, -35.785, -35.754, -35.728, -35.703,
    -35.68, -35.63, -35.58, -35.53, -35.48,
    -35.43, -35.38, -35.33, -35.28, -35.23,
    -35.18
};

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

// Curves are now composed from a high-T base and the DM-inspired low-T tail
// below (sources.thin_loss.low_t), instead of storing every combination. The
// legacy merged names are aliases for those compositions

// DM tail (Dalgarno & McCray 1972, x = 1e-3 curve), log T 2.0-3.9.
static constexpr f64 logt_dm_tail[] = {
    2.0, 2.1, 2.2, 2.3, 2.4, 2.5,
    2.6, 2.7, 2.8, 2.9, 3.0, 3.1,
    3.2, 3.3, 3.4, 3.5, 3.6, 3.7,
    3.8, 3.9
};
static constexpr f64 loglambda_dm_tail[] = {
    -39.8742, -39.77355, -39.6619, -39.54245, -39.4229, -39.31015,
    -39.2078, -39.118, -39.0389, -38.96955, -38.9083, -38.8546,
    -38.8085, -38.771, -38.7409, -38.7182, -38.7005, -38.68445,
    -38.6641, -38.62025
};

// Colgan with digitised H and Ca line contributions removed between 1e4 and 1e5 K
// Shares logt_colgan.
static constexpr f64 loglambda_colgan_noHCa1e5[] = {
    -35.472108, -35.439479, -35.172253, -35.122568, -35.017579, -34.809261,
    -34.6015, -34.406582, -34.26633, -34.185057, -34.165886, -34.152725,
    -34.143716, -34.13693, -34.13361, -34.133432, -34.135866, -34.128024,
    -34.125478, -34.089648, -34.088124, -34.195424, -34.345823, -34.348393,
    -34.317007, -34.290722, -34.289003, -34.341045, -34.431224, -34.624483,
    -34.86694, -35.028975, -35.080509, -35.060571, -35.019733, -35.000004,
    -35.051611, -35.221755, -35.414517, -35.525813, -35.569135, -35.574857,
    -35.561505, -35.539689, -35.514904, -35.488959, -35.460711, -35.429084,
    -35.393586, -35.354568, -35.312614, -35.268274, -35.222037, -35.17423,
    -35.125141
};

// CHIANTI 11 bases from log T = 4.0 (the coronal ones with coronal abundances; noHCa1e5: H and Ca removed below
// 1e5 K). All share logt_chianti11.
static constexpr f64 logt_chianti11[] = {
    4.0, 4.1, 4.2, 4.3, 4.4, 4.5,
    4.6, 4.7, 4.8, 4.9, 5.0, 5.1,
    5.2, 5.3, 5.4, 5.5, 5.6, 5.7,
    5.8, 5.9, 6.0, 6.1, 6.2, 6.3,
    6.4, 6.5, 6.6, 6.7, 6.8, 6.9,
    7.0, 7.1, 7.2, 7.3, 7.4, 7.5,
    7.6, 7.7, 7.8, 7.9, 8.0, 8.1,
    8.2, 8.3, 8.4, 8.5, 8.6, 8.7,
    8.8, 8.9, 9.0
};
static constexpr f64 loglambda_chianti11[] = {
    -36.26906695, -35.30361639, -34.75892639, -34.83513426, -35.02858389, -35.07898128,
    -35.02696069, -34.88000427, -34.66164472, -34.47829362, -34.40492003, -34.43016752,
    -34.40600168, -34.37727264, -34.37891769, -34.5457062, -34.74213813, -34.78453547,
    -34.84629722, -34.90084789, -34.9094717, -34.93654124, -34.98205735, -35.09580173,
    -35.29146756, -35.45774517, -35.54569685, -35.57186923, -35.56762967, -35.55762895,
    -35.57009279, -35.63009141, -35.7221193, -35.78055512, -35.79600552, -35.7849314,
    -35.75911752, -35.72536635, -35.68773392, -35.64848599, -35.60833294, -35.566971,
    -35.52399746, -35.47941804, -35.43350434, -35.38658314, -35.33890719, -35.29067594,
    -35.24201771, -35.19303623, -35.14380958
};
static constexpr f64 loglambda_chianti11_noHCa1e5[] = {
    -37.02843522, -36.39514457, -36.00959474, -35.87757904, -35.66209001, -35.36218228,
    -35.15370977, -34.93300986, -34.68256871, -34.48798932, -34.41107203, -34.43016752,
    -34.40600168, -34.37727264, -34.37891769, -34.5457062, -34.74213813, -34.78453547,
    -34.84629722, -34.90084789, -34.9094717, -34.93654124, -34.98205735, -35.09580173,
    -35.29146756, -35.45774517, -35.54569685, -35.57186923, -35.56762967, -35.55762895,
    -35.57009279, -35.63009141, -35.7221193, -35.78055512, -35.79600552, -35.7849314,
    -35.75911752, -35.72536635, -35.68773392, -35.64848599, -35.60833294, -35.566971,
    -35.52399746, -35.47941804, -35.43350434, -35.38658314, -35.33890719, -35.29067594,
    -35.24201771, -35.19303623, -35.14380958
};
static constexpr f64 loglambda_chianti11coronal[] = {
    -36.09537918, -35.22831424, -34.7117957, -34.77534502, -34.94030406, -34.94170753,
    -34.86136473, -34.72754182, -34.53952609, -34.41806824, -34.39069605, -34.42120051,
    -34.39333852, -34.35740017, -34.34406315, -34.4608016, -34.55266747, -34.51289561,
    -34.48509404, -34.46620543, -34.45541916, -34.4843441, -34.53920551, -34.68049078,
    -34.92906841, -35.12676208, -35.21226191, -35.21849297, -35.19413193, -35.17251718,
    -35.1885802, -35.27721869, -35.42982002, -35.5552902, -35.62046845, -35.64090297,
    -35.63489039, -35.61479752, -35.58850777, -35.56058495, -35.53220355, -35.50212365,
    -35.46885006, -35.43200571, -35.39204344, -35.34967898, -35.30549697, -35.25996024,
    -35.21339401, -35.16603268, -35.11805825
};
static constexpr f64 loglambda_chianti11coronal_noHCa1e5[] = {
    -36.44575276, -35.8717271, -35.52649559, -35.4580363, -35.36777201, -35.1282759,
    -34.94379317, -34.76416964, -34.55522729, -34.42649619, -34.39664846, -34.42120051,
    -34.39333852, -34.35740017, -34.34406315, -34.4608016, -34.55266747, -34.51289561,
    -34.48509404, -34.46620543, -34.45541916, -34.4843441, -34.53920551, -34.68049078,
    -34.92906841, -35.12676208, -35.21226191, -35.21849297, -35.19413193, -35.17251718,
    -35.1885802, -35.27721869, -35.42982002, -35.5552902, -35.62046845, -35.64090297,
    -35.63489039, -35.61479752, -35.58850777, -35.56058495, -35.53220355, -35.50212365,
    -35.46885006, -35.43200571, -35.39204344, -35.34967898, -35.30549697, -35.25996024,
    -35.21339401, -35.16603268, -35.11805825
};

// Low-FIP element contribution for thin_loss_fip (full grid, log T 2.0-9.0).
static constexpr f64 logt_chianti11_lowfip[] = {
    2., 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3., 3.1, 3.2, 3.3, 3.4,
    3.5, 3.6, 3.7, 3.8, 3.9, 4., 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9,
    5., 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6., 6.1, 6.2, 6.3, 6.4,
    6.5, 6.6, 6.7, 6.8, 6.9, 7., 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9,
    8., 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.
};
static constexpr f64 loglambda_chianti11_lowfip[] = {
    -99., -99., -99., -99., -99., -99., -99., -99., -99., -99., -99., -99.,
    -99., -99., -99., -99., -99., -99., -99., -99., -36.91362952, -36.36266996,
    -36.03608964, -36.00351468, -36.01364048, -35.84549787, -35.69601425,
    -35.59135059, -35.48513245, -35.64076721, -36.21747233, -36.44572557,
    -36.26986257, -36.04205156, -35.79239541, -35.54670529, -35.33885379,
    -35.17976538, -35.06796174, -34.99986259, -34.97749576, -35.00744921,
    -35.06811436, -35.22518056, -35.50924503, -35.73137923, -35.81382513,
    -35.80379751, -35.76496011, -35.73614628, -35.75504667, -35.86468133,
    -36.07198034, -36.28027393, -36.43119828, -36.52334644, -36.57284044,
    -36.59742934, -36.61310991, -36.63249717, -36.6607824, -36.69466833,
    -36.72674222, -36.75141893, -36.76645211, -36.77208829, -36.76908962,
    -36.75848972, -36.7413107, -36.71827506, -36.69015572
};


#define MOSSCAP_TABLE(name, logt, loglambda) \
    {name, CoolingTable { .count = sizeof(logt) / sizeof(logt[0]), .log_temp = logt, .log_lambda = loglambda }}

static std::map<std::string, CoolingTable>&
get_cooling_tables() {
    static std::map<std::string, CoolingTable> tables = {
        MOSSCAP_TABLE("simple", logt_simple, loglambda_simple),
        MOSSCAP_TABLE("dm_coronal", logt_dm_coronal, loglambda_dm_coronal),
        MOSSCAP_TABLE("dm_tail", logt_dm_tail, loglambda_dm_tail),
        MOSSCAP_TABLE("colgan", logt_colgan, loglambda_colgan),
        MOSSCAP_TABLE("colgan_nohca1e5", logt_colgan, loglambda_colgan_noHCa1e5),
        MOSSCAP_TABLE("chianti11", logt_chianti11, loglambda_chianti11),
        MOSSCAP_TABLE("chianti11_nohca1e5", logt_chianti11, loglambda_chianti11_noHCa1e5),
        MOSSCAP_TABLE("chianti11coronal", logt_chianti11, loglambda_chianti11coronal),
        MOSSCAP_TABLE("chianti11coronal_nohca1e5", logt_chianti11, loglambda_chianti11coronal_noHCa1e5),
        MOSSCAP_TABLE("chianti11_lowfip", logt_chianti11_lowfip, loglambda_chianti11_lowfip),
    };
    return tables;
}
#undef MOSSCAP_TABLE

/// Legacy names: base + dm_tail (tail nodes below t_join, base nodes at or above it).
struct CurveAlias {
    const char* base;
    const char* tail;
    f64 tail_scale;
    f64 t_join;
};

static const std::map<std::string, CurveAlias>& get_curve_aliases() {
    static const std::map<std::string, CurveAlias> aliases = {
        {"dm", CurveAlias{.base = "dm_coronal", .tail = "dm_tail", .tail_scale = 1.0, .t_join = 1e4}},
        {"colgan_dm", CurveAlias{.base = "colgan", .tail = "dm_tail", .tail_scale = 1.0, .t_join = 1e4}},
        {"colgan_nohca1e5_dm", CurveAlias{.base = "colgan_nohca1e5", .tail = "dm_tail", .tail_scale = 1.0, .t_join = 1e4}},
        {"chianti11_dm", CurveAlias{.base = "chianti11", .tail = "dm_tail", .tail_scale = 1.0, .t_join = 1e4}},
        {"chianti11_nohca1e5_dm", CurveAlias{.base = "chianti11_nohca1e5", .tail = "dm_tail", .tail_scale = 1.0, .t_join = 1e4}},
        {"chianti11_nohca1e5_01dm", CurveAlias{.base = "chianti11_nohca1e5", .tail = "dm_tail", .tail_scale = 0.1, .t_join = 1e4}},
        {"chianti11coronal_dm", CurveAlias{.base = "chianti11coronal", .tail = "dm_tail", .tail_scale = 1.0, .t_join = 1e4}},
        {"chianti11coronal_nohca1e5_dm", CurveAlias{.base = "chianti11coronal_nohca1e5", .tail = "dm_tail", .tail_scale = 1.0, .t_join = 1e4}},
        {"chianti11coronal_nohca1e5_01dm", CurveAlias{.base = "chianti11coronal_nohca1e5", .tail = "dm_tail", .tail_scale = 0.1, .t_join = 1e4}},
    };
    return aliases;
}

static std::string lower(std::string name) {
    std::transform(name.begin(), name.end(), name.begin(), [](char c) { return std::tolower(c); });
    return name;
}

static CoolingTable load_cooling_table(const std::string& curve) {
    const auto& tables = get_cooling_tables();
    auto iter = tables.find(lower(curve));
    if (iter == tables.end()) {
        throw std::runtime_error(fmt::format("No known thin loss table: {}", curve));
    }
    return iter->second;
}

template <typename FTraits, typename Ctx>
void thin_loss_kernel(const Simulation& sim, const Ctx& ctx, const Fp4d& S) {
    using Cons = typename FTraits::cons;
    using Prim = typename FTraits::prim;
    constexpr int n_hydro = FTraits::num_vars;
    constexpr fp_t m_p = ConstantsF64::u;

    JasUnpack(sim, state, eos, dt_sub);
    JasUnpack(state, Q, sz, mu0);

    dex_parallel_for(
        "Compute thin loss",
        FlatLoop<3>(sz.zc, sz.yc, sz.xc),
        KOKKOS_LAMBDA (int k, int j, int i) {
            yakl::SArray<fp_t, 1, n_hydro> w;
            CellIndex cell_idx{.i = i, .j = j, .k = k};
            const auto q = QtyView(Q, cell_idx);
            cons_to_prim<FTraits>(eos.gamma, mu0, q, w);

            const fp_t nh_tot = w(I(Prim::Rho)) / (eos.mass_per_h * m_p);
            fp_t y = eos.y;
            if (!eos.is_constant) {
                y = eos.y_space(cell_idx.k, cell_idx.j, cell_idx.i);
            }
            auto temperature = temperature_si(w(I(Prim::Pres)), nh_tot, eos.total_abund, y);
            fp_t ne = y * nh_tot;

            JasUse(ctx, dt_sub);
            // NOTE(cmo): Under TRAC the exact integration is over the full dt_sub, and the resulting rate is
            // scaled linearly by the trac term (as Lare does).
            fp_t rate;
            if constexpr (std::is_same_v<Ctx, FipThinLossContext>) {
                const FipTownsendCurve curve{
                    .ctx = ctx,
                    .fip_bias = ctx.fip_bias(q, I(Cons::Rho))
                };
                rate = townsend_energy_rate(curve, temperature, nh_tot, ne, eos.gamma, dt_sub, ctx.min_temperature);
            } else {
                rate = townsend_energy_rate(ctx.curve, temperature, nh_tot, ne, eos.gamma, dt_sub, ctx.min_temperature);
            }
            S(I(Cons::Ene), cell_idx.k, cell_idx.j, cell_idx.i) += rate;
        }
    );
    Kokkos::fence();
}

/// Register a thin loss source term. Only one thin loss variant may be active,
/// otherwise the losses would be double counted.
template <typename Ctx>
static void register_thin_loss(Simulation& sim, const std::string& name, std::shared_ptr<Ctx> ctx) {
    for (const char* existing : {"thin_loss", "thin_loss_fip"}) {
        if (source_term_index(sim, existing) != sim.compute_source_terms.size()) {
            throw std::runtime_error(
                fmt::format("Cannot register \"{}\": thin loss source \"{}\" already registered.", name, existing)
            );
        }
    }

    sim.compute_source_terms.push_back(SourceTerm{
        .name = name,
        .fn = invoke_fluid_traits(
            sim.num_dim,
            sim.fluid_type,
            [=]<typename FTraits>(FTraits) -> SourceFn {
                return [=] (const Simulation& sim, const Fp4d& out) {
                    return thin_loss_kernel<FTraits>(sim, *ctx, source_target(sim, out));
                };
            }
        ),
        .get_context = [=]() { return ctx.get(); },
        .trac_role = TracRole::EnergySourceSink
    });
}

/// The thin-loss curve, possibly spliced from two registered tables:
///     sources:
///       thin_loss:
///         curve: chianti11       # base table, or a legacy alias (e.g. chianti11_DM = chianti11 + dm_tail)
///         scale: 1.0             # multiplies the base Λ
///         low_t:                 # optional low-temperature tail (not with an alias)
///           curve: dm_tail       # tail table
///           t_join: 1e4          # [K] tail nodes with T < t_join, base nodes with T >= t_join
///           scale: 1.0           # multiplies the tail Λ
/// Without low_t a table is used whole. The loss is n_e n_H Λ throughout.
struct ThinLossTable {
    std::vector<f64> log_t;
    std::vector<f64> log_l;
    /// present if called with low_fip_name. Sets 0 in the low temperature tail,
    /// and scaled with base_scale in the rest of the curve
    std::vector<f64> lambda_low_fip;
};

static ThinLossTable build_thin_loss_table(
    const YAML::Node& config,
    const std::string& prefix,
    const std::string& low_fip_name = ""
) {
    ThinLossTable out;
    auto& log_t = out.log_t;
    auto& log_l = out.log_l;
    const std::string curve_name = get_or<std::string>(config, prefix + ".curve", "DM");
    const auto& aliases = get_curve_aliases();
    const auto alias = aliases.find(lower(curve_name));
    const bool is_alias = alias != aliases.end();
    const bool has_low_t = has_key(config, prefix + ".low_t");
    if (is_alias && has_low_t) {
        throw std::runtime_error(fmt::format(
            "{}: curve \"{}\" already includes a low-T tail; use its base \"{}\" with low_t instead.",
            prefix, curve_name, alias->second.base
        ));
    }

    const auto base = load_cooling_table(is_alias ? std::string(alias->second.base) : curve_name);
    const f64 base_scale = get_or<f64>(config, prefix + ".scale", 1.0);
    if (!(base_scale > 0.0)) {
        throw std::runtime_error(fmt::format("{}.scale must be > 0.", prefix));
    }
    const f64 base_shift = std::log10(base_scale);

    auto fill_low_fip = [&](size_t first) {
        if (low_fip_name.empty()) {
            return;
        }
        const auto low_fip = load_cooling_table(low_fip_name);
        if (first == 0 && static_cast<i64>(log_t.size()) != low_fip.count) {
            throw std::runtime_error(fmt::format(
                "{}: curves \"{}\" and \"{}\" must share a temperature grid.", prefix, curve_name, low_fip_name
            ));
        }

        // Fill the low_t tail region with zeros
        out.lambda_low_fip.assign(log_t.size(), 0.0);
        for (size_t i = first; i < log_t.size(); ++i) {
            const auto* hit = std::find(low_fip.log_temp, low_fip.log_temp + low_fip.count, log_t[i]);
            if (hit == low_fip.log_temp + low_fip.count) {
                throw std::runtime_error(fmt::format(
                    "{}: low_fip_curve \"{}\" has no node at log T = {} of curve \"{}\".",
                    prefix, low_fip_name, log_t[i], curve_name
                ));
            }
            out.lambda_low_fip[i] = std::pow(10.0, low_fip.log_lambda[hit - low_fip.log_temp]) * base_scale;
        }
    };

    if (!is_alias && !has_low_t) {
        for (i64 i = 0; i < base.count; ++i) {
            log_t.push_back(base.log_temp[i]);
            log_l.push_back(base.log_lambda[i] + base_shift);
        }
        fill_low_fip(0);
        return out;
    }

    const auto tail = load_cooling_table(
        is_alias ? std::string(alias->second.tail) : get_or<std::string>(config, prefix + ".low_t.curve", "")
    );

    const f64 t_join = is_alias ? alias->second.t_join : get_or<f64>(config, prefix + ".low_t.t_join", 1e4);
    const f64 tail_scale = is_alias ? alias->second.tail_scale : get_or<f64>(config, prefix + ".low_t.scale", 1.0);

    if (!(t_join > 0.0) || !(tail_scale > 0.0)) {
        throw std::runtime_error(fmt::format("{}.low_t: t_join and scale must be > 0.", prefix));
    }

    const f64 log_join = std::log10(t_join);
    const f64 tail_shift = std::log10(tail_scale);
    for (i64 i = 0; i < tail.count; ++i) {
        if (tail.log_temp[i] < log_join) {
            log_t.push_back(tail.log_temp[i]);
            log_l.push_back(tail.log_lambda[i] + tail_shift);
        }
    }
    const size_t n_tail = log_t.size();
    for (i64 i = 0; i < base.count; ++i) {
        if (base.log_temp[i] >= log_join) {
            log_t.push_back(base.log_temp[i]);
            log_l.push_back(base.log_lambda[i] + base_shift);
        }
    }

    if (n_tail == 0 || n_tail == log_t.size()) {
        throw std::runtime_error(fmt::format(
            "{}.low_t: t_join = {} leaves no tail or no base nodes.", prefix, t_join
        ));
    }

    fmt::println(
        "{}: tail {} nodes (log T {:.3f}-{:.3f}, scale {}) spliced to base {} nodes (log T {:.3f}-{:.3f}, scale {})",
        prefix, n_tail, log_t.front(), log_t[n_tail - 1], tail_scale, log_t.size() - n_tail, log_t[n_tail], log_t.back(), base_scale
    );
    fill_low_fip(n_tail);
    return out;
}

void setup_thin_loss(Simulation& sim, YAML::Node& config) {
    const bool enable = get_or<bool>(config, "sources.thin_loss.enable", false);
    if (!enable) {
        return;
    }

    const ThinLossTable table = build_thin_loss_table(config, "sources.thin_loss");
    const auto& log_t = table.log_t;
    const auto& log_l = table.log_l;
    const fp_t min_temperature = get_or<fp_t>(config, "sources.thin_loss.min_temperature", 5e2_fp);

    i64 n_temps = log_t.size();
    i64 n_bins = n_temps - 1;

    TownsendCurve<Fp1dHost> curve{
        .temps = Fp1dHost("thin_temp_bin_edges", n_temps),
        .lambdas = Fp1dHost("thin_lambda", n_temps),
        .alpha_k = Fp1dHost("thin_alpha", n_bins)
    };

    for (int i = 0; i < n_temps; ++i) {
        curve.temps(i) = std::pow(10.0_fp, log_t[i]);
        curve.lambdas(i) = std::pow(10.0_fp, log_l[i]);
    }
    for (int i = 0; i < n_bins; ++i) {
        curve.alpha_k(i) = (log_l[i + 1] - log_l[i]) / (log_t[i + 1] - log_t[i]);
    }

    auto ctx = std::make_shared<ThinLossContext>(ThinLossContext{
        .curve = TownsendCurve<Fp1d>{
            .temps = curve.temps.createDeviceCopy(),
            .lambdas = curve.lambdas.createDeviceCopy(),
            .alpha_k = curve.alpha_k.createDeviceCopy()
        },
        .min_temperature = min_temperature
    });

    register_thin_loss(sim, "thin_loss", ctx);
}

/// Thin losses with a passive FIP bias tracer f advected with the plasma, using
/// Λ(T, f) = Λ_high(T) + f Λ_low(T). Config:
///     sources:
///       thin_loss_fip:
///         enable: true
///         curve: chianti11_dm                # total at FIP bias 1 (also scale, low_t: as sources.thin_loss)
///         low_fip_curve: <table>             # low-FIP element contribution at FIP bias 1
///         fip_bias_range: [1.0, 4.0]         # tracer clamp
///         tracer_index: 0                    # index into simulation.n_extra_fields
///         min_temperature: 500.0
/// Λ_high is the base curve minus the low-FIP curve. With low_t the low-FIP contribution is taken as zero on
/// the tail nodes (the legacy low-FIP table is ~1e-98 there) and is looked up node by node on the base nodes,
/// so low_fip_curve must contain every base node above t_join; the base scale multiplies it too. The tracer is stored like
/// any other: Q holds rho * fip_bias for the continuity equation.
void setup_thin_loss_fip(Simulation& sim, YAML::Node& config) {
    const bool enable = get_or<bool>(config, "sources.thin_loss_fip.enable", false);
    if (!enable) {
        return;
    }

    const std::string base_name = get_or<std::string>(config, "sources.thin_loss_fip.curve", "");
    const std::string low_fip_name = get_or<std::string>(config, "sources.thin_loss_fip.low_fip_curve", "");
    if (base_name.empty() || low_fip_name.empty()) {
        throw std::runtime_error("sources.thin_loss_fip requires both curve and low_fip_curve.");
    }
    const ThinLossTable base = build_thin_loss_table(config, "sources.thin_loss_fip", low_fip_name);

    fp_t min_fip_bias = 1.0_fp;
    fp_t max_fip_bias = 4.0_fp;
    if (config["sources"]["thin_loss_fip"]["fip_bias_range"]) {
        const auto range = config["sources"]["thin_loss_fip"]["fip_bias_range"];
        if (!range.IsSequence() || range.size() != 2) {
            throw std::runtime_error("sources.thin_loss_fip.fip_bias_range must be [min, max].");
        }
        min_fip_bias = range[0].as<fp_t>();
        max_fip_bias = range[1].as<fp_t>();
    }
    if (min_fip_bias < 0.0_fp || max_fip_bias < min_fip_bias) {
        throw std::runtime_error(fmt::format(
            "sources.thin_loss_fip.fip_bias_range = [{}, {}] is invalid.", min_fip_bias, max_fip_bias
        ));
    }
    const fp_t min_temperature = get_or<fp_t>(config, "sources.thin_loss_fip.min_temperature", 5e2_fp);

    // NOTE(claude): User tracers (simulation.n_extra_fields) sit directly after the
    // hydro variables; any dex tracers follow them.
    const int n_hydro = get_num_hydro_vars(sim.num_dim, sim.fluid_type);
    int n_user_tracers = sim.state.num_tracers;
    if (sim.dex.interface_config.enable && sim.dex.interface_config.advect) {
        n_user_tracers = sim.dex.interface_config.field_start_idx - n_hydro;
    }
    const int tracer_index = get_or<int>(config, "sources.thin_loss_fip.tracer_index", 0);
    if (tracer_index < 0 || tracer_index >= n_user_tracers) {
        throw std::runtime_error(fmt::format(
            "sources.thin_loss_fip.tracer_index = {}, but only {} tracer(s) allocated by simulation.n_extra_fields.",
            tracer_index,
            n_user_tracers
        ));
    }

    const i64 n_temps = base.log_t.size();
    const i64 n_bins = n_temps - 1;
    auto temps = Fp1dHost("thin_fip_temp_bin_edges", n_temps);
    auto lambdas_high_fip = Fp1dHost("thin_fip_lambda_high", n_temps);
    auto lambdas_low_fip = Fp1dHost("thin_fip_lambda_low", n_temps);
    auto inv_log_dtemp = Fp1dHost("thin_fip_inv_log_dtemp", n_bins);
    for (int i = 0; i < n_temps; ++i) {
        temps(i) = std::pow(10.0_fp, base.log_t[i]);
        const fp_t lambda_base = std::pow(10.0_fp, base.log_l[i]);
        lambdas_low_fip(i) = base.lambda_low_fip[i];
        lambdas_high_fip(i) = lambda_base - lambdas_low_fip(i);

        // NOTE(claude): Λ is linear in f, so positivity at both ends of the
        // clamp range guarantees it everywhere (α is taken from log Λ).
        for (fp_t f : {min_fip_bias, max_fip_bias}) {
            if (lambdas_high_fip(i) + f * lambdas_low_fip(i) <= 0.0_fp) {
                throw std::runtime_error(fmt::format(
                    "sources.thin_loss_fip: Λ ≤ 0 at log T = {} for FIP bias {} (curve \"{}\", low_fip_curve \"{}\").",
                    base.log_t[i],
                    f,
                    base_name,
                    low_fip_name
                ));
            }
        }
    }
    for (int i = 0; i < n_bins; ++i) {
        inv_log_dtemp(i) = 1.0_fp / (std::log(10.0_fp) * (base.log_t[i + 1] - base.log_t[i]));
    }

    auto ctx = std::make_shared<FipThinLossContext>(FipThinLossContext{
        .temps = temps.createDeviceCopy(),
        .lambdas_high_fip = lambdas_high_fip.createDeviceCopy(),
        .lambdas_low_fip = lambdas_low_fip.createDeviceCopy(),
        .inv_log_dtemp = inv_log_dtemp.createDeviceCopy(),
        .min_fip_bias = min_fip_bias,
        .max_fip_bias = max_fip_bias,
        .min_temperature = min_temperature,
        .tracer_idx = n_hydro + tracer_index
    });

    register_thin_loss(sim, "thin_loss_fip", ctx);
}

}
