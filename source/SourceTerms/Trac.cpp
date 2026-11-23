#include "Trac.hpp"
#include "../MosscapConfig.hpp"
#include "../SourceTerms.hpp"
#include "ConductionCommon.hpp"

namespace Mosscap {

template <typename FTraits>
static void trac_kernel(const Simulation& sim, bool field_aligned) {
    using Prim = typename FTraits::prim;
    using Cons = typename FTraits::cons;
    JasUnpack(sim, state, eos, sources);
    JasUnpack(state, W, sz, dx, cond);
    const auto& S = sources.S;
    const TracFields trac = sim.trac;
    const fp_t m_p = state.p_mass;
    const fp_t kappa0 = cond.kappa0;
    const bool spitzer = cond.spitzer;
    const fp_t inv_delta = 1.0_fp / cond.trac.delta;
    const fp_t t_min = cond.trac.t_min;
    const bool include_dex = trac.dex_net.initialized();

    dex_parallel_for(
        "TRAC",
        FlatLoop<3>(sz.zc, sz.yc, sz.xc),
        KOKKOS_LAMBDA (int k, int j, int i) {
            const bool use_field = field_aligned;
            const QtyView w(W, CellIndex{.i = i, .j = j, .k = k});
            const fp_t T = cell_temperature<FTraits>(m_p, eos, w);
            const fp_t kappa_sp = spitzer_kappa(kappa0, spitzer, T);
            const fp_t net = trac.net(k, j, i);
            const fp_t net_tot = include_dex ? net + trac.dex_net(k, j, i) : net;

            fp_t kappa = kappa_sp;
            fp_t s = 1.0_fp;
            if (T >= t_min && net_tot != 0.0_fp) {
                // NOTE(cmo): field-aligned grid spacing, from the simulated components of B only
                fp_t ds = dx;
                if constexpr (FTraits::is_mhd) {
                    if (use_field) {
                        fp_t b_sum = std::abs(w(I(Prim::Bx)));
                        fp_t b_sq = square(w(I(Prim::Bx)));
                        if constexpr (FTraits::num_dim > 1) {
                            b_sum += std::abs(w(I(Prim::By)));
                            b_sq += square(w(I(Prim::By)));
                        }
                        if constexpr (FTraits::num_dim > 2) {
                            b_sum += std::abs(w(I(Prim::Bz)));
                            b_sq += square(w(I(Prim::Bz)));
                        }
                        if (b_sq > 0.0_fp) {
                            ds = dx * b_sum / std::sqrt(b_sq);
                        }
                    }
                }
                const fp_t kappa_trac = ds * inv_delta * std::sqrt(kappa_sp * std::abs(net_tot) / std::max(T, 1e-30_fp));
                if (kappa_trac > kappa_sp) {
                    kappa = kappa_trac;
                    s = kappa_sp / kappa_trac;
                }
            }
            trac.kappa(k, j, i) = kappa;
            trac.scale(k, j, i) = s;
            S(I(Cons::Ene), k, j, i) += s * net;
            trac.net(k, j, i) = 0.0_fp;
        }
    );
    Kokkos::fence();
}

/// Wrap an EnergySourceSink term for TRAC. During a step the term writes into
/// sim.trac.source_scratch then its energy rate is added to sim.trac.net, and
/// other components propagate to S.  Evaluated alone (`out` set, e.g. for
/// output), its energy rate in `out` is multiplied by the stored scale.
template <typename FTraits>
static SourceFn wrap_trac_source(SourceFn inner) {
    using Cons = typename FTraits::cons;
    constexpr int n_hydro = FTraits::num_vars;

    return [=](const Simulation& sim, const Fp4d& out) {
        const TracFields trac = sim.trac;
        const auto& sz = sim.state.sz;
        const int ene = I(Cons::Ene);

        // NOTE(cmo): The wrapped source was called with a an argument, i.e. output.
        if (out.initialized()) {
            inner(sim, out);
            dex_parallel_for(
                "TRAC scale output",
                FlatLoop<3>(sz.zc, sz.yc, sz.xc),
                KOKKOS_LAMBDA (int k, int j, int i) {
                    out(ene, k, j, i) *= trac.scale(k, j, i);
                }
            );
            Kokkos::fence();
            return;
        }

        // NOTE(cmo): Default path
        const Fp4d& scratch = trac.source_scratch;
        inner(sim, scratch);
        const auto& S = sim.sources.S;
        dex_parallel_for(
            "TRAC collect",
            FlatLoop<3>(sz.zc, sz.yc, sz.xc),
            KOKKOS_LAMBDA (int k, int j, int i) {
                for (int v = 0; v < n_hydro; ++v) {
                    if (v == ene) {
                        trac.net(k, j, i) += scratch(v, k, j, i);
                    } else {
                        S(v, k, j, i) += scratch(v, k, j, i);
                    }
                    scratch(v, k, j, i) = 0.0_fp;
                }
            }
        );
        Kokkos::fence();
    };
}

void setup_trac(Simulation& sim, YAML::Node& config) {
    if (!sim.state.cond.trac.enable) {
        return;
    }

    // NOTE(cmo): Source term ordering now matters. Energy sources/sinks need to
    // be evaluated before TRAC, which needs to be evaluated before conduction.
    // Others can happen wherever -- we shift them to the end.
    auto& terms = sim.compute_source_terms;
    std::vector<SourceTerm> sources;
    std::vector<SourceTerm> conduction;
    std::vector<SourceTerm> others;
    for (const auto& term : terms) {
        const auto role = term.trac_role.role;
        if (role == TracRole::None) {
            others.push_back(term);
        } else if (role == TracRole::Conduction) {
            conduction.push_back(term);
        } else {
            sources.push_back(term);
        }
    }

    if (conduction.empty()) {
        throw std::runtime_error("conduction.trac.enable needs a thermal conduction source (classic or hyperbolic).");
    }
    if (sources.empty()) {
        fmt::println("TRAC: no optically thin loss or heating terms, so kappa_eff = kappa_sp everywhere.");
    }

    bool field_aligned = true;
    if (source_term_index(sim, "thermal_conduction") != terms.size()) {
        const auto* ctx = (ThermalConductionContext*)terms[source_term_index(sim, "thermal_conduction")].get_context();
        field_aligned = ctx->anisotropic;
    }

    const auto& sz = sim.state.sz;
    sim.trac.net = Fp3d("trac_net", sz.zc, sz.yc, sz.xc);
    sim.trac.kappa = Fp3d("trac_kappa", sz.zc, sz.yc, sz.xc);
    sim.trac.scale = Fp3d("trac_scale", sz.zc, sz.yc, sz.xc);
    sim.trac.source_scratch = Fp4d("trac_source_scratch", sim.sources.S.extent(0), sz.zc, sz.yc, sz.xc);
    sim.trac.net = 0.0_fp;
    sim.trac.kappa = 0.0_fp;
    sim.trac.scale = 1.0_fp;
    sim.trac.source_scratch = 0.0_fp;
    if (sim.state.cond.trac.include_dex) {
        if (!sim.dex.interface_config.enable) {
            throw std::runtime_error("conduction.trac.include_dex needs dex.enable.");
        }
        sim.trac.dex_net = Fp3d("trac_dex_net", sz.zc, sz.yc, sz.xc);
        sim.trac.dex_net = 0.0_fp;
    }

    SourceTerm trac_term{
        .name = "trac",
        .fn = invoke_fluid_traits(
            sim.num_dim,
            sim.fluid_type,
            [=]<typename FTraits>(FTraits) -> SourceFn {
                return [=] (const Simulation& sim, const Fp4d& out) {
                    // NOTE(cmo): Requesting TRAC as a diagnostic source term
                    // isn't really meaningful, and its fields are always
                    // written when enabled.
                    if (out.initialized()) {
                        return;
                    }
                    trac_kernel<FTraits>(sim, field_aligned);
                };
            }
        ),
        .trac_role = TracRole::None
    };

    // Wrap all the energy source terms
    for (auto& term : sources) {
        term.fn = invoke_fluid_traits(
            sim.num_dim,
            sim.fluid_type,
            [&]<typename FTraits>(FTraits) -> SourceFn {
                return wrap_trac_source<FTraits>(term.fn);
            }
        );
    }

    // Build the new source_terms [*energy_sources/sinks, trac, conduction, *others]
    std::vector<SourceTerm> ordered(sources.begin(), sources.end());
    ordered.push_back(trac_term);
    ordered.insert(ordered.end(), conduction.begin(), conduction.end());
    ordered.insert(ordered.end(), others.begin(), others.end());
    terms = std::move(ordered);

    std::string order;
    for (const auto& t : terms) {
        order += (order.empty() ? "" : ", ") + t.name;
    }
    fmt::println(
        "TRAC: delta = {}, t_min = {} K, {}{}; source terms: {}",
        sim.state.cond.trac.delta,
        sim.state.cond.trac.t_min,
        sim.state.cond.trac.include_dex
            ? "Dex losses included (lagged), Dex path lengths scaled by s, "
            : "",
        field_aligned ? "field-aligned ds" : "ds = dx",
        order
    );
}

}
