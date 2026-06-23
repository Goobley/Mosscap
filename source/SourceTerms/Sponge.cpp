#include "Sponge.hpp"
#include "../Simulation.hpp"
#include "../MosscapConfig.hpp"
#include "../SourceTerms.hpp"

// NOTE(cmo): Sacrificial sponge layers implemented as per Wilson 2016 (https://theses.gla.ac.uk/7209/)

namespace Mosscap {

template <typename FTraits>
void sponge_kernel(const Simulation& sim, const SpongeParams& sponge) {
    JasUnpack(sim, state);

    using Cons = typename FTraits::cons;

    const auto& Q = sim.state.Q;
    const auto& S = sim.sources.S;
    const auto& sz = sim.state.sz;
    const auto& dt = sim.dt;

    dex_parallel_for(
        "Apply sponge",
        FlatLoop<3>(sz.zc, sz.yc, sz.xc),
        KOKKOS_LAMBDA (int k, int j, int i) {
            vec3 pos = state.get_pos(i, j, k);
            bool in_x = pos(0) > sponge.xs && pos(0) < sponge.xe;
            bool in_y = true;
            bool in_z = true;
            if (FTraits::num_dim > 1) {
                in_y = pos(1) > sponge.ys && pos(1) < sponge.ye;
            }
            if (FTraits::num_dim > 2) {
                in_z = pos(2) > sponge.zs && pos(2) < sponge.ze;
            }

            if (in_x && in_y && in_z) {
                return;
            }
            fp_t sigma_x = 0.0_fp;
            fp_t sigma_y = 0.0_fp;
            fp_t sigma_z = 0.0_fp;
            fp_t max_sigma = 0.0_fp;
            const decltype(state.boundaries.xs_const) eq_state(0.0);
            if (!in_x) {
                if (pos(0) < sponge.xs) {
                    sigma_x = sponge.A * std::exp(sponge.B * std::abs(pos(0) - sponge.xs));
                } else {
                    sigma_x = sponge.A * std::exp(sponge.B * std::abs(pos(0) - sponge.xe));
                }
                if (sigma_x > max_sigma) {
                    max_sigma = sigma_x;
                    if (sponge.use_edge_vals) {
                        i32 x_idx = (pos(0) < sponge.xs) ? sz.ng : sz.xc - sz.ng - 1;
                        for (int v = 0; v < S.extent(0); ++v) {
                            eq_state(v) = Q(v, k, j, x_idx);
                        }
                    } else {
                        for (int v = 0; v < S.extent(0); ++v) {
                            eq_state(v) = (pos(0) < sponge.xs) ? state.boundaries.xs_const(v) : state.boundaries.xe_const(v);
                        }
                    }
                }
            }
            if (!in_y) {
                if (pos(1) < sponge.ys) {
                    sigma_y = sponge.A * std::exp(sponge.B * std::abs(pos(1) - sponge.ys));
                } else {
                    sigma_y = sponge.A * std::exp(sponge.B * std::abs(pos(1) - sponge.ye));
                }
                if (sigma_y > max_sigma) {
                    max_sigma = sigma_y;
                    if (sponge.use_edge_vals) {
                        i32 y_idx = (pos(1) < sponge.ys) ? sz.ng : sz.yc - sz.ng - 1;
                        for (int v = 0; v < S.extent(0); ++v) {
                            eq_state(v) = Q(v, k, y_idx, i);
                        }
                    } else {
                        for (int v = 0; v < S.extent(0); ++v) {
                            eq_state(v) = (pos(1) < sponge.ys) ? state.boundaries.ys_const(v) : state.boundaries.ye_const(v);
                        }
                    }
                }
            }
            if (!in_z) {
                if (pos(2) < sponge.zs) {
                    sigma_z = sponge.A * std::exp(sponge.B * std::abs(pos(2) - sponge.zs));
                } else {
                    sigma_z = sponge.A * std::exp(sponge.B * std::abs(pos(2) - sponge.ze));
                }
                if (sigma_z > max_sigma) {
                    max_sigma = sigma_z;
                    if (sponge.use_edge_vals) {
                        i32 z_idx = (pos(2) < sponge.zs) ? sz.ng : sz.zc - sz.ng - 1;
                        for (int v = 0; v < S.extent(0); ++v) {
                            eq_state(v) = Q(v, z_idx, j, i);
                        }
                    } else {
                        for (int v = 0; v < S.extent(0); ++v) {
                            eq_state(v) = (pos(2) < sponge.zs) ? state.boundaries.zs_const(v) : state.boundaries.ze_const(v);
                        }
                    }
                }
            }
            const fp_t sigma = std::min(std::max(sigma_x, std::max(sigma_y, sigma_z)), 1.0_fp);
            // if (sigma <= 0.0_fp) {
            //     return;
            // }

            if constexpr (is_instance(FTraits::fluid_type, FluidType::GlmMhd)) {
                if (sponge.damp_psi_to_zero) {
                    eq_state(I(Cons::Psi)) = 0.0_fp;
                }
                if (sponge.ignore_psi) {
                    eq_state(I(Cons::Psi)) = Q(I(Cons::Psi), k, j, i);
                }
            }

            const auto& Q0 = eq_state;
            for (int v = 0; v < S.extent(0); ++v) {
                S(v, k, j, i) += - sigma * (Q(v, k, j, i) - Q0(v)) / dt;
            }
        }
    );
    Kokkos::fence();


}

void setup_sponge(Simulation& sim, YAML::Node& config) {
    auto sponge = std::make_shared<SpongeParams>(SpongeParams{
        .A = get_or<fp_t>(config, "sources.sponge.A", 0.5_fp),
        .B = get_or<fp_t>(config, "sources.sponge.B", 0.05_fp),
        .xs = get_or<fp_t>(config, "sources.sponge.xs", 0.0_fp),
        .xe = get_or<fp_t>(config, "sources.sponge.xe", 1.0_fp),
        .ys = get_or<fp_t>(config, "sources.sponge.ys", 0.0_fp),
        .ye = get_or<fp_t>(config, "sources.sponge.ye", 1.0_fp),
        .zs = get_or<fp_t>(config, "sources.sponge.zs", 0.0_fp),
        .ze = get_or<fp_t>(config, "sources.sponge.ze", 1.0_fp),
        .use_edge_vals = get_or<bool>(config, "sources.sponge.use_edge_vals", true),
        .damp_psi_to_zero = get_or<bool>(config, "sources.sponge.damp_psi_to_zero", true),
        .ignore_psi = get_or<bool>(config, "sources.sponge.ignore_psi", false)
    });
    if (sponge->damp_psi_to_zero && sponge->ignore_psi) {
        throw std::runtime_error("Cannot set both damp_psi_to_zero and ignore_psi in Sponge.");
    }
    auto apply_sponge = invoke_fluid_traits(
        sim.num_dim,
        sim.fluid_type,
        [=]<typename FTraits>(FTraits) -> std::function<void(const Simulation&)> {
            return [=] (const Simulation& sim) {
                return sponge_kernel<FTraits>(sim, *sponge);
            };
        }
    );

    if (source_term_index(sim, "sponge") != sim.compute_source_terms.size()) {
        throw std::runtime_error("Source \"sponge\" already registered.");
    }

    sim.compute_source_terms.push_back(SourceTerm{
        .name = "sponge",
        .fn = apply_sponge,
        .get_context = [=]() { return sponge.get(); }
    });
}

}