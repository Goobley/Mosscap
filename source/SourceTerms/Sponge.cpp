#include "Sponge.hpp"
#include "../Simulation.hpp"
#include "../MosscapConfig.hpp"
#include "../SourceTerms.hpp"

// NOTE(cmo): Sacrificial sponge layers implemented as per Wilson 2016 (https://theses.gla.ac.uk/7209/)

namespace Mosscap {

struct SpongeParams {
    /// Amplitude on exp
    fp_t A;
    /// Decay param in exp
    fp_t B;
    /// damp for x <= xs
    fp_t xs;
    /// damp for x >= xe
    fp_t xe;
    /// damp for y <= ys
    fp_t ys;
    /// damp for y >= ye
    fp_t ye;
    /// damp for z <= zs
    fp_t zs;
    /// damp for z >= ze
    fp_t ze;
};

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
            const decltype(state.boundaries.xs_const)* eq_state = nullptr;
            if (!in_x) {
                if (pos(0) < sponge.xs) {
                    sigma_x = sponge.A * std::exp(sponge.B * std::abs(pos(0) - sponge.xs));
                } else {
                    sigma_x = sponge.A * std::exp(sponge.B * std::abs(pos(0) - sponge.xe));
                }
                if (sigma_x > max_sigma) {
                    max_sigma = sigma_x;
                    eq_state = (pos(0) < sponge.xs) ? &state.boundaries.xs_const : &state.boundaries.xe_const;
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
                    eq_state = (pos(1) < sponge.ys) ? &state.boundaries.ys_const : &state.boundaries.ye_const;
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
                    eq_state = (pos(2) < sponge.zs) ? &state.boundaries.zs_const : &state.boundaries.ze_const;
                }
            }
            const fp_t sigma = std::max(sigma_x, std::max(sigma_y, sigma_z));
            if (!eq_state) {
                return;
            }

            const auto& Q0 = *eq_state;
            for (int v = 0; v < S.extent(0); ++v) {
                S(v, k, j, i) += - sigma * (Q(v, k, j, i) - Q0(v)) / dt;
            }
        }
    );
    Kokkos::fence();


}

void setup_sponge(Simulation& sim, YAML::Node& config) {
    SpongeParams sponge{
        .A = get_or<fp_t>(config, "sources.sponge.A", 0.5_fp),
        .B = get_or<fp_t>(config, "sources.sponge.B", 0.05_fp),
        .xs = get_or<fp_t>(config, "sources.sponge.xs", 0.0_fp),
        .xe = get_or<fp_t>(config, "sources.sponge.xe", 1.0_fp),
        .ys = get_or<fp_t>(config, "sources.sponge.ys", 0.0_fp),
        .ye = get_or<fp_t>(config, "sources.sponge.ye", 1.0_fp),
        .zs = get_or<fp_t>(config, "sources.sponge.zs", 0.0_fp),
        .ze = get_or<fp_t>(config, "sources.sponge.ze", 1.0_fp)
    };
    auto apply_sponge = invoke_fluid_traits(
        sim.num_dim,
        sim.fluid_type,
        [=]<typename FTraits>(FTraits) -> std::function<void(const Simulation&)> {
            return [=] (const Simulation& sim) {
                return sponge_kernel<FTraits>(sim, sponge);
            };
        }
    );

    if (source_term_index(sim, "sponge") != sim.compute_source_terms.size()) {
        throw std::runtime_error("Source \"sponge\" already registered.");
    }

    sim.compute_source_terms.push_back(SourceTerm{
        .name = "sponge",
        .fn = apply_sponge
    });
}

}