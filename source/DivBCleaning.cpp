#include "DivBCleaning.hpp"
#include "Simulation.hpp"
#include "MosscapConfig.hpp"

namespace Mosscap {

void clean_divb(const Simulation& sim) {
    if (sim.clean_divb) {
        sim.clean_divb(sim);
    }
}

template <typename FTraits>
void linde_cleaning(const Simulation& sim) {
    constexpr fp_t divB_diff = 0.8_fp;
    const auto& state = sim.state;
    JasUnpack(state, sz, Q);
    Fp3d divB("divB", Q.extent(1), Q.extent(2), Q.extent(3));

    // NOTE(cmo): This loop needs to go into the ghost cells, but still requires
    // a 1 cell border.
    int nx = sz.xc - 2;
    int ny = std::max(sz.yc - 2, 1);
    int nz = std::max(sz.zc - 2, 1);
    const fp_t space_factor = 0.5_fp / state.dx;
    dex_parallel_for(
        "Compute divB",
        FlatLoop<3>(nz, ny, nx),
        KOKKOS_LAMBDA (int ki, int ji, int ii) {
            const int k = nz == 1 ? ki : ki + 1;
            const int j = ny == 1 ? ji : ji + 1;
            const int i = ii + 1;

            using Cons = FTraits::cons;
            fp_t div = (Q(I(Cons::Bx), k, j, i + 1) - Q(I(Cons::Bx), k, j, i - 1)) * space_factor;
            if constexpr (FTraits::num_dim > 1) {
                div += (Q(I(Cons::By), k, j + 1, i) - Q(I(Cons::By), k, j - 1, i)) * space_factor;
            }
            if constexpr (FTraits::num_dim > 2) {
                div += (Q(I(Cons::Bz), k + 1, j, i) - Q(I(Cons::Bz), k - 1, j, i)) * space_factor;
            }

            divB(k, j, i) = div;
        }
    );
    Kokkos::fence();

    const fp_t inv_mu0 = 1.0_fp / state.mu0;
    const fp_t eta_dt = divB_diff * square(sim.state.dx);
    dex_parallel_for(
        "Apply Linde source",
        FlatLoop<3>(nz, ny, nx),
        KOKKOS_LAMBDA (int ki, int ji, int ii) {
            const int k = nz == 1 ? ki : ki + 1;
            const int j = ny == 1 ? ji : ji + 1;
            const int i = ii + 1;
            using Cons = FTraits::cons;

            const fp_t grad_divB_x = (divB(k, j, i + 1) - divB(k, j, i - 1)) * space_factor;
            const fp_t Bx = Q(I(Cons::Bx), k, j, i);
            Q(I(Cons::Bx), k, j, i) += grad_divB_x * eta_dt;
            Q(I(Cons::Ene), k, j, i) += Bx * grad_divB_x * eta_dt * inv_mu0;
            if constexpr (FTraits::num_dim > 1) {
                const fp_t By = Q(I(Cons::By), k, j, i);
                const fp_t grad_divB_y = (divB(k, j + 1, i) - divB(k, j - 1, i)) * space_factor;
                Q(I(Cons::By), k, j, i) += grad_divB_y * eta_dt;
                Q(I(Cons::Ene), k, j, i) += By * grad_divB_y * eta_dt * inv_mu0;
            }
            if constexpr (FTraits::num_dim > 2) {
                const fp_t Bz = Q(I(Cons::Bz), k, j, i);
                const fp_t grad_divB_z = (divB(k + 1, j, i) - divB(k - 1, j, i)) * space_factor;
                Q(I(Cons::Bz), k, j, i) += grad_divB_z * eta_dt;
                Q(I(Cons::Ene), k, j, i) += Bz * grad_divB_z * eta_dt * inv_mu0;
            }
        }
    );
    Kokkos::fence();
}

void setup_divb_cleaning(Simulation& sim, YAML::Node& config) {
    if (sim.fluid_type == FluidType::Mhd) {
        if (sim.state.sz.ng < 2) {
            throw std::runtime_error("For divB cleaning, need at least 2 ghost cells");
        }
        bool do_cleaning = get_or<bool>(config, "simulation.divb_cleaning", true);
        if (!do_cleaning) {
            return;
        }
        if (sim.num_dim == 1) {
            sim.clean_divb = linde_cleaning<FluidTraits<1, FluidType::Mhd>>;
        } else if (sim.num_dim == 2) {
            sim.clean_divb = linde_cleaning<FluidTraits<2, FluidType::Mhd>>;
        } else if (sim.num_dim == 3) {
            sim.clean_divb = linde_cleaning<FluidTraits<3, FluidType::Mhd>>;
        }
    }
}

}