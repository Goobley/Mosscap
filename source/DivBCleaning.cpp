#include "DivBCleaning.hpp"
#include "Simulation.hpp"
#include "MosscapConfig.hpp"
#include "SourceTerms.hpp"
#include "Boundaries.hpp"

namespace Mosscap {

// NOTE(cmo): Neither Linde nor Projection methods really work

void clean_divb(const Simulation& sim) {
    // NOTE(cmo): Leaving this in case we decide to have other methods that need
    // it, such as projection.
    if (sim.clean_divb) {
        sim.clean_divb(sim);
        fill_bcs(sim);
    }
}

template <typename FTraits, bool extended>
void glm_source(const Simulation& sim, fp_t glm_alpha) {
    const auto& state = sim.state;
    JasUnpack(state, sz, Q, mu0);
    const auto& S = sim.sources.S;
    int nx = sz.xc - 2;
    int ny = std::max(sz.yc - 2, 1);
    int nz = std::max(sz.zc - 2, 1);

    const fp_t c_h = sim.state.glm_ch;
    const fp_t dt_sub = sim.dt_sub;
    const fp_t inv_dt_sub = 1.0_fp / sim.dt_sub;
    const fp_t coeff = std::exp(-glm_alpha * c_h * dt_sub / state.dx);
    const fp_t inv_2dx = 1.0_fp / (2.0_fp * state.dx);
    const fp_t inv_mu0 = 1.0_fp / mu0;

    dex_parallel_for(
        "GLM MHD Source",
        FlatLoop<3>(nz, ny, nx),
        KOKKOS_LAMBDA (int ki, int ji, int ii) {
            const int k = nz == 1 ? ki : ki + 1;
            const int j = ny == 1 ? ji : ji + 1;
            const int i = ii + 1;
            using Cons = typename FTraits::cons;

            // Q(I(Prim::Psi), k, j, i) *= coeff;
            // NOTE(cmo): Implement as source term, even though it's technically multiplicative damping
            const fp_t psi_target = Q(I(Cons::Psi), k, j, i) * coeff;
            S(I(Cons::Psi), k, j, i) += (psi_target - Q(I(Cons::Psi), k, j, i)) * inv_dt_sub;

            JasUse(inv_2dx, inv_mu0);
            if constexpr (extended) {
                fp_t Bim1 = Q(I(Cons::Bx), k, j, i-1);
                fp_t Bi = Q(I(Cons::Bx), k, j, i+1);
                fp_t divB_i = (Bi - Bim1) * inv_2dx;
                if constexpr (FTraits::num_dim > 1) {
                    Bim1 = Q(I(Cons::By), k, j-1, i);
                    Bi = Q(I(Cons::By), k, j+1, i);
                    divB_i += (Bi - Bim1) * inv_2dx;
                }
                if constexpr (FTraits::num_dim > 2) {
                    Bim1 = Q(I(Cons::Bz), k-1, j, i);
                    Bi = Q(I(Cons::Bz), k+1, j, i);
                    divB_i += (Bi - Bim1) * inv_2dx;
                }
                S(I(Cons::MomX), k, j, i) -= divB_i * Q(I(Cons::Bx), k, j, i) * inv_mu0;
                if constexpr (FTraits::num_dim > 1) {
                    S(I(Cons::MomY), k, j, i) -= divB_i * Q(I(Cons::By), k, j, i) * inv_mu0;
                }
                if constexpr (FTraits::num_dim > 2) {
                    S(I(Cons::MomZ), k, j, i) -= divB_i * Q(I(Cons::Bz), k, j, i) * inv_mu0;
                }

                fp_t grad_psi = (Q(I(Cons::Psi), k, j, i + 1) - Q(I(Cons::Psi), k, j, i - 1)) * inv_2dx;
                S(I(Cons::Ene), k, j, i) -= Q(I(Cons::Bx), k, j, i) * grad_psi * inv_mu0;
                if constexpr (FTraits::num_dim > 1) {
                    grad_psi = (Q(I(Cons::Psi), k, j + 1, i) - Q(I(Cons::Psi), k, j - 1, i)) * inv_2dx;
                    S(I(Cons::Ene), k, j, i) -= Q(I(Cons::By), k, j, i) * grad_psi * inv_mu0;
                }
                if constexpr (FTraits::num_dim > 2) {
                    grad_psi = (Q(I(Cons::Psi), k + 1, j, i) - Q(I(Cons::Psi), k - 1, j, i)) * inv_2dx;
                    S(I(Cons::Ene), k, j, i) -= Q(I(Cons::Bz), k, j, i) * grad_psi * inv_mu0;
                }
            }
        }
    );
}

template <typename FTraits>
Fp3d cg_poisson(const Simulation& sim, Fp3d divB, int max_iters) {
    const auto& state = sim.state;
    JasUnpack(state, sz);
    int nx = sz.xc - 4;
    int ny = std::max(sz.yc - 4, 1);
    int nz = std::max(sz.zc - 4, 1);
    Fp3d phi("phi", divB.extent(0), divB.extent(1), divB.extent(2));
    phi = 0.0_fp;

    auto dot = [nx, ny, nz](const Fp3d& a, const Fp3d& b) {
        fp_t result;
        dex_parallel_reduce(
            "dot",
            FlatLoop<3>(nz, ny, nx),
            KOKKOS_LAMBDA (int ki, int ji, int ii, fp_t& running_dot) {
                const int k = nz == 1 ? ki : ki + 2;
                const int j = ny == 1 ? ji : ji + 2;
                const int i = ii + 2;

                running_dot += a(k, j, i) * b(k, j, i);
            },
            Kokkos::Sum<fp_t>(result)
        );
        return result;
    };

    Fp3d resid("residual", divB.extent(0), divB.extent(1), divB.extent(2));
    divB.deep_copy_to(resid);
    Fp3d p("search dir", divB.extent(0), divB.extent(1), divB.extent(2));
    divB.deep_copy_to(p);
    Fp3d Ap("A*p", divB.extent(0), divB.extent(1), divB.extent(2));
    Kokkos::fence();
    fp_t norm_resid_old = dot(resid, resid);

    int iter;
    for (iter = 0; iter < max_iters; ++iter) {
        dex_parallel_for(
            "A * p",
            FlatLoop<3>(nz, ny, nx),
            KOKKOS_LAMBDA (int ki, int ji, int ii) {
                const int k = nz == 1 ? ki : ki + 2;
                const int j = ny == 1 ? ji : ji + 2;
                const int i = ii + 2;

                fp_t sum = p(k, j, i + 1) + p(k, j, i - 1);
                if constexpr (FTraits::num_dim > 1) {
                    sum += p(k, j + 1, i) + p(k, j - 1, i);
                }
                if constexpr (FTraits::num_dim > 2) {
                    sum += p(k + 1, j, i) + p(k - 1, j, i);
                }
                sum -= 2.0_fp * FTraits::num_dim * p(k, j, i);
                sum /= (2.0 * square(state.dx));
                Ap(k, j, i) = sum;
            }
        );
        Kokkos::fence();

        fp_t alpha = norm_resid_old / dot(p, Ap);
        dex_parallel_for(
            "cg update",
            FlatLoop<3>(nz, ny, nx),
            KOKKOS_LAMBDA (int ki, int ji, int ii) {
                const int k = nz == 1 ? ki : ki + 2;
                const int j = ny == 1 ? ji : ji + 2;
                const int i = ii + 2;
                phi(k, j, i) += alpha * p(k, j, i);
                resid(k, j, i) -= alpha * Ap(k, j, i);
            }
        );
        Kokkos::fence();
        fp_t norm_resid_new = dot(resid, resid);
        const fp_t resid_ratio = norm_resid_new / norm_resid_old;
        dex_parallel_for(
            "cg update search dir",
            FlatLoop<3>(nz, ny, nx),
            KOKKOS_LAMBDA (int ki, int ji, int ii) {
                const int k = nz == 1 ? ki : ki + 2;
                const int j = ny == 1 ? ji : ji + 2;
                const int i = ii + 2;
                p(k, j, i) = resid(k, j, i) + resid_ratio * p(k, j, i);
            }
        );
        Kokkos::fence();
        norm_resid_old = norm_resid_new;
        if (std::sqrt(norm_resid_old) < 1e-1_fp) {
            break;
        }
    }
    return phi;
}


template <typename FTraits, int Order = 1>
void apply_grad_phi(const Simulation& sim, Fp3d phi) {
    static_assert(Order == 1 || Order == 2, "Gradient only set up for first or second order");
    const auto& state = sim.state;
    JasUnpack(state, sz, Q);
    int nx = sz.xc - 2 * sz.ng;
    int ny = std::max(sz.yc - 2 * sz.ng, 1);
    int nz = std::max(sz.zc - 2 * sz.ng, 1);
    fp_t space_factor = 1.0_fp / (2.0_fp * state.dx );
    if constexpr (Order == 2) {
        space_factor = 1.0_fp / state.dx;
    }
    const fp_t mu0 = sim.state.mu0;

    dex_parallel_for(
        "B update",
        FlatLoop<3>(nz, ny, nx),
        KOKKOS_LAMBDA (int ki, int ji, int ii) {
            using Cons = typename FTraits::cons;
            const int k = nz == 1 ? ki : ki + sz.ng;
            const int j = ny == 1 ? ji : ji + sz.ng;
            const int i = ii + sz.ng;
            fp_t grad_phi_x = (phi(k, j, i + 1) - phi(k, j, i - 1)) * space_factor;
            if constexpr (Order == 2) {
                grad_phi_x *= (2.0_fp / 3.0_fp);
                grad_phi_x -= ((phi(k, j, i + 2) - phi(k, j, i - 2))) * (1.0_fp / 12.0_fp) * space_factor;
            }
            fp_t prev_e_mag = square(Q(I(Cons::Bx), k, j, i));
            Q(I(Cons::Bx), k, j, i) -= grad_phi_x;

            if constexpr (FTraits::num_dim > 1) {
                fp_t grad_phi_y = (phi(k, j + 1, i) - phi(k, j - 1, i)) * space_factor;
                if constexpr (Order == 2) {
                    grad_phi_y *= (2.0_fp / 3.0_fp);
                    grad_phi_y -= ((phi(k, j + 2, i) - phi(k, j - 2, i))) * (1.0_fp / 12.0_fp) * space_factor;
                }
                prev_e_mag += square(Q(I(Cons::By), k, j, i));
                Q(I(Cons::By), k, j, i) -= grad_phi_y;
            }
            if constexpr (FTraits::num_dim > 2) {
                fp_t grad_phi_z = (phi(k + 1, j, i) - phi(k - 1, j, i)) * space_factor;
                if constexpr (Order == 2) {
                    grad_phi_z *= (2.0_fp / 3.0_fp);
                    grad_phi_z -= ((phi(k + 2, j, i) - phi(k - 2, j, i))) * (1.0_fp / 12.0_fp) * space_factor;
                }
                prev_e_mag += square(Q(I(Cons::Bz), k, j, i));
                Q(I(Cons::Bz), k, j, i) -= grad_phi_z;
            }
            prev_e_mag /= (2.0_fp * mu0);
            const fp_t new_e_mag = (square(Q(I(Cons::Bx), k, j, i)) + square(Q(I(Cons::By), k, j, i)) + square(Q(I(Cons::Bz), k, j, i))) / (2.0_fp * mu0);
            Q(I(Cons::Ene), k, j, i) += new_e_mag - prev_e_mag;
        }
    );
    Kokkos::fence();
}

template <typename FTraits, int Order=1>
void magnetic_field_projection_gs(const Simulation& sim) {
    int num_rb_iter = 5000;

    fp_t max_divb = 0.0_fp;
    Fp3d divB = compute_divb_impl<FTraits, Order>(sim, &max_divb);

    // Fp3d phi = rbgs_poisson<FTraits>(sim, divB, num_rb_iter);
    // Fp3d phi = jacobi_poisson<FTraits>(sim, divB, num_rb_iter);
    Fp3d phi = cg_poisson<FTraits>(sim, divB, num_rb_iter);
    apply_grad_phi<FTraits, Order>(sim, phi);

    fill_bcs(sim);
    divB = compute_divb_impl<FTraits, Order>(sim, &max_divb);
}

template <typename FTraits, int Order = 1>
Fp3d compute_divb_impl(const Simulation& sim, fp_t* max_divb_out=nullptr) {
    const auto& state = sim.state;
    JasUnpack(state, sz, Q);
    Fp3d divB("divB", Q.extent(1), Q.extent(2), Q.extent(3));
    divB = 0.0_fp;

    static_assert(Order == 1 || Order == 2, "Only implemented for first and second order central differences");

    // NOTE(cmo): This loop needs to go into the ghost cells, but still requires
    // a 1 cell border.
    int nx = sz.xc - 2 * Order;
    int ny = std::max(sz.yc - 2 * Order, 1);
    int nz = std::max(sz.zc - 2 * Order, 1);
    fp_t space_factor = 1.0_fp / (2.0_fp * state.dx);
    if constexpr (Order == 2) {
        space_factor = 1.0_fp / (12.0_fp * state.dx);
    }
    fp_t max_divb = 0.0_fp;
    dex_parallel_reduce(
        "Compute divB",
        FlatLoop<3>(nz, ny, nx),
        KOKKOS_LAMBDA (int ki, int ji, int ii, fp_t& max_divb) {
            const int k = nz == 1 ? ki : ki + Order;
            const int j = ny == 1 ? ji : ji + Order;
            const int i = ii + Order;

            using Cons = typename FTraits::cons;
            fp_t div = 0.0_fp;
            JasUse(Q, space_factor);
            if constexpr (Order == 1) {
                div = (Q(I(Cons::Bx), k, j, i + 1) - Q(I(Cons::Bx), k, j, i - 1)) * space_factor;
                if constexpr (FTraits::num_dim > 1) {
                    div += (Q(I(Cons::By), k, j + 1, i) - Q(I(Cons::By), k, j - 1, i)) * space_factor;
                }
                if constexpr (FTraits::num_dim > 2) {
                    div += (Q(I(Cons::Bz), k + 1, j, i) - Q(I(Cons::Bz), k - 1, j, i)) * space_factor;
                }
            } else if constexpr (Order == 2) {
                constexpr i32 IBX = I(Cons::Bx);
                div = (-Q(IBX, k, j, i + 2) + 8.0_fp * Q(IBX, k, j, i + 1) - 8.0_fp * Q(IBX, k, j, i - 1) + Q(IBX, k, j, i - 2)) * space_factor;
                if constexpr (FTraits::num_dim > 1) {
                    constexpr i32 IBY = I(Cons::By);
                    div += (-Q(IBY, k, j + 2, i) + 8.0_fp * Q(IBY, k, j + 1, i) - 8.0_fp * Q(IBY, k, j - 1, i) + Q(IBY, k, j - 2, i)) * space_factor;
                }
                if constexpr (FTraits::num_dim > 2) {
                    constexpr i32 IBZ = I(Cons::Bz);
                    div += (-Q(IBZ, k + 2, j, i) + 8.0_fp * Q(IBZ, k + 1, j, i) - 8.0_fp * Q(IBZ, k - 1, j, i) + Q(IBZ, k - 2, j, i)) * space_factor;
                }
            }

            divB(k, j, i) = div;
            const bool in_domain =
                ((i >= sz.ng) && i < (sz.xc - sz.ng)) &&
                (ny == 1 || ((j >= sz.ng) && i < (sz.yc - sz.ng))) &&
                (nz == 1 || ((k >= sz.ng) && i < (sz.zc - sz.ng)));
            if (in_domain) {
                max_divb = std::max(div, max_divb);
            }
        },
        Kokkos::Max<fp_t>(max_divb)
    );

    if (max_divb_out) {
        *max_divb_out  = max_divb;
    }
    return divB;
}

Fp3d compute_divb(const Simulation& sim, fp_t* max_divb_out) {
    if (sim.fluid_type == FluidType::Hydro) {
        throw std::runtime_error("Computing divB on a non-magnetic fluid.");
    }

    return invoke_fluid_traits(
        sim.num_dim,
        sim.fluid_type,
        [&]<typename FTraits>(FTraits) {
            return compute_divb_impl<FTraits>(sim, max_divb_out);
        }
    );
}

template <typename FTraits>
KOKKOS_INLINE_FUNCTION fp_t compute_divB_upwind(const Fp4d& Q, const Fluxes& fluxes, fp_t inv_dx, int k, int j, int i) {
    using Cons = typename FTraits::cons;
    // NOTE(cmo): Closer to pluto implementation
    // Cell left interface at idx i, right interface at i+1
    // Upwind b for the normal component of divergence (i.e. second derivative?)
    // Note: pluto does this whilst it still has access to the LR state.
    fp_t Bim1 = fluxes.Fx(I(Cons::Rho), k, j, i) > 0.0_fp ? Q(I(Cons::Bx), k, j, i-1) : Q(I(Cons::Bx), k, j, i);
    fp_t Bi = fluxes.Fx(I(Cons::Rho), k, j, i+1) > 0.0_fp ? Q(I(Cons::Bx), k, j, i) : Q(I(Cons::Bx), k, j, i+1);
    fp_t divB_i = (Bi - Bim1) * inv_dx;
    if constexpr (FTraits::num_dim > 1) {
        Bim1 = fluxes.Fy(I(Cons::Rho), k, j, i) > 0.0_fp ? Q(I(Cons::By), k, j-1, i) : Q(I(Cons::By), k, j, i);
        Bi = fluxes.Fy(I(Cons::Rho), k, j+1, i) > 0.0_fp ? Q(I(Cons::By), k, j, i) : Q(I(Cons::By), k, j+1, i);
        divB_i += (Bi - Bim1) * inv_dx;
    }
    if constexpr (FTraits::num_dim > 2) {
        Bim1 = fluxes.Fz(I(Cons::Rho), k, j, i) > 0.0_fp ? Q(I(Cons::Bz), k-1, j, i) : Q(I(Cons::Bz), k, j, i);
        Bi = fluxes.Fz(I(Cons::Rho), k+1, j, i) > 0.0_fp ? Q(I(Cons::Bz), k, j, i) : Q(I(Cons::Bz), k+1, j, i);
        divB_i += (Bi - Bim1) * inv_dx;
    }
    return divB_i;
}

template <typename FTraits>
void janhunen_cleaning(const Simulation& sim, fp_t divB_diff) {
    const auto& state = sim.state;
    JasUnpack(state, sz, Q, W);
    const auto& S = sim.sources.S;
    const auto& fluxes = sim.fluxes;
    const fp_t inv_dx = 1.0_fp / state.dx;

    constexpr bool divB_central = false;
    Fp3d divB;
    if constexpr (divB_central) {
        divB = compute_divb_impl<FTraits>(sim);
    }

    int nx = sz.xc - 4;
    int ny = std::max(sz.yc - 4, 1);
    int nz = std::max(sz.zc - 4, 1);
    dex_parallel_for(
        "Apply Janhunen source",
        FlatLoop<3>(nz, ny, nx),
        KOKKOS_LAMBDA (int ki, int ji, int ii) {
            const int k = nz == 1 ? ki : ki + 2;
            const int j = ny == 1 ? ji : ji + 2;
            const int i = ii + 2;
            using Cons = typename FTraits::cons;
            using Prim = typename FTraits::prim;

            fp_t divB_i;
            JasUse(divB, divB_central, fluxes, Q, inv_dx);
            if constexpr (divB_central) {
                divB_i = divB(k, j, i);
            } else {
                divB_i = compute_divB_upwind<FTraits>(Q, fluxes, inv_dx, k, j, i);
            }

            S(I(Cons::Bx), k, j, i) -= W(I(Prim::Vx), k, j, i)  * divB_i;
            if constexpr (FTraits::num_dim > 1) {
                S(I(Cons::By), k, j, i) -= W(I(Prim::Vy), k, j, i) * divB_i;
            }
            if constexpr (FTraits::num_dim > 2) {
                S(I(Cons::Bz), k, j, i) -= W(I(Prim::Vz), k, j, i) * divB_i;
            }
        }
    );
    Kokkos::fence();
}

template <typename FTraits>
void powell_cleaning(const Simulation& sim, fp_t divB_diff) {
    const auto& state = sim.state;
    JasUnpack(state, sz, Q, W);
    const auto& S = sim.sources.S;
    const auto& fluxes = sim.fluxes;

    fp_t inv_mu0 = 1.0_fp / state.mu0;
    fp_t inv_dx = 1.0_fp / state.dx;

    constexpr bool divB_central = false;
    Fp3d divB;
    if constexpr (divB_central) {
        divB = compute_divb_impl<FTraits>(sim);
    }

    int nx = sz.xc - 4;
    int ny = std::max(sz.yc - 4, 1);
    int nz = std::max(sz.zc - 4, 1);
    dex_parallel_for(
        "Apply Powell sources",
        FlatLoop<3>(nz, ny, nx),
        KOKKOS_LAMBDA (int ki, int ji, int ii) {
            const int k = nz == 1 ? ki : ki + 2;
            const int j = ny == 1 ? ji : ji + 2;
            const int i = ii + 2;
            using Cons = typename FTraits::cons;
            using Prim = typename FTraits::prim;

            fp_t divB_i;
            JasUse(divB, divB_central, fluxes, Q, inv_dx);
            if constexpr (divB_central) {
                divB_i = divB(k, j, i);
            } else {
                divB_i = compute_divB_upwind<FTraits>(Q, fluxes, inv_dx, k, j, i);
            }

            S(I(Cons::MomX), k, j, i) -= W(I(Prim::Bx), k, j, i)  * divB_i * inv_mu0;
            S(I(Cons::Bx), k, j, i) -= W(I(Prim::Vx), k, j, i)  * divB_i;
            S(I(Cons::Ene), k, j, i) -= W(I(Prim::Vx), k, j, i) * W(I(Prim::Bx), k, j, i)  * divB_i * inv_mu0;
            if constexpr (FTraits::num_dim > 1) {
                S(I(Cons::MomY), k, j, i) -= W(I(Prim::By), k, j, i)  * divB_i * inv_mu0;
                S(I(Cons::By), k, j, i) -= W(I(Prim::Vy), k, j, i) * divB_i;
                S(I(Cons::Ene), k, j, i) -= W(I(Prim::Vy), k, j, i) * W(I(Prim::By), k, j, i)  * divB_i * inv_mu0;
            }
            if constexpr (FTraits::num_dim > 2) {
                S(I(Cons::MomZ), k, j, i) -= W(I(Prim::Bz), k, j, i)  * divB_i * inv_mu0;
                S(I(Cons::Bz), k, j, i) -= W(I(Prim::Vz), k, j, i) * divB_i;
                S(I(Cons::Ene), k, j, i) -= W(I(Prim::Vz), k, j, i) * W(I(Prim::Bz), k, j, i)  * divB_i * inv_mu0;
            }
        }
    );
    Kokkos::fence();
}


template <typename FTraits>
void linde_cleaning(const Simulation& sim, fp_t divB_diff) {
    const auto& state = sim.state;
    JasUnpack(state, sz, Q);
    const auto& S = sim.sources.S;
    const auto& fluxes = sim.fluxes;

    constexpr bool divB_central = false;
    Fp3d divB;
    if constexpr (divB_central) {
        divB = compute_divb_impl<FTraits>(sim);
    }

    const fp_t inv_mu0 = 1.0_fp / state.mu0;
    // NOTE(cmo): This routine is not responsible for integrating the source
    // terms, so we need to divide by dt, unlike amrvac.
    const fp_t eta = divB_diff * square(state.dx) / fp_t(sim.num_dim) / sim.dt_sub;
    // NOTE(cmo): Shrink one cell further to handle gradient calculation -- hence needing at least 2 ghost cells
    const fp_t space_factor = 1.0_fp / (2.0_fp * state.dx);
    int nx = sz.xc - 4;
    int ny = std::max(sz.yc - 4, 1);
    int nz = std::max(sz.zc - 4, 1);
    dex_parallel_for(
        "Apply Linde source",
        FlatLoop<3>(nz, ny, nx),
        KOKKOS_LAMBDA (int ki, int ji, int ii) {
            const int k = nz == 1 ? ki : ki + 2;
            const int j = ny == 1 ? ji : ji + 2;
            const int i = ii + 2;
            using Cons = typename FTraits::cons;

            fp_t grad_divB_x;
            JasUse(divB_central, fluxes, divB, Q, space_factor);
            if constexpr (divB_central) {
                grad_divB_x = (divB(k, j, i + 1) - divB(k, j, i - 1)) * space_factor;
            } else {
                fp_t divB_ip = compute_divB_upwind<FTraits>(Q, fluxes, 2.0_fp * space_factor, k, j, i+1);
                fp_t divB_im = compute_divB_upwind<FTraits>(Q, fluxes, 2.0_fp * space_factor, k, j, i-1);
                grad_divB_x  = (divB_ip - divB_im) * space_factor;
            }
            const fp_t Bx = Q(I(Cons::Bx), k, j, i);
            S(I(Cons::Bx), k, j, i) += grad_divB_x * eta;
            S(I(Cons::Ene), k, j, i) += Bx * grad_divB_x * eta * inv_mu0;

            if constexpr (FTraits::num_dim > 1) {
                const fp_t By = Q(I(Cons::By), k, j, i);
                fp_t grad_divB_y;
                if constexpr (divB_central) {
                    grad_divB_y = (divB(k, j + 1, i) - divB(k, j - 1, i)) * space_factor;
                } else {
                    fp_t divB_ip = compute_divB_upwind<FTraits>(Q, fluxes, 2.0_fp * space_factor, k, j+1, i);
                    fp_t divB_im = compute_divB_upwind<FTraits>(Q, fluxes, 2.0_fp * space_factor, k, j-1, i);
                    grad_divB_y  = (divB_ip - divB_im) * space_factor;
                }
                S(I(Cons::By), k, j, i) += grad_divB_y * eta;
                S(I(Cons::Ene), k, j, i) += By * grad_divB_y * eta * inv_mu0;
            }
            if constexpr (FTraits::num_dim > 2) {
                const fp_t Bz = Q(I(Cons::Bz), k, j, i);
                fp_t grad_divB_z;
                if constexpr (divB_central) {
                    grad_divB_z = (divB(k + 1, j, i) - divB(k - 1, j, i)) * space_factor;
                } else {
                    fp_t divB_ip = compute_divB_upwind<FTraits>(Q, fluxes, 2.0_fp * space_factor, k+1, j, i);
                    fp_t divB_im = compute_divB_upwind<FTraits>(Q, fluxes, 2.0_fp * space_factor, k-1, j, i);
                    grad_divB_z  = (divB_ip - divB_im) * space_factor;
                }
                S(I(Cons::Bz), k, j, i) += grad_divB_z * eta;
                S(I(Cons::Ene), k, j, i) += Bz * grad_divB_z * eta * inv_mu0;
            }
        }
    );
    Kokkos::fence();
}

template <typename FTraits>
std::function<void(const Simulation&, fp_t)> select_scheme(DivBCleaningScheme scheme) {
    if (scheme == DivBCleaningScheme::Linde) {
        return linde_cleaning<FTraits>;
    } else if (scheme == DivBCleaningScheme::Janhunen) {
        return janhunen_cleaning<FTraits>;
    } else if (scheme == DivBCleaningScheme::LindeJanhunen) {
        return [](const Simulation& sim, fp_t divB_diff) {
            linde_cleaning<FTraits>(sim, divB_diff);
            janhunen_cleaning<FTraits>(sim, divB_diff);
        };
    } else if (scheme == DivBCleaningScheme::Powell8Wave) {
        return powell_cleaning<FTraits>;
    }
    throw std::runtime_error("Unknown divB cleaning scheme.");
}

void setup_base_field_divb_cleaning(Simulation& sim, YAML::Node& config) {
    constexpr const char* source_name = "divb_cleaning";
    if (!is_instance(sim.fluid_type, FluidType::Mhd)) {
        return;
    }

    if (sim.state.sz.ng < 2) {
        throw std::runtime_error("For divB cleaning, need at least 2 ghost cells");
    }
    bool do_cleaning = get_or<bool>(config, "simulation.divb_cleaning", !is_instance(sim.fluid_type, FluidType::GlmMhd));
    if (!do_cleaning) {
        return;
    }
    if (source_term_index(sim, source_name) != sim.compute_source_terms.size()) {
        throw std::runtime_error(fmt::format("Source \"{}\" already registered.", source_name));
    }
    std::string cleaning_type = get_or<std::string>(config, "simulation.divb_cleaning_scheme", "powell8wave");
    DivBCleaningScheme scheme = find_associated_enum<DivBCleaningScheme>(DivBCleaningName, NumDivBCleaningScheme, cleaning_type);

    fp_t divb_diff = get_or<fp_t>(config, "simulation.divb_diff", 0.2_fp);
    if (scheme == DivBCleaningScheme::Projection) {
        sim.clean_divb = select_fluid_traits<const Simulation&>(
            sim.num_dim,
            sim.fluid_type,
            [] <typename FTraits> (FTraits, const Simulation& sim) {
                return magnetic_field_projection_gs<FTraits>(sim);
            }
        );
    } else {
        auto cleaning_source = invoke_fluid_traits(
            sim.num_dim,
            sim.fluid_type,
            [scheme] <typename FTraits> (FTraits) {
                return select_scheme<FTraits>(scheme);
            }
        );
        sim.compute_source_terms.push_back(SourceTerm{
            .name = cleaning_type,
            .fn = [divb_diff, cleaning_source] (const Simulation& sim) {
                cleaning_source(sim, divb_diff);
            }
        });
    }
}

void setup_glm_divb_cleaning(Simulation& sim, YAML::Node& config) {
    if (!is_instance(sim.fluid_type, FluidType::GlmMhd)) {
        return;
    }

    fp_t glm_alpha = get_or<fp_t>(config, "simulation.glm_alpha", 0.1_fp);
    bool glm_extended = get_or<bool>(config, "simulation.glm_extended_source", false);

    auto glm_source_fn = select_fluid_traits<const Simulation&>(
        sim.num_dim,
        sim.fluid_type,
        [glm_alpha, glm_extended] <typename FTraits> (FTraits, const Simulation& sim) {
            if (glm_extended) {
                return glm_source<FTraits, true>(sim, glm_alpha);
            }
            return glm_source<FTraits, false>(sim, glm_alpha);
        }
    );
    sim.compute_source_terms.push_back(SourceTerm{
        .name = "GLM Source",
        .fn = glm_source_fn
    });
}

void setup_divb_cleaning(Simulation& sim, YAML::Node& config) {
    setup_base_field_divb_cleaning(sim, config);
    setup_glm_divb_cleaning(sim, config);
}

}