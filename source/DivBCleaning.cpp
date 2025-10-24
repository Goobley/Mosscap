#include "DivBCleaning.hpp"
#include "Simulation.hpp"
#include "MosscapConfig.hpp"
#include "SourceTerms.hpp"

namespace Mosscap {

void clean_divb(const Simulation& sim) {
    // NOTE(cmo): Leaving this in case we decide to have other methods that need
    // it, such as projection.
    if (sim.clean_divb) {
        sim.clean_divb(sim);
    }
}

template <typename FTraits>
fp_t compute_max_residual(const Simulation& sim, Fp3d divB, Fp3d phi) {
    const auto& state = sim.state;
    JasUnpack(state, sz, Q);
    int nx = sz.xc - 2;
    int ny = std::max(sz.yc - 2, 1);
    int nz = std::max(sz.zc - 2, 1);
    constexpr fp_t dim_factor = (2.0_fp * fp_t(FTraits::num_dim));
    const fp_t inv_dx2 = 1.0_fp / square(state.dx);

    fp_t resid2 = 0.0_fp;
    dex_parallel_reduce(
        "phi residual",
        FlatLoop<3>(nz, ny, nx),
        KOKKOS_LAMBDA (int ki, int ji, int ii, fp_t& running_max) {
            const int k = nz == 1 ? ki : ki + 1;
            const int j = ny == 1 ? ji : ji + 1;
            const int i = ii + 1;
            fp_t d2phi = phi(k, j, i + 1) + phi(k, j, i - 1);
            if constexpr (FTraits::num_dim > 1) {
                d2phi += phi(k, j + 1, i) + phi(k, j - 1, i);
            }
            if constexpr (FTraits::num_dim > 2) {
                d2phi += phi(k + 1, j, i) + phi(k - 1, j, i);
            }
            d2phi -= dim_factor * phi(k, j, i);
            d2phi *= inv_dx2;
            running_max = std::max(running_max, square(d2phi - divB(k, j, i)));
        },
        Kokkos::Max<fp_t>(resid2)
    );
    return std::sqrt(resid2);
}

template <typename FTraits>
Fp3d rbgs_poisson(const Simulation& sim, Fp3d divB, int max_iters) {
    // NOTE(cmo): This is very crude
    const auto& state = sim.state;
    JasUnpack(state, sz, Q);
    int nx = sz.xc - 2;
    int ny = std::max(sz.yc - 2, 1);
    int nz = std::max(sz.zc - 2, 1);
    Fp3d phi("phi", divB.extent(0), divB.extent(1), divB.extent(2));
    phi = 0.0_fp;
    constexpr fp_t denom = 1.0_fp / (2.0_fp * fp_t(FTraits::num_dim));
    Kokkos::fence();
    for (int iter = 0; iter < max_iters; ++iter) {
        dex_parallel_for(
            "rbgs_red",
            FlatLoop<3>(nz, ny, nx),
            KOKKOS_LAMBDA (int ki, int ji, int ii) {
                const int k = nz == 1 ? ki : ki + 1;
                const int j = ny == 1 ? ji : ji + 1;
                const int i = ii + 1;
                if (((i + j + k) & 1) == 0) {
                    fp_t sum = phi(k, j, i + 1) + phi(k, j, i - 1);
                    if constexpr (FTraits::num_dim > 1) {
                        sum += phi(k, j + 1, i) + phi(k, j - 1, i);
                    }
                    if constexpr (FTraits::num_dim > 2) {
                        sum += phi(k + 1, j, i) + phi(k - 1, j, i);
                    }
                    phi(k, j, i) = (sum - square(state.dx) * divB(k, j, i)) * denom;
                }
            }
        );
        Kokkos::fence();

        dex_parallel_for(
            "rbgs black",
            FlatLoop<3>(nz, ny, nx),
            KOKKOS_LAMBDA (int ki, int ji, int ii) {
                const int k = nz == 1 ? ki : ki + 1;
                const int j = ny == 1 ? ji : ji + 1;
                const int i = ii + 1;
                if (((i + j + k) & 1) == 1) {
                    fp_t sum = phi(k, j, i + 1) + phi(k, j, i - 1);
                    if constexpr (FTraits::num_dim > 1) {
                        sum += phi(k, j + 1, i) + phi(k, j - 1, i);
                    }
                    if constexpr (FTraits::num_dim > 2) {
                        sum += phi(k + 1, j, i) + phi(k - 1, j, i);
                    }
                    phi(k, j, i) = (sum - square(state.dx) * divB(k, j, i)) * denom;
                }
            }
        );
        Kokkos::fence();
    }
    const fp_t resid = compute_max_residual<FTraits>(sim, divB, phi);
    fmt::println("Residual {}", resid);
    return phi;
}

template <typename FTraits>
void apply_grad_phi(const Simulation& sim, Fp3d phi) {
    const auto& state = sim.state;
    JasUnpack(state, sz, Q);
    int nx = sz.xc - 2;
    int ny = std::max(sz.yc - 2, 1);
    int nz = std::max(sz.zc - 2, 1);
    const fp_t space_factor = 1.0_fp / (2.0_fp * state.dx);

    dex_parallel_for(
        "B update",
        FlatLoop<3>(nz, ny, nx),
        KOKKOS_LAMBDA (int ki, int ji, int ii) {
            using Cons = FTraits::cons;
            const int k = nz == 1 ? ki : ki + 1;
            const int j = ny == 1 ? ji : ji + 1;
            const int i = ii + 1;
            const fp_t grad_phi_x = (phi(k, j, i + 1) - phi(k, j, i - 1)) * space_factor;
            Q(I(Cons::Bx), k, j, i) -= grad_phi_x;

            if constexpr (FTraits::num_dim > 1) {
                const fp_t grad_phi_y = (phi(k, j + 1, i) - phi(k, j - 1, i)) * space_factor;
                Q(I(Cons::By), k, j, i) -= grad_phi_y;
            }
            if constexpr (FTraits::num_dim > 2) {
                const fp_t grad_phi_z = (phi(k + 1, j, i) - phi(k - 1, j, i)) * space_factor;
                Q(I(Cons::Bz), k, j, i) -= grad_phi_z;
            }
        }
    );
    Kokkos::fence();
}

template <typename FTraits>
void magnetic_field_projection_gs(const Simulation& sim) {
    constexpr int num_rb_iter = 50;

    fp_t max_divb = 0.0_fp;
    Fp3d divB = compute_divb_impl<FTraits>(sim, &max_divb);
    fmt::println("DivB pre {}", max_divb);

    Fp3d phi = rbgs_poisson<FTraits>(sim, divB, num_rb_iter);
    fmt::println("phi computed");
    apply_grad_phi<FTraits>(sim, phi);

    divB = compute_divb_impl<FTraits>(sim, &max_divb);
    fmt::println("DivB post {}", max_divb);
}

template <typename FTraits>
Fp3d compute_divb_impl(const Simulation& sim, fp_t* max_divb_out) {
    const auto& state = sim.state;
    JasUnpack(state, sz, Q, W);
    Fp3d divB("divB", Q.extent(1), Q.extent(2), Q.extent(3));

    // NOTE(cmo): This loop needs to go into the ghost cells, but still requires
    // a 1 cell border.
    int nx = sz.xc - 2;
    int ny = std::max(sz.yc - 2, 1);
    int nz = std::max(sz.zc - 2, 1);
    const fp_t space_factor = 1.0_fp / (2.0_fp * state.dx);
    fp_t max_divb = 0.0_fp;
    dex_parallel_reduce(
        "Compute divB",
        FlatLoop<3>(nz, ny, nx),
        KOKKOS_LAMBDA (int ki, int ji, int ii, fp_t& max_divb) {
            const int k = nz == 1 ? ki : ki + 1;
            const int j = ny == 1 ? ji : ji + 1;
            const int i = ii + 1;

            using Cons = FTraits::cons;
            fp_t div = (W(I(Cons::Bx), k, j, i + 1) - W(I(Cons::Bx), k, j, i - 1)) * space_factor;
            if constexpr (FTraits::num_dim > 1) {
                div += (W(I(Cons::By), k, j + 1, i) - W(I(Cons::By), k, j - 1, i)) * space_factor;
            }
            if constexpr (FTraits::num_dim > 2) {
                div += (W(I(Cons::Bz), k + 1, j, i) - W(I(Cons::Bz), k - 1, j, i)) * space_factor;
            }

            divB(k, j, i) = div;
            max_divb = std::max(div, max_divb);
        },
        Kokkos::Max<fp_t>(max_divb)
    );

    if (max_divb_out) {
        *max_divb_out  = max_divb;
    }
    return divB;
}

Fp3d compute_divb(const Simulation& sim, fp_t* max_divb_out) {
    if (sim.fluid_type != FluidType::Mhd) {
        throw std::runtime_error("Computing divB on a non-magnetic fluid.");
    }

    if (sim.num_dim == 1) {
        return compute_divb_impl<FluidTraits<1, FluidType::Mhd>>(sim, max_divb_out);
    } else if (sim.num_dim == 2) {
        return compute_divb_impl<FluidTraits<2, FluidType::Mhd>>(sim, max_divb_out);
    } else {
        return compute_divb_impl<FluidTraits<3, FluidType::Mhd>>(sim, max_divb_out);
    }
}

template <typename FTraits>
void janhunen_cleaning(const Simulation& sim, fp_t divB_diff) {
    const auto& state = sim.state;
    JasUnpack(state, sz, Q, W);
    const auto& S = sim.sources.S;

    fp_t max_divb = 0.0_fp;
    Fp3d divB = compute_divb_impl<FTraits>(sim, &max_divb);
    fmt::println("DivB pre {}", max_divb);

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
            using Cons = FTraits::cons;
            using Prim = FTraits::prim;

            S(I(Cons::Bx), k, j, i) -= W(I(Prim::Vx), k, j, i)  * divB(k, j, i);
            if constexpr (FTraits::num_dim > 1) {
                S(I(Cons::By), k, j, i) -= W(I(Prim::Vy), k, j, i) * divB(k, j, i);
            }
            if constexpr (FTraits::num_dim > 2) {
                S(I(Cons::Bz), k, j, i) -= W(I(Prim::Vz), k, j, i) * divB(k, j, i);
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

    fp_t max_divb = 0.0_fp;
    Fp3d divB = compute_divb_impl<FTraits>(sim, &max_divb);
    fmt::println("DivB pre {}", max_divb);

    const fp_t inv_mu0 = 1.0_fp / state.mu0;
    // NOTE(cmo): This routine is not responsible for integrating the source
    // terms, so we need to divide by dt, unlike amrvac.
    const fp_t dt_sub = sim.dt_sub;
    // const fp_t eta = divB_diff * square(sim.state.dx) / fp_t(sim.num_dim) / sim.dt_sub;
    const fp_t eta = divB_diff * square(state.dx) / fp_t(sim.num_dim) / sim.dt_sub;
    // const fp_t eta_dt = divB_diff * square(sim.state.dx) / fp_t(sim.num_dim);
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
            using Cons = FTraits::cons;

            const fp_t grad_divB_x = (divB(k, j, i + 1) - divB(k, j, i - 1)) * space_factor;
            const fp_t Bx = Q(I(Cons::Bx), k, j, i);
            S(I(Cons::Bx), k, j, i) += grad_divB_x * eta;
            // S(I(Cons::Ene), k, j, i) += Bx * grad_divB_x * eta * inv_mu0;

            if constexpr (FTraits::num_dim > 1) {
                const fp_t By = Q(I(Cons::By), k, j, i);
                const fp_t grad_divB_y = (divB(k, j + 1, i) - divB(k, j - 1, i)) * space_factor;
                S(I(Cons::By), k, j, i) += grad_divB_y * eta;
                // S(I(Cons::Ene), k, j, i) += By * grad_divB_y * eta * inv_mu0;
            }
            if constexpr (FTraits::num_dim > 2) {
                const fp_t Bz = Q(I(Cons::Bz), k, j, i);
                const fp_t grad_divB_z = (divB(k + 1, j, i) - divB(k - 1, j, i)) * space_factor;
                S(I(Cons::Bz), k, j, i) += grad_divB_z * eta;
                // S(I(Cons::Ene), k, j, i) += Bz * grad_divB_z * eta * inv_mu0;
            }
        }
    );
    Kokkos::fence();

    if (std::abs(sim.time - 0.3_fp) < 1e-6_fp) {
        yakl::SimpleNetCDF nc;
        nc.create("simple_out.nc", yakl::NETCDF_MODE_REPLACE);
        nc.write(state.Q, "Q", {"var", "z", "y", "x"});
        nc.write(S, "S", {"var", "z", "y", "x"});
        nc.write(sim.dt, "dt");
        nc.write(sim.dt_sub, "dt_sub");
    }
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
    }
    throw std::runtime_error("Unknown divB cleaning scheme.");
}


void setup_divb_cleaning(Simulation& sim, YAML::Node& config) {
    constexpr const char* source_name = "divb_cleaning";
    if (sim.fluid_type == FluidType::Mhd) {
        if (sim.state.sz.ng < 2) {
            throw std::runtime_error("For divB cleaning, need at least 2 ghost cells");
        }
        bool do_cleaning = get_or<bool>(config, "simulation.divb_cleaning", true);
        if (!do_cleaning) {
            return;
        }
        if (source_term_index(sim, source_name) != sim.compute_source_terms.size()) {
            throw std::runtime_error(fmt::format("Source \"{}\" already registered.", source_name));
        }
        std::string cleaning_type = get_or<std::string>(config, "simulation.divb_cleaning_scheme", "linde");
        DivBCleaningScheme scheme = find_associated_enum<DivBCleaningScheme>(DivBCleaningName, NumDivBCleaningScheme, cleaning_type);

        fp_t divb_diff = get_or<fp_t>(config, "simulation.divb_diff", 0.8_fp);
        if (sim.num_dim == 1) {
            if (scheme != DivBCleaningScheme::Projection) {
                auto cleaning_source = select_scheme<FluidTraits<1, FluidType::Mhd>>(scheme);
                sim.compute_source_terms.push_back(SourceTerm{
                    .name = cleaning_type,
                    .fn = [divb_diff, cleaning_source] (const Simulation& sim) {
                        cleaning_source(sim, divb_diff);
                    }
                });
            } else {
                sim.clean_divb = magnetic_field_projection_gs<FluidTraits<1, FluidType::Mhd>>;
            }
        } else if (sim.num_dim == 2) {
            if (scheme != DivBCleaningScheme::Projection) {
                auto cleaning_source = select_scheme<FluidTraits<2, FluidType::Mhd>>(scheme);
                sim.compute_source_terms.push_back(SourceTerm{
                    .name = cleaning_type,
                    .fn = [divb_diff, cleaning_source] (const Simulation& sim) {
                        cleaning_source(sim, divb_diff);
                    }
                });
            } else {
                sim.clean_divb = magnetic_field_projection_gs<FluidTraits<2, FluidType::Mhd>>;
            }
        } else if (sim.num_dim == 3) {
            if (scheme != DivBCleaningScheme::Projection) {
                auto cleaning_source = select_scheme<FluidTraits<3, FluidType::Mhd>>(scheme);
                sim.compute_source_terms.push_back(SourceTerm{
                    .name = cleaning_type,
                    .fn = [divb_diff, cleaning_source] (const Simulation& sim) {
                        cleaning_source(sim, divb_diff);
                    }
                });
            } else {
                sim.clean_divb = magnetic_field_projection_gs<FluidTraits<1, FluidType::Mhd>>;
            }
        }
    }
}

}