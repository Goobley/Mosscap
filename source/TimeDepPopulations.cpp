#include "TimeDepPopulations.hpp"
#include "../DexRT/source/State.hpp"
#include "../DexRT/source/Collisions.hpp"
#include "KokkosBatched_Gesv.hpp"
#include "KokkosBatched_LU_Decl.hpp"
#include "KokkosBatched_SolveLU_Decl.hpp"
#include "KokkosBlas.hpp"

template <typename T=fp_t, typename State>
fp_t time_dep_impl(State& state, const Fp2d& prev_pops, const KineticEqOptions& args = KineticEqOptions()) {
    // NOTE(cmo): This implementation works, and uses less memory, however it's
    // not very computationally efficient, as we're processing a small matrix
    // per entire thread team. That said, it profiles faster
    yakl::timer_start("Kinetic eq");
    JasUnpack(args, ignore_change_below_ntot_frac, dt);
    fp_t global_max_change = FP(0.0);
    for (int ia = 0; ia < state.adata_host.num_level.extent(0); ++ia) {
        if (args.only_atom >= 0 && args.only_atom != ia) {
            continue;
        }
        JasUnpack(state, pops);
        JasUnpack(args, theta, predicted_pops);
        const auto& Gamma = state.Gamma[ia];
        const fp_t abundance = state.adata_host.abundance(ia);
        const auto nh_tot = state.atmos.nh_tot;

        constexpr bool iterative_improvement = true;
        constexpr int num_refinement_passes = 2;

        const i64 Nspace = Gamma.extent(2);
        const int pops_start = state.adata_host.level_start(ia);
        const int num_level = state.adata_host.num_level(ia);

        FlatLoop<2> nxn_loop(num_level, num_level);

        if (args.theta < FP(1.0) && args.initial_iter) {
            if (!args.predicted_pops.initialized()) {
                throw std::runtime_error("Need to provide scratch space for the predicted populations");
            }
            size_t scratch_size = ScratchView<T**>::shmem_size(num_level, num_level);
            scratch_size += ScratchView<T*>::shmem_size(num_level);

            Kokkos::parallel_for(
                "Compute predicted pops",
                TeamPolicy(Nspace, std::min(square(num_level), 128)).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
                KOKKOS_LAMBDA (const KTeam& team) {
                    const i64 ks = team.league_rank();

                    ScratchView<T**> Gammak(team.team_scratch(0), num_level, num_level);
                    ScratchView<T*> pred_pops(team.team_scratch(0), num_level);

                    // Copy over Gamma chunk
                    Kokkos::parallel_for(
                        Kokkos::TeamVectorRange(team, nxn_loop.num_iter),
                        [&] (const int x) {
                            const auto args = nxn_loop.unpack(x);
                            const int i = args[0];
                            const int j = args[1];

                            Gammak(i, j) = Gamma(i, j, ks);
                        }
                    );
                    team.team_barrier();

                    // Fixup gamma
                    Kokkos::parallel_for(
                        Kokkos::TeamVectorRange(team, num_level),
                        [&] (const int i) {
                            T diag = T(FP(0.0));
                            Gammak(i, i) = diag;
                            for (int j = 0; j < num_level; ++j) {
                                diag += Gammak(j, i);
                            }
                            Gammak(i, i) = -diag;
                        }
                    );

                    // Set up pops vec
                    Kokkos::parallel_for(
                        Kokkos::TeamVectorRange(team, num_level),
                        [&] (const int i) {
                            pred_pops(i) = predicted_pops(pops_start + i, ks);
                        }
                    );
                    team.team_barrier();

                    // pred_pops = Gamma * n
                    KokkosBlas::Experimental::Gemv<KokkosBlas::Mode::TeamVector, KokkosBlas::Algo::Gemv::Default>::invoke(
                        team,
                        'n',
                        T(1),
                        Gammak,
                        pred_pops,
                        T(0),
                        pred_pops
                    );
                    team.team_barrier();

                    // Copy pops vec
                    Kokkos::parallel_for(
                        Kokkos::TeamVectorRange(team, num_level),
                        [&] (const int i) {
                            predicted_pops(pops_start + i, ks) = pred_pops(i);
                        }
                    );
                    team.team_barrier();
                }
            );
            Kokkos::fence();
        }

        // NOTE(cmo): This allocation could be avoided, but we would need to fuse everything into one kernel.
        yakl::Array<T, 2, yakl::memDevice> new_pops("new_pops", Nspace, num_level);

        size_t scratch_size = ScratchView<T**>::shmem_size(num_level, num_level);
        if (iterative_improvement) {
            scratch_size *= 2;
            scratch_size += 2 * ScratchView<T*>::shmem_size(num_level);
        }


        Kokkos::parallel_for(
            "Kinetic Eq",
            TeamPolicy(Nspace, std::min(square(num_level), 128)).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
            KOKKOS_LAMBDA (const KTeam& team) {
                const i64 ks = team.league_rank();

                ScratchView<T**> Gammak(team.team_scratch(0), num_level, num_level);
                KView<T*> new_popsk(&new_pops(ks, 0), new_pops.extent(1));

                // Copy over Gamma chunk
                Kokkos::parallel_for(
                    Kokkos::TeamVectorRange(team, nxn_loop.num_iter),
                    [&] (const int x) {
                        const auto args = nxn_loop.unpack(x);
                        const int i = args[0];
                        const int j = args[1];

                        Gammak(i, j) = Gamma(i, j, ks);
                    }
                );
                team.team_barrier();

                // Fixup gamma
                Kokkos::parallel_for(
                    Kokkos::TeamVectorRange(team, num_level),
                    [&] (const int i) {
                        T diag = T(FP(0.0));
                        Gammak(i, i) = diag;
                        for (int j = 0; j < num_level; ++j) {
                            diag += Gammak(j, i);
                        }
                        // NOTE(cmo): For stat eq we would set to -diag, but for
                        // time_dep it's 1 - Gamma*dt, so 1 + diag * dt
                        Gammak(i, i) = FP(1.0) + theta * diag * dt;
                    }
                );

                if (theta < FP(1.0)) {
                    // Setup rhs
                    Kokkos::parallel_for(
                        Kokkos::TeamVectorRange(team, num_level),
                        [&] (const int i) {
                            new_popsk(i) =  (FP(1.0) - theta) * dt * predicted_pops(pops_start + i, ks) + prev_pops(pops_start + i, ks);
                        }
                    );
                } else {
                    // Setup rhs
                    Kokkos::parallel_for(
                        Kokkos::TeamVectorRange(team, num_level),
                        [&] (const int i) {
                            new_popsk(i) =  prev_pops(pops_start + i, ks);
                        }
                    );
                }
                team.team_barrier();

                // Convert Gamma to time-dependent backwards Euler
                Kokkos::parallel_for(
                    Kokkos::TeamVectorRange(team, nxn_loop.num_iter),
                    [&] (const int x) {
                        const auto args = nxn_loop.unpack(x);
                        const int i = args[0];
                        const int j = args[1];

                        if (i != j) {
                            Gammak(i, j) = -Gammak(i, j) * theta * dt;
                        }
                    }
                );
                team.team_barrier();

                ScratchView<T**> Gamma_copy;
                ScratchView<T*> lhs;
                ScratchView<T*> residuals;
                if (iterative_improvement) {
                    Gamma_copy = ScratchView<T**>(team.team_scratch(0), num_level, num_level);
                    lhs = ScratchView<T*>(team.team_scratch(0), num_level);
                    residuals = ScratchView<T*>(team.team_scratch(0), num_level);

                    Kokkos::parallel_for(
                        Kokkos::TeamVectorRange(team, nxn_loop.num_iter),
                        [&] (const int x) {
                            const auto args = nxn_loop.unpack(x);
                            const int i = args[0];
                            const int j = args[1];

                            if (i == 0) {
                                lhs(j) = new_popsk(j);
                                residuals(j) = new_popsk(j);
                            }
                            Gamma_copy(i, j) = Gammak(i, j);
                        }
                    );
                    team.team_barrier();
                }

                // LU factorise
                KokkosBatched::LU<KTeam, KokkosBatched::Mode::Team, KokkosBatched::Algo::LU::Unblocked>::invoke(
                    team, Gammak
                );
                team.team_barrier();
                // LU Solve
                KokkosBatched::TeamSolveLU<
                    KTeam,
                    KokkosBatched::Trans::NoTranspose,
                    KokkosBatched::Algo::Trsm::Unblocked
                >::invoke(
                    team,
                    Gammak,
                    new_popsk
                );
                team.team_barrier();

                if (iterative_improvement) {
                    for (int refinement = 0; refinement < num_refinement_passes; ++refinement) {
                        // r_i = b_i
                        Kokkos::parallel_for(
                            Kokkos::TeamVectorRange(team, residuals.extent(0)),
                            [&] (int i) {
                                residuals(i) = lhs(i);
                            }
                        );
                        team.team_barrier();
                        // r -= Gamma x
                        KokkosBlas::Experimental::Gemv<KokkosBlas::Mode::TeamVector, KokkosBlas::Algo::Gemv::Default>::invoke(
                            team,
                            'n',
                            T(-1),
                            Gamma_copy,
                            new_popsk,
                            T(1),
                            residuals
                        );
                        team.team_barrier();
                        // Solve Gamma x' = r (already factorised)
                        KokkosBatched::TeamSolveLU<
                            KTeam,
                            KokkosBatched::Trans::NoTranspose,
                            KokkosBatched::Algo::Trsm::Unblocked
                        >::invoke(
                            team,
                            Gammak,
                            residuals
                        );
                        team.team_barrier();
                        // x += x'
                        Kokkos::parallel_for(
                            Kokkos::TeamVectorRange(
                                team,
                                new_popsk.extent(0)
                            ),
                            [&] (int i) {
                                new_popsk(i) += residuals(i);
                            }
                        );
                        team.team_barrier();
                    }
                }
            }
        );
        Kokkos::fence();

        typedef Kokkos::MaxLoc<fp_t, Kokkos::pair<int, int>> Reducer;
        typedef Reducer::value_type ReductionVal;
        ReductionVal max_change_loc;
        dex_parallel_reduce(
            "Update pops and compute max change",
            FlatLoop<2>(Nspace, num_level),
            KOKKOS_LAMBDA (int ks, int i, ReductionVal& rval) {
                fp_t change = FP(0.0);
                const fp_t n_total_k = nh_tot(ks) * abundance;
                fp_t new_pop_scaled = new_pops(ks, i);

                // compute change
                if (pops(pops_start + i, ks) < ignore_change_below_ntot_frac * n_total_k) {
                    change = FP(0.0);
                } else {
                    change = std::abs(FP(1.0) - pops(pops_start + i, ks) / new_pop_scaled);
                }

                // update
                pops(pops_start + i, ks) = new_pop_scaled;

                // reduce update
                if (change > rval.val) {
                    rval.val = change;
                    rval.loc = Kokkos::make_pair(ks, i);
                }
            },
            Reducer(max_change_loc)
        );

        const fp_t max_change = max_change_loc.val;
        auto temp_h = Fp1d("temp_readback", &state.atmos.temperature(max_change_loc.loc.first), 1).createHostCopy();

        state.println(
            "     Max Change (ele: {}, Z={}): {} (@ l={}, ks={}) [T={}]",
            ia,
            state.adata_host.Z(ia),
            max_change,
            max_change_loc.loc.second,
            max_change_loc.loc.first,
            temp_h(0)
        );
        global_max_change = std::max(max_change, global_max_change);

    }
    yakl::timer_stop("Kinetic eq");
    return global_max_change;
}

template <typename State>
fp_t time_dep_update(State& state, const Fp2d& prev_pops, const KineticEqOptions& args) {
#ifdef HAVE_MPI
    fp_t max_rel_change;
    if (state.mpi_state.rank == 0) {
        max_rel_change = time_dep_impl<StatEqPrecision>(state, prev_pops, args);
    }
    MPI_Bcast(&max_rel_change, 1, get_FpMpi(), 0, state.mpi_state.comm);
    return max_rel_change;
#else
    return time_dep_impl<StatEqPrecision>(state, prev_pops, args);
#endif
}


template <typename T=fp_t, typename State>
fp_t time_dep_nr_post_update_impl(State& state, const Fp2d& prev_pops, const TimeDepNrPostUpdateOptions& args) {
    yakl::timer_start("Charge conservation");
    JasUnpack(args, ignore_change_below_ntot_frac, conserve_pressure, dt, theta, predicted_pops);
    JasUnpack(state, atmos, nh_lte);

    // TODO(cmo): Add background n_e term like in Lw.
    // NOTE(cmo): Only considers H for now
    // TODO(cmo): He contribution?
    assert(state.have_h && "Need to have H active for non-lte EOS");
    const auto& pops = state.pops;
    const auto& GammaH = state.Gamma[0];
    const int num_level = GammaH.extent(0);
    const int num_eqn = GammaH.extent(0) + 1;
    const int num_space = GammaH.extent(2);
    JasUnpack(state.atmos, ne, nh_tot, pressure, temperature);
    // NOTE(cmo): GammaH_flat is how we access Gamma/C in the following
    const auto& GammaH_flat = state.Gamma[0];

    const fp_t total_abund = state.config.total_abund;
    constexpr fp_t ne_pert_size = FP(1e-2);
    constexpr bool iterative_improvement = true;
    constexpr int num_refinement_passes = 2;

    const auto& H_atom = extract_atom(state.adata, state.adata_host, 0);

    size_t scratch_size = ScratchView<T**>::shmem_size(num_level, num_level); // Gammak
    scratch_size += ScratchView<fp_t**>::shmem_size(num_level, num_level); // C
    // scratch_size += ScratchView<fp_t**>::shmem_size(num_level, num_level); // C_ne_pert
    scratch_size += ScratchView<fp_t**>::shmem_size(num_level, num_level); // dC
    scratch_size += ScratchView<T**>::shmem_size(num_eqn, num_eqn); // dF
    scratch_size += ScratchView<T*>::shmem_size(num_eqn); // F
    scratch_size += ScratchView<T*>::shmem_size(num_level); // new_popsk
    scratch_size += ScratchView<T*>::shmem_size(num_level); // prev_popsk
    if (iterative_improvement) {
        scratch_size += ScratchView<T**>::shmem_size(num_eqn, num_eqn); // dF copy
        scratch_size += 2 * ScratchView<T*>::shmem_size(num_eqn); // lhs_copy/residuals
    }
    yakl::Array<T, 2, yakl::memDevice> new_F("new_F", num_space, num_eqn);
    Fp3d C("C", num_level, num_level, num_space);
    Fp3d dC("dC", num_level, num_level, num_space);
    C = FP(0.0);
    dC = FP(0.0);
    FlatLoop<2> nlxnl_loop(num_level, num_level);
    FlatLoop<2> nexne_loop(num_eqn, num_eqn);

    Fp2d n_star = state.pops.createDeviceObject();
    compute_lte_pops(&state, n_star);
    Kokkos::fence();
    const auto n_star_slice = slice_pops(
        n_star,
        state.adata_host,
        state.atoms_with_gamma_mapping[0]
    );

    // NOTE(cmo): These terms can be computed in shared memory (see the
    // commented blocks), but it's much slower because we're only doing one set
    // of rates per thread team.
    dex_parallel_for(
        "Compute C, dC",
        FlatLoop<1>(num_space),
        KOKKOS_LAMBDA (i64 ks) {
            for (int i = 0; i < num_level; ++i) {
                for (int j = 0; j < num_level; ++j) {
                    C(i, j, ks) = FP(0.0);
                    dC(i, j, ks) = FP(0.0);
                }
            }

            compute_C_ne_pert(
                atmos,
                H_atom,
                n_star_slice,
                nh_lte,
                ks,
                C,
                dC, // This contains C_ne_pert
                ne_pert_size
            );

            const fp_t ne_k = atmos.ne(ks);
            const fp_t recip_dNe = FP(1.0) / (ne_pert_size * ne_k);
            for (int i = 0; i < num_level; ++i) {
                for (int j = 0; j < num_level; ++j) {
                    fp_t dCdNe = (dC(i, j, ks) - C(i, j, ks)) * recip_dNe;
                    dC(i, j, ks) = dCdNe;
                }
            }

            for (int i = 0; i < num_level; ++i) {
                fp_t diag = FP(0.0);
                fp_t diag_dC = FP(0.0);
                C(i, i, ks) = diag;
                dC(i, i, ks) = diag_dC;
                for (int j = 0; j < num_level; ++j) {
                    diag += C(j, i, ks);
                    diag_dC += dC(j, i, ks);
                }
                C(i, i, ks) = -diag;
                dC(i, i, ks) = -diag_dC;
            }
        }
    );
    Kokkos::fence();

    Kokkos::parallel_for(
        "Charge Conservation",
        TeamPolicy(num_space, std::min(square(num_eqn), 128)).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA (const KTeam& team) {
            const i64 ks = team.league_rank();

            ScratchView<fp_t**> Ck(team.team_scratch(0), num_level, num_level);
            ScratchView<fp_t**> dCk(team.team_scratch(0), num_level, num_level);
            const fp_t ne_k = atmos.ne(ks);

            ScratchView<T**> Gammak(team.team_scratch(0), num_level, num_level);
            ScratchView<T*> new_popsk(team.team_scratch(0), num_level);
            ScratchView<T*> prev_popsk(team.team_scratch(0), num_level);
            // Copy over Gamma and new_pops chunks
            Kokkos::parallel_for(
                Kokkos::TeamVectorRange(team, nlxnl_loop.num_iter),
                [&] (const int x) {
                    const auto args = nlxnl_loop.unpack(x);
                    const int i = args[0];
                    const int j = args[1];

                    Gammak(i, j) = GammaH_flat(i, j, ks);
                    Ck(i, j) = C(i, j, ks);
                    dCk(i, j) = dC(i, j, ks);
                    if (i == 0) {
                        new_popsk(j) = pops(j, ks);
                    }
                }
            );

            if (theta < FP(1.0)) {
                Kokkos::parallel_for(
                    Kokkos::TeamVectorRange(team, num_level),
                    [&] (const int i) {
                        prev_popsk(i) = (FP(1.0) - theta) * dt * predicted_pops(i, ks) + prev_pops(i, ks);
                    }
                );
            } else {
                Kokkos::parallel_for(
                    Kokkos::TeamVectorRange(team, num_level),
                    [&] (const int i) {
                        prev_popsk(i) = prev_pops(i, ks);
                    }
                );
            }
            team.team_barrier();

            // Fixup gamma
            Kokkos::parallel_for(
                Kokkos::TeamVectorRange(team, num_level),
                [&] (const int i) {
                    T diag = T(FP(0.0));
                    Gammak(i, i) = diag;
                    for (int j = 0; j < num_level; ++j) {
                        diag += Gammak(j, i);
                    }
                    Gammak(i, i) = -diag;
                }
            );
            team.team_barrier();

            ScratchView<T**> dF(team.team_scratch(0), num_eqn, num_eqn);
            ScratchView<T*> F(team.team_scratch(0), num_eqn);
            // Compute LHS, based on Lightspinner impl
            Kokkos::parallel_for(
                Kokkos::TeamVectorRange(team, num_eqn),
                [&] (const int i) {
                    if (i < (num_level - 1)) {
                        T Fi = FP(0.0);
                        for (int j = 0; j < num_level; ++j) {
                            Fi -= Gammak(i, j) * new_popsk(j);
                        }
                        Fi *= theta * dt;
                        Fi += new_popsk(i) - prev_popsk(i);
                        F(i) = Fi;
                    } else if (i == (num_level - 1)) {
                        if (conserve_pressure) {
                            using ConstantsFP::k_B;
                            T N = pressure(ks) / (k_B * temperature(ks));
                            T dntot = N;
                            for (int j = 0; j < num_level; ++j) {
                                dntot -= total_abund * new_popsk(j);
                            }
                            dntot -= ne_k;
                            F(i) = -dntot;
                        } else {
                            T dntot = H_atom.abundance * nh_tot(ks);
                            for (int j = 0; j < num_level; ++j) {
                                dntot -= new_popsk(j);
                            }
                            F(i) = -dntot;
                        }
                    } else if (i == (num_eqn - 1)) {
                        T charge = FP(0.0);
                        for (int j = 0; j < num_level; ++j) {
                            charge += (H_atom.stage(j) - FP(1.0)) * new_popsk(j);
                        }
                        charge -= ne_k;
                        F(i) = charge;
                    }
                }
            );

            // Compute Jacobian dF
            Kokkos::parallel_for(
                Kokkos::TeamVectorRange(team, nexne_loop.num_iter),
                [&] (const int x) {
                    const auto args = nexne_loop.unpack(x);
                    const int i = args[0];
                    const int j = args[1];
                    if (i < num_level && j < num_level) {
                        dF(i, j) = theta * Gammak(i, j) * dt;
                        if (i == j) {
                            dF(i, j) -= FP(1.0);
                        }
                    } else {
                        dF(i, j) = T(0);
                    }
                }
            );
            team.team_barrier();

            Kokkos::parallel_for(
                Kokkos::TeamVectorRange(team, nlxnl_loop.num_iter),
                [&] (const int x) {
                    const auto args = nlxnl_loop.unpack(x);
                    const int i = args[0];
                    const int j = args[1];
                    const int num_cont = H_atom.continua.extent(0);
                    if (x < num_cont) {
                        const int kr = x;
                        const auto& cont = H_atom.continua(kr);
                        const T precon_Rji = Gammak(cont.i, cont.j) - Ck(cont.i, cont.j);
                        T entry = -(precon_Rji / ne_k) * new_popsk(cont.j);
                        entry *= -theta * dt;
                        Kokkos::atomic_add(&dF(cont.i, num_eqn-1), entry);
                    }

                    T entry = -dCk(i, j) * new_popsk(j);
                    entry *= -theta * dt;
                    Kokkos::atomic_add(&dF(i, num_eqn-1), entry);
                }
            );
            team.team_barrier();

            // Setup conservation equations
            Kokkos::parallel_for(
                Kokkos::TeamVectorRange(team, nexne_loop.num_iter),
                [&] (const int x) {
                    const auto args = nexne_loop.unpack(x);
                    const int i = args[0];
                    const int j = args[1];

                    if (i == (num_level-1) && j < num_level) {
                        if (conserve_pressure) {
                            // NOTE(cmo): Pressure conservation eqn
                            dF(i, j) = total_abund;
                        } else {
                            // NOTE(cmo): Number conservation eqn for H
                            dF(i, j) = FP(1.0);
                        }
                    } else if (i == (num_level-1) && j == num_level) {
                        if (conserve_pressure) {
                            // NOTE(cmo): Pressure conservation eqn (ne term)
                            dF(i, j) = FP(1.0);
                        } else {
                            // NOTE(cmo): Number conservation eqn for H
                            dF(i, j) = FP(0.0);
                        }
                    } else if (i == (num_eqn - 1) && j < num_level) {
                        dF(i, j) = -(H_atom.stage(j) - FP(1.0));
                    } else if (i == (num_eqn - 1) && j == (num_eqn - 1)) {
                        dF(i, j) = FP(1.0);
                    }
                }
            );
            team.team_barrier();

            ScratchView<T**> dF_copy;
            ScratchView<T*> lhs;
            ScratchView<T*> residuals;
            if (iterative_improvement) {
                dF_copy = ScratchView<T**>(team.team_scratch(0), num_eqn, num_eqn);
                lhs = ScratchView<T*>(team.team_scratch(0), num_eqn);
                residuals = ScratchView<T*>(team.team_scratch(0), num_eqn);

                Kokkos::parallel_for(
                    Kokkos::TeamVectorRange(team, nexne_loop.num_iter),
                    [&] (const int x) {
                        const auto args = nexne_loop.unpack(x);
                        const int i = args[0];
                        const int j = args[1];

                        if (i == 0) {
                            lhs(j) = F(j);
                            residuals(j) = F(j);
                        }
                        dF_copy(i, j) = dF(i, j);
                    }
                );
                team.team_barrier();
            }


            // LU factorise
            KokkosBatched::LU<KTeam, KokkosBatched::Mode::Team, KokkosBatched::Algo::LU::Unblocked>::invoke(
                team, dF
            );
            team.team_barrier();
            // LU Solve
            KokkosBatched::TeamSolveLU<
                KTeam,
                KokkosBatched::Trans::NoTranspose,
                KokkosBatched::Algo::Trsm::Unblocked
            >::invoke(
                team,
                dF,
                F
            );
            team.team_barrier();

            if (iterative_improvement) {
                for (int refinement = 0; refinement < num_refinement_passes; ++refinement) {
                    // r_i = b_i
                    Kokkos::parallel_for(
                        Kokkos::TeamVectorRange(team, residuals.extent(0)),
                        [&] (int i) {
                            residuals(i) = lhs(i);
                        }
                    );
                    team.team_barrier();
                    // r -= dF @ x
                    KokkosBlas::Experimental::Gemv<KokkosBlas::Mode::TeamVector, KokkosBlas::Algo::Gemv::Default>::invoke(
                        team,
                        'n',
                        T(-1),
                        dF_copy,
                        F,
                        T(1),
                        residuals
                    );
                    team.team_barrier();
                    // Solve dF F' = r (already factorised)
                    KokkosBatched::TeamSolveLU<
                        KTeam,
                        KokkosBatched::Trans::NoTranspose,
                        KokkosBatched::Algo::Trsm::Unblocked
                    >::invoke(
                        team,
                        dF,
                        residuals
                    );
                    team.team_barrier();
                    // F += F'
                    Kokkos::parallel_for(
                        Kokkos::TeamVectorRange(
                            team,
                            new_popsk.extent(0)
                        ),
                        [&] (int i) {
                            F(i) += residuals(i);
                        }
                    );
                    team.team_barrier();
                }
            }

            // Store result
            Kokkos::parallel_for(
                Kokkos::TeamVectorRange(team, num_eqn),
                [&] (const int i) {
                    new_F(ks, i) = F(i);
                }
            );
            team.team_barrier();
        }
    );
    Kokkos::fence();

    const auto& F = new_F;
    typedef Kokkos::MaxLoc<fp_t, Kokkos::pair<int, int>> Reducer;
    typedef Reducer::value_type ReductionVal;
    ReductionVal max_change_loc;
    dex_parallel_reduce(
        "Update pops and compute max change",
        FlatLoop<2>(num_space, num_eqn),
        KOKKOS_LAMBDA (int ks, int i, ReductionVal& rval) {
            fp_t change = FP(0.0);
            fp_t step_size = FP(1.0);

            constexpr bool clamp_step_size = true;
            if (i < num_level) {
                fp_t update = F(ks, i);
                fp_t updated = pops(i, ks) + update;
                if (clamp_step_size && updated < FP(0.0)) {
                    fp_t ne_update = F(ks, num_eqn-1);
                    step_size = std::max(FP(0.95) * ne(ks) / std::abs(ne_update), FP(1e-4));
                    update *= step_size;
                }
                if (pops(i, ks) > (ignore_change_below_ntot_frac * nh_tot(ks))) {
                    change = std::abs(update / (pops(i, ks)));
                }
                pops(i, ks) += update;
                // if (pops(i, ks) < FP(1e-3) || std::isnan(pops(i, ks))) {
                //     pops(i, ks) = FP(1e-3);
                // }
            } else {
                fp_t ne_update = F(ks, num_eqn-1);
                fp_t updated = ne(ks) + ne_update;
                if (clamp_step_size && updated < FP(0.0)) {
                    step_size = std::max(FP(0.95) * ne(ks) / std::abs(ne_update), FP(1e-4));
                    ne_update *= step_size;
                }
                change = std::abs(ne_update / ne(ks));
                ne(ks) += ne_update;
                // if (ne(ks) < FP(1e-3) || std::isnan(ne(ks))) {
                //     ne(ks) = FP(1e-3);
                // }
            }

            // reduce update
            if (change > rval.val) {
                rval.val = change;
                rval.loc = Kokkos::make_pair(ks, i);
            }
        },
        Reducer(max_change_loc)
    );
    Kokkos::fence();

    if (conserve_pressure) {
        const auto& full_pops = state.pops;
        dex_parallel_for(
            "Update and rescale pops (pressure)",
            FlatLoop<1>(num_space),
            KOKKOS_LAMBDA (i64 k) {
                fp_t pops_sum = FP(0.0);
                for (int i = 0; i < num_level; ++i) {
                    pops_sum += pops(i, k);
                }
                fp_t nh_tot_ratio = pops_sum / nh_tot(k);
                nh_tot(k) = pops_sum;

                for (int i = 0; i < full_pops.extent(0); ++i) {
                    full_pops(i, k) *= nh_tot_ratio;
                }
            }
        );
        Kokkos::fence();
    }

    fp_t max_change = max_change_loc.val;

    int max_change_level = max_change_loc.loc.second;
    i64 max_change_ks = max_change_loc.loc.first;
    state.println(
        "     NR Update Max Change (level: {}): {} (@ {})",
        max_change_level == (num_eqn - 1) ? "n_e": std::to_string(max_change_level),
        max_change,
        max_change_ks
    );
    yakl::timer_stop("Charge conservation");
    return max_change;
}

template <typename State>
fp_t time_dep_nr_post_update(State& state, const Fp2d& prev_pops, const TimeDepNrPostUpdateOptions& args) {
#ifdef HAVE_MPI
    fp_t max_rel_change;
    if (state.mpi_state.rank == 0) {
        max_rel_change = time_dep_nr_post_update_impl<StatEqPrecision>(state, prev_pops, args);
    }
    MPI_Bcast(&max_rel_change, 1, get_FpMpi(), 0, state.mpi_state.comm);
    return max_rel_change;
#else
    return time_dep_nr_post_update_impl<StatEqPrecision>(state, prev_pops, args);
#endif

}

template fp_t time_dep_update<State>(State& state, const Fp2d& prev_pops, const KineticEqOptions& args);
template fp_t time_dep_nr_post_update<State>(State& state, const Fp2d& prev_pops, const TimeDepNrPostUpdateOptions& args);