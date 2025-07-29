#include "TimeStepping.hpp"
#include "Simulation.hpp"
#include "Hydro.hpp"
#include "Boundaries.hpp"

template <int NumDim, typename Lambda>
void integrate_flux(const std::string& step_name, const GridSize& sz, const Fluxes& flux, const Lambda& updater) {
    int nx = sz.xc - 2 * sz.ng;
    int ny = std::max(sz.yc - 2 * sz.ng, 1);
    int nz = std::max(sz.zc - 2 * sz.ng, 1);
    dex_parallel_for(
        step_name,
        FlatLoop<3>(nz, ny, nx),
        KOKKOS_LAMBDA (int ki, int ji, int ii) {
            const int k = nz == 1 ? ki : ki + sz.ng;
            const int j = ny == 1 ? ji : ji + sz.ng;
            const int i = ii + sz.ng;
            #pragma unroll
            for (int var = 0; var < flux.Fx.extent(0); ++var) {
                fp_t q_update = FP(0.0);
                q_update += flux.Fx(var, k, j, i) - flux.Fx(var, k, j, i+1);
                if constexpr (NumDim > 1) {
                    q_update += flux.Fy(var, k, j, i) - flux.Fy(var, k, j+1, i);
                }
                if constexpr (NumDim > 2) {
                    q_update += flux.Fz(var, k, j, i) - flux.Fz(var, k+1, j, i);
                }
                updater(var, k, j, i, q_update);
            }
        }
    );
}

// NOTE(cmo): Due to C++ explicit specialisation rules, the time_step function
// must be fully defined before the associated init.
template <>
template <int NumDim>
void TimeStepper<TimeStepScheme::Rk2>::time_step(Simulation& sim, fp_t dt) {
    const auto& state = sim.state;
    const auto& Q = state.Q;
    const auto& Q_old = sim.ts_storage.Q_old[0];
    const auto& flux = sim.fluxes;

    Q.deep_copy_to(Q_old);
    compute_hydro_fluxes(sim);

    integrate_flux<NumDim>(
        "RK2 Step 0",
        state.sz,
        flux,
        KOKKOS_LAMBDA (const int var, const int k, const int j, const int i, fp_t q_update) {
            q_update *= dt / state.dx;
            state.Q(var, k, j, i) += q_update;
    });
    Kokkos::fence();

    fill_bcs(sim);
    compute_hydro_fluxes(sim);

    integrate_flux<NumDim>(
        "RK2 Step 1",
        state.sz,
        flux,
        KOKKOS_LAMBDA (const int var, const int k, const int j, const int i, fp_t q_update) {
            q_update *= dt / state.dx;
            state.Q(var, k, j, i) = FP(0.5) * (Q_old(var, k, j, i) + state.Q(var, k, j, i) + q_update);
    });
    Kokkos::fence();

    fill_bcs(sim);
    sim.time += dt;
    sim.current_step += 1;
}

template<>
bool TimeStepper<TimeStepScheme::Rk2>::init(Simulation& sim) {
    sim.ts_storage.Q_old.emplace_back(
        Fp4d(
            "Q_old_0",
            sim.state.Q.extent(0),
            sim.state.Q.extent(1),
            sim.state.Q.extent(2),
            sim.state.Q.extent(3)
        )
    );
    std::vector<std::function<void(Simulation&, fp_t)>> dimensioned_schemes = {
        TimeStepper<TimeStepScheme::Rk2>::time_step<1>,
        TimeStepper<TimeStepScheme::Rk2>::time_step<2>,
        TimeStepper<TimeStepScheme::Rk2>::time_step<3>
    };
    sim.time_step = dimensioned_schemes.at(sim.num_dim - 1);

    return true;
}

template <>
template <int NumDim>
void TimeStepper<TimeStepScheme::SspRk3>::time_step(Simulation& sim, fp_t dt) {
    const auto& state = sim.state;
    const auto& Q = state.Q;
    const auto& Q_old = sim.ts_storage.Q_old[0];
    const auto& flux = sim.fluxes;

    Q.deep_copy_to(Q_old);
    compute_hydro_fluxes(sim);

    integrate_flux<NumDim>(
        "SSPRK3 Step 0",
        state.sz,
        flux,
        KOKKOS_LAMBDA (const int var, const int k, const int j, const int i, fp_t q_update) {
            q_update *= dt / state.dx;
            state.Q(var, k, j, i) += q_update;
    });
    Kokkos::fence();

    fill_bcs(sim);
    compute_hydro_fluxes(sim);

    integrate_flux<NumDim>(
        "SSPRK3 Step 1",
        state.sz,
        flux,
        KOKKOS_LAMBDA (const int var, const int k, const int j, const int i, fp_t q_update) {
            q_update *= dt / state.dx;
            state.Q(var, k, j, i) = FP(0.75) * Q_old(var, k, j, i) + FP(0.25) * (state.Q(var, k, j, i) + q_update);
    });
    Kokkos::fence();

    fill_bcs(sim);
    compute_hydro_fluxes(sim);

    integrate_flux<NumDim>(
        "SSPRK3 Step 2",
        state.sz,
        flux,
        KOKKOS_LAMBDA (const int var, const int k, const int j, const int i, fp_t q_update) {
            q_update *= dt / state.dx;
            state.Q(var, k, j, i) = (FP(1.0) / FP(3.0)) * Q_old(var, k, j, i) + (FP(2.0) / FP(3.0)) * (state.Q(var, k, j, i) + q_update);
    });
    Kokkos::fence();

    fill_bcs(sim);
    sim.time += dt;
    sim.current_step += 1;
}

template<>
bool TimeStepper<TimeStepScheme::SspRk3>::init(Simulation& sim) {
    sim.ts_storage.Q_old.emplace_back(
        Fp4d(
            "Q_old_0",
            sim.state.Q.extent(0),
            sim.state.Q.extent(1),
            sim.state.Q.extent(2),
            sim.state.Q.extent(3)
        )
    );
    std::vector<std::function<void(Simulation&, fp_t)>> dimensioned_schemes = {
        TimeStepper<TimeStepScheme::SspRk3>::time_step<1>,
        TimeStepper<TimeStepScheme::SspRk3>::time_step<2>,
        TimeStepper<TimeStepScheme::SspRk3>::time_step<3>
    };
    sim.time_step = dimensioned_schemes.at(sim.num_dim - 1);

    return true;
}

template <>
template <int NumDim>
void TimeStepper<TimeStepScheme::SspRk4>::time_step(Simulation& sim, fp_t dt) {
    const auto& state = sim.state;
    const auto& Q = state.Q;
    const auto& Q_old = sim.ts_storage.Q_old[0];
    const auto& flux = sim.fluxes;

    Q.deep_copy_to(Q_old);
    compute_hydro_fluxes(sim);

    integrate_flux<NumDim>(
        "SSPRK4 Step 0",
        state.sz,
        flux,
        KOKKOS_LAMBDA (const int var, const int k, const int j, const int i, fp_t q_update) {
            q_update *= dt / state.dx;
            state.Q(var, k, j, i) += FP(0.5) * q_update;
    });
    Kokkos::fence();

    fill_bcs(sim);
    compute_hydro_fluxes(sim);

    integrate_flux<NumDim>(
        "SSPRK4 Step 1",
        state.sz,
        flux,
        KOKKOS_LAMBDA (const int var, const int k, const int j, const int i, fp_t q_update) {
            q_update *= dt / state.dx;
            state.Q(var, k, j, i) += FP(0.5) * q_update;
    });
    Kokkos::fence();

    fill_bcs(sim);
    compute_hydro_fluxes(sim);

    integrate_flux<NumDim>(
        "SSPRK4 Step 2",
        state.sz,
        flux,
        KOKKOS_LAMBDA (const int var, const int k, const int j, const int i, fp_t q_update) {
            q_update *= dt / state.dx;
            state.Q(var, k, j, i) = (FP(2.0) / FP(3.0)) * Q_old(var, k, j, i) + (FP(1.0) / FP(3.0)) * state.Q(var, k, j, i) + (FP(1.0) / (6.0)) * q_update;
    });
    Kokkos::fence();

    fill_bcs(sim);
    compute_hydro_fluxes(sim);

    integrate_flux<NumDim>(
        "SSPRK4 Step 3",
        state.sz,
        flux,
        KOKKOS_LAMBDA (const int var, const int k, const int j, const int i, fp_t q_update) {
            q_update *= dt / state.dx;
            state.Q(var, k, j, i) += (0.5) * q_update;
    });
    Kokkos::fence();


    fill_bcs(sim);
    sim.time += dt;
    sim.current_step += 1;
}

template<>
bool TimeStepper<TimeStepScheme::SspRk4>::init(Simulation& sim) {
    sim.ts_storage.Q_old.emplace_back(
        Fp4d(
            "Q_old_0",
            sim.state.Q.extent(0),
            sim.state.Q.extent(1),
            sim.state.Q.extent(2),
            sim.state.Q.extent(3)
        )
    );
    std::vector<std::function<void(Simulation&, fp_t)>> dimensioned_schemes = {
        TimeStepper<TimeStepScheme::SspRk4>::time_step<1>,
        TimeStepper<TimeStepScheme::SspRk4>::time_step<2>,
        TimeStepper<TimeStepScheme::SspRk4>::time_step<3>
    };
    sim.time_step = dimensioned_schemes.at(sim.num_dim - 1);
    return true;
}