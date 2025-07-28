#if !defined(MOSSCAP_SIMULATION_HPP)

#include "Types.hpp"
#include "State.hpp"
#include "Reconstruct.hpp"
#include "Boundaries.hpp"

struct ScratchSpace {
    Fp4d RR; /// Left-hand reconstruction [w, k, j, i]
    Fp4d RL; /// Right-hand reconstruction [w, k, j, i]
    Fp4d Fx; /// x flux
    Fp4d Fy; /// y flux
    Fp4d Fz; /// z flux
};

struct Simulation {
    i64 current_step;
    fp_t max_cfl;
    fp_t time;
    fp_t max_time;
    State state;
    ScratchSpace scratch;
    std::function<fp_t(const State&)> compute_timestep;
    std::function<void(const State&)> time_integrate;
    void step();
    void write();
};

#else
#endif