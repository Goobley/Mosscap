#if !defined(MOSSCAP_DEX_INTERFACE)
#define MOSSCAP_DEX_INTERFACE

#include "Config.hpp"
#include "../DexRT/source/State.hpp"
#include "../DexRT/source/CascadeState.hpp"

#ifdef DEXRT_USE_MAGMA
#error "MAGMA not supported for Dex in the Mosscap integration"
#endif

namespace YAML { struct Node; }

namespace Mosscap {

// NOTE(cmo): Only supporting 2D for now
using DexState = ::State;
using DexCascState = ::CascadeState;

struct Simulation;

struct DexInterface {
    DexState state;
    DexCascState casc_state;

    bool init(Simulation& sim, YAML::Node& config);
    bool init_atmosphere(Simulation& sim, i32 max_mip_level);
    bool update_atmosphere(Simulation& sim);
};

template <typename T, typename U, int rank, int mem_space>
inline
yakl::Array<U, rank, mem_space> maybe_convert_fp_array(const yakl::Array<T, rank, mem_space>& in) {
    if constexpr (std::is_same<T, U>::value) {
        return in;
    }

    yakl::Array<U, rank, mem_space> result(in.label(), yakl::DimsT<i64>(in.get_dimensions()));
    Kokkos::parallel_for<
        std::conditional_t<
            mem_space == yakl::memDevice,
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultHostExecutionSpace
        >
    >(
        "Convert and copy array",
        in.size(),
        KOKKOS_LAMBDA (i64 i) {
            result.data()[i] = U(in.data()[i]);
        }
    );
    Kokkos::fence();
    return result;
}

KOKKOS_INLINE_FUNCTION fp_t vturb_fn(fp_t temperature, fp_t nh_tot, fp_t ne) {
    return 2e3_fp;
}

}

#else
#endif