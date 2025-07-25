#if !defined(MOSSCAP_TYPES_HPP)
#define MOSSCAP_TYPES_HPP
#include "Config.hpp"
#include "LoopUtils.hpp"

constexpr auto memDevice = yakl::memDevice;
// constexpr auto memDevice = yakl::memHost;
typedef yakl::Array<fp_t, 1, memDevice> Fp1d;
typedef yakl::Array<fp_t, 2, memDevice> Fp2d;
typedef yakl::Array<fp_t, 3, memDevice> Fp3d;
typedef yakl::Array<fp_t, 4, memDevice> Fp4d;
typedef yakl::Array<fp_t, 5, memDevice> Fp5d;

typedef yakl::Array<fp_t const, 1, memDevice> FpConst1d;
typedef yakl::Array<fp_t const, 2, memDevice> FpConst2d;
typedef yakl::Array<fp_t const, 3, memDevice> FpConst3d;
typedef yakl::Array<fp_t const, 4, memDevice> FpConst4d;
typedef yakl::Array<fp_t const, 5, memDevice> FpConst5d;

typedef yakl::Array<fp_t, 1, yakl::memHost> Fp1dHost;
typedef yakl::Array<fp_t, 2, yakl::memHost> Fp2dHost;
typedef yakl::Array<fp_t, 3, yakl::memHost> Fp3dHost;
typedef yakl::Array<fp_t, 4, yakl::memHost> Fp4dHost;
typedef yakl::Array<fp_t, 5, yakl::memHost> Fp5dHost;

typedef yakl::Array<fp_t const, 1, yakl::memHost> FpConst1dHost;
typedef yakl::Array<fp_t const, 2, yakl::memHost> FpConst2dHost;
typedef yakl::Array<fp_t const, 3, yakl::memHost> FpConst3dHost;
typedef yakl::Array<fp_t const, 4, yakl::memHost> FpConst4dHost;
typedef yakl::Array<fp_t const, 5, yakl::memHost> FpConst5dHost;

typedef yakl::SArray<fp_t, 1, 2> vec2;
typedef yakl::SArray<fp_t, 1, 3> vec3;
typedef yakl::SArray<fp_t, 1, 4> vec4;
template <int N>
using vec = yakl::SArray<fp_t, 1, N>;
typedef yakl::SArray<fp_t, 2, 2, 2> mat2x2;
typedef yakl::SArray<int, 2, 2, 2> imat2x2;
typedef yakl::SArray<int32_t, 1, 2> ivec2;
typedef yakl::SArray<int32_t, 1, 3> ivec3;
template <int N>
using ivec = yakl::SArray<int32_t, 1, N>;

typedef Kokkos::LayoutRight Layout;
template <class T, typename... Args>
using KView = Kokkos::View<T, Layout, Args...>;

typedef Kokkos::DefaultExecutionSpace::memory_space DefaultMemSpace;
typedef Kokkos::HostSpace HostSpace;
constexpr bool HostDevSameSpace = std::is_same_v<DefaultMemSpace, HostSpace>;

typedef Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> TeamPolicy;
typedef TeamPolicy::member_type KTeam;
using Kokkos::DefaultExecutionSpace;
typedef DefaultExecutionSpace::scratch_memory_space KScratchSpace;

template <class T>
using KScratchView = KView<T, KScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

struct CellIndex {
    i32 i; // x index
    i32 j; // y index
    i32 k; // z index
};

/// Access the fields at a location given by idx
struct QtyView {
    const Fp4d& q;
    const CellIndex idx;

    KOKKOS_INLINE_FUNCTION QtyView(const Fp4d& q_, const CellIndex idx_) : q(q_), idx(idx_) {}

    KOKKOS_INLINE_FUNCTION fp_t& operator()(const int f) const {
        return q(f, idx.k, idx.j, idx.i);
    }
};

template <typename T>
constexpr KOKKOS_INLINE_FUNCTION T square(const T t) {
    return t * t;
}

template <typename T>
constexpr KOKKOS_INLINE_FUNCTION T cube(const T t) {
    return t * t * t;
}


#else
#endif