#if !defined(MOSSCAP_CONFIG_HPP)
#define MOSSCAP_CONFIG_HPP
#include <cstdint>

typedef int32_t i32;
typedef int64_t i64;
typedef uint32_t u32;
typedef uint64_t u64;
typedef float f32;
typedef double f64;

namespace Mosscap {
#ifdef MOSSCAP_SINGLE_PRECISION
    using fp_t = f32;
#else
    using fp_t = f64;
#endif

    // NOTE(cmo): I didn't realise custom number literals were possible until
    // amrex. I though it was strings only!
    inline namespace literals {
        constexpr fp_t operator""_fp(long double x) {
            return fp_t(x);
        }
    }
}

#else
#endif