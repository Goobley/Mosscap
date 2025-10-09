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
    // typedef f32 fp_t;
    #define CMO_EXPAND(x) x
    #ifdef _MSC_VER
        #define FP(x) (CMO_EXPAND(x)##f)
    #else
        #define FP(x) (x##f)
    #endif
    using fp_t = f32;
#else
    // typedef double fp_t;
    #define FP(x) (x)
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