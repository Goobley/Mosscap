#if !defined(MOSSCAP_ATMOS_COMMON_HPP)
#define MOSSCAP_ATMOS_COMMON_HPP

#include "YAKL.h"
#include "Config.hpp"

namespace Mosscap {

KOKKOS_INLINE_FUNCTION fp_t vturb_fn(fp_t temperature, fp_t nh_tot, fp_t ne) {
    return 2e3_fp;
}

}

#else
#endif
