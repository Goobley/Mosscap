#if !defined(MOSSCAP_GLM_MHD_HPP)
#define MOSSCAP_GLM_MHD_HPP


namespace Mosscap {

struct Simulation;
void update_glm_ch(Simulation& sim);

template <int Axis, typename FTraits, typename WType>
KOKKOS_INLINE_FUNCTION void glm_set_mid_state(const fp_t c_h, WType& wL, WType& wR) {
    if constexpr (!is_instance(FTraits::fluid_type, FluidType::GlmMhd)) {
        return;
    } else {
        using Prim = FTraits::prim;
        constexpr int IB1 = MagneticField<Axis, FTraits>();
        const fp_t dB = wR(IB1) - wL(IB1);
        const fp_t dpsi = wR(I(Prim::Psi)) - wL(I(Prim::Psi));

        const fp_t Bm = 0.5_fp * (wL(IB1) + wR(IB1)) - 0.5_fp * dpsi / c_h;
        const fp_t psi_m = 0.5_fp * (wL(I(Prim::Psi)) + wR(I(Prim::Psi))) - 0.5_fp * c_h * dB;
        wL(IB1) = Bm;
        wR(IB1) = Bm;
        wL(I(Prim::Psi)) = psi_m;
        wR(I(Prim::Psi)) = psi_m;
    }
}

}


#else
#endif