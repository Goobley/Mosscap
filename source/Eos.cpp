#include "Eos.hpp"
#include "Simulation.hpp"
#include "yaml-cpp/yaml.h"
#include "MosscapConfig.hpp"
#include "YAKL_netcdf.h"
#include "TabulatedLteH.hpp"
#include "AnalyticLteH.hpp"
#include "DexrtEos.hpp"

namespace Mosscap {

bool Eos::init(Simulation& sim, const YAML::Node& config) {
    std::string eos_str = get_or<std::string>(config, "eos.type", "ideal");
    EosType type = find_associated_enum<EosType>(EosTypeName, NumEosType, eos_str);
    avg_mass = get_or<fp_t>(config, "eos.avg_mass", 1.0_fp);

    switch (type) {
        case EosType::Ideal: {
            fp_t gamma = get_or<fp_t>(config, "eos.gamma", 1.4_fp);
            fp_t ion_frac = get_or<fp_t>(config, "eos.ion_frac", 1.0_fp);
            return init_ideal(gamma, ion_frac, sim);
        } break;
        case EosType::AnalyticLteH: {
            fp_t gamma = get_or<fp_t>(config, "eos.gamma", 5.0_fp / 3.0_fp);
            bool include_ionisation_energy = get_or<bool>(config, "eos.include_ionisation_energy", false);
            return init_analytic_lte_h(gamma, sim, include_ionisation_energy);
        } break;
        case EosType::TabulatedLteH: {
            fp_t gamma = get_or<fp_t>(config, "eos.gamma", 5.0_fp / 3.0_fp);
            std::string eos_table = get_or<std::string>(config, "eos.table_path", "mosscap_lte_h_tables.nc");
            return init_tabulated_lte_h(gamma, sim, eos_table);
        } break;
        case EosType::DexPressure: {
            fp_t gamma = get_or<fp_t>(config, "eos.gamma", 5.0_fp / 3.0_fp);
            return init_dexrt(gamma, sim);
        } break;
    }

    return true;
}

bool Eos::init_analytic_lte_h(fp_t gamma_, Simulation& sim, bool include_ionisation_energy) {
    is_constant = false;
    gamma = gamma_;
    const auto& sz = sim.state.sz;
    y_space = Fp3d("y_space", sz.zc, sz.yc, sz.xc);
    T_space = Fp3d("T_space", sz.zc, sz.yc, sz.xc);

    AnalyticLteH lte_eos;
    lte_eos.init(include_ionisation_energy);

    sim.update_eos = [lte_eos](const Simulation& sim) {
        if (sim.num_dim == 1) {
            lte_eos.update_eos<1>(sim);
        } else if (sim.num_dim == 2) {
            lte_eos.update_eos<2>(sim);
        } else {
            lte_eos.update_eos<3>(sim);
        }
    };

    return true;
}

bool Eos::init_tabulated_lte_h(fp_t gamma_, Simulation& sim, const std::string& table_path) {
    is_constant = false;
    gamma = gamma_;

    const auto& sz = sim.state.sz;
    y_space = Fp3d("y_space", sz.zc, sz.yc, sz.xc);
    T_space = Fp3d("T_space", sz.zc, sz.yc, sz.xc);

    TabulatedLteH lte_h;
    lte_h.init(table_path);
    y_space = -1.0_fp;

    sim.update_eos = [lte_h](const Simulation& sim) {
        if (sim.num_dim == 1) {
            lte_h.update_eos<1>(sim);
        } else if (sim.num_dim == 2) {
            lte_h.update_eos<2>(sim);
        } else {
            lte_h.update_eos<3>(sim);
        }
    };
    return true;
}

bool Eos::init_dexrt(fp_t gamma_, Simulation& sim) {
    is_constant = false;
    gamma = gamma_;

    if (sim.num_dim != 2 && sim.dex.interface_config.enable) {
        throw std::runtime_error("Dex EOS only supports 2D models with dex enabled.");
    }

    const auto& sz = sim.state.sz;
    y_space = Fp3d("y_space", sz.zc, sz.yc, sz.xc);
    y_space = 1.0_fp;

    DexPressureEos dex_eos;
    dex_eos.init();

    sim.update_eos = [dex_eos](const Simulation& sim) {
        dex_eos.update_eos(sim);
    };

    return true;
}

}