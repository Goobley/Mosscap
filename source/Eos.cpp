#include "Eos.hpp"
#include "Simulation.hpp"
#include "yaml-cpp/yaml.h"
#include "MosscapConfig.hpp"
#include "YAKL_netcdf.h"
#include "TabulatedLteH.hpp"
#include "AnalyticLteH.hpp"

namespace Mosscap {

bool Eos::init(Simulation& sim, const YAML::Node& config) {
    std::string eos_str = get_or<std::string>(config, "eos.type", "ideal");
    EosType type = find_associated_enum<EosType>(EosTypeName, NumEosType, eos_str);
    avg_mass = get_or<fp_t>(config, "eos.avg_mass", FP(1.0));

    switch (type) {
        case EosType::Ideal: {
            fp_t gamma = get_or<fp_t>(config, "eos.gamma", FP(1.4));
            fp_t ion_frac = get_or<fp_t>(config, "eos.ion_frac", FP(1.0));
            return init_ideal(gamma, ion_frac, sim);
        } break;
        case EosType::AnalyticLteH: {
            fp_t gamma = get_or<fp_t>(config, "eos.gamma", FP(5.0) / FP(3.0));
            bool include_ionisation_energy = get_or<bool>(config, "eos.include_ionisation_energy", false);
            return init_analytic_lte_h(gamma, sim, include_ionisation_energy);
        } break;
        case EosType::TabulatedLteH: {
            fp_t gamma = get_or<fp_t>(config, "eos.gamma", FP(5.0) / FP(3.0));
            std::string eos_table = get_or<std::string>(config, "eos.table_path", "mosscap_lte_h_tables.nc");
            return init_tabulated_lte_h(gamma, sim, eos_table);
        } break;
    }

    return true;
}

bool Eos::init_analytic_lte_h(fp_t gamma, Simulation& sim, bool include_ionisation_energy) {
    is_constant = false;
    gamma = gamma;
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

bool Eos::init_tabulated_lte_h(fp_t gamma, Simulation& sim, const std::string& table_path) {
    is_constant = false;
    gamma = gamma;

    const auto& sz = sim.state.sz;
    y_space = Fp3d("y_space", sz.zc, sz.yc, sz.xc);
    T_space = Fp3d("T_space", sz.zc, sz.yc, sz.xc);

    TabulatedLteH lte_h;
    lte_h.init(table_path);
    y_space = FP(-1.0);

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

}