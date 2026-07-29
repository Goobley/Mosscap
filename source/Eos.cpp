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
    // NOTE(claude): eos.avg_mass was always the mass per hydrogen atom, never
    // an average particle mass, so it has been renamed. The old key is still
    // accepted, with a warning.
    const bool has_new_key = has_key(config, "eos.mass_per_h");
    const bool has_old_key = has_key(config, "eos.avg_mass");
    if (has_new_key && has_old_key) {
        throw std::runtime_error(
            "Both eos.mass_per_h and the deprecated eos.avg_mass are set. Remove eos.avg_mass."
        );
    }
    if (has_old_key) {
        fmt::println(
            "WARNING: eos.avg_mass is deprecated and will be removed; it is the mass per "
            "hydrogen atom, not an average particle mass. Rename it to eos.mass_per_h."
        );
        mass_per_h = get_or<fp_t>(config, "eos.avg_mass", 1.0_fp);
    } else {
        mass_per_h = get_or<fp_t>(config, "eos.mass_per_h", 1.0_fp);
    }
    this->type = type;

    switch (type) {
        case EosType::Ideal: {
            fp_t gamma = get_or<fp_t>(config, "eos.gamma", 1.4_fp);
            fp_t ion_frac = get_or<fp_t>(config, "eos.ion_frac", 1.0_fp);
            return init_ideal(gamma, ion_frac, sim);
        } break;
        case EosType::AnalyticLteH: {
            fp_t gamma = get_or<fp_t>(config, "eos.gamma", 5.0_fp / 3.0_fp);
            bool include_ionisation_energy = get_or<bool>(config, "eos.include_ionisation_energy", false);
            has_ion_e = include_ionisation_energy;
            return init_analytic_lte_h(gamma, sim, include_ionisation_energy);
        } break;
        case EosType::TabulatedLteH: {
            fp_t gamma = get_or<fp_t>(config, "eos.gamma", 5.0_fp / 3.0_fp);
            std::string eos_table = get_or<std::string>(config, "eos.table_path", "mosscap_lte_h_tables.nc");
            return init_tabulated_lte_h(gamma, sim, eos_table);
        } break;
        case EosType::TracerEos: {
            fp_t gamma = get_or<fp_t>(config, "eos.gamma", 5.0_fp / 3.0_fp);
            has_ion_e = true;
            fp_t temperature_floor = get_or<fp_t>(config, "eos.min_temperature", 2e3);
            bool renorm_tracer_hpops = get_or<bool>(config, "eos.renorm_tracer_hpops", false);

            return init_tracer(gamma, sim, temperature_floor, renorm_tracer_hpops);
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
        invoke_fluid_traits(sim.num_dim, sim.fluid_type, [&]<typename FTraits>(FTraits) {
            lte_eos.update_eos<FTraits>(sim);
        });
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
        invoke_fluid_traits(sim.num_dim, sim.fluid_type, [&]<typename FTraits>(FTraits) {
            lte_h.update_eos<FTraits>(sim);
        });
    };

    return true;
}

bool Eos::init_tracer(fp_t gamma_, Simulation& sim, fp_t min_temperature, bool renorm_tracer_hpops) {
    is_constant = false;
    gamma = gamma_;

    if (sim.num_dim != 2 && sim.dex.interface_config.enable) {
        throw std::runtime_error("Tracer EOS only supports 2D models with dex enabled.");
    }

    const auto& sz = sim.state.sz;
    y_space = Fp3d("y_space", sz.zc, sz.yc, sz.xc);
    y_space = 1.0_fp;

    if (min_temperature > 0.0_fp) {
        floor_heat = Fp3d("eos_floor_heat", sz.zc, sz.yc, sz.xc);
        floor_heat = 0.0_fp;
    }

    DexPressureEosOptions opts;
    opts.renorm_tracer_hpops = renorm_tracer_hpops;
    DexPressureEos dex_eos;
    dex_eos.init(sim, min_temperature, opts);

    sim.update_eos = [dex_eos](const Simulation& sim) {
        invoke_fluid_traits(sim.num_dim, sim.fluid_type, [&]<typename FTraits>(FTraits) {
            dex_eos.update_eos<FTraits>(sim);
        });
    };

    return true;
}

}