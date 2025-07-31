#include "Eos.hpp"
#include "Simulation.hpp"
#include "yaml-cpp/yaml.h"
#include "MosscapConfig.hpp"

bool Eos::init(const Simulation& sim, const YAML::Node& config) {
    std::string eos_str = get_or<std::string>(config, "eos.type", "ideal");
    EosType type = find_associated_enum<EosType>(EosTypeName, NumEosType, eos_str);

    switch (type) {
        case EosType::Ideal: {
            fp_t gamma = get_or<fp_t>(config, "eos.gamma", FP(1.4));
            return init_ideal(gamma);
        } break;
    }

    return true;
}