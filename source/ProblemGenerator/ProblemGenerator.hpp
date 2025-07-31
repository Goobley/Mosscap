#if !defined(MOSSCAP_PROBLEM_GENERATOR_HPP)
#define MOSSCAP_PROBLEM_GENERATOR_HPP
#include <functional>
#include <map>
#include <yaml-cpp/yaml.h>
#include <fmt/core.h>
#include "../Simulation.hpp"
#include "../JasPP.hpp"

template <typename T>
struct DispatchFactory {
    std::map<std::string, std::function<void(T&, const YAML::Node&)>> registry;

    void dispatch(const std::string& identifier, T& arg, const YAML::Node& cfg) const {
        auto it = registry.find(identifier);
        if (it != registry.end()) {
            return it->second(arg, cfg);
        }
        throw std::runtime_error(fmt::format("No method for {} found.", identifier));
    };

    bool register_method(const std::string& identifier, const std::function<void(T&, const YAML::Node&)>& constructor) {
        auto it = registry.find(identifier);
        if (it != registry.end()) {
            return false;
        }
        registry[identifier] = constructor;
        return true;
    }
};

inline DispatchFactory<Simulation>& get_problem_generator() {
    // NOTE(cmo): This is a Meyer's singleton, it should be available at static init time (i.e. before main)
    static DispatchFactory<Simulation> factory;
    return factory;
}

#define MOSSCAP_PROB_FN_NAME(PROB_NAME) JasConcat(config_, PROB_NAME)
#define MOSSCAP_PROB_FN_SIGNATURE(PROB_NAME) void MOSSCAP_PROB_FN_NAME(PROB_NAME)(Simulation& sim, const YAML::Node& config)
#define MOSSCAP_PROB_FN_REGISTER(PROB_NAME) MOSSCAP_PROB_FN_SIGNATURE(PROB_NAME); \
static const bool JasConcat(register_, PROB_NAME) = get_problem_generator().register_method(JasStringify(PROB_NAME), MOSSCAP_PROB_FN_NAME(PROB_NAME));

// NOTE(cmo): The preamble is to try and ensure the compiler doesn't remove the
// static variables which otherwise aren't used.
#define MOSSCAP_PROBLEM_PREAMBLE(PROB_NAME) (void)JasConcat(register_, PROB_NAME); \
[[maybe_unused]] constexpr const char* PROBLEM_NAME = JasStringify(PROB_NAME)
#define MOSSCAP_NEW_PROBLEM(PROB_NAME) MOSSCAP_PROB_FN_REGISTER(PROB_NAME) \
MOSSCAP_PROB_FN_SIGNATURE(PROB_NAME)

#else
#endif