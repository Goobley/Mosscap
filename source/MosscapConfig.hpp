#if !defined(MOSSCAP_MOSSCAP_CONFIG_HPP)
#define MOSSCAP_MOSSCAP_CONFIG_HPP

#include "yaml-cpp/yaml.h"
#include "fmt/core.h"
#include <cstddef>
#include <string_view>
#include <string>
#include <algorithm>

namespace impl
{
    // https://stackoverflow.com/a/59522794
    template <typename T>
    [[nodiscard]] constexpr std::string_view RawTypeName()
    {
        #ifndef _MSC_VER
        return __PRETTY_FUNCTION__;
        #else
        return __FUNCSIG__;
        #endif
    }

    struct TypeNameFormat
    {
        std::size_t junk_leading = 0;
        std::size_t junk_total = 0;
    };

    constexpr TypeNameFormat type_name_format = []{
        TypeNameFormat ret;
        std::string_view sample = RawTypeName<int>();
        ret.junk_leading = sample.find("int");
        ret.junk_total = sample.size() - 3;
        return ret;
    }();
    static_assert(type_name_format.junk_leading != std::size_t(-1), "Unable to determine the type name format on this compiler.");

    template <typename T>
    static constexpr auto type_name_storage = []{
        std::array<char, RawTypeName<T>().size() - type_name_format.junk_total + 1> ret{};
        std::copy_n(RawTypeName<T>().data() + type_name_format.junk_leading, ret.size() - 1, ret.data());
        return ret;
    }();
}

template <typename T>
[[nodiscard]] constexpr std::string_view type_name()
{
    return {impl::type_name_storage<T>.data(), impl::type_name_storage<T>.size() - 1};
}

template <typename T>
[[nodiscard]] constexpr const char* type_name_c_str()
{
    return impl::type_name_storage<T>.data();
}

/// Return the config variable at `id` of the form x.y.z as type T, or `default_val` if not present
template <typename T>
T get_or(const YAML::Node& f, const std::string& id, const T& default_val) {
    size_t pos = id.find('.');
    if (pos != std::string::npos) {
        // split head and tail around the .: head.tail.x.y.z -> head, tail.x.y.z
        std::string head(id.substr(0, pos));
        std::string tail(id.substr(pos+1));
        if (f[head]) {
            return get_or(f[head], tail, default_val);
        }
        return default_val;
    }

    if (f[id]) {
        try {
            return f[id].as<T>();
        } catch (const YAML::BadConversion& e) {
            throw std::runtime_error(
                fmt::format(
                    "Unable to convert key \"{}\" to type \"{}\". Has value: \"{}\".",
                    id,
                    type_name<T>(),
                    f[id].as<std::string>()
                )
            );
        }
    }

    return default_val;
}

/// Set the config variable at `id` of the form x.y.z to val if it doesn't already exist.
template <typename T>
void set_if_not_present(YAML::Node& f, const std::string& id, const T& val) {
    size_t pos = id.find('.');
    if (pos != std::string::npos) {
        // split head and tail around the .: head.tail.x.y.z -> head, tail.x.y.z
        std::string head(id.substr(0, pos));
        std::string tail(id.substr(pos+1));
        auto node = f[head];
        return set_if_not_present(node, tail, val);
    }
    f[id] = val;
}

template <typename T>
T find_associated_enum(const char* const* names, int num_names, std::string to_compare) {
    std::transform(
        to_compare.begin(),
        to_compare.end(),
        to_compare.begin(),
        [](char c) { return std::tolower(c); }
    );

    for (int i = 0; i < num_names; ++i) {
        if (to_compare == names[i]) {
            return T(i);
        }
    }

    throw std::runtime_error(fmt::format("Unable to find associated key in \"{}\" for \"{}\".", type_name<T>(), to_compare));
}

#else
#endif