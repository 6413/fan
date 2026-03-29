module;
#include <string>
#include <string_view>
#include <sstream>
#include <type_traits>
#include <cstdint>

export module fan.formatter;

namespace fan::detail {
  template <typename T>
  std::string to_str(const T& v) {
    if constexpr (std::is_same_v<std::decay_t<T>, std::string>) return v;
    else if constexpr (std::is_same_v<std::decay_t<T>, std::string_view>) return std::string(v);
    else if constexpr (std::is_same_v<std::decay_t<T>, const char*>) return v ? v : "";
    else if constexpr (std::is_same_v<std::decay_t<T>, char>) return std::string(1, v);
    else if constexpr (std::is_arithmetic_v<std::decay_t<T>>) return std::to_string(v);
    else if constexpr (std::is_convertible_v<T, std::string>) return std::string(v);
    else {
      std::ostringstream oss;
      oss << v;
      return oss.str();
    }
  }
}

export namespace fan {
  template <typename ...Args>
  std::string format_join(const char* sep, const Args&... args) {
    std::string result;
    int idx = 0;
    ((result += fan::detail::to_str(args) + (++idx == (int)sizeof...(args) ? "" : sep)), ...);
    return result;
  }

  template <typename ...Args> std::string format_args(const Args&... args) { return format_join(" ", args...); }
  template <typename ...Args> std::string format_args_raw(const Args&... args) { return format_join("", args...); }
  template <typename ...Args> std::string format_args_comma(const Args&... args) { return format_join(", ", args...); }
  template <typename ...Args> std::string format_args_no_space(const Args&... args) { return format_join("", args...); }
  template <typename ...Args> std::string format_args_with_space(const Args&... args) { return format_join(" ", args...); }
  template <typename ...Args> std::string format_error_args(const Args&... args) { return format_join(" ", args...); }

  template<typename T>
  auto convert_uint8(T value) {
    if constexpr (std::is_same_v<T, uint8_t>) return static_cast<int>(value);
    else if constexpr (std::is_same_v<T, std::string_view>) return std::string(value);
    else return value;
  }

  template <typename ...Args>
  std::string format_args_n8(const Args&... args) {
    std::string result;
    int idx = 0;
    ((result += fan::detail::to_str(convert_uint8(args)) + (++idx == (int)sizeof...(args) ? "" : ", ")), ...);
    return result;
  }
}