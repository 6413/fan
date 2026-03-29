module;

#include <fan/utility.h>

#include <sstream>
#include <iostream>
#include <string>
#include <string_view>
#include <source_location>
#if defined(fan_std23)
  #include <stacktrace>
#endif
#include <ostream>
#include <iomanip>
#include <unordered_map>
#include <type_traits>
#include <bitset>

export module fan.print;

import fan.types;
import fan.utility;
import fan.types.color;
import fan.time;

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

  template <typename T>
  struct is_bitset : std::false_type {};
  template <size_t N>
  struct is_bitset<std::bitset<N>> : std::true_type {};

  template <typename T>
  concept has_subscript_and_size = requires(const T& t) {
    t.size(); t[0];
  } && (!std::is_same_v<T, std::string>)
    && (!std::is_same_v<T, std::string_view>)
    && (!std::is_base_of_v<std::string_view, std::remove_cvref_t<T>>)
    && (!is_bitset<std::remove_cvref_t<T>>::value);

  template <typename T>
  requires has_subscript_and_size<T>
  std::ostream& operator<<(std::ostream& os, const T& container) noexcept {
    for (uintptr_t i = 0; i < container.size(); i++) os << container[i] << ' ';
    return os;
  }

  template <typename T>
  requires requires(const T& t) { t.size.size(); }
  std::ostream& operator<<(std::ostream& os, const T& container_within) noexcept {
    for (uintptr_t i = 0; i < container_within.size(); i++) {
      for (uintptr_t j = 0; j < container_within[i].size(); j++)
        os << container_within[i][j] << ' ';
      os << '\n';
    }
    return os;
  }

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

  template <typename ...Args>
  std::string format_args_tabbed(std::streamsize tab_width, const Args&... args) {
    std::ostringstream oss;
    std::ios init(nullptr);
    init.copyfmt(oss);
    int idx = 0;
    ((oss << (idx == 0 ? "" : (fan::is_negative(args) ? "" : " "))
      << std::setw(tab_width) << std::left << args
      << (++idx == (int)sizeof...(args) ? "" : " ")
      << (fan::is_negative(args) ? " " : "")), ...);
    oss.copyfmt(init);
    return oss.str();
  }

  template <typename ...Args>
  std::wstring format_wargs_with_space(const Args&... args) {
    std::wostringstream oss;
    ((oss << args << L' '), ...);
    return oss.str();
  }

  template <typename ...Args>
  std::wstring format_wargs(const Args&... args) {
    std::wostringstream oss;
    ((oss << args << L" "), ...);
    return oss.str();
  }

  std::string format_warning(const std::string& message) {
#ifndef fan_disable_warnings
    std::string ret = "fan warning:" + message;
    write_error_to_disk(ret);
    return ret;
#else
    return "";
#endif
  }

  std::string format_warning_no_space(const std::string& message) {
#ifndef fan_disable_warnings
    std::string ret = "fan warning:" + message;
    write_error_to_disk(ret);
    return ret;
#else
    return "";
#endif
  }

  template <typename ...Args> void print(const Args&... args) { std::cout << format_args(args...) << '\n'; }
  template <typename ...Args> void printr(const Args&... args) { std::cout << format_args_raw(args...); }
  template <typename ...Args> void printc(const Args&... args) { std::cout << format_args_comma(args...) << '\n'; }
  template <typename ...Args> void printt(std::streamsize tab_width, const Args&... args) { std::cout << format_args_tabbed(tab_width, args...) << '\n'; }
  template <typename ...Args> void printn8(const Args&... args) { std::cout << format_args_n8(args...) << '\n'; }
  template <typename ...Args> void print_no_space(const Args&... args) { std::cout << format_args_no_space(args...) << '\n'; }
  template <typename ...Args> void printnln(const Args&... args) { std::cout << format_args_with_space(args...); }
  template <typename ...Args> void wprint_no_endline(const Args&... args) { std::wcout << format_wargs_with_space(args...); }
  template <typename ...Args> void wprint(const Args&... args) { std::wcout << format_wargs(args...) << '\n'; }

  template <typename ...Args>
  void print_color(const fan::color& c, const Args&... args) {
    uint32_t rgba = c.get_rgba();
    uint8_t r = (rgba >> 24) & 0xFF;
    uint8_t g = (rgba >> 16) & 0xFF;
    uint8_t b = (rgba >> 8) & 0xFF;
    std::cout << "\033[38;2;" << (int)r << ";" << (int)g << ";" << (int)b << "m" << format_args(args...) << "\033[0m\n";
  }

  template <typename ...Args>
  void print_success(const Args&... args) {
    std::string message = format_args(args...);
    write_error_to_disk(message);
    print_color(fan::colors::green, message);
  }

  template <typename ...Args>
  void print_warning(const Args&... args) {
#ifndef fan_disable_warnings
    std::string message = format_args(args...);
    write_error_to_disk(message);
    print_color(fan::colors::yellow, message);
#endif
  }

  template <typename ...Args>
  void print_error(const Args&... args) {
#ifndef fan_disable_errors
    std::string message = format_args(args...);
    write_error_to_disk(message);
    print_color(fan::colors::red, message);
#endif
  }

  void print_warning_no_space(const std::string& message) {
#ifndef fan_disable_warnings
    write_error_to_disk(message);
    std::cout << format_warning_no_space(message) << '\n';
#endif
  }

  template <typename ...Args>
  void throw_error(const Args&... args) {
    fan::throw_error_impl(format_error_args(args...).c_str());
  }

  template <int throttle_ms = 1000, typename... Args>
  void print_throttled(const Args&... args) {
    static std::unordered_map<std::size_t, fan::time::timer> timers;
    std::string msg = format_args(args...);
    std::size_t key = std::hash<std::string>{}(msg);
    auto& t = timers[key];
    if (!t.started()) { fan::printr(msg, '\n'); t.start_millis(throttle_ms); return; }
    if (t.finished()) { fan::printr(msg, '\n'); t.start_millis(throttle_ms); }
  }

  template<typename... Args, FAN_UNIQUE_CALL>
  void print_every(int throttle_ms, const Args&... args) {
    if (fan::time::every<
      #if defined(fan_compiler_msvc) || defined(fan_compiled_clang)
        token
      #else
        line, file
      #endif
    >(throttle_ms)) {
      fan::printr(fan::format_args(args...), '\n');
    }
  }

  void print_once(const auto&... args) {
    static std::unordered_map<std::string, int> count_map;
    std::string key = format_args(args...);
    if (++count_map[key] == 1)
      std::cout << key << '\n';
  }

  namespace debug {
    void print_stacktrace() {
#if defined(fan_std23)
      std::stacktrace st;
      fan::print(st.current());
#elif defined(fan_platform_unix)
#else
      fan::print("stacktrace not supported");
#endif
    }
  }
}

export std::string operator""_str(const char* str, std::size_t) {
  return std::string{str};
}