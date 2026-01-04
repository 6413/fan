module;

#include <fan/utility.h>

#include <sstream>
#include <iostream>
#include <string>
#include <string_view>
#if defined(fan_std23)
  #include <stacktrace>
#endif
#include <ostream>
#include <iomanip>
#include <unordered_map>
#include <chrono>
#include <type_traits>
#include <bitset>

export module fan.print;

import fan.utility;
// for colored prints
import fan.types.color;

export namespace fan {
  template <typename ...Args>
  constexpr std::string format_args(const Args&... args) {
    std::ostringstream oss;
    int idx = 0;
    ((oss << args << (++idx == sizeof...(args) ? "" : " ")), ...);
    return oss.str();
  }
  template <typename ...Args>
  constexpr std::string format_args_raw(const Args&... args) {
    std::ostringstream oss;
    ((oss << args), ...);
    return oss.str();
  }
  template <typename ...Args>
  constexpr std::string format_args_comma(const Args&... args) {
    std::ostringstream oss;
    int idx = 0;
    ((oss << args << (++idx == sizeof...(args) ? "" : ", ")), ...);
    return oss.str();
  }
  template <typename ...Args>
  constexpr std::string format_args_tabbed(std::streamsize tab_width, const Args&... args) {
    std::ostringstream oss;
    std::ios init(nullptr);
    init.copyfmt(oss);

    int idx = 0;
    ((oss << (idx == 0 ? "" : (fan::is_negative(args) ? "" : " "))
      << std::setw(tab_width)
      << std::left << args
      << (++idx == sizeof...(args) ? "" : " ")
      << (fan::is_negative(args) ? " " : "")), ...);

    oss.copyfmt(init);
    return oss.str();
  }
  template<typename T>
  auto convert_uint8(T value) {
    if constexpr (std::is_same_v<T, uint8_t>) {
      return static_cast<int>(value);
    }
    else if constexpr (std::is_same_v<T, std::string_view>) {
      return std::string(value);
    }
    else {
      return value;
    }
  }
  template <typename ...Args>
  constexpr std::string format_args_n8(const Args&... args) {
    std::ostringstream oss;
    int idx = 0;
    ((oss << convert_uint8(args) << (++idx == sizeof...(args) ? "" : ", ")), ...);
    return oss.str();
  }
  template <typename ...Args>
  constexpr std::string format_args_no_space(const Args&... args) {
    std::ostringstream oss;
    ((oss << args), ...);
    return oss.str();
  }
  std::string format_warning(const std::string& message) {
  #ifndef fan_disable_warnings
    return "fan warning: " + message;
  #else
    return "";
  #endif
  }
  std::string format_warning_no_space(const std::string& message) {
  #ifndef fan_disable_warnings
    return "fan warning:" + message;
  #else
    return "";
  #endif
  }
  template <typename ...Args>
  constexpr std::string format_args_with_space(const Args&... args) {
    std::ostringstream oss;
    ((oss << args << ' '), ...);
    return oss.str();
  }
  template <typename ...Args>
  constexpr std::wstring format_wargs_with_space(const Args&... args) {
    std::wostringstream oss;
    ((oss << args << L' '), ...);
    return oss.str();
  }
  template <typename ...Args>
  constexpr std::wstring format_wargs(const Args&... args) {
    std::wostringstream oss;
    ((oss << args << L" "), ...);
    return oss.str();
  }
  template <typename ...Args>
  constexpr std::string format_error_args(const Args&... args) {
    std::ostringstream oss;
    ((oss << args << " "), ...);
    std::string str = oss.str();
    if (!str.empty()) {
      str.pop_back();
    }
    return str;
  }

  template <typename ...Args>
  constexpr void print(const Args&... args) {
    std::cout << format_args(args...) << '\n';
  }
  template <typename ...Args>
  constexpr void print_color(const fan::color& c, const Args&... args) {
    uint32_t rgba = c.get_rgba();
    uint8_t r = (rgba >> 24) & 0xFF;
    uint8_t g = (rgba >> 16) & 0xFF;
    uint8_t b = (rgba >> 8) & 0xFF;
    std::cout << "\033[38;2;" << static_cast<int>(r) << ";" << static_cast<int>(g) << ";" << static_cast<int>(b) << "m" << format_args(args...) << "\033[0m\n";
  }
  template <typename ...Args>
  constexpr void printr(const Args&... args) {
    std::cout << format_args_raw(args...);
  }
  template <typename ...Args>
  constexpr void printc(const Args&... args) {
    std::cout << format_args_comma(args...) << '\n';
  }
  template <typename ...Args>
  constexpr void printt(std::streamsize tab_width, const Args&... args) {
    std::cout << format_args_tabbed(tab_width, args...) << '\n';
  }
  template <typename ...Args>
  constexpr void printn8(const Args&... args) {
    std::cout << format_args_n8(args...) << '\n';
  }
  template <typename ...Args>
  constexpr void print_no_space(const Args&... args) {
    std::cout << format_args_no_space(args...) << '\n';
  }
  void print_success(const std::string& message) {
    print_color(fan::colors::green, message);
  }
  void print_warning(const std::string& message) {
  #ifndef fan_disable_warnings
    print_color(fan::colors::yellow, message);
  #endif
  }
  void print_error(const std::string& message) {
  #ifndef fan_disable_errors
    print_color(fan::colors::red, message);
  #endif
  }
  void print_warning_no_space(const std::string& message) {
  #ifndef fan_disable_warnings
    std::cout << format_warning_no_space(message) << '\n';
  #endif
  }
  template <typename ...Args>
  constexpr void printnln(const Args&... args) {
    std::cout << format_args_with_space(args...);
  }
  template <typename ...Args>
  constexpr void wprint_no_endline(const Args&... args) {
    std::wcout << format_wargs_with_space(args...);
  }
  template <typename ...Args>
  constexpr void wprint(const Args&... args) {
    std::wcout << format_wargs(args...) << '\n';
  }
  template <typename ...Args>
  constexpr void throw_error(const Args&... args) {
    std::string error_msg = format_error_args(args...);
    fan::throw_error_impl(error_msg.c_str());
  }

  template <typename T>
  struct is_bitset : std::false_type {};

  template <size_t N>
  struct is_bitset<std::bitset<N>> : std::true_type {};

  template <typename T>
  concept has_subscript_and_size = requires(const T& t) { 
    t.size(); 
    t[0]; 
  } && (!std::is_same_v<T, std::string>) 
    && (!is_bitset<std::remove_cvref_t<T>>::value);

  template <typename T>
  requires has_subscript_and_size<T>
  std::ostream& operator<<(std::ostream& os, const T& container) noexcept {
    for (uintptr_t i = 0; i < container.size(); i++) {
      os << container[i] << ' ';
    }
    return os;
  }

  template <typename T>
  requires requires(const T& t) { t.size.size(); }
  std::ostream& operator<<(std::ostream& os, const T& container_within) noexcept {
    for (uintptr_t i = 0; i < container_within.size(); i++) {
      for (uintptr_t j = 0; j < container_within[i].size(); j++) {
        os << container_within[i][j] << ' ';
      }
      os << '\n';
    }
    return os;
  }

  template <int throttle_ms = 1000>
  void print_throttled(const auto&... args) {
    static std::unordered_map<std::string, std::chrono::steady_clock::time_point> last_print_time;
    static std::unordered_map<std::string, int> count_map;

    std::ostringstream oss;
    oss << fan::format_args(args...);
    std::string key = oss.str();

    auto now = std::chrono::steady_clock::now();
    count_map[key]++;

    if (last_print_time.find(key) == last_print_time.end() ||
      std::chrono::duration_cast<std::chrono::milliseconds>(now - last_print_time[key]).count() >= throttle_ms) {

      std::cout << key << " (" << count_map[key] << " times)" << std::endl;
      last_print_time[key] = now;
    }
  }

  template <bool print_count = true>
  void print_once(const auto&... args) {
    static std::unordered_map<std::string, std::string> last_value;
    static std::unordered_map<std::string, int> count_map;

    std::ostringstream oss;
    oss << fan::format_args(args...);
    std::string key = oss.str();

    auto& last = last_value[key];
    auto& count = count_map[key];

    count++;

    if (last != key) {
      last = key;

      if constexpr (print_count) {
        std::cout << key << " (" << count << " times)" << std::endl;
      } else {
        std::cout << key << std::endl;
      }
    }
  }


  namespace debug {

    void print_stacktrace() {
      
      #if defined(fan_std23)
      std::stacktrace st;
      fan::print(st.current());
      #elif defined(fan_platform_unix)
      // waiting for stacktrace to be released for clang lib++
      #else
      fan::print("stacktrace not supported");
      #endif
    }
  }
}

export std::string operator""_str(const char* str, std::size_t) {
  return std::string{ str };
}