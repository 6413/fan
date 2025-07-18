module;

#include <fan/types/types.h>

// With windows clang build there can be msvc and clang both defined
#if defined(fan_compiler_msvc) && !defined(fan_compiler_clang)

#else

#include <iostream>
#include <string>
#include <string_view>
//#include <stacktrace>
#include <sstream>
#include <ostream>
#include <iomanip>
#include <unordered_map>
#include <chrono>

#endif

export module fan.print;

#if defined(fan_compiler_msvc) && !defined(fan_compiler_clang)
import std;
#endif

export namespace fan {

  template <typename ...Args>
  constexpr void print(const Args&... args) {
    int idx = 0;
    ((std::cout << args << (++idx == sizeof...(args) ? "" : " ")), ...);
    std::cout << '\n';
  }
  // print raw
  template <typename ...Args>
  constexpr void printr(const Args&... args) {
    int idx = 0;
    ((std::cout << args), ...);
  }
  // print comma
  template <typename ...Args>
  constexpr void printc(const Args&... args) {
    int idx = 0;
    ((std::cout << args << (++idx == sizeof...(args) ? "" : ", ")), ...);
    std::cout << '\n';
  }

  //print tab size
  template <typename ...Args>
  constexpr void prints(std::streamsize w, const Args&... args) {
    std::ios init(0);
    init.copyfmt(std::cout);
    std::setw(w);
    fan::print(args...);
    std::cout.copyfmt(init);
  }
  //print tab
  template <typename ...Args>
  constexpr void printt(const Args&... args) {
    fan::prints(2, args...);
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
  constexpr void printn8(const Args&... args) {
    int idx = 0;
    ((std::cout << convert_uint8(args) << (++idx == sizeof...(args) ? "" : ", ")), ...);
    std::cout << '\n';
  }

  template <typename ...Args>
  constexpr void print_no_space(const Args&... args) {
    ((std::cout << args), ...) << '\n';
  }

  void print_warning(const std::string& message) {
    #ifndef fan_disable_warnings
    fan::print("fan warning: " + message);
    #endif
  }
  void print_warning_no_space(const std::string& message) {
    #ifndef fan_disable_warnings
    fan::print_no_space("fan warning:", message);
    #endif
  }

  template <typename ...Args>
  constexpr void print_no_endline(const Args&... args) {
    ((std::cout << args << ' '), ...);
  }

  template <typename ...Args>
  constexpr void wprint_no_endline(const Args&... args) {
    ((std::wcout << args << ' '), ...);
  }

  template <typename ...Args>
  constexpr void wprint(const Args&... args) {
    ((std::wcout << args << " "), ...) << '\n';
  }

  template <typename ...Args>
  constexpr void throw_error(const Args&... args) {
    std::ostringstream out;
    ((out << args << " "), ...);
    std::string str = out.str();
    str.pop_back();
    fan::throw_error_impl(str.c_str());
  }

  template <typename T>
  requires requires(const T& t) { t.size(); } && (!std::is_same_v<T, std::string>)
  std::ostream& operator<<(std::ostream& os, const T& container) noexcept {
    for (uintptr_t i = 0; i < container.size(); i++) {
      os << container[i] << ' ';
    }
    return os;
  }

  template <typename T>
  requires requires(const T& t) { t.size.size(); }
  std::ostream& operator<<(std::ostream& os, const T& container_within) noexcept
  {
    for (uintptr_t i = 0; i < container_within.size(); i++) {
      for (uintptr_t j = 0; j < container_within[i].size(); j++) {
        os << container_within[i][j] << ' ';
      }
      os << '\n';
    }
    return os;
  }

  template <typename T>
  constexpr void print_throttled(const T& value, int throttle_ms = 1000) {
    static std::unordered_map<std::string, std::chrono::steady_clock::time_point> last_print_time;
    static std::unordered_map<std::string, int> count_map;

    std::ostringstream oss;
    oss << value;
    std::string key = oss.str();

    auto now = std::chrono::steady_clock::now();
    count_map[key]++;

    if (last_print_time.find(key) == last_print_time.end() ||
      std::chrono::duration_cast<std::chrono::milliseconds>(now - last_print_time[key]).count() >= throttle_ms) {

      std::cout << key << " (" << count_map[key] << " times)" << std::endl;
      last_print_time[key] = now;
    }
  }
  namespace debug {

    void print_stacktrace() {
      
      #if defined(fan_std23)
      //std::stacktrace st;
      //fan::print(st.current());
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