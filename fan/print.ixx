module;

#include <fan/types/types.h>

#ifdef fan_compiler_msvc

#else

#include <iostream>
#include <string>
//#include <stacktrace>
#include <sstream>
#include <ostream>

#endif

export module fan:print;

#ifdef fan_compiler_msvc
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
  template <typename ...Args>
  constexpr void printc(const Args&... args) {
    int idx = 0;
    ((std::cout << args << (++idx == sizeof...(args) ? "" : ", ")), ...);
    std::cout << '\n';
  }

  template<typename T>
  auto convert_uint8(T value) {
    if constexpr (std::is_same_v<T, uint8_t>) {
      return static_cast<int>(value);
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
    fan::print(args...);
    fan::throw_error_impl();
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

export constexpr std::string operator""_str(const char* str, std::size_t) {
  return std::string(str);
}