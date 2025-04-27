module;

#include <fan/types/types.h>

#include <iostream>
#include <string>
#include <sstream>

#if defined(fan_std23)
  #include <stacktrace>
#endif

export module fan.types.print;

export namespace fan {

  template <typename ...Args>
  constexpr void print(const Args&... args) {
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
 //   fan::print(args...);
    fan::throw_error_impl();
  }

  template <typename T>
  std::ostream& operator<<(std::ostream& os, const std::vector<T>& vector) noexcept
  {
    for (uintptr_t i = 0; i < vector.size(); i++) {
      os << vector[i] << ' ';
    }
    return os;
  }

  template <typename T>
  std::ostream& operator<<(std::ostream& os, const std::vector<std::vector<T>>& vector) noexcept
  {
    for (uintptr_t i = 0; i < vector.size(); i++) {
      for (uintptr_t j = 0; j < vector[i].size(); j++) {
        os << vector[i][j] << ' ';
      }
      os << '\n';
    }
    return os;
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
