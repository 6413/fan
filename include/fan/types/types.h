#pragma once

#define STRINGIFY(p0) #p0
#define STRINGIFY_DEFINE(a) STRINGIFY(a)

#ifndef FAN_INCLUDE_PATH
  #define _FAN_PATH(p0) <fan/p0>
#else
  #define _FAN_PATH(p0) <FAN_INCLUDE_PATH/fan/p0>
#endif

#include <iostream>
#include <array>
#include <vector>
#include <sstream>
#include <functional>
#include <type_traits>
#include <cstdint>

typedef intptr_t si_t;
//typedef uintptr_t uint_t;
typedef intptr_t sint_t;
typedef int8_t sint8_t;
typedef int16_t sint16_t;
typedef int32_t sint32_t;
//typedef int64_t sint64_t;

typedef float f32_t;
typedef double f64_t;

typedef double f_t;

typedef f32_t cf_t;

namespace fan {

  constexpr auto uninitialized = -1;

  // converts enum to int
  template <typename Enumeration>
  constexpr auto eti(Enumeration const value)
    -> typename std::underlying_type<Enumeration>::type
  {
    return static_cast<
      typename std::underlying_type<Enumeration>::type
    >(value);
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

  template <typename ...Args>
  constexpr void print_no_space(const Args&... args) {
    ((std::cout << args), ...) << '\n';
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
  constexpr void print(const Args&... args) {
    ((std::cout << args << ' '), ...) << '\n';
  }

  template <typename ...Args>
  constexpr void wprint(const Args&... args) {
    ((std::wcout << args << " "), ...) << '\n';
  }

  template <typename T>
  constexpr uintptr_t vector_byte_size(const typename std::vector<T>& vector)
  {
    return sizeof(T) * vector.size();
  }

  template <typename T>
  std::wstring to_wstring(const T a_value, const int n = 2)
  {
    std::wostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
  }

  template <typename T, typename T2>
  constexpr bool is_flag(T value, T2 flag) {
    return (value & flag) == flag;
  }

  //enum class platform_t { windows, linux };

  #if defined(_WIN32) || defined(_WIN64)

  //constexpr platform_t platform = platform_t::windows;
  #define fan_platform_windows

  #ifdef _MSC_VER
  #define fan_compiler_visual_studio
  #elif defined(__clang__)
  #define fan_compiler_clang
  #elif defined(__GNUC__)
  #define fan_compiler_gcc
  #endif
  #elif defined(__ANDROID__ )

  #define fan_platform_android

  #elif defined(__linux__)

 // constexpr platform_t platform = platform_t::windows;
  #define fan_platform_linux
  #define fan_platform_unix

  #elif defined(__unix__)

  #define fan_platform_unix

  #elif defined(__FreeBSD__)

  #define fan_platform_freebsd
  #define fan_platform_unix

  #endif

  template <bool _Test, uintptr_t _Ty1, uintptr_t _Ty2>
  struct conditional_value {
    static constexpr auto value = _Ty1;
  };

  template <uintptr_t _Ty1, uintptr_t _Ty2>
  struct conditional_value<false, _Ty1, _Ty2> {
    static constexpr auto value = _Ty2;
  };

  template <bool _Test, uintptr_t _Ty1, uintptr_t _Ty2>
  struct conditional_value_t {
    static constexpr auto value = conditional_value<_Test, _Ty1, _Ty2>::value;
  };

  #define fan_validate_buffer(buffer, function) \
	if (buffer != static_cast<decltype(buffer)>(fan::uninitialized)) \
	{ \
		function; \
	}

  // prints warning if value is -1
  #define fan_validate_value(value, text) if (value == (decltype(value))fan::uninitialized) { fan::throw_error(text); }

  template <typename T>
  constexpr T clamp(T value, T min, T max) {
    if (value < min) {
      return min;
    }
    if (value > max) {
      return max;
    }
    return value;
  }

  template <typename T, typename T2>
  constexpr T min(T x, T2 y) {
    return x < y ? x : y;
  }
  template <typename T, typename T2>
  constexpr T max(T x, T2 y) {
    return x > y ? x : y;
  }

  template <typename T, uint32_t duplicate_id = 0>
  class class_duplicator : public T {

    using T::T;

  };

  static std::wstring str_to_wstr(const std::string& s)
  {
    std::wstring ret(s.begin(), s.end());
    return ret;
  }

  // slow
  template <typename T>
  static std::vector<T> string_to_values(const std::string& str)
  {
    std::vector<T> values;

    std::stringstream ss;
    ss << str;

    std::string temp;
    T found;
    while (!ss.eof()) {

      ss >> temp;

      if (std::stringstream(temp) >> found) {
        values.push_back(found);
      }
    }
    return values;
  }

   // slow
  template <typename T>
  static std::vector<T> string_to_values(const std::wstring& str)
  {
    std::vector<T> values;

    std::wstringstream ss;
    ss << str;

    std::wstring temp;
    T found;
    while (!ss.eof()) {

      ss >> temp;

      if (std::wstringstream(temp) >> found) {
        values.push_back(found);
      }
    }
    return values;
  }

  static void print_warning(const std::string& message) {
    #ifndef fan_disable_warnings
    fan::print("fan warning: ", message);
    #endif
  }
  static void print_warning_no_space(const std::string& message) {
    #ifndef fan_disable_warnings
    fan::print_no_space("fan warning:", message);
    #endif
  }

  static void throw_error(const std::string& message) {
    fan::print(message);
    #ifdef fan_compiler_visual_studio
    system("pause");
    #endif
#if __cpp_exceptions
    throw std::runtime_error("");
#endif
    //exit(1);
  }

  template <typename T>
  struct ptr_maker_t {

    void open() {
      ptr = new T;
    }

    void close() {
      delete ptr;
    }

    T& operator*() {
      return *ptr;
    }
    T& operator*() const {
      return *ptr;
    }
    T* operator->() {
      return ptr;
    }
    T* operator->() const {
      return ptr;
    }
    T& operator[](uintptr_t i) const {
      return ptr[i];
    }
    T& operator[](uintptr_t i) {
      return ptr[i];
    }

    T* ptr;

  };

  template <typename T, typename T2>
  struct pair_t {
    T first;
    T2 second;
  };

  template <typename T, typename T2>
  pair_t<T, T2> make_pair(T a, T2 b) {
    return pair_t<T, T2>{a, b};
  }

  template <typename T, typename U>
  constexpr auto ofof(U T::* member) {
    return ((::size_t) & reinterpret_cast<char const volatile&>((((T*)0)->*member)));
  }

  template <typename T, typename U>
  constexpr auto offsetless(void* ptr, U T::* member) {
    return (T*)((uint8_t*)(ptr)-((char*)&((T*)nullptr->*member) - (char*)nullptr));
  }

  template <typename T>
  std::string combine_values(T t) {
    if constexpr (std::is_same<T, const char*>::value) {
      return t;
    }
    else {
      return std::to_string(t);
    }
  }

  template <typename T2, typename ...T>
  static std::string combine_values(T2 first, T... args) {
    if constexpr (std::is_same<T2, const char*>::value ||
      std::is_same<T2, std::string>::value) {
      return first + f(args...);
    }
    else {
      return std::to_string(first) + f(args...);
    }
  }

  template<typename Callable>
  using return_type_of_t = typename decltype(std::function{std::declval<Callable>()})::result_type;

  constexpr uint64_t get_hash(const std::string& str) {
    uint64_t result = 0xcbf29ce484222325; // FNV offset basis

    uint32_t i = 0;

    while(str[i] != 0) {
        result ^= str[i];
        result *= 1099511628211; // FNV prime
        i++;
    }

    return result;
  }
}


//template <platform_t T_platform>
//concept platform_windows = T_platform == platform_t::windows;

//template <platform_t T_platform>
//concept platform_linux = T_platform == platform_t::linux;

#define _CONCAT(_0_m, _1_m) _0_m ## _1_m
#define CONCAT(_0_m, _1_m) _CONCAT(_0_m, _1_m)
#define _CONCAT2(_0_m, _1_m) _0_m ## _1_m
#define CONCAT2(_0_m, _1_m) _CONCAT(_0_m, _1_m)
#define _CONCAT3(_0_m, _1_m, _2_m) _0_m ## _1_m ## _2_m
#define CONCAT3(_0_m, _1_m, _2_m) _CONCAT3(_0_m, _1_m, _2_m)
#define _CONCAT4(_0_m, _1_m, _2_m, _3_m) _0_m ## _1_m ## _2_m ## _3_m
#define CONCAT4(_0_m, _1_m, _2_m, _3_m) _CONCAT4(_0_m, _1_m, _2_m, _3_m)

#define OFFSETLESS(ptr_m, t_m, d_m) \
	(t_m *)((uint8_t *)(ptr_m) - offsetof(t_m, d_m))

#define fan_debug_none 0
#define fan_debug_low 1
#define fan_debug_medium 2
#define fan_debug_high 3

#define __ca__ ,

#ifndef fan_debug
#define fan_debug fan_debug_low
#endif

#ifndef fan_use_uninitialized
  #define fan_use_uninitialized 0
#endif

#ifdef fan_platform_windows
#pragma comment(lib, "Onecore.lib")
#endif