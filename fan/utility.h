// Macro based utility
#pragma once

#define STRINGIFY(p0) #p0
#define STRINGIFY_DEFINE(a) STRINGIFY(a)

#ifndef _CONCAT
  #define _CONCAT(_0_m, _1_m) _0_m ## _1_m
#endif
#ifndef CONCAT
  #define CONCAT(_0_m, _1_m) _CONCAT(_0_m, _1_m)
#endif
#define _CONCAT2(_0_m, _1_m) _0_m ## _1_m
#define CONCAT2(_0_m, _1_m) _CONCAT(_0_m, _1_m)
#define _CONCAT3(_0_m, _1_m, _2_m) _0_m ## _1_m ## _2_m
#define CONCAT3(_0_m, _1_m, _2_m) _CONCAT3(_0_m, _1_m, _2_m)
#define _CONCAT4(_0_m, _1_m, _2_m, _3_m) _0_m ## _1_m ## _2_m ## _3_m
#define CONCAT4(_0_m, _1_m, _2_m, _3_m) _CONCAT4(_0_m, _1_m, _2_m, _3_m)
#define _CONCAT5(_0_m, _1_m, _2_m, _3_m, _4_m) _0_m ## _1_m ## _2_m ## _3_m ## _4_m
#define CONCAT5(_0_m, _1_m, _2_m, _3_m, _4_m) _CONCAT5(_0_m, _1_m, _2_m, _3_m, _4_m)

#define _FAN_PATH(p0) <fan/p0>
#define _PATH_QUOTE(p0) STRINGIFY(p0)

#if defined(__clang__)
  #define fan_compiler_clang
#elif defined(__GNUC__)
  #define fan_compiler_gcc
#endif

#if defined(_WIN32) || defined(_WIN64)
  #define fan_platform_windows
  #if defined(_MSC_VER)
    #define fan_compiler_msvc
  #endif
#elif defined(__ANDROID__ )
  #define fan_platform_android
#elif defined(__linux__)
  #define fan_platform_linux
  #define fan_platform_unix
#elif defined(__unix__)
  #define fan_platform_unix
#elif defined(__FreeBSD__)
  #define fan_platform_freebsd
  #define fan_platform_unix
#endif

#if defined(fan_platform_windows)
  #pragma execution_character_set("utf-8")
#endif

#define PARENS ()
#define EXPAND(...) EXPAND4(EXPAND4(EXPAND4(EXPAND4(__VA_ARGS__))))
#define EXPAND4(...) EXPAND3(EXPAND3(EXPAND3(EXPAND3(__VA_ARGS__))))
#define EXPAND3(...) EXPAND2(EXPAND2(EXPAND2(EXPAND2(__VA_ARGS__))))
#define EXPAND2(...) EXPAND1(EXPAND1(EXPAND1(EXPAND1(__VA_ARGS__))))
#define EXPAND1(...) __VA_ARGS__

#define __FAN__GET_ARG_COUNT(...) __FAN__INTERNAL_GET_ARG_COUNT(__VA_ARGS__ __VA_OPT__(,) 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
#define __FAN__INTERNAL_GET_ARG_COUNT(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, n, ...) n

#define __FAN__FOREACH(f, ...) __VA_OPT__(EXPAND(__FAN__FE_HELPER(f, 0, __VA_ARGS__)))
#define __FAN__FE_HELPER(f, i, x, ...) f(x, i); __VA_OPT__(__FAN__FE_AGAIN PARENS (f, i + 1, __VA_ARGS__))
#define __FAN__FE_AGAIN() __FAN__FE_HELPER

#define __FAN__FOREACH_NS(f, ...) __VA_OPT__(EXPAND(__FAN__FENS_HELPER(f, __VA_ARGS__)))
#define __FAN__FENS_HELPER(f, x, ...) f(x) __VA_OPT__(, __FAN__FENS_AGAIN PARENS (f, __VA_ARGS__))
#define __FAN__FENS_AGAIN() __FAN__FENS_HELPER

#define fan_ternary_void(f0, f1) [&]{ f0;}() : [&]{f1;}()

#define __FAN_PRINT_EACH__(x) fan::printnln("", #x"=",x, ",")
#define fan_print(...) __FAN__FOREACH_NS(__FAN_PRINT_EACH__, __VA_ARGS__);fan::print("")

#define fan_validate_buffer(buffer, function) \
  if (buffer != static_cast<decltype(buffer)>(fan::uninitialized)) \
  { \
    function; \
  }

#define fan_validate_value(value, text) if (value == (decltype(value))fan::uninitialized) { fan::throw_error(text); }

#if defined(_DEBUG) || defined(DEBUG)
    #define FAN_DEBUG_BUILD 1
#else
    #define FAN_DEBUG_BUILD 0
#endif

#define fan_debug_none 0
#define fan_debug_low 1
#define fan_debug_medium 2
#define fan_debug_high 3
#define fan_debug_insane 4

#define __ca__ ,

#ifndef FAN_DEBUG
#if FAN_DEBUG_BUILD
  #define FAN_DEBUG fan_debug_high
  #define __sanit 1
#else
  #define FAN_DEBUG fan_debug_none
#endif
#endif

#ifndef fan_use_uninitialized
#define fan_use_uninitialized 0
#endif

#ifdef fan_platform_windows
#pragma comment(lib, "Onecore.lib")
#endif

using DWORD = unsigned long;

#define __FAN__INSERTVARNAME(x, idx) var__ x

#define fan_init_struct(type, ...) [&] { \
  type var__; \
  __FAN__FOREACH(__FAN__INSERTVARNAME, __VA_ARGS__) \
  return var__; \
}()

#define fan_init_struct0(type, name, ...) type name = type([&] { \
  type var__; \
  __FAN__FOREACH(__FAN__INSERTVARNAME, __VA_ARGS__) \
  return var__; \
}())

#define fan_init_id_t0(type, properties, name, ...) type name = type([&] { \
  properties var__; \
  __FAN__FOREACH(__FAN__INSERTVARNAME, __VA_ARGS__) \
  return var__; \
}())

#define fan_init_id_t1(type, ...) [&] { \
  type ## _id_t ::properties_t var__; \
  __FAN__FOREACH(__FAN__INSERTVARNAME, __VA_ARGS__) \
  return var__; \
}()

#define fan_init_id_t2(type, ...) \
  fan_init_id_t0(type, __super__secret__name0, __VA_ARGS__)

#define fan_init_id_t3(type, f, ...) [&] { \
  type ## _id_t ::properties_t var__; \
  var__ f; \
  __FAN__FOREACH(__FAN__INSERTVARNAME, __VA_ARGS__) \
  return var__; \
}()

#define fan_get_current_directory() [] { \
  std::string dir_path = __FILE__; \
  return dir_path.substr(0, dir_path.rfind((char)std::filesystem::path::preferred_separator)) +  \
    (char)std::filesystem::path::preferred_separator; \
  }()

#ifndef lstd_defstruct
  #define lstd_defstruct(type_name) \
    struct type_name{ \
      using lstd_current_type = type_name;
#endif

namespace fan {
  template<auto F>
  using invoke_t = decltype(F());
}

#define fan_temporary_struct_maker(...) fan::invoke_t<+[](){ struct { __VA_ARGS__ } s{}; return s; }>
#define st(...) \
  fan::invoke_t<+[](){ struct { __VA_ARGS__ } s{}; return s; }>
#define st_raw(body) \
  fan::invoke_t<+[](){ struct body s{}; return s; }>

#define EXPAND_L(p0) p0
#define lstd_preprocessor_get_argn(p0, p1, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, n, ...) n
#define lstd_preprocessor_get_arg_count(...) EXPAND_L(lstd_preprocessor_get_argn(__VA_ARGS__, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0))

#ifndef lstd_preprocessor_ignore_first_of_every_2
#define _lstd_preprocessor_ignore_first_of_every_2_2(p0, p1) p1
#define _lstd_preprocessor_ignore_first_of_every_2_4(p0, p1, ...) p1, EXPAND_L(_lstd_preprocessor_ignore_first_of_every_2_2(__VA_ARGS__))
#define _lstd_preprocessor_ignore_first_of_every_2_6(p0, p1, ...) p1, EXPAND_L(_lstd_preprocessor_ignore_first_of_every_2_4(__VA_ARGS__))
#define _lstd_preprocessor_ignore_first_of_every_2_8(p0, p1, ...) p1, EXPAND_L(_lstd_preprocessor_ignore_first_of_every_2_6(__VA_ARGS__))
#define _lstd_preprocessor_ignore_first_of_every_2_10(p0, p1, ...) p1, EXPAND_L(_lstd_preprocessor_ignore_first_of_every_2_8(__VA_ARGS__))
#define _lstd_preprocessor_ignore_first_of_every_2_12(p0, p1, ...) p1, EXPAND_L(_lstd_preprocessor_ignore_first_of_every_2_10(__VA_ARGS__))
#define _lstd_preprocessor_ignore_first_of_every_2_14(p0, p1, ...) p1, EXPAND_L(_lstd_preprocessor_ignore_first_of_every_2_12(__VA_ARGS__))
#define _lstd_preprocessor_ignore_first_of_every_2_16(p0, p1, ...) p1, EXPAND_L(_lstd_preprocessor_ignore_first_of_every_2_14(__VA_ARGS__))
#define _lstd_preprocessor_ignore_first_of_every_2_start(n, ...) CONCAT(_lstd_preprocessor_ignore_first_of_every_2_,n(__VA_ARGS__))
#define lstd_preprocessor_ignore_first_of_every_2(...) _lstd_preprocessor_ignore_first_of_every_2_start(lstd_preprocessor_get_arg_count(__VA_ARGS__), __VA_ARGS__)
#endif

#ifndef lstd_preprocessor_combine_every_2
#define _lstd_preprocessor_combine_every_2_2(p0, p1) p0 p1
#define _lstd_preprocessor_combine_every_2_4(p0, p1, ...) p0 p1, EXPAND_L(_lstd_preprocessor_combine_every_2_2(__VA_ARGS__))
#define _lstd_preprocessor_combine_every_2_6(p0, p1, ...) p0 p1, EXPAND_L(_lstd_preprocessor_combine_every_2_4(__VA_ARGS__))
#define _lstd_preprocessor_combine_every_2_8(p0, p1, ...) p0 p1, EXPAND_L(_lstd_preprocessor_combine_every_2_6(__VA_ARGS__))
#define _lstd_preprocessor_combine_every_2_10(p0, p1, ...) p0 p1, EXPAND_L(_lstd_preprocessor_combine_every_2_8(__VA_ARGS__))
#define _lstd_preprocessor_combine_every_2_12(p0, p1, ...) p0 p1, EXPAND_L(_lstd_preprocessor_combine_every_2_10(__VA_ARGS__))
#define _lstd_preprocessor_combine_every_2_14(p0, p1, ...) p0 p1, EXPAND_L(_lstd_preprocessor_combine_every_2_12(__VA_ARGS__))
#define _lstd_preprocessor_combine_every_2_16(p0, p1, ...) p0 p1, EXPAND_L(_lstd_preprocessor_combine_every_2_14(__VA_ARGS__))
#define _lstd_preprocessor_combine_every_2_start(n, ...) CONCAT(_lstd_preprocessor_combine_every_2_,n(__VA_ARGS__))
#define lstd_preprocessor_combine_every_2(...) _lstd_preprocessor_combine_every_2_start(lstd_preprocessor_get_arg_count(__VA_ARGS__), __VA_ARGS__)
#endif

#undef __FAN_PRINT_EACH
#define __FAN_PRINT_EACH(x) #x

#define fan_enum_string(name, ...) \
static constexpr const char* name##_strings[] = {__FAN__FOREACH_NS(__FAN_PRINT_EACH, __VA_ARGS__)}; \
enum name { __VA_ARGS__ }

#define fan_enum_class_string(name, ...) \
static constexpr const char* name##_strings[] = {__FAN__FOREACH_NS(__FAN_PRINT_EACH, __VA_ARGS__)}; \
enum class name { __VA_ARGS__ }

#if defined(__EMSCRIPTEN__)
  #ifndef FAN_WASM
    #define FAN_WASM
  #endif
#endif

#if __cplusplus >= 202004L && defined(fan_compiler_msvc) && !defined(fan_compiler_clang) && __has_include("stacktrace") && !defined(FAN_WASM)
  #define fan_std23
#endif

#ifndef __forceinline
  #if defined(fan_compiler_clang) || defined(fan_compiler_gcc)
    #define __forceinline __attribute__((always_inline))
  #elif defined(fan_compiler_msvc)
    // already defined
  #else
    #define __forceinline inline __attribute__((always_inline))
  #endif
#endif

#define fan_make_flexible_array(type, name, ...) \
  std::array<type, std::initializer_list<type>{__VA_ARGS__}.size()> name = {__VA_ARGS__}

#if defined(fan_compiler_msvc) || defined(fan_compiler_clang)
  #define FAN_UNIQUE_CALL auto token = +[]{}
  #define FAN_UNIQUE_CALL_PASS token
#else
  #define FAN_UNIQUE_CALL std::uint64_t line = __builtin_LINE()
  #define FAN_UNIQUE_CALL_PASS line
#endif

#pragma pack(push, 1)
  #include <fan/types/bll_types.h>
#pragma pack(pop)

#ifndef offsetof
  #if defined(fan_compiler_msvc)
    #define offsetof(type, member) ((std::size_t)(&((type*)0)->member))
  #else
    #define offsetof(type, member) __builtin_offsetof(type, member)
  #endif
#endif

#ifndef OFFSETLESS
  #define OFFSETLESS(ptr_m, t_m, d_m) \
    ((t_m *)((std::uint8_t *)(ptr_m) - offsetof(t_m, d_m)))
#endif