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

#include <cstdint>

// offsetof
#include <cstddef>
#include <stdio.h>

// for logging
#include <fstream>
#include <iostream>
#include <string>

#if defined(__clang__)
	#define fan_compiler_clang
#elif defined(__GNUC__)
	#define fan_compiler_gcc
#endif

#if defined(_WIN32) || defined(_WIN64)

//constexpr platform_t platform = platform_t::windows;
	#define fan_platform_windows

	#if defined(_MSC_VER)
		#define fan_compiler_msvc
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


// override to utf-8 - if file already utf8 it breaks somehow, probably msvc bug
#if defined(fan_platform_windows)
	#pragma execution_character_set("utf-8")
	#include <cstdlib>
#endif

#define __FAN__FOREACH_1(f, x) f(x, 0)
#define __FAN__FOREACH_2(f, x, ...)  f(x, 1); __FAN__FOREACH_1(f, __VA_ARGS__)
#define __FAN__FOREACH_3(f, x, ...)  f(x, 2); __FAN__FOREACH_2(f, __VA_ARGS__)
#define __FAN__FOREACH_4(f, x, ...)  f(x, 3); __FAN__FOREACH_3(f, __VA_ARGS__)
#define __FAN__FOREACH_5(f, x, ...)  f(x, 4); __FAN__FOREACH_4(f, __VA_ARGS__)
#define __FAN__FOREACH_6(f, x, ...)  f(x, 5); __FAN__FOREACH_5(f, __VA_ARGS__)
#define __FAN__FOREACH_7(f, x, ...)  f(x, 6); __FAN__FOREACH_6(f, __VA_ARGS__)
#define __FAN__FOREACH_8(f, x, ...)  f(x, 7); __FAN__FOREACH_7(f, __VA_ARGS__)
#define __FAN__FOREACH_9(f, x, ...)  f(x, 8); __FAN__FOREACH_8(f, __VA_ARGS__)
#define __FAN__FOREACH_10(f, x, ...)  f(x, 9); __FAN__FOREACH_9(f, __VA_ARGS__)
#define __FAN__FOREACH_11(f, x, ...)  f(x, 10); __FAN__FOREACH_10(f, __VA_ARGS__)
#define __FAN__FOREACH_12(f, x, ...)  f(x, 11); __FAN__FOREACH_11(f, __VA_ARGS__)
#define __FAN__FOREACH_13(f, x, ...)  f(x, 12); __FAN__FOREACH_12(f, __VA_ARGS__)
#define __FAN__FOREACH_14(f, x, ...)  f(x, 13); __FAN__FOREACH_13(f, __VA_ARGS__)
#define __FAN__FOREACH_15(f, x, ...)  f(x, 14); __FAN__FOREACH_14(f, __VA_ARGS__)
#define __FAN__FOREACH_16(f, x, ...)  f(x, 15); __FAN__FOREACH_15(f, __VA_ARGS__)
#define __FAN__FOREACH_17(f, x, ...)  f(x, 16); __FAN__FOREACH_16(f, __VA_ARGS__)
#define __FAN__FOREACH_18(f, x, ...)  f(x, 17); __FAN__FOREACH_17(f, __VA_ARGS__)
#define __FAN__FOREACH_19(f, x, ...)  f(x, 18); __FAN__FOREACH_18(f, __VA_ARGS__)
#define __FAN__FOREACH_20(f, x, ...)  f(x, 19); __FAN__FOREACH_19(f, __VA_ARGS__)
#define __FAN__FOREACH_21(f, x, ...)  f(x, 20); __FAN__FOREACH_20(f, __VA_ARGS__)
#define __FAN__FOREACH_22(f, x, ...)  f(x, 21); __FAN__FOREACH_21(f, __VA_ARGS__)
#define __FAN__FOREACH_23(f, x, ...)  f(x, 22); __FAN__FOREACH_22(f, __VA_ARGS__)
#define __FAN__FOREACH_24(f, x, ...)  f(x, 23); __FAN__FOREACH_23(f, __VA_ARGS__)
#define __FAN__FOREACH_25(f, x, ...)  f(x, 24); __FAN__FOREACH_24(f, __VA_ARGS__)
#define __FAN__FOREACH_26(f, x, ...)  f(x, 25); __FAN__FOREACH_25(f, __VA_ARGS__)
#define __FAN__FOREACH_27(f, x, ...)  f(x, 26); __FAN__FOREACH_26(f, __VA_ARGS__)
#define __FAN__FOREACH_28(f, x, ...)  f(x, 27); __FAN__FOREACH_27(f, __VA_ARGS__)
#define __FAN__FOREACH_29(f, x, ...)  f(x, 28); __FAN__FOREACH_28(f, __VA_ARGS__)
#define __FAN__FOREACH_30(f, x, ...)  f(x, 29); __FAN__FOREACH_29(f, __VA_ARGS__)


#define __FAN__FOREACH_N(_30,_29,_28,_27,_26,_25,_24,_23,_22,_21,_20,_19,_18,_17,_16,_15,_14,_13,_12,_11,_10,_9,_8,_7,_6,_5,_4,_3,_2,_1,N,...) __FAN__FOREACH_##N

#define __FAN__FOREACH_NS_1(f, x) f(x)
#define __FAN__FOREACH_NS_2(f, x, ...)   f(x), __FAN__FOREACH_NS_1(f, __VA_ARGS__)
#define __FAN__FOREACH_NS_3(f, x, ...)   f(x), __FAN__FOREACH_NS_2(f, __VA_ARGS__)
#define __FAN__FOREACH_NS_4(f, x, ...)   f(x), __FAN__FOREACH_NS_3(f, __VA_ARGS__)
#define __FAN__FOREACH_NS_5(f, x, ...)   f(x), __FAN__FOREACH_NS_4(f, __VA_ARGS__)
#define __FAN__FOREACH_NS_6(f, x, ...)   f(x), __FAN__FOREACH_NS_5(f, __VA_ARGS__)
#define __FAN__FOREACH_NS_7(f, x, ...)   f(x), __FAN__FOREACH_NS_6(f, __VA_ARGS__)
#define __FAN__FOREACH_NS_8(f, x, ...)   f(x), __FAN__FOREACH_NS_7(f, __VA_ARGS__)
#define __FAN__FOREACH_NS_9(f, x, ...)   f(x), __FAN__FOREACH_NS_8(f, __VA_ARGS__)
#define __FAN__FOREACH_NS_10(f, x, ...)  f(x), __FAN__FOREACH_NS_9(f, __VA_ARGS__)
#define __FAN__FOREACH_NS_11(f, x, ...)  f(x), __FAN__FOREACH_NS_10(f, __VA_ARGS__)
#define __FAN__FOREACH_NS_12(f, x, ...)  f(x), __FAN__FOREACH_NS_11(f, __VA_ARGS__)
#define __FAN__FOREACH_NS_13(f, x, ...)  f(x), __FAN__FOREACH_NS_12(f, __VA_ARGS__)
#define __FAN__FOREACH_NS_14(f, x, ...)  f(x), __FAN__FOREACH_NS_13(f, __VA_ARGS__)
#define __FAN__FOREACH_NS_15(f, x, ...)  f(x), __FAN__FOREACH_NS_14(f, __VA_ARGS__)
#define __FAN__FOREACH_NS_16(f, x, ...)  f(x), __FAN__FOREACH_NS_15(f, __VA_ARGS__)
#define __FAN__FOREACH_NS_17(f, x, ...)  f(x), __FAN__FOREACH_NS_16(f, __VA_ARGS__)
#define __FAN__FOREACH_NS_18(f, x, ...)  f(x), __FAN__FOREACH_NS_17(f, __VA_ARGS__)
#define __FAN__FOREACH_NS_19(f, x, ...)  f(x), __FAN__FOREACH_NS_18(f, __VA_ARGS__)
#define __FAN__FOREACH_NS_20(f, x, ...)  f(x), __FAN__FOREACH_NS_19(f, __VA_ARGS__)
#define __FAN__FOREACH_NS_21(f, x, ...)  f(x), __FAN__FOREACH_NS_20(f, __VA_ARGS__)
#define __FAN__FOREACH_NS_22(f, x, ...)  f(x), __FAN__FOREACH_NS_21(f, __VA_ARGS__)
#define __FAN__FOREACH_NS_23(f, x, ...)  f(x), __FAN__FOREACH_NS_22(f, __VA_ARGS__)
#define __FAN__FOREACH_NS_24(f, x, ...)  f(x), __FAN__FOREACH_NS_23(f, __VA_ARGS__)
#define __FAN__FOREACH_NS_25(f, x, ...)  f(x), __FAN__FOREACH_NS_24(f, __VA_ARGS__)
#define __FAN__FOREACH_NS_26(f, x, ...)  f(x), __FAN__FOREACH_NS_25(f, __VA_ARGS__)
#define __FAN__FOREACH_NS_27(f, x, ...)  f(x), __FAN__FOREACH_NS_26(f, __VA_ARGS__)
#define __FAN__FOREACH_NS_28(f, x, ...)  f(x), __FAN__FOREACH_NS_27(f, __VA_ARGS__)
#define __FAN__FOREACH_NS_29(f, x, ...)  f(x), __FAN__FOREACH_NS_28(f, __VA_ARGS__)
#define __FAN__FOREACH_NS_30(f, x, ...)  f(x), __FAN__FOREACH_NS_29(f, __VA_ARGS__)

#define __FAN__FOREACH_NS_N(_30,_29,_28,_27,_26,_25,_24,_23,_22,_21,_20,_19,_18,_17,_16,_15,_14,_13,_12,_11,_10,_9,_8,_7,_6,_5,_4,_3,_2,_1,N,...) __FAN__FOREACH_NS_##N


// NEEDS /Zc:__cplusplus /Zc:preprocessor
#define __FAN__FOREACH(f, ...) __FAN__FOREACH_N(__VA_ARGS__,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1)(f, __VA_ARGS__)

// NEEDS /Zc:__cplusplus /Zc:preprocessor
#define __FAN__FOREACH_NS(f, ...) __FAN__FOREACH_NS_N(__VA_ARGS__,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1)(f, __VA_ARGS__)

#define fan_ternary_void(f0, f1) [&]{ f0;}() : [&]{f1;}()

#define __FAN_PRINT_EACH__(x) fan::printnln("", #x"=",x, ",")
#define fan_print(...) __FAN__FOREACH_NS(__FAN_PRINT_EACH__, __VA_ARGS__);fan::print("")

#define fan_validate_buffer(buffer, function) \
	if (buffer != static_cast<decltype(buffer)>(fan::uninitialized)) \
	{ \
		function; \
	}

	// prints warning if value is -1
#define fan_validate_value(value, text) if (value == (decltype(value))fan::uninitialized) { fan::throw_error(text); }

#ifndef OFFSETLESS
	#define OFFSETLESS(ptr_m, t_m, d_m) \
		((t_m *)((uint8_t *)(ptr_m) - offsetof(t_m, d_m)))
#endif

#if defined(_DEBUG) || defined(DEBUG)
		#define fan_debug_build 1
#else
		#define fan_debug_build 0
#endif

#define fan_debug_none 0
#define fan_debug_low 1
#define fan_debug_medium 2
#define fan_debug_high 3
#define fan_debug_insane 4

#define __ca__ ,

#ifndef FAN_DEBUG
#if fan_debug_build
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

// requires /Zc:preprocessor in msvc commandline properties

#define fan_init_struct(type, ...) [&] { \
	type var__; \
	__FAN__FOREACH(__FAN__INSERTVARNAME, __VA_ARGS__); \
	return var__; \
}()

#define fan_init_struct0(type, name, ...) type name = type([&] { \
	type var__; \
	__FAN__FOREACH(__FAN__INSERTVARNAME, __VA_ARGS__); \
	return var__; \
}())

#define fan_init_id_t0(type, properties, name, ...) type name = type([&] { \
	properties var__; \
	__FAN__FOREACH(__FAN__INSERTVARNAME, __VA_ARGS__); \
	return var__; \
}())

#define fan_init_id_t1(type, ...) [&] { \
	type ## _id_t ::properties_t var__; \
	__FAN__FOREACH(__FAN__INSERTVARNAME, __VA_ARGS__); \
	return var__; \
}()

#define fan_init_id_t2(type, ...) \
	fan_init_id_t0(type, __super__secret__name0, __VA_ARGS__)

#define fan_init_id_t3(type, f, ...) [&] { \
	type ## _id_t ::properties_t var__; \
	var__ f; \
	__FAN__FOREACH(__FAN__INSERTVARNAME, __VA_ARGS__); \
	return var__; \
}()

// gets the current file's directory path
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

#define fan_temporary_struct_maker(data) __return_type_of<decltype([]{ struct {data}v; return v; })>

#define EXPAND(p0) p0

#define lstd_preprocessor_get_argn(p0, p1, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, n, ...) n
#define lstd_preprocessor_get_arg_count(...) EXPAND(lstd_preprocessor_get_argn(__VA_ARGS__, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0))

#ifndef lstd_preprocessor_ignore_first_of_every_2
#define _lstd_preprocessor_ignore_first_of_every_2_2(p0, p1) p1
#define _lstd_preprocessor_ignore_first_of_every_2_4(p0, p1, ...) p1, EXPAND(_lstd_preprocessor_ignore_first_of_every_2_2(__VA_ARGS__))
#define _lstd_preprocessor_ignore_first_of_every_2_6(p0, p1, ...) p1, EXPAND(_lstd_preprocessor_ignore_first_of_every_2_4(__VA_ARGS__))
#define _lstd_preprocessor_ignore_first_of_every_2_8(p0, p1, ...) p1, EXPAND(_lstd_preprocessor_ignore_first_of_every_2_6(__VA_ARGS__))
#define _lstd_preprocessor_ignore_first_of_every_2_10(p0, p1, ...) p1, EXPAND(_lstd_preprocessor_ignore_first_of_every_2_8(__VA_ARGS__))
#define _lstd_preprocessor_ignore_first_of_every_2_12(p0, p1, ...) p1, EXPAND(_lstd_preprocessor_ignore_first_of_every_2_10(__VA_ARGS__))
#define _lstd_preprocessor_ignore_first_of_every_2_14(p0, p1, ...) p1, EXPAND(_lstd_preprocessor_ignore_first_of_every_2_12(__VA_ARGS__))
#define _lstd_preprocessor_ignore_first_of_every_2_16(p0, p1, ...) p1, EXPAND(_lstd_preprocessor_ignore_first_of_every_2_14(__VA_ARGS__))
#define _lstd_preprocessor_ignore_first_of_every_2_start(n, ...) CONCAT(_lstd_preprocessor_ignore_first_of_every_2_,n(__VA_ARGS__))
#define lstd_preprocessor_ignore_first_of_every_2(...) _lstd_preprocessor_ignore_first_of_every_2_start(lstd_preprocessor_get_arg_count(__VA_ARGS__), __VA_ARGS__)
#endif

#ifndef lstd_preprocessor_combine_every_2
#define _lstd_preprocessor_combine_every_2_2(p0, p1) p0 p1
#define _lstd_preprocessor_combine_every_2_4(p0, p1, ...) p0 p1, EXPAND(_lstd_preprocessor_combine_every_2_2(__VA_ARGS__))
#define _lstd_preprocessor_combine_every_2_6(p0, p1, ...) p0 p1, EXPAND(_lstd_preprocessor_combine_every_2_4(__VA_ARGS__))
#define _lstd_preprocessor_combine_every_2_8(p0, p1, ...) p0 p1, EXPAND(_lstd_preprocessor_combine_every_2_6(__VA_ARGS__))
#define _lstd_preprocessor_combine_every_2_10(p0, p1, ...) p0 p1, EXPAND(_lstd_preprocessor_combine_every_2_8(__VA_ARGS__))
#define _lstd_preprocessor_combine_every_2_12(p0, p1, ...) p0 p1, EXPAND(_lstd_preprocessor_combine_every_2_10(__VA_ARGS__))
#define _lstd_preprocessor_combine_every_2_14(p0, p1, ...) p0 p1, EXPAND(_lstd_preprocessor_combine_every_2_12(__VA_ARGS__))
#define _lstd_preprocessor_combine_every_2_16(p0, p1, ...) p0 p1, EXPAND(_lstd_preprocessor_combine_every_2_14(__VA_ARGS__))
#define _lstd_preprocessor_combine_every_2_start(n, ...) CONCAT(_lstd_preprocessor_combine_every_2_,n(__VA_ARGS__))
#define lstd_preprocessor_combine_every_2(...) _lstd_preprocessor_combine_every_2_start(lstd_preprocessor_get_arg_count(__VA_ARGS__), __VA_ARGS__)
#endif

#undef __FAN_PRINT_EACH
#define __FAN_PRINT_EACH(x) #x

// no initializers allowed
#define fan_enum_string(name, ...) \
static constexpr const char* name##_strings[] = {__FAN__FOREACH_NS(__FAN_PRINT_EACH, __VA_ARGS__)}; \
enum name { __VA_ARGS__ }

#define fan_enum_class_string(name, ...) \
static constexpr const char* name##_strings[] = {__FAN__FOREACH_NS(__FAN_PRINT_EACH, __VA_ARGS__)}; \
enum class name { __VA_ARGS__ }

#if __cplusplus >= 202004L && defined(fan_compiler_msvc) && !defined(fan_compiler_clang) && __has_include("stacktrace") // wsl
	#define fan_std23
#endif

#if __cplusplus >= 202302L && !defined(fan_compiler_msvc) && __has_include("stacktrace")
	//#define fan_std23
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

  struct log_t {
    std::string data;
    std::string filename = "fan_errors.txt";
    ~log_t() {
      if (data.size()) {
        std::ofstream out(filename, std::ios::binary);
        out << data;
      }
    }
  };

  inline log_t& get_error_log() {
    static log_t log;
    return log;
  }

#ifndef __throw_error_impl
	namespace fan {
		struct exception_t {
			const char* reason;
		};

		inline void throw_error_impl(const char* reason = "") {
      std::string res(reason);
      if (res.size()) {
        get_error_log().data += res + '\n';
      }
			printf("%s\n", reason);
#ifdef fan_compiler_msvc
			//system("pause");
#endif
#if __cpp_exceptions
			throw exception_t{ .reason = reason };
#endif
		}
	}
	#define __throw_error_impl
#endif

#pragma pack(push, 1)
  #include <fan/types/bll_types.h>
#pragma pack(pop)