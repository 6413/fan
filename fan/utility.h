// Macro based utility
#pragma once

#define STRINGIFY(p0) #p0
#define STRINGIFY_DEFINE(a) STRINGIFY(a)
#define REMOVE_TEMPLTE(a)  

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

#pragma pack(push, 1)
#include <fan/types/bll_types.h>
#pragma pack(pop)

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

#define __FAN_PRINT_EACH__(x) fan::print_no_endline("", #x"=",x, ",")
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

#ifndef fan_debug
#if fan_debug_build
	#define fan_debug fan_debug_high
	#define __sanit 1
#else
	#define fan_debug fan_debug_none
#endif
#endif

#ifndef fan_use_uninitialized
#define fan_use_uninitialized 0
#endif

#ifdef fan_platform_windows
#pragma comment(lib, "Onecore.lib")
#endif

using DWORD = unsigned long;

#if defined(_MSC_VER)
	#ifndef __builtin_memset
		#define __builtin_memset std::memset
	#endif
	#ifndef __builtin_memcpy
		#define __builtin_memcpy std::memcpy
	#endif
	#ifndef __builtin_memmove
		#define __builtin_memmove std::memmove
	#endif
#endif

#define __platform_libc

#ifndef __clz
#define __clz __clz
#ifndef __clz32
	inline uint8_t __clz32(uint32_t p0)
	{
	#if defined(__GNUC__)
		return __builtin_clz(p0);
	#elif defined(_MSC_VER)
		DWORD trailing_zero = 0;
		if (_BitScanReverse(&trailing_zero, p0)) {
			return uint8_t((DWORD)31 - trailing_zero);
		}
		else {
			return 0;
		}
	#else
	#error ?
	#endif
	}
	#define __clz32 __clz32
#endif

#ifndef __clz64
inline uint8_t __clz64(uint64_t p0) {
#if defined(__GNUC__)
	return __builtin_clzll(p0);
#elif defined(_WIN64)

	DWORD trailing_zero = 0;
	if (_BitScanReverse64(&trailing_zero, p0)) {
		return uint8_t((DWORD)63 - trailing_zero);
	}
	else {
		return 0;
	}
#else
	fan::throw_error_impl();
//#error ?
#endif
}
#define __clz64 __clz64
#endif
#if defined(__x86_64__) || defined(_M_AMD64)
	#define SYSTEM_BIT 64
	#define SYSTEM_BYTE 8
#elif defined(__i386__) || defined(_WIN32)
	#define SYSTEM_BIT 32
	#define SYSTEM_BYTE 4
#else 
	#error failed to find platform
#endif

static uint8_t __clz(uintptr_t p0) {
	#if SYSTEM_BIT == 32
		return __clz32(p0);
	#elif SYSTEM_BIT == 64
		return __clz64(p0);
	#else
		#error ?
	#endif
}
#endif

#ifndef __return_type_of
	#define __return_type_of fan::return_type_of_t
#endif

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

#ifndef ENDIAN
	#if defined(__BYTE_ORDER)
		#if __BYTE_ORDER == __BIG_ENDIAN
			#define ENDIAN 0
		#elif __BYTE_ORDER == __LITTLE_ENDIAN
			#define ENDIAN 1
		#else
			#error ?
		#endif
	#elif defined(__BYTE_ORDER__)
		#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
			#define ENDIAN 0
		#elif __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
			#define ENDIAN 1
		#else
			#error ?
		#endif
	#elif defined(__x86_64__) && __x86_64__ == 1
		#define ENDIAN 1
	#elif defined(fan_platform_windows)
		#define ENDIAN 1
	#else
		#error ?
	#endif
#endif


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

// std::unreachable

#ifndef __unreachable
#if defined(fan_compiler_msvc)
#define __unreachable() __assume(false)
#elif defined(fan_compiler_clang) || defined(fan_compiler_gcc)
#define __unreachable() __builtin_unreachable()
#endif
#endif

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

// WITCH compatibility
#ifndef __compile_time_64log2
	#define __compile_time_64log2(v) ( \
		(v) >= 0x8000000000000000 ? 0x3f : \
		(v) >= 0x4000000000000000 ? 0x3e : \
		(v) >= 0x2000000000000000 ? 0x3d : \
		(v) >= 0x1000000000000000 ? 0x3c : \
		(v) >= 0x0800000000000000 ? 0x3b : \
		(v) >= 0x0400000000000000 ? 0x3a : \
		(v) >= 0x0200000000000000 ? 0x39 : \
		(v) >= 0x0100000000000000 ? 0x38 : \
		(v) >= 0x0080000000000000 ? 0x37 : \
		(v) >= 0x0040000000000000 ? 0x36 : \
		(v) >= 0x0020000000000000 ? 0x35 : \
		(v) >= 0x0010000000000000 ? 0x34 : \
		(v) >= 0x0008000000000000 ? 0x33 : \
		(v) >= 0x0004000000000000 ? 0x32 : \
		(v) >= 0x0002000000000000 ? 0x31 : \
		(v) >= 0x0001000000000000 ? 0x30 : \
		(v) >= 0x0000800000000000 ? 0x2f : \
		(v) >= 0x0000400000000000 ? 0x2e : \
		(v) >= 0x0000200000000000 ? 0x2d : \
		(v) >= 0x0000100000000000 ? 0x2c : \
		(v) >= 0x0000080000000000 ? 0x2b : \
		(v) >= 0x0000040000000000 ? 0x2a : \
		(v) >= 0x0000020000000000 ? 0x29 : \
		(v) >= 0x0000010000000000 ? 0x28 : \
		(v) >= 0x0000008000000000 ? 0x27 : \
		(v) >= 0x0000004000000000 ? 0x26 : \
		(v) >= 0x0000002000000000 ? 0x25 : \
		(v) >= 0x0000001000000000 ? 0x24 : \
		(v) >= 0x0000000800000000 ? 0x23 : \
		(v) >= 0x0000000400000000 ? 0x22 : \
		(v) >= 0x0000000200000000 ? 0x21 : \
		(v) >= 0x0000000100000000 ? 0x20 : \
		(v) >= 0x0000000080000000 ? 0x1f : \
		(v) >= 0x0000000040000000 ? 0x1e : \
		(v) >= 0x0000000020000000 ? 0x1d : \
		(v) >= 0x0000000010000000 ? 0x1c : \
		(v) >= 0x0000000008000000 ? 0x1b : \
		(v) >= 0x0000000004000000 ? 0x1a : \
		(v) >= 0x0000000002000000 ? 0x19 : \
		(v) >= 0x0000000001000000 ? 0x18 : \
		(v) >= 0x0000000000800000 ? 0x17 : \
		(v) >= 0x0000000000400000 ? 0x16 : \
		(v) >= 0x0000000000200000 ? 0x15 : \
		(v) >= 0x0000000000100000 ? 0x14 : \
		(v) >= 0x0000000000080000 ? 0x13 : \
		(v) >= 0x0000000000040000 ? 0x12 : \
		(v) >= 0x0000000000020000 ? 0x11 : \
		(v) >= 0x0000000000010000 ? 0x10 : \
		(v) >= 0x0000000000008000 ? 0x0f : \
		(v) >= 0x0000000000004000 ? 0x0e : \
		(v) >= 0x0000000000002000 ? 0x0d : \
		(v) >= 0x0000000000001000 ? 0x0c : \
		(v) >= 0x0000000000000800 ? 0x0b : \
		(v) >= 0x0000000000000400 ? 0x0a : \
		(v) >= 0x0000000000000200 ? 0x09 : \
		(v) >= 0x0000000000000100 ? 0x08 : \
		(v) >= 0x0000000000000080 ? 0x07 : \
		(v) >= 0x0000000000000040 ? 0x06 : \
		(v) >= 0x0000000000000020 ? 0x05 : \
		(v) >= 0x0000000000000010 ? 0x04 : \
		(v) >= 0x0000000000000008 ? 0x03 : \
		(v) >= 0x0000000000000004 ? 0x02 : \
		(v) >= 0x0000000000000002 ? 0x01 : \
		0 \
	)
#endif

#ifndef __compile_time_32log2
	#define __compile_time_32log2(v) ( \
		(v) >= 0x80000000 ? 0x1f : \
		(v) >= 0x40000000 ? 0x1e : \
		(v) >= 0x20000000 ? 0x1d : \
		(v) >= 0x10000000 ? 0x1c : \
		(v) >= 0x08000000 ? 0x1b : \
		(v) >= 0x04000000 ? 0x1a : \
		(v) >= 0x02000000 ? 0x19 : \
		(v) >= 0x01000000 ? 0x18 : \
		(v) >= 0x00800000 ? 0x17 : \
		(v) >= 0x00400000 ? 0x16 : \
		(v) >= 0x00200000 ? 0x15 : \
		(v) >= 0x00100000 ? 0x14 : \
		(v) >= 0x00080000 ? 0x13 : \
		(v) >= 0x00040000 ? 0x12 : \
		(v) >= 0x00020000 ? 0x11 : \
		(v) >= 0x00010000 ? 0x10 : \
		(v) >= 0x00008000 ? 0x0f : \
		(v) >= 0x00004000 ? 0x0e : \
		(v) >= 0x00002000 ? 0x0d : \
		(v) >= 0x00001000 ? 0x0c : \
		(v) >= 0x00000800 ? 0x0b : \
		(v) >= 0x00000400 ? 0x0a : \
		(v) >= 0x00000200 ? 0x09 : \
		(v) >= 0x00000100 ? 0x08 : \
		(v) >= 0x00000080 ? 0x07 : \
		(v) >= 0x00000040 ? 0x06 : \
		(v) >= 0x00000020 ? 0x05 : \
		(v) >= 0x00000010 ? 0x04 : \
		(v) >= 0x00000008 ? 0x03 : \
		(v) >= 0x00000004 ? 0x02 : \
		(v) >= 0x00000002 ? 0x01 : \
		0 \
	)
#endif

#ifndef __compile_time_log2
	#define __compile_time_log2 CONCAT3(__compile_time_,SYSTEM_BIT,log2)
#endif

#ifndef __fast_8log2
	#define __fast_8log2 __fast_8log2
	inline std::uint8_t __fast_8log2(std::uint8_t v){
		return 31 - __clz32(v);
	}
#endif
#ifndef __fast_16log2
	#define __fast_16log2 __fast_16log2
	inline std::uint8_t __fast_16log2(std::uint16_t v){
		return 31 - __clz32(v);
	}
#endif
#ifndef __fast_32log2
	#define __fast_32log2 __fast_32log2
	inline std::uint8_t __fast_32log2(std::uint32_t v){
		return 31 - __clz32(v);
	}
#endif
#ifndef __fast_64log2
	#define __fast_64log2 __fast_64log2
	inline std::uint8_t __fast_64log2(std::uint64_t v){
		return 63 - __clz64(v);
	}
#endif
#ifndef __fast_log2
	#define __fast_log2 CONCAT3(__fast_,SYSTEM_BIT,log2)
#endif

#ifndef __throw_error_impl
	namespace fan {
		struct exception_t {
			const char* reason;
		};

		inline void throw_error_impl(const char* reason = "") {
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

#ifndef __abort
	#define __abort fan::throw_error_impl
#endif

#ifndef __generic_malloc
	#define __generic_malloc(n) std::malloc(n)
#endif
#ifndef __generic_realloc
	#define __generic_realloc(ptr, n) std::realloc(ptr, n)
#endif
#ifndef __generic_free
	#define __generic_free(ptr) std::free(ptr)
#endif