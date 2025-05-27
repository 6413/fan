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

#define _FAN_PATH(p0) <fan/p0>

#define _PATH_QUOTE(p0) STRINGIFY(p0)

#include <cstdint>
// offsetof
#include <cstddef>

#pragma pack(push, 1)

#ifndef __gfunc_api
  #define __gfunc_api static
#endif

#ifndef __gfunc_module_api
  #define __gfunc_module_api inline
#endif

#ifndef fan_api
  #define fan_api __gfunc_api
#endif

#ifndef fan_module_api
  #define fan_module_api __gfunc_module_api
#endif

#include <fan/types/bll_types.h>

template <typename T>
struct address_wrapper_t {
	using value_type = T;
	constexpr operator value_type() {
		return *(value_type*)(((std::uint8_t*)this) + sizeof(*this));
	}
};
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

typedef std::intptr_t si_t;
//typedef uintptr_t uint_t;
typedef std::intptr_t sint_t;
typedef std::int8_t sint8_t;
typedef std::int16_t sint16_t;
typedef std::int32_t sint32_t;
//typedef int64_t sint64_t;

typedef float f32_t;
typedef double f64_t;

typedef double f_t;

typedef f32_t cf_t;

namespace fan {

	inline void throw_error_impl() {
#ifdef fan_compiler_msvc
		system("pause");
#endif
#if __cpp_exceptions
		throw;
#endif
	}

#ifndef __abort
	#define __abort fan::throw_error_impl
#endif

	constexpr auto uninitialized = -1;
}

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

namespace fan {

	#define fan_ternary_void(f0, f1) [&]{ f0;}() : [&]{f1;}()

	#define __FAN_PRINT_EACH__(x) fan::print_no_endline("", #x"=",x, ",")
	#define fan_print(...) __FAN__FOREACH_NS(__FAN_PRINT_EACH__, __VA_ARGS__);fan::print("")

	//template <typename T>
	//fan::wstring to_wstring(const T a_value, const int n = 2)
	//{
	//	std::wostringstream out;
	//	out.precision(n);
	//	out << std::fixed << a_value;
	//	return out.str().c_str();
	//}


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

	template <typename Callable>
	struct return_type_of_membr;

	template <typename R, typename C, typename... Args>
	struct return_type_of_membr<R(C::*)(Args...)> {
		using type = R;
	};

	template <typename R, typename C, typename... Args>
	struct return_type_of_membr<R(C::*)(Args...) const> {
		using type = R;
	};

	template <typename Callable>
	using return_type_of_membr_t = typename return_type_of_membr<Callable>::type;

	/*template<typename Callable>
	using return_type_of_t = typename decltype(std::function{ std::declval<Callable>() })::result_type;*/

	static constexpr std::uint64_t get_hash(const char* str) {
		std::uint64_t result = 0xcbf29ce484222325; // FNV offset basis

		std::uint32_t i = 0;

		if (str == nullptr) {
			return 0;
		}

		while (str[i] != 0) {
			result ^= (std::uint64_t)str[i];
			result *= 1099511628211; // FNV prime
			i++;
		}

		return result;
	}

	//#define fan_requires_rule(type, rule) \
	//	[] <typename dont_shadow_me2_t>() constexpr { \
	//		return requires(dont_shadow_me2_t t) { rule; } == true; \
	//	}.template operator()<type>()

	//#define fan_has_function(type, func_call) \
	//	fan_requires_rule(type, t.func_call)

	//#define fan_if_has_function(ptr, func_name, params) \
	//	if constexpr (fan_has_function(std::remove_reference_t<std::remove_pointer_t<decltype(ptr)>>, func_name params)) \
	//	[&] <typename runtime_t>(runtime_t* This) { \
	//		if constexpr (fan_has_function(runtime_t, func_name params)) { \
	//			This->func_name params ;\
	//		} \
	//	}(ptr)

	//#define fan_if_has_function_get(ptr, func_name, params, data) \
	//	if constexpr (fan_has_function(std::remove_reference_t<std::remove_pointer_t<decltype(ptr)>>, func_name params)) \
	//	[&] <typename runtime_t>(runtime_t* This) { \
	//		if constexpr (fan_has_function(runtime_t, func_name params)) { \
	//			data = This->func_name params ;\
	//		} \
	//	}(ptr)

	//#define fan_has_variable(type, var_name) \
	//	fan_requires_rule(type, t.var_name)

	//#define fan_if_has_variable(ptr, var_name, todo) \
	//[&] <typename dont_shadow_me_t>(dont_shadow_me_t* This) { \
	//	if constexpr (fan_has_variable(dont_shadow_me_t, var_name)) { \
	//		todo ;\
	//	} \
	//}(ptr);

}

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
		#define __builtin_memset memset
	#endif
	#ifndef __builtin_memcpy
		#define __builtin_memcpy memcpy
	#endif
	#ifndef __builtin_memmove
		#define __builtin_memmove memmove
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

namespace fan {
	#define fan_temporary_struct_maker(data) __return_type_of<decltype([]{ struct {data}v; return v; })>

	template <typename T>
	struct any_type_wrap_t {

		operator T& () {
			return v;
		}
		operator T() const {
			return v;
		}
		void operator=(const auto& nv) {
			v = nv;
		}
		T v;
	};

	template <typename T0, typename T1>
	struct mark : any_type_wrap_t<T1> {
		using mark_type_t = T0;
		using type_t = T1;

		void operator=(const auto& nv) {
			any_type_wrap_t<T1>::operator=(nv);
		}
	};

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

	//// __VA_ARGS__ is not compile time in clang according to clang
	#define fan_make_flexible_array(type, name, ...) \
	std::array<type, std::initializer_list<type>{__VA_ARGS__}.size()> name = {__VA_ARGS__}

	template<class T, typename U>
	std::int64_t member_offset(U T::* member)
	{
		return reinterpret_cast<std::int64_t>(
			&(reinterpret_cast<T const volatile*>(0)->*member)
			);
	}

	#undef __FAN_PRINT_EACH
	#define __FAN_PRINT_EACH(x) #x

	// no initializers allowed
	#define fan_enum_string(name, ...) \
	static constexpr const char* name##_strings[] = {__FAN__FOREACH_NS(__FAN_PRINT_EACH, __VA_ARGS__)}; \
	enum name { __VA_ARGS__ }

	#define fan_enum_class_string(name, ...) \
	static constexpr const char* name##_strings[] = {__FAN__FOREACH_NS(__FAN_PRINT_EACH, __VA_ARGS__)}; \
	enum class name { __VA_ARGS__ }

}

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

template<typename T1, typename T2>
struct pair {
  T1 first;
  T2 second;
};

template<typename It>
struct iterator_traits {
    using reference = decltype(*(declval<It>()));
};

template <typename T>
T&& declval() noexcept;


template<bool B, typename T, typename F>
struct conditional { using type = T; };

template<typename T, typename F>
struct conditional<false, T, F> { using type = F; };

template<bool B, typename T, typename F>
using conditional_t = typename conditional<B, T, F>::type;

template<typename T>
struct is_const { static constexpr bool value = false; };

template<typename T>
struct is_const<const T> { static constexpr bool value = true; };

template<typename T>
constexpr bool is_const_v = is_const<T>::value;

namespace fan_detail {
  template<typename It>
  struct iterator_traits {
    using reference = decltype(*(declval<It>()));
  };

  template <typename Container>
  auto get_begin(Container& container) -> decltype(container.begin()) {
    return container.begin();
  }

  template <typename Container>
  auto get_end(Container& container) -> decltype(container.end()) {
    return container.end();
  }

  template <typename Container>
  auto get_size(Container& container) -> decltype(container.size()) {
    return container.size();
  }
}

template <typename It>
class enumerate_iterator {
  It _iter;
  std::size_t _index;
public:
  using value_type = pair<std::size_t, typename fan_detail::iterator_traits<It>::reference>;
  using reference = value_type;
  using pointer = void;

  enumerate_iterator(It iter, std::size_t index) : _iter(iter), _index(index) {}

  reference operator*() const {
    return { _index, *_iter };
  }

  enumerate_iterator& operator++() {
    ++_iter;
    ++_index;
    return *this;
  }

  bool operator!=(const enumerate_iterator& other) const {
    return _iter != other._iter;
  }
};

template <typename Container>
class enumerate_view {
  Container& _container;
public:
  using iterator = enumerate_iterator<
    conditional_t<
    is_const_v<Container>,
    decltype(fan_detail::get_begin(declval<const Container&>())),
    decltype(fan_detail::get_begin(declval<Container&>()))
    >
  >;

  enumerate_view(Container& container) : _container(container) {}

  iterator begin() {
    return { fan_detail::get_begin(_container), 0 };
  }

  iterator end() {
    return { fan_detail::get_end(_container), fan_detail::get_size(_container) };
  }
};

namespace fan {
  struct enumerate_fn {
    template <typename Container>
    auto operator()(Container& container) const {
      return enumerate_view<Container>{container};
    }
  };

  inline constexpr enumerate_fn enumerate{};
}

template <typename Container>
auto operator|(Container& container, const fan::enumerate_fn& view) {
  return view(container);
}
