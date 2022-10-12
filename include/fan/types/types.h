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
#include <regex>

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

// TBD
#if __cplusplus >= 199711L && defined(fan_compiler_visual_studio)
#define fan_std23
#endif
#undef fan_std23

#if defined(fan_std23)
#include <stacktrace>
#endif

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

	struct string;

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
	constexpr void print_no_space(const Args&... args);

	template <typename ...Args>
	constexpr void print_no_endline(const Args&... args);

	template <typename ...Args>
	constexpr void wprint_no_endline(const Args&... args);

	template <typename ...Args>
	constexpr void print(const Args&... args);

	template <typename ...Args>
	constexpr void wprint(const Args&... args);

	static void throw_error(const fan::string& message);

}

#include _FAN_PATH(types/function.h)
#include _FAN_PATH(types/fstring.h)
#include _FAN_PATH(time/time.h)

namespace fan {

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

	static void throw_error(const fan::string& message) {
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
	constexpr uintptr_t vector_byte_size(const typename std::vector<T>& vector)
	{
		return sizeof(T) * vector.size();
	}

	template <typename T>
	fan::string to_string(const T a_value, const int n = 2)
	{
		std::ostringstream out;
		out.precision(n);
		out << std::fixed << a_value;
		return out.str().c_str();
	}
	template <typename T>
	std::wstring to_wstring(const T a_value, const int n = 2)
	{
		std::wostringstream out;
		out.precision(n);
		out << std::fixed << a_value;
		return out.str().c_str();
	}

	template <typename T, typename T2>
	constexpr bool is_flag(T value, T2 flag) {
		return (value & flag) == flag;
	}

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

	static std::wstring str_to_wstr(const fan::string& s)
	{
		std::wstring ret(s.begin(), s.end());
		return ret;
	}

	static void print_warning(const fan::string& message) {
#ifndef fan_disable_warnings
		fan::print("fan warning: ", message);
#endif
	}
	static void print_warning_no_space(const fan::string& message) {
#ifndef fan_disable_warnings
		fan::print_no_space("fan warning:", message);
#endif
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

#pragma pack(push, 1)
	template <typename Member, std::size_t O>
	struct Pad {
		char pad[O];
		Member m;
	};
#pragma pack(pop)

	template<typename Member>
	struct Pad<Member, 0> {
		Member m;
	};

	template <typename Base, typename Member, std::size_t O>
	struct MakeUnion {
		union U {
			char c;
			Base base;
			Pad<Member, O> pad;
			constexpr U() noexcept : c{} {};
		};
		constexpr static U u{};
	};

	template <typename Member, typename Base, typename Orig>
	struct ofof_impl {
		template<std::size_t off, auto union_part = &MakeUnion<Base, Member, off>::u>
		static constexpr std::ptrdiff_t offset2(Member Orig::* member) {
			if constexpr (off > sizeof(Base)) {
				throw 1;
			}
			else {
				const auto diff1 = &((static_cast<const Orig*>(&union_part->base))->*member);
				const auto diff2 = &union_part->pad.m;
				if (diff1 > diff2) {
					constexpr auto MIN = sizeof(Member) < alignof(Orig) ? sizeof(Member) : alignof(Orig);
					return offset2<off + MIN>(member);
				}
				else {
					return off;
				}
			}
		}
	};


	template<class Member, class Base>
	std::tuple<Member, Base> get_types(Member Base::*);

	template <class TheBase = void, class TT>
	inline constexpr std::ptrdiff_t ofof(TT member) {
		using T = decltype(get_types(std::declval<TT>()));
		using Member = std::tuple_element_t<0, T>;
		using Orig = std::tuple_element_t<1, T>;
		using Base = std::conditional_t<std::is_void_v<TheBase>, Orig, TheBase>;
		return ofof_impl<Member, Base, Orig>::template offset2<0>(member);
	}

	template <auto member, class TheBase = void>
	inline constexpr std::ptrdiff_t ofof() {
		return ofof<TheBase>(member);
	}

	//template <typename T, typename U>
	//constexpr auto offsetless(void* ptr, U T::* member) {
	//  return (T*)((uint8_t*)(ptr)-((char*)&((T*)nullptr->*member) - (char*)nullptr));
	//}

	template <typename T>
	fan::string combine_values(T t) {
		if constexpr (std::is_same<T, const char*>::value) {
			return t;
		}
		else {
			return std::to_string(t);
		}
	}

	template <typename T2, typename ...T>
	static fan::string combine_values(T2 first, T... args) {
		if constexpr (std::is_same<T2, const char*>::value ||
			std::is_same<T2, fan::string>::value) {
			return first + f(args...);
		}
		else {
			return std::to_string(first) + f(args...);
		}
	}

	template<typename Callable>
	using return_type_of_t = typename decltype(std::function{ std::declval<Callable>() })::result_type;

	inline uint64_t get_hash(const fan::string& str) {
		uint64_t result = 0xcbf29ce484222325; // FNV offset basis

		uint32_t i = 0;

		while (str[i] != 0) {
			result ^= str[i];
			result *= 1099511628211; // FNV prime
			i++;
		}

		return result;
}

	constexpr const char* file_name(const char* path) {
		const char* file = path;
		while (*path) {
#if defined(fan_platform_windows)
			if (*path++ == '\\') {
#elif defined(fan_platform_unix)
			if (*path++ == '/') {
#endif
				file = path;
			}
			}
		return file;
		}

	namespace debug {

		static void print_stacktrace() {
#ifdef fan_std23
			std::stacktrace st;
			fan::print(st.current());
#elif defined(fan_platform_unix)
			// waiting for stacktrace to be released for clang lib++
#else
			fan::print("stacktrace not supported");
#endif
		}
	}

	namespace performance {
		void measure(auto l) {
			fan::time::clock c;
			c.start();
			l();
			fan::print(c.elapsed());
		}
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

static void PR_abort() {
	fan::throw_error("");
}