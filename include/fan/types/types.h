#pragma once

#define STRINGIFY(p0) #p0
#define STRINGIFY_DEFINE(a) STRINGIFY(a)

#define _CONCAT(_0_m, _1_m) _0_m ## _1_m
#define CONCAT(_0_m, _1_m) _CONCAT(_0_m, _1_m)
#define _CONCAT2(_0_m, _1_m) _0_m ## _1_m
#define CONCAT2(_0_m, _1_m) _CONCAT(_0_m, _1_m)
#define _CONCAT3(_0_m, _1_m, _2_m) _0_m ## _1_m ## _2_m
#define CONCAT3(_0_m, _1_m, _2_m) _CONCAT3(_0_m, _1_m, _2_m)
#define _CONCAT4(_0_m, _1_m, _2_m, _3_m) _0_m ## _1_m ## _2_m ## _3_m
#define CONCAT4(_0_m, _1_m, _2_m, _3_m) _CONCAT4(_0_m, _1_m, _2_m, _3_m)

#ifndef FAN_INCLUDE_PATH
	#define _FAN_PATH(p0) <fan/p0>
#else
	#define FAN_INCLUDE_PATH_END fan/
	#define _FAN_PATH(p0) <FAN_INCLUDE_PATH/fan/p0>
#define _FAN_PATH_QUOTE(p0) STRINGIFY_DEFINE(FAN_INCLUDE_PATH) "/fan/" STRINGIFY(p0)

#endif

#define _PATH_QUOTE(p0) STRINGIFY(p0)

#include <iostream>
#include <array>
#include <vector>
#include <sstream>
#include <functional>
#include <type_traits>
#include <cstdint>
#include <regex>
#include <charconv>

#pragma pack(push, 1)

#ifndef __empty_struct
  #define __empty_struct __empty_struct
  struct __empty_struct {

  };
#endif

#ifndef __MemorySet
  #define __MemorySet(src, dst, size) memset(dst, src, size)
#endif

#ifndef __abort
  #define __abort() fan::throw_error("")
#endif

template <typename T>
struct address_wrapper_t {
  using value_type = T;
  constexpr operator value_type() {
    return *(value_type*)(((uint8_t*)this) + sizeof(*this));
  }
};
#pragma pack(pop)

// override to utf-8 - if file already utf8 it breaks somehow, probably msvc bug
#pragma execution_character_set("utf-8")

#if defined(_WIN32) || defined(_WIN64)

//constexpr platform_t platform = platform_t::windows;
#define fan_platform_windows

#if defined(__clang__)
	#define fan_compiler_clang
#elif defined(_MSC_VER)
	#define fan_compiler_visual_studio
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
#if __cplusplus >= 202004L && defined(fan_compiler_visual_studio) && !defined(fan_compiler_clang)
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
	void print(const Args&... args);

	template <typename ...Args>
	constexpr void wprint(const Args&... args);

  template <typename ...Args>
  static void throw_error(const Args&... args);

  template<typename T>
  struct has_bracket_operator
  {
    template<typename U>
    static constexpr decltype(std::declval<U>()[0], bool{}) test(int)
    {
      return true;
    }

    template<typename>
    static constexpr bool test(...)
    {
      return false;
    }

    static constexpr bool value = test<T>(0);
  };
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
	void print(const Args&... args) {
		((std::cout << args << ' '), ...) << '\n';
	}

  template <typename... T>
  static FMT_INLINE auto print_format(fmt::format_string<T...> fmt, T&&... args) {
    fan::print(fmt::vformat(fmt, fmt::make_format_args(args...)));
  }

	template <typename ...Args>
	constexpr void wprint(const Args&... args) {
		((std::wcout << args << " "), ...) << '\n';
	}

  template <typename... T>
  static void throw_error_format(fmt::format_string<T...> fmt, T&&... args) {
    fan::print_format(fmt, args...);
    #ifdef fan_compiler_visual_studio
    system("pause");
    #endif
    #if __cpp_exceptions
    throw std::runtime_error("");
    #endif
    //exit(1);
  }

  template <typename ...Args>
  static void throw_error(const Args&... args) {
    fan::print(args...);
    #ifdef fan_compiler_visual_studio
    system("pause");
    #endif
    #if __cpp_exceptions
    throw std::runtime_error("");
    #endif
    //exit(1);
  }

  static void assert_test(bool test) {
    if (!test) {
      fan::throw_error("assert failed");
    }
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
	//template <typename T>
	//fan::wstring to_wstring(const T a_value, const int n = 2)
	//{
	//	std::wostringstream out;
	//	out.precision(n);
	//	out << std::fixed << a_value;
	//	return out.str().c_str();
	//}

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

	constexpr inline uint64_t get_hash(const fan::string& str) {
		uint64_t result = 0xcbf29ce484222325; // FNV offset basis

		uint32_t i = 0;

		while (str[i] != 0) {
			result ^= (uint64_t)str[i];
			result *= 1099511628211; // FNV prime
			i++;
		}

		return result;
	}

	template <class T>
	class has_member_type {
		struct One { char a[1]; };
		struct Two { char a[2]; };

		template <class U>
		static One foo(typename U::type*);

		template <class U>
		static Two foo(...);

	public:
		static const bool value = sizeof(foo<T>(nullptr)) == sizeof(One);
	};

  template <typename From>
  class auto_cast {
  public:
    explicit constexpr auto_cast(From & t) noexcept
      : val{ t }
    {
    }

    template <typename To>
    constexpr operator To() const noexcept(noexcept(reinterpret_cast<To*>(&std::declval<From>())))
    {
      return *reinterpret_cast<To*>(&val);
    }

  private:
    From & val;
  };

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

  // unstable
  template <typename, typename, typename = void>
  struct has_function : std::false_type {};
  
  // unstable
  template <typename C, typename Ret, typename... Args>
  struct has_function<C, Ret(Args...), std::void_t<decltype(std::declval<C>().operator()(std::declval<Args>()...))>> : std::is_convertible<decltype(std::declval<C>().operator()(std::declval<Args>()...)), Ret>::type {};



  //#define fan_has_variable(type, var_name) [&](type p = type())constexpr{ return requires{p.var_name;}; }()

  //template <typename T>
  //constexpr bool has_variable() {

  //}
  #define fan_has_variable_struct(var_name) \
  template <typename T> \
  struct CONCAT(has_,var_name) { \
    template <typename U> \
    static auto test(int) -> decltype(std::declval<U>().var_name, std::true_type()); \
 \
    template <typename> \
    static auto test(...) -> std::false_type; \
 \
    static constexpr bool value = decltype(test<T>(0))::value; \
  }; \
  template <typename U> \
  static constexpr bool CONCAT(CONCAT(has_,var_name), _v) = CONCAT(has_, var_name)<U>::value;


  #define fan_has_function_concept(func_name) \
   template<typename U, typename... Args> \
    struct CONCAT(has_,func_name) { \
      static constexpr bool value = requires(U u, Args... args) { \
        u.func_name(args...); \
      }; \
    }; \
    template <typename U, typename... Args> \
    static constexpr bool CONCAT(CONCAT(has_,func_name), _v) = CONCAT(has_, func_name)<U, Args...>::value;
}


//template <platform_t T_platform>
//concept platform_windows = T_platform == platform_t::windows;

//template <platform_t T_platform>
//concept platform_linux = T_platform == platform_t::linux;

#ifndef OFFSETLESS
	#define OFFSETLESS(ptr_m, t_m, d_m) \
		((t_m *)((uint8_t *)(ptr_m) - offsetof(t_m, d_m)))
#endif

#define fan_debug_none 0
#define fan_debug_low 1
#define fan_debug_medium 2
#define fan_debug_high 3
#define fan_debug_insanity 4

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

#ifndef __clz
#define __clz __clz
static uint8_t __clz32(uint32_t p0) {
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
static uint8_t __clz64(uint64_t p0) {
	#if defined(__GNUC__)
	return __builtin_clzll(p0);
	#elif defined(_MSC_VER)
	DWORD trailing_zero = 0;
	if (_BitScanReverse64(&trailing_zero, p0)) {
		return uint8_t((DWORD)63 - trailing_zero);
	}
	else {
		return 0;
	}
	#else
	#error ?
	#endif
}

#if defined(__x86_64__) || defined(_M_AMD64)
	#define SYSTEM_BIT 64
	#define SYSTEM_BYTE 8
#elif defined(__i386__)
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

#define __FAN__INSERTVARNAME(x) var__ x

#define __FAN__FOREACH_1(f, x) f(x)
#define __FAN__FOREACH_2(f, x, ...)  f(x); __FAN__FOREACH_1(f, __VA_ARGS__)
#define __FAN__FOREACH_3(f, x, ...)  f(x); __FAN__FOREACH_2(f, __VA_ARGS__)
#define __FAN__FOREACH_4(f, x, ...)  f(x); __FAN__FOREACH_3(f, __VA_ARGS__)
#define __FAN__FOREACH_5(f, x, ...)  f(x); __FAN__FOREACH_4(f, __VA_ARGS__)
#define __FAN__FOREACH_6(f, x, ...)  f(x); __FAN__FOREACH_5(f, __VA_ARGS__)
#define __FAN__FOREACH_7(f, x, ...)  f(x); __FAN__FOREACH_6(f, __VA_ARGS__)
#define __FAN__FOREACH_8(f, x, ...)  f(x); __FAN__FOREACH_7(f, __VA_ARGS__)
#define __FAN__FOREACH_9(f, x, ...)  f(x); __FAN__FOREACH_8(f, __VA_ARGS__)
#define __FAN__FOREACH_10(f, x, ...)  f(x); __FAN__FOREACH_9(f, __VA_ARGS__)
#define __FAN__FOREACH_11(f, x, ...)  f(x); __FAN__FOREACH_10(f, __VA_ARGS__)
#define __FAN__FOREACH_12(f, x, ...)  f(x); __FAN__FOREACH_11(f, __VA_ARGS__)
#define __FAN__FOREACH_13(f, x, ...)  f(x); __FAN__FOREACH_12(f, __VA_ARGS__)
#define __FAN__FOREACH_14(f, x, ...)  f(x); __FAN__FOREACH_13(f, __VA_ARGS__)
#define __FAN__FOREACH_15(f, x, ...)  f(x); __FAN__FOREACH_14(f, __VA_ARGS__)
#define __FAN__FOREACH_16(f, x, ...)  f(x); __FAN__FOREACH_15(f, __VA_ARGS__)
#define __FAN__FOREACH_17(f, x, ...)  f(x); __FAN__FOREACH_16(f, __VA_ARGS__)
#define __FAN__FOREACH_18(f, x, ...)  f(x); __FAN__FOREACH_17(f, __VA_ARGS__)
#define __FAN__FOREACH_19(f, x, ...)  f(x); __FAN__FOREACH_18(f, __VA_ARGS__)
#define __FAN__FOREACH_20(f, x, ...)  f(x); __FAN__FOREACH_19(f, __VA_ARGS__)
#define __FAN__FOREACH_21(f, x, ...)  f(x); __FAN__FOREACH_20(f, __VA_ARGS__)
#define __FAN__FOREACH_22(f, x, ...)  f(x); __FAN__FOREACH_21(f, __VA_ARGS__)
#define __FAN__FOREACH_23(f, x, ...)  f(x); __FAN__FOREACH_22(f, __VA_ARGS__)
#define __FAN__FOREACH_24(f, x, ...)  f(x); __FAN__FOREACH_23(f, __VA_ARGS__)
#define __FAN__FOREACH_25(f, x, ...)  f(x); __FAN__FOREACH_24(f, __VA_ARGS__)
#define __FAN__FOREACH_26(f, x, ...)  f(x); __FAN__FOREACH_25(f, __VA_ARGS__)
#define __FAN__FOREACH_27(f, x, ...)  f(x); __FAN__FOREACH_26(f, __VA_ARGS__)
#define __FAN__FOREACH_28(f, x, ...)  f(x); __FAN__FOREACH_27(f, __VA_ARGS__)
#define __FAN__FOREACH_29(f, x, ...)  f(x); __FAN__FOREACH_28(f, __VA_ARGS__)
#define __FAN__FOREACH_30(f, x, ...)  f(x); __FAN__FOREACH_29(f, __VA_ARGS__)

#define __FAN__FOREACH_N(n, ...) __FAN__FOREACH_##n
#define __FAN__FOREACH_N(_30,_29,_28,_27,_26,_25,_24,_23,_22,_21,_20,_19,_18,_17,_16,_15,_14,_13,_12,_11,_10,_9,_8,_7,_6,_5,_4,_3,_2,_1,N,...) __FAN__FOREACH_##N
#define __FAN__FOREACH(f, ...) __FAN__FOREACH_N(__VA_ARGS__,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1)(f, __VA_ARGS__)

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

#define fan_init_id_t0(type, name, ...) type ## _id_t name = type ## _id_t([&] { \
  type ## _id_t ::properties_t var__; \
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