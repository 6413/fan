#pragma once

#include <iostream>
#include <array>
#include <vector>
#include <sstream>
#include <functional>
#include <any>

typedef intptr_t si_t;
typedef uintptr_t uint_t;
typedef intptr_t sint_t;
typedef int8_t sint8_t;
typedef int16_t sint16_t;
typedef int32_t sint32_t;
typedef int64_t sint64_t;

typedef float f32_t;
typedef double f64_t;

typedef double f_t;

typedef f32_t cf_t;

namespace fan {

	using fstring = std::wstring;

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
		for (uint_t i = 0; i < vector.size(); i++) {
			os << vector[i] << ' ';
		}
		return os;
	}

	template <typename T>
	std::ostream& operator<<(std::ostream& os, const std::vector<std::vector<T>>& vector) noexcept
	{
		for (uint_t i = 0; i < vector.size(); i++) {
			for (uint_t j = 0; j < vector[i].size(); j++) {
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
	constexpr void print(const Args&... args) {
		((std::cout << args << ' '), ...) << '\n';
	}

	template <typename ...Args>
	constexpr void wprint(const Args&... args) {
		((std::wcout << args << " "), ...) << '\n';
	}

	template <typename T>
	constexpr uint_t vector_byte_size(const typename std::vector<T>& vector)
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

	enum class platform_t { windows, linux };

	#if defined(_WIN32) || defined(_WIN64)

		constexpr platform_t platform = platform_t::windows;
		#define fan_platform_windows

		#ifdef _MSC_VER
			#define fan_compiler_visual_studio
		#elif defined(__clang__)
			#define fan_compiler_clang
		#elif defined(__GNUC__)
			#define fan_compiler_gcc
		#endif

	#elif defined(__linux__)

		constexpr platform_t platform = platform_t::windows;
		#define fan_platform_linux
		#define fan_platform_unix

	#elif defined(__unix__)

		#define fan_platform_unix

	#elif defined(__FreeBSD__)

		#define fan_platform_freebsd
		#define fan_platform_unix
	
	#endif

	template <typename T>
	struct nested_value_type : std::type_identity<T> { };

	template <typename T>
	using nested_value_type_t = typename nested_value_type<T>::type;

	template <typename T, std::size_t N>
	struct nested_value_type<std::array<T, N>>
		: std::type_identity<nested_value_type_t<T>> { };

	template <typename T>
	concept is_arithmetic_t = std::is_arithmetic<T>::value;

	template <typename T>
	concept is_not_arithmetic_t = !is_arithmetic_t<T>;

	template <bool _Test, uint_t _Ty1, uint_t _Ty2>
	struct conditional_value {
		static constexpr auto value = _Ty1;
	};

	template <uint_t _Ty1, uint_t _Ty2>
	struct conditional_value<false, _Ty1, _Ty2> {
		static constexpr auto value = _Ty2;
	};

	template <bool _Test, uint_t _Ty1, uint_t _Ty2>
	struct conditional_value_t {
		static constexpr auto value = conditional_value<_Test, _Ty1, _Ty2>::value;
	};

	#define fan_validate_buffer(buffer, function) \
	if (buffer != static_cast<decltype(buffer)>(fan::uninitialized)) \
	{ \
		function; \
	}

	// prints warning if value is -1
	#define fan_validate_value(value, text) if (value == (decltype(value))fan::uninitialized) { }

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

	template <typename T, uint32_t duplicate_id = 0>
	class class_duplicator : public T {
	
		using T::T;

	};


	static fan::fstring str_to_wstr(const std::string& s)
	{
		std::wstring ret(s.begin(), s.end());
		return ret;
	}
}

//template <platform_t T_platform>
//concept platform_windows = T_platform == platform_t::windows;

//template <platform_t T_platform>
//concept platform_linux = T_platform == platform_t::linux;