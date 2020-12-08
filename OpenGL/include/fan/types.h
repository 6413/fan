#pragma once

#include <iostream>

typedef uintptr_t ui_t;
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

namespace fan {

	// converts enum to int
	template <typename Enumeration>
	constexpr auto eti(Enumeration const value)
	-> typename std::underlying_type<Enumeration>::type
	{
		return static_cast<
			typename std::underlying_type<Enumeration>::type
		>(value);
	}

	template <typename ...Args>
	constexpr void print(const Args&... args) {
		((std::cout << args << " "), ...) << '\n';
	}

	enum class platform_t { windows, linux };

	#if defined(_WIN32) || defined(_WIN64)
		constexpr platform_t platform = platform_t::windows;
		#define FAN_PLATFORM_WINDOWS

	#elif defined(__linux__)
		constexpr platform_t platform = platform_t::windows;
		#define FAN_PLATFORM_LINUX
	#endif
}

//template <platform_t T_platform>
//concept platform_windows = T_platform == platform_t::windows;

//template <platform_t T_platform>
//concept platform_linux = T_platform == platform_t::linux;

#include <fan/vector.h>
#include <fan/da.h>