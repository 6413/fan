#pragma once

#ifndef __empty_struct
  #define __empty_struct __empty_struct
  struct __empty_struct {

  };
#endif

#ifndef __is_type_same
#define __is_type_same std::is_same_v
#endif

#ifndef __abort
  #define __abort() fan::throw_error_impl()
#endif

#ifndef __cta
  #define __cta(x) static_assert(x)
#endif

#ifndef __generic_malloc
	#define __generic_malloc(n) fan::memory_profile_malloc_cb(n)
#endif
#ifndef __generic_realloc
	#define __generic_realloc(ptr, n) fan::memory_profile_realloc_cb(ptr, n)
#endif
#ifndef __generic_free
	#define __generic_free(ptr) fan::memory_profile_free_cb(ptr)
#endif

#ifndef __unreachable_or
  #if defined(fan_compiler_msvc)
    #define __unreachable_or(...) __assume(0)
  #elif defined(fan_compiler_clang) || defined(fan_compiler_gcc)
    #define __unreachable_or(...) __unreachable()
  #else
    #error ?
  #endif
#endif

#ifndef __remove_reference_t 
  #define __remove_reference_t std::remove_reference_t
#endif


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

  
// std::unreachable

#ifndef __unreachable
  #if defined(fan_compiler_msvc)
    #define __unreachable() __assume(false)
  #elif defined(fan_compiler_clang) || defined(fan_compiler_gcc)
    #define __unreachable() __builtin_unreachable()
  #endif
#endif

#ifndef __return_type_of
	#define __return_type_of fan::return_type_of_t
#endif