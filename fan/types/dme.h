#pragma once

#include <cstdint>
#pragma pack(push, 1)

#ifndef __empty_struct
  struct __empty_struct{};
  #define __empty_struct __empty_struct
#endif

template<typename T, uintptr_t idx, const char* str, typename dt_t = void>
struct __dme_t : T {
  using dt = dt_t;
  static constexpr uintptr_t I = idx;
  constexpr operator uintptr_t() const { return I; }
  operator T& () { return *this; }
  static constexpr uintptr_t AN() { return I; }
  const char* sn = str;
};

template<typename T>
using return_type_of_t = decltype((*(T*)nullptr)());

inline char __dme_empty_string[] = "";

#define __dme(varname, data) \
  static inline constexpr char varname##_str[] = #varname; \
  struct varname##_t {data}; \
  __dme_t<dme_type_t, __COUNTER__ - DME_INTERNAL__BEG - 1,  varname##_str, varname##_t> varname

template <typename main_t, uintptr_t index, typename T = __empty_struct>
struct __dme_inherit_t{
  using dme_type_t = T;
  constexpr auto* NA(uintptr_t I) const { return &((__dme_t<dme_type_t, 0, __dme_empty_string> *)this)[I]; }
  static constexpr uintptr_t GetMemberAmount() { return sizeof(main_t) / sizeof(dme_type_t); }
  static constexpr auto DME_INTERNAL__BEG = index;
};

#define __dme_inherit(main_t, ...) __dme_inherit_t<main_t, __COUNTER__, ##__VA_ARGS__>

#pragma pack(pop)

#undef __empty_struct