#pragma once

#pragma pack(push, 1)
template <typename T, unsigned long long idx, typename dt_t = void>
struct wrapper_t : T {
  using dt = dt_t;
  static constexpr unsigned long long I = idx;
  constexpr operator unsigned long long() const { return I; }
  operator T& () { return *this; }
  static constexpr unsigned long long AN() { return I; }
  const char* sn;
};

template<typename T>
using return_type_of_t = decltype((*(T*)nullptr)());

#define __dme2(varname, init) wrapper_t<dme_type_t, __COUNTER__ - DME_INTERNAL__BEG - 1> varname{init, #varname}
#define __dme2_data(varname, init, data) wrapper_t<dme_type_t, __COUNTER__ - DME_INTERNAL__BEG - 1,  \
  return_type_of_t<decltype([]{ struct { data }v; return v; })>> varname{init, #varname}

template <typename main_t, typename T, unsigned long long index>
struct base_t {
  using dme_type_t = T;
  constexpr wrapper_t<dme_type_t, 0>* NA(unsigned long long I) const { return &((wrapper_t<dme_type_t, 0> *)this)[I]; }
  static constexpr unsigned long long GetMemberAmount() { return sizeof(main_t) / sizeof(dme_type_t); }
  static constexpr auto DME_INTERNAL__BEG = index;
};

#define dme_inherit__(main_t, type) base_t<main_t, type, __COUNTER__>
#pragma pack(pop)