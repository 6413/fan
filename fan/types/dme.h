#pragma once

#pragma pack(push, 1)

template <typename T, unsigned long long idx, const char* str, typename dt_t = void>
struct wrapper_t : T {
  using dt = dt_t;
  static constexpr unsigned long long I = idx;
  constexpr operator unsigned long long() const { return I; }
  operator T& () { return *this; }
  static constexpr unsigned long long AN() { return I; }
  const char* sn = str;
};

template<typename T>
using return_type_of_t = decltype((*(T*)nullptr)());

inline char dme_empty_string__[] = "";

#define __dme2(varname) static constexpr inline char internalstr_##varname[] = #varname;\
  wrapper_t<dme_type_t, __COUNTER__ - DME_INTERNAL__BEG - 1,  internalstr_##varname> varname
#define __dme2_data(varname, data) private: \
 static inline constexpr char internalstr_##varname[] = #varname;\
 struct internal_##varname {data}; public: wrapper_t<dme_type_t, __COUNTER__ - DME_INTERNAL__BEG - 1,  internalstr_##varname, internal_##varname> varname


template <typename main_t, typename T, unsigned long long index>
struct base_t {
  using dme_type_t = T;
  constexpr auto* NA(unsigned long long I) const { return &((wrapper_t<dme_type_t, 0, dme_empty_string__> *)this)[I]; }
  static constexpr unsigned long long GetMemberAmount() { return sizeof(main_t) / sizeof(dme_type_t); }
  static constexpr auto DME_INTERNAL__BEG = index;
};

#define dme_inherit__(main_t, type) base_t<main_t, type, __COUNTER__>
#pragma pack(pop)