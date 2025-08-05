#pragma once

#if !defined(__dme)

#include <cstdint>
#include <type_traits>


#pragma pack(push, 1)
  #ifndef __empty_struct
    struct __empty_struct{};
    #define __empty_struct __empty_struct
  #endif


  template<typename T, const char* str, uintptr_t dt_size>
  struct __dme_t : T {
    __dme_t& operator=(const T v) requires (!std::is_same_v<T, __empty_struct>) {
      *dynamic_cast<T*>(this) = v;
      return *this;
    }
    operator const char* () const {
      return sn;
    }
    const char *sn = str;
    uint32_t m_DSS = dt_size;
  };

  template<typename T>
  using return_type_of_t = decltype((*(T*)nullptr)());

  inline char __dme_empty_string[] = "";

  #define __dme(varname, ...) \
    struct varname##structed_dt{ \
      __VA_ARGS__ \
    }; \
    struct varname##_t : varname##structed_dt{ \
      constexpr operator uintptr_t() const { return __COUNTER__ - DME_INTERNAL__BEG - 1; } \
      static inline constexpr uintptr_t dss = (sizeof(#__VA_ARGS__) > 1) * sizeof(varname##structed_dt); \
    }; \
    inline static struct varname##_t varname; \
    static inline constexpr char varname##_str[] = #varname; \
    __dme_t<value_type, varname##_str, varname##_t::dss> varname##_ram


  template <typename main_t, uintptr_t index, typename T = __empty_struct>
  struct __dme_inherit_t{
    constexpr static main_t& items() { static main_t m; return m; }
    using value_type = T;
    using dme_type_t = __dme_t<value_type, __dme_empty_string, 0>;
    constexpr auto* NA(uintptr_t I) { return &((dme_type_t *)this)[I]; }
    static constexpr uintptr_t GetMemberAmount() { return sizeof(main_t) / sizeof(dme_type_t); }
    static constexpr uintptr_t size() { return GetMemberAmount(); }
    static constexpr auto DME_INTERNAL__BEG = index;
    static constexpr const char* name(uintptr_t i) { return items().NA(i)->sn; }
    // these are read only since constexpr static main_t
    static constexpr auto* begin() { return items().NA(0); }
    static constexpr auto* end() { return items().NA(size()); }
  };

  #define __dme_inherit(main_t, ...) __dme_inherit_t<main_t, __COUNTER__, ##__VA_ARGS__>
  #define __dme_get(dme_var, enum_value) dme_var.enum_value##_ram

#pragma pack(pop)
#endif