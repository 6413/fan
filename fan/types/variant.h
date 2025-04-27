#pragma once

#include <fan/types/types.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <tuple>
#include <string>
#include <typeinfo>

namespace fan {
  template <typename T>
  struct function_traits
    : public function_traits<decltype(&T::operator())> {};

  template <typename ClassType, typename ReturnType, typename... Args>
  struct function_traits<ReturnType(ClassType::*)(Args...) const> {
    enum { arity = sizeof...(Args) };

    typedef ReturnType result_type;

    template <size_t i>
    struct arg {
      typedef typename std::tuple_element<i, std::tuple<Args...>>::type type;
    };
    using tupl = std::tuple<Args...>;
  };
#define __FAN__ADDSEMICOLON(x, idx) x

  template <class T, class Tuple>
  struct index_t {
    static const std::size_t value = -1;
  };

  template <class T, class... Types>
  struct index_t<T, std::tuple<T, Types...>> {
    static const std::size_t value = 0;
  };

  template <class T, class U, class... Types>
  struct index_t<T, std::tuple<U, Types...>> {
    static const std::size_t value = 1 + index_t<T, std::tuple<Types...>>::value;
  };
}

#define fan_variant_functions(...) \
 inline static auto name##__##lambdaaa = [](__VA_ARGS__){ }; \
  using tupl = std::functionraits<decltype(name##__##lambdaaa)>::tupl; \
  static constexpr std::size_t len = std::tuple_size_v<tupl>; \
  \
  union { \
    __FAN__FOREACH(__FAN__ADDSEMICOLON, __VA_ARGS__); \
  };\
  uint32_t active = 0; \
  template<size_t ...Is> \
  constexpr void internal_get_runtime_value(std::index_sequence<Is...>, size_t i, const auto& lambda) { \
    ((void)(Is == i && (lambda(*(std::tuple_element_t<Is, tupl>*)this), true)), ...); \
  } \
  template<size_t ...Is> \
  constexpr void internal_get_runtime_value_idx(std::index_sequence<Is...>, size_t i, const auto& lambda) { \
    ((void)(Is == i && (lambda<Is>(*(std::tuple_element_t<Is, tupl>*)this), true)), ...); \
  } \
  \
  constexpr void get_value(size_t idx, const auto& lambda) \
  { \
    internal_get_runtime_value(std::make_index_sequence<len>{}, idx, lambda); \
  } \
  constexpr void get_value_idx(size_t idx, const auto& lambda) \
  { \
    internal_get_runtime_value_idx(std::make_index_sequence<len>{}, idx, lambda); \
  } \
  \
  constexpr void get_value(const auto& lambda) \
  { \
    get_value(active, lambda); \
  } \
  template <typename T> \
  void operator=(const T& v) { \
    active = fan::index_t<T, tupl>::value; \
    if (active == -1) { \
      std::size_t idx = 0; \
      std::apply([&v, &idx, this]<typename... T2>(T2&&... args) { \
        (([&v, &idx, this](auto& args){ \
          if constexpr(std::is_convertible_v<decltype(args), T>) { \
            active = idx; \
          } \
          ++idx; \
        }(args)), ...); \
      }, tupl{}); \
    }\
    if (active == -1) { \
    } \
    get_value(active, [&v, this]<typename T2>(T2& v2) { \
      if constexpr (std::is_same_v<T, T2> || std::is_convertible_v<T, T2>) { \
        new (&v2) T2(); \
        v2 = v; \
      } \
    });\
  }

// cant have same type twice
#define fan_variant(name, ...) \
  struct name { \
    name(){} \
    ~name(){} \
    fan_variant_functions(__VA_ARGS__) \
  }

//unnamed variant
#define fan_uvariant(...) \
  struct { \
    fan_variant_functions(__VA_ARGS__) \
  }