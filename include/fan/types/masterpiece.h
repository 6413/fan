#pragma once

#include <cstdint>
#include <type_traits>
#include <variant>

namespace fan {
  #pragma pack(push, 1)
  // reversing masterpiece
  template<typename>
  struct templated_base_case;

  template <template<typename...> class T, typename... TArgs>
  struct templated_base_case<T<TArgs...>>
  {
    using type = T<>;
  };

  template<
    typename T,
    typename = typename templated_base_case<T>::type>
  struct reverse_impl;

  template<
    template <typename...> class T,
    typename... TArgs>
  struct reverse_impl<
    typename templated_base_case<T<TArgs...>>::type,
    T<TArgs...>>
  {
    using type = T<TArgs...>;
  };

  template<
    template<typename...> class T,
    typename x,
    typename... xs,
    typename... done>
  struct reverse_impl<
    T<x, xs...>,
    T<done...>>
  {
    using type = typename reverse_impl<T<xs...>, T<x, done...>>::type;
  };

  template <typename... args>
  struct masterpiece_reversed_t;

  template <>
  struct masterpiece_reversed_t<> {

  };

  template <typename T, typename ...Rest>
  struct masterpiece_reversed_t<T, Rest...> : masterpiece_reversed_t<Rest...> {

    static constexpr uint32_t count = sizeof...(Rest);

    using value_type = T;

    value_type x_;

    using base = masterpiece_reversed_t<Rest...>;

  protected:
    template <uint32_t N, typename... Ts>
    struct get;

    template <uint32_t N, typename T2, typename... Ts>
    struct get<N, fan::masterpiece_reversed_t<T2, Ts...>>
    {
      using type = typename get<N + 1, fan::masterpiece_reversed_t<Ts...>>::type;
    };

    template <typename T2, typename... Ts>
    struct get<count, fan::masterpiece_reversed_t<T2, Ts...>>
    {
      using type = T2;
    };

  public:

    template <int N>
    using get_type = get <N, masterpiece_reversed_t<T, Rest...>>;

    template <int N>
    using get_type_t = typename get_type<N>::type;

    template <typename _T2, typename _Ty = masterpiece_reversed_t<T, Rest...>, uint32_t depth = count>
    constexpr auto get_value(_Ty* a = nullptr) {
      constexpr uint32_t i = get_index_with_type<_T2>();
      if constexpr (depth == count) {
        a = this;
      }
      if constexpr (i == depth) {
        return &a->x_;
      }
      if constexpr (depth > i) {
        return get_value<i, typename _Ty::base, depth - 1>(a);
      }
    }

    template <uint32_t i, typename _Ty = masterpiece_reversed_t<T, Rest...>, uint32_t depth = count>
    constexpr auto get_value(_Ty* a = nullptr) {
      if constexpr (depth == count) {
        a = this;
      }
      if constexpr (i == depth) {
        return &a->x_;
      }
      if constexpr (depth > i) {
        return get_value<i, typename _Ty::base, depth - 1>(a);
      }
    }

    //template <uint32_t i, typename _Ty = masterpiece_reversed_t<T, Rest...>, uint32_t depth = count>
    //constexpr auto get_value(_Ty* a = nullptr) const {
    //  return get_value<i, _Ty>(a);
    //}

    T* get_single() {
      return &x_;
    }

    template<size_t ...Is>
    constexpr void internal_get_runtime_value(std::index_sequence<Is...>, size_t i, const auto& lambda) {
      ((void)(Is == i && (lambda(get_value<Is>()), true)), ...);
    }
    
    constexpr void get_value(size_t idx, const auto& lambda)
    {
      internal_get_runtime_value(std::make_index_sequence<size()>{}, idx, lambda);
    }

    template <typename get_type, typename _Ty = masterpiece_reversed_t<T, Rest...>, uint32_t depth = count>
    static constexpr uint32_t get_index_with_type() {
      if constexpr (std::is_same<get_type, typename _Ty::value_type>::value) {
        return depth;
      }
      else if constexpr (depth > 0) {
        return get_index_with_type<get_type, typename _Ty::base, depth - 1>();
      }
      else {
      //  static_assert("failed to find index");
      }
      return -1;
    }
    template <uint32_t depth = 0>
    constexpr int iterate_ret(auto lambda){
      if constexpr(depth > count) {
        return depth;
      }
      else {
        if (lambda(std::integral_constant<uint32_t, depth>{}, get_value<depth>())) {
          return depth;
        }
        return iterate_ret<depth + 1>(lambda);
      }
      return depth;
    }
    template <uint32_t depth = 0>
    constexpr void iterate(auto lambda) {
      if constexpr(depth > count) {
        return;
      }
      else {
        lambda(std::integral_constant<uint32_t, depth>{}, get_value<depth>());
        return iterate<depth + 1>(lambda);
      }
    }
    template <uint32_t depth = count>
    constexpr int reverse_iterate_ret(auto lambda) {
      if constexpr (depth == (uint32_t)-1) {
        return depth;
      }
      else {
        if (lambda(std::integral_constant<uint32_t, depth>{}, get_value<depth>())) {
          return depth;
        }
        return reverse_iterate_ret<depth - 1>(lambda);
      }
      return depth;
    }
    template <uint32_t depth = count>
    constexpr void reverse_iterate(auto lambda) {
      if constexpr (depth == (uint32_t)-1) {
        return;
      }
      else {
        lambda(std::integral_constant<uint32_t, depth>{}, get_value<depth>());
        return reverse_iterate<depth - 1>(lambda);
      }
    }
    static constexpr size_t size() {
      return count + 1;
    }
  };

  template <typename T>
  struct masterpiece_reversed_t<T> {
    T x_;

    static constexpr uint32_t count = 1;

    using value_type = T;

    template <uint32_t i>
    constexpr auto get_value() {
      return &x_;
    }

    template <typename T2>
    constexpr T* get_value() {
      static_assert(std::is_same_v<T, T2>);
      return &x_;
    }

    static constexpr size_t size() {
      return count;
    }

    template<size_t ...Is>
    constexpr void internal_get_runtime_value(std::index_sequence<Is...>, size_t i, const auto& lambda) {
      ((void)(Is == i && (lambda(get_value<Is>()), true)), ...);
    }

    constexpr void get_value(size_t idx, const auto& lambda)
    {
      internal_get_runtime_value(std::make_index_sequence<size()>{}, idx, lambda);
    }


  protected:
    template <uint32_t N, typename... Ts>
    struct get;

    template <uint32_t N, typename T2, typename... Ts>
    struct get<N, fan::masterpiece_reversed_t<T2, Ts...>>
    {
      using type = typename get<N + 1, fan::masterpiece_reversed_t<Ts...>>::type;
    };

    template <typename T2, typename... Ts>
    struct get<count, fan::masterpiece_reversed_t<T2, Ts...>>
    {
      using type = T2;
    };

  public:

    template <int N>
    using get_type = masterpiece_reversed_t<T>;

    template <int N>
    using get_type_t = T;

    constexpr void iterate(auto lambda) {
      lambda(std::integral_constant<uint32_t, 0>{}, get_value<0>());
    }

    template <uint32_t depth = 0>
    constexpr int iterate_ret(auto lambda){
      if (lambda(std::integral_constant<uint32_t, depth>{}, get_value<depth>())) {
        return depth;
      }
      return count;
    }
    template <uint32_t depth = count>
    constexpr int reverse_iterate_ret(auto lambda) {
      if constexpr (depth == (uint32_t)-1) {
        return depth;
      }
      else {
        if (lambda(std::integral_constant<uint32_t, depth>{}, get_value<depth>())) {
          return depth;
        }
        return reverse_iterate_ret<depth - 1>(lambda);
      }
      return depth;
    }
    template <uint32_t depth = count>
    constexpr void reverse_iterate(auto lambda) {
      if constexpr (depth == (uint32_t)-1) {
        return;
      }
      else {
        lambda(std::integral_constant<uint32_t, depth>{}, get_value<depth>());
        return reverse_iterate<depth - 1>(lambda);
      }
    }
  };

  template <typename... args>
  using masterpiece_t = typename reverse_impl<masterpiece_reversed_t<args...>>::type;

  #pragma pack(pop)
}

#define fan_EVAL(...) __VA_ARGS__
#define fan_EAT(...)
#define fan_EMPTY
#define fan_STR2(x) STRINGIFY(x)
#define fan_STR(x) fan_STR2(x)

#define fan_MAP01(f, x) f(x)
#define fan_MAP02(f, x, ...) f(x) fan_EVAL(fan_MAP01(f, __VA_ARGS__))
#define fan_MAP03(f, x, ...) f(x) fan_EVAL(fan_MAP02(f, __VA_ARGS__))
#define fan_MAP04(f, x, ...) f(x) fan_EVAL(fan_MAP03(f, __VA_ARGS__))
#define fan_MAP05(f, x, ...) f(x) fan_EVAL(fan_MAP04(f, __VA_ARGS__))
#define fan_MAP06(f, x, ...) f(x) fan_EVAL(fan_MAP05(f, __VA_ARGS__))
#define fan_MAP07(f, x, ...) f(x) fan_EVAL(fan_MAP06(f, __VA_ARGS__))
#define fan_MAP08(f, x, ...) f(x) fan_EVAL(fan_MAP07(f, __VA_ARGS__))
#define fan_MAP09(f, x, ...) f(x) fan_EVAL(fan_MAP08(f, __VA_ARGS__))
#define fan_MAP10(f, x, ...) f(x) fan_EVAL(fan_MAP09(f, __VA_ARGS__))
#define fan_MAP11(f, x, ...) f(x) fan_EVAL(fan_MAP10(f, __VA_ARGS__))
#define fan_MAP12(f, x, ...) f(x) fan_EVAL(fan_MAP11(f, __VA_ARGS__))
#define fan_MAP13(f, x, ...) f(x) fan_EVAL(fan_MAP12(f, __VA_ARGS__))
#define fan_MAP14(f, x, ...) f(x) fan_EVAL(fan_MAP13(f, __VA_ARGS__))
#define fan_MAP15(f, x, ...) f(x) fan_EVAL(fan_MAP14(f, __VA_ARGS__))
#define fan_MAP16(f, x, ...) f(x) fan_EVAL(fan_MAP15(f, __VA_ARGS__))
#define fan_MAP17(f, x, ...) f(x) fan_EVAL(fan_MAP16(f, __VA_ARGS__))
#define fan_MAP18(f, x, ...) f(x) fan_EVAL(fan_MAP17(f, __VA_ARGS__))
#define fan_MAP19(f, x, ...) f(x) fan_EVAL(fan_MAP18(f, __VA_ARGS__))
#define fan_MAP20(f, x, ...) f(x) fan_EVAL(fan_MAP19(f, __VA_ARGS__))
#define fan_MAP21(f, x, ...) f(x) fan_EVAL(fan_MAP20(f, __VA_ARGS__))
#define fan_MAP22(f, x, ...) f(x) fan_EVAL(fan_MAP21(f, __VA_ARGS__))
#define fan_MAP23(f, x, ...) f(x) fan_EVAL(fan_MAP22(f, __VA_ARGS__))
#define fan_MAP24(f, x, ...) f(x) fan_EVAL(fan_MAP23(f, __VA_ARGS__))
#define fan_MAP25(f, x, ...) f(x) fan_EVAL(fan_MAP24(f, __VA_ARGS__))
#define fan_MAP26(f, x, ...) f(x) fan_EVAL(fan_MAP25(f, __VA_ARGS__))
#define fan_MAP27(f, x, ...) f(x) fan_EVAL(fan_MAP26(f, __VA_ARGS__))
#define fan_MAP28(f, x, ...) f(x) fan_EVAL(fan_MAP27(f, __VA_ARGS__))
#define fan_MAP29(f, x, ...) f(x) fan_EVAL(fan_MAP28(f, __VA_ARGS__))
#define fan_MAP30(f, x, ...) f(x) fan_EVAL(fan_MAP29(f, __VA_ARGS__))
#define fan_MAP31(f, x, ...) f(x) fan_EVAL(fan_MAP30(f, __VA_ARGS__))
#define fan_MAP32(f, x, ...) f(x) fan_EVAL(fan_MAP31(f, __VA_ARGS__))
#define fan_MAP33(f, x, ...) f(x) fan_EVAL(fan_MAP32(f, __VA_ARGS__))
#define fan_MAP34(f, x, ...) f(x) fan_EVAL(fan_MAP33(f, __VA_ARGS__))
#define fan_MAP35(f, x, ...) f(x) fan_EVAL(fan_MAP34(f, __VA_ARGS__))
#define fan_MAP36(f, x, ...) f(x) fan_EVAL(fan_MAP35(f, __VA_ARGS__))
#define fan_MAP37(f, x, ...) f(x) fan_EVAL(fan_MAP36(f, __VA_ARGS__))
#define fan_MAP38(f, x, ...) f(x) fan_EVAL(fan_MAP37(f, __VA_ARGS__))
#define fan_MAP39(f, x, ...) f(x) fan_EVAL(fan_MAP38(f, __VA_ARGS__))
#define fan_MAP40(f, x, ...) f(x) fan_EVAL(fan_MAP39(f, __VA_ARGS__))
#define fan_MAP41(f, x, ...) f(x) fan_EVAL(fan_MAP40(f, __VA_ARGS__))
#define fan_MAP42(f, x, ...) f(x) fan_EVAL(fan_MAP41(f, __VA_ARGS__))
#define fan_MAP43(f, x, ...) f(x) fan_EVAL(fan_MAP42(f, __VA_ARGS__))
#define fan_MAP44(f, x, ...) f(x) fan_EVAL(fan_MAP43(f, __VA_ARGS__))
#define fan_MAP45(f, x, ...) f(x) fan_EVAL(fan_MAP44(f, __VA_ARGS__))
#define fan_MAP46(f, x, ...) f(x) fan_EVAL(fan_MAP45(f, __VA_ARGS__))
#define fan_MAP47(f, x, ...) f(x) fan_EVAL(fan_MAP46(f, __VA_ARGS__))
#define fan_MAP48(f, x, ...) f(x) fan_EVAL(fan_MAP47(f, __VA_ARGS__))
#define fan_MAP49(f, x, ...) f(x) fan_EVAL(fan_MAP48(f, __VA_ARGS__))
#define fan_MAP50(f, x, ...) f(x) fan_EVAL(fan_MAP49(f, __VA_ARGS__))
#define fan_MAP51(f, x, ...) f(x) fan_EVAL(fan_MAP50(f, __VA_ARGS__))
#define fan_MAP52(f, x, ...) f(x) fan_EVAL(fan_MAP51(f, __VA_ARGS__))
#define fan_MAP53(f, x, ...) f(x) fan_EVAL(fan_MAP52(f, __VA_ARGS__))
#define fan_MAP54(f, x, ...) f(x) fan_EVAL(fan_MAP53(f, __VA_ARGS__))
#define fan_MAP55(f, x, ...) f(x) fan_EVAL(fan_MAP54(f, __VA_ARGS__))
#define fan_MAP56(f, x, ...) f(x) fan_EVAL(fan_MAP55(f, __VA_ARGS__))
#define fan_MAP57(f, x, ...) f(x) fan_EVAL(fan_MAP56(f, __VA_ARGS__))
#define fan_MAP58(f, x, ...) f(x) fan_EVAL(fan_MAP57(f, __VA_ARGS__))
#define fan_MAP59(f, x, ...) f(x) fan_EVAL(fan_MAP58(f, __VA_ARGS__))
#define fan_MAP60(f, x, ...) f(x) fan_EVAL(fan_MAP59(f, __VA_ARGS__))
#define fan_MAP61(f, x, ...) f(x) fan_EVAL(fan_MAP60(f, __VA_ARGS__))
#define fan_MAP62(f, x, ...) f(x) fan_EVAL(fan_MAP61(f, __VA_ARGS__))
#define fan_MAP63(f, x, ...) f(x) fan_EVAL(fan_MAP62(f, __VA_ARGS__))
#define fan_MAP64(f, x, ...) f(x) fan_EVAL(fan_MAP63(f, __VA_ARGS__))

#define fan_GET_NTH_ARG( \
  _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, \
  _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, \
  _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, \
  _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, \
  _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, \
  _51, _52, _53, _54, _55, _56, _57, _58, _59, _60, \
  _61, _62, _63, _64, N, ...) N

#define fan_MAP(f, ...) fan_EVAL(fan_EVAL(fan_GET_NTH_ARG(__VA_ARGS__, \
  fan_MAP64, fan_MAP63, fan_MAP62, fan_MAP61, \
  fan_MAP60, fan_MAP59, fan_MAP58, fan_MAP57, fan_MAP56, \
  fan_MAP55, fan_MAP54, fan_MAP53, fan_MAP52, fan_MAP51, \
  fan_MAP50, fan_MAP49, fan_MAP48, fan_MAP47, fan_MAP46, \
  fan_MAP45, fan_MAP44, fan_MAP43, fan_MAP42, fan_MAP41, \
  fan_MAP40, fan_MAP39, fan_MAP38, fan_MAP37, fan_MAP36, \
  fan_MAP35, fan_MAP34, fan_MAP33, fan_MAP32, fan_MAP31, \
  fan_MAP30, fan_MAP29, fan_MAP28, fan_MAP27, fan_MAP26, \
  fan_MAP25, fan_MAP24, fan_MAP23, fan_MAP22, fan_MAP21, \
  fan_MAP20, fan_MAP19, fan_MAP18, fan_MAP17, fan_MAP16, \
  fan_MAP15, fan_MAP14, fan_MAP13, fan_MAP12, fan_MAP11, \
  fan_MAP10, fan_MAP09, fan_MAP08, fan_MAP07, fan_MAP06, \
  fan_MAP05, fan_MAP04, fan_MAP03, fan_MAP02, fan_MAP01 \
  ))(f, __VA_ARGS__))

#define fan_REFLECTION_FIELD_NAME(x) fan_STR(fan_EVAL(x))

#define fan_REFLECTION_ALL(x) fan_EVAL x
#define fan_REFLECTION_SECOND(x) fan_EAT x
#define fan_REFLECTION_FIELD(x) fan_REFLECTION_ALL(x);
#define fan_REFLECTION_METHOD2(x) lambda(x/*, fan_REFLECTION_FIELD_NAME(x), std::integral_constant<uint64_t, fan::get_hash(fan_REFLECTION_FIELD_NAME(x))>{}*/);
#define fan_REFLECTION_METHOD(x) fan_REFLECTION_METHOD2(fan_REFLECTION_SECOND(x))

// making iterate constexpr fails to link in clang
#define fan_REFLECTION_VISTOR_METHOD(...) \
  inline void iterate_masterpiece(const auto& lambda) { \
    fan_MAP(fan_REFLECTION_METHOD, __VA_ARGS__) \
  } \
  inline void iterate_masterpiece(const auto& lambda) const { \
    \
      fan_MAP(fan_REFLECTION_METHOD, __VA_ARGS__) \
  }


#define fan_masterpiece_make(...) \
  fan_MAP(fan_REFLECTION_FIELD, __VA_ARGS__) \
  fan_REFLECTION_VISTOR_METHOD(__VA_ARGS__)
  //_PP_REFLECTION_VISTOR_METHOD(const, __VA_ARGS__)


//#define OBJECT_NAME_METHOD(obj) \
//  static constexpr const char* object_name() { \
//    return #obj; \
//  }

namespace fan {
  template <typename T>
  struct mp_t : T {
    using type_t = T;

    constexpr auto get_tuple_refless() {
      T t;
      return fan::make_struct_tuple<T>(t);
    }

    constexpr auto get_tuple() {
      return fan::make_struct_tuple_ref(*(T*)this);
    }
    constexpr auto get_tuple() const {
      T t;
      return fan::make_struct_tuple<T>(t);
    }

    template <std::size_t n>
    constexpr auto& get() {
      return std::get<n>(get_tuple());
    }

    constexpr type_t& get() {
      return *(T*)this;
    }

    constexpr operator type_t() {
      return *(T*)this;
    }

    constexpr operator auto() {
      return get_tuple();
    }

    // []<auto i>(auto& v){}
    constexpr void iterate(auto l) {
      fan::iterate_struct(*(T*)this, l);
    }

    constexpr auto begin() {
      return &std::get<0>(get_tuple());
    }
    constexpr auto end() {
      constexpr std::size_t n = std::tuple_size_v<decltype(get_tuple())>;
      return begin() + sizeof(T) - sizeof(decltype(std::get<n - 1>(get_tuple())));
    }

    constexpr std::size_t size() const {
      using T2 = decltype(get_tuple());
      return std::tuple_size_v<T2>;
    }

    template<size_t ...Is>
    constexpr void internal_get_runtime_value(std::index_sequence<Is...>, size_t i, const auto& lambda) {
      ((void)(Is == i && (lambda(get<Is>()), true)), ...);
    }

    constexpr void get_value(size_t idx, const auto& lambda)
    {
      constexpr std::size_t n = std::tuple_size_v<decltype(get_tuple())>;
      internal_get_runtime_value(std::make_index_sequence<n>{}, idx, lambda);
    }

    friend std::ostream& operator<<(std::ostream& os, const mp_t<T>& m) {
      os << fan::struct_to_string(*(T*)(&m));
      return os;
    }

  };

  template <typename T>
  static constexpr std::size_t get_biggest_sizeof() {
    constexpr auto max_stack_size = 0xffff;
    static_assert(sizeof(T) <= max_stack_size, "too big struct for stack");
    std::size_t max_sizeof = 0;
    fan::iterate_struct<T>(T{}, [&max_sizeof]<auto i0, typename T2>(T2&) {
      if (max_sizeof < sizeof(T2)) {
        max_sizeof = sizeof(T2);
      }
    });
    return max_sizeof;
  }

  //template <typename in_type_t>
  //struct union_mp {
  //  uint8_t m_data[get_biggest_sizeof<in_type_t>()];
  //  uint32_t m_current = -1;

  //  constexpr auto get_tuple() {
  //    return fan::make_struct_tuple_ref(*(in_type_t*)this);
  //  }

  //  template <std::size_t n>
  //  constexpr auto& get() {
  //    return std::get<n>(get_tuple());
  //  }

  //  template <typename T>
  //  constexpr T& get() {
  //    bool exit = false;
  //    in_type_t it{};
  //    fan::iterate_struct(it, [this, &exit]<auto i0, typename T0>(T0&) {
  //      if constexpr (std::is_same_v<T, T0>) {
  //        exit = true;
  //        if (m_current != i0) {
  //          new (m_data) T();
  //        }
  //        m_current = i0;
  //      }
  //    });
  //    return *reinterpret_cast<T*>(m_data);
  //  }
  //  template<size_t ...Is>
  //  constexpr void internal_get_runtime_value(std::index_sequence<Is...>, size_t i, const auto& lambda) {
  //    ((void)(Is == i && (lambda(get<Is>()), true)), ...);
  //  }

  //  constexpr void get_value(size_t idx, const auto& lambda)
  //  {
  //    constexpr std::size_t n = std::tuple_size_v<decltype(get_tuple())>;
  //    internal_get_runtime_value(std::make_index_sequence<n>{}, idx, lambda);
  //  }
  //  constexpr void current(const auto& lambda) {
  //    get_value(m_current, lambda);
  //  }
  //};

  template<typename... Ts>
  std::variant<Ts...> make_variant(const std::tuple<Ts...>& tup) {
    std::variant<Ts...> var;
    return var;
  }

  template <typename T>
  auto make_union_mp() {
    fan::mp_t<T>t;
    return make_variant(t.get_tuple_refless());
  }

  template <typename T>
  using union_mp = fan::return_type_of_t<decltype(make_union_mp<T>)>;

}