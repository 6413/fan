#pragma once

namespace fan {

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

    static constexpr std::size_t count = sizeof...(Rest);

    using value_type = T;

    value_type x_;

    using base = masterpiece_reversed_t<Rest...>;

  protected:
    template <uint32_t N, typename... Ts>
    struct get;

    template <uint32_t N, typename T, typename... Ts>
    struct get<N, fan::masterpiece_reversed_t<T, Ts...>>
    {
      using type = typename get<N + 1, fan::masterpiece_reversed_t<Ts...>>::type;
    };

    template <typename T, typename... Ts>
    struct get<count, fan::masterpiece_reversed_t<T, Ts...>>
    {
      using type = T;
    };
  public:

    template <int N>
    using get_type = get <N, masterpiece_reversed_t<T, Rest...>>;

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

    template <uint32_t depth = 0>
    constexpr void iterate(auto lambda) {
      if constexpr(depth > count) {
        return;
      }
      else {
        lambda(std::integral_constant<uint32_t, depth>{}, get_value<depth>());
        iterate<depth + 1>(lambda);
      }
    }
  };

  template <typename T>
  struct masterpiece_reversed_t<T> {
    T x_;

    static constexpr std::size_t count = 1;

    using value_type = T;

    template <uint32_t i>
    constexpr auto get_value() {
      return &x_;
    }

    constexpr void iterate(auto lambda) {
      lambda(std::integral_constant<uint32_t, 0>{});
    }
  };

  template <typename... args>
  using masterpiece_t = typename reverse_impl<masterpiece_reversed_t<args...>>::type;
}