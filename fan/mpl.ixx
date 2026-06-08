module;

#include <coroutine>

export module fan.mpl;

import std;

export namespace fan {
  template <typename T, typename = void>
  struct is_awaitable : std::false_type {};

  template <typename T>
  struct is_awaitable<T, std::void_t<
    decltype(std::declval<T>().await_ready()),
    decltype(std::declval<T>().await_suspend(std::declval<std::coroutine_handle<>>())),
    decltype(std::declval<T>().await_resume())
    >> : std::true_type {};

  template <typename T>
  constexpr bool is_awaitable_v = is_awaitable<T>::value;

  template<bool B, typename T, typename F>
  struct conditional {
    using type = T;
  };

  template<typename T, typename F>
  struct conditional<false, T, F> {
    using type = F;
  };

  template<bool B, typename T, typename F>
  using conditional_t = typename conditional<B, T, F>::type;

  template<typename T>
  struct is_const {
    static constexpr bool value = false;
  };

  template<typename T>
  struct is_const<const T> {
    static constexpr bool value = true;
  };

  template<typename T>
  constexpr bool is_const_v = is_const<T>::value;

  template<typename T1, typename T2>
  struct pair {
    T1 first;
    T2 second;
  };

  template <typename T>
  T&& declval() noexcept;

  template<typename T>
  using return_type_of_t = decltype((*(T*)nullptr)());

  template <typename Callable>
  struct return_type_of_membr;

  template <typename R, typename C, typename... Args>
  struct return_type_of_membr<R(C::*)(Args...)> {
    using type = R;
  };

  template <typename R, typename C, typename... Args>
  struct return_type_of_membr<R(C::*)(Args...) const> {
    using type = R;
  };

  template <typename Callable>
  using return_type_of_membr_t = typename return_type_of_membr<Callable>::type;

  template <bool _Test, std::uintptr_t _Ty1, std::uintptr_t _Ty2>
  struct conditional_value {
    static constexpr auto value = _Ty1;
  };

  template <std::uintptr_t _Ty1, std::uintptr_t _Ty2>
  struct conditional_value<false, _Ty1, _Ty2> {
    static constexpr auto value = _Ty2;
  };

  template <bool _Test, std::uintptr_t _Ty1, std::uintptr_t _Ty2>
  struct conditional_value_t {
    static constexpr auto value = conditional_value<_Test, _Ty1, _Ty2>::value;
  };

  constexpr auto uninitialized = -1;

  template<class T, typename U>
  std::int64_t member_offset(U T::* member) {
    return reinterpret_cast<std::int64_t>(
      &(reinterpret_cast<T const volatile*>(0)->*member)
    );
  }

  template<typename T>
  using component_type_t = std::conditional_t<
    requires { typename T::value_type; },
  typename T::value_type,
    T
  >;

  template <typename T>
  constexpr bool is_negative(const T& value) {
    if constexpr (std::is_arithmetic_v<T>) {
      return value < T {0};
    }
    return false;
  }

  template<typename T>
  constexpr int get_component_count() {
    if constexpr (requires { T::size(); }) {
      return T::size();
    }
    else {
      return 1;
    }
  }

  template <typename T, typename M> 
  M get_member_type(M T::*);

  template <typename T, typename M> 
  T get_class_type(M T::*);

  template <typename T, typename R, R T::*M> 
  std::size_t offset_of() { 
    return reinterpret_cast<std::size_t>(&(((T*)0)->*M)); 
  }

  template<typename... Ts>
  struct type_pack {};

  template<typename T>
  struct fn_traits : fn_traits<decltype(&T::operator())> {};

  template<typename C, typename R, typename... Args>
  struct fn_traits<R (C::*)(Args...) const> {
    template<template<typename...> typename Z, typename... X>
    static auto apply(X&&... x) {
      return Z<X..., Args...>::call(std::forward<X>(x)...);
    }
  };

  template<typename C, typename R, typename... Args>
  struct fn_traits<R (C::*)(Args...)> : fn_traits<R (C::*)(Args...) const> {};

  template<typename R, typename... Args>
  struct fn_traits<R(Args...)> {
    template<template<typename...> typename Z, typename... X>
    static auto apply(X&&... x) {
      return Z<X..., Args...>::call(std::forward<X>(x)...);
    }
  };

  template<typename R, typename... Args>
  struct fn_traits<R (*)(Args...)> : fn_traits<R(Args...)> {};

  template<typename T>
  struct lambda_traits : lambda_traits<decltype(&T::operator())> {};

  template<typename C, typename R, typename... Args>
  struct lambda_traits<R (C::*)(Args...) const> {
    using args = type_pack<Args...>;
  };

  template<typename C, typename R, typename... Args>
  struct lambda_traits<R (C::*)(Args...)> : lambda_traits<R (C::*)(Args...) const> {};

  template<typename R, typename... Args>
  struct lambda_traits<R(Args...)> {
    using args = type_pack<Args...>;
  };

  template<typename R, typename... Args>
  struct lambda_traits<R (*)(Args...)> : lambda_traits<R(Args...)> {};
}