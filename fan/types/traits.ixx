module;

#include <type_traits>
#include <coroutine>

export module fan:types.traits;

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
}