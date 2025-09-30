module;

#include <fan/utility.h>

#include <type_traits>
#include <coroutine>

export module fan.utility;

export import fan.time;

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

// think better place for these
namespace fan {

  template<bool B, typename T, typename F>
  struct conditional { using type = T; };

  template<typename T, typename F>
  struct conditional<false, T, F> { using type = F; };

  template<bool B, typename T, typename F>
  using conditional_t = typename conditional<B, T, F>::type;

  template<typename T>
  struct is_const { static constexpr bool value = false; };

  template<typename T>
  struct is_const<const T> { static constexpr bool value = true; };

  template<typename T>
  constexpr bool is_const_v = is_const<T>::value;

  template<typename T1, typename T2>
  struct pair {
    T1 first;
    T2 second;
  };

  template<typename It>
  struct iterator_traits {
    using reference = decltype(*(declval<It>()));
  };

  template <typename T>
  T&& declval() noexcept;

  namespace fan_detail {
    template<typename It>
    struct iterator_traits {
      using reference = decltype(*(declval<It>()));
    };

    template <typename Container>
    auto get_begin(Container& container) -> decltype(container.begin()) {
      return container.begin();
    }

    template <typename Container>
    auto get_end(Container& container) -> decltype(container.end()) {
      return container.end();
    }

    template <typename Container>
    auto get_size(Container& container) -> decltype(container.size()) {
      return container.size();
    }
  }

  template <typename It>
  class enumerate_iterator {
    It _iter;
    std::size_t _index;
  public:
    using value_type = pair<std::size_t, typename fan_detail::iterator_traits<It>::reference>;
    using reference = value_type;
    using pointer = void;

    enumerate_iterator(It iter, std::size_t index) : _iter(iter), _index(index) {}

    reference operator*() const {
      return { _index, *_iter };
    }

    enumerate_iterator& operator++() {
      ++_iter;
      ++_index;
      return *this;
    }

    bool operator!=(const enumerate_iterator& other) const {
      return _iter != other._iter;
    }
  };

  template <typename Container>
  class enumerate_view {
    Container& _container;
  public:
    using iterator = enumerate_iterator<
      conditional_t<
      is_const_v<Container>,
      decltype(fan_detail::get_begin(declval<const Container&>())),
      decltype(fan_detail::get_begin(declval<Container&>()))
      >
    >;

    enumerate_view(Container& container) : _container(container) {}
    iterator begin() {
      return { fan_detail::get_begin(_container), 0 };
    }
    iterator end() {
      return { fan_detail::get_end(_container), fan_detail::get_size(_container) };
    }
  };

  struct enumerate_fn {
    template <typename Container>
    auto operator()(Container& container) const {
      return enumerate_view<Container>{container};
    }
  };

  template <typename Container>
  auto operator|(Container& container, const fan::enumerate_fn& view) {
    return view(container);
  }

  export constexpr enumerate_fn enumerate{};

  export {
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

    /*template<typename Callable>
    using return_type_of_t = typename decltype(std::function{ std::declval<Callable>() })::result_type;*/

    constexpr std::uint64_t get_hash(const char* str) {
      std::uint64_t result = 0xcbf29ce484222325; // FNV offset basis

      std::uint32_t i = 0;

      if (str == nullptr) {
        return 0;
      }

      while (str[i] != 0) {
        result ^= (std::uint64_t)str[i];
        result *= 1099511628211; // FNV prime
        i++;
      }

      return result;
    }

    template <bool _Test, uintptr_t _Ty1, uintptr_t _Ty2>
    struct conditional_value {
      static constexpr auto value = _Ty1;
    };

    template <uintptr_t _Ty1, uintptr_t _Ty2>
    struct conditional_value<false, _Ty1, _Ty2> {
      static constexpr auto value = _Ty2;
    };

    template <bool _Test, uintptr_t _Ty1, uintptr_t _Ty2>
    struct conditional_value_t {
      static constexpr auto value = conditional_value<_Test, _Ty1, _Ty2>::value;
    };

    constexpr auto uninitialized = -1;

    template<class T, typename U>
    std::int64_t member_offset(U T::* member)
    {
      return reinterpret_cast<std::int64_t>(
        &(reinterpret_cast<T const volatile*>(0)->*member)
        );
    }
    #ifndef __throw_error_impl
    #define __throw_error_impl throw_error_impl

    struct exception_t {
      const char* reason;
    };

    inline void throw_error_impl(const char* reason = "") {
#ifdef fan_compiler_msvc
      //system("pause");
#endif
#if __cpp_exceptions
      throw exception_t{ .reason = reason };
#endif
    }
    #else
    using fan::throw_error_impl;
    #endif


    template <typename T>
    constexpr bool is_negative(const T& value) {
      if constexpr (std::is_arithmetic_v<T>) {
        return value < T{ 0 };
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

    template<typename T>
    using component_type_t = std::conditional_t<
      requires { typename T::value_type; },
      typename T::value_type,
      T
    >;
  }
}