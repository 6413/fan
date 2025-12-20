module;

#include <fan/utility.h>

#include <type_traits>
#include <coroutine>
#include <iterator>
#include <source_location>
#include <set>
#include <stacktrace>
#include <map>
#include <functional> // raii_nr_t
#include <cstring>

namespace raii_build {
  #include <fan/types/raii_nr.h>
}

namespace heap_profiler {
  #include <fan/memory/memory.h>
}

fan_track_allocations();

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

  template<typename It>
  struct iterator_traits {
    using reference = decltype(*(declval<It>()));
  };

  template <typename T>
  T&& declval() noexcept;

  export {
    template<typename container_t>
    struct bll_iterator_t {
      using node_t      = decltype(std::declval<container_t>().GetNodeFirst());
      using value_type  = decltype(std::declval<container_t>()[std::declval<node_t>()]);
      using reference   = value_type&;
      using index_type  = node_t;

      bll_iterator_t(container_t* c, node_t n) 
        : container(c), current(n) {}

      auto& operator*() const {
        if constexpr (requires(container_t* c, node_t n) { c->StartSafeNext(n); }) {
          container->StartSafeNext(current);
          safe_next_active = true;
        }
        return (*container)[current];
      }
      auto get_index() const {
        return current;
      }
      bll_iterator_t& operator++() {
        if constexpr (requires(container_t* c, node_t n) { c->StartSafeNext(n); }) {
          current = container->EndSafeNext();
          safe_next_active = false;
        }
        else if constexpr (requires(node_t n, container_t* c) { n.Next(c); }) {
          current = current.Next(container);
        }
        return *this;
      }

      bool operator!=(const bll_iterator_t& other) const {
        return current != other.current;
      }

      container_t* container;
      node_t       current;
      mutable bool safe_next_active = false;
    };
  }
  namespace fan_detail {
    template<typename T>
    struct iterator_traits {
      using reference = decltype(*(declval<T>()));
    };

    template<typename T>
    concept has_get_node_first = requires(T& t) {
      { t.GetNodeFirst() };
      { t.dst };
    };

    template<typename T>
    auto get_begin(T& container) {
      using base_t = std::remove_const_t<T>;
      if constexpr (has_get_node_first<base_t>) {
        return bll_iterator_t<base_t>{const_cast<base_t*>(&container), const_cast<base_t&>(container).GetNodeFirst()};
      }
      else {
        return container.begin();
      }
    }
    template<typename T>
    auto get_end(T& container) {
      using base_t = std::remove_const_t<T>;
      if constexpr (has_get_node_first<base_t>) {
        return bll_iterator_t<base_t>{const_cast<base_t*>(&container), const_cast<base_t&>(container).dst};
      }
      else {
        return container.end();
      }
    }
    template<typename T>
    auto get_first(T& container) {
      using base_t = std::remove_const_t<T>;
      if constexpr (has_get_node_first<base_t>) {
        return container.GetNodeFirst();
      }
      else {
        return typename T::size_type(0);
      }
    }
    template<typename T>
    auto get_size(T& container) {
      using base_t = std::remove_const_t<T>;
      if constexpr (has_get_node_first<base_t>) {
        return container.dst;
      }
      else {
        return container.size();
      }
    }
  }

  template <typename T>
  struct enumerate_iterator_t {
    template<typename U>
    static auto get_index_type_impl(int) -> typename U::index_type;
    template<typename U>
    static auto get_index_type_impl(...) -> std::size_t;

    using iter_index_t = decltype(get_index_type_impl<T>(0));
    using iter_reference = typename fan_detail::iterator_traits<T>::reference;
    using iter_value = std::remove_reference_t<iter_reference>;

    struct reference_proxy {
      iter_index_t index;
      iter_reference value;
      operator pair<iter_index_t, iter_value>() const {
        return {index, value};
      }
    };

    using value_type = pair<iter_index_t, iter_value>;
    using reference = reference_proxy;
    using pointer = void;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::forward_iterator_tag;

    enumerate_iterator_t(T iter, iter_index_t index) : _iter(iter), _index(index) {}
    reference operator*() const {
      if constexpr (requires { _iter.get_index(); }) {
        return {_iter.get_index(), *_iter};
      }
      else {
        return {_index, *_iter};
      }
    }
    enumerate_iterator_t& operator++() {
      ++_iter;
      if constexpr (!requires { _iter.get_index(); }) {
        ++_index;
      }
      return *this;
    }
    bool operator!=(const enumerate_iterator_t& other) const {
      return _iter != other._iter;
    }
    T _iter;
    iter_index_t _index;
  };

  template <typename container_t>
  struct enumerate_view_t {
    using iterator = enumerate_iterator_t<
      conditional_t<
      is_const_v<container_t>,
      decltype(fan_detail::get_begin(declval<const container_t&>())),
      decltype(fan_detail::get_begin(declval<container_t&>()))
      >
    >;

    enumerate_view_t(container_t& container) : _container(container) {}

    iterator begin() {
      return { fan_detail::get_begin(_container), fan_detail::get_first(_container) };
    }
    iterator end() {
      return { fan_detail::get_end(_container), fan_detail::get_size(_container) };
    }

    container_t& _container;
  };

  struct enumerate_fn {
    template <typename container_t>
    auto operator()(container_t& container) const {
      return enumerate_view_t<container_t>{container};
    }
  };

  template <typename container_t>
  auto operator|(container_t& container, const fan::enumerate_fn& view) {
    return view(container);
  }

  export {
    constexpr enumerate_fn enumerate{};

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
    std::int64_t member_offset(U T::* member) {
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


export namespace fan {
  using heap_profiler_t = heap_profiler::heap_profiler_t;
  void* memory_profile_malloc_cb(std::size_t n) {
    return heap_profiler_t::instance().allocate_memory(n);
  }
  void* memory_profile_realloc_cb(void* ptr, std::size_t n) {
    return heap_profiler_t::instance().reallocate_memory(ptr, n);
  }
  void memory_profile_free_cb(void* ptr) {
    heap_profiler_t::instance().deallocate_memory(ptr);
  }
}

export namespace fan{
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
  template<typename... Ts>
  struct type_pack {};
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

export namespace fan{
  template <
    typename nr_t,
    typename user_t,
    typename... params
  >
  using raii_nr_t = raii_build::raii_nr_t<nr_t, user_t, params...>;

  template<typename bll_t, typename user_fn_t, typename... param_t>
  auto add_bll_raii_impl(bll_t& bll, user_fn_t&& user_fn, fan::type_pack<param_t...>){
    using storage_t = std::remove_reference_t<bll_t>;
    using node_ref_t = typename storage_t::nr_t;
    using handle_t = raii_nr_t<node_ref_t, storage_t, param_t...>;
    using fn_t = typename handle_t::fn_t;
    using add_fn_t = typename handle_t::add_fn;
    using remove_fn_t = typename handle_t::remove_fn;
    struct callbacks_t{
      static node_ref_t add_impl(storage_t* s, fn_t cb){
        auto nr = s->NewNodeLast();
        (*s)[nr] = [cb](param_t... d){
          cb(nullptr, d...);
        };
        return nr;
      }
      static void remove_impl(storage_t* s, const node_ref_t& nr){
        if (s->NodeList.Current) {
          s->unlrec(nr);
        }
      }
      storage_t* s;
      fn_t cb;
    };
    return handle_t(
      &bll,
      add_fn_t(callbacks_t::add_impl),
      remove_fn_t(callbacks_t::remove_impl),
      [user_fn = std::forward<user_fn_t>(user_fn)](storage_t*, param_t... d){
        user_fn(d...);
      }
    );
  }
  template<typename bll_t, typename user_fn_t>
  auto add_bll_raii_cb(bll_t& bll, user_fn_t&& user_fn){
    using traits = fan::lambda_traits<std::remove_reference_t<user_fn_t>>;
    using args_pack = typename traits::args;
    return add_bll_raii_impl(bll, std::forward<user_fn_t>(user_fn), args_pack{});
  }

  template<typename owner_t, typename storage_t, typename node_ref_t, typename user_fn_t, typename... param_t>
  auto add_bll_raii_struct_impl(owner_t* owner, storage_t owner_t::*storage_member, user_fn_t&& user_fn, fan::type_pack<param_t...>){
    using handle_t = raii_nr_t<node_ref_t, owner_t, param_t...>;
    using fn_t = typename handle_t::fn_t;
    auto add = [storage_member](owner_t* o, fn_t cb){
      auto& s = o->*storage_member;
      auto nr = s.NewNodeLast();
      s[nr] = [cb](param_t... d){
        cb(nullptr, d...);
      };
      return nr;
    };
    auto remove = [storage_member](owner_t* o, const node_ref_t& nr){
      auto& s = o->*storage_member;
      if (s.NodeList.Current) {
        s.unlrec(nr);
      }
    };
    return handle_t(
      owner,
      std::move(add),
      std::move(remove),
      [user_fn = std::forward<user_fn_t>(user_fn)](owner_t*, param_t... d){
        user_fn(d...);
      }
    );
  }
  template<typename owner_t, typename storage_t, typename user_fn_t>
  auto add_bll_raii_struct_cb(owner_t* owner, storage_t owner_t::*storage_member, user_fn_t&& user_fn){
    using traits = fan::lambda_traits<std::remove_reference_t<user_fn_t>>;
    using args_pack = typename traits::args;
    return add_bll_raii_struct_impl<owner_t, storage_t, typename storage_t::nr_t>(
      owner,
      storage_member,
      std::forward<user_fn_t>(user_fn),
      args_pack{}
    );
  }
  template <typename T, typename M> 
  M get_member_type(M T::*);

  template <typename T, typename M> 
  T get_class_type(M T::*);

  template <typename T, typename R, R T::*M> 
  std::size_t offset_of() { 
    return reinterpret_cast<std::size_t>(&(((T*)0)->*M)); 
  }
}

//export {
//  template<typename T>
//  concept has_bll_methods = requires(T& t) {
//    { t.GetNodeFirst() };
//    { t.dst };
//  };
//
//  template<typename T>
//    requires has_bll_methods<T>
//  auto begin(T& container) {
//    return fan::bll_iterator_t<T>{&container, container.GetNodeFirst()};
//  }
//
//  template<typename T>
//    requires has_bll_methods<T>
//  auto end(T& container) {
//    return fan::bll_iterator_t<T>{&container, container.dst};
//  }
//}