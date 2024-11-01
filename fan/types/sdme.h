#pragma once

// sdme_build_struct_arithmetic_type -- used for when sdme_create_struct type needs to be arithmetic type aka non struct

#include <concepts>

template<typename...> struct type_list {};
template<class L, typename T> struct type_list_append;
template<typename... Ts, typename T> struct type_list_append<type_list<Ts...>, T> { using type = type_list<Ts..., T>; };

template<unsigned long long I, typename L> struct type_at;
template<typename H, typename... T> struct type_at<0, type_list<H, T...>> { using type = H; };
template<unsigned long long I, typename H, typename... T> struct type_at<I, type_list<H, T...>> { using type = typename type_at<I - 1, type_list<T...>>::type; };

template<typename L> struct sum_sizeof;
template<typename... Ts> struct sum_sizeof<type_list<Ts...>> { static constexpr auto value = (sizeof(Ts) + ...); };
template<> struct sum_sizeof<type_list<>> { static constexpr auto value = 0; };

template<typename L> struct type_list_size;
template<typename... Ts> struct type_list_size<type_list<Ts...>> { static constexpr auto value = sizeof...(Ts); };

template<unsigned N, typename L> struct state_t {
  static constexpr unsigned n = N;
  using list = L;
};

namespace { struct tu_tag {}; }

template<unsigned N, std::same_as<tu_tag> T> struct reader {
  friend auto state_func(reader<N, T>);
};

template<unsigned N, typename L, std::same_as<tu_tag> T> struct setter {
  friend auto state_func(reader<N, T>) { return L{}; }
  static constexpr state_t<N, L> state{};
};

template struct setter<0, type_list<>, tu_tag>;

template<std::same_as<tu_tag> T, auto E, unsigned N = 0>
[[nodiscard]] consteval auto get_state() {
  if constexpr (requires(reader<N, T> r) { state_func(r); })
    return get_state<T, E, N + 1>();
  else
    return state_t < N - 1, decltype(state_func(reader<N - 1, T>{})) > {};
}

template<std::same_as<tu_tag> T = tu_tag, auto E = [] {}, auto S = get_state<T, E>() >
using get_list = typename std::remove_cvref_t<decltype(S)>::list;

template<typename Type, std::same_as<tu_tag> T, auto E>
[[nodiscard]] consteval auto append_impl() {
  using cur_state = decltype(get_state<T, E>());
  using new_list = typename type_list_append<typename cur_state::list, Type>::type;
  return setter<cur_state::n + 1, new_list, T>{}.state;
}

template<typename Type, std::same_as<tu_tag> T = tu_tag, auto E = [] {}, auto S = append_impl<Type, T, E>() >
constexpr auto append = [] { return S; };

template<typename L, unsigned long long N>
struct sum_sizes {
  static constexpr auto value = sizeof(typename type_at<N - 1, L>::type) + sum_sizes<L, N - 1>::value;
};
template<typename L> struct sum_sizes<L, 0> { static constexpr auto value = 0; };

#pragma pack(push, 1)

template<typename M> struct getPointerType {
  template<typename C, typename T> static T get_type(T C::* v);
  using type = decltype(get_type(static_cast<M>(nullptr)));
};

template <typename T>
struct any_type_wrap_t {
  operator T& () {
    return v;
  }
  operator T() const {
    return v;
  }
  void operator=(const auto& nv) {
    v = nv;
  }
  T v;
};

template<typename T, int member_index, int offsetof_to_this>
struct wrap_t : T {
  // struct member index
  static constexpr int I = member_index;
  // gives offsetof to current pointer, 
  // counting previous struct members
  static constexpr int otf = offsetof_to_this;
  operator T& () { return *this; }
};

template<typename T, int offsetof_to_this>
struct wrap_of_wrap_t : 
#ifdef sdme_build_struct_arithmetic_type
  any_type_wrap_t<T>
#else
  T 
#endif
{
  // gives offsetof to current pointer, 
// counting previous struct members
  static constexpr auto otf = offsetof_to_this;
  operator T& () {
    return *this;
  }
};

#define __sdme2(type, var) wrap_t<wrap_of_wrap_t<type, sum_sizeof<get_list<>>::value - beg_off>, \
    __COUNTER__ - beg - 1, []{ append<wrap_of_wrap_t<type, sum_sizeof<get_list<>>::value - beg_off>>(); \
    return sum_sizeof<get_list<>>::value - beg_off; }()> var

/*for runtime index access*/
template<unsigned long long... Indices>
struct index_sequence {
  static constexpr unsigned long long size() { return sizeof...(Indices); }
};

template<unsigned long long N, unsigned long long... Indices>
struct make_index_sequence_helper : make_index_sequence_helper<N - 1, N - 1, Indices...> {};

template<unsigned long long... Indices>
struct make_index_sequence_helper<0, Indices...> {
  using type = index_sequence<Indices...>;
};

template<unsigned long long N>
using make_index_sequence = typename make_index_sequence_helper<N>::type;

template<typename T> struct inherit_t {

  template<auto c>
    requires std::is_member_pointer_v<decltype(c)>
  constexpr auto AN() const {
    return getPointerType<decltype(c)>::type::otf;
  }

  template<int i>
  constexpr int AN() const {
    return type_at<i + T::beg_len, decltype(T::get())>::type::otf;
  }

  template <int i>
  auto* NA() const {
    return (typename type_at<i + T::beg_len, decltype(T::get())>::type*)(((unsigned char*)this) + AN<i>());
  }

  // number to address with constexpr
  template <int i>
  constexpr void* NAC() const {
    return (((unsigned char*)this) + AN<i>());
  }

  //template <auto c>
  //requires std::is_member_pointer_v<decltype(c)>
  //constexpr auto* NA() const {
  //  return (typename type_at<decltype(T::*c)::I + T::beg_len, decltype(T::get())>::type*)(((unsigned char*)this) + AN<c>());
  //}

  // internal get runtime value
  template<unsigned long long ...Is>
  constexpr void igrv(index_sequence<Is...>, unsigned long long i, const auto& lambda) const {
    ((void)(Is == i && (lambda(*NA<Is>()), true)), ...);
  }

  constexpr void get_value(unsigned long long idx, const auto& lambda) const {
    igrv(make_index_sequence<size()>{}, idx, lambda);
  }

  static constexpr auto get_sizeof_this_trick();

  //template<auto c> 
  //requires std::is_member_pointer_v<decltype(c)>
  //constexpr decltype(auto) get_n() const { return this->*c; }

  static constexpr unsigned long long size() { return T::member_count; }
  static constexpr auto GetMemberAmount() { return T::member_count; }
};

// static constexpr cheat - using enum instead of static constexpr auto
#define sdme_internal_begin__ \
    enum {\
    beg = __COUNTER__, \
    beg_len = type_list_size<get_list<>>::value, \
    beg_off = sum_sizes<get_list<>, beg_len>::value \
    }; \
    static constexpr auto get() { return get_list<>(); }

#define sdme_internal_end__ \
      enum { \
        member_count = type_list_size<get_list<>>::value - beg_len \
      }; \
    };

#define sdme_internal_end_manual__ \
      enum { \
        member_count = type_list_size<get_list<>>::value - beg_len \
      }; \

#define sdme_create_struct(name) struct name : inherit_t<name> { sdme_internal_begin__

#undef sdme_build_struct_arithmetic_type