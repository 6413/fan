module;

#include <fan/types/types.h>
#include <tuple>
#include <ostream>
#include <string>
#include <functional>

// slow header to use

export module fan.types.magic;

import fan.fmt;
import fan.print;
import fan.types.fstring;

export namespace fan {

  template<typename ...Args>
constexpr std::size_t va_count(Args&&...) { return sizeof...(Args); }

template<size_t a, size_t b> struct assert_equality {
  static_assert(a == b, "Not equal");
  static constexpr bool result = (a == b);
};

template <size_t a, size_t b>
constexpr bool assert_equality_v = assert_equality<a, b>::result;

namespace impl {
  struct universal_type_t {
    template <typename T>
    operator T
    // known to be needed with msvc only - special check for clang needed because in windows clang its using msvc somehow
    #if defined(fan_compiler_msvc) && !defined(fan_compiler_clang)
      &
    #endif
    ();
  };

  template <typename T, typename... Args>
  consteval auto member_count() {
    //static_assert(std::is_aggregate_v<std::remove_cvref_t<T>>);

    if constexpr (requires {T{{Args{}}..., {universal_type_t{}}}; } == false) {
      return sizeof...(Args);
    }
    else {
      return member_count<T, Args..., universal_type_t>();
    }
  }
}

template <class T>
constexpr std::size_t count_struct_members() {
  return impl::member_count<T>();
}


 #define __FAN_REF_EACH(x) std::ref(x)
 #define __FAN_NREF_EACH(x) x
 #define GENERATE_CALL_F(count, ...) \
template <std::size_t _N, typename T> \
requires (count == _N) \
constexpr auto generate_variable_list_ref(T& struct_value) { \
    auto& [__VA_ARGS__] = struct_value; \
    return std::make_tuple(__FAN__FOREACH_NS(__FAN_REF_EACH, __VA_ARGS__)); \
}\
template <std::size_t _N, typename T> \
requires (count == _N) \
constexpr auto generate_variable_list_nref(const T& struct_value) { \
  \
    auto [__VA_ARGS__] = struct_value; \
    return std::make_tuple(__FAN__FOREACH_NS(__FAN_NREF_EACH, __VA_ARGS__)); \
}

  GENERATE_CALL_F(1, a)
  GENERATE_CALL_F(2, a, b)
  GENERATE_CALL_F(3, a, b, c)
  GENERATE_CALL_F(4, a, b, c, d)
  GENERATE_CALL_F(5, a, b, c, d, e)
  GENERATE_CALL_F(6, a, b, c, d, e, f)
  GENERATE_CALL_F(7, a, b, c, d, e, f, g)
  GENERATE_CALL_F(8, a, b, c, d, e, f, g, h)
  GENERATE_CALL_F(9, a, b, c, d, e, f, g, h, i)
  GENERATE_CALL_F(10, a, b, c, d, e, f, g, h, i, j)
  GENERATE_CALL_F(11, a, b, c, d, e, f, g, h, i, j, k)
  GENERATE_CALL_F(12, a, b, c, d, e, f, g, h, i, j, k, l)
  GENERATE_CALL_F(13, a, b, c, d, e, f, g, h, i, j, k, l, m)
  GENERATE_CALL_F(14, a, b, c, d, e, f, g, h, i, j, k, l, m, n)
  GENERATE_CALL_F(15, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o)
  GENERATE_CALL_F(16, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p)
  GENERATE_CALL_F(17, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q)
  GENERATE_CALL_F(18, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r)
  GENERATE_CALL_F(19, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s)
  GENERATE_CALL_F(20, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t)
  GENERATE_CALL_F(21, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u)
  GENERATE_CALL_F(22, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v)
  GENERATE_CALL_F(23, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w)
  GENERATE_CALL_F(24, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x)
  GENERATE_CALL_F(25, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y)
  GENERATE_CALL_F(26, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z)
  GENERATE_CALL_F(27, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, aa)
  GENERATE_CALL_F(28, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, aa, ab)
  GENERATE_CALL_F(29, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, aa, ab, ac)
  GENERATE_CALL_F(30, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, aa, ab, ac, ad)

  template <typename T>
  constexpr auto make_struct_tuple_ref(T& st) {
    static_assert(count_struct_members<T>() <= 30, "struct limited to 30");
    return generate_variable_list_ref<count_struct_members<T>()>(st);
  }

  template <typename T>
  constexpr auto make_struct_tuple(const T& st) {
    return generate_variable_list_nref<count_struct_members<T>()>(st);
  }

  template <typename T>
  constexpr auto make_struct_tuple_ref(T&& st) {
    T s;
    return generate_variable_list_ref<count_struct_members<T>()>(s);
  }
  template <typename T>
  constexpr auto make_struct_tuple_ref(const T& st) {
    T s;
    return generate_variable_list_ref<count_struct_members<T>()>(s);
  }
  

  template <typename T, typename F, std::size_t... I>
  void iterate_struct_impl(const T& st, F lambda, std::index_sequence<I...>) {
    auto tuple = make_struct_tuple(st);
    std::apply([&lambda](const auto&...args) {
      (lambda.template operator() < I > (std::forward<decltype(args)>(args)), ...);
      }, tuple);
  }

   template <typename T, typename F, std::size_t... I>
   constexpr void iterate_struct_impl(T& st, F lambda, std::index_sequence<I...>) {
     if constexpr (!std::is_empty_v<T>) {
       auto tuple = make_struct_tuple_ref(st);
       std::apply([&lambda](auto&...args) {
         (lambda.template operator() < I > (std::forward<decltype(args)>(args)), ...);
         }, tuple);
     }
   }

  template <typename T>
  constexpr void iterate_struct(const T& st, auto lambda) {
    iterate_struct_impl(st, lambda, std::make_index_sequence<count_struct_members<T>()>{});
  }

  template <typename T>
  constexpr void iterate_struct(T& st, auto lambda) {
    iterate_struct_impl(st, lambda, std::make_index_sequence<count_struct_members<T>()>{});
  }

  template <typename T>
  struct is_printable {
  private:
    template <typename U>
    static auto test(int) -> decltype(std::declval<std::ostream&>() << std::declval<U>(), std::true_type());

    template <typename>
    static auto test(...) -> std::false_type;

  public:
    static constexpr bool value = decltype(test<T>(0))::value;
  };

  template <typename T>
  constexpr bool is_printable_v = is_printable<T>::value;

  template <typename T>
  fan::string struct_to_string(T& st) {
    fan::string formatted_string = "{\n";
    iterate_struct(st, [&formatted_string, &st]<std::size_t i, typename T2>(T2& v) {
      // static_assert(is_printable_v<T2>, "struct member missing operator<< or not printable");
      if constexpr (!is_printable_v<T2>) {
        auto f = struct_to_string(v);
        std::string indented;
        std::istringstream f_stream(f);
        for (std::string line; std::getline(f_stream, line); ) {
          indented += "  " + line + '\n';
        }
        formatted_string += indented;
      }
      else {
        std::ostringstream os;
        os << v;
        formatted_string += fan::format("  Member index {}: {{\n    Type:{}, Value:{}\n  }}",
          i, typeid(T2).name(), os.str()
        );
        formatted_string += "\n";
        if constexpr (i + 1 != count_struct_members<T>()) {
          formatted_string += "\n";
        }
      }
    });
    formatted_string += "}";
    return formatted_string;
  }
  template <typename T>
  constexpr void print_struct(const T& st) {
    static_assert(count_struct_members<T>() <= 30, "limited to 30 members");
    fan::print(struct_to_string(st));
  }
}