module;

//std::format doesnt exist for clang in linux without libc++


#if __has_include("format")
  #include <format>
  namespace current_fmt = std;
#else
#define FMT_HEADER_ONLY
#include <fmt/format.h>
#include <fmt/xchar.h>
  namespace current_fmt = fmt;
#endif

#include <vector>
#include <sstream>
#include <string_view>
// std::quoted
#include <iomanip>
#include <iostream>

export module fan.fmt;

export import fan.print;
export import fan.types.fstring;
import fan.types.vector;
import fan.utility;



export namespace fan {

  template <typename... T>
  constexpr std::string format(const std::string_view& fmt, T&&... args) {
    return current_fmt::vformat(fmt, current_fmt::make_format_args(args...));
  }

  template <typename... args_t>
  constexpr auto printf(current_fmt::format_string<args_t...> fmt, args_t&&... args) {
    fan::print(current_fmt::format(fmt, std::forward<args_t>(args)...));
  }

  template <typename... args_t>
  constexpr auto printf_n8(current_fmt::format_string<args_t...> fmt, args_t&&... args) {
    fan::printn8(current_fmt::format(fmt, std::forward<args_t>(args)...));
  }

  template <typename... args_t>
  constexpr auto throw_errorf(current_fmt::format_string<args_t...> fmt, args_t&&... args) {
    fan::print(current_fmt::format(fmt, std::forward<args_t>(args)...));
    fan::throw_error();
  }

  template <typename... args_t>
  constexpr auto printf_throttled(current_fmt::format_string<args_t...> fmt, args_t&&... args) {
    std::string formatted = current_fmt::format(fmt, std::forward<args_t>(args)...);
    print_throttled(formatted, 1000);
  }

  template <typename... args_t>
  constexpr auto printf_throttled(int throttle_ms, current_fmt::format_string<args_t...> fmt, args_t&&... args) {
    std::string formatted = current_fmt::format(fmt, std::forward<args_t>(args)...);
    print_throttled(formatted, throttle_ms);
  }
  template <typename... args_t>
  constexpr void printfp(current_fmt::format_string<args_t...> fmt, args_t&&... args) {
    std::string formatted = current_fmt::format(fmt, std::forward<args_t>(args)...);
    std::streamsize width = formatted.length();
    
    std::ios init(nullptr);
    init.copyfmt(std::cout);
    std::cout << std::left << std::setw(width) << formatted << '\n';
    std::cout.copyfmt(init);
  }
}

template<typename T>
struct current_fmt::formatter<fan::vec2_wrap_t<T>> {
  auto parse(current_fmt::format_parse_context& ctx) { return ctx.end(); }
  auto format(const fan::vec2_wrap_t<T>& obj, current_fmt::format_context& ctx) { return current_fmt::format_to(ctx.out(), "{}", obj.to_string()); }
};
template<typename T>
struct current_fmt::formatter<fan::vec3_wrap_t<T>> {
  auto parse(current_fmt::format_parse_context& ctx) { return ctx.end(); }
  auto format(const fan::vec3_wrap_t<T>& obj, current_fmt::format_context& ctx) { return current_fmt::format_to(ctx.out(), "{}", obj.to_string()); }
};
template<typename T>
struct current_fmt::formatter<fan::vec4_wrap_t<T>> {
  auto parse(current_fmt::format_parse_context& ctx) { return ctx.end(); }
  auto format(const fan::vec4_wrap_t<T>& obj, current_fmt::format_context& ctx) { return current_fmt::format_to(ctx.out(), "{}", obj.to_string()); }
};