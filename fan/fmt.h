#pragma once

#include <fan/types/fstring.h>

//std::format doesnt exist for clang in linux
#define FMT_HEADER_ONLY
#include <fmt/format.h>
#include <fmt/xchar.h>

#include <vector>

namespace fan {

  std::vector<std::string> split(std::string str, std::string token);
  std::vector<std::string> split_quoted(const std::string& input);

  template <typename... T>
  static FMT_INLINE auto format(fmt::format_string<T...> fmt, T&&... args)
    -> fan::string {
    return fmt::vformat(fmt, fmt::make_format_args(args...));
  }

  template <typename... args_t>
  constexpr static auto print_format(fmt::format_string<args_t...> fmt, args_t&&... args) {
    fan::print(fmt::format(fmt, std::forward<args_t>(args)...));
  }

  template <typename... args_t>
  constexpr static auto printn8_format(fmt::format_string<args_t...> fmt, args_t&&... args) {
    fan::printn8(fmt::format(fmt, std::forward<args_t>(args)...));
  }

  template <typename... args_t>
  constexpr static auto throw_error_format(fmt::format_string<args_t...> fmt, args_t&&... args) {
    fan::print(fmt::format(fmt, std::forward<args_t>(args)...));
    throw_error_impl();
  }
}

template<typename T>
struct fmt::formatter<fan::vec2_wrap_t<T>> {
  auto parse(fmt::format_parse_context& ctx) { return ctx.end(); }
  auto format(const fan::vec2_wrap_t<T>& obj, fmt::format_context& ctx) { return fmt::format_to(ctx.out(), "{}", obj.to_string()); }
};
template<typename T>
struct fmt::formatter<fan::vec3_wrap_t<T>> {
  auto parse(fmt::format_parse_context& ctx) { return ctx.end(); }
  auto format(const fan::vec3_wrap_t<T>& obj, fmt::format_context& ctx) { return fmt::format_to(ctx.out(), "{}", obj.to_string()); }
};
template<typename T>
struct fmt::formatter<fan::vec4_wrap_t<T>> {
  auto parse(fmt::format_parse_context& ctx) { return ctx.end(); }
  auto format(const fan::vec4_wrap_t<T>& obj, fmt::format_context& ctx) { return fmt::format_to(ctx.out(), "{}", obj.to_string()); }
};