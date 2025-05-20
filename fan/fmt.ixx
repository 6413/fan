module;

#include <fan/types/types.h>

//std::format doesnt exist for clang in linux
#define FMT_HEADER_ONLY
#include <fmt/format.h>
#include <fmt/xchar.h>

#include <vector>
#include <sstream>

// std::quoted
#include <iomanip>

export module fan:fmt;

export import :print;
export import :types.fstring;
import :types.vector;

export namespace fan {

  std::vector<std::string> split(std::string str, std::string token) {
    std::vector<std::string>result;
    while (str.size()) {
      std::size_t index = str.find(token);
      if (index != std::string::npos) {
        result.push_back(str.substr(0, index));
        str = str.substr(index + token.size());
        if (str.size() == 0)result.push_back(str);
      }
      else {
        result.push_back(str);
        str = "";
      }
    }
    return result;
  }

  std::vector<std::string> split_quoted(const std::string& input) {
    std::vector<std::string> args;
    std::istringstream stream(input);
    std::string arg;

    while (stream >> std::quoted(arg)) {
      args.push_back(arg);
    }

    return args;
  }

  template <typename... T>
  FMT_INLINE auto format(fmt::format_string<T...> fmt, T&&... args)
    -> std::string {
    return fmt::vformat(fmt, fmt::make_format_args(args...));
  }

  template <typename... args_t>
  constexpr auto print_format(fmt::format_string<args_t...> fmt, args_t&&... args) {
    fan::print(fmt::format(fmt, std::forward<args_t>(args)...));
  }

  template <typename... args_t>
  constexpr auto printn8_format(fmt::format_string<args_t...> fmt, args_t&&... args) {
    fan::printn8(fmt::format(fmt, std::forward<args_t>(args)...));
  }

  template <typename... args_t>
  constexpr auto throw_error_format(fmt::format_string<args_t...> fmt, args_t&&... args) {
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