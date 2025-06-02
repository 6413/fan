module;

#include <fan/types/types.h>

//std::format doesnt exist for clang in linux
//#define FMT_HEADER_ONLY
//#include <fmt/format.h>
//#include <fmt/xchar.h>

#include <format>

#include <vector>
#include <sstream>

// std::quoted
#include <iomanip>

export module fan.fmt;

export import fan.print;
export import fan.types.fstring;
import fan.types.vector;

namespace current_fmt = std;

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
  auto format(current_fmt::format_string<T...> fmt, T&&... args)
    -> std::string {
    return current_fmt::vformat(fmt, current_fmt::make_format_args(args...));
  }

  template <typename... args_t>
  constexpr auto print_format(current_fmt::format_string<args_t...> fmt, args_t&&... args) {
    fan::print(current_fmt::format(fmt, std::forward<args_t>(args)...));
  }

  template <typename... args_t>
  constexpr auto printn8_format(current_fmt::format_string<args_t...> fmt, args_t&&... args) {
    fan::printn8(current_fmt::format(fmt, std::forward<args_t>(args)...));
  }

  template <typename... args_t>
  constexpr auto throw_error_format(current_fmt::format_string<args_t...> fmt, args_t&&... args) {
    fan::print(current_fmt::format(fmt, std::forward<args_t>(args)...));
    throw_error_impl();
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