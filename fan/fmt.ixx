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
  std::string format_tabbed_from_string(std::streamsize tab_width, const std::string& formatted) {
    std::istringstream iss(formatted);
    std::vector<std::string> columns;
    std::string token;
    while (iss >> token) {
      columns.push_back(token);
    }
    
    std::ostringstream oss;
    std::ios init(nullptr);
    init.copyfmt(oss);
    
    for (size_t i = 0; i < columns.size(); ++i) {
      bool is_neg = !columns[i].empty() && columns[i][0] == '-';
      if (i > 0) {
        oss << (is_neg ? "" : " ");
      }
      oss << std::setw(tab_width) << std::left << columns[i];
      if (i < columns.size() - 1) {
        oss << " ";
      }
    }
    
    oss.copyfmt(init);
    return oss.str();
  }

  template <typename... T>
  constexpr std::string format(std::string_view fmt, T&&... args) {
    return current_fmt::vformat(fmt, current_fmt::make_format_args(args...));
  }
  template <typename ...Args>
  std::string format_args_with_newline(const Args&... args) {
    return fan::format_args(args...) + "\n";
  }
  template <typename... args_t>
  std::string format_with_newline(std::string_view fmt, args_t&&... args) {
    return format(fmt, std::forward<args_t>(args)...) + "\n";
  }
  template <typename... args_t>
  std::string format_with_newline(current_fmt::format_string<args_t...> fmt, args_t&&... args) {
    return format(fmt, std::forward<args_t>(args)...) + "\n";
  }

  template <typename... args_t>
  constexpr auto printf(current_fmt::format_string<args_t...> fmt, args_t&&... args) {
    std::cout << format_with_newline(fmt, std::forward<args_t>(args)...);
  }
  template <typename... args_t>
  constexpr auto printf_n8(current_fmt::format_string<args_t...> fmt, args_t&&... args) {
    std::cout << format_args_n8(format(fmt, std::forward<args_t>(args)...)) << '\n';
  }
  template <typename... args_t>
  constexpr auto throw_errorf(current_fmt::format_string<args_t...> fmt, args_t&&... args) {
    std::cout << format_with_newline(fmt, std::forward<args_t>(args)...);
    fan::throw_error();
  }
  template <typename... args_t>
  constexpr auto printf_throttled(current_fmt::format_string<args_t...> fmt, args_t&&... args) {
    print_throttled(format(fmt, std::forward<args_t>(args)...), 1000);
  }
  template <typename... args_t>
  constexpr auto printf_throttled(int throttle_ms, current_fmt::format_string<args_t...> fmt, args_t&&... args) {
    print_throttled(format(fmt, std::forward<args_t>(args)...), throttle_ms);
  }
  template <typename... args_t>
  constexpr void printft(std::streamsize tab_width, current_fmt::format_string<args_t...> fmt, args_t&&... args) {
    std::cout << format_tabbed_from_string(tab_width, format(fmt, std::forward<args_t>(args)...)) << '\n';
  }

  template <typename... T> 
  using format_string = current_fmt::format_string<T...>;
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