module;

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
import fan.types.color;
import fan.types.vector;
import fan.utility;

export namespace fan {
  std::string format_tabbed_from_string(std::streamsize tab_width, const std::string& formatted) {
    std::istringstream iss(formatted);
    std::vector<std::vector<std::string>> all_rows;
    std::vector<size_t> max_widths;
    std::string line;

    while (std::getline(iss, line)) {
      if (line.empty()) continue;

      std::istringstream line_stream(line);
      std::vector<std::string> row;
      std::string token;

      while (line_stream >> token) {
        row.push_back(token);
      }

      if (!row.empty()) {
        all_rows.push_back(row);

        for (size_t i = 0; i < row.size(); ++i) {
          if (max_widths.size() <= i) {
            max_widths.resize(i + 1, 0);
          }
          max_widths[i] = std::max(max_widths[i], row[i].length());
        }
      }
    }

    for (auto& width : max_widths) {
      width = std::max(width, static_cast<size_t>(tab_width));
    }

    std::ostringstream result;
    for (const auto& row : all_rows) {
      for (size_t i = 0; i < row.size(); ++i) {
        if (i > 0) result << " ";

        if (i < row.size() - 1) {
          result << std::setw(max_widths[i]) << std::left << row[i];
        }
        else {
          result << row[i];
        }
      }
      result << "\n";
    }

    return result.str();
  }

  // Custom format implementation for fan types
  template <typename... T>
  constexpr std::string format(std::string_view fmt_str, T&&... args) {
    auto convert_arg = [](auto&& arg) -> auto {
      if constexpr (requires { arg.to_string(); }) {
        return arg.to_string();
      } else {
        return arg;
      }
    };
    
    auto converted_args = std::make_tuple(convert_arg(args)...);
    
    return std::apply([fmt_str](auto&&... converted) {
      return current_fmt::vformat(fmt_str, current_fmt::make_format_args(converted...));
    }, converted_args);
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