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
import fan.types.compile_time_string;

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

  template<typename T>
  std::string format_to_string(const T& v) {
    std::ostringstream oss;
    oss << v;
    return oss.str();
  }
  std::string format_to_string(const std::string_view& v) { return std::string(v); }
  std::string format_to_string(const char* v) { return v ? std::string(v) : std::string(); }
  template<std::size_t N>
  std::string format_to_string(const fan::ct_string<N>& v) { return std::string(v.ptr, v.len); }
  std::string format_to_string(const fan::str_view_t& v) { return std::string(v); }

  template<typename... Args>
  std::string format(std::string_view fmt, Args&&... args) {
    std::array<std::string, sizeof...(Args)> parts {format_to_string(std::forward<Args>(args))...};
    std::string out;
    out.reserve(fmt.size() + 32);
    std::size_t pos = 0;
    std::size_t idx = 0;
    while (true) {
      auto p = fmt.find("{}", pos);
      if (p == std::string_view::npos) {
        out.append(fmt.substr(pos));
        break;
      }
      out.append(fmt.substr(pos, p - pos));
      if (idx < parts.size()) out.append(parts[idx++]);
      pos = p + 2;
    }
    return out;
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