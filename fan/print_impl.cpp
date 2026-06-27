module;

#if __has_include(<print>)
  #define USE_STD_PRINT
#endif

#include <cstdio>

module fan.print;

import std;

import fan.time;

namespace fan::detail {
  static void print_raw(std::string_view msg) {
#if defined(USE_STD_PRINT)
    std::print("{}", msg);
#else
    std::cout << msg;
#endif
  }

  static void print_line_raw(std::string_view msg) {
    print_raw(msg);
    print_raw("\n");
  }

  std::string format_tabbed_impl(std::streamsize tab_width, const std::vector<std::string>& str_args, const std::vector<bool>& neg_args) {
    std::ostringstream oss;
    std::ios init(nullptr);
    init.copyfmt(oss);
    for (std::size_t i = 0; i < str_args.size(); ++i) {
      oss << (i == 0 ? "" : (neg_args[i] ? "" : " "))
          << std::setw(tab_width) << std::left << str_args[i]
          << (i == str_args.size() - 1 ? "" : " ")
          << (neg_args[i] ? " " : "");
    }
    oss.copyfmt(init);
    return oss.str();
  }

  std::wstring format_wargs_impl(const std::vector<std::wstring>& args, bool with_space) {
    std::wostringstream oss;
    for (std::size_t i = 0; i < args.size(); ++i) {
      oss << args[i];
      if (with_space || i < args.size() - 1) {
        oss << L" ";
      }
    }
    return oss.str();
  }

  void print_impl(const std::string& msg) {
    print_raw(msg);
  }

  void printr_impl(const std::string& msg) {
    print_raw(msg);
  }

  void wprint_impl(const std::wstring& msg, bool newline) {
    std::wcout << msg;
    if (newline) {
      std::wcout << L'\n';
    }
  }

  void print_color_impl(const fan::color& c, const std::string& msg) {
    std::uint32_t rgba = c.get_rgba();
    std::uint8_t r = (rgba >> 24) & 0xFF;
    std::uint8_t g = (rgba >> 16) & 0xFF;
    std::uint8_t b = (rgba >> 8) & 0xFF;
    print_raw(std::format("\033[38;2;{};{};{}m{}\033[0m", (int)r, (int)g, (int)b, msg));
  }

  void print_throttled_impl(int throttle_ms, std::size_t hash_key, const std::string& msg) {
    static std::unordered_map<std::size_t, fan::time::timer> timers;
    auto& t = timers[hash_key];
    if (!t.started()) { 
      print_line_raw(msg);
      t.start_millis(throttle_ms); 
      return; 
    }
    if (t.finished()) { 
      print_line_raw(msg);
      t.start_millis(throttle_ms); 
    }
  }

  void print_once_impl(const std::string& msg) {
    static std::unordered_map<std::string, int> count_map;
    if (++count_map[msg] == 1) {
      print_line_raw(msg);
    }
  }

  void print_stacktrace_impl() {
  #if defined(fan_std23)
    print_line_raw(std::to_string(std::stacktrace::current()));
  #elif defined(fan_platform_unix)
    // Fallback if needed
  #else
    print_line_raw("stacktrace not supported");
  #endif
  }
}

namespace fan {
  void fan::print_progress(std::size_t done, std::size_t total) {
    if (!total) { return; }
    f64_t pct = std::min<f64_t>(100.0, (f64_t(done) / total) * 100.0);
    char buf[41];
    int filled = static_cast<int>(pct / 2.5);
    std::memset(buf, '=', filled);
    std::memset(buf + filled, ' ', 40 - filled);
    buf[40] = '\0';
  #if defined(USE_STD_PRINT)
    std::print("\r[{}] {:>5.3f}%", buf, pct);
    std::fflush(stdout);  // flush the C stream, not cout
  #else
    std::cout << std::format("\r[{}] {:>5.3f}%", buf, pct) << std::flush;
  #endif
  }

  void fan::flush_console() {
  #if defined(USE_STD_PRINT)
    std::fflush(stdout);
  #else
    std::cout.flush();
  #endif
  }
}