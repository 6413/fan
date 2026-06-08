module;

#if defined(fan_std23)
#endif

module fan.print;

import std;

import fan.time;

namespace fan::detail {
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
    std::cout << msg;
  }

  void printr_impl(const std::string& msg) {
    std::cout << msg;
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
    std::cout << "\033[38;2;" << (int)r << ";" << (int)g << ";" << (int)b << "m" << msg << "\033[0m";
  }

  void print_throttled_impl(int throttle_ms, std::size_t hash_key, const std::string& msg) {
    static std::unordered_map<std::size_t, fan::time::timer> timers;
    auto& t = timers[hash_key];
    if (!t.started()) { 
      std::cout << msg << '\n'; 
      t.start_millis(throttle_ms); 
      return; 
    }
    if (t.finished()) { 
      std::cout << msg << '\n'; 
      t.start_millis(throttle_ms); 
    }
  }

  void print_once_impl(const std::string& msg) {
    static std::unordered_map<std::string, int> count_map;
    if (++count_map[msg] == 1) {
      std::cout << msg << '\n';
    }
  }

  void print_stacktrace_impl() {
  #if defined(fan_std23)
    std::cout << std::to_string(std::stacktrace::current()) << '\n';
  #elif defined(fan_platform_unix)
    // Fallback if needed
  #else
    std::cout << "stacktrace not supported\n";
  #endif
  }
}