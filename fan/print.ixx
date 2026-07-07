module;

#include <fan/utility.h>

export module fan.print;

import std;

import fan.types;
import fan.utility;
import fan.types.color;
import fan.time;
import fan.print.error;
import fan.formatter;

export namespace fan {
  template <typename T>
  struct is_bitset : std::false_type {};
  template <std::size_t N>
  struct is_bitset<std::bitset<N>> : std::true_type {};

  template <typename T>
  concept has_subscript_and_size =
    requires(const T& t) {
      t.size();
      t[0];
    }
    && fan::streamable<decltype(std::declval<const T&>()[0])>
    && (!std::is_same_v<T, std::string>)
    && (!std::is_same_v<T, std::string_view>)
    && (!std::is_base_of_v<std::string_view, std::remove_cvref_t<T>>)
    && (!is_bitset<std::remove_cvref_t<T>>::value);

  template <typename T>
  requires has_subscript_and_size<T>
  std::ostream& operator<<(std::ostream& os, const T& container) noexcept {
    for (std::uintptr_t i = 0; i < container.size(); i++) os << container[i] << ' ';
    return os;
  }

  template <typename T>
  requires requires(const T& t) { t.size.size();
  }
  std::ostream& operator<<(std::ostream& os, const T& container_within) noexcept {
    for (std::uintptr_t i = 0; i < container_within.size(); i++) {
      for (std::uintptr_t j = 0; j < container_within[i].size(); j++)
        os << container_within[i][j] << ' ';
      os << '\n';
    }
    return os;
  }

  namespace detail {
    std::string format_tabbed_impl(std::streamsize tab_width, const std::vector<std::string>& str_args, const std::vector<bool>& neg_args);
    std::wstring format_wargs_impl(const std::vector<std::wstring>& args, bool with_space);
    void print_impl(const std::string& msg);
    void printr_impl(const std::string& msg);
    void wprint_impl(const std::wstring& msg, bool newline);
    void print_color_impl(const fan::color& c, const std::string& msg);
    void print_throttled_impl(int throttle_ms, std::size_t hash_key, const std::string& msg);
    void print_once_impl(const std::string& msg);
    void print_stacktrace_impl();

    template<typename T> std::wstring to_wstr(const T& v) {
      if constexpr (std::is_same_v<std::decay_t<T>, std::wstring>) return v;
      else if constexpr (std::is_same_v<std::decay_t<T>, const wchar_t*>) return v ? v : L"";
      else if constexpr (std::is_same_v<std::decay_t<T>, wchar_t>) return std::wstring(1, v);
      else if constexpr (std::is_arithmetic_v<std::decay_t<T>>) return std::to_wstring(v);
      else {
         std::string s = fan::format_args(v);
         return std::wstring(s.begin(), s.end());
      }
    }
  }

  template <typename ...Args>
  std::string format_args_tabbed(std::streamsize tab_width, const Args&... args) {
    return detail::format_tabbed_impl(tab_width, {fan::format_args(args)...}, {fan::is_negative(args)...});
  }

  template <typename ...Args>
  std::wstring format_wargs_with_space(const Args&... args) {
    return detail::format_wargs_impl({detail::to_wstr(args)...}, true);
  }

  template <typename ...Args>
  std::wstring format_wargs(const Args&... args) {
    return detail::format_wargs_impl({detail::to_wstr(args)...}, false);
  }

  std::string format_warning(const std::string& message) {
  #ifndef fan_disable_warnings
    std::string ret = "fan warning:" + message;
    write_error_to_disk(ret);
    return ret;
  #else
    return "";
  #endif
  }

  std::string format_warning_no_space(const std::string& message) {
  #ifndef fan_disable_warnings
    std::string ret = "fan warning:" + message;
    write_error_to_disk(ret);
    return ret;
  #else
    return "";
  #endif
  }


  template <typename ...Args>
  void print_color_raw(const fan::color& c, const Args&... args) {
    detail::print_color_impl(c, format_args(args...));
  }

  template <typename ...Args>
  void print_color(const fan::color& c, const Args&... args) {
    print_color_raw(c, args..., '\n');
  }

  std::string paint(const fan::color& c, const auto&... args) {
    std::string msg = fan::format_args(args...);
    std::uint32_t rgba = c.get_rgba();
    std::uint8_t r = (rgba >> 24) & 0xFF, g = (rgba >> 16) & 0xFF, b = (rgba >> 8) & 0xFF;
    return "\033[38;2;" + std::to_string(r) + ";" + std::to_string(g) + ";" + std::to_string(b) + "m" + msg + "\033[0m";
  }
  std::string paint_ok(const auto&... args)   { return paint(fan::colors::green,  args...); }
  std::string paint_warn(const auto&... args) { return paint(fan::colors::yellow, args...); }
  std::string paint_err(const auto&... args)  { return paint(fan::colors::red,    args...); }

  template <typename ...Args> void print_impl(const Args&... args) { detail::print_impl(format_args(args...) + '\n'); }
  
  template <typename ...Args> 
  void print_dbg(const Args&... args) { 
    std::string msg = format_args(args...);
    push_memory_log(msg, log_level_e::info);
    #if FAN_DEBUG >= fan_debug_insane
      detail::print_impl(msg + '\n');
    #endif
  }
  template <typename ...Args>
  void print_dbg_tag(const std::string& tag, const Args&... args) {
    std::string msg = format_args(args...);
    push_memory_log(tag, msg, log_level_e::info);
  #if FAN_DEBUG >= fan_debug_insane
    detail::print_impl("[" + tag + "] " + msg + '\n');
  #endif
  }

  template <typename ...Args>
  void print_log(
    log_level_e level,
    std::string_view tag,
    const Args&... args
  ) {
    std::string msg = format_args(args...);

    if (tag.empty()) {
      push_memory_log(msg, level);
    }
    else {
      push_memory_log(std::string(tag), msg, level);
    }

  #if FAN_DEBUG >= fan_debug_insane
    if (tag.empty()) {
      detail::print_impl(msg + '\n');
    }
    else {
      detail::print_impl("[" + std::string(tag) + "] " + msg + '\n');
    }
  #endif
  }

#if !defined(FAN_REFLECTION)
  template <typename ...Args> void print(const Args&... args) { fan::print_impl(args...); }
#endif


  template <typename ...Args> void printr(const Args&... args) { detail::printr_impl(format_args_raw(args...)); }
  template <typename ...Args> void printc(const Args&... args) { detail::print_impl(format_args_comma(args...) + '\n'); }
  template <typename ...Args> void printt(std::streamsize tab_width, const Args&... args) { detail::print_impl(format_args_tabbed(tab_width, args...) + '\n'); }
  template <typename ...Args> void printn8(const Args&... args) { detail::print_impl(format_args_n8(args...) + '\n'); }
  template <typename ...Args> void print_no_space(const Args&... args) { detail::print_impl(format_args_no_space(args...) + '\n'); }
  template <typename ...Args> void printnln(const Args&... args) { detail::printr_impl(format_args_with_space(args...)); }
  template <typename ...Args> void wprint_no_endline(const Args&... args) { detail::wprint_impl(format_wargs_with_space(args...), false); }
  template <typename ...Args> void wprint(const Args&... args) { detail::wprint_impl(format_wargs(args...), true); }

  template <typename ...Args>
  void printr_ok(const Args&... args) {
    std::string message = format_args(args...);
    write_error_to_disk(message);
    print_color_raw(fan::colors::green, message);
  }

  template <typename ...Args>
  void printr_success(const Args&... args) {
    printr_ok(args...);
  }

  template <typename ...Args>
  void printr_warning(const Args&... args) {
  #ifndef fan_disable_warnings
    std::string message = format_args(args...);
    write_error_to_disk(message);
    print_color_raw(fan::colors::yellow, message);
    if (message.size() && message.back() == '\n') message.pop_back();
    push_memory_log(message, log_level_e::warning);
  #endif
  }

  template <typename ...Args>
  void printr_error(const Args&... args) {
  #ifndef fan_disable_errors
    std::string message = format_args(args...);
    write_error_to_disk(message);
    print_color_raw(fan::colors::red, message);
    if (message.size() && message.back() == '\n') message.pop_back();
    push_memory_log(message, log_level_e::error);
  #endif
  }

  template <typename ...Args>
  void print_ok(const Args&... args) {
    printr_ok(args..., "\n");
  }

  template <typename ...Args>
  void print_success(const Args&... args) {
    print_ok(args...);
  }

  template <typename ...Args>
  void print_warning(const Args&... args) {
  #ifndef fan_disable_warnings
    printr_warning(args..., "\n");
  #endif
  }

  template <typename ...Args>
  void print_error(const Args&... args) {
  #ifndef fan_disable_errors
    printr_error(args..., "\n");
  #endif
  }

  void print_warning_no_space(const std::string& message) {
  #ifndef fan_disable_warnings
    write_error_to_disk(message);
    push_memory_log(message, log_level_e::warning);
    detail::print_impl(format_warning_no_space(message) + '\n');
  #endif
  }

  template <int throttle_ms = 1000, typename... Args>
  void print_throttled(const Args&... args) {
    std::string msg = format_args(args...);
    std::size_t key = std::hash<std::string>{}(msg);
    detail::print_throttled_impl(throttle_ms, key, msg);
  }

  template<typename... Args, FAN_UNIQUE_CALL>
  void print_every(int throttle_ms, const Args&... args) {
    if (fan::time::every<FAN_UNIQUE_CALL_PASS>(throttle_ms)) {
      detail::printr_impl(fan::format_args(args...) + '\n');
    }
  }

  void print_once(const auto&... args) {
    detail::print_once_impl(format_args(args...));
  }

  void print_results(const auto& result, const auto& expected) {
    if (result != expected) {
      std::string a = result;
      std::string b = expected;
      char failed_char_pos = 0;
      for (int i = 0; i < a.size() && i < b.size(); ++i) {
        if (a[i] != b[i]) {
          failed_char_pos = i;
          break;
        }
      }
      fan::print_error("FAIL");
      fan::print_error(result);
      fan::print_success(expected);
      for (int spaces = 0; spaces < failed_char_pos; ++spaces) fan::printr(" ");
      fan::print_error("^");
    }
    else {
      fan::print_success("SUCCESS");
      fan::print_success(result);
      fan::print_success(expected);
    }
  }

  void flush_console();
  void print_progress(std::size_t done, std::size_t total);

  namespace debug {
    void print_stacktrace() {
      detail::print_stacktrace_impl();
    }
  }
}