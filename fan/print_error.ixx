module;

export module fan.print.error;

import std;

export namespace fan {
  struct log_t { std::string filename = "fan_errors.txt"; };
  log_t& get_error_log();
  void write_error_to_disk(const std::string& msg);
  struct exception_t { const char* reason; };
  void throw_error_impl(const char* reason = "");

  template <typename ...Args>
  void throw_error(const Args&... args) {
    std::string msg;
    ((msg += [&]() -> std::string {
      if constexpr (std::is_same_v<std::decay_t<decltype(args)>, std::string>) return args;
      else if constexpr (std::is_same_v<std::decay_t<decltype(args)>, const char*>) return args;
      else if constexpr (std::is_arithmetic_v<std::decay_t<decltype(args)>>) return std::to_string(args);
      else if constexpr (std::is_convertible_v<decltype(args), std::string>) return std::string(args);
      else return "?";
    }() + " "), ...);
    if (!msg.empty()) msg.pop_back();
    throw_error_impl(msg.c_str());
  }

  void assert(bool test) {
    if (!test) {
      fan::throw_error_impl("assert failed");
    }
  }
}

export std::string operator""_str(const char* str, std::size_t) {
  return std::string{str};
}