module;

#include <string>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <type_traits>

export module fan.print.error;

export namespace fan {
  struct log_t {
    std::string filename = "fan_errors.txt";
  };

  log_t& get_error_log() {
    static log_t log;
    return log;
  }

  void write_error_to_disk(const std::string& msg) {
    auto& log = get_error_log();
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    std::ofstream out(log.filename, std::ios::binary | std::ios::app);
    out << oss.str() << " - " << msg << '\n';
  }

  struct exception_t {
    const char* reason;
  };

  void throw_error_impl(const char* reason = "") {
    std::string res(reason);
    if (res.size()) write_error_to_disk(res);
#if __cpp_exceptions
    throw exception_t{.reason = reason};
#endif
  }

  template <typename ...Args>
  void throw_error(const Args&... args) {
    std::string msg;
    ((msg += [&]() -> std::string {
      if constexpr (std::is_same_v<std::decay_t<decltype(args)>, std::string>) return args;
      else if constexpr (std::is_same_v<std::decay_t<decltype(args)>, const char*>) return args ? args : "";
      else if constexpr (std::is_arithmetic_v<std::decay_t<decltype(args)>>) return std::to_string(args);
      else if constexpr (std::is_convertible_v<decltype(args), std::string>) return std::string(args);
      else return "?";
    }() + " "), ...);
    if (!msg.empty()) msg.pop_back();
    fan::throw_error_impl(msg.c_str());
  }
}

export std::string operator""_str(const char* str, std::size_t) {
  return std::string{str};
}