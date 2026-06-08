module;

#include <ctime>

module fan.print.error;

import std;

namespace fan {
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

  void throw_error_impl(const char* reason) {
    std::string res(reason);
    if (res.size()) write_error_to_disk(res);
#if __cpp_exceptions
    throw exception_t{.reason = reason};
#endif
  }
}