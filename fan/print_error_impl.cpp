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

  void push_memory_log(const std::string& tag, const std::string& msg, log_level_e level) {
    auto& log = get_error_log();
    std::lock_guard<std::mutex> lock(log.mtx);
    log.buffer.push_back({tag, msg, level});
    if (log.buffer.size() > log.max_size) {
      log.buffer.pop_front();
    }
  }
  void push_memory_log(const std::string& msg, log_level_e level) {
    push_memory_log("", msg, level);
  }

  std::vector<log_entry_t> dump_memory_logs() {
    auto& log = get_error_log();
    std::lock_guard<std::mutex> lock(log.mtx);
    return std::vector<log_entry_t>(log.buffer.begin(), log.buffer.end());
  }

  std::vector<log_entry_t> dump_memory_logs_since(std::uint64_t& cursor) {
    auto& log = get_error_log();
    std::lock_guard<std::mutex> lock(log.mtx);
    
    std::vector<log_entry_t> result;
    if (cursor >= log.total_logs_pushed) return result;
    
    std::size_t available = log.total_logs_pushed - cursor;
    if (available > log.buffer.size()) {
      available = log.buffer.size();
    }
    
    auto it = log.buffer.end() - available;
    result.assign(it, log.buffer.end());
    
    cursor = log.total_logs_pushed;
    return result;
  }

  void clear_memory_logs() {
    auto& log = get_error_log();
    std::lock_guard<std::mutex> lock(log.mtx);
    log.buffer.clear();
  }

  void throw_error_impl(const char* reason) {
    std::string res(reason);
    if (res.size()) {
      write_error_to_disk(res);
      push_memory_log(res, log_level_e::error);
    }
#if __cpp_exceptions
    throw exception_t{.reason = reason};
#endif
  }
}