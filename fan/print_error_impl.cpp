module;

module fan.print.error;

import std;

// NOTE: Do NOT #include <ctime> here. `import std;` already provides
// std::chrono, std::time_t, std::tm. See gcc_bug.txt.

namespace fan {
  log_t& get_error_log() {
    static log_t log;
    return log;
  }

  void write_error_to_disk(const std::string& msg) {
    auto& log = get_error_log();
    auto now = std::chrono::system_clock::now();
    // NOTE: use std::format chrono (no <ctime> / localtime_r) to stay
    // compatible with `import std;` + GCC -freflection (see gcc_bug.txt).
    std::string ts;
    try {
      ts = std::format("{:%Y-%m-%d %H:%M:%S}",
        std::chrono::floor<std::chrono::seconds>(now));
    }
    catch (...) {
      ts = "unknown-time";
    }
    std::string new_entry = ts + " - " + msg + '\n';

    std::lock_guard<std::mutex> lock(log.mtx);
    std::ifstream in(log.filename, std::ios::binary);
    std::string existing((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();

    std::string tmp_path = log.filename + ".tmp";
    {
      std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
      out << new_entry << existing;
    }
    std::filesystem::rename(tmp_path, log.filename);
  }

  void push_memory_log(const std::string& tag, const std::string& msg, log_level_e level) {
    auto& log = get_error_log();
    std::lock_guard<std::mutex> lock(log.mtx);
    log.buffer.push_back({tag, msg, level});
    if (log.buffer.size() > log.max_size) {
      log.buffer.pop_front();
    }
    ++log.total_logs_pushed;
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
      res += std::format("\n{}", std::stacktrace::current());
      write_error_to_disk(res);
      push_memory_log(res, log_level_e::error);
    }
#if __cpp_exceptions
    throw exception_t{.reason = reason};
#endif
  }
}