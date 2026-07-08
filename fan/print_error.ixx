module;

export module fan.print.error;

import std;

export namespace fan {
  enum class log_level_e : std::uint8_t {
    info,
    warning,
    error
  };

  struct log_entry_t {
    std::string tag;
    std::string msg;
    log_level_e level;
  };

  struct log_t { 
    std::string filename = "fan_log.txt"; 
    std::deque<log_entry_t> buffer;
    std::size_t max_size = 1000;
    std::uint64_t total_logs_pushed = 0;
    std::mutex mtx;
  };
  
  log_t& get_error_log();
  void write_error_to_disk(const std::string& msg);
  struct exception_t { const char* reason; };
  void throw_error_impl(const char* reason = "");

  void push_memory_log(
    const std::string& tag,
    const std::string& msg,
    log_level_e level = log_level_e::info
  );
  void push_memory_log(const std::string& msg, log_level_e level = log_level_e::info);

  std::vector<log_entry_t> dump_memory_logs();
  std::vector<log_entry_t> dump_memory_logs_since(std::uint64_t& cursor);
  void clear_memory_logs();

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