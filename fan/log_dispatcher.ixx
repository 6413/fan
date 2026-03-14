module;

#include <string>
#include <vector>
#include <functional>

export module fan.log_dispatcher;

export namespace fan {
  struct log_dispatcher_t {
    struct entry_t { std::string prefix; std::function<void(std::string_view)> handler; };

    log_dispatcher_t& on(std::string prefix, std::function<void(std::string_view)> h) {
      entries.push_back({std::move(prefix), std::move(h)}); return *this;
    }
    log_dispatcher_t& otherwise(std::function<void(std::string_view)> h) {
      default_handler = std::move(h); return *this;
    }
    void operator()(std::string_view line) const {
      for (auto& e : entries)
        if (line.starts_with(e.prefix)) { e.handler(line); return; }
      if (default_handler) default_handler(line);
    }

    std::vector<entry_t> entries;
    std::function<void(std::string_view)> default_handler;
  };
}