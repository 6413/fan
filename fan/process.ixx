module;

#include <uv.h>
#include <string>
#include <vector>
#include <functional>
#include <coroutine>

export module fan.process;

import fan.event;
export import fan.print;
export import fan.log_dispatcher;

export namespace fan::process {
  inline fan::log_dispatcher_t default_logger() {
    return fan::log_dispatcher_t {}
      .on("ERROR:", [](std::string_view l) { fan::print_error(l); })
      .on("error:", [](std::string_view l) { fan::print_error(l); })
      .on("WARNING:", [](std::string_view l) { fan::print_warning(l); })
      .on("warning:", [](std::string_view l) { fan::print_warning(l); })
      .otherwise([](std::string_view l) { fan::print(l); });
  }

  struct spawn_t {
    void start(const std::vector<std::string>& args, std::function<void(std::string_view)> on_line);
    bool is_running() const;
    int exit_code() const;

    void* state_ = nullptr;
  };

  struct run_result_t {
    bool ok() const { return spawned && exit_code == 0; }

    bool spawned = false;
    int  exit_code = -1;
  };

  struct run_awaitable {
    run_awaitable(const std::vector<std::string>& args, std::function<void(std::string_view)> on_line);

    bool await_ready()                        const noexcept;
    void await_suspend(std::coroutine_handle<> handle);
    run_result_t   await_resume()                       const noexcept;

    spawn_t proc_;
    run_result_t result_;
  };

  run_awaitable run_async(const std::vector<std::string>& args, std::function<void(std::string_view)> on_line);
}