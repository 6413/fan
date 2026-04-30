module;

#include <coroutine>

export module fan.process;

import std;
import fan.event;
import fan.event.types;
import fan.print;
import fan.log_dispatcher;

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
    bool         await_ready()                        const noexcept;
    void         await_suspend(std::coroutine_handle<> handle);
    run_result_t await_resume()                       const noexcept;
    spawn_t      proc_;
    run_result_t result_;
  };

  run_awaitable run_async(const std::vector<std::string>& args, std::function<void(std::string_view)> on_line);

  struct ipc_server_conn_t {
    struct read_awaitable {
      bool await_ready() const noexcept;
      void await_suspend(std::coroutine_handle<> handle);
      std::string await_resume() const;
      ipc_server_conn_t* conn_;
    };
    read_awaitable read();
    void send(std::string_view msg);
    bool is_connected() const;
    void close();
    void* state_ = nullptr;
  };

  struct ipc_server_t {
    ipc_server_t(std::string_view path, std::function<fan::event::task_t(ipc_server_conn_t)> on_connect);
    void close();
    void* state_ = nullptr;
  };

  struct ipc_client_t {
    struct connect_awaitable {
      std::string path_;
      void* st_ = nullptr;
      bool await_ready() const noexcept;
      void await_suspend(std::coroutine_handle<> handle);
      ipc_client_t await_resume() const;
    };
    struct read_awaitable {
      bool await_ready() const noexcept;
      void await_suspend(std::coroutine_handle<> handle);
      std::string await_resume() const;
      ipc_client_t* client_;
    };
    static connect_awaitable connect(std::string_view path);
    read_awaitable read();
    void send(std::string_view msg);
    bool is_connected() const;
    void close();
    void* state_ = nullptr;
  };
  
  fan::event::task_t spawn_self_impl(std::function<void()> child_fn);

  inline fan::event::task_t spawn_self(std::function<void()> child_fn) {
    return spawn_self_impl(std::move(child_fn));
  }

  inline std::string spawn_self_env = "FAN_SPAWN_SELF_ID";

  inline std::string ipc_default_path(std::string_view name) {
#if defined(fan_platform_windows)
    return std::string("\\\\.\\pipe\\") + std::string(name);
#else
    return std::string("/tmp/") + std::string(name) + ".sock";
#endif
  }
}