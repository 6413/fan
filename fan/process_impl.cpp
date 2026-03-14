module;

#include <uv.h>
#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <coroutine>
#include <sstream>

#define DEBUG_PRINTS 0

#if DEBUG_PRINTS
  #define DPRINT(...) fan::print(__VA_ARGS__)
#else
  #define DPRINT(...) ((void)0)
#endif

module fan.process;

namespace fan::process {

  struct spawn_state_t {
    uv_process_t process {};
    uv_pipe_t pipe {};
    std::string line_buf;
    std::string last_line;
    int exit_code = 0;
    bool running = false;
    bool process_exited = false;
    bool pipe_closed = false;
    std::function<void(std::string_view)> on_line;
    std::coroutine_handle<> co_handle {};

    void try_resume() {
      if (process_exited && pipe_closed && co_handle) {
        auto h = co_handle;
        co_handle = {};
        h();
      }
    }
  };

  void spawn_t::start(const std::vector<std::string>& args, std::function<void(std::string_view)> on_line) {
    auto* st = new spawn_state_t;
    st->on_line = std::move(on_line);
    st->running = true;

    auto loop = fan::event::get_loop();
    uv_pipe_init(loop, &st->pipe, 0);

    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
    argv.push_back(nullptr);

    uv_stdio_container_t stdio[3];
    stdio[0].flags = UV_IGNORE;
    stdio[1].flags = (uv_stdio_flags)(UV_CREATE_PIPE | UV_WRITABLE_PIPE);
    stdio[1].data.stream = reinterpret_cast<uv_stream_t*>(&st->pipe);
    stdio[2].flags = (uv_stdio_flags)(UV_CREATE_PIPE | UV_WRITABLE_PIPE);
    stdio[2].data.stream = reinterpret_cast<uv_stream_t*>(&st->pipe);

    uv_process_options_t opts {};
    opts.file = argv[0];
    opts.args = argv.data();
    opts.stdio = stdio;
    opts.stdio_count = 3;
    opts.flags = UV_PROCESS_WINDOWS_HIDE;
    opts.exit_cb = [](uv_process_t* proc, int64_t exit_status, int) {
      auto* st = static_cast<spawn_state_t*>(proc->data);
      DPRINT("[process] exit_cb: exit_status=", exit_status, " last_line=", st->last_line);
      st->exit_code = (int)exit_status;
      st->running = false;
      st->process_exited = true;
      uv_close(reinterpret_cast<uv_handle_t*>(proc), [](uv_handle_t* h) {
        auto* st = static_cast<spawn_state_t*>(h->data);
        st->try_resume();
      });
    };

    st->process.data = st;
    st->pipe.data = st;

    if (uv_spawn(loop, &st->process, &opts) != 0) {
      st->running = false;
      state_ = st;
      return;
    }

    state_ = st;
    DPRINT("[process] st=", (void*)st);
    DPRINT("[process] pid=", st->process.pid);

    int r = uv_read_start(reinterpret_cast<uv_stream_t*>(&st->pipe),
      [](uv_handle_t*, size_t n, uv_buf_t* buf) {
        buf->base = new char[n];
        buf->len = (unsigned long)n;
      },
      [](uv_stream_t* stream, ssize_t nread, const uv_buf_t* buf) {
        auto* st = static_cast<spawn_state_t*>(stream->data);
        if (nread > 0) {
          st->line_buf.append(buf->base, nread);
          size_t pos;
          while ((pos = st->line_buf.find_first_of("\n\r")) != std::string::npos) {
            std::string line = st->line_buf.substr(0, pos);
            st->line_buf.erase(0, pos + 1);
            if (line.empty()) continue;
            st->last_line = line;
            st->on_line(line);
          }
        }
        if (buf->base) delete[] buf->base;
        if (nread == UV_EOF || nread < 0) {
          DPRINT("[process] pipe EOF: nread=", nread, " last_line=", st->last_line);
          if (!st->line_buf.empty()) {
            st->on_line(st->line_buf);
            st->line_buf.clear();
          }
          uv_read_stop(stream);
          uv_close(reinterpret_cast<uv_handle_t*>(stream), [](uv_handle_t* h) {
            auto* st = static_cast<spawn_state_t*>(h->data);
            DPRINT("[process] pipe close cb");
            st->pipe_closed = true;
            st->try_resume();
          });
        }
      }
    );
    DPRINT("[process] uv_read_start result:", r);
  }

  bool spawn_t::is_running() const { return state_ && static_cast<spawn_state_t*>(state_)->running; }
  int spawn_t::exit_code() const { return state_ ? static_cast<spawn_state_t*>(state_)->exit_code : -1; }

  run_awaitable::run_awaitable(const std::vector<std::string>& args, std::function<void(std::string_view)> on_line) {
    bool success = true;
    try { proc_.start(args, std::move(on_line)); }
    catch (...) { success = false; }
    result_.spawned = success && proc_.is_running();
  }

  bool run_awaitable::await_ready() const noexcept { return false; }

  void run_awaitable::await_suspend(std::coroutine_handle<> handle) {
    DPRINT("[process] suspend st=", (void*)proc_.state_);
    if (proc_.state_)
      static_cast<spawn_state_t*>(proc_.state_)->co_handle = handle;
  }

  run_result_t run_awaitable::await_resume() const noexcept {
    DPRINT("[process] coroutine resumed");
    if (result_.spawned && proc_.state_)
      const_cast<run_result_t&>(result_).exit_code = static_cast<spawn_state_t*>(proc_.state_)->exit_code;
    return result_;
  }

  run_awaitable run_async(const std::vector<std::string>& args, std::function<void(std::string_view)> on_line) {
    return run_awaitable(args, std::move(on_line));
  }
}