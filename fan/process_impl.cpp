module;
#include <fan/utility.h>
#include <uv.h>
#include <string>
#include <string_view>
#include <vector>
#include <functional>
#include <coroutine>
#include <algorithm>
#include <expected>
#include <unordered_map>
#include <atomic>

#define DEBUG_PRINTS 0

#if DEBUG_PRINTS
  #define DPRINT(...) fan::print(__VA_ARGS__)
#else
  #define DPRINT(...) ((void)0)
#endif

module fan.process;

import fan.print.error;

namespace fan::process {
  struct spawn_state_t {
    void try_resume() {
      if (process_exited && pipe_closed && co_handle) {
        auto h = co_handle;
        co_handle = {};
        h();
      }
    }

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
  };

  void spawn_t::start(const std::vector<std::string>& args, std::function<void(std::string_view)> on_line) {
    auto* st = new spawn_state_t;
    st->on_line = std::move(on_line);
    st->running = true;

    auto* loop = (uv_loop_t*)fan::event::get_loop();
    uv_pipe_init(loop, &st->pipe, 0);

    std::vector<char*> argv;
    for (auto& a : args) { argv.push_back(const_cast<char*>(a.c_str())); }
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

    uv_read_start(reinterpret_cast<uv_stream_t*>(&st->pipe),
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
            if (line.empty()) { continue; }
            st->last_line = line;
            st->on_line(line);
          }
        }
        if (buf->base) { delete[] buf->base; }
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
    if (proc_.state_) {
      static_cast<spawn_state_t*>(proc_.state_)->co_handle = handle;
    }
  }

  run_result_t run_awaitable::await_resume() const noexcept {
    DPRINT("[process] coroutine resumed");
    if (result_.spawned && proc_.state_) {
      const_cast<run_result_t&>(result_).exit_code = static_cast<spawn_state_t*>(proc_.state_)->exit_code;
    }
    return result_;
  }

  run_awaitable run_async(const std::vector<std::string>& args, std::function<void(std::string_view)> on_line) {
    return run_awaitable(args, std::move(on_line));
  }

  struct ipc_conn_t {
    uv_pipe_t   pipe {};
    std::string linebuf;
    std::vector<std::string> message_queue;
    std::coroutine_handle<> read_handle {};
    bool connected = false;
    std::string error_msg = "pipe disconnected";
  };

  struct ipc_server_state_t {
    uv_pipe_t server {};
    std::function<fan::event::task_t(ipc_server_conn_t)> on_connect;
    std::vector<fan::event::task_t> active_tasks;
  };

  struct ipc_client_state_t {
    uv_pipe_t   pipe {};
    std::string linebuf;
    std::string error_msg = "pipe disconnected";
    bool connected = false;
    std::vector<std::string> message_queue;
    std::coroutine_handle<> connect_handle {};
    std::coroutine_handle<> read_handle {};
  };

  static void ipc_alloc_cb(uv_handle_t*, size_t n, uv_buf_t* buf) {
    buf->base = new char[n];
    buf->len  = (unsigned long)n;
  }

  static void ipc_write_msg(uv_stream_t* stream, std::string_view msg) {
    auto* req = new uv_write_t;
    auto* buf = new std::string(msg);
    buf->push_back('\n');
    uv_buf_t b = uv_buf_init(buf->data(), (unsigned int)buf->size());
    req->data = buf;
    uv_write(req, stream, &b, 1, [](uv_write_t* r, int) {
      delete static_cast<std::string*>(r->data);
      delete r;
    });
  }

  ipc_server_t::ipc_server_t(std::string_view path, std::function<fan::event::task_t(ipc_server_conn_t)> on_connect) {
    auto* st = new ipc_server_state_t;
    st->on_connect = std::move(on_connect);
    state_ = st;
    st->server.data = st;
    auto* loop = (uv_loop_t*)fan::event::get_loop();
    uv_pipe_init(loop, &st->server, 0);
#if !defined(fan_platform_windows)
    uv_fs_t fs;
    uv_fs_unlink(loop, &fs, path.data(), nullptr);
#endif
    uv_pipe_bind(&st->server, path.data());
    uv_listen((uv_stream_t*)&st->server, 128, [](uv_stream_t* srv, int status) {
      if (status < 0) { return; }
      auto* st   = static_cast<ipc_server_state_t*>(srv->data);
      auto* conn = new ipc_conn_t;
      conn->connected = true;
      uv_pipe_init(srv->loop, &conn->pipe, 0);
      uv_accept(srv, (uv_stream_t*)&conn->pipe);
      conn->pipe.data = conn;
      uv_read_start((uv_stream_t*)&conn->pipe, ipc_alloc_cb, [](uv_stream_t* s, ssize_t nread, const uv_buf_t* b) {
        auto* conn = static_cast<ipc_conn_t*>(s->data);
        if (nread > 0) {
          conn->linebuf.append(b->base, nread);
          size_t pos;
          while ((pos = conn->linebuf.find('\n')) != std::string::npos) {
            std::string msg = conn->linebuf.substr(0, pos);
            conn->linebuf.erase(0, pos + 1);
            if (!msg.empty()) {
              conn->message_queue.push_back(msg);
              if (conn->read_handle) { auto h = conn->read_handle; conn->read_handle = {}; h(); }
            }
          }
        }
        if (b->base) { delete[] b->base; }
        if (nread == UV_EOF || nread < 0) {
          conn->connected = false;
          uv_read_stop(s);
          if (!uv_is_closing((uv_handle_t*)s)) {
            uv_close((uv_handle_t*)s, [](uv_handle_t* h) { delete static_cast<ipc_conn_t*>(h->data); });
          }
          if (conn->read_handle) { auto h = conn->read_handle; conn->read_handle = {}; h(); }
        }
      });
      st->active_tasks.push_back(st->on_connect(ipc_server_conn_t{conn}));
    });
  }

  void ipc_server_t::close() {
    auto* st = static_cast<ipc_server_state_t*>(state_);
    if (!st) { return; }
    uv_close((uv_handle_t*)&st->server, [](uv_handle_t* h) { delete static_cast<ipc_server_state_t*>(h->data); });
    state_ = nullptr;
  }

  ipc_server_conn_t::read_awaitable ipc_server_conn_t::read() { return {this}; }
  bool ipc_server_conn_t::read_awaitable::await_ready() const noexcept {
    auto* conn = static_cast<ipc_conn_t*>(conn_->state_);
    return !conn->connected || !conn->message_queue.empty();
  }
  void ipc_server_conn_t::read_awaitable::await_suspend(std::coroutine_handle<> handle) {
    static_cast<ipc_conn_t*>(conn_->state_)->read_handle = handle;
  }
  std::string ipc_server_conn_t::read_awaitable::await_resume() const {
    auto* conn = static_cast<ipc_conn_t*>(conn_->state_);
    if (!conn->message_queue.empty()) {
      auto msg = std::move(conn->message_queue.front());
      conn->message_queue.erase(conn->message_queue.begin());
      return msg;
    }
    throw fan::task_cancelled_exception{conn->error_msg};
  }

  void ipc_server_conn_t::send(std::string_view msg) {
    auto* conn = static_cast<ipc_conn_t*>(state_);
    if (conn && conn->connected) { ipc_write_msg((uv_stream_t*)&conn->pipe, msg); }
  }

  bool ipc_server_conn_t::is_connected() const {
    auto* conn = static_cast<ipc_conn_t*>(state_);
    return conn && conn->connected;
  }

  void ipc_server_conn_t::close() {
    auto* conn = static_cast<ipc_conn_t*>(state_);
    if (!conn) { return; }
    if (conn->connected) {
      conn->connected = false;
      if (!uv_is_closing((uv_handle_t*)&conn->pipe)) {
        uv_close((uv_handle_t*)&conn->pipe, [](uv_handle_t* h) { delete static_cast<ipc_conn_t*>(h->data); });
      }
    }
    state_ = nullptr;
  }

  ipc_client_t::connect_awaitable ipc_client_t::connect(std::string_view path) { return {std::string(path), nullptr}; }
  bool ipc_client_t::connect_awaitable::await_ready() const noexcept { return false; }
  void ipc_client_t::connect_awaitable::await_suspend(std::coroutine_handle<> handle) {
    auto* st = new ipc_client_state_t;
    st_ = st; st->connect_handle = handle; st->pipe.data = st;
    auto* loop = (uv_loop_t*)fan::event::get_loop();
    uv_pipe_init(loop, &st->pipe, 0);
    auto* req = new uv_connect_t;
    req->data = st;
    uv_pipe_connect(req, &st->pipe, path_.c_str(), [](uv_connect_t* req, int status) {
      auto* st = static_cast<ipc_client_state_t*>(req->data);
      delete req;
      if (status < 0) {
        st->error_msg = uv_strerror(status);
        if (st->connect_handle) { auto h = st->connect_handle; st->connect_handle = {}; h(); }
        return;
      }
      st->connected = true;
      uv_read_start((uv_stream_t*)&st->pipe, ipc_alloc_cb, [](uv_stream_t* s, ssize_t nread, const uv_buf_t* b) {
        auto* st = static_cast<ipc_client_state_t*>(s->data);
        if (nread > 0) {
          st->linebuf.append(b->base, nread);
          size_t pos;
          while ((pos = st->linebuf.find('\n')) != std::string::npos) {
            std::string msg = st->linebuf.substr(0, pos);
            st->linebuf.erase(0, pos + 1);
            if (!msg.empty()) {
              st->message_queue.push_back(msg);
              if (st->read_handle) { auto h = st->read_handle; st->read_handle = {}; h(); }
            }
          }
        }
        if (b->base) { delete[] b->base; }
        if (nread == UV_EOF || nread < 0) {
          st->connected = false;
          uv_read_stop(s);
          if (!uv_is_closing((uv_handle_t*)s)) { uv_close((uv_handle_t*)s, nullptr); }
          if (st->read_handle) { auto h = st->read_handle; st->read_handle = {}; h(); }
        }
      });
      if (st->connect_handle) { auto h = st->connect_handle; st->connect_handle = {}; h(); }
    });
  }

  ipc_client_t ipc_client_t::connect_awaitable::await_resume() const {
    auto* st = static_cast<ipc_client_state_t*>(st_);
    if (st->connected) { ipc_client_t c; c.state_ = st; return c; }
    std::string err = st->error_msg;
    if (!uv_is_closing((uv_handle_t*)&st->pipe)) {
      uv_close((uv_handle_t*)&st->pipe, [](uv_handle_t* h) { delete static_cast<ipc_client_state_t*>(h->data); });
    }
    throw fan::task_cancelled_exception{err};
  }

  ipc_client_t::read_awaitable ipc_client_t::read() { return {this}; }
  bool ipc_client_t::read_awaitable::await_ready() const noexcept {
    auto* st = static_cast<ipc_client_state_t*>(client_->state_);
    return !st->connected || !st->message_queue.empty();
  }
  void ipc_client_t::read_awaitable::await_suspend(std::coroutine_handle<> handle) {
    static_cast<ipc_client_state_t*>(client_->state_)->read_handle = handle;
  }
  std::string ipc_client_t::read_awaitable::await_resume() const {
    auto* st = static_cast<ipc_client_state_t*>(client_->state_);
    if (!st->message_queue.empty()) {
      auto msg = std::move(st->message_queue.front());
      st->message_queue.erase(st->message_queue.begin());
      return msg;
    }
    throw fan::task_cancelled_exception{st->error_msg};
  }

  void ipc_client_t::send(std::string_view msg) {
    auto* st = static_cast<ipc_client_state_t*>(state_);
    if (st && st->connected) { ipc_write_msg((uv_stream_t*)&st->pipe, msg); }
  }

  bool ipc_client_t::is_connected() const {
    auto* st = static_cast<ipc_client_state_t*>(state_);
    return st && st->connected;
  }

  void ipc_client_t::close() {
    auto* st = static_cast<ipc_client_state_t*>(state_);
    if (!st) { return; }
    if (st->connected) {
      st->connected = false;
      if (!uv_is_closing((uv_handle_t*)&st->pipe)) {
        uv_close((uv_handle_t*)&st->pipe, [](uv_handle_t* h) { delete static_cast<ipc_client_state_t*>(h->data); });
      }
    }
    state_ = nullptr;
  }

  static std::unordered_map<std::string, std::function<void()>>& child_registry() {
    static std::unordered_map<std::string, std::function<void()>> r;
    return r;
  }

  static std::string get_self_path() {
  #if defined(fan_platform_windows)
    char buf[4096];
    GetModuleFileNameA(nullptr, buf, sizeof(buf));
    return buf;
  #else
    char buf[4096];
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n < 0) { fan::throw_error_impl("spawn_self: readlink failed"); }
    buf[n] = '\0';
    return buf;
  #endif
  }

  fan::event::task_t spawn_self_impl(std::function<void()> child_fn) {
    static std::atomic<uint32_t> counter {0};
    std::string id = std::to_string(++counter);

    child_registry()[id] = std::move(child_fn);

  #if defined(fan_platform_windows)
    ::SetEnvironmentVariableA(spawn_self_env.c_str(), id.c_str());
    std::vector<std::string> args = {get_self_path()};
    auto result = co_await run_async(args, [](std::string_view line) { fan::print("[child]", line); });
    ::SetEnvironmentVariableA(spawn_self_env.c_str(), nullptr);
  #else
    std::vector<std::string> args = {get_self_path()};
    setenv(spawn_self_env.c_str(), id.c_str(), 1);
    auto result = co_await run_async(args, [](std::string_view line) { fan::print("[child]", line); });
    unsetenv(spawn_self_env.c_str());
  #endif

    child_registry().erase(id);
    co_return;
  }

  struct spawn_self_child_checker_t {
    spawn_self_child_checker_t() {
      const char* id = ::getenv(fan::process::spawn_self_env.c_str());
      if (!id) { return; }
      auto& reg = child_registry();
      auto it = reg.find(id);
      if (it != reg.end()) {
        it->second();
      }
      ::exit(0);
    }
  } spawn_self_child_checker;
}