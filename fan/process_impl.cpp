module;

#include <fan/utility.h>
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
    uv_pipe_init((uv_loop_t*)loop, &st->pipe, 0);

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

    if (uv_spawn((uv_loop_t*)loop, &st->process, &opts) != 0) {
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

  struct ipc_conn_t {
    uv_pipe_t  pipe {};
    std::string linebuf;
  };

  struct ipc_server_state_t {
    uv_pipe_t server {};
    std::vector<ipc_conn_t*> conns;
    std::function<void(std::string_view)> on_message;
  };

  void ipc_server_t::listen(std::string_view path, std::function<void(std::string_view)> on_message) {
    auto* st = new ipc_server_state_t;
    st->on_message = std::move(on_message);
    state_ = st;
    st->server.data = st;

    auto* loop = (uv_loop_t*)fan::event::get_loop();
    uv_pipe_init(loop, &st->server, 0);
    uv_fs_t fs;
    uv_fs_unlink(loop, &fs, path.data(), nullptr);
    uv_pipe_bind(&st->server, path.data());
    uv_listen((uv_stream_t*)&st->server, 128, [](uv_stream_t* srv, int status) {
      if (status < 0) { return; }
      auto* st   = static_cast<ipc_server_state_t*>(srv->data);
      auto* conn = new ipc_conn_t;
      uv_pipe_init(srv->loop, &conn->pipe, 0);
      uv_accept(srv, (uv_stream_t*)&conn->pipe);
      st->conns.push_back(conn);

      struct ctx_t { ipc_server_state_t* st; ipc_conn_t* conn; };
      auto* ctx = new ctx_t{st, conn};
      conn->pipe.data = ctx;

      uv_read_start((uv_stream_t*)&conn->pipe, ipc_alloc_cb,
        [](uv_stream_t* s, ssize_t nread, const uv_buf_t* b) {
          auto* ctx = static_cast<ctx_t*>(s->data);
          if (nread > 0) {
            ctx->conn->linebuf.append(b->base, nread);
            size_t pos;
            while ((pos = ctx->conn->linebuf.find('\n')) != std::string::npos) {
              std::string msg = ctx->conn->linebuf.substr(0, pos);
              ctx->conn->linebuf.erase(0, pos + 1);
              if (!msg.empty()) { ctx->st->on_message(msg); }
            }
          }
          if (b->base) { delete[] b->base; }
          if (nread == UV_EOF || nread < 0) {
            uv_read_stop(s);
            uv_close((uv_handle_t*)s, [](uv_handle_t* h) {
              auto* ctx = static_cast<ctx_t*>(h->data);
              auto& c = ctx->st->conns;
              c.erase(std::remove(c.begin(), c.end(), ctx->conn), c.end());
              delete ctx->conn;
              delete ctx;
            });
          }
        });
    });
  }

  void ipc_server_t::send(std::string_view msg) {
    auto* st = static_cast<ipc_server_state_t*>(state_);
    if (!st) { return; }
    for (auto* conn : st->conns) { ipc_write_msg((uv_stream_t*)&conn->pipe, msg); }
  }

  void ipc_server_t::close() {
    auto* st = static_cast<ipc_server_state_t*>(state_);
    if (!st) { return; }
    st->server.data = st;
    uv_close((uv_handle_t*)&st->server, [](uv_handle_t* h) {
      delete static_cast<ipc_server_state_t*>(h->data);
    });
    state_ = nullptr;
  }

  struct ipc_client_state_t {
    uv_pipe_t  pipe {};
    std::string linebuf;
    bool connected = false;
    std::function<void(std::string_view)> on_message;
    std::coroutine_handle<> co_handle {};
  };

  void ipc_client_t::connect(std::string_view path, std::function<void(std::string_view)> on_message) {
    auto* st = new ipc_client_state_t;
    st->on_message = std::move(on_message);
    state_ = st;
    st->pipe.data = st;

    auto* loop = (uv_loop_t*)fan::event::get_loop();
    uv_pipe_init(loop, &st->pipe, 0);

    auto* req = new uv_connect_t;
    req->data = st;
    uv_pipe_connect(req, &st->pipe, path.data(), [](uv_connect_t* req, int status) {
      auto* st = static_cast<ipc_client_state_t*>(req->data);
      delete req;
      if (status < 0) {
        if (st->co_handle) { auto h = st->co_handle; st->co_handle = {}; h(); }
        return;
      }
      st->connected = true;
      if (st->co_handle) { auto h = st->co_handle; st->co_handle = {}; h(); }
      uv_read_start((uv_stream_t*)&st->pipe, ipc_alloc_cb,
        [](uv_stream_t* s, ssize_t nread, const uv_buf_t* b) {
          auto* st = static_cast<ipc_client_state_t*>(s->data);
          if (nread > 0) {
            st->linebuf.append(b->base, nread);
            size_t pos;
            while ((pos = st->linebuf.find('\n')) != std::string::npos) {
              std::string msg = st->linebuf.substr(0, pos);
              st->linebuf.erase(0, pos + 1);
              if (!msg.empty()) { st->on_message(msg); }
            }
          }
          if (b->base) { delete[] b->base; }
          if (nread == UV_EOF || nread < 0) {
            auto* st = static_cast<ipc_client_state_t*>(s->data);
            st->connected = false;
            uv_read_stop(s);
            uv_close((uv_handle_t*)s, nullptr);
          }
        });
    });
  }

  void ipc_client_t::send(std::string_view msg) {
    auto* st = static_cast<ipc_client_state_t*>(state_);
    if (!st || !st->connected) { return; }
    ipc_write_msg((uv_stream_t*)&st->pipe, msg);
  }

  bool ipc_client_t::is_connected() const {
    auto* st = static_cast<ipc_client_state_t*>(state_);
    return st && st->connected;
  }

  void ipc_client_t::close() {
    auto* st = static_cast<ipc_client_state_t*>(state_);
    if (!st) { return; }
    st->pipe.data = st;
    uv_close((uv_handle_t*)&st->pipe, [](uv_handle_t* h) {
      delete static_cast<ipc_client_state_t*>(h->data);
    });
    state_ = nullptr;
  }

  connect_awaitable::connect_awaitable(std::string_view path,
                                       std::function<void(std::string_view)> on_message) {
    client_.connect(path, std::move(on_message));
  }

  bool connect_awaitable::await_ready() const noexcept { return false; }

  void connect_awaitable::await_suspend(std::coroutine_handle<> handle) {
    if (client_.state_) {
      static_cast<ipc_client_state_t*>(client_.state_)->co_handle = handle;
    }
  }

  connect_result_t connect_awaitable::await_resume() const noexcept {
    return {client_.is_connected()};
  }

  connect_awaitable connect_async(std::string_view path,
                                  std::function<void(std::string_view)> on_message) {
    return {path, std::move(on_message)};
  }

  inline std::string ipc_default_path(std::string_view name) {
  #if defined(fan_platform_windows)
    return std::string("\\\\.\\pipe\\") + std::string(name);
  #else
    return std::string("/tmp/") + std::string(name) + ".sock";
  #endif
  }
}