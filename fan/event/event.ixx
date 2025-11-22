module;

#include <utility>
#include <coroutine>
#include <string>
#include <functional>
#include <filesystem>
#include <thread>

#include <uv.h>
#undef min
#undef max

export module fan.event;

import fan.print;
import fan.utility;
export import :types;

export namespace fan::event {

  loop_t loop_new();
  loop_t& get_loop();
  void loop_stop(loop_t loop);
  int loop_close(loop_t loop);

  struct error_code_t {
    int code;
    constexpr error_code_t(int code) noexcept : code(code) {}
    constexpr operator int() const noexcept { return code; }
    constexpr bool await_ready() const noexcept { return true; }
    constexpr void await_suspend(std::coroutine_handle<>) const noexcept {}
    void throw_if() const;
    void await_resume() const;
  };

  void print_event_handles(loop_t loop = get_loop());

  struct timer_t {
    struct timer_data {
      uv_timer_t timer_handle;
      std::coroutine_handle<> co_handle;
      int ready;
      timer_data();
    };

    struct timer_deleter {
      void operator()(timer_data* data) const noexcept;
    };

    timer_t();
    timer_t(uint64_t timeout, uint64_t repeat = 0);
    error_code_t start(uint64_t timeout, uint64_t repeat = 0) noexcept;
    error_code_t again() noexcept;
    void set_repeat(uint64_t repeat) noexcept;
    error_code_t stop() noexcept;
    bool await_ready() const noexcept;
    void await_suspend(std::coroutine_handle<> h) noexcept;
    void await_resume() noexcept;

    std::unique_ptr<timer_data, timer_deleter> data;
  };

  struct deferred_resume_t {
    struct queued_resume_t {
      std::shared_ptr<void> keepalive;
      std::coroutine_handle<> h;
    };

    template<typename promise_t>
    static void schedule_resume(std::coroutine_handle<promise_t> h) {
      auto& p = h.promise();
      std::shared_ptr<void> keepalive(p.self_keepalive, p.self_keepalive.get());
      resume_queue.push_back(queued_resume_t{ keepalive, h });
    }

    static void process_resumes();

    static inline std::vector<queued_resume_t> resume_queue;
  };

  template<typename promise_t>
  void schedule_resume(std::coroutine_handle<promise_t> h) {
    deferred_resume_t::schedule_resume(h);
  }

  // requires the derived function to have "check_condition" function
  template <typename derived_t>
  struct condition_awaiter {
    bool await_ready() const {
      return static_cast<const derived_t*>(this)->check_condition();
    }
    void await_resume() {}
  };

  struct idle_task {
  private:
    struct idle_data {
      std::function<task_t()> callback;
      task_t task;
      bool has_task;
      idle_data(std::function<task_t()> cb);
    };

    static void idle_callback(uv_idle_t* handle);
    static void close_callback(uv_handle_t* handle);

  public:
    static idle_id_t task_idle(std::function<task_t()> callback);
    static void idle_stop(idle_id_t idle_handle);
  };

  idle_id_t task_idle(std::function<task_t()> callback);
  void idle_stop(idle_id_t idle_handle);

  struct counter_awaitable_t {
    struct state_t {
      size_t remaining;
      std::coroutine_handle<> continuation;
    };
    state_t* st;
    bool await_ready() const noexcept;
    void await_suspend(std::coroutine_handle<> h) noexcept;
    void await_resume() const noexcept;
  };

  template <typename... tasks_t>
  fan::event::task_t when_all(tasks_t&&... tasks) {
    counter_awaitable_t::state_t state{ sizeof...(tasks_t), {} };
    auto run_task = [&](auto&& t) -> fan::event::task_t {
      co_await std::forward<decltype(t)>(t);
      if (--state.remaining == 0 && state.continuation) {
        state.continuation.resume();
      }
      };
    (fan::event::task_idle([&]() -> fan::event::task_t {
      co_await run_task(std::forward<tasks_t>(tasks));
      }), ...);
    co_await counter_awaitable_t{ &state };
  }

  template<typename T = void>
  struct signal_awaitable_t {
    bool ready;
    std::coroutine_handle<> waiting_coroutine;
    T value;
    void signal(T val = T{});
    bool await_ready() const;
    void await_suspend(std::coroutine_handle<> h);
    T await_resume() const;
  };

  template<>
  struct signal_awaitable_t<void> {
    bool ready;
    std::coroutine_handle<> waiting_coroutine;
    void signal();
    bool await_ready() const;
    void await_suspend(std::coroutine_handle<> h);
    void await_resume() const;
  };

  struct uv_fs_awaitable {
    uv_fs_t req;
    std::coroutine_handle<> handle;
    bool closed;
    uv_fs_awaitable();
    ~uv_fs_awaitable();
    uv_fs_awaitable(const uv_fs_awaitable&) = delete;
    uv_fs_awaitable& operator=(const uv_fs_awaitable&) = delete;
    uv_fs_awaitable(uv_fs_awaitable&&) = delete;
    uv_fs_awaitable& operator=(uv_fs_awaitable&&) = delete;
    bool await_ready() const noexcept;
    void await_suspend(std::coroutine_handle<> h) noexcept;
    void await_resume() noexcept;
  };

  struct uv_fs_open_awaitable : uv_fs_awaitable {
    uv_fs_open_awaitable(const std::string& path, int flags, int mode);
    static void on_open_cb(uv_fs_t* r);
    int result() const noexcept;
  };

  struct uv_fs_size_awaitable : uv_fs_awaitable {
    uv_fs_size_awaitable(int file);
    uv_fs_size_awaitable(const std::string& path);
    static void on_size_cb(uv_fs_t* r);
    int64_t result() const noexcept;
  };

  struct uv_fs_read_awaitable : uv_fs_awaitable {
    uv_fs_read_awaitable(int file, uv_buf_t buf, int64_t offset);
    static void on_read_cb(uv_fs_t* r);
    ssize_t result() const noexcept;
  };

  struct uv_fs_write_awaitable : uv_fs_awaitable {
    uv_fs_write_awaitable(int fd, const char* buffer, size_t length, int64_t offset);
    static void on_write_cb(uv_fs_t* req);
    ssize_t result() const noexcept;
  };

  struct uv_fs_close_awaitable : uv_fs_awaitable {
    uv_fs_close_awaitable(int file);
    static void on_close_cb(uv_fs_t* r);
  };

  struct fs_watcher_t {
    struct file_event {
      std::string filename;
      int events;
      std::chrono::steady_clock::time_point timestamp;
    };

    uv_fs_event_t fs_event;
    loop_t loop;
    uv_timer_t timer;
    std::string watch_path;
    std::function<void(const std::string&, int)> event_callback;
    std::unordered_map<std::string, file_event> pending_events;

    static void timer_callback(uv_timer_t* handle);
    static void fs_callback(uv_fs_event_t* handle, const char* filename, int events, int status);
    void process_latest_events();
    fs_watcher_t(const std::string& path);
    bool start(std::function<void(const std::string&, int)> callback);
    void stop();
  };


  fan::event::task_t task_timer(uint64_t time, auto l) {
    while (true) {
      if constexpr (fan::is_awaitable_v<decltype(l())>) {
        bool ret = co_await l();
        if (ret) {
          break;
        }
      }
      else {
        if (l()) {
          break;
        }
      }
      co_await event::timer_t(time);
    }
  }


  struct fd_waiter_t {
    uv_poll_t poll_handle;
    std::coroutine_handle<> co_handle;
    bool ready = false;
    int events_received = 0;
  };

  fan::event::task_value_resume_t<void> wait_fd(loop_t loop, int fd, int events);
  void sleep(unsigned int msec);
  void loop(fan::event::loop_t loop = fan::event::get_loop(), bool once = false);
  uint64_t now();
  std::string strerror(int err);

  //thread stuff

  template <typename cb_t, typename ...args_t>
  void thread_create(cb_t&& cb, args_t&&... args) {
    std::jthread([cb = std::forward<cb_t>(cb), args = std::make_tuple(std::forward<args_t>(args)...)]() mutable {
      std::apply(cb, args);
      }).detach();
  }
}

export namespace fan {
  using co_sleep = event::timer_t;
}

export namespace fan::io::file {
  fan::event::task_value_resume_t<int> async_open(const std::string& path, int flags = fan::fs_in, int mode = fan::perm_0644) {
    fan::event::uv_fs_open_awaitable open_req(path, flags, mode);
    co_await open_req;
    auto r = open_req.result();
    if (r < 0) {
      fan::throw_error("failed to open file:" + path, "error:"_str + uv_strerror(r));
    }
    co_return r;
  }

  fan::event::task_value_resume_t<ssize_t> async_read(int file, char* buffer, size_t buffer_size, int64_t offset) {
    uv_buf_t buf = uv_buf_init(buffer, buffer_size);
    fan::event::uv_fs_read_awaitable read_req(file, buf, offset);
    co_await read_req;
    auto r = read_req.result();
    if (r < 0) {
      fan::throw_error("error reading file", std::to_string(r));
    }
    co_return r;
  }
  fan::event::task_value_resume_t<ssize_t> async_read(int file, std::string* buffer, int64_t offset, std::size_t buffer_size = 4096) {
    buffer->resize(buffer_size);
    uv_buf_t buf = uv_buf_init(buffer->data(), buffer_size);
    fan::event::uv_fs_read_awaitable read_req(file, buf, offset);
    co_await read_req;
    auto r = read_req.result();
    if (r < 0) {
      fan::throw_error("error reading file", std::to_string(r));
    }
    buffer->resize(r);
    co_return r;
  }

  fan::event::task_value_resume_t<ssize_t> async_write(int fd, const char* buffer, size_t length, int64_t offset) {
    fan::event::uv_fs_write_awaitable write_req(fd, buffer, length, offset);
    co_await write_req;
    ssize_t written = write_req.result();
    if (written < 0) {
      fan::throw_error("failed to write to file: error:"_str + uv_strerror(static_cast<int>(written)));
    }
    co_return written;
  }

  fan::event::task_t async_close(int file) {
    fan::event::uv_fs_close_awaitable close_req(file);
    co_await close_req;
  }

  
  fan::event::task_value_resume_t<intptr_t> async_size(int file) {
    fan::event::uv_fs_size_awaitable size_req(file);
    co_await size_req;
    co_return size_req.result();
  }

  fan::event::task_value_resume_t<intptr_t> async_size(const std::string& path) {
    fan::event::uv_fs_size_awaitable size_req(path);
    co_await size_req;
    co_return size_req.result();
  }

  template <typename lambda_t>
  fan::event::task_t async_read_cb(const std::string& path, lambda_t&& lambda, int buffer_size = 64) {
    int fd = co_await fan::io::file::async_open(path);
    int offset = 0;
    std::string buffer;
    buffer.resize(buffer_size);
    while (true) {
      std::size_t result = co_await fan::io::file::async_read(fd, buffer.data(), buffer.size(), offset);
      if (result == 0) {
        break;
      }
      std::string chunk(buffer.data(), result);
      if constexpr (fan::is_awaitable_v<decltype(lambda(chunk))>) {
        co_await lambda(chunk);
      }
      else {
        lambda(chunk);
      }
      offset += result;
    }
    co_await fan::io::file::async_close(fd);
  }
  fan::event::task_value_resume_t<std::string> async_read(const std::string& path, int buffer_size = 4096) {
    int fd = co_await fan::io::file::async_open(path);
    int offset = 0;
    std::string buffer;
    buffer.resize(buffer_size);
    std::string content;

    while (true) {
      std::size_t result = co_await fan::io::file::async_read(fd, buffer.data(), buffer.size(), offset);
      if (result == 0) {
        break;
      }
      content.append(buffer.data(), result);
      offset += result;
    }

    co_await fan::io::file::async_close(fd);
    co_return content;
  }

  fan::event::task_t async_write(const std::string& path, const std::string& data) {
    int fd = co_await fan::io::file::async_open(path, fan::fs_out);

    size_t offset = 0;
    size_t buffer_size = 4096;
    size_t total_written = 0;

    while (total_written < data.size()) {
      size_t remaining = data.size() - total_written;
      size_t to_write = std::min(remaining, buffer_size);

      std::string buffer(data.data() + total_written, to_write);
      std::size_t written = co_await fan::io::file::async_write(fd, buffer.data(), buffer.size(), offset + total_written);
      if (written == 0) {
        fan::throw_error("write failed");
      }
      total_written += written;
    }
    co_await fan::io::file::async_close(fd);
  }

  struct async_read_t {
    std::string path;
    int fd = -1;
    intptr_t offset = 0;
    intptr_t size = 0;

    fan::event::task_resume_t open(const std::string& file_path) {
      path = file_path;
      fd = co_await fan::io::file::async_open(path);
      size = co_await fan::io::file::async_size(fd);
    }
    fan::event::task_resume_t close() {
      co_await fan::io::file::async_close(fd);
    }
    fan::event::task_value_resume_t<std::string> read() {
      std::string buffer;
      intptr_t read = co_await fan::io::file::async_read(fd, &buffer, offset);
      if (read > 0) {
        offset += read;
      }
      if (read < 0) {
        fan::throw_error("fs read error:" + fan::event::strerror(read));
      }
      co_return buffer;
    }
  };

  struct async_write_t {
    std::string path;
    int fd = -1;
    intptr_t offset = 0;

    fan::event::task_resume_t open(const std::string& file_path) {
      path = file_path;
      fd = co_await fan::io::file::async_open(path, fan::fs_out);
    }
    fan::event::task_resume_t close() const {
      co_await fan::io::file::async_close(fd);
    }
    fan::event::task_value_resume_t<intptr_t> write(const std::string& data, std::size_t buffer_size = 4096) {
      intptr_t result = co_await fan::io::file::async_write(fd, data.data() + offset, std::min(data.size() - offset, buffer_size), offset);
      if (result > 0) {
        offset += result;
      }
      co_return result;
    }
  };
}

export namespace fan::io {
  struct ev_next_tick_awaiter {
    uv_idle_t* idle_handle = nullptr;
    std::coroutine_handle<> coro;
    bool await_ready() noexcept;
    void await_suspend(std::coroutine_handle<> h) noexcept;
    void await_resume() noexcept;
  };

  struct async_directory_iterator_t {
    std::vector<std::filesystem::directory_entry> entries;
    std::function<fan::event::task_t(const std::filesystem::directory_entry&)> callback;
    std::string base_path;
    std::string next_path;
    bool sort_alphabetically = false;
    size_t current_index = 0;
    bool stopped = false;
    bool operation_in_progress = false;
    bool switch_requested = false;
    fan::event::task_t iteration_task;
    void stop();
  };

  fan::event::task_t iterate_directory(async_directory_iterator_t* state);
  void async_directory_iterate(async_directory_iterator_t* state, const std::string& path);
}

struct cleaner_t {
  ~cleaner_t();
} cleaner;