module;

#include <coroutine>
#define POSITION2_WINDOW_CENTER fan::vec2(fan::graphics::ctx().window->get_size() / 2)
#define POSITION3_WINDOW_CENTER fan::vec3(POSITION2_WINDOW_CENTER, 0)

#include <fan/utility.h>

export module fan.event;

import std;

import fan.mpl;

import fan.event.types;

export namespace fan::event {

  loop_t loop_new();
  loop_t& get_loop();
  void loop_stop(loop_t loop = fan::event::get_loop());
  int loop_close(loop_t loop = fan::event::get_loop());

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
    struct timer_data;
    struct timer_deleter {
      void operator()(timer_data* data) const noexcept;
    };

    timer_t();
    timer_t(std::uint64_t timeout, std::uint64_t repeat = 0);
    ~timer_t() = default;

    timer_t(timer_t&&) = default;
    timer_t& operator=(timer_t&&) = default;
    timer_t(const timer_t&) = delete;
    timer_t& operator=(const timer_t&) = delete;

    error_code_t start(std::uint64_t timeout, std::uint64_t repeat = 0) noexcept;
    error_code_t again() noexcept;
    void set_repeat(std::uint64_t repeat) noexcept;
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

  template <typename derived_t>
  struct condition_awaiter {
    bool await_ready() const {
      return static_cast<const derived_t*>(this)->check_condition();
    }
    void await_resume() {}
  };

  struct idle_task {
    static idle_id_t task_idle(std::function<task_t()> callback);
    static void idle_stop(idle_id_t idle_handle);
  };

  idle_id_t task_idle(std::function<task_t()> callback);
  void idle_stop(idle_id_t idle_handle);

  struct counter_awaitable_t {
    struct state_t {
      std::size_t remaining;
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
    void signal(T val = T{}) {
      value = std::move(val);
      ready = true;
      if (waiting_coroutine) waiting_coroutine.resume();
    }
    bool await_ready() const { return ready; }
    void await_suspend(std::coroutine_handle<> h) { waiting_coroutine = h; }
    T await_resume() const { return value; }
    bool ready = false;
    std::coroutine_handle<> waiting_coroutine{};
    T value{};
  };

  template<>
  struct signal_awaitable_t<void> {
    void signal() {
      ready = true;
      if (waiting_coroutine) waiting_coroutine.resume();
    }
    bool await_ready() const { return ready; }
    void await_suspend(std::coroutine_handle<> h) { waiting_coroutine = h; }
    void await_resume() const {}
    bool ready = false;
    std::coroutine_handle<> waiting_coroutine{};
  };

  struct fs_watcher_t {
    void* internal_state = nullptr;
    fs_watcher_t(const std::string& path);
    ~fs_watcher_t();
    bool start(std::function<void(const std::string&, int)> callback);
    void stop();
    std::string get_watch_path();
  };

  fan::event::task_t after(std::uint64_t time, auto l) {
    co_await fan::event::timer_t(time);
    if constexpr (fan::is_awaitable_v<decltype(l())>) { co_await l(); } else { l(); }
  }

  fan::event::task_t every(std::uint64_t time, auto l) {
    while (true) {
      if constexpr (fan::is_awaitable_v<decltype(l())>) {
        if (co_await l()) break;
      } else {
        if (l()) break;
      }
      co_await event::timer_t(time);
    }
  }

  void sleep(unsigned int msec);
  void loop(fan::event::loop_t l = fan::event::get_loop(), bool once = false);
  void run_once(fan::event::loop_t l = fan::event::get_loop());
  std::uint64_t now();
  std::string strerror(int err);

  template <typename cb_t, typename ...args_t>
  void thread_create(cb_t&& cb, args_t&&... args) {
    std::jthread([cb = std::forward<cb_t>(cb), args = std::make_tuple(std::forward<args_t>(args)...)]() mutable {
      std::apply(cb, args);
    }).detach();
  }

  struct poll_awaitable_t {
    poll_awaitable_t(loop_t loop, int fd, int events);
    ~poll_awaitable_t();
    bool await_ready() const noexcept;
    void await_suspend(std::coroutine_handle<> h) noexcept;
    int await_resume() noexcept;

    void* poll_handle = nullptr;
    std::coroutine_handle<> co_handle = nullptr;
    bool ready = false;
    int events_received = 0;
  };

  poll_awaitable_t poll_task(loop_t loop, int fd, int events);

  struct uv_fs_open_awaitable {
    alignas(8) std::uint8_t data[512];
    uv_fs_open_awaitable(const std::string& path, int flags, int mode);
    ~uv_fs_open_awaitable();
    bool await_ready() const noexcept { return false; }
    void await_suspend(std::coroutine_handle<> h) noexcept;
    int result() const noexcept;
    void await_resume() noexcept {}
  };

  struct uv_fs_read_awaitable {
    alignas(8) std::uint8_t data[512];
    uv_fs_read_awaitable(int file, char* buffer, std::size_t size, std::int64_t offset);
    ~uv_fs_read_awaitable();
    bool await_ready() const noexcept { return false; }
    void await_suspend(std::coroutine_handle<> h) noexcept;
    std::intptr_t result() const noexcept;
    void await_resume() noexcept {}
  };

  struct uv_fs_write_awaitable {
    alignas(8) std::uint8_t data[512];
    uv_fs_write_awaitable(int fd, const char* buffer, std::size_t length, std::int64_t offset);
    ~uv_fs_write_awaitable();
    bool await_ready() const noexcept { return false; }
    void await_suspend(std::coroutine_handle<> h) noexcept;
    std::intptr_t result() const noexcept;
    void await_resume() noexcept {}
  };

  struct uv_fs_close_awaitable {
    alignas(8) std::uint8_t data[512];
    uv_fs_close_awaitable(int file);
    ~uv_fs_close_awaitable();
    bool await_ready() const noexcept { return false; }
    void await_suspend(std::coroutine_handle<> h) noexcept;
    void await_resume() noexcept {}
  };

  struct uv_fs_size_awaitable {
    alignas(8) std::uint8_t data[512];
    uv_fs_size_awaitable(int file);
    uv_fs_size_awaitable(const std::string& path);
    ~uv_fs_size_awaitable();
    bool await_ready() const noexcept { return false; }
    void await_suspend(std::coroutine_handle<> h) noexcept;
    std::int64_t result() const noexcept;
    void await_resume() noexcept {}
  };
}

export namespace fan {
  using co_sleep = event::timer_t;

  template<typename func_t, FAN_UNIQUE_CALL>
  auto do_once(func_t&& f) requires (!fan::is_awaitable_v<decltype(f())>) { return f(); }

  template<typename func_t, FAN_UNIQUE_CALL>
  fan::event::task_t do_once(func_t&& f) requires (fan::is_awaitable_v<decltype(f())>) { co_return co_await f(); }
}

export namespace fan::io::file {
  fan::event::runv_t<int> async_open(const std::string& path, int flags = fan::fs_in, int mode = fan::perm_0644);
  fan::event::runv_t<std::intptr_t> async_read(int file, char* buffer, std::size_t buffer_size, std::int64_t offset);
  fan::event::runv_t<std::intptr_t> async_write(int fd, const char* buffer, std::size_t length, std::int64_t offset);
  fan::event::task_t async_close(int file);
  fan::event::runv_t<std::intptr_t> async_size(int file);
  fan::event::runv_t<std::intptr_t> async_size(const std::string& path);
  fan::event::runv_t<std::intptr_t> async_read(int file, std::string* buffer, std::int64_t offset, std::size_t buffer_size = 4096);

  template <typename lambda_t>
  fan::event::task_t async_read_cb(const std::string& path, lambda_t&& lambda, int buffer_size = 64) {
    int fd = co_await fan::io::file::async_open(path);
    int offset = 0;
    std::string buffer;
    buffer.resize(buffer_size);
    while (true) {
      std::intptr_t result = co_await fan::io::file::async_read(fd, buffer.data(), buffer.size(), offset);
      if (result <= 0) break;
      std::string chunk(buffer.data(), result);
      if constexpr (fan::is_awaitable_v<decltype(lambda(chunk))>) {
        co_await lambda(chunk);
      } else {
        lambda(chunk);
      }
      offset += result;
    }
    co_await fan::io::file::async_close(fd);
  }

  fan::event::runv_t<std::string> async_read(const std::string& path, int buffer_size = 4096);
  fan::event::task_t async_write(const std::string& path, const std::string& data);

  struct async_read_t {
    std::string path;
    int fd = -1;
    std::intptr_t offset = 0;
    std::intptr_t size = 0;

    fan::event::run_t open(const std::string& file_path);
    fan::event::run_t close();
    fan::event::runv_t<std::string> read();
  };

  struct async_write_t {
    std::string path;
    int fd = -1;
    std::intptr_t offset = 0;

    fan::event::run_t open(const std::string& file_path);
    fan::event::run_t close() const;
    fan::event::runv_t<std::intptr_t> write(const std::string& data, std::size_t buffer_size = 4096);
  };
}

export namespace fan::io {
  struct async_directory_iterator_t {
    void* internal_state = nullptr;
    std::function<fan::event::task_t(const std::string&, bool)> callback;
    bool sort_alphabetically = false;
    bool operation_in_progress = false;

    async_directory_iterator_t();
    ~async_directory_iterator_t();
    void stop();

    std::size_t get_current_index() const;
    std::size_t get_entries_size() const;
    bool is_finished() const;
  };

  void async_directory_iterate(async_directory_iterator_t* state, const std::string& path);
}