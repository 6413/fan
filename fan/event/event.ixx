module;
#include <coroutine>
#include <functional>
#include <queue>
#include <chrono>
#include <memory>
#include <exception>

#include <uv.h>
#undef min
#undef max

using namespace std::chrono_literals;

export module fan.event;

import fan.types.print;

export namespace fan {
  template <typename T, typename = void>
struct is_awaitable : std::false_type {};

template <typename T>
struct is_awaitable<T, std::void_t<
    decltype(std::declval<T>().await_ready()),
    decltype(std::declval<T>().await_suspend(std::declval<std::coroutine_handle<>>())),
    decltype(std::declval<T>().await_resume())
>> : std::true_type {};

template <typename T>
inline constexpr bool is_awaitable_v = is_awaitable<T>::value;

  namespace event {
    struct error_code_t {
      int code;
      constexpr error_code_t(int code) noexcept : code(code) {}
      constexpr operator int() const noexcept { return code; }
      void throw_if() const { if (code) throw code; }
      constexpr bool await_ready() const noexcept { return true; }
      constexpr void await_suspend(std::coroutine_handle<>) const noexcept {}
      void await_resume() const { throw_if(); }
    };
    struct timer_t {
      struct timer_data {
        uv_timer_t timer_handle;
        std::coroutine_handle<> co_handle = nullptr;
        int ready{ 0 };
      };

      struct timer_deleter {
        void operator()(timer_data* data) const noexcept {
          uv_close(reinterpret_cast<uv_handle_t*>(&data->timer_handle), [](uv_handle_t* timer_handle) {
            delete static_cast<timer_data*>(timer_handle->data);
            });
        }
      };

      std::unique_ptr<timer_data, timer_deleter> data;

      timer_t() : data(new timer_data{}, timer_deleter{}) {
        uv_timer_init(uv_default_loop(), &data->timer_handle);
        data->timer_handle.data = data.get();
      }
      // timeout in ms
      timer_t(uint64_t timeout, uint64_t repeat = 0)
        : data(new timer_data{}, timer_deleter{}) {
        uv_timer_init(uv_default_loop(), &data->timer_handle);
        data->timer_handle.data = data.get();
        start(timeout, repeat);
      }

      // timeout in ms
      error_code_t start(uint64_t timeout, uint64_t repeat = 0) noexcept {
        return uv_timer_start(&data->timer_handle, [](uv_timer_t* timer_handle) {
          auto data = static_cast<timer_data*>(timer_handle->data);
          ++data->ready;
          if (data->co_handle) {
            data->co_handle();
          }
          }, timeout, repeat);
      }
      error_code_t again() noexcept {
        return uv_timer_again(&data->timer_handle);
      }
      void set_repeat(uint64_t repeat) noexcept {
        return uv_timer_set_repeat(&data->timer_handle, repeat);
      }
      error_code_t stop() noexcept {
        return uv_timer_stop(&data->timer_handle);
      }
      bool await_ready() const noexcept { return data->ready; }

      void await_suspend(std::coroutine_handle<> h) noexcept { data->co_handle = h; }

      void await_resume() noexcept {
        data->co_handle = nullptr;
        --data->ready;
      };
    };

    template<typename T, typename suspend_type_t>
    struct task_value_wrap_t;

    template<typename T, typename suspend_type_t>
    struct task_value_promise_t {
      T value;
      std::exception_ptr exception = nullptr;
      std::coroutine_handle<> continuation = nullptr;

      task_value_wrap_t<T, suspend_type_t> get_return_object();
      suspend_type_t initial_suspend() noexcept { return {}; }
      auto final_suspend() noexcept {
        struct final_awaiter {
          bool await_ready() noexcept { return false; }

          void await_suspend(std::coroutine_handle<task_value_promise_t> h) noexcept {
            if (h.promise().continuation)
              h.promise().continuation.resume();
          }

          void await_resume() noexcept {}
        };

        return final_awaiter{};
      }
      void return_value(T&& val) {
        value = std::move(val);
      }
      void return_value(const T& val) {
        value = val;
      }
      void unhandled_exception() {
        exception = std::current_exception();
      }
    };

    template<typename suspend_type_t>
    struct task_value_promise_t<void, suspend_type_t> {
      std::exception_ptr exception = nullptr;
      std::coroutine_handle<> continuation = nullptr;

      task_value_wrap_t<void, suspend_type_t> get_return_object();
      suspend_type_t initial_suspend() noexcept { return {}; }
      auto final_suspend() noexcept {
        struct final_awaiter {
          bool await_ready() noexcept { return false; }
          void await_suspend(std::coroutine_handle<task_value_promise_t> h) noexcept {
            if (h.promise().continuation)
              h.promise().continuation.resume();
          }
          void await_resume() noexcept {}
        };
        return final_awaiter{};
      }
      void return_void() {}
      void unhandled_exception() {
        exception = std::current_exception();
      }
    };

    template<typename T, typename suspend_type_t>
    struct task_value_wrap_t {
      using promise_type = task_value_promise_t<T, suspend_type_t>;

      task_value_wrap_t() {}
      task_value_wrap_t(std::coroutine_handle<promise_type> ch) : handle(ch) {}
      task_value_wrap_t(const task_value_wrap_t&) = delete;
      task_value_wrap_t(task_value_wrap_t&& other) : handle(other.handle) {
        other.handle = nullptr;
      }
      task_value_wrap_t& operator=(task_value_wrap_t&& other) {
        destroy();
        std::swap(handle, other.handle);
        return *this;
      }
      void destroy() {
        if (handle) {
          handle.destroy();
          handle = nullptr;
        }
      }
      ~task_value_wrap_t() {
        if (handle) {
          handle.destroy();
        }
      }
      bool valid() const {
        return static_cast<bool>(handle);
      }
      bool await_ready() const noexcept {
        return handle.done();
      }
      void await_suspend(std::coroutine_handle<> continuation_handle) noexcept {
        handle.promise().continuation = continuation_handle;
      }
      T await_resume() {
        if (handle.promise().exception) {
          std::rethrow_exception(handle.promise().exception);
        }
        return std::move(handle.promise().value);
      }

      std::coroutine_handle<promise_type> handle = nullptr;
    };

    template <typename T, typename suspend_type_t>
    task_value_wrap_t<T, suspend_type_t> task_value_promise_t<T, suspend_type_t>::get_return_object() {
      return task_value_wrap_t<T, suspend_type_t>{std::coroutine_handle<task_value_promise_t<T, suspend_type_t>>::from_promise(*this)};
    }

    template<typename suspend_type_t>
    struct task_value_wrap_t<void, suspend_type_t> {
      using promise_type = task_value_promise_t<void, suspend_type_t>;

      task_value_wrap_t() {}
      task_value_wrap_t(std::coroutine_handle<promise_type> ch) : handle(ch) {}
      task_value_wrap_t(const task_value_wrap_t&) = delete;
      task_value_wrap_t(task_value_wrap_t&& other) : handle(other.handle) {
        other.handle = nullptr;
      }
      task_value_wrap_t& operator=(task_value_wrap_t&& other) {
        destroy();
        std::swap(handle, other.handle);
        return *this;
      }
      void destroy() {
        if (handle) {
          handle.destroy();
          handle = nullptr;
        }
      }
      ~task_value_wrap_t() {
        if (handle) {
          handle.destroy();
        }
      }
      bool valid() const {
        return static_cast<bool>(handle);
      }
      bool await_ready() const noexcept {
        return handle.done();
      }
      void await_suspend(std::coroutine_handle<> continuation_handle) noexcept {
        handle.promise().continuation = continuation_handle;
      }
      void await_resume() {
        if (handle.promise().exception) {
          std::rethrow_exception(handle.promise().exception);
        }
      }

      std::coroutine_handle<promise_type> handle = nullptr;
    };

    template <typename suspend_type_t>
    task_value_wrap_t<void, suspend_type_t> task_value_promise_t<void, suspend_type_t>::get_return_object() {
      return task_value_wrap_t<void, suspend_type_t>{std::coroutine_handle<task_value_promise_t<void, suspend_type_t>>::from_promise(*this)};
    }

    using task_suspend_t = task_value_wrap_t<void, std::suspend_always>;
    using task_resume_t = task_value_wrap_t<void, std::suspend_never>;

    template <typename T>
    using task_value_t = task_value_wrap_t<T, std::suspend_always>;

    template <typename T>
    using task_value_resume_t = task_value_wrap_t<T, std::suspend_never>;

    using task_t = task_resume_t;

    struct uv_fs_awaitable {
      uv_fs_t req;
      std::coroutine_handle<> handle;

      uv_fs_awaitable() {
        req.data = this;
      }

      ~uv_fs_awaitable() {
        uv_fs_req_cleanup(&req);
      }

      uv_fs_awaitable(const uv_fs_awaitable&) = delete;
      uv_fs_awaitable& operator=(const uv_fs_awaitable&) = delete;
      uv_fs_awaitable(uv_fs_awaitable&&) = delete;
      uv_fs_awaitable& operator=(uv_fs_awaitable&&) = delete;

      bool await_ready() const noexcept { return false; }

      void await_suspend(std::coroutine_handle<> h) noexcept {
        handle = h;
      }

      void await_resume() noexcept {}
    };

    struct uv_fs_open_awaitable : uv_fs_awaitable {
      uv_fs_open_awaitable(const std::string& path, int flags, int mode) {
        req.data = this;
        uv_fs_open(uv_default_loop(), &req, path.c_str(), flags, mode, on_open_cb);
      }

      static void on_open_cb(uv_fs_t* r) {
        auto self = static_cast<uv_fs_open_awaitable*>(r->data);
        if (self->handle) {
          self->handle.resume();
        }
      }

      int result() const noexcept {
        return req.result;
      }
    };

    struct uv_fs_read_awaitable : uv_fs_awaitable {
      uv_fs_read_awaitable(int file, uv_buf_t buf, int64_t offset) {
        int ret = uv_fs_read(uv_default_loop(), &req, file, &buf, 1, offset, on_read_cb);
        if (ret < 0) {
          fan::throw_error("uv_error"_str + uv_strerror(ret));
        }
      }

      static void on_read_cb(uv_fs_t* r) {
        auto self = static_cast<uv_fs_read_awaitable*>(r->data);
        self->handle.resume();
      }

      ssize_t result() const noexcept {
        return req.result;
      }
    };

    struct uv_fs_close_awaitable : uv_fs_awaitable {
      uv_fs_close_awaitable(int file) {
        req.data = this;
        uv_fs_close(uv_default_loop(), &req, file, on_close_cb);
      }

      static void on_close_cb(uv_fs_t* r) {
        auto self = static_cast<uv_fs_close_awaitable*>(r->data);
        self->handle.resume();
      }
    };


    struct fs_watcher_t {
      struct file_event {
        std::string filename;
        int events;
        std::chrono::steady_clock::time_point timestamp;
      };

      uv_fs_event_t fs_event;
      uv_loop_t* loop;
      uv_timer_t timer;
      std::string watch_path;
      std::function<void(const std::string&, int)> event_callback;
      std::unordered_map<std::string, file_event> pending_events;

      static void timer_callback(uv_timer_t* handle) {
        fs_watcher_t* watcher = static_cast<fs_watcher_t*>(handle->data);
        watcher->process_latest_events();
      }

      static void fs_event_callback(uv_fs_event_t* handle, const char* filename, int events, int status) {
        if (status < 0) return;

        fs_watcher_t* watcher = static_cast<fs_watcher_t*>(handle->data);

        if (filename) {
          std::string file_str(filename);
          auto now = std::chrono::steady_clock::now();

          watcher->pending_events[file_str] = {
              file_str,
              events,
              now
          };
        }
      }

      void process_latest_events() {
        for (auto& event_pair : pending_events) {
          if (event_callback) {
            event_callback(event_pair.second.filename, event_pair.second.events);
          }
        }
        pending_events.clear();
      }

      fs_watcher_t(uv_loop_t* event_loop, const std::string& path)
        : loop(event_loop), watch_path(path) {
        fs_event.data = this;
        timer.data = this;
      }

      bool start(std::function<void(const std::string&, int)> callback) {
        event_callback = callback;

        int result = uv_fs_event_init(loop, &fs_event);
        if (result < 0) return false;

        result = uv_fs_event_start(&fs_event, fs_event_callback,
          watch_path.c_str(), UV_FS_EVENT_RECURSIVE);
        if (result < 0) return false;

        result = uv_timer_init(loop, &timer);
        if (result < 0) return false;

        result = uv_timer_start(&timer, timer_callback, 0, 50);
        if (result < 0) return false;

        return true;
      }

      void stop() {
        uv_fs_event_stop(&fs_event);
        uv_timer_stop(&timer);
      }
    };

    fan::event::task_t timer_task(uint64_t time, auto l) {
      while (true) {
        if (l()) {
          break;
        }
        co_await event::timer_t(time);
      }
    }

    //thread stuff
    using thread_id_t = uv_thread_t;

    template <typename cb_t, typename ...args_t>
    struct thread_task_t {
      cb_t cb;
      std::tuple<args_t...> args;
      thread_task_t(cb_t&& cb, args_t&&... args) :
        cb(std::forward<cb_t>(cb)),
        args(std::forward<args_t>(args)...) { }
      void operator()() {
        std::apply(cb, args);
      }
    };

    template <typename cb_t, typename ...args_t>
    thread_id_t thread_create(cb_t&& cb, args_t&&... args, int* error = 0) {
      thread_id_t id;
      thread_task_t<cb_t, args_t...>* cb_copy = new thread_task_t<cb_t, args_t...>(
        std::forward<cb_t>(cb),
        std::forward<args_t>(args)...
      );
      int uv_error = uv_thread_create(&id, [](void* arg){
        auto* task = static_cast<thread_task_t<cb_t, args_t...>*>(arg);
        (*task)();
        delete task;
      }, cb_copy);
      if (error && uv_error < 0) {
        *error = uv_error;
        delete cb_copy;
        fan::print_warning(std::string("thread create error:") + uv_strerror(*error));
      }
      return id;
    }
    void sleep(unsigned int msec) {
      uv_sleep(msec);
    }
    void loop() {
      uv_run(uv_default_loop(), UV_RUN_DEFAULT);
    }
  }
  using co_sleep = event::timer_t;
}

export namespace fan {
  namespace io {
    namespace file {
      fan::event::task_value_resume_t<int> async_open(const std::string& path) {
        fan::event::uv_fs_open_awaitable open_req(path, O_RDONLY, 0);
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
      fan::event::task_value_resume_t<ssize_t> async_read(int file, std::string* buffer, int64_t offset, unsigned int buffer_size = 4096) {
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

      fan::event::task_t async_close(int file) {
        fan::event::uv_fs_close_awaitable close_req(file);
        co_await close_req;
      }

      /*
      requires cb to be async
      fan::event::task_value_t<int64_t> async_size(int file) {
        uv_fs_t stat_req;
        int r = uv_fs_fstat(uv_default_loop(), &stat_req, file, NULL);
        if (r < 0) {
          co_return -1;
        }
        co_return stat_req.statbuf.st_size;
      }

      fan::event::task_value_t<int64_t> async_size(const std::string& path) {
        uv_fs_t stat_req;
        int r = uv_fs_stat(uv_default_loop(), &stat_req, path.c_str(), NULL);
        if (r < 0) {
          co_return -1;
        }
        co_return stat_req.statbuf.st_size;
      }*/

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

          if constexpr (is_awaitable_v<decltype(lambda(chunk))>) {
            co_await lambda(chunk);
          }
          else {
            lambda(chunk);
          }

          offset += result;
        }

        co_await fan::io::file::async_close(fd);
      }
    }
  }
}