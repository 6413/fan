#pragma once
#include <coroutine>
#include <functional>
#include <queue>
#include <chrono>
#include <memory>

#include <uv.h>
#undef min
#undef max

using namespace std::chrono_literals;

namespace fan {
  namespace ev {
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
        std::coroutine_handle<> co_handle;
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

      timer_t(uint64_t  timeout, uint64_t repeat = 0)
        : data(new timer_data{}, timer_deleter{}) {
        uv_timer_init(uv_default_loop(), &data->timer_handle);
        data->timer_handle.data = data.get();
        start(timeout, repeat);
      }

      error_code_t start(uint64_t  timeout, uint64_t repeat = 0) noexcept {
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

    struct task_promise_t {
      std::coroutine_handle<> _continuation;
      std::exception_ptr _exception;

      task_promise_t() : _continuation{ std::noop_coroutine() } { }
      task_promise_t(const task_promise_t&) = delete;
      task_promise_t(task_promise_t&&) = default;

      std::coroutine_handle<task_promise_t> get_return_object() noexcept {
        return std::coroutine_handle<task_promise_t>::from_promise(*this);
      }
      std::coroutine_handle<> continuation() const noexcept { return _continuation; }
      void set_continuation(std::coroutine_handle<> h) noexcept { _continuation = h; }
      std::suspend_never initial_suspend() { return {}; }
      auto final_suspend() noexcept {
        struct final_awaiter_t {
          bool await_ready() const noexcept { return false; }
          auto await_suspend(std::coroutine_handle<task_promise_t> ch) const noexcept { return ch.promise()._continuation; }
          void await_resume() const noexcept {}
        };
        return final_awaiter_t{};
      }
      void unhandled_exception() { _exception = std::current_exception(); }
      void return_void() {}
    };

    struct task_t {
      using promise_type = task_promise_t;
      task_t() {}
      task_t(std::coroutine_handle<task_promise_t> ch) : _handle(ch) { }
      task_t(const task_t&) = delete;
      task_t(task_t&& f) : _handle(f._handle) {
        f._handle = nullptr;
      }
      task_t& operator=(task_t&& f) {
        destroy();
        std::swap(_handle, f._handle);
        return *this;
      }
      void destroy() {
        if (_handle) {
          _handle.destroy();
          _handle = nullptr;
        }
      }
      ~task_t() { if (_handle) { _handle.destroy(); } }

      bool valid() const { return static_cast<bool>(_handle); }
      bool await_ready() const noexcept {
        return _handle.done();
      }
      void await_suspend(std::coroutine_handle<> ch) noexcept {
        _handle.promise()._continuation = ch;
      }
      void await_resume() {
        if (_handle.promise()._exception) {
          std::rethrow_exception(_handle.promise()._exception);
        }
      }

      std::coroutine_handle<task_promise_t> _handle = nullptr;
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
    fan::ev::task_t timer_task(uint64_t time, auto l) {
      while (true) {
        if (l()) {
          break;
        }
        co_await ev::timer_t(time);
      }
    }
  }
  using co_sleep = ev::timer_t;
}