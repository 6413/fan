#pragma once

#include <coroutine>
#include <queue>
#include <chrono>

namespace fan {
  namespace ev {
    struct EventLoop {
      using Task = std::function<void()>;
      std::queue<Task> tasks;
      void run() {
        while (!tasks.empty()) {
          auto task_t = tasks.front();
          tasks.pop();
          task_t();
        }
      }
    };

    inline static EventLoop eventLoop;

    struct Timer {
      std::chrono::system_clock::time_point expires;
      std::function<void()> callback;
      bool operator<(const Timer& other) const {
        return expires > other.expires;
      }
    };

    inline static std::priority_queue<Timer> timers;

    static void add_timer(std::chrono::system_clock::duration duration, std::function<void()> callback) {
      timers.push(Timer{ std::chrono::system_clock::now() + duration, callback });
    }

    static void process_timers() {
      while (!timers.empty()) {
        auto timer = timers.top();
        if (timer.expires > std::chrono::system_clock::now()) break;
        timers.pop();
        timer.callback();
      }
    }

    struct task_t {
      task_t() = default;
      bool m_resume = false;
      struct promise_type {
        task_t get_return_object() { return task_t{ handle::from_promise(*this) }; }
        std::suspend_never initial_suspend() { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        void return_void() {}
        void unhandled_exception() {}
        std::coroutine_handle<> coro;
      };
      using handle = std::coroutine_handle<promise_type>;
      handle coro;
      explicit task_t(handle h) : coro(h) {}
      ~task_t() { if (coro) coro.destroy(); }

      bool await_ready() { return m_resume; } // add this method
      void await_suspend(std::coroutine_handle<>) {} // add this method
      void await_resume() {}
    };

    struct sleep_for {
      std::chrono::system_clock::duration duration;
      bool await_ready() { return duration.count() <= 0; }
      void await_suspend(std::coroutine_handle<> h) {
        add_timer(duration, [=] { h.resume(); });
        eventLoop.tasks.push(process_timers);
      }
      void await_resume() {}
    };

    static task_t co_resume() {
      task_t t;
      t.m_resume = true;
      return t;
    }
    static task_t co_suspend() {
      task_t t;
      t.m_resume = false;
      return t;
    }
  }
}