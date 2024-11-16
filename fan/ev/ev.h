#pragma once
#include <coroutine>
#include <functional>
#include <queue>
#include <chrono>
#include <thread> // unnecessary

using namespace std::chrono_literals;

struct event_loop_t {
  // Check if there are any events or tasks
  bool events_exist() {
    return !timers.empty() || !tasks.empty();
  }

  // Add a timer to the event loop
  void add_timer(std::chrono::steady_clock::time_point when, std::coroutine_handle<> handle) {
    timers.push({ when, handle });
  }

  // Schedule a task
  void schedule(std::function<void()> task) {
    tasks.push(std::move(task));
  }

  // Run one event or task
  bool run_one() {
    auto now = std::chrono::steady_clock::now();

    // Check and execute the next timer if it is ready
    if (!timers.empty() && timers.top().when <= now) {
      auto handle = timers.top().handle;
      timers.pop();
      handle.resume();
      return true;
    }

    // Execute the next scheduled task
    if (!tasks.empty()) {
      auto task = std::move(tasks.front());
      tasks.pop();
      task();
      return true;
    }

    return false;
  }

  // Run the event loop
  void run() {
    while (events_exist()) {
      if (!run_one()) {
        std::this_thread::sleep_for(10ms);  // Sleep to prevent busy waiting
      }
    }
  }

  struct timer_t {
    std::chrono::steady_clock::time_point when;
    std::coroutine_handle<> handle;
    bool operator>(const timer_t& other) const {
      return when > other.when;
    }
  };

  std::priority_queue<timer_t, std::vector<timer_t>, std::greater<timer_t>> timers;
  std::queue<std::function<void()>> tasks;
};

// Global event loop instance
inline event_loop_t& get_event_loop() {
  static event_loop_t event_loop;
  return event_loop;
}

template <typename T>
struct task_t {
  struct promise_type {
    using result_type_t = std::conditional_t<std::is_void_v<T>, __empty_struct, T>;
    result_type_t result;
    std::coroutine_handle<> continuation;

    task_t get_return_object() {
      auto handle = std::coroutine_handle<promise_type>::from_promise(*this);
      get_event_loop().schedule([handle]() { handle.resume(); });
      return task_t(handle);
    }

    std::suspend_always initial_suspend() { return {}; }

    std::suspend_always final_suspend() noexcept {
      if (continuation) {
        get_event_loop().schedule([h = continuation]() { h.resume(); });
      }
      return {};
    }

    void return_value(result_type_t value) { result = value; }

    void unhandled_exception() { std::terminate(); }
  };

  std::coroutine_handle<promise_type> handle;

  task_t(std::coroutine_handle<promise_type> h) : handle(h) {}

  ~task_t() {
    if (handle) {
      handle.destroy();
    }
  }

  task_t(const task_t&) = delete;
  task_t& operator=(const task_t&) = delete;

  task_t(task_t&& other) noexcept : handle(other.handle) {
    other.handle = nullptr;
  }

  task_t& operator=(task_t&& other) noexcept {
    if (this != &other) {
      if (handle) {
        handle.destroy();
      }
      handle = other.handle;
      other.handle = nullptr;
    }
    return *this;
  }

  bool await_ready() const { return false; }

  void await_suspend(std::coroutine_handle<> continuation_handle) {
    handle.promise().continuation = continuation_handle;
  }

  auto await_resume() { return handle.promise().result; }

  // Sleep for a specified duration
  struct sleep_awaitable {
    std::chrono::milliseconds duration;

    bool await_ready() const noexcept { return false; }

    void await_suspend(std::coroutine_handle<> handle) {
      auto ready_at = std::chrono::steady_clock::now() + duration;
      get_event_loop().add_timer(ready_at, handle);
    }

    void await_resume() const noexcept {}
  };
};

static task_t<void>::sleep_awaitable co_sleep_for(std::chrono::milliseconds duration) {
  return task_t<void>::sleep_awaitable{ duration };
}

struct yield_awaitable_t {
  bool await_ready() { return false; }
  void await_suspend(std::coroutine_handle<> h) {
    get_event_loop().schedule([h]() { h.resume(); });
  }
  void await_resume() {}
};

yield_awaitable_t resume_other_tasks() { return {}; }