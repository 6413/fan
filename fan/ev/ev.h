#pragma once
#include <coroutine>
#include <functional>
#include <queue>
#include <chrono>
#include <thread> // unnecessary

using namespace std::chrono_literals;

template <typename T>
struct task;

template <typename T>
struct task_promise {
  T value;
  std::exception_ptr exception;

  task<T> get_return_object();

  std::suspend_always initial_suspend() { return {}; }
  std::suspend_always final_suspend() noexcept { return {}; }

  void return_value(T v) { value = v; }

  void unhandled_exception() { exception = std::current_exception(); }

  task_promise() = default;
  task_promise(const task_promise&) = delete;
  task_promise& operator=(const task_promise&) = delete;
  task_promise(task_promise&&) = default;
  task_promise& operator=(task_promise&&) = default;
};

template <typename T>
struct task {
  using promise_type = task_promise<T>;
  std::coroutine_handle<promise_type> coro;

  task(std::coroutine_handle<promise_type> h) : coro(h) {}
  ~task() { if (coro) coro.destroy(); }

  task(const task&) = delete;
  task& operator=(const task&) = delete;

  task(task&& other) noexcept : coro(other.coro) {
    other.coro = nullptr;
  }

  task& operator=(task&& other) noexcept {
    if (this != &other) {
      if (coro) {
        coro.destroy();
      }
      coro = other.coro;
      other.coro = nullptr;
    }
    return *this;
  }

  bool await_ready() const noexcept { return false; }
  void await_suspend(std::coroutine_handle<> h) noexcept { coro.resume(); }
  T await_resume() {
    if (coro.promise().exception) {
      std::rethrow_exception(coro.promise().exception);
    }
    return coro.promise().value;
  }
};

template <typename T>
task<T> task_promise<T>::get_return_object() {
  return task<T>{std::coroutine_handle<task_promise>::from_promise(*this)};
}

template <>
struct task_promise<void> {
  std::exception_ptr exception;

  task<void> get_return_object();

  std::suspend_always initial_suspend() { return {}; }
  std::suspend_always final_suspend() noexcept { return {}; }

  void return_void() {}

  void unhandled_exception() { exception = std::current_exception(); }

  task_promise() = default;
  task_promise(const task_promise&) = delete;
  task_promise& operator=(const task_promise&) = delete;
  task_promise(task_promise&&) = default;
  task_promise& operator=(task_promise&&) = default;
};

template <>
struct task<void> {
  using promise_type = task_promise<void>;
  std::coroutine_handle<promise_type> coro;

  task() = default;
  task(std::coroutine_handle<promise_type> h) : coro(h) {}
  ~task() { if (coro) coro.destroy(); }

  task(const task&) = delete;
  task& operator=(const task&) = delete;

  task(task&& other) noexcept : coro(other.coro) {
    other.coro = nullptr;
  }

  task& operator=(task&& other) noexcept {
    if (this != &other) {
      if (coro) {
        coro.destroy();
      }
      coro = other.coro;
      other.coro = nullptr;
    }
    return *this;
  }

  bool await_ready() const noexcept { return false; }
  void await_suspend(std::coroutine_handle<> h) noexcept { coro.resume(); }
  void await_resume() {
    if (coro.promise().exception) {
      std::rethrow_exception(coro.promise().exception);
    }
  }
};

task<void> task_promise<void>::get_return_object() {
  return task<void>{std::coroutine_handle<task_promise>::from_promise(*this)};
}


struct uv_sleep_awaitable {
  uv_timer_t timer_req;
  std::coroutine_handle<> handle;
  task<void>* parent_task;
  bool completed{ false };

  uv_sleep_awaitable(task<void>* parent, uv_loop_t* loop, uint64_t delay)
    : parent_task(parent) {
    timer_req.data = this;
    uv_timer_init(loop, &timer_req);
    uv_timer_start(&timer_req, on_timer_cb, delay, 0);
  }

  ~uv_sleep_awaitable() {
    if (!completed) {
      uv_timer_stop(&timer_req);
      uv_close((uv_handle_t*)&timer_req, nullptr);
    }
  }

  static void on_timer_cb(uv_timer_t* timer) {
    auto* self = static_cast<uv_sleep_awaitable*>(timer->data);
    self->completed = true;
    self->handle.resume();
    // Resume the parent task as well
    if (self->parent_task) {
      self->parent_task->coro.resume();
    }
  }

  bool await_ready() const noexcept { return false; }
  void await_suspend(std::coroutine_handle<> h) noexcept {
    handle = h;
  }
  void await_resume() noexcept {}
};

task<void> co_sleep_for(task<void>* parent_task, std::chrono::milliseconds delay, uv_loop_t* loop = uv_default_loop()) {
  co_await uv_sleep_awaitable(parent_task, loop, delay.count());
}
