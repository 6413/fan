#include <fan/types/types.h>

#include <iostream>
#include <chrono>
#include <functional>
#include <queue>
#include <coroutine>

#include _FAN_PATH(event/event.h);

using namespace std::chrono_literals;

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

EventLoop eventLoop;

struct Timer {
  std::chrono::system_clock::time_point expires;
  std::function<void()> callback;
  bool operator<(const Timer& other) const {
      return expires > other.expires;
  }
};

std::priority_queue<Timer> timers;

void add_timer(std::chrono::system_clock::duration duration, std::function<void()> callback) {
    timers.push(Timer{std::chrono::system_clock::now() + duration, callback});
}

void process_timers() {
  while (!timers.empty()) {
    auto timer = timers.top();
    if (timer.expires > std::chrono::system_clock::now()) break;
    timers.pop();
    timer.callback();
  }
}

struct sleep_for {
    std::chrono::system_clock::duration duration;
    bool await_ready() { return duration.count() <= 0; }
    void await_suspend(std::coroutine_handle<> h) {
        add_timer(duration, [=] { h.resume(); });
        eventLoop.tasks.push(process_timers);
    }
    void await_resume() {}
};

struct task_t {
  task_t() = default;
  bool m_resume = false;
  struct promise_type {
    task_t get_return_object() { return task_t{handle::from_promise(*this)}; }
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

  static task_t resume() {
    task_t t;
    t.m_resume = true;
    return t;
  }
  static task_t suspend() {
    task_t t;
    t.m_resume = false;
    return t;
  }
};

fan::time::clock c;

task_t foo() {
  fan::print("foo");
  return task_t::suspend();
}

task_t bar() {
  co_await std::suspend_always{};
  c.start(fan::time::nanoseconds(1e+9));
  fan::print("a");
  co_await foo(); // how to remove co_await here and do it automatically based on foo return
  fan::print("b");
}

int main() {
  std::cout << "main() start\n";
  
  task_t t2 = bar();
  t2.coro.resume();
  while (!c.finished()) {

  }
  // continue bar execution
  fan::print("c");
  if (!t2.coro.done()) {
    t2.coro.resume();
  }
  while (!t2.coro.done()) {
    std::this_thread::sleep_for(1ms);
    process_timers();
  }
  std::cout << "main() end\n";
  return 0;
}