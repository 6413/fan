#include <fan/types/types.h>

#include <iostream>
#include <queue>
#include <chrono>
#include <coroutine>

template<typename T2, typename T = std::conditional_t<std::is_same_v<T2, void>, uint8_t, T2>>
struct task_t {
  task_t() = default;
  T value;
  bool m_resume = false;
  struct promise_type;
  using handle = std::coroutine_handle<promise_type>;

  operator T&() {
    return value;
  }

  struct promise_type {
    T value;
    void return_value(T v) { value = v; }
    std::suspend_never initial_suspend() { return {}; }
    std::suspend_always final_suspend() noexcept { return {}; }
    task_t get_return_object() {
      auto h = handle::from_promise(*this);
      return task_t{h, h.promise().value};
    }
    void unhandled_exception() {}
  };

  bool await_ready() { return m_resume; }
  void await_suspend(std::coroutine_handle<>) {}
  void await_resume() {}

  task_t(task_t const&) = delete;
  task_t(task_t&& rhs) : coro(rhs.coro), value(rhs.value) { rhs.coro = nullptr; }
  ~task_t() { if (coro) coro.destroy(); }

//private:
  handle coro;
  explicit task_t(handle h, T v) : coro(h), value(v) {}
};

task_t<void> co_suspend() {
  task_t<void> t;
  t.m_resume = false;
  co_return t;
}

task_t<void> co_resume() {
  task_t<void> t;
  t.m_resume = true;
  co_return t;
}

task_t<void> f() {
  fan::print("0");
  co_await co_suspend();
  fan::print("1");
}

int main() {

  auto co = f();
  fan::print("2");
  co.coro.resume();
  fan::print("3");
  
}
