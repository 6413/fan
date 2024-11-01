#undef loco_assimp
#include <fan/fmt.h>
#include <fan/math/random.h>
#include <chrono>
#include <coroutine>
#include <queue>
#include <thread>

using namespace std::chrono_literals;

struct event_loop_t;

event_loop_t* g_event_loop = nullptr;

struct event_loop_t {

  event_loop_t() {
    g_event_loop = this;
  }

  bool events_exist() {
    return !timers.empty();
  }

  void add_timer(std::chrono::steady_clock::time_point when, std::coroutine_handle<> handle) {
    timers.push({ when, handle });
  }

  bool run_one() {
    if (timers.empty()) return false;
    auto now = std::chrono::steady_clock::now();
    auto& timer = timers.top();
    if (timer.when <= now) {
      auto handle = timer.handle;
      timers.pop();
      handle.resume();
      return true;
    }
    return false;
  }

  void run() {
    while (events_exist()) {
      if (!run_one()) {
        std::this_thread::sleep_for(10ms);
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
};

struct task_t {
  struct promise_type {
    task_t get_return_object() {
      return task_t{ std::coroutine_handle<promise_type>::from_promise(*this) };
    }
    std::suspend_always initial_suspend() { return {}; }
    std::suspend_always final_suspend() noexcept { return {}; }
    void return_void() {}
    void unhandled_exception() {}
  };

  bool await_ready() const noexcept { return false; }

  void await_suspend(std::coroutine_handle<> handle) {
    auto ready_at = std::chrono::steady_clock::now() + duration_;
    g_event_loop->add_timer(ready_at, handle);
  }

  void await_resume() const noexcept {}

  inline static uint32_t task_counter{ 0 };
  uint32_t task_id;

  explicit task_t(std::chrono::milliseconds ms = std::chrono::milliseconds(0))
    : duration_(ms) { }

  task_t(std::coroutine_handle<promise_type> h) : handle_(h), duration_(0ms), task_id(++task_counter) { }

  ~task_t() {
    if (handle_) handle_.destroy();
  }

  void start() {
    if (handle_) handle_.resume();
  }

private:
  std::chrono::milliseconds duration_;
  std::coroutine_handle<promise_type> handle_{ nullptr };
};

task_t sleep_for(std::chrono::milliseconds duration) {
  return task_t(duration);
}


//-----------------------------------------------------------

template<typename T>
struct task_value_t {

  struct promise_type;
  using handle_type = std::coroutine_handle<promise_type>;

  struct promise_type // required
  {
    T value_;

    task_value_t get_return_object() {
      return task_value_t(handle_type::from_promise(*this));
    }
    std::suspend_always initial_suspend() { return {}; }
    std::suspend_always final_suspend() noexcept { return {}; }
    void unhandled_exception() { }

    template<std::convertible_to<T> From>
    std::suspend_always yield_value(From&& from) {
      value_ = std::forward<From>(from);
      return {};
    }
    void return_void() {}
  };

  handle_type h_;

  task_value_t(handle_type h) : h_(h) {}
  ~task_value_t() { h_.destroy(); }
  explicit operator bool() {
    fill();
    return !h_.done();
  }
  T operator()() {
    fill();
    full_ = false;
    return std::move(h_.promise().value_);
  }

private:
  bool full_ = false;

  void fill() {
    if (!full_) {
      h_();
      full_ = true;
    }
  }
};



//-----------------------------------------------------------


auto g_time_since_start = std::chrono::steady_clock::now();

using steady_clock_t = std::chrono::steady_clock;
using time_point_t = std::chrono::time_point<steady_clock_t>;

time_point_t time_since_start() {
  return time_point_t(steady_clock_t::now() - g_time_since_start);
}

constexpr const char* started_str = "called from: {}(), task id: {}, thread id:{}, started sleep at:{:.2f}s";
constexpr const char* slept_str = "called from: {}(), task id: {}, thread id:{}, slept:{:.2f}s, since start:{:.2f}";

#define print_time_and_thread_id \
  __FUNCTION__, \
  task->task_id, \
  std::hash<std::thread::id>{}(std::this_thread::get_id()), /*to get size_t*/ \
  now.time_since_epoch().count() / 1e9

#define print_sleep_time_and_thread_id \
  __FUNCTION__, \
  task->task_id, \
  std::hash<std::thread::id>{}(std::this_thread::get_id()), /*to get size_t*/ \
  std::chrono::duration<double>(sleep_time).count(), \
  now.time_since_epoch().count() / 1e9

task_t example(task_t* task) {
  auto now = time_since_start();
  auto sleep_time = std::chrono::milliseconds(fan::random::value_i64(300, 2000));
  fan::print_format(started_str, print_time_and_thread_id);
  co_await sleep_for(sleep_time);
  now = time_since_start();
  fan::print_format(slept_str, print_sleep_time_and_thread_id);
  sleep_time = std::chrono::milliseconds(fan::random::value_i64(300, 2000));
  co_await sleep_for(sleep_time);
  now = time_since_start();
  fan::print_format(slept_str, print_sleep_time_and_thread_id);
}

task_value_t<int> numbers(task_value_t<int>* task) {
  co_yield 1;
  fan::print("a");
  co_yield 2;
  fan::print("b");
  co_yield 3;
}

int main() {
  event_loop_t event_loop;
  task_t main_f = [&](task_t* main_task) -> task_t {

    task_value_t<int> get = numbers(&get);

    while (get) {
      fan::print(get());
    }

    task_t task = example(&task);
    task.start();
    task_t task2 = [](task_t* task) -> task_t {
      return example(task);
      }(&task2);
    task2.start();

    event_loop.run();

    co_return;
    }(&main_f);
  main_f.start();
  return 0;
}