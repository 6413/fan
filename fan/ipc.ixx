module;

#include <coroutine>

export module fan.ipc;

import std;
import fan.event.types;

export namespace fan::ipc {

  struct shared_memory_t {
    shared_memory_t(const char* name, std::size_t size, bool owner);
    ~shared_memory_t();
    void* ptr = nullptr;
    void* impl_ = nullptr;
    std::size_t size_ = 0;
  };

  template<typename T, std::uint32_t N>
  struct ring_buffer_t {
    static_assert((N& (N - 1)) == 0);

    void push(const T& item) {
      std::uint32_t head = head_.load(std::memory_order_relaxed);
      while (head - tail_.load(std::memory_order_acquire) >= N) {}
      slots_[head & (N - 1)] = item;
      head_.store(head + 1, std::memory_order_release);
    }

    bool try_pop(T& out) {
      std::uint32_t tail = tail_.load(std::memory_order_relaxed);
      if (head_.load(std::memory_order_acquire) == tail) { return false; }
      out = slots_[tail & (N - 1)];
      tail_.store(tail + 1, std::memory_order_release);
      return true;
    }

    std::atomic<std::uint32_t> head_ {};
    std::atomic<std::uint32_t> tail_ {};
    T slots_[N] {};
  };

  struct async_consumer_t {
    async_consumer_t(const char* event_name);
    ~async_consumer_t();

    struct pop_awaitable_t {
      bool await_ready() const noexcept { return false; }
      void await_suspend(std::coroutine_handle<> h);
      void await_resume() const noexcept {}
      async_consumer_t* self;
    };

    pop_awaitable_t wait();
    void* impl_ = nullptr;
  };

  struct async_producer_t {
    async_producer_t(const char* event_name);
    ~async_producer_t();

    void signal();
    void* impl_ = nullptr;
  };

}