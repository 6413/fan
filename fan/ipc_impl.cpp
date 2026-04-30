module;
#include <uv.h>
#include <thread>
#include <coroutine>
#include <cstring>
#include <cstdio>

#include <fan/utility.h>
#if defined(fan_platform_windows)
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <semaphore.h>
#endif

module fan.ipc;
import fan.event;
import fan.print.error;

namespace fan::ipc {

#if defined(fan_platform_windows)

  struct shm_impl_t { HANDLE h; bool owner; };

  shared_memory_t::shared_memory_t(const char* name, size_t size, bool owner) : size_(size) {
    auto* im = new shm_impl_t;
    im->owner = owner;
    im->h = CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
      (DWORD)(size >> 32), (DWORD)(size & 0xFFFFFFFF), name);
    if (!im->h) { fan::throw_error_impl("shared_memory_t: CreateFileMapping failed"); }
    ptr = MapViewOfFile(im->h, FILE_MAP_ALL_ACCESS, 0, 0, size);
    if (!ptr) { fan::throw_error_impl("shared_memory_t: MapViewOfFile failed"); }
    impl_ = im;
  }
  shared_memory_t::~shared_memory_t() {
    auto* im = static_cast<shm_impl_t*>(impl_);
    if (ptr) { UnmapViewOfFile(ptr); }
    if (im && im->h) { CloseHandle(im->h); }
    delete im;
  }

  struct sem_impl_t { HANDLE h; bool owner; };

  static sem_impl_t* sem_open_impl(const char* name, bool owner) {
    auto* im = new sem_impl_t;
    im->owner = owner;
    im->h = CreateSemaphoreA(nullptr, 0, LONG_MAX, name);
    if (!im->h) { fan::throw_error_impl("sem: CreateSemaphore failed"); }
    return im;
  }
  static void sem_signal(sem_impl_t* im) { ReleaseSemaphore(im->h, 1, nullptr); }
  static void sem_wait_impl(sem_impl_t* im) { WaitForSingleObject(im->h, INFINITE); }
  static void sem_close_impl(sem_impl_t* im) { CloseHandle(im->h); delete im; }

#else

  struct shm_impl_t { int fd; char name[256]; bool owner; };

  shared_memory_t::shared_memory_t(const char* name, size_t size, bool owner) : size_(size) {
    auto* im = new shm_impl_t;
    im->owner = owner;
    snprintf(im->name, 256, "/%s", name);
    im->fd = shm_open(im->name, O_CREAT | O_RDWR, 0666);
    if (im->fd < 0) { fan::throw_error_impl("shared_memory_t: shm_open failed"); }
    if (owner) { ftruncate(im->fd, size); }
    ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, im->fd, 0);
    if (ptr == MAP_FAILED) { fan::throw_error_impl("shared_memory_t: mmap failed"); }
    impl_ = im;
  }
  shared_memory_t::~shared_memory_t() {
    auto* im = static_cast<shm_impl_t*>(impl_);
    if (ptr) { munmap(ptr, size_); }
    if (im) {
      close(im->fd);
      if (im->owner) { shm_unlink(im->name); }
      delete im;
    }
  }

  struct sem_impl_t { sem_t* s; char name[256]; bool owner; };

  static sem_impl_t* sem_open_impl(const char* name, bool owner) {
    auto* im = new sem_impl_t;
    im->owner = owner;
    snprintf(im->name, 256, "/%s", name);
    im->s = sem_open(im->name, O_CREAT, 0666, 0);
    if (im->s == SEM_FAILED) { fan::throw_error_impl("sem: sem_open failed"); }
    return im;
  }
  static void sem_signal(sem_impl_t* im) { sem_post(im->s); }
  static void sem_wait_impl(sem_impl_t* im) { sem_wait(im->s); }
  static void sem_close_impl(sem_impl_t* im) {
    sem_close(im->s);
    if (im->owner) { sem_unlink(im->name); }
    delete im;
  }

#endif

  struct consumer_impl_t {
    sem_impl_t* sem = nullptr;
    uv_async_t async {};
    std::thread watcher;
    std::coroutine_handle<> pending {};
    bool stop = false;
  };

  async_consumer_t::async_consumer_t(const char* event_name) {
    auto* im = new consumer_impl_t;
    im->sem = sem_open_impl(event_name, true);
    im->async.data = im;
    uv_async_init((uv_loop_t*)fan::event::get_loop(), &im->async, [](uv_async_t* h) {
      auto* im = static_cast<consumer_impl_t*>(h->data);
      if (im->pending) {
        auto h2 = im->pending;
        im->pending = {};
        h2.resume();
      }
    });
    im->watcher = std::thread([im] {
      while (!im->stop) {
        sem_wait_impl(im->sem);
        if (im->stop) { break; }
        uv_async_send(&im->async);
      }
    });
    impl_ = im;
  }

  async_consumer_t::~async_consumer_t() {
    auto* im = static_cast<consumer_impl_t*>(impl_);
    im->stop = true;
    sem_signal(im->sem);
    im->watcher.join();
    uv_close((uv_handle_t*)&im->async, [](uv_handle_t* h) {
      auto* im = static_cast<consumer_impl_t*>(h->data);
      sem_close_impl(im->sem);
      delete im;
    });
  }

  async_consumer_t::pop_awaitable_t async_consumer_t::wait() { return {this}; }

  void async_consumer_t::pop_awaitable_t::await_suspend(std::coroutine_handle<> h) {
    static_cast<consumer_impl_t*>(self->impl_)->pending = h;
  }

  struct producer_impl_t { sem_impl_t* sem = nullptr; };

  async_producer_t::async_producer_t(const char* event_name) {
    auto* im = new producer_impl_t;
    im->sem = sem_open_impl(event_name, false);
    impl_ = im;
  }
  async_producer_t::~async_producer_t() {
    auto* im = static_cast<producer_impl_t*>(impl_);
    sem_close_impl(im->sem);
    delete im;
  }
  void async_producer_t::signal() {
    sem_signal(static_cast<producer_impl_t*>(impl_)->sem);
  }

}