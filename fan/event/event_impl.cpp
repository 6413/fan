module;

#include <coroutine>
#include <functional>
#include <chrono>
#include <filesystem>
#include <algorithm>

#include <uv.h>
#undef min
#undef max

module fan.event;

namespace fan::event {
  loop_t loop_new() {
    return uv_loop_new();
  }
  loop_t& get_loop() {
    static loop_t event_loop = uv_default_loop();
    return event_loop;
  }
  void loop_stop(loop_t loop = fan::event::get_loop()) {
    uv_stop(loop);
  }
  int loop_close(loop_t loop = fan::event::get_loop()) {
    return uv_loop_close(loop);
  }

  void error_code_t::throw_if() const { if (code) throw code; }
  void error_code_t::await_resume() const { throw_if(); }

  void print_event_handles(loop_t loop) {
    fan::print("========================");
    fan::print("Active handles:", loop->active_handles);
    fan::print("Active requests:", loop->active_reqs.count);
    uv_walk(loop, [](uv_handle_t* handle, void* arg) {
      const char* type_name = uv_handle_type_name(handle->type);
      fan::print("Handle:", type_name, "active:", uv_is_active(handle), "closing:", uv_is_closing(handle));
      }, nullptr);
    fan::print("========================");
  }

  timer_t::timer_data::timer_data() : co_handle(nullptr), ready(0) {}

  void timer_t::timer_deleter::operator()(timer_t::timer_data* data) const noexcept {
    uv_close(reinterpret_cast<uv_handle_t*>(&data->timer_handle), [](uv_handle_t* timer_handle) {
      delete static_cast<timer_t::timer_data*>(timer_handle->data);
      });
  }

  timer_t::timer_t() : data(new timer_data{}, timer_deleter{}) {
    uv_timer_init(fan::event::get_loop(), &data->timer_handle);
    data->timer_handle.data = data.get();
  }

  timer_t::timer_t(uint64_t timeout, uint64_t repeat)
    : data(new timer_data{}, timer_deleter{}) {
    uv_timer_init(fan::event::get_loop(), &data->timer_handle);
    data->timer_handle.data = data.get();
    start(timeout, repeat);
  }

  error_code_t timer_t::start(uint64_t timeout, uint64_t repeat) noexcept {
    return uv_timer_start(&data->timer_handle, [](uv_timer_t* timer_handle) {
      auto data = static_cast<timer_t::timer_data*>(timer_handle->data);
      ++data->ready;
      if (data->co_handle) {
        data->co_handle();
      }
      }, timeout, repeat);
  }

  error_code_t timer_t::again() noexcept {
    return uv_timer_again(&data->timer_handle);
  }

  void timer_t::set_repeat(uint64_t repeat) noexcept {
    uv_timer_set_repeat(&data->timer_handle, repeat);
  }

  error_code_t timer_t::stop() noexcept {
    return uv_timer_stop(&data->timer_handle);
  }

  bool timer_t::await_ready() const noexcept { return data->ready; }

  void timer_t::await_suspend(std::coroutine_handle<> h) noexcept { data->co_handle = h; }

  void timer_t::await_resume() noexcept {
    data->co_handle = nullptr;
    --data->ready;
  }
  void deferred_resume_t::process_resumes() {
    for (auto& e : resume_queue) {
      if (e.h) {
        e.h.resume();
      }
    }
    resume_queue.clear();
  }

  idle_task::idle_data::idle_data(std::function<task_t()> cb)
    : callback(std::move(cb)), has_task(false) {
  }

  void idle_task::idle_callback(uv_idle_t* handle) {
    auto* data = static_cast<idle_data*>(handle->data);
    if (!data->has_task) {
      data->task = data->callback();
      data->has_task = true;
    }
    if (data->task.await_ready()) {
      data->task.await_resume();
      data->has_task = false;
    }
  }

  void idle_task::close_callback(uv_handle_t* handle) {
    auto* idle_handle = reinterpret_cast<uv_idle_t*>(handle);
    delete static_cast<idle_data*>(idle_handle->data);
    delete idle_handle;
  }

  idle_id_t idle_task::task_idle(std::function<task_t()> callback) {
    auto* idle_handle = new uv_idle_t;
    idle_handle->data = new idle_data(std::move(callback));
    if (uv_idle_init(get_loop(), idle_handle) != 0 ||
      uv_idle_start(idle_handle, idle_callback) != 0) {
      delete static_cast<idle_data*>(idle_handle->data);
      delete idle_handle;
      return nullptr;
    }
    return idle_handle;
  }

  void idle_task::idle_stop(idle_id_t idle_handle) {
    if (idle_handle) {
      uv_idle_stop(idle_handle);
      uv_close(reinterpret_cast<uv_handle_t*>(idle_handle), close_callback);
    }
  }

  idle_id_t task_idle(std::function<task_t()> callback) {
    return idle_task::task_idle(std::move(callback));
  }

  void idle_stop(idle_id_t idle_handle) {
    idle_task::idle_stop(idle_handle);
  }

  bool counter_awaitable_t::await_ready() const noexcept {
    return st->remaining == 0;
  }

  void counter_awaitable_t::await_suspend(std::coroutine_handle<> h) noexcept {
    st->continuation = h;
  }

  void counter_awaitable_t::await_resume() const noexcept {}

  template<typename T>
  void signal_awaitable_t<T>::signal(T val) {
    value = std::move(val);
    ready = true;
    if (waiting_coroutine) {
      waiting_coroutine.resume();
    }
  }

  template<typename T>
  bool signal_awaitable_t<T>::await_ready() const {
    return ready;
  }

  template<typename T>
  void signal_awaitable_t<T>::await_suspend(std::coroutine_handle<> h) {
    waiting_coroutine = h;
  }

  template<typename T>
  T signal_awaitable_t<T>::await_resume() const {
    return value;
  }

  void signal_awaitable_t<void>::signal() {
    ready = true;
    if (waiting_coroutine) {
      waiting_coroutine.resume();
    }
  }

  bool signal_awaitable_t<void>::await_ready() const {
    return ready;
  }

  void signal_awaitable_t<void>::await_suspend(std::coroutine_handle<> h) {
    waiting_coroutine = h;
  }

  void signal_awaitable_t<void>::await_resume() const {}

  uv_fs_awaitable::uv_fs_awaitable() { req.data = this; }
  uv_fs_awaitable::~uv_fs_awaitable() { uv_fs_req_cleanup(&req); closed = true; }
  bool uv_fs_awaitable::await_ready() const noexcept { return false; }
  void uv_fs_awaitable::await_suspend(std::coroutine_handle<> h) noexcept { handle = h; }
  void uv_fs_awaitable::await_resume() noexcept {}

  uv_fs_open_awaitable::uv_fs_open_awaitable(const std::string& path, int flags, int mode) {
    req.data = this;
    uv_fs_open(get_loop(), &req, path.c_str(), flags, mode, on_open_cb);
  }
  void uv_fs_open_awaitable::on_open_cb(uv_fs_t* r) {
    auto self = static_cast<uv_fs_open_awaitable*>(r->data);
    if (self->closed) {
      return;
    }
    if (self->handle) {
      self->handle.resume();
    }
  }
  int uv_fs_open_awaitable::result() const noexcept { return req.result; }

  uv_fs_size_awaitable::uv_fs_size_awaitable(int file) {
    uv_fs_fstat(get_loop(), &req, file, on_size_cb);
  }
  uv_fs_size_awaitable::uv_fs_size_awaitable(const std::string& path) {
    uv_fs_stat(fan::event::get_loop(), &req, path.c_str(), on_size_cb);
  }
  void uv_fs_size_awaitable::on_size_cb(uv_fs_t* r) {
    auto self = static_cast<uv_fs_size_awaitable*>(r->data);
    if (self->closed) {
      return;
    }
    if (self->handle) {
      self->handle.resume();
    }
  }
  int64_t uv_fs_size_awaitable::result() const noexcept {
    return req.result < 0 ? -1 : req.statbuf.st_size;
  }

  uv_fs_read_awaitable::uv_fs_read_awaitable(int file, uv_buf_t buf, int64_t offset) {
    int ret = uv_fs_read(fan::event::get_loop(), &req, file, &buf, 1, offset, on_read_cb);
    if (ret < 0) {
      fan::throw_error("uv_error"_str + uv_strerror(ret));
    }
  }
  void uv_fs_read_awaitable::on_read_cb(uv_fs_t* r) {
    auto self = static_cast<uv_fs_read_awaitable*>(r->data);
    if (self->closed) {
      return;
    }
    self->handle.resume();
  }
  ssize_t uv_fs_read_awaitable::result() const noexcept { return req.result; }

  uv_fs_write_awaitable::uv_fs_write_awaitable(int fd, const char* buffer, size_t length, int64_t offset) {
    req.data = this;
    uv_buf_t buf = uv_buf_init(const_cast<char*>(buffer), length);
    uv_fs_write(fan::event::get_loop(), &req, fd, &buf, 1, offset, on_write_cb);
  }
  void uv_fs_write_awaitable::on_write_cb(uv_fs_t* req) {
    auto self = static_cast<uv_fs_write_awaitable*>(req->data);
    if (self->closed) {
      return;
    }
    self->handle.resume();
  }
  ssize_t uv_fs_write_awaitable::result() const noexcept { return req.result; }

  uv_fs_close_awaitable::uv_fs_close_awaitable(int file) {
    req.data = this;
    uv_fs_close(fan::event::get_loop(), &req, file, on_close_cb);
  }
  void uv_fs_close_awaitable::on_close_cb(uv_fs_t* r) {
    auto self = static_cast<uv_fs_close_awaitable*>(r->data);
    if (self->closed) {
      return;
    }
    self->handle.resume();
  }

  void fs_watcher_t::timer_callback(uv_timer_t* handle) {
    fs_watcher_t* watcher = static_cast<fs_watcher_t*>(handle->data);
    watcher->process_latest_events();
  }

  void fs_watcher_t::fs_callback(uv_fs_event_t* handle, const char* filename, int events, int status) {
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

  void fs_watcher_t::process_latest_events() {
    for (auto& event_pair : pending_events) {
      if (event_callback) {
        event_callback(event_pair.second.filename, event_pair.second.events);
      }
    }
    pending_events.clear();
  }

  fs_watcher_t::fs_watcher_t(const std::string& path)
    : loop(fan::event::get_loop()), watch_path(path) {
    fs_event.data = this;
    timer.data = this;
  }

  bool fs_watcher_t::start(std::function<void(const std::string&, int)> callback) {
    event_callback = callback;
    int result = uv_fs_event_init(loop, &fs_event);
    if (result < 0) return false;
    result = uv_fs_event_start(&fs_event, fs_callback, watch_path.c_str(), UV_FS_EVENT_RECURSIVE);
    if (result < 0) return false;
    result = uv_timer_init(loop, &timer);
    if (result < 0) return false;
    result = uv_timer_start(&timer, timer_callback, 0, 50);
    if (result < 0) return false;
    return true;
  }

  void fs_watcher_t::stop() {
    uv_fs_event_stop(&fs_event);
    uv_timer_stop(&timer);
  }

  fan::event::task_value_resume_t<void> wait_fd(loop_t loop, int fd, int events) {
    fd_waiter_t waiter;
    waiter.poll_handle.data = &waiter;
    int result = uv_poll_init(loop, &waiter.poll_handle, fd);
    if (result != 0) {
      throw std::runtime_error("Failed to init poll handle");
    }
    result = uv_poll_start(&waiter.poll_handle, events, [](uv_poll_t* handle, int status, int events) {
      auto* waiter = static_cast<fd_waiter_t*>(handle->data);
      waiter->ready = true;
      waiter->events_received = events;
      uv_poll_stop(handle);
      if (waiter->co_handle) {
        waiter->co_handle();
      }
      });
    if (result != 0) {
      throw std::runtime_error("Failed to start poll");
    }
    if (!waiter.ready) {
      struct awaiter {
        fd_waiter_t* w;
        bool await_ready() const { return w->ready; }
        void await_suspend(std::coroutine_handle<> h) { w->co_handle = h; }
        void await_resume() const {}
      };
      co_await awaiter{ &waiter };
    }
    uv_close(reinterpret_cast<uv_handle_t*>(&waiter.poll_handle), nullptr);
  }

  void sleep(unsigned int msec) {
    uv_sleep(msec);
  }

  void loop(fan::event::loop_t loop, bool once) {
    uv_run(loop, once ? UV_RUN_ONCE : UV_RUN_DEFAULT);
  }

  uint64_t now() {
    return uv_now(get_loop()) * 1000000;
  }

  std::string strerror(int err) {
    return uv_strerror(err);
  }
}

namespace fan::io {
  bool ev_next_tick_awaiter::await_ready() noexcept { return false; }

  void ev_next_tick_awaiter::await_suspend(std::coroutine_handle<> h) noexcept {
    coro = h;
    idle_handle = new uv_idle_t;
    uv_idle_init(fan::event::get_loop(), idle_handle);
    idle_handle->data = this;
    uv_idle_start(idle_handle, [](uv_idle_t* handle) {
      auto* awaiter = static_cast<ev_next_tick_awaiter*>(handle->data);
      uv_idle_stop(handle);
      uv_close(reinterpret_cast<uv_handle_t*>(handle), [](uv_handle_t* h) {
        delete reinterpret_cast<uv_idle_t*>(h);
        });
      awaiter->coro.resume();
      });
  }

  void ev_next_tick_awaiter::await_resume() noexcept {}

  void async_directory_iterator_t::stop() {
    stopped = true;
    switch_requested = false;
  }

  fan::event::task_t iterate_directory(async_directory_iterator_t* state) {
    while (state->current_index < state->entries.size()) {
      if (state->stopped) co_return;
      co_await state->callback(state->entries[state->current_index]);
      ++state->current_index;
      if (state->stopped) co_return;
      co_await ev_next_tick_awaiter{};
    }
    co_return;
  }

  void async_directory_iterate(async_directory_iterator_t* state, const std::string& path) {
    if (state->operation_in_progress) {
      state->stopped = true;
      state->switch_requested = true;
      state->next_path = path;
      return;
    }
    state->operation_in_progress = true;
    state->stopped = false;
    state->switch_requested = false;
    state->base_path = path;
    state->entries.clear();
    state->current_index = 0;
    uv_fs_t* req = new uv_fs_t;
    std::memset(req, 0, sizeof(uv_fs_t));
    req->data = state;
    int ret = uv_fs_scandir(fan::event::get_loop(), req, path.c_str(), 0, [](uv_fs_t* req) {
      auto* state = static_cast<async_directory_iterator_t*>(req->data);
      if (!state->stopped) {
        uv_dirent_t ent;
        while (uv_fs_scandir_next(req, &ent) != fan::eof) {
          std::filesystem::path full_path = std::filesystem::path(state->base_path) / ent.name;
          try {
            state->entries.emplace_back(full_path);
          }
          catch (...) {}
        }
        if (state->sort_alphabetically) {
          std::sort(state->entries.begin(), state->entries.end(),
            [](const std::filesystem::directory_entry& a, const std::filesystem::directory_entry& b) -> bool {
              if (a.is_directory() == b.is_directory()) {
                std::string a_stem = a.path().stem().string();
                std::string b_stem = b.path().stem().string();
                std::transform(a_stem.begin(), a_stem.end(), a_stem.begin(),
                  [](unsigned char c) { return std::tolower(c); });
                std::transform(b_stem.begin(), b_stem.end(), b_stem.begin(),
                  [](unsigned char c) { return std::tolower(c); });
                return a_stem < b_stem;
              }
              return a.is_directory() && !b.is_directory();
            }
          );
        }
        if (!state->stopped) {
          state->iteration_task = iterate_directory(state);
        }
      }
      uv_fs_req_cleanup(req);
      delete req;
      state->operation_in_progress = false;
      if (state->switch_requested) {
        std::string new_path = state->next_path;
        state->switch_requested = false;
        uv_idle_t* idle = new uv_idle_t;
        idle->data = state;
        uv_idle_init(fan::event::get_loop(), idle);
        uv_idle_start(idle, [](uv_idle_t* handle) {
          auto* state = static_cast<async_directory_iterator_t*>(handle->data);
          std::string path = state->next_path;
          uv_idle_stop(handle);
          uv_close(reinterpret_cast<uv_handle_t*>(handle), [](uv_handle_t* h) {
            delete reinterpret_cast<uv_idle_t*>(h);
            });
          async_directory_iterate(state, path);
        });
      }
    });
    if (ret < 0) {
      delete req;
      state->operation_in_progress = false;
      fan::throw_error("error fs_scandir:"_str + fan::event::strerror(ret));
    }
  }
}

cleaner_t::~cleaner_t() {
  uv_loop_close(fan::event::get_loop());
}