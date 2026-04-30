module;

#include <coroutine>
#include <uv.h>
#undef min
#undef max

module fan.event;

import std;

import fan.print;
import fan.print.error;

namespace fan::event {

  struct awaitable_internal_t {
    uv_fs_t req;
    std::coroutine_handle<> handle;
  };
  static_assert(sizeof(awaitable_internal_t) <= 512, "Awaitable Shell buffer too small");

  uv_fs_open_awaitable::uv_fs_open_awaitable(const std::string& path, int flags, int mode) {
    auto* internal = reinterpret_cast<awaitable_internal_t*>(data);
    internal->handle = nullptr;
    internal->req.data = internal;
    uv_fs_open((uv_loop_t*)get_loop(), &internal->req, path.c_str(), flags, mode, [](uv_fs_t* r) {
      auto* in = static_cast<awaitable_internal_t*>(r->data);
      if (in->handle) in->handle.resume();
    });
  }
  uv_fs_open_awaitable::~uv_fs_open_awaitable() {
    uv_fs_req_cleanup(&reinterpret_cast<awaitable_internal_t*>(data)->req);
  }
  void uv_fs_open_awaitable::await_suspend(std::coroutine_handle<> h) noexcept {
    reinterpret_cast<awaitable_internal_t*>(data)->handle = h;
  }
  int uv_fs_open_awaitable::result() const noexcept {
    return reinterpret_cast<const awaitable_internal_t*>(data)->req.result;
  }

  uv_fs_read_awaitable::uv_fs_read_awaitable(int file, char* buffer, std::size_t size, std::int64_t offset) {
    auto* internal = reinterpret_cast<awaitable_internal_t*>(data);
    internal->handle = nullptr;
    internal->req.data = internal;
    uv_buf_t uv_buf = uv_buf_init(buffer, size);
    uv_fs_read((uv_loop_t*)get_loop(), &internal->req, file, &uv_buf, 1, offset, [](uv_fs_t* r) {
      auto* in = static_cast<awaitable_internal_t*>(r->data);
      if (in->handle) in->handle.resume();
    });
  }
  uv_fs_read_awaitable::~uv_fs_read_awaitable() {
    uv_fs_req_cleanup(&reinterpret_cast<awaitable_internal_t*>(data)->req);
  }
  void uv_fs_read_awaitable::await_suspend(std::coroutine_handle<> h) noexcept {
    reinterpret_cast<awaitable_internal_t*>(data)->handle = h;
  }
  std::intptr_t uv_fs_read_awaitable::result() const noexcept {
    return reinterpret_cast<const awaitable_internal_t*>(data)->req.result;
  }

  uv_fs_write_awaitable::uv_fs_write_awaitable(int fd, const char* buffer, std::size_t length, std::int64_t offset) {
    auto* internal = reinterpret_cast<awaitable_internal_t*>(data);
    internal->handle = nullptr;
    internal->req.data = internal;
    uv_buf_t uv_buf = uv_buf_init(const_cast<char*>(buffer), length);
    uv_fs_write((uv_loop_t*)get_loop(), &internal->req, fd, &uv_buf, 1, offset, [](uv_fs_t* r) {
      auto* in = static_cast<awaitable_internal_t*>(r->data);
      if (in->handle) in->handle.resume();
    });
  }
  uv_fs_write_awaitable::~uv_fs_write_awaitable() {
    uv_fs_req_cleanup(&reinterpret_cast<awaitable_internal_t*>(data)->req);
  }
  void uv_fs_write_awaitable::await_suspend(std::coroutine_handle<> h) noexcept {
    reinterpret_cast<awaitable_internal_t*>(data)->handle = h;
  }
  std::intptr_t uv_fs_write_awaitable::result() const noexcept {
    return reinterpret_cast<const awaitable_internal_t*>(data)->req.result;
  }

  uv_fs_close_awaitable::uv_fs_close_awaitable(int file) {
    auto* internal = reinterpret_cast<awaitable_internal_t*>(data);
    internal->handle = nullptr;
    internal->req.data = internal;
    uv_fs_close((uv_loop_t*)get_loop(), &internal->req, file, [](uv_fs_t* r) {
      auto* in = static_cast<awaitable_internal_t*>(r->data);
      if (in->handle) in->handle.resume();
    });
  }
  uv_fs_close_awaitable::~uv_fs_close_awaitable() {
    uv_fs_req_cleanup(&reinterpret_cast<awaitable_internal_t*>(data)->req);
  }
  void uv_fs_close_awaitable::await_suspend(std::coroutine_handle<> h) noexcept {
    reinterpret_cast<awaitable_internal_t*>(data)->handle = h;
  }

  uv_fs_size_awaitable::uv_fs_size_awaitable(int file) {
    auto* internal = reinterpret_cast<awaitable_internal_t*>(data);
    internal->handle = nullptr;
    internal->req.data = internal;
    uv_fs_fstat((uv_loop_t*)get_loop(), &internal->req, file, [](uv_fs_t* r) {
      auto* in = static_cast<awaitable_internal_t*>(r->data);
      if (in->handle) in->handle.resume();
    });
  }
  uv_fs_size_awaitable::uv_fs_size_awaitable(const std::string& path) {
    auto* internal = reinterpret_cast<awaitable_internal_t*>(data);
    internal->handle = nullptr;
    internal->req.data = internal;
    uv_fs_stat((uv_loop_t*)get_loop(), &internal->req, path.c_str(), [](uv_fs_t* r) {
      auto* in = static_cast<awaitable_internal_t*>(r->data);
      if (in->handle) in->handle.resume();
    });
  }
  uv_fs_size_awaitable::~uv_fs_size_awaitable() {
    uv_fs_req_cleanup(&reinterpret_cast<awaitable_internal_t*>(data)->req);
  }
  void uv_fs_size_awaitable::await_suspend(std::coroutine_handle<> h) noexcept {
    reinterpret_cast<awaitable_internal_t*>(data)->handle = h;
  }
  std::int64_t uv_fs_size_awaitable::result() const noexcept {
    auto* internal = reinterpret_cast<const awaitable_internal_t*>(data);
    return internal->req.result < 0 ? -1 : internal->req.statbuf.st_size;
  }

  loop_t loop_new() { return uv_loop_new(); }
  loop_t& get_loop() {
    static uv_loop_t* event_loop = uv_default_loop();
    return (loop_t&)event_loop;
  }
  void loop_stop(loop_t loop) { uv_stop((uv_loop_t*)loop); }
  int loop_close(loop_t loop) { return uv_loop_close((uv_loop_t*)loop); }

  void error_code_t::throw_if() const { if (code) throw code; }
  void error_code_t::await_resume() const { throw_if(); }

  void print_event_handles(loop_t loop) {
    uv_loop_t* uvloop = (uv_loop_t*)loop;
    fan::print("========================");
    fan::print("Active handles:", uvloop->active_handles);
    fan::print("Active requests:", uvloop->active_reqs.count);
    uv_walk(uvloop, [](uv_handle_t* handle, void* arg) {
      const char* type_name = uv_handle_type_name(handle->type);
      fan::print("Handle:", type_name, "active:", uv_is_active(handle), "closing:", uv_is_closing(handle));
      }, nullptr);
    fan::print("========================");
  }

  struct timer_t::timer_data {
    uv_timer_t timer_handle;
    std::coroutine_handle<> co_handle;
    int ready;
    timer_data() : co_handle(nullptr), ready(0) {}
  };

  void timer_t::timer_deleter::operator()(timer_t::timer_data* data) const noexcept {
    uv_close(reinterpret_cast<uv_handle_t*>(&data->timer_handle), [](uv_handle_t* timer_handle) {
      delete static_cast<timer_t::timer_data*>(timer_handle->data);
    });
  }

  timer_t::timer_t() : data(new timer_data{}, timer_deleter{}) {
    uv_timer_init((uv_loop_t*)fan::event::get_loop(), &data->timer_handle);
    data->timer_handle.data = data.get();
  }

  timer_t::timer_t(std::uint64_t timeout, std::uint64_t repeat)
    : data(new timer_data{}, timer_deleter{}) {
    uv_timer_init((uv_loop_t*)fan::event::get_loop(), &data->timer_handle);
    data->timer_handle.data = data.get();
    start(timeout, repeat);
  }

  error_code_t timer_t::start(std::uint64_t timeout, std::uint64_t repeat) noexcept {
    return uv_timer_start(&data->timer_handle, [](uv_timer_t* timer_handle) {
      auto data = static_cast<timer_t::timer_data*>(timer_handle->data);
      ++data->ready;
      if (data->co_handle) data->co_handle();
    }, timeout, repeat);
  }

  error_code_t timer_t::again() noexcept { return uv_timer_again(&data->timer_handle); }
  void timer_t::set_repeat(std::uint64_t repeat) noexcept { uv_timer_set_repeat(&data->timer_handle, repeat); }
  error_code_t timer_t::stop() noexcept { return uv_timer_stop(&data->timer_handle); }
  bool timer_t::await_ready() const noexcept { return data->ready; }
  void timer_t::await_suspend(std::coroutine_handle<> h) noexcept { data->co_handle = h; }
  void timer_t::await_resume() noexcept { data->co_handle = nullptr; --data->ready; }

  void deferred_resume_t::process_resumes() {
    for (auto& e : resume_queue) {
      if (e.h) e.h.resume();
    }
    resume_queue.clear();
  }

  struct idle_data_t {
    std::function<task_t()> callback;
    task_t task;
    bool has_task;
    idle_data_t(std::function<task_t()> cb) : callback(std::move(cb)), has_task(false) {}
  };

  idle_id_t idle_task::task_idle(std::function<task_t()> callback) {
    auto* idle_handle = new uv_idle_t;
    idle_handle->data = new idle_data_t(std::move(callback));
    if (uv_idle_init((uv_loop_t*)get_loop(), idle_handle) != 0 ||
        uv_idle_start(idle_handle, [](uv_idle_t* handle) {
          auto* data = static_cast<idle_data_t*>(handle->data);
          if (!data->has_task) {
            data->task = data->callback();
            data->has_task = true;
          }
          if (data->task.await_ready()) {
            data->task.await_resume();
            data->has_task = false;
          }
        }) != 0) {
      delete static_cast<idle_data_t*>(idle_handle->data);
      delete idle_handle;
      return nullptr;
    }
    return idle_handle;
  }

  void idle_task::idle_stop(idle_id_t idle_handle) {
    if (idle_handle) {
      uv_idle_stop((uv_idle_t*)idle_handle);
      uv_close(reinterpret_cast<uv_handle_t*>(idle_handle), [](uv_handle_t* handle) {
        auto* ih = reinterpret_cast<uv_idle_t*>(handle);
        delete static_cast<idle_data_t*>(ih->data);
        delete ih;
      });
    }
  }

  idle_id_t task_idle(std::function<task_t()> callback) { return idle_task::task_idle(std::move(callback)); }
  void idle_stop(idle_id_t idle_handle) { idle_task::idle_stop(idle_handle); }

  bool counter_awaitable_t::await_ready() const noexcept { return st->remaining == 0; }
  void counter_awaitable_t::await_suspend(std::coroutine_handle<> h) noexcept { st->continuation = h; }
  void counter_awaitable_t::await_resume() const noexcept {}

  struct fs_watcher_internal_t {
    struct file_event {
      std::string filename;
      int events;
      std::chrono::steady_clock::time_point timestamp;
    };
    uv_fs_event_t fs_event;
    uv_timer_t timer;
    std::string watch_path;
    std::function<void(const std::string&, int)> event_callback;
    std::unordered_map<std::string, file_event> pending_events;
  };

  fs_watcher_t::fs_watcher_t(const std::string& path) {
    internal_state = new fs_watcher_internal_t();
    auto* state = static_cast<fs_watcher_internal_t*>(internal_state);
    state->watch_path = path;
    state->fs_event.data = state;
    state->timer.data = state;
  }

  fs_watcher_t::~fs_watcher_t() {
    delete static_cast<fs_watcher_internal_t*>(internal_state);
  }

  bool fs_watcher_t::start(std::function<void(const std::string&, int)> callback) {
    auto* state = static_cast<fs_watcher_internal_t*>(internal_state);
    state->event_callback = callback;
    int result = uv_fs_event_init((uv_loop_t*)get_loop(), &state->fs_event);
    if (result < 0) return false;
    result = uv_fs_event_start(&state->fs_event, [](uv_fs_event_t* handle, const char* filename, int events, int status) {
      if (status < 0) return;
      auto* state = static_cast<fs_watcher_internal_t*>(handle->data);
      if (filename) {
        std::string file_str(filename);
        state->pending_events[file_str] = { file_str, events, std::chrono::steady_clock::now() };
      }
    }, state->watch_path.c_str(), UV_FS_EVENT_RECURSIVE);
    if (result < 0) return false;
    result = uv_timer_init((uv_loop_t*)get_loop(), &state->timer);
    if (result < 0) return false;
    result = uv_timer_start(&state->timer, [](uv_timer_t* handle) {
      auto* state = static_cast<fs_watcher_internal_t*>(handle->data);
      for (auto& event_pair : state->pending_events) {
        if (state->event_callback) state->event_callback(event_pair.second.filename, event_pair.second.events);
      }
      state->pending_events.clear();
    }, 0, 50);
    return result >= 0;
  }

  void fs_watcher_t::stop() {
    auto* state = static_cast<fs_watcher_internal_t*>(internal_state);
    uv_fs_event_stop(&state->fs_event);
    uv_timer_stop(&state->timer);
  }

  std::string fs_watcher_t::get_watch_path() {
    return static_cast<fs_watcher_internal_t*>(internal_state)->watch_path;
  }

  void sleep(unsigned int msec) { uv_sleep(msec); }
  void loop(fan::event::loop_t l, bool once) { uv_run((uv_loop_t*)l, once ? UV_RUN_ONCE : UV_RUN_DEFAULT); }
  void run_once(fan::event::loop_t l) { loop(l, true); }
  std::uint64_t now() { return uv_now((uv_loop_t*)get_loop()) * 1000000; }
  std::string strerror(int err) { return uv_strerror(err); }

  poll_awaitable_t::poll_awaitable_t(loop_t loop, int fd, int events) {
    poll_handle = new uv_poll_t();
    uv_poll_t* ph = static_cast<uv_poll_t*>(poll_handle);
    ph->data = this;
    uv_poll_init((uv_loop_t*)loop, ph, fd);
    uv_poll_start(ph, events, [](uv_poll_t* handle, int, int events) {
      auto* self = static_cast<poll_awaitable_t*>(handle->data);
      self->events_received = events;
      self->ready = true;
      uv_poll_stop(handle);
      if (self->co_handle) self->co_handle();
    });
  }

  poll_awaitable_t::~poll_awaitable_t() {
    uv_poll_t* ph = static_cast<uv_poll_t*>(poll_handle);
    if (ph) {
      uv_close((uv_handle_t*)ph, [](uv_handle_t* h) { delete (uv_poll_t*)h; });
    }
  }

  bool poll_awaitable_t::await_ready() const noexcept { return ready; }
  void poll_awaitable_t::await_suspend(std::coroutine_handle<> h) noexcept { co_handle = h; }
  int poll_awaitable_t::await_resume() noexcept { return events_received; }
  poll_awaitable_t poll_task(loop_t loop, int fd, int events) { return poll_awaitable_t(loop, fd, events); }
}

namespace fan::io::file {

  fan::event::runv_t<std::intptr_t> async_read(int file, std::string* buffer, std::int64_t offset, std::size_t buffer_size) {
    buffer->resize(buffer_size);
    std::intptr_t r = co_await async_read(file, buffer->data(), buffer_size, offset);
    if (r < 0) fan::throw_error("error reading file", std::to_string(r));
    buffer->resize(r);
    co_return r;
  }

  fan::event::runv_t<std::string> async_read(const std::string& path, int buffer_size) {
    int fd = co_await async_open(path);
    int offset = 0;
    std::string buffer;
    buffer.resize(buffer_size);
    std::string content;

    while (true) {
      std::intptr_t result = co_await async_read(fd, buffer.data(), buffer.size(), offset);
      if (result <= 0) break;
      content.append(buffer.data(), result);
      offset += result;
    }
    co_await async_close(fd);
    co_return content;
  }

  fan::event::task_t async_write(const std::string& path, const std::string& data) {
    int fd = co_await async_open(path, fan::fs_out);
    std::size_t offset = 0;
    std::size_t buffer_size = 4096;
    std::size_t total_written = 0;

    while (total_written < data.size()) {
      std::size_t remaining = data.size() - total_written;
      std::size_t to_write = std::min(remaining, buffer_size);
      std::string buffer(data.data() + total_written, to_write);
      std::intptr_t written = co_await async_write(fd, buffer.data(), buffer.size(), offset + total_written);
      if (written <= 0) fan::throw_error("write failed");
      total_written += written;
    }
    co_await async_close(fd);
  }

  fan::event::run_t async_read_t::open(const std::string& file_path) {
    path = file_path;
    fd = co_await fan::io::file::async_open(path);
    size = co_await fan::io::file::async_size(fd);
  }

  fan::event::run_t async_read_t::close() {
    co_await fan::io::file::async_close(fd);
  }

  fan::event::runv_t<std::string> async_read_t::read() {
    std::string buffer;
    std::intptr_t read_bytes = co_await fan::io::file::async_read(fd, &buffer, offset);
    if (read_bytes > 0) offset += read_bytes;
    if (read_bytes < 0) fan::throw_error("fs read error:" + fan::event::strerror((int)read_bytes));
    co_return buffer;
  }
}

namespace fan::io {
  struct ev_next_tick_awaiter {
    uv_idle_t* idle_handle = nullptr;
    std::coroutine_handle<> coro;
    bool await_ready() noexcept { return false; }
    void await_suspend(std::coroutine_handle<> h) noexcept {
      coro = h;
      idle_handle = new uv_idle_t;
      uv_idle_init((uv_loop_t*)fan::event::get_loop(), idle_handle);
      idle_handle->data = this;
      uv_idle_start(idle_handle, [](uv_idle_t* handle) {
        auto* awaiter = static_cast<ev_next_tick_awaiter*>(handle->data);
        uv_idle_stop(handle);
        uv_close(reinterpret_cast<uv_handle_t*>(handle), [](uv_handle_t* h) { delete reinterpret_cast<uv_idle_t*>(h); });
        awaiter->coro.resume();
      });
    }
    void await_resume() noexcept {}
  };

  struct dir_iter_internal {
    std::vector<std::filesystem::directory_entry> entries;
    std::string base_path;
    std::string next_path;
    std::size_t current_index = 0;
    bool stopped = false;
    bool switch_requested = false;
    fan::event::task_t iteration_task;
  };

  async_directory_iterator_t::async_directory_iterator_t() {
    internal_state = new dir_iter_internal();
  }

  async_directory_iterator_t::~async_directory_iterator_t() {
    delete static_cast<dir_iter_internal*>(internal_state);
  }

  void async_directory_iterator_t::stop() {
    auto* s = static_cast<dir_iter_internal*>(internal_state);
    s->stopped = true;
    s->switch_requested = false;
  }

  fan::event::task_t iterate_directory(async_directory_iterator_t* iter) {
    auto* s = static_cast<dir_iter_internal*>(iter->internal_state);
    while (s->current_index < s->entries.size()) {
      if (s->stopped) co_return;

      const auto& entry = s->entries[s->current_index];
      co_await iter->callback(entry.path().string(), entry.is_directory());

      ++s->current_index;
      if (s->stopped) co_return;
      co_await ev_next_tick_awaiter {};
    }
    co_return;
  }

  void async_directory_iterate(async_directory_iterator_t* iter, const std::string& path) {
    auto* s = static_cast<dir_iter_internal*>(iter->internal_state);
    if (iter->operation_in_progress) {
      s->stopped = true;
      s->switch_requested = true;
      s->next_path = path;
      return;
    }
    iter->operation_in_progress = true;
    s->stopped = false;
    s->switch_requested = false;
    s->base_path = path;
    s->entries.clear();
    s->current_index = 0;

    uv_fs_t* req = new uv_fs_t;
    std::memset(req, 0, sizeof(uv_fs_t));
    req->data = iter;
    int ret = uv_fs_scandir((uv_loop_t*)fan::event::get_loop(), req, path.c_str(), 0, [](uv_fs_t* r) {
      auto* it = static_cast<async_directory_iterator_t*>(r->data);
      auto* state = static_cast<dir_iter_internal*>(it->internal_state);
      
      if (!state->stopped) {
        uv_dirent_t ent;
        while (uv_fs_scandir_next(r, &ent) != fan::eof) {
          try { state->entries.emplace_back(std::filesystem::path(state->base_path) / ent.name); } catch (...) {}
        }
        if (it->sort_alphabetically) {
          std::sort(state->entries.begin(), state->entries.end(), [](const std::filesystem::directory_entry& a, const std::filesystem::directory_entry& b) {
            if (a.is_directory() == b.is_directory()) {
              std::string a_stem = a.path().stem().string();
              std::string b_stem = b.path().stem().string();
              std::transform(a_stem.begin(), a_stem.end(), a_stem.begin(), ::tolower);
              std::transform(b_stem.begin(), b_stem.end(), b_stem.begin(), ::tolower);
              return a_stem < b_stem;
            }
            return a.is_directory() && !b.is_directory();
          });
        }
        if (!state->stopped) state->iteration_task = iterate_directory(it);
      }
      uv_fs_req_cleanup(r);
      delete r;
      it->operation_in_progress = false;

      if (state->switch_requested) {
        std::string new_path = state->next_path;
        state->switch_requested = false;
        uv_idle_t* idle = new uv_idle_t;
        idle->data = it;
        uv_idle_init((uv_loop_t*)fan::event::get_loop(), idle);
        uv_idle_start(idle, [](uv_idle_t* handle) {
          auto* it = static_cast<async_directory_iterator_t*>(handle->data);
          std::string path = static_cast<dir_iter_internal*>(it->internal_state)->next_path;
          uv_idle_stop(handle);
          uv_close(reinterpret_cast<uv_handle_t*>(handle), [](uv_handle_t* h) { delete reinterpret_cast<uv_idle_t*>(h); });
          async_directory_iterate(it, path);
        });
      }
    });
    if (ret < 0) {
      delete req;
      iter->operation_in_progress = false;
      fan::throw_error("error fs_scandir:" + fan::event::strerror(ret));
    }
  }

  std::size_t async_directory_iterator_t::get_current_index() const {
    return static_cast<dir_iter_internal*>(internal_state)->current_index;
  }

  std::size_t async_directory_iterator_t::get_entries_size() const {
    return static_cast<dir_iter_internal*>(internal_state)->entries.size();
  }

  bool async_directory_iterator_t::is_finished() const {
    auto* s = static_cast<dir_iter_internal*>(internal_state);
    return s->current_index >= s->entries.size();
  }
}

struct cleaner_t {
  ~cleaner_t() {
    uv_loop_close((uv_loop_t*)fan::event::get_loop());
  }
} cleaner;