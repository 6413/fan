module;

#include <coroutine>
#include <ctype.h>

module fan.event;

import fan.print;
import fan.print.error;
import fan.event.uv_raw;
import fan.io.file;

namespace fan::event {

  struct awaitable_internal_t {
    fan::uv::fs_t req;
    std::coroutine_handle<> handle;
  };
  static_assert(sizeof(awaitable_internal_t) <= 512, "Awaitable Shell buffer too small");

  uv_fs_open_awaitable::uv_fs_open_awaitable(const std::string& path, int flags, int mode) {
    auto* internal = reinterpret_cast<awaitable_internal_t*>(data);
    internal->handle = nullptr;
    internal->req.data = internal;
    fan::uv::fs_open((fan::uv::loop_t*)get_loop(), &internal->req, path.c_str(), flags, mode, [](fan::uv::fs_t* r) {
      auto* in = static_cast<awaitable_internal_t*>(r->data);
      if (in->handle) in->handle.resume();
    });
  }
  uv_fs_open_awaitable::~uv_fs_open_awaitable() {
    fan::uv::fs_req_cleanup(&reinterpret_cast<awaitable_internal_t*>(data)->req);
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
    fan::uv::buf_t uv_buf = fan::uv::buf_init(buffer, size);
    fan::uv::fs_read((fan::uv::loop_t*)get_loop(), &internal->req, file, &uv_buf, 1, offset, [](fan::uv::fs_t* r) {
      auto* in = static_cast<awaitable_internal_t*>(r->data);
      if (in->handle) in->handle.resume();
    });
  }
  uv_fs_read_awaitable::~uv_fs_read_awaitable() {
    fan::uv::fs_req_cleanup(&reinterpret_cast<awaitable_internal_t*>(data)->req);
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
    fan::uv::buf_t uv_buf = fan::uv::buf_init(const_cast<char*>(buffer), length);
    fan::uv::fs_write((fan::uv::loop_t*)get_loop(), &internal->req, fd, &uv_buf, 1, offset, [](fan::uv::fs_t* r) {
      auto* in = static_cast<awaitable_internal_t*>(r->data);
      if (in->handle) in->handle.resume();
    });
  }
  uv_fs_write_awaitable::~uv_fs_write_awaitable() {
    fan::uv::fs_req_cleanup(&reinterpret_cast<awaitable_internal_t*>(data)->req);
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
    fan::uv::fs_close((fan::uv::loop_t*)get_loop(), &internal->req, file, [](fan::uv::fs_t* r) {
      auto* in = static_cast<awaitable_internal_t*>(r->data);
      if (in->handle) in->handle.resume();
    });
  }
  uv_fs_close_awaitable::~uv_fs_close_awaitable() {
    fan::uv::fs_req_cleanup(&reinterpret_cast<awaitable_internal_t*>(data)->req);
  }
  void uv_fs_close_awaitable::await_suspend(std::coroutine_handle<> h) noexcept {
    reinterpret_cast<awaitable_internal_t*>(data)->handle = h;
  }

  uv_fs_size_awaitable::uv_fs_size_awaitable(int file) {
    auto* internal = reinterpret_cast<awaitable_internal_t*>(data);
    internal->handle = nullptr;
    internal->req.data = internal;
    fan::uv::fs_fstat((fan::uv::loop_t*)get_loop(), &internal->req, file, [](fan::uv::fs_t* r) {
      auto* in = static_cast<awaitable_internal_t*>(r->data);
      if (in->handle) in->handle.resume();
    });
  }
  uv_fs_size_awaitable::uv_fs_size_awaitable(const std::string& path) {
    auto* internal = reinterpret_cast<awaitable_internal_t*>(data);
    internal->handle = nullptr;
    internal->req.data = internal;
    fan::uv::fs_stat((fan::uv::loop_t*)get_loop(), &internal->req, path.c_str(), [](fan::uv::fs_t* r) {
      auto* in = static_cast<awaitable_internal_t*>(r->data);
      if (in->handle) in->handle.resume();
    });
  }
  uv_fs_size_awaitable::~uv_fs_size_awaitable() {
    fan::uv::fs_req_cleanup(&reinterpret_cast<awaitable_internal_t*>(data)->req);
  }
  void uv_fs_size_awaitable::await_suspend(std::coroutine_handle<> h) noexcept {
    reinterpret_cast<awaitable_internal_t*>(data)->handle = h;
  }
  std::int64_t uv_fs_size_awaitable::result() const noexcept {
    auto* internal = reinterpret_cast<const awaitable_internal_t*>(data);
    return internal->req.result < 0 ? -1 : internal->req.statbuf.st_size;
  }

  loop_t loop_new() {
    auto* l = new fan::uv::loop_t;
    fan::uv::loop_init(l);
    return (loop_t)l;
  }
  loop_t& get_loop() {
    static fan::uv::loop_t* event_loop = fan::uv::default_loop();
    return (loop_t&)event_loop;
  }
  void loop_stop(loop_t loop) { fan::uv::stop((fan::uv::loop_t*)loop); }
  int loop_close(loop_t loop) { return fan::uv::loop_close((fan::uv::loop_t*)loop); }

  void error_code_t::throw_if() const { if (code) throw code; }
  void error_code_t::await_resume() const { throw_if(); }

  void print_event_handles(loop_t loop) {
    fan::uv::loop_t* uvloop = (fan::uv::loop_t*)loop;
    fan::print_dbg("========================");
    fan::print_dbg("Active handles:", uvloop->active_handles);
    fan::print_dbg("Active requests:", uvloop->active_reqs.count);
    fan::uv::walk(uvloop, [](fan::uv::handle_t* handle, void*) {
      const char* type_name = fan::uv::handle_type_name(handle->type);
      fan::print_dbg("Handle:", type_name, "active:", fan::uv::is_active(handle), "closing:", fan::uv::is_closing(handle));
    }, nullptr);
    fan::print_dbg("========================");
  }

  struct timer_t::timer_data {
    fan::uv::timer_t timer_handle;
    std::coroutine_handle<> co_handle;
    int ready;
    timer_data() : co_handle(nullptr), ready(0) {}
  };

  void timer_t::timer_deleter::operator()(timer_t::timer_data* data) const noexcept {
    fan::uv::close(reinterpret_cast<fan::uv::handle_t*>(&data->timer_handle), [](fan::uv::handle_t* h) {
      delete static_cast<timer_t::timer_data*>(h->data);
    });
  }

  timer_t::timer_t() : data(new timer_data{}, timer_deleter{}) {
    fan::uv::timer_init((fan::uv::loop_t*)fan::event::get_loop(), &data->timer_handle);
    data->timer_handle.data = data.get();
  }

  timer_t::timer_t(std::uint64_t timeout, std::uint64_t repeat)
    : data(new timer_data{}, timer_deleter{}) {
    fan::uv::timer_init((fan::uv::loop_t*)fan::event::get_loop(), &data->timer_handle);
    data->timer_handle.data = data.get();
    start(timeout, repeat);
  }

  error_code_t timer_t::start(std::uint64_t timeout, std::uint64_t repeat) noexcept {
    return fan::uv::timer_start(&data->timer_handle, [](fan::uv::timer_t* h) {
      auto* d = static_cast<timer_t::timer_data*>(h->data);
      ++d->ready;
      if (d->co_handle) d->co_handle();
    }, timeout, repeat);
  }

  error_code_t timer_t::again() noexcept { return fan::uv::timer_again(&data->timer_handle); }
  void timer_t::set_repeat(std::uint64_t repeat) noexcept { fan::uv::timer_set_repeat(&data->timer_handle, repeat); }
  error_code_t timer_t::stop() noexcept { return fan::uv::timer_stop(&data->timer_handle); }
  bool timer_t::await_ready() const noexcept { return data->ready; }
  void timer_t::await_suspend(std::coroutine_handle<> h) noexcept { data->co_handle = h; }
  void timer_t::await_resume() noexcept { data->co_handle = nullptr; --data->ready; }

  void deferred_resume_t::process_resumes() {
    for (auto& e : get_queue()) {
      if (e.h) e.h.resume();
    }
    get_queue().clear();
  }

  struct idle_data_t {
    std::function<task_t()> callback;
    task_t task;
    bool has_task;
    idle_data_t(std::function<task_t()> cb) : callback(std::move(cb)), has_task(false) {}
  };

  idle_id_t idle_task::task_idle(std::function<task_t()> callback) {
    auto* idle_handle = new fan::uv::idle_t;
    idle_handle->data = new idle_data_t(std::move(callback));
    if (fan::uv::idle_init((fan::uv::loop_t*)get_loop(), idle_handle) != 0 ||
        fan::uv::idle_start(idle_handle, [](fan::uv::idle_t* handle) {
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
      fan::uv::idle_stop((fan::uv::idle_t*)idle_handle);
      fan::uv::close(reinterpret_cast<fan::uv::handle_t*>(idle_handle), [](fan::uv::handle_t* handle) {
        auto* ih = reinterpret_cast<fan::uv::idle_t*>(handle);
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
  int active_handles = 0;
  void on_handle_close() {
    fan::print_dbg_tag("FS_WATCHER DEBUG", "Handle closed. Remaining active handles:", active_handles - 1);
    if (--active_handles == 0) {
      fan::print_dbg_tag("FS_WATCHER DEBUG", "All handles closed. Deleting internal state.");
      delete this;
    }
  }
  
  std::vector<fan::uv::fs_event_t*> fs_events;
  std::unordered_map<fan::uv::fs_event_t*, std::string> event_paths;
  fan::uv::timer_t timer;
  std::string watch_path;
  std::function<void(const std::string&, int)> event_callback;
  std::unordered_map<std::string, file_event> pending_events;

  int attach_watch(const std::string& target_path) {
    fan::print_dbg_tag("FS_WATCHER DEBUG", "Attempting to attach watch to:", target_path);
    auto* ev = new fan::uv::fs_event_t;
    ev->data = this;
    
    int result = fan::uv::fs_event_init((fan::uv::loop_t*)get_loop(), ev);
    if (result < 0) { 
      fan::print_dbg_tag("FS_WATCHER DEBUG", "FAILED to init fs_event for:", target_path, "Error:", fan::event::strerror(result));
      delete ev; 
      return result; 
    }
    
    event_paths[ev] = target_path;

    result = fan::uv::fs_event_start(ev, [](fan::uv::fs_event_t* handle, const char* filename, int events, int status) {
      auto* state = static_cast<fs_watcher_internal_t*>(handle->data);
      
      std::string file_str = filename ? filename : "UNKNOWN_FILE";
      std::string full_path = state->event_paths[handle] + "/" + file_str;
      
      fan::print_dbg_tag("FS_WATCHER DEBUG", "RAW INOTIFY EVENT FIRED! Status:", status, "| Events bitmask:", events, "| File:", full_path);

      if (status < 0) return;
      if (filename) {
        // Dynamically attach a watcher if a new folder is created at runtime
        std::error_code ec;
        if (std::filesystem::is_directory(full_path, ec)) {
          fan::print_dbg_tag("FS_WATCHER DEBUG", "Detected new runtime directory, attaching new watcher:", full_path);
          state->attach_watch(full_path);
        }

        state->pending_events[full_path] = { full_path, events, std::chrono::steady_clock::now() };
        fan::print_dbg_tag("FS_WATCHER DEBUG", "Event queued for processing on next timer tick.");
      }
    }, target_path.c_str(), 0);

    if (result < 0) {
      fan::print_dbg_tag("FS_WATCHER DEBUG", "FAILED to start fs_event for:", target_path, "Error:", fan::event::strerror(result));
      event_paths.erase(ev);
      fan::uv::close((fan::uv::handle_t*)ev, [](fan::uv::handle_t* h) { delete (fan::uv::fs_event_t*)h; });
      return result;
    }
    
    fs_events.push_back(ev);
    active_handles++;
    fan::print_dbg_tag("FS_WATCHER DEBUG", "Successfully attached watch. Total handles:", active_handles);
    return 0;
  }
};

fs_watcher_t::fs_watcher_t(const std::string& path) {
  internal_state = new fs_watcher_internal_t();
  auto* state = static_cast<fs_watcher_internal_t*>(internal_state);
  state->watch_path = std::filesystem::absolute(path).generic_string();
  fan::print_dbg_tag("FS_WATCHER DEBUG", "Watcher object constructed. Absolute target path:", state->watch_path);
  state->timer.data = state;
}

fs_watcher_t::~fs_watcher_t() {
  fan::print_dbg_tag("FS_WATCHER DEBUG", "Watcher destructor called.");
  stop();
  internal_state = nullptr;
}

std::expected<void, std::string> fs_watcher_t::start(std::function<void(const std::string&, int)> callback) {
  fan::print_dbg_tag("FS_WATCHER DEBUG", "start() called.");
  auto* state = static_cast<fs_watcher_internal_t*>(internal_state);
  
  if (state->active_handles > 0) {
    fan::print_dbg_tag("FS_WATCHER DEBUG", "start() aborted: Watcher is already active.");
    return std::unexpected("Watcher is already active");
  }
  state->event_callback = callback;

  fan::print_dbg_tag("FS_WATCHER DEBUG", "Attaching root watcher...");
  int res = state->attach_watch(state->watch_path);
  if (res < 0) return std::unexpected(fan::event::strerror(res));

#if defined(__linux__) || defined(__unix__)
  fan::print_dbg_tag("FS_WATCHER DEBUG", "UNIX Environment detected. Executing manual recursive traversal...");
  std::error_code ec;
  auto it = std::filesystem::recursive_directory_iterator(state->watch_path, std::filesystem::directory_options::skip_permission_denied, ec);
  
  if (ec) {
    fan::print_dbg_tag("FS_WATCHER DEBUG", "Filesystem error while starting iterator:", ec.message());
  }

  int dir_count = 0;
  for (; it != std::filesystem::recursive_directory_iterator(); ++it) {
    if (it->is_directory(ec)) {
      std::string dir_name = it->path().filename().string();
      
      // Skip hidden folders (.git, .xmake) and build output folders
      if ((dir_name.length() > 0 && dir_name[0] == '.') || dir_name == "build") {
        fan::print_dbg_tag("FS_WATCHER DEBUG", "Skipping ignored directory:", it->path().string());
        it.disable_recursion_pending();
        continue;
      }

      int r = state->attach_watch(it->path().string());
      if (r < 0) {
         fan::print_dbg_tag("FS_WATCHER DEBUG", "CRITICAL WARNING: Failed to attach watcher to", it->path().string(), "- Error:", fan::event::strerror(r));
      } else {
         dir_count++;
      }
    }
  }
  fan::print_dbg_tag("FS_WATCHER DEBUG", "Recursive traversal complete. Successfully attached to", dir_count, "subdirectories.");
#endif

  fan::print_dbg_tag("FS_WATCHER DEBUG", "Initializing timer...");
  int result = fan::uv::timer_init((fan::uv::loop_t*)get_loop(), &state->timer);
  if (result < 0) {
    fan::print_dbg_tag("FS_WATCHER DEBUG", "Timer init failed:", fan::event::strerror(result));
    stop();
    return std::unexpected(fan::event::strerror(result));
  }
  
  result = fan::uv::timer_start(&state->timer, [](fan::uv::timer_t* handle) {
    auto* state = static_cast<fs_watcher_internal_t*>(handle->data);
    
    // Heartbeat logic
    static int ticks = 0;
    if (ticks == 0 || ticks % 40 == 0) { // Prints immediately on first tick, then roughly every 2 seconds
      fan::print_dbg_tag("FS_WATCHER DEBUG", "Heartbeat tick. Event loop is healthy. Active handles:", state->active_handles);
    }
    ticks++;

    if (!state->pending_events.empty()) {
      fan::print_dbg_tag("FS_WATCHER DEBUG", "Dispatching", state->pending_events.size(), "queued events to user callback.");
    }

    for (auto& event_pair : state->pending_events) {
      if (state->event_callback) {
        state->event_callback(event_pair.second.filename, event_pair.second.events);
      }
    }
    state->pending_events.clear();
  }, 0, 50);

  if (result < 0) {
    fan::print_dbg_tag("FS_WATCHER DEBUG", "Timer start failed:", fan::event::strerror(result));
    stop();
    return std::unexpected(fan::event::strerror(result));
  }
  
  state->active_handles++;
  fan::print_dbg_tag("FS_WATCHER DEBUG", "start() complete successfully.");
  return {};
}

void fs_watcher_t::stop() {
  auto* state = static_cast<fs_watcher_internal_t*>(internal_state);
  if (!state || state->active_handles == 0) {
    return;
  }
  
  fan::print_dbg_tag("FS_WATCHER DEBUG", "stop() initiated. Closing", state->fs_events.size(), "filesystem handles and timer.");

  for (auto* ev : state->fs_events) {
    if (ev->loop != nullptr && ev->type != fan::uv::unknown_handle && !fan::uv::is_closing(reinterpret_cast<fan::uv::handle_t*>(ev))) {
      fan::uv::fs_event_stop(ev);
      fan::uv::close(reinterpret_cast<fan::uv::handle_t*>(ev), [](fan::uv::handle_t* handle) {
        static_cast<fs_watcher_internal_t*>(handle->data)->on_handle_close();
        delete (fan::uv::fs_event_t*)handle;
      });
    } else {
      --state->active_handles;
    }
  }
  state->fs_events.clear();

  if (state->timer.loop != nullptr && state->timer.type != fan::uv::unknown_handle && !fan::uv::is_closing(reinterpret_cast<fan::uv::handle_t*>(&state->timer))) {
    fan::uv::timer_stop(&state->timer);
    fan::uv::close(reinterpret_cast<fan::uv::handle_t*>(&state->timer), [](fan::uv::handle_t* handle) {
      static_cast<fs_watcher_internal_t*>(handle->data)->on_handle_close();
    });
  } else {
    --state->active_handles;
  }

  if (state->active_handles == 0) {
    delete state;
    internal_state = nullptr;
  } else {
    fan::uv::run((fan::uv::loop_t*)get_loop(), fan::uv::run_nowait);
  }
}

std::string fs_watcher_t::get_watch_path() {
  return static_cast<fs_watcher_internal_t*>(internal_state)->watch_path;
}
  void sleep(unsigned int msec) { fan::uv::sleep(msec); }
  void loop(fan::event::loop_t l, bool once) { fan::uv::run((fan::uv::loop_t*)l, once ? fan::uv::run_once : fan::uv::run_default); }
  void run_once(fan::event::loop_t l) { loop(l, true); }
  std::uint64_t now() { return fan::uv::now((fan::uv::loop_t*)get_loop()) * 1000000; }
  std::string strerror(int err) { return fan::uv::strerror(err); }

  static std::vector<std::function<void()>> async_queue;
  static std::mutex async_mtx;
  static fan::uv::async_t async_handle;
  static std::once_flag async_once;
  static std::atomic<bool> async_ready = false;

  void process_main_queue() {
    std::vector<std::function<void()>> q;
    {
      std::lock_guard lock(async_mtx);
      q.swap(async_queue);
    }
    for (auto& cb : q) {
      cb();
    }
  }

  void init_dispatcher() {
    std::call_once(async_once, [] {
      fan::uv::async_init((fan::uv::loop_t*)get_loop(), &async_handle, [](fan::uv::async_t*) {
        process_main_queue();
      });
      async_ready.store(true, std::memory_order_release);
    });
  }

  void close_dispatcher() {
    if (async_ready.exchange(false, std::memory_order_acq_rel)) {
      fan::uv::close(reinterpret_cast<fan::uv::handle_t*>(&async_handle), nullptr);
    }
  }

  void post_to_main(std::function<void()> cb) {
    {
      std::lock_guard lock(async_mtx);
      async_queue.push_back(std::move(cb));
    }
    if (async_ready.load(std::memory_order_acquire)) {
      fan::uv::async_send(&async_handle);
    }
  }

  poll_awaitable_t::poll_awaitable_t(loop_t loop, int fd, int events) {
    poll_handle = new fan::uv::poll_t();
    fan::uv::poll_t* ph = static_cast<fan::uv::poll_t*>(poll_handle);
    ph->data = this;
    fan::uv::poll_init((fan::uv::loop_t*)loop, ph, fd);
    fan::uv::poll_start(ph, events, [](fan::uv::poll_t* handle, int, int events) {
      auto* self = static_cast<poll_awaitable_t*>(handle->data);
      self->events_received = events;
      self->ready = true;
      fan::uv::poll_stop(handle);
      if (self->co_handle) self->co_handle();
    });
  }

  poll_awaitable_t::~poll_awaitable_t() {
    fan::uv::poll_t* ph = static_cast<fan::uv::poll_t*>(poll_handle);
    if (ph) {
      fan::uv::close((fan::uv::handle_t*)ph, [](fan::uv::handle_t* h) { delete (fan::uv::poll_t*)h; });
    }
  }

  bool poll_awaitable_t::await_ready() const noexcept { return ready; }
  void poll_awaitable_t::await_suspend(std::coroutine_handle<> h) noexcept { co_handle = h; }
  int poll_awaitable_t::await_resume() noexcept { return events_received; }
  poll_awaitable_t poll_task(loop_t loop, int fd, int events) { return poll_awaitable_t(loop, fd, events); }
}

namespace fan::io::file {

  fan::event::runv_t<int> async_open(const std::string& path, int flags, int mode) {
    fan::event::uv_fs_open_awaitable req(path, flags, mode);
    co_await req;
    co_return req.result();
  }

  fan::event::runv_t<std::intptr_t> async_read(int file, char* buffer, std::size_t buffer_size, std::int64_t offset) {
    fan::event::uv_fs_read_awaitable req(file, buffer, buffer_size, offset);
    co_await req;
    co_return req.result();
  }

  fan::event::runv_t<std::intptr_t> async_write(int fd, const char* buffer, std::size_t length, std::int64_t offset) {
    fan::event::uv_fs_write_awaitable req(fd, buffer, length, offset);
    co_await req;
    co_return req.result();
  }

  fan::event::task_t async_close(int file) {
    fan::event::uv_fs_close_awaitable req(file);
    co_await req;
  }

  fan::event::runv_t<std::intptr_t> async_size(int file) {
    fan::event::uv_fs_size_awaitable req(file);
    co_await req;
    co_return req.result();
  }

  fan::event::runv_t<std::intptr_t> async_size(const std::string& path) {
    fan::event::uv_fs_size_awaitable req(path);
    co_await req;
    co_return req.result();
  }

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

  fan::event::run_t async_write_t::open(const std::string& file_path) {
    path = file_path;
    fd = co_await fan::io::file::async_open(path, fan::fs_out);
  }

  fan::event::run_t async_write_t::close() const {
    co_await fan::io::file::async_close(fd);
  }

  fan::event::runv_t<std::intptr_t> async_write_t::write(const std::string& data, std::size_t buffer_size) {
    std::intptr_t result = co_await fan::io::file::async_write(fd, data.data() + offset, std::min(data.size() - offset, buffer_size), offset);
    if (result > 0) offset += result;
    co_return result;
  }
}

namespace fan::io {
  struct ev_next_tick_awaiter {
    fan::uv::idle_t* idle_handle = nullptr;
    std::coroutine_handle<> coro;
    bool await_ready() noexcept { return false; }
    void await_suspend(std::coroutine_handle<> h) noexcept {
        coro = h;
        idle_handle = new fan::uv::idle_t;
        fan::uv::idle_init((fan::uv::loop_t*)fan::event::get_loop(), idle_handle);
        idle_handle->data = this;
        fan::uv::idle_start(idle_handle, [](fan::uv::idle_t* handle) {
          auto* awaiter = static_cast<ev_next_tick_awaiter*>(handle->data);
          fan::uv::idle_stop(handle);
          fan::uv::close(reinterpret_cast<fan::uv::handle_t*>(handle), [](fan::uv::handle_t* h) {
            delete reinterpret_cast<fan::uv::idle_t*>(h);
          });
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
    try {
      while (s->current_index < s->entries.size()) {
        if (s->stopped) break;
        const auto& entry = s->entries[s->current_index];
        std::string path_str = entry.path().string();
        bool is_dir = entry.is_directory();
        auto cb_task = iter->callback(path_str, is_dir);
        co_await cb_task;
        ++s->current_index;
        if (s->stopped) break;
        ev_next_tick_awaiter awaiter{};
        co_await awaiter;
      }
    }
    catch (...) {}
    
    iter->operation_in_progress = false;

    if (s->switch_requested) {
      s->switch_requested = false;
      fan::uv::idle_t* idle = new fan::uv::idle_t;
      idle->data = iter;
      fan::uv::idle_init((fan::uv::loop_t*)fan::event::get_loop(), idle);
      fan::uv::idle_start(idle, [](fan::uv::idle_t* handle) {
        auto* it = static_cast<async_directory_iterator_t*>(handle->data);
        std::string path = static_cast<dir_iter_internal*>(it->internal_state)->next_path;
        fan::uv::idle_stop(handle);
        fan::uv::close(reinterpret_cast<fan::uv::handle_t*>(handle), [](fan::uv::handle_t* h) {
          delete reinterpret_cast<fan::uv::idle_t*>(h);
        });
        async_directory_iterate(it, path);
      });
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

    fan::uv::fs_t* req = new fan::uv::fs_t;
    std::memset(req, 0, sizeof(fan::uv::fs_t));
    req->data = iter;
    int ret = fan::uv::fs_scandir((fan::uv::loop_t*)fan::event::get_loop(), req, path.c_str(), 0, [](fan::uv::fs_t* r) {
      auto* it = static_cast<async_directory_iterator_t*>(r->data);
      auto* state = static_cast<dir_iter_internal*>(it->internal_state);

      if (!state->stopped) {
        fan::uv::dirent_t ent;
        while (1) {
          auto res = fan::uv::fs_scandir_next(r, &ent);
          if (res < 0) break;
          try { state->entries.emplace_back(std::filesystem::path(state->base_path) / ent.name); } catch (...) {}
        }
        if (it->sort_alphabetically) {
          std::sort(state->entries.begin(), state->entries.end(), [](const std::filesystem::directory_entry& a, const std::filesystem::directory_entry& b) {
            std::error_code ec;
            bool a_is_dir = a.is_directory(ec);
            if (ec) a_is_dir = false;

            bool b_is_dir = b.is_directory(ec);
            if (ec) b_is_dir = false;

            if (a_is_dir == b_is_dir) {
              std::string a_stem = a.path().stem().string();
              std::string b_stem = b.path().stem().string();
              std::transform(a_stem.begin(), a_stem.end(), a_stem.begin(), ::tolower);
              std::transform(b_stem.begin(), b_stem.end(), b_stem.begin(), ::tolower);
              return a_stem < b_stem;
            }
            return a_is_dir && !b_is_dir;
          });
        }
        if (!state->stopped) {
          state->iteration_task = iterate_directory(it);
        } else {
          it->operation_in_progress = false;

          if (state->switch_requested) {
            state->switch_requested = false;
            fan::uv::idle_t* idle = new fan::uv::idle_t;
            idle->data = it;
            fan::uv::idle_init((fan::uv::loop_t*)fan::event::get_loop(), idle);
            fan::uv::idle_start(idle, [](fan::uv::idle_t* handle) {
              auto* it = static_cast<async_directory_iterator_t*>(handle->data);
              std::string path = static_cast<dir_iter_internal*>(it->internal_state)->next_path;
              fan::uv::idle_stop(handle);
              fan::uv::close(reinterpret_cast<fan::uv::handle_t*>(handle), [](fan::uv::handle_t* h) {
                delete reinterpret_cast<fan::uv::idle_t*>(h);
              });
              async_directory_iterate(it, path);
            });
          }
        }
      }
      fan::uv::fs_req_cleanup(r);
      delete r;
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