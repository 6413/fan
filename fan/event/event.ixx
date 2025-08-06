module;
#include <coroutine>
#include <functional>
#include <queue>
#include <chrono>
#include <memory>
#include <exception>
#include <filesystem>
#include <algorithm>
#include <cstring>
#include <thread>

#include <uv.h>
#undef min
#undef max

using namespace std::chrono_literals;

export module fan.event;

import fan.print;
import fan.types.traits;

export namespace fan{
  inline constexpr int fs_o_append      = UV_FS_O_APPEND;
  inline constexpr int fs_o_creat       = UV_FS_O_CREAT;
  inline constexpr int fs_o_excl        = UV_FS_O_EXCL;
  inline constexpr int fs_o_filemap     = UV_FS_O_FILEMAP;
  inline constexpr int fs_o_random      = UV_FS_O_RANDOM;
  inline constexpr int fs_o_rdonly     = UV_FS_O_RDONLY;
  inline constexpr int fs_o_rdwr        = UV_FS_O_RDWR;
  inline constexpr int fs_o_sequential  = UV_FS_O_SEQUENTIAL;
  inline constexpr int fs_o_short_lived = UV_FS_O_SHORT_LIVED;
  inline constexpr int fs_o_temporary   = UV_FS_O_TEMPORARY;
  inline constexpr int fs_o_trunc       = UV_FS_O_TRUNC;
  inline constexpr int fs_o_wronly      = UV_FS_O_WRONLY;

  inline constexpr int fs_o_direct      = UV_FS_O_DIRECT;
  inline constexpr int fs_o_directory   = UV_FS_O_DIRECTORY;
  inline constexpr int fs_o_dsync       = UV_FS_O_DSYNC; 
  inline constexpr int fs_o_exlock      = UV_FS_O_EXLOCK; 
  inline constexpr int fs_o_noatime     = UV_FS_O_NOATIME;
  inline constexpr int fs_o_noctty      = UV_FS_O_NOCTTY;
  inline constexpr int fs_o_nofollow    = UV_FS_O_NOFOLLOW;
  inline constexpr int fs_o_nonblock    = UV_FS_O_NONBLOCK;
  inline constexpr int fs_o_symlink     = UV_FS_O_SYMLINK;
  inline constexpr int fs_o_sync        = UV_FS_O_SYNC;

  inline constexpr int fs_in        = UV_FS_O_RDONLY;
  inline constexpr int fs_out       = UV_FS_O_WRONLY | UV_FS_O_CREAT | UV_FS_O_TRUNC;
  inline constexpr int fs_app       = UV_FS_O_WRONLY | UV_FS_O_APPEND;
  inline constexpr int fs_trunc     = UV_FS_O_TRUNC;
  inline constexpr int fs_ate       = UV_FS_O_RDWR;
  inline constexpr int fs_nocreate  = UV_FS_O_EXCL;
  inline constexpr int fs_noreplace = UV_FS_O_EXCL;


  // User (owner) permissions
  constexpr int s_irusr = 0400;  // Read permission bit for the owner of the file
  constexpr int s_iread = 0400;  // Obsolete synonym for BSD compatibility

  constexpr int s_iwusr = 0200;  // Write permission bit for the owner of the file
  constexpr int s_iwrite = 0200;  // Obsolete synonym for BSD compatibility

  constexpr int s_ixusr = 0100;  // Execute (for ordinary files) or search (for directories) permission bit for the owner
  constexpr int s_iexec = 0100;  // Obsolete synonym for BSD compatibility

  constexpr int s_irwxu = (s_irusr | s_iwusr | s_ixusr);  // Equivalent to (S_IRUSR | S_IWUSR | S_IXUSR)

  // Group permissions
  constexpr int s_irgrp = 040;   // Read permission bit for the group owner of the file
  constexpr int s_iwgrp = 020;   // Write permission bit for the group owner of the file
  constexpr int s_ixgrp = 010;   // Execute or search permission bit for the group owner of the file

  constexpr int s_irwxg = (s_irgrp | s_iwgrp | s_ixgrp);  // Equivalent to (S_IRGRP | S_IWGRP | S_IXGRP)

  // Other users' permissions
  constexpr int s_iroth = 04;    // Read permission bit for other users
  constexpr int s_iwoth = 02;    // Write permission bit for other users
  constexpr int s_ixoth = 01;    // Execute or search permission bit for other users

  constexpr int s_irwxo = (s_iroth | s_iwoth | s_ixoth);  // Equivalent to (S_IROTH | S_IWOTH | S_IXOTH)

  // Special permission bits
  constexpr int s_isuid = 04000; // Set-user-ID on execute bit
  constexpr int s_isgid = 02000; // Set-group-ID on execute bit
  constexpr int s_isvtx = 01000; // Sticky bit

  constexpr int s_usr_rw = (s_irusr | s_iwusr);   // User read/write
  constexpr int s_grp_r = s_irgrp;                 // Group read
  constexpr int s_oth_r = s_iroth;                 // Other read

  constexpr int perm_0644 = s_usr_rw | s_grp_r | s_oth_r;

  constexpr int fs_change = UV_CHANGE;
  constexpr int fs_rename = UV_RENAME;
}

template<typename promise_type_t>
struct final_awaiter {
  bool await_ready() noexcept { return false; }

  void await_suspend(std::coroutine_handle<promise_type_t> h) noexcept {
    if (h.promise().continuation) {
      h.promise().continuation.resume();
    }
  }

  void await_resume() noexcept {}
};

template<typename T, typename suspend_type_t>
struct task_value_wrap_t;

template<typename T, typename suspend_type_t>
struct task_value_promise_t {
  T value;
  std::exception_ptr exception = nullptr;
  std::coroutine_handle<> continuation = nullptr;

  task_value_wrap_t<T, suspend_type_t> get_return_object();
  suspend_type_t initial_suspend() noexcept { return {}; }
  auto final_suspend() noexcept {
    return final_awaiter<task_value_promise_t<T, suspend_type_t>>{};
  }
  void return_value(T&& val) {
    value = std::move(val);
  }
  void return_value(const T& val) {
    value = val;
  }
  void unhandled_exception() {
    exception = std::current_exception();
  }
};

template<typename suspend_type_t>
struct task_value_promise_t<void, suspend_type_t> {
  std::exception_ptr exception = nullptr;
  std::coroutine_handle<> continuation = nullptr;

  task_value_wrap_t<void, suspend_type_t> get_return_object();
  suspend_type_t initial_suspend() noexcept { return {}; }
  auto final_suspend() noexcept {
    return final_awaiter<task_value_promise_t<void, suspend_type_t>>{};
  }
  void return_void() {}
  void unhandled_exception() {
    exception = std::current_exception();
  }
};

template<typename T, typename suspend_type_t>
struct task_value_wrap_t {
  using promise_type = task_value_promise_t<T, suspend_type_t>;

  task_value_wrap_t() {}
  task_value_wrap_t(std::coroutine_handle<promise_type> ch) : handle(ch) {}
  task_value_wrap_t(const task_value_wrap_t&) = delete;
  task_value_wrap_t(task_value_wrap_t&& other) : handle(other.handle) {
    other.handle = nullptr;
  }
  task_value_wrap_t& operator=(task_value_wrap_t&& other) {
    destroy();
    std::swap(handle, other.handle);
    return *this;
  }
  void destroy() {
    if (handle) {
      handle.destroy();
      handle = nullptr;
    }
  }
  ~task_value_wrap_t() {
    if (handle) {
      handle.destroy();
    }
  }
  bool valid() const {
    return static_cast<bool>(handle);
  }
  bool await_ready() const noexcept {
    return handle.done();
  }
  void await_suspend(std::coroutine_handle<> continuation_handle) noexcept {
    handle.promise().continuation = continuation_handle;
  }
  T await_resume() {
    if (handle.promise().exception) {
      std::rethrow_exception(handle.promise().exception);
    }
    return std::move(handle.promise().value);
  }

  std::coroutine_handle<promise_type> handle = nullptr;
};

template <typename T, typename suspend_type_t>
task_value_wrap_t<T, suspend_type_t> task_value_promise_t<T, suspend_type_t>::get_return_object() {
  return task_value_wrap_t<T, suspend_type_t>{std::coroutine_handle<task_value_promise_t<T, suspend_type_t>>::from_promise(*this)};
}

template<typename suspend_type_t>
struct task_value_wrap_t<void, suspend_type_t> {
  using promise_type = task_value_promise_t<void, suspend_type_t>;

  task_value_wrap_t() {}
  task_value_wrap_t(std::coroutine_handle<promise_type> ch) : handle(ch) {}
  task_value_wrap_t(const task_value_wrap_t&) = delete;
  task_value_wrap_t(task_value_wrap_t&& other) : handle(other.handle) {
    other.handle = nullptr;
  }
  task_value_wrap_t& operator=(task_value_wrap_t&& other) {
    destroy();
    std::swap(handle, other.handle);
    return *this;
  }
  void destroy() {
    if (handle) {
      handle.destroy();
      handle = nullptr;
    }
  }
  ~task_value_wrap_t() {
    if (handle) {
      handle.destroy();
      handle = nullptr;
    }
  }
  bool valid() const {
    return static_cast<bool>(handle);
  }
  bool await_ready() const noexcept {
    return handle.done();
  }
  void await_suspend(std::coroutine_handle<> continuation_handle) noexcept {
    handle.promise().continuation = continuation_handle;
  }
  void await_resume() {
    if (handle.promise().exception) {
      std::rethrow_exception(handle.promise().exception);
    }
  }

  std::coroutine_handle<promise_type> handle = nullptr;
};

template <typename suspend_type_t>
task_value_wrap_t<void, suspend_type_t> task_value_promise_t<void, suspend_type_t>::get_return_object() {
  return task_value_wrap_t<void, suspend_type_t>{std::coroutine_handle<task_value_promise_t<void, suspend_type_t>>::from_promise(*this)};
}

export namespace fan {

  namespace event {
    
    uv_loop_t*& get_event_loop() {
      static uv_loop_t* event_loop = uv_default_loop();
      return event_loop;
    }

    struct error_code_t {
      int code;
      constexpr error_code_t(int code) noexcept : code(code) {}
      constexpr operator int() const noexcept { return code; }
      void throw_if() const { if (code) throw code; }
      constexpr bool await_ready() const noexcept { return true; }
      constexpr void await_suspend(std::coroutine_handle<>) const noexcept {}
      void await_resume() const { throw_if(); }
    };
    struct timer_t {
      struct timer_data {
        uv_timer_t timer_handle;
        std::coroutine_handle<> co_handle = nullptr;
        int ready{ 0 };
      };

      struct timer_deleter {
        void operator()(timer_data* data) const noexcept {
          uv_close(reinterpret_cast<uv_handle_t*>(&data->timer_handle), [](uv_handle_t* timer_handle) {
            delete static_cast<timer_data*>(timer_handle->data);
            });
        }
      };

      std::unique_ptr<timer_data, timer_deleter> data;

      timer_t() : data(new timer_data{}, timer_deleter{}) {
        uv_timer_init(fan::event::get_event_loop(), &data->timer_handle);
        data->timer_handle.data = data.get();
      }
      // timeout in ms
      timer_t(uint64_t timeout, uint64_t repeat = 0)
        : data(new timer_data{}, timer_deleter{}) {
        uv_timer_init(fan::event::get_event_loop(), &data->timer_handle);
        data->timer_handle.data = data.get();
        start(timeout, repeat);
      }

      // timeout in ms
      error_code_t start(uint64_t timeout, uint64_t repeat = 0) noexcept {
        return uv_timer_start(&data->timer_handle, [](uv_timer_t* timer_handle) {
          auto data = static_cast<timer_data*>(timer_handle->data);
          ++data->ready;
          if (data->co_handle) {
            data->co_handle();
          }
          }, timeout, repeat);
      }
      error_code_t again() noexcept {
        return uv_timer_again(&data->timer_handle);
      }
      void set_repeat(uint64_t repeat) noexcept {
        return uv_timer_set_repeat(&data->timer_handle, repeat);
      }
      error_code_t stop() noexcept {
        return uv_timer_stop(&data->timer_handle);
      }
      bool await_ready() const noexcept { return data->ready; }

      void await_suspend(std::coroutine_handle<> h) noexcept { data->co_handle = h; }

      void await_resume() noexcept {
        data->co_handle = nullptr;
        --data->ready;
      };
    };


    using task_suspend_t = task_value_wrap_t<void, std::suspend_always>;
    using task_resume_t = task_value_wrap_t<void, std::suspend_never>;

    template <typename T>
    using task_value_t = task_value_wrap_t<T, std::suspend_always>;

    template <typename T>
    using task_value_resume_t = task_value_wrap_t<T, std::suspend_never>;

    using task_t = task_resume_t;

    struct uv_fs_awaitable {
      uv_fs_t req;
      std::coroutine_handle<> handle;
      bool closed = false;

      uv_fs_awaitable() { req.data = this; }
      ~uv_fs_awaitable() { uv_fs_req_cleanup(&req); closed = true; }
      uv_fs_awaitable(const uv_fs_awaitable&) = delete;
      uv_fs_awaitable& operator=(const uv_fs_awaitable&) = delete;
      uv_fs_awaitable(uv_fs_awaitable&&) = delete;
      uv_fs_awaitable& operator=(uv_fs_awaitable&&) = delete;

      bool await_ready() const noexcept { return false; }
      void await_suspend(std::coroutine_handle<> h) noexcept {
        handle = h;
      }
      void await_resume() noexcept {}
    };

    struct uv_fs_open_awaitable : uv_fs_awaitable {
      uv_fs_open_awaitable(const std::string& path, int flags, int mode) {
        req.data = this;
        uv_fs_open(get_event_loop(), &req, path.c_str(), flags, mode, on_open_cb);
      }

      static void on_open_cb(uv_fs_t* r) {
        auto self = static_cast<uv_fs_open_awaitable*>(r->data);
        if (self->closed) {
          return;
        }
        if (self->handle) {
          self->handle.resume();
        }
      }

      int result() const noexcept {
        return req.result;
      }
    };

    struct uv_fs_size_awaitable : uv_fs_awaitable {
      uv_fs_size_awaitable(int file) {
        uv_fs_fstat(get_event_loop(), &req, file, on_size_cb);
      }
      uv_fs_size_awaitable(const std::string& path) {
        uv_fs_stat(fan::event::get_event_loop(), &req, path.c_str(), on_size_cb);
      }
      static void on_size_cb(uv_fs_t* r) {
        auto self = static_cast<uv_fs_size_awaitable*>(r->data);
        if (self->closed) {
          return;
        }
        if (self->handle) {
          self->handle.resume();
        }
      }
      int64_t result() const noexcept {
        return req.result < 0 ? -1 : req.statbuf.st_size;
      }
    };


    struct uv_fs_read_awaitable : uv_fs_awaitable {
      uv_fs_read_awaitable(int file, uv_buf_t buf, int64_t offset) {
        int ret = uv_fs_read(fan::event::get_event_loop(), &req, file, &buf, 1, offset, on_read_cb);
        if (ret < 0) {
          fan::throw_error("uv_error"_str + uv_strerror(ret));
        }
      }
      static void on_read_cb(uv_fs_t* r) {
        auto self = static_cast<uv_fs_read_awaitable*>(r->data);
        if (self->closed) {
          return;
        }
        self->handle.resume();
      }
      ssize_t result() const noexcept {
        return req.result;
      }
    };

    struct uv_fs_write_awaitable : uv_fs_awaitable {
      uv_fs_write_awaitable(int fd, const char* buffer, size_t length, int64_t offset) {
        req.data = this;
        uv_buf_t buf = uv_buf_init(const_cast<char*>(buffer), length);
        uv_fs_write(fan::event::get_event_loop(), &req, fd, &buf, 1, offset, on_write_cb);
      }
      static void on_write_cb(uv_fs_t* req) {
        auto self = static_cast<uv_fs_write_awaitable*>(req->data);
        if (self->closed) {
          return;
        }
        self->handle.resume();
      }
      ssize_t result() const noexcept {
        return req.result;
      }
    };

    struct uv_fs_close_awaitable : uv_fs_awaitable {
      uv_fs_close_awaitable(int file) {
        req.data = this;
        uv_fs_close(fan::event::get_event_loop(), &req, file, on_close_cb);
      }
      static void on_close_cb(uv_fs_t* r) {
        auto self = static_cast<uv_fs_close_awaitable*>(r->data);
        if (self->closed) {
          return;
        }
        self->handle.resume();
      }
    };

    struct fs_watcher_t {
      struct file_event {
        std::string filename;
        int events;
        std::chrono::steady_clock::time_point timestamp;
      };

      uv_fs_event_t fs_event;
      uv_loop_t* loop;
      uv_timer_t timer;
      std::string watch_path;
      std::function<void(const std::string&, int)> event_callback;
      std::unordered_map<std::string, file_event> pending_events;

      static void timer_callback(uv_timer_t* handle) {
        fs_watcher_t* watcher = static_cast<fs_watcher_t*>(handle->data);
        watcher->process_latest_events();
      }

      static void fs_event_callback(uv_fs_event_t* handle, const char* filename, int events, int status) {
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

      void process_latest_events() {
        for (auto& event_pair : pending_events) {
          if (event_callback) {
            event_callback(event_pair.second.filename, event_pair.second.events);
          }
        }
        pending_events.clear();
      }

      fs_watcher_t(const std::string& path)
        : loop(fan::event::get_event_loop()), watch_path(path) {
        fs_event.data = this;
        timer.data = this;
      }

      bool start(std::function<void(const std::string&, int)> callback) {
        event_callback = callback;

        int result = uv_fs_event_init(loop, &fs_event);
        if (result < 0) return false;

        result = uv_fs_event_start(&fs_event, fs_event_callback,
          watch_path.c_str(), UV_FS_EVENT_RECURSIVE);
        if (result < 0) return false;

        result = uv_timer_init(loop, &timer);
        if (result < 0) return false;

        result = uv_timer_start(&timer, timer_callback, 0, 50);
        if (result < 0) return false;

        return true;
      }

      void stop() {
        uv_fs_event_stop(&fs_event);
        uv_timer_stop(&timer);
      }
    };

    fan::event::task_t task_timer(uint64_t time, auto l) {
      while (true) {
        if constexpr (fan::is_awaitable_v<decltype(l())>) {
          bool ret = co_await l();
          if (ret) {
            break;
          }
        }
        else {
          if (l()) {
            break;
          }
        }
        co_await event::timer_t(time);
      }
    }

    //thread stuff

    template <typename cb_t, typename ...args_t>
    void thread_create(cb_t&& cb, args_t&&... args) {
      std::jthread([cb = std::forward<cb_t>(cb), args = std::make_tuple(std::forward<args_t>(args)...)]() mutable {
        std::apply(cb, args);
        }).detach();
    }
    void sleep(unsigned int msec) {
      uv_sleep(msec);
    }
    void loop(bool once = false) {
      uv_run(fan::event::get_event_loop(), once ? UV_RUN_ONCE : UV_RUN_DEFAULT);
    }
    uint64_t now() {
      return uv_now(get_event_loop()) * 1000000;
    }

    std::string strerror(int err) {
      return uv_strerror(err);
    }

    using idle_id_t = uv_idle_t*;

    struct idle_task {
    private:
      struct idle_data {
        std::function<task_t()> callback;
        task_t task;
        bool has_task = false;

        idle_data(std::function<task_t()> cb) : callback(std::move(cb)) {}
      };

      static void idle_callback(uv_idle_t* handle) {
        auto* data = static_cast<idle_data*>(handle->data);

        // Start new task if none active
        if (!data->has_task) {
          data->task = data->callback();
          data->has_task = true;
        }

        // Check if task is done
        if (data->task.await_ready()) {
          data->task.await_resume();
          data->has_task = false;
        }
      }

      static void close_callback(uv_handle_t* handle) {
        auto* idle_handle = reinterpret_cast<uv_idle_t*>(handle);
        delete static_cast<idle_data*>(idle_handle->data);
        delete idle_handle;
      }

    public:
      static idle_id_t task_idle(std::function<task_t()> callback) {
        auto* idle_handle = new uv_idle_t;
        idle_handle->data = new idle_data(std::move(callback));

        if (uv_idle_init(get_event_loop(), idle_handle) != 0 ||
          uv_idle_start(idle_handle, idle_callback) != 0) {
          delete static_cast<idle_data*>(idle_handle->data);
          delete idle_handle;
          return nullptr;
        }

        return idle_handle;
      }

      static void idle_stop(idle_id_t idle_handle) {
        if (idle_handle) {
          uv_idle_stop(idle_handle);
          uv_close(reinterpret_cast<uv_handle_t*>(idle_handle), close_callback);
        }
      }
    };


    idle_id_t task_idle(std::function<task_t()> callback) {
      return idle_task::task_idle(std::move(callback));
    }

    void idle_stop(idle_id_t idle_handle) {
      idle_task::idle_stop(idle_handle);
    }
  }
  using co_sleep = event::timer_t;
}

export namespace fan::io::file {
  fan::event::task_value_resume_t<int> async_open(const std::string& path, int flags = fan::fs_in, int mode = fan::perm_0644) {
    fan::event::uv_fs_open_awaitable open_req(path, flags, mode);
    co_await open_req;
    auto r = open_req.result();
    if (r < 0) {
      fan::throw_error("failed to open file:" + path, "error:"_str + uv_strerror(r));
    }
    co_return r;
  }

  fan::event::task_value_resume_t<ssize_t> async_read(int file, char* buffer, size_t buffer_size, int64_t offset) {
    uv_buf_t buf = uv_buf_init(buffer, buffer_size);
    fan::event::uv_fs_read_awaitable read_req(file, buf, offset);
    co_await read_req;
    auto r = read_req.result();
    if (r < 0) {
      fan::throw_error("error reading file", std::to_string(r));
    }
    co_return r;
  }
  fan::event::task_value_resume_t<ssize_t> async_read(int file, std::string* buffer, int64_t offset, std::size_t buffer_size = 4096) {
    buffer->resize(buffer_size);
    uv_buf_t buf = uv_buf_init(buffer->data(), buffer_size);
    fan::event::uv_fs_read_awaitable read_req(file, buf, offset);
    co_await read_req;
    auto r = read_req.result();
    if (r < 0) {
      fan::throw_error("error reading file", std::to_string(r));
    }
    buffer->resize(r);
    co_return r;
  }

  fan::event::task_value_resume_t<ssize_t> async_write(int fd, const char* buffer, size_t length, int64_t offset) {
    fan::event::uv_fs_write_awaitable write_req(fd, buffer, length, offset);
    co_await write_req;
    ssize_t written = write_req.result();
    if (written < 0) {
      fan::throw_error("failed to write to file: error:"_str + uv_strerror(static_cast<int>(written)));
    }
    co_return written;
  }

  fan::event::task_t async_close(int file) {
    fan::event::uv_fs_close_awaitable close_req(file);
    co_await close_req;
  }

  
  fan::event::task_value_resume_t<intptr_t> async_size(int file) {
    fan::event::uv_fs_size_awaitable size_req(file);
    co_await size_req;
    co_return size_req.result();
  }

  fan::event::task_value_resume_t<intptr_t> async_size(const std::string& path) {
    fan::event::uv_fs_size_awaitable size_req(path);
    co_await size_req;
    co_return size_req.result();
  }

  template <typename lambda_t>
  fan::event::task_t async_read_cb(const std::string& path, lambda_t&& lambda, int buffer_size = 64) {
    int fd = co_await fan::io::file::async_open(path);
    int offset = 0;
    std::string buffer;
    buffer.resize(buffer_size);
    while (true) {
      std::size_t result = co_await fan::io::file::async_read(fd, buffer.data(), buffer.size(), offset);
      if (result == 0) {
        break;
      }
      std::string chunk(buffer.data(), result);
      if constexpr (fan::is_awaitable_v<decltype(lambda(chunk))>) {
        co_await lambda(chunk);
      }
      else {
        lambda(chunk);
      }
      offset += result;
    }
    co_await fan::io::file::async_close(fd);
  }
  fan::event::task_value_resume_t<std::string> async_read(const std::string& path, int buffer_size = 4096) {
    int fd = co_await fan::io::file::async_open(path);
    int offset = 0;
    std::string buffer;
    buffer.resize(buffer_size);
    std::string content;

    while (true) {
      std::size_t result = co_await fan::io::file::async_read(fd, buffer.data(), buffer.size(), offset);
      if (result == 0) {
        break;
      }
      content.append(buffer.data(), result);
      offset += result;
    }

    co_await fan::io::file::async_close(fd);
    co_return content;
  }

  fan::event::task_t async_write(const std::string& path, const std::string& data) {
    int fd = co_await fan::io::file::async_open(path, fan::fs_out);

    size_t offset = 0;
    size_t buffer_size = 4096;
    size_t total_written = 0;

    while (total_written < data.size()) {
      size_t remaining = data.size() - total_written;
      size_t to_write = std::min(remaining, buffer_size);

      std::string buffer(data.data() + total_written, to_write);
      std::size_t written = co_await fan::io::file::async_write(fd, buffer.data(), buffer.size(), offset + total_written);
      if (written == 0) {
        fan::throw_error("write failed");
      }
      total_written += written;
    }
    co_await fan::io::file::async_close(fd);
  }

  struct async_read_t {
    std::string path;
    int fd = -1;
    intptr_t offset = 0;
    intptr_t size = 0;

    fan::event::task_resume_t open(const std::string& file_path) {
      path = file_path;
      fd = co_await fan::io::file::async_open(path);
      size = co_await fan::io::file::async_size(fd);
    }
    fan::event::task_resume_t close() {
      co_await fan::io::file::async_close(fd);
    }
    fan::event::task_value_resume_t<std::string> read() {
      std::string buffer;
      intptr_t read = co_await fan::io::file::async_read(fd, &buffer, offset);
      if (read > 0) {
        offset += read;
      }
      if (read < 0) {
        fan::throw_error("fs read error:" + fan::event::strerror(read));
      }
      co_return buffer;
    }
  };

  struct async_write_t {
    std::string path;
    int fd = -1;
    intptr_t offset = 0;

    fan::event::task_resume_t open(const std::string& file_path) {
      path = file_path;
      fd = co_await fan::io::file::async_open(path, fan::fs_out);
    }
    fan::event::task_resume_t close() const {
      co_await fan::io::file::async_close(fd);
    }
    fan::event::task_value_resume_t<intptr_t> write(const std::string& data, std::size_t buffer_size = 4096) {
      intptr_t result = co_await fan::io::file::async_write(fd, data.data() + offset, std::min(data.size() - offset, buffer_size), offset);
      if (result > 0) {
        offset += result;
      }
      co_return result;
    }
  };
}

export namespace fan::io {

  struct ev_next_tick_awaiter {
    uv_idle_t* idle_handle = nullptr;
    std::coroutine_handle<> coro;

    bool await_ready() noexcept { return false; }

    void await_suspend(std::coroutine_handle<> h) noexcept {
      coro = h;
      idle_handle = new uv_idle_t;
      uv_idle_init(fan::event::get_event_loop(), idle_handle);
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

    void await_resume() noexcept {}
  };


  struct async_directory_iterator_t {
    std::vector<std::filesystem::directory_entry> entries;
    std::function<fan::event::task_t(const std::filesystem::directory_entry&)> callback;
    std::string base_path;
    std::string next_path;
    bool sort_alphabetically = false;
    size_t current_index = 0;
    bool stopped = false;
    bool operation_in_progress = false;
    bool switch_requested = false;
    fan::event::task_t iteration_task;
    
    void stop() {
      stopped = true;
      switch_requested = false;
    }
  };

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
      memset(req, 0, sizeof(uv_fs_t));
      req->data = state;

      int ret = uv_fs_scandir(fan::event::get_event_loop(), req, path.c_str(), 0,
        [](uv_fs_t* req) {
          auto* state = static_cast<async_directory_iterator_t*>(req->data);

          if (!state->stopped) {
            uv_dirent_t ent;
            while (uv_fs_scandir_next(req, &ent) != UV_EOF) {
              std::filesystem::path full_path = std::filesystem::path(state->base_path) / ent.name;
              state->entries.emplace_back(full_path);
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
            uv_idle_init(fan::event::get_event_loop(), idle);
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

struct cleaner_t {
  ~cleaner_t() {
    uv_loop_close(fan::event::get_event_loop());
  }
}cleaner;