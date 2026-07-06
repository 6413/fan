module;
#include <cstdint>
#include <cstddef>
#include <uv.h>

export module fan.event.uv_raw;

export namespace fan::uv {

  // ─── types ───────────────────────────────────────────────────────────────

  using loop_t        = ::uv_loop_t;
  using handle_t      = ::uv_handle_t;
  using stream_t      = ::uv_stream_t;
  using pipe_t        = ::uv_pipe_t;
  using process_t     = ::uv_process_t;
  using timer_t       = ::uv_timer_t;
  using idle_t        = ::uv_idle_t;
  using async_t       = ::uv_async_t;
  using signal_t      = ::uv_signal_t;
  using fs_t          = ::uv_fs_t;
  using buf_t         = ::uv_buf_t;
  using process_options_t = ::uv_process_options_t;
  using stdio_container_t = ::uv_stdio_container_t;
  using timespec_t    = ::uv_timespec_t;
  using stat_t        = ::uv_stat_t;

  using timer_cb      = ::uv_timer_cb;
  using idle_cb       = ::uv_idle_cb;
  using async_cb      = ::uv_async_cb;
  using signal_cb     = ::uv_signal_cb;
  using fs_cb         = ::uv_fs_cb;
  using read_cb       = ::uv_read_cb;
  using write_cb      = ::uv_write_cb;
  using close_cb      = ::uv_close_cb;
  using exit_cb       = ::uv_exit_cb;
  using alloc_cb      = ::uv_alloc_cb;

  using fs_req_type   = ::uv_fs_type;
  using run_mode      = ::uv_run_mode;

  inline constexpr auto run_default  = UV_RUN_DEFAULT;
  inline constexpr auto run_once     = UV_RUN_ONCE;
  inline constexpr auto run_nowait   = UV_RUN_NOWAIT;

  inline constexpr auto create_pipe  = UV_CREATE_PIPE;
  inline constexpr auto readable_pipe= UV_READABLE_PIPE;
  inline constexpr auto writable_pipe= UV_WRITABLE_PIPE;
  inline constexpr auto ignore       = UV_IGNORE;

  // ─── loop ────────────────────────────────────────────────────────────────

  inline loop_t* default_loop() {
    return ::uv_default_loop();
  }

  inline int loop_init(loop_t* loop) {
    return ::uv_loop_init(loop);
  }

  inline int loop_close(loop_t* loop) {
    return ::uv_loop_close(loop);
  }

  inline int run(loop_t* loop, run_mode mode = UV_RUN_DEFAULT) {
    return ::uv_run(loop, mode);
  }

  inline void stop(loop_t* loop) {
    ::uv_stop(loop);
  }

  inline uint64_t now(loop_t* loop) {
    return ::uv_now(loop);
  }

  inline void update_time(loop_t* loop) {
    ::uv_update_time(loop);
  }

  // ─── handle ──────────────────────────────────────────────────────────────

  inline int is_closing(const handle_t* handle) {
    return ::uv_is_closing(handle);
  }

  inline void close(handle_t* handle, close_cb cb = nullptr) {
    if (handle == nullptr || handle->loop == nullptr || handle->type == UV_UNKNOWN_HANDLE || is_closing(handle)) {
      if (cb) {
        cb(handle);
      }
      return;
    }
    ::uv_close(handle, cb);
  }

  inline int is_active(const handle_t* handle) {
    return ::uv_is_active(handle);
  }

  // ─── timer ───────────────────────────────────────────────────────────────

  inline int timer_init(loop_t* loop, timer_t* handle) {
    return ::uv_timer_init(loop, handle);
  }

  inline int timer_start(timer_t* handle, timer_cb cb, uint64_t timeout, uint64_t repeat) {
    return ::uv_timer_start(handle, cb, timeout, repeat);
  }

  inline int timer_stop(timer_t* handle) {
    return ::uv_timer_stop(handle);
  }

  inline int timer_again(timer_t* handle) {
    return ::uv_timer_again(handle);
  }

  inline void timer_set_repeat(timer_t* handle, uint64_t repeat) {
    ::uv_timer_set_repeat(handle, repeat);
  }

  inline uint64_t timer_get_repeat(const timer_t* handle) {
    return ::uv_timer_get_repeat(handle);
  }

  // ─── idle ────────────────────────────────────────────────────────────────

  inline int idle_init(loop_t* loop, idle_t* handle) {
    return ::uv_idle_init(loop, handle);
  }

  inline int idle_start(idle_t* handle, idle_cb cb) {
    return ::uv_idle_start(handle, cb);
  }

  inline int idle_stop(idle_t* handle) {
    return ::uv_idle_stop(handle);
  }

  // ─── async ───────────────────────────────────────────────────────────────

  inline int async_init(loop_t* loop, async_t* handle, async_cb cb) {
    return ::uv_async_init(loop, handle, cb);
  }

  inline int async_send(async_t* handle) {
    return ::uv_async_send(handle);
  }

  // ─── signal ──────────────────────────────────────────────────────────────

  inline int signal_init(loop_t* loop, signal_t* handle) {
    return ::uv_signal_init(loop, handle);
  }

  inline int signal_start(signal_t* handle, signal_cb cb, int signum) {
    return ::uv_signal_start(handle, cb, signum);
  }

  inline int signal_stop(signal_t* handle) {
    return ::uv_signal_stop(handle);
  }

  // ─── pipe / stream ───────────────────────────────────────────────────────

  inline int pipe_init(loop_t* loop, pipe_t* handle, int ipc) {
    return ::uv_pipe_init(loop, handle, ipc);
  }

  inline int read_start(stream_t* stream, alloc_cb alloc, read_cb read) {
    return ::uv_read_start(stream, alloc, read);
  }

  inline int read_stop(stream_t* stream) {
    return ::uv_read_stop(stream);
  }

  inline buf_t buf_init(char* base, unsigned int len) {
    return ::uv_buf_init(base, len);
  }

  // ─── process ─────────────────────────────────────────────────────────────

  inline int spawn(loop_t* loop, process_t* handle, const process_options_t* options) {
    return ::uv_spawn(loop, handle, options);
  }

  inline int process_kill(process_t* handle, int signum) {
    return ::uv_process_kill(handle, signum);
  }

  inline int kill(int pid, int signum) {
    return ::uv_kill(pid, signum);
  }

  inline uint64_t process_get_pid(const process_t* handle) {
    return ::uv_process_get_pid(handle);
  }

  // ─── filesystem ──────────────────────────────────────────────────────────

  inline int fs_open(loop_t* loop, fs_t* req, const char* path, int flags, int mode, fs_cb cb) {
    return ::uv_fs_open(loop, req, path, flags, mode, cb);
  }

  inline int fs_close(loop_t* loop, fs_t* req, int fd, fs_cb cb) {
    return ::uv_fs_close(loop, req, fd, cb);
  }

  inline int fs_read(loop_t* loop, fs_t* req, int fd, buf_t* bufs, unsigned int nbufs, int64_t offset, fs_cb cb) {
    return ::uv_fs_read(loop, req, fd, bufs, nbufs, offset, cb);
  }

  inline int fs_write(loop_t* loop, fs_t* req, int fd, const buf_t* bufs, unsigned int nbufs, int64_t offset, fs_cb cb) {
    return ::uv_fs_write(loop, req, fd, bufs, nbufs, offset, cb);
  }

  inline int fs_stat(loop_t* loop, fs_t* req, const char* path, fs_cb cb) {
    return ::uv_fs_stat(loop, req, path, cb);
  }

  inline int fs_fstat(loop_t* loop, fs_t* req, int fd, fs_cb cb) {
    return ::uv_fs_fstat(loop, req, fd, cb);
  }

  inline int fs_unlink(loop_t* loop, fs_t* req, const char* path, fs_cb cb) {
    return ::uv_fs_unlink(loop, req, path, cb);
  }

  inline int fs_mkdir(loop_t* loop, fs_t* req, const char* path, int mode, fs_cb cb) {
    return ::uv_fs_mkdir(loop, req, path, mode, cb);
  }

  inline int fs_rename(loop_t* loop, fs_t* req, const char* path, const char* new_path, fs_cb cb) {
    return ::uv_fs_rename(loop, req, path, new_path, cb);
  }

  inline int fs_scandir(loop_t* loop, fs_t* req, const char* path, int flags, fs_cb cb) {
    return ::uv_fs_scandir(loop, req, path, flags, cb);
  }

  inline int fs_scandir_next(fs_t* req, ::uv_dirent_t* ent) {
    return ::uv_fs_scandir_next(req, ent);
  }

  inline void fs_req_cleanup(fs_t* req) {
    ::uv_fs_req_cleanup(req);
  }

  inline ssize_t fs_get_result(const fs_t* req) {
    return req->result;
  }

  inline stat_t* fs_get_stat(fs_t* req) {
    return &req->statbuf;
  }

  // ─── error ───────────────────────────────────────────────────────────────

  inline const char* strerror(int err) {
    return ::uv_strerror(err);
  }

  inline const char* err_name(int err) {
    return ::uv_err_name(err);
  }

  // ─── sleep ───────────────────────────────────────────────────────────────

  inline void sleep(unsigned int msec) {
    ::uv_sleep(msec);
  }

  // ─── walk / introspection ────────────────────────────────────────────────

  using walk_cb = ::uv_walk_cb;

  inline void walk(loop_t* loop, walk_cb cb, void* arg) {
    ::uv_walk(loop, cb, arg);
  }

  inline const char* handle_type_name(::uv_handle_type type) {
    return ::uv_handle_type_name(type);
  }

  // ─── fs_event ────────────────────────────────────────────────────────────

  using fs_event_t = ::uv_fs_event_t;
  using fs_event_cb = ::uv_fs_event_cb;

  inline int fs_event_init(loop_t* loop, fs_event_t* handle) {
    return ::uv_fs_event_init(loop, handle);
  }

  inline int fs_event_start(fs_event_t* handle, fs_event_cb cb, const char* path, unsigned int flags) {
    return ::uv_fs_event_start(handle, cb, path, flags);
  }

  inline int fs_event_stop(fs_event_t* handle) {
    return ::uv_fs_event_stop(handle);
  }

  // ─── poll ────────────────────────────────────────────────────────────────

  using poll_t = ::uv_poll_t;
  using poll_cb = ::uv_poll_cb;

  inline int poll_init(loop_t* loop, poll_t* handle, int fd) {
    return ::uv_poll_init(loop, handle, fd);
  }

  inline int poll_start(poll_t* handle, int events, poll_cb cb) {
    return ::uv_poll_start(handle, events, cb);
  }

  inline int poll_stop(poll_t* handle) {
    return ::uv_poll_stop(handle);
  }

  // ─── dirent ──────────────────────────────────────────────────────────────

  using dirent_t = ::uv_dirent_t;

  inline constexpr unsigned int fs_event_recursive = UV_FS_EVENT_RECURSIVE;
}
