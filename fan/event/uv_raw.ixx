module;
#include <cstdint>
#include <uv.h>

export module fan.event.uv_raw;

export namespace fan::uv {
  using uv_loop_t = ::uv_loop_t;
  using uv_fs_t = ::uv_fs_t;
  using uv_async_t = ::uv_async_t;
  using uv_buf_t = ::uv_buf_t;
  using uv_handle_t = ::uv_handle_t;

  inline int fs_open(
    uv_loop_t* loop,
    uv_fs_t* req,
    const char* path,
    int flags,
    int mode,
    void(*cb)(uv_fs_t*)
  ) {
    return ::uv_fs_open(loop, req, path, flags, mode, cb);
  }

  inline int fs_read(
    uv_loop_t* loop,
    uv_fs_t* req,
    int fd,
    uv_buf_t* buf,
    unsigned int nbufs,
    std::int64_t offset,
    void(*cb)(uv_fs_t*)
  ) {
    return ::uv_fs_read(loop, req, fd, buf, nbufs, offset, cb);
  }

  inline int fs_write(
    uv_loop_t* loop,
    uv_fs_t* req,
    int fd,
    const uv_buf_t* bufs,
    unsigned int nbufs,
    std::int64_t offset,
    void(*cb)(uv_fs_t*)
  ) {
    return ::uv_fs_write(loop, req, fd, bufs, nbufs, offset, cb);
  }

  inline int fs_close(
    uv_loop_t* loop,
    uv_fs_t* req,
    int fd,
    void(*cb)(uv_fs_t*)
  ) {
    return ::uv_fs_close(loop, req, fd, cb);
  }

  inline int fs_fstat(
    uv_loop_t* loop,
    uv_fs_t* req,
    int fd,
    void(*cb)(uv_fs_t*)
  ) {
    return ::uv_fs_fstat(loop, req, fd, cb);
  }

  inline int fs_stat(
    uv_loop_t* loop,
    uv_fs_t* req,
    const char* path,
    void(*cb)(uv_fs_t*)
  ) {
    return ::uv_fs_stat(loop, req, path, cb);
  }

  inline void fs_req_cleanup(uv_fs_t* req) {
    ::uv_fs_req_cleanup(req);
  }

  inline int async_init(
    uv_loop_t* loop,
    uv_async_t* handle,
    void* data,
    void(*cb)(uv_async_t*)
  ) {
    handle->data = data;
    return ::uv_async_init(loop, handle, cb);
  }

  inline int async_send(uv_async_t* handle) {
    return ::uv_async_send(handle);
  }

  inline void close_handle(uv_handle_t* handle) {
    ::uv_close(handle, nullptr);
  }

  inline void close_async(uv_async_t* handle) {
    ::uv_close(reinterpret_cast<uv_handle_t*>(handle), nullptr);
  }

  inline uv_loop_t* default_loop() {
    return ::uv_default_loop();
  }

  inline int run(uv_loop_t* loop) {
    return ::uv_run(loop, UV_RUN_DEFAULT);
  }

  inline int run_once(uv_loop_t* loop) {
    return ::uv_run(loop, UV_RUN_ONCE);
  }

  inline void stop(uv_loop_t* loop) {
    ::uv_stop(loop);
  }

  inline int loop_close(uv_loop_t* loop) {
    return ::uv_loop_close(loop);
  }

  inline const char* strerror(int err) {
    return ::uv_strerror(err);
  }

  inline void close(
    uv_handle_t* handle,
    void(*cb)(uv_handle_t*)
  ) {
    ::uv_close(handle, cb);
  }

}