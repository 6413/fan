module;

#include <coroutine>
#include <memory>
#include <exception>

#include <uv.h>
#undef min
#undef max

export module fan.event:types;

export namespace fan{
  constexpr int fs_o_append      = UV_FS_O_APPEND;
  constexpr int fs_o_creat       = UV_FS_O_CREAT;
  constexpr int fs_o_excl        = UV_FS_O_EXCL;
  constexpr int fs_o_filemap     = UV_FS_O_FILEMAP;
  constexpr int fs_o_random      = UV_FS_O_RANDOM;
  constexpr int fs_o_rdonly     = UV_FS_O_RDONLY;
  constexpr int fs_o_rdwr        = UV_FS_O_RDWR;
  constexpr int fs_o_sequential  = UV_FS_O_SEQUENTIAL;
  constexpr int fs_o_short_lived = UV_FS_O_SHORT_LIVED;
  constexpr int fs_o_temporary   = UV_FS_O_TEMPORARY;
  constexpr int fs_o_trunc       = UV_FS_O_TRUNC;
  constexpr int fs_o_wronly      = UV_FS_O_WRONLY;
  
  constexpr int fs_o_direct      = UV_FS_O_DIRECT;
  constexpr int fs_o_directory   = UV_FS_O_DIRECTORY;
  constexpr int fs_o_dsync       = UV_FS_O_DSYNC; 
  constexpr int fs_o_exlock      = UV_FS_O_EXLOCK; 
  constexpr int fs_o_noatime     = UV_FS_O_NOATIME;
  constexpr int fs_o_noctty      = UV_FS_O_NOCTTY;
  constexpr int fs_o_nofollow    = UV_FS_O_NOFOLLOW;
  constexpr int fs_o_nonblock    = UV_FS_O_NONBLOCK;
  constexpr int fs_o_symlink     = UV_FS_O_SYMLINK;
  constexpr int fs_o_sync        = UV_FS_O_SYNC;
  
  constexpr int fs_in        = UV_FS_O_RDONLY;
  constexpr int fs_out       = UV_FS_O_WRONLY | UV_FS_O_CREAT | UV_FS_O_TRUNC;
  constexpr int fs_app       = UV_FS_O_WRONLY | UV_FS_O_APPEND;
  constexpr int fs_trunc     = UV_FS_O_TRUNC;
  constexpr int fs_ate       = UV_FS_O_RDWR;
  constexpr int fs_nocreate  = UV_FS_O_EXCL;
  constexpr int fs_noreplace = UV_FS_O_EXCL;


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
  constexpr int eof = UV_EOF;
}

template <typename promise_type>
struct coroutine_handle_owner {
  std::coroutine_handle<promise_type> h;

  explicit coroutine_handle_owner(std::coroutine_handle<promise_type> hh) : h(hh) {}
  coroutine_handle_owner(const coroutine_handle_owner&) = delete;
  coroutine_handle_owner& operator=(const coroutine_handle_owner&) = delete;

  ~coroutine_handle_owner() {
    if (h) {
      h.destroy();
      h = 0;
    }
  }
};

template<typename promise_type_t>
struct final_awaiter {
  bool await_ready() noexcept {
    return false;
  }

  std::coroutine_handle<> await_suspend(std::coroutine_handle<promise_type_t> h) noexcept {
    auto& p = h.promise();
    auto next = p.continuation;
    auto keep = std::move(p.self_keepalive);
    return next ? next : std::noop_coroutine();
  }

  void await_resume() noexcept {}
};

template<typename T, typename suspend_type_t>
struct task_value_wrap_t;

template <typename promise_type>
struct cancel_tag_t {
  promise_type* p;
  bool await_ready() const noexcept {
    return !p->cancelled;
  }
  bool await_suspend(std::coroutine_handle<promise_type>) const noexcept {
    return false;
  }
  bool await_resume() const noexcept {
    return p->cancelled;
  }
};

struct cancel_task_impl {};

struct task_cancelled_exception {};

template<typename awaitable, typename promise_type>
struct cancellable_awaitable_t {
  awaitable awaitable;
  promise_type* promise;

  auto await_ready() {
    if (promise->cancelled) {
      throw task_cancelled_exception{};
    }
    return awaitable.await_ready();
  }

  template<typename handle>
  auto await_suspend(handle h) {
    if (promise->cancelled) {
      throw task_cancelled_exception{};
    }
    return awaitable.await_suspend(h);
  }

  auto await_resume() {
    if (promise->cancelled) {
      throw task_cancelled_exception{};
    }
    return awaitable.await_resume();
  }
};

template<typename T, typename suspend_type_t>
struct task_value_promise_t {
  T value;
  std::exception_ptr exception = nullptr;
  std::coroutine_handle<> continuation = nullptr;
  std::shared_ptr<coroutine_handle_owner<task_value_promise_t>> self_keepalive;

  bool cancelled = false;

  task_value_wrap_t<T, suspend_type_t> get_return_object();
  suspend_type_t initial_suspend() noexcept { return {}; }
  auto final_suspend() noexcept {
    return final_awaiter<task_value_promise_t<T, suspend_type_t>>{};
  }
  void return_value(T&& val) { value = std::move(val); }
  void return_value(const T& val) { value = val; }
  void unhandled_exception() { 
    try {
      std::rethrow_exception(std::current_exception());
    } catch (const task_cancelled_exception&) {
      return;
    } catch (...) {
      exception = std::current_exception();
    }
  }

  auto await_transform(cancel_task_impl) noexcept {
    return cancel_tag_t<task_value_promise_t<T, suspend_type_t>>{ this };
  }

  template<typename awaitable>
  auto await_transform(awaitable&& a) noexcept {
    return cancellable_awaitable_t<awaitable, task_value_promise_t<T, suspend_type_t>>{
      std::forward<awaitable>(a), this
    };
  }
};

template<typename suspend_type_t>
struct task_value_promise_t<void, suspend_type_t> {
  std::exception_ptr exception = nullptr;
  std::coroutine_handle<> continuation = nullptr;
  std::shared_ptr<coroutine_handle_owner<task_value_promise_t>> self_keepalive;

  bool cancelled = false;

  task_value_wrap_t<void, suspend_type_t> get_return_object();
  suspend_type_t initial_suspend() noexcept { return {}; }
  auto final_suspend() noexcept {
    return final_awaiter<task_value_promise_t<void, suspend_type_t>>{};
  }
  void return_void() {}
  void unhandled_exception() { 
    try {
      std::rethrow_exception(std::current_exception());
    } catch (const task_cancelled_exception&) {
      return;
    } catch (...) {
      exception = std::current_exception();
    }
  }

  auto await_transform(cancel_task_impl) noexcept {
    return cancel_tag_t<task_value_promise_t<void, suspend_type_t>>{ this };
  }

  template<typename awaitable>
  auto await_transform(awaitable&& a) noexcept {
    return cancellable_awaitable_t<awaitable, task_value_promise_t<void, suspend_type_t>>{
      std::forward<awaitable>(a), this
    };
  }
};

template<typename T, typename suspend_type_t>
struct task_value_wrap_t {
  using promise_type = task_value_promise_t<T, suspend_type_t>;
  using owner_t = coroutine_handle_owner<promise_type>;
  std::shared_ptr<owner_t> owner;

  bool await_ready() const noexcept {
    return owner && owner->h.done();
  }

  void await_suspend(std::coroutine_handle<> cont) noexcept {
    auto& p = owner->h.promise();
    p.continuation = cont;
    if (!p.self_keepalive) {
      p.self_keepalive = owner;
    }
  }

  T await_resume() {
    auto& p = owner->h.promise();
    if (p.exception) {
      std::rethrow_exception(p.exception);
    }
    return std::move(p.value);
  }
};

template<typename suspend_type_t>
struct task_value_wrap_t<void, suspend_type_t> {
  using promise_type = task_value_promise_t<void, suspend_type_t>;
  using owner_t = coroutine_handle_owner<promise_type>;
  std::shared_ptr<owner_t> owner;

  bool await_ready() const noexcept {
    return owner && owner->h.done();
  }
  void await_suspend(std::coroutine_handle<> cont) noexcept {
    auto& p = owner->h.promise();
    p.continuation = cont;
    if (!p.self_keepalive) {
      p.self_keepalive = owner;
    }
  }
  void await_resume() {
    auto& p = owner->h.promise();
    if (p.exception) {
      std::rethrow_exception(p.exception);
    }
  }
  bool valid() const {
    return owner && !owner->h.done();
  }
  void join() {
    if (!owner) return;
    auto& h = owner->h;
    while (!h.done()) {
      h.resume();
    }
  }
  void request_stop() {
    if (!owner) return;
    owner->h.promise().cancelled = true;
  }
  void stop_and_join() {
    if (valid()) {
      request_stop();
      join();
    }
  }
  void destroy() {
    *this = {};
  }
};

template<typename T, typename suspend_t>
task_value_wrap_t<T, suspend_t>
task_value_promise_t<T, suspend_t>::get_return_object() {
  using promise_type = task_value_promise_t<T, suspend_t>;
  auto h = std::coroutine_handle<promise_type>::from_promise(*this);
  auto own = std::make_shared<coroutine_handle_owner<promise_type>>(h);
  self_keepalive = own;
  return task_value_wrap_t<T, suspend_t>{ std::move(own) };
}

template<typename suspend_t>
task_value_wrap_t<void, suspend_t>
task_value_promise_t<void, suspend_t>::get_return_object() {
  using promise_type = task_value_promise_t<void, suspend_t>;
  auto h = std::coroutine_handle<promise_type>::from_promise(*this);
  auto own = std::make_shared<coroutine_handle_owner<promise_type>>(h);
  self_keepalive = own;
  return task_value_wrap_t<void, suspend_t>{ std::move(own) };
}

export namespace fan::event {
  using loop_t = uv_loop_t*;
  using idle_id_t = uv_idle_t*;

  using task_suspend_t = task_value_wrap_t<void, std::suspend_always>;
  using task_resume_t = task_value_wrap_t<void, std::suspend_never>;

  template <typename T>
  using task_value_t = task_value_wrap_t<T, std::suspend_always>;

  template <typename T>
  using task_value_resume_t = task_value_wrap_t<T, std::suspend_never>;

  using task_t = task_resume_t;
  using cancel_task = cancel_task_impl;
}