module;

#include <coroutine>

export module fan.event.types;

import std;

export namespace fan {
  extern const int fs_o_append;
  extern const int fs_o_creat;
  extern const int fs_o_excl;
  extern const int fs_o_filemap;
  extern const int fs_o_random;
  extern const int fs_o_rdonly;
  extern const int fs_o_rdwr;
  extern const int fs_o_sequential;
  extern const int fs_o_short_lived;
  extern const int fs_o_temporary;
  extern const int fs_o_trunc;
  extern const int fs_o_wronly;
  
  extern const int fs_o_direct;
  extern const int fs_o_directory;
  extern const int fs_o_dsync; 
  extern const int fs_o_exlock; 
  extern const int fs_o_noatime;
  extern const int fs_o_noctty;
  extern const int fs_o_nofollow;
  extern const int fs_o_nonblock;
  extern const int fs_o_symlink;
  extern const int fs_o_sync;
  
  extern const int fs_in;
  extern const int fs_out;
  extern const int fs_app;
  extern const int fs_trunc;
  extern const int fs_ate;
  extern const int fs_nocreate;
  extern const int fs_noreplace;

  constexpr int s_irusr = 0400;  
  constexpr int s_iread = 0400;  
  constexpr int s_iwusr = 0200;  
  constexpr int s_iwrite = 0200;  
  constexpr int s_ixusr = 0100;  
  constexpr int s_iexec = 0100;  
  constexpr int s_irwxu = (s_irusr | s_iwusr | s_ixusr);  

  constexpr int s_irgrp = 040;   
  constexpr int s_iwgrp = 020;   
  constexpr int s_ixgrp = 010;   
  constexpr int s_irwxg = (s_irgrp | s_iwgrp | s_ixgrp);  

  constexpr int s_iroth = 04;    
  constexpr int s_iwoth = 02;    
  constexpr int s_ixoth = 01;    
  constexpr int s_irwxo = (s_iroth | s_iwoth | s_ixoth);  

  constexpr int s_isuid = 04000; 
  constexpr int s_isgid = 02000; 
  constexpr int s_isvtx = 01000; 

  constexpr int s_usr_rw = (s_irusr | s_iwusr);   
  constexpr int s_grp_r = s_irgrp;                 
  constexpr int s_oth_r = s_iroth;                 

  constexpr int perm_0644 = s_usr_rw | s_grp_r | s_oth_r;

  extern const int fs_change;
  extern const int fs_rename;
  extern const int eof;

  struct cancel_task_impl {};
  struct task_cancelled_exception {
    std::string message;
  };

  struct any_proxy_t {
    template<typename T>
    operator T() {
      return std::any_cast<T>(val);
    }
    std::any& val;
  };
}

export template <typename promise_type>
struct coroutine_handle_owner {
  std::coroutine_handle<promise_type> h;

  explicit coroutine_handle_owner(std::coroutine_handle<promise_type> hh) : h(hh) {}
  coroutine_handle_owner(const coroutine_handle_owner&) = delete;
  coroutine_handle_owner& operator=(const coroutine_handle_owner&) = delete;
  ~coroutine_handle_owner() {
    if (h) {
      h.destroy();
      h = nullptr;
    }
  }
};

export template<typename promise_type_t>
struct final_awaiter {
  bool await_ready() noexcept { return false; }
  std::coroutine_handle<> await_suspend(std::coroutine_handle<promise_type_t> h) noexcept {
    auto& p = h.promise();
    auto next = p.continuation;
    auto keep = std::move(p.self_keepalive);
    return next ? next : std::noop_coroutine();
  }
  void await_resume() noexcept {}
};

export template<typename T, typename suspend_type_t>
struct task_value_wrap_t;

export template<typename suspend_type_t>
struct task_auto_wrap_t;

export template <typename promise_type>
struct cancel_tag_t {
  bool await_ready() const noexcept { return !p->cancelled; }
  bool await_suspend(std::coroutine_handle<promise_type>) const noexcept { return false; }
  bool await_resume() const noexcept { return p->cancelled; }
  promise_type* p;
};

export template<typename awaitable_t, typename promise_type>
struct cancellable_awaitable_t {
  awaitable_t awaitable;
  promise_type* promise;

  cancellable_awaitable_t(awaitable_t&& a, promise_type* p)
    : awaitable(std::forward<awaitable_t>(a)), promise(p) {}

  auto await_ready() {
    if (promise->cancelled) { throw fan::task_cancelled_exception{}; }
    return awaitable.await_ready();
  }
  template<typename handle>
  auto await_suspend(handle h) {
    if (promise->cancelled) { throw fan::task_cancelled_exception{}; }
    return awaitable.await_suspend(h);
  }
  auto await_resume() {
    if (promise->cancelled) { throw fan::task_cancelled_exception{}; }
    return awaitable.await_resume();
  }
};

export template<typename T, typename suspend_type_t>
struct task_value_promise_t {
  task_value_wrap_t<T, suspend_type_t> get_return_object();
  suspend_type_t initial_suspend() noexcept { return {}; }
  auto final_suspend() noexcept { return final_awaiter<task_value_promise_t<T, suspend_type_t>>{}; }
  void return_value(T&& val) { value = std::move(val); }
  void return_value(const T& val) { value = val; }
  void unhandled_exception() {
    try {
      std::rethrow_exception(std::current_exception());
    }
    catch (const fan::task_cancelled_exception& e) {
      if constexpr (std::is_same_v<T, std::string>) { value = e.message; }
    }
    catch (...) {
      exception = std::current_exception();
    }
  }
  auto await_transform(fan::cancel_task_impl) noexcept {
    return cancel_tag_t<task_value_promise_t<T, suspend_type_t>>{ this };
  }
  template<typename awaitable>
  auto await_transform(awaitable&& a) noexcept {
    using current_promise_t = task_value_promise_t<T, suspend_type_t>;
    return cancellable_awaitable_t<awaitable, current_promise_t>(std::forward<awaitable>(a), this);
  }

  T value;
  std::exception_ptr exception = nullptr;
  std::coroutine_handle<> continuation = nullptr;
  std::shared_ptr<coroutine_handle_owner<task_value_promise_t>> self_keepalive;
  bool cancelled = false;
};

template<typename suspend_type_t>
struct task_value_promise_t<void, suspend_type_t> {
  task_value_wrap_t<void, suspend_type_t> get_return_object();
  suspend_type_t initial_suspend() noexcept { return {}; }
  auto final_suspend() noexcept { return final_awaiter<task_value_promise_t<void, suspend_type_t>>{}; }
  void return_void() {}
  void unhandled_exception() { 
    try {
      std::rethrow_exception(std::current_exception());
    }
    catch (const fan::task_cancelled_exception&) {
      return;
    }
    catch (...) {
      exception = std::current_exception();
    }
  }
  auto await_transform(fan::cancel_task_impl) noexcept {
    return cancel_tag_t<task_value_promise_t<void, suspend_type_t>>{ this };
  }
  template<typename awaitable>
  auto await_transform(awaitable&& a) noexcept {
    using current_promise_t = task_value_promise_t<void, suspend_type_t>;
    return cancellable_awaitable_t<awaitable, current_promise_t>(std::forward<awaitable>(a), this);
  }

  std::exception_ptr exception = nullptr;
  std::coroutine_handle<> continuation = nullptr;
  std::shared_ptr<coroutine_handle_owner<task_value_promise_t>> self_keepalive;
  bool cancelled = false;
};

export template<typename suspend_type_t>
struct task_auto_promise_t {
  task_auto_wrap_t<suspend_type_t> get_return_object();
  suspend_type_t initial_suspend() noexcept { return {}; }
  auto final_suspend() noexcept { return final_awaiter<task_auto_promise_t<suspend_type_t>>{}; }
  
  template<typename T>
  void return_value(T&& val) { value = std::forward<T>(val); }

  void unhandled_exception() {
    try {
      std::rethrow_exception(std::current_exception());
    }
    catch (const fan::task_cancelled_exception& e) {
      value = e.message;
    }
    catch (...) {
      exception = std::current_exception();
    }
  }
  auto await_transform(fan::cancel_task_impl) noexcept {
    return cancel_tag_t<task_auto_promise_t<suspend_type_t>>{ this };
  }
  template<typename awaitable>
  auto await_transform(awaitable&& a) noexcept {
    using current_promise_t = task_auto_promise_t<suspend_type_t>;
    return cancellable_awaitable_t<awaitable, current_promise_t>(std::forward<awaitable>(a), this);
  }

  std::any value;
  std::exception_ptr exception = nullptr;
  std::coroutine_handle<> continuation = nullptr;
  std::shared_ptr<coroutine_handle_owner<task_auto_promise_t>> self_keepalive;
  bool cancelled = false;
};

export template<typename T, typename suspend_type_t>
struct task_value_wrap_t {
  using promise_type = task_value_promise_t<T, suspend_type_t>;
  using owner_t = coroutine_handle_owner<promise_type>;

  bool await_ready() const noexcept { return owner && owner->h.done(); }
  void await_suspend(std::coroutine_handle<> cont) noexcept {
    auto& p = owner->h.promise();
    p.continuation = cont;
    if (!p.self_keepalive) { p.self_keepalive = owner; }
    
    if constexpr (std::is_same_v<suspend_type_t, std::suspend_always>) {
      owner->h.resume();
    }
  }
  T await_resume() {
    auto& p = owner->h.promise();
    if (p.exception) { std::rethrow_exception(p.exception); }
    return std::move(p.value);
  }

  bool done() const { return !owner || owner->h.done(); }
  
  std::shared_ptr<owner_t> owner;
};

template<typename suspend_type_t>
struct task_value_wrap_t<void, suspend_type_t> {
  using promise_type = task_value_promise_t<void, suspend_type_t>;
  using owner_t = coroutine_handle_owner<promise_type>;

  task_value_wrap_t() = default;
  task_value_wrap_t(std::shared_ptr<owner_t> o) : owner(std::move(o)) {}
  task_value_wrap_t(const task_value_wrap_t&) = default;
  task_value_wrap_t(task_value_wrap_t&& other) noexcept : owner(std::move(other.owner)) {
    other.owner = nullptr;
  }
  task_value_wrap_t& operator=(const task_value_wrap_t& other) {
    if (this != &other) {
      if (owner) { owner->h.promise().cancelled = true; }
      owner = other.owner;
    }
    return *this;
  }
  task_value_wrap_t& operator=(task_value_wrap_t&& other) noexcept {
    if (this != &other) {
      if (owner) { owner->h.promise().cancelled = true; }
      owner = std::move(other.owner);
      other.owner = nullptr;
    }
    return *this;
  }
  ~task_value_wrap_t() {
    if (owner) { owner->h.promise().cancelled = true; }
  }
  bool await_ready() const noexcept { return owner && owner->h.done(); }
  void await_suspend(std::coroutine_handle<> cont) noexcept {
    auto& p = owner->h.promise();
    p.continuation = cont;
    if (!p.self_keepalive) { p.self_keepalive = owner; }

    if constexpr (std::is_same_v<suspend_type_t, std::suspend_always>) {
      owner->h.resume();
    }
  }
  void await_resume() {
    auto& p = owner->h.promise();
    if (p.exception) { std::rethrow_exception(p.exception); }
  }
  bool valid() const { return owner && !owner->h.done(); }
  void join() {
    if (!owner) return;
    auto& h = owner->h;
    while (!h.done()) { h.resume(); }
  }
  void resume() {
    if (!owner) return;
    owner->h.resume();
  }
  void request_stop() {
    if (!owner) return;
    owner->h.promise().cancelled = true;
  }
  void stop_and_join() {
    if (valid()) { request_stop(); join(); }
  }
  void destroy() { *this = {}; }

  bool done() const { return !owner || owner->h.done(); }

  std::shared_ptr<owner_t> owner;
};

export template<typename suspend_type_t>
struct task_auto_wrap_t {
  using promise_type = task_auto_promise_t<suspend_type_t>;
  using owner_t = coroutine_handle_owner<promise_type>;

  task_auto_wrap_t() = default;
  task_auto_wrap_t(std::shared_ptr<owner_t> o) : owner(std::move(o)) {}
  task_auto_wrap_t(const task_auto_wrap_t&) = default;
  task_auto_wrap_t(task_auto_wrap_t&& other) noexcept : owner(std::move(other.owner)) {
    other.owner = nullptr;
  }
  task_auto_wrap_t& operator=(const task_auto_wrap_t& other) {
    if (this != &other) {
      if (owner) { owner->h.promise().cancelled = true; }
      owner = other.owner;
    }
    return *this;
  }
  task_auto_wrap_t& operator=(task_auto_wrap_t&& other) noexcept {
    if (this != &other) {
      if (owner) { owner->h.promise().cancelled = true; }
      owner = std::move(other.owner);
      other.owner = nullptr;
    }
    return *this;
  }
  ~task_auto_wrap_t() {
    if (owner) { owner->h.promise().cancelled = true; }
  }
  bool await_ready() const noexcept { return owner && owner->h.done(); }
  void await_suspend(std::coroutine_handle<> cont) noexcept {
    auto& p = owner->h.promise();
    p.continuation = cont;
    if (!p.self_keepalive) { p.self_keepalive = owner; }

    if constexpr (std::is_same_v<suspend_type_t, std::suspend_always>) {
      owner->h.resume();
    }
  }
  fan::any_proxy_t await_resume() {
    auto& p = owner->h.promise();
    if (p.exception) { std::rethrow_exception(p.exception); }
    return fan::any_proxy_t{p.value};
  }
  fan::any_proxy_t get() {
    return fan::any_proxy_t{owner->h.promise().value};
  }
  bool valid() const { return owner && !owner->h.done(); }
  void join() {
    if (!owner) return;
    auto& h = owner->h;
    while (!h.done()) { h.resume(); }
  }
  void resume() {
    if (!owner) return;
    owner->h.resume();
  }
  void request_stop() {
    if (!owner) return;
    owner->h.promise().cancelled = true;
  }
  void stop_and_join() {
    if (valid()) { request_stop(); join(); }
  }
  void destroy() { *this = {}; }

  bool done() const { return !owner || owner->h.done(); }

  std::shared_ptr<owner_t> owner;
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

template<typename suspend_t>
task_auto_wrap_t<suspend_t>
task_auto_promise_t<suspend_t>::get_return_object() {
  using promise_type = task_auto_promise_t<suspend_t>;
  auto h = std::coroutine_handle<promise_type>::from_promise(*this);
  auto own = std::make_shared<coroutine_handle_owner<promise_type>>(h);
  self_keepalive = own;
  return task_auto_wrap_t<suspend_t>{ std::move(own) };
}

export namespace fan::event {
  using loop_t = void*;
  using idle_id_t = void*;

  using wait_t = task_value_wrap_t<void, std::suspend_always>;
  using run_t = task_value_wrap_t<void, std::suspend_never>;

  template <typename T>
  using waitv_t = task_value_wrap_t<T, std::suspend_always>;

  template <typename T>
  using runv_t = task_value_wrap_t<T, std::suspend_never>;

  using task_t = run_t;
  // auto
  using waita_t = task_auto_wrap_t<std::suspend_always>;
  // auto
  using runa_t = task_auto_wrap_t<std::suspend_never>;
  using cancel_task = cancel_task_impl;
}