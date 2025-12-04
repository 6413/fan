//
/*
#pragma once

#include <functional>
#include <unordered_map>
#include <utility>
*/

#define AUTO_RAII_FIELD(nr_t, name, owner_t, ...)                      \
  owner_t(const owner_t& o) : name(o.name) { name.rebind(this); }     \
  owner_t(owner_t&& o)      : name(std::move(o.name)) {               \
    name.rebind(this);                                               \
  }                                                                   \
  owner_t& operator=(const owner_t& o) {                              \
    if (this != &o) { name = o.name; name.rebind(this); }             \
    return *this;                                                     \
  }                                                                   \
  owner_t& operator=(owner_t&& o) {                                   \
    if (this != &o) { name = std::move(o.name); name.rebind(this); }  \
    return *this;                                                     \
  }

#define AUTO_RAII_DEBUG_FIELD(nr_t, name, owner_t, ...)                \
  owner_t(const owner_t& o) : name(o.name) { name.rebind(this); }     \
  owner_t(owner_t&& o)      : name(std::move(o.name)) {               \
    name.rebind(this);                                               \
  }                                                                   \
  owner_t& operator=(const owner_t& o) {                              \
    if (this != &o) { name = o.name; name.rebind(this); }             \
    return *this;                                                     \
  }                                                                   \
  owner_t& operator=(owner_t&& o) {                                   \
    if (this != &o) { name = std::move(o.name); name.rebind(this); }  \
    return *this;                                                     \
  }


template <
  typename nr_t,
  typename user_t,
  typename... params
>
struct raii_nr_t : nr_t {

  using fn_t      = std::function<void(user_t*, params...)>;
  using add_fn    = std::function<nr_t(user_t*, fn_t)>;
  using remove_fn = std::function<void(user_t*, const nr_t&)>;

  struct state_t {
    user_t*  user_ptr = nullptr;
    add_fn   add_cb;
    remove_fn remove_cb;
    fn_t     user_fn;
  };

  [[no_unique_address]]
  state_t state;

  user_t* user() const { return state.user_ptr; }
  void rebind(user_t* p) { state.user_ptr = p; }

  operator const nr_t&() const { return *static_cast<const nr_t*>(this); }
  operator nr_t&()             { return *static_cast<nr_t*>(this); }

  raii_nr_t() { nr_t::sic(); }
  ~raii_nr_t() { remove(); }

  fn_t make_wrapper() const {
    return [this](user_t*, params... ps) {
      state.user_fn(state.user_ptr, ps...);
    };
  }

  template <typename fn_raw_t>
  raii_nr_t(user_t* p, add_fn add, remove_fn remove, fn_raw_t&& fn_raw) {
    nr_t::sic();
    state.user_ptr = p;
    state.add_cb   = std::move(add);
    state.remove_cb = std::move(remove);
    state.user_fn =
      [f = std::forward<fn_raw_t>(fn_raw)]
      (user_t* self, auto... ps) { f(self, ps...); };
    create_new();
  }

  raii_nr_t(const raii_nr_t& o) {
    nr_t::sic();
    state = o.state;
    create_new();
  }

  raii_nr_t(raii_nr_t&& o) noexcept {
    nr_t::sic();
    state = o.state;
    create_new();
    o.remove();
  }

  raii_nr_t& operator=(const raii_nr_t& o) {
    if (this != &o) {
      remove();
      nr_t::sic();
      state = o.state;
      create_new();
    }
    return *this;
  }

  raii_nr_t& operator=(raii_nr_t&& o) noexcept {
    if (this != &o) {
      remove();
      nr_t::sic();
      state = o.state;
      create_new();
      o.remove();
    }
    return *this;
  }

  void create_new() {
    if (state.user_ptr && state.add_cb && state.user_fn) {
      auto wrapper = make_wrapper();
      *static_cast<nr_t*>(this) =
        state.add_cb(state.user_ptr, wrapper);
    }
  }

  void remove() {
    if (!nr_t::iic()) {
      if (state.user_ptr && state.remove_cb)
        state.remove_cb(state.user_ptr, *this);
      nr_t::sic();
    }
  }
};

template <
  typename nr_t,
  typename user_t,
  typename... params
>
struct raii_nr_debug_t : raii_nr_t<nr_t, user_t, params...> {

  using base_t = raii_nr_t<nr_t, user_t, params...>;
  using fn_t   = typename base_t::fn_t;
  using base_t::state;

  static inline std::unordered_map<void*, bool> alive_map;

  void* key = nullptr;
  fn_t stored;

  raii_nr_debug_t() { nr_t::sic(); }

  void make_key() {
    key = new int(0);
    alive_map[key] = true;
  }

  void delete_key() {
    alive_map.erase(key);
    delete static_cast<int*>(key);
    key = nullptr;
  }

  template <typename F>
  fn_t wrap(F&& raw) {
    return fn_t(std::forward<F>(raw));
  }

  fn_t wrap_debug() const {
    void* k = key;
    return [this, k](user_t* self, params... ps) {
      if (!alive_map[k]) return;
      stored(self, ps...);
    };
  }

  template <typename fn_raw_t>
  raii_nr_debug_t(user_t* p,
      typename base_t::add_fn add,
      typename base_t::remove_fn remove,
      fn_raw_t&& fn_raw)
  {
    nr_t::sic();
    make_key();
    stored = wrap(std::forward<fn_raw_t>(fn_raw));
    state.user_ptr = p;
    state.add_cb   = std::move(add);
    state.remove_cb = std::move(remove);
    state.user_fn  = wrap_debug();
    base_t::create_new();
  }

  raii_nr_debug_t(const raii_nr_debug_t& o) {
    nr_t::sic();
    state  = o.state;
    stored = o.stored;
    make_key();
    state.user_fn = wrap_debug();
    base_t::create_new();
  }

  raii_nr_debug_t(raii_nr_debug_t&& o) noexcept {
    nr_t::sic();
    state  = o.state;
    stored = std::move(o.stored);
    make_key();
    state.user_fn = wrap_debug();
    base_t::create_new();
    o.delete_key();
    o.remove();
  }

  ~raii_nr_debug_t() {
    if (key) delete_key();
  }

  raii_nr_debug_t& operator=(const raii_nr_debug_t& o) {
    if (this != &o) {
      if (key) delete_key();
      base_t::remove();
      nr_t::sic();
      state  = o.state;
      stored = o.stored;
      make_key();
      state.user_fn = wrap_debug();
      base_t::create_new();
    }
    return *this;
  }

  raii_nr_debug_t& operator=(raii_nr_debug_t&& o) noexcept {
    if (this != &o) {
      if (key) delete_key();
      base_t::remove();
      nr_t::sic();
      state  = o.state;
      stored = std::move(o.stored);
      make_key();
      state.user_fn = wrap_debug();
      base_t::create_new();
      o.delete_key();
      o.remove();
    }
    return *this;
  }

  static void verify() {
    for (auto& i : alive_map)
      if (i.second) printf("DBG FAIL %p\n", i.first);
  }
};