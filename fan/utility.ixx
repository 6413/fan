module;

namespace raii_build {
  #include <fan/types/raii_nr.h>
}

export module fan.utility;

import std;

export import fan.mpl;
import fan.types;
import fan.memory;
import fan.time;

export namespace fan {

  template<typename It>
  struct iterator_traits {
    using reference = decltype(*(declval<It>()));
  };

  template<typename container_t>
  struct bll_iterator_t {
    using node_t      = decltype(std::declval<container_t>().GetNodeFirst());
    using value_type  = decltype(std::declval<container_t>()[std::declval<node_t>()]);
    using reference   = value_type&;
    using index_type  = node_t;

    bll_iterator_t(container_t* c, node_t n) 
      : container(c), current(n) {}

    auto& operator*() const {
      if constexpr (requires(container_t* c, node_t n) { c->StartSafeNext(n); }) {
        container->StartSafeNext(current);
        safe_next_active = true;
      }
      return (*container)[current];
    }
    auto get_index() const {
      return current;
    }
    bll_iterator_t& operator++() {
      if constexpr (requires(container_t* c, node_t n) { c->StartSafeNext(n); }) {
        current = container->EndSafeNext();
        safe_next_active = false;
      }
      else if constexpr (requires(node_t n, container_t* c) { n.Next(c); }) {
        current = current.Next(container);
      }
      return *this;
    }

    bool operator!=(const bll_iterator_t& other) const {
      return current != other.current;
    }

    container_t* container;
    node_t       current;
    mutable bool safe_next_active = false;
  };

  namespace fan_detail {
    template<typename T>
    struct iterator_traits {
      using reference = decltype(*(declval<T>()));
    };

    template<typename T>
    concept has_get_node_first = requires(T& t) {
      { t.GetNodeFirst() };
      { t.dst };
    };

    template<typename T>
    auto get_begin(T& container) {
      using base_t = std::remove_const_t<T>;
      if constexpr (has_get_node_first<base_t>) {
        return bll_iterator_t<base_t>{
          const_cast<base_t*>(&container),
          const_cast<base_t&>(container).GetNodeFirst()
        };
      }
      else {
        return container.begin();
      }
    }

    template<typename T>
    auto get_end(T& container) {
      using base_t = std::remove_const_t<T>;
      if constexpr (has_get_node_first<base_t>) {
        return bll_iterator_t<base_t>{
          const_cast<base_t*>(&container),
          const_cast<base_t&>(container).dst
        };
      }
      else {
        return container.end();
      }
    }

    template<typename T>
    auto get_first(T& container) {
      using base_t = std::remove_const_t<T>;
      if constexpr (has_get_node_first<base_t>) {
        return const_cast<base_t&>(container).GetNodeFirst();
      }
      else {
        return typename base_t::size_type(0);
      }
    }

    template<typename T>
    auto get_size(T& container) {
      using base_t = std::remove_const_t<T>;
      if constexpr (has_get_node_first<base_t>) {
        return const_cast<base_t&>(container).dst;
      }
      else {
        return const_cast<base_t&>(container).size();
      }
    }
  }

  template <typename T>
  struct enumerate_iterator_t {
    template<typename U>
    static auto get_index_type_impl(int) -> typename U::index_type;
    template<typename U>
    static auto get_index_type_impl(...) -> std::size_t;

    using iter_index_t = decltype(get_index_type_impl<T>(0));
    using iter_reference = typename fan_detail::iterator_traits<T>::reference;
    using iter_value = std::remove_reference_t<iter_reference>;

    struct reference_proxy {
      iter_index_t index;
      iter_reference value;
      operator pair<iter_index_t, iter_value>() const {
        return {index, value};
      }
    };

    using value_type = pair<iter_index_t, iter_value>;
    using reference = reference_proxy;
    using pointer = void;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::forward_iterator_tag;

    enumerate_iterator_t(T iter, iter_index_t index) : _iter(iter), _index(index) {}
    reference operator*() const {
      if constexpr (requires { _iter.get_index(); }) {
        return {_iter.get_index(), *_iter};
      }
      else {
        return {_index, *_iter};
      }
    }
    enumerate_iterator_t& operator++() {
      ++_iter;
      if constexpr (!requires { _iter.get_index(); }) {
        ++_index;
      }
      return *this;
    }
    bool operator!=(const enumerate_iterator_t& other) const {
      return _iter != other._iter;
    }
    T _iter;
    iter_index_t _index;
  };

  template <typename container_t>
  struct enumerate_view_t {
    using base_t = std::remove_reference_t<container_t>;
    using iterator = enumerate_iterator_t<
      conditional_t<
      is_const_v<base_t>,
      decltype(fan_detail::get_begin(declval<const base_t&>())),
      decltype(fan_detail::get_begin(declval<base_t&>()))
      >
    >;

    enumerate_view_t(container_t&& container) : _container(std::forward<container_t>(container)) {}

    iterator begin() {
      auto it = fan_detail::get_begin(_container);
      using iter_t = decltype(it);

      if constexpr (requires(iter_t t) { t.get_index(); }) {
        return {it, typename iterator::iter_index_t{}};
      }
      else {
        return {it, fan_detail::get_first(_container)};
      }
    }

    iterator end() {
      auto it = fan_detail::get_end(_container);
      using iter_t = decltype(it);

      if constexpr (requires(iter_t t) { t.get_index(); }) {
        return {it, typename iterator::iter_index_t{}};
      }
      else {
        return {it, fan_detail::get_size(_container)};
      }
    }

    iterator begin() const {
      return const_cast<enumerate_view_t*>(this)->begin();
    }

    iterator end() const {
      return const_cast<enumerate_view_t*>(this)->end();
    }

    container_t _container;
  };

  struct enumerate_fn {
    template <typename container_t>
    auto operator()(container_t&& container) const {
      return enumerate_view_t<container_t>{std::forward<container_t>(container)};
    }
  };

  template <typename container_t>
  auto operator|(container_t&& container, const fan::enumerate_fn& view) {
    return view(std::forward<container_t>(container));
  }

  constexpr enumerate_fn enumerate{};

  constexpr std::uint64_t get_hash(const char* str) {
    std::uint64_t result = 0xcbf29ce484222325; // FNV offset basis
    std::uint32_t i = 0;
    if (str == nullptr) {
      return 0;
    }
    while (str[i] != 0) {
      result ^= (std::uint64_t)str[i];
      result *= 1099511628211; // FNV prime
      i++;
    }
    return result;
  }

  template <
    typename nr_t,
    typename user_t,
    typename... params
  >
  using raii_nr_t = raii_build::raii_nr_t<nr_t, user_t, params...>;

  template<typename bll_t, typename user_fn_t, typename... param_t>
  auto add_bll_raii_impl(bll_t& bll, user_fn_t&& user_fn, fan::type_pack<param_t...>){
    using storage_t = std::remove_reference_t<bll_t>;
    using node_ref_t = typename storage_t::nr_t;
    using handle_t = raii_nr_t<node_ref_t, storage_t, param_t...>;
    using fn_t = typename handle_t::fn_t;
    using add_fn_t = typename handle_t::add_fn;
    using remove_fn_t = typename handle_t::remove_fn;
    
    struct callbacks_t {
      static node_ref_t add_impl(storage_t* s, fn_t cb){
        auto nr = s->NewNodeLast();
        (*s)[nr] = [cb](param_t... d){
          cb(nullptr, d...);
        };
        return nr;
      }
      static void remove_impl(storage_t* s, const node_ref_t& nr){
        if (s->NodeList.Current) {
          s->unlrec(nr);
        }
      }
      storage_t* s;
      fn_t cb;
    };

    return handle_t(
      &bll,
      add_fn_t(callbacks_t::add_impl),
      remove_fn_t(callbacks_t::remove_impl),
      [user_fn = std::forward<user_fn_t>(user_fn)](storage_t*, param_t... d){
        user_fn(d...);
      }
    );
  }

  template<typename bll_t, typename user_fn_t>
  auto add_bll_raii_cb(bll_t& bll, user_fn_t&& user_fn){
    using traits = fan::lambda_traits<std::remove_reference_t<user_fn_t>>;
    using args_pack = typename traits::args;
    return add_bll_raii_impl(bll, std::forward<user_fn_t>(user_fn), args_pack{});
  }

  template<typename owner_t, typename storage_t, typename node_ref_t, typename user_fn_t, typename... param_t>
  auto add_bll_raii_struct_impl(owner_t* owner, storage_t owner_t::*storage_member, user_fn_t&& user_fn, fan::type_pack<param_t...>){
    using handle_t = raii_nr_t<node_ref_t, owner_t, param_t...>;
    using fn_t = typename handle_t::fn_t;
    
    auto add = [storage_member](owner_t* o, fn_t cb){
      auto& s = o->*storage_member;
      auto nr = s.NewNodeLast();
      s[nr] = [cb](param_t... d){
        cb(nullptr, d...);
      };
      return nr;
    };

    auto remove = [storage_member](owner_t* o, const node_ref_t& nr){
      auto& s = o->*storage_member;
      if (s.NodeList.Current) {
        s.unlrec(nr);
      }
    };

    return handle_t(
      owner,
      std::move(add),
      std::move(remove),
      [user_fn = std::forward<user_fn_t>(user_fn)](owner_t*, param_t... d){
        user_fn(d...);
      }
    );
  }

  template<typename owner_t, typename storage_t, typename user_fn_t>
  auto add_bll_raii_struct_cb(owner_t* owner, storage_t owner_t::*storage_member, user_fn_t&& user_fn){
    using traits = fan::lambda_traits<std::remove_reference_t<user_fn_t>>;
    using args_pack = typename traits::args;
    return add_bll_raii_struct_impl<owner_t, storage_t, typename storage_t::nr_t>(
      owner,
      storage_member,
      std::forward<user_fn_t>(user_fn),
      args_pack{}
    );
  }

  template<FAN_UNIQUE_CALL, typename T = f32_t>
  auto& static_var() {
    static T var{};
    return var;
  }

  template <std::ranges::contiguous_range R>
  auto subspan(R&& v, size_t start, size_t length) {
    return std::span(v).subspan(start, length);
  }

  struct restore_flag_t {
    bool& flag;
    bool prev;
    restore_flag_t(bool& f, bool val) : flag(f), prev(f) { f = val; }
    ~restore_flag_t() { flag = prev; }
  };
}