module;
#include <cstdint>
#include <vector>
#include <functional>
#include <tuple>
#include <type_traits>
export module fan.ecs;

import fan.types;
import fan.types.vector;

namespace fan::detail {
  template <typename T> struct fn_traits;
  template <typename R, typename C, typename... A>
  struct fn_traits<R(C::*)(A...) const> { using args = std::tuple<std::decay_t<A>...>; };
  template <typename R, typename C, typename... A>
  struct fn_traits<R(C::*)(A...)> { using args = std::tuple<std::decay_t<A>...>; };
}

export namespace fan::ecs {
  struct c_pos { fan::vec2 v; };
  struct c_vel { fan::vec2 v; };
  struct c_hp { int current; int max; };
  struct c_life { f32_t timer; };
  struct tag_bullet{};
  struct c_cost { int32_t value = 0; };
}

export namespace fan {
  template <typename... comps_t>
  struct ecs_t : std::vector<comps_t>... {
    static_assert(sizeof...(comps_t) <= 64, "Max 64 components per ECS");

    template <typename T>
    static consteval uint64_t bit() {
      constexpr std::array<bool, sizeof...(comps_t)> matches = { std::is_same_v<T, comps_t>... };
      for (size_t i = 0; i < matches.size(); ++i) {
        if (matches[i]) { return 1ULL << i; }
      }
      return 0;
    }

    template <typename... Ts>
    static consteval uint64_t mask_of() {
      return (bit<Ts>() | ...);
    }

    template <typename T>
    std::vector<T>& pool() { return *this; }

    std::vector<uint64_t> mask;
    std::vector<uint32_t> gen;
    std::vector<uint32_t> free_list;

    uint32_t create() {
      if (!free_list.empty()) {
        uint32_t id = free_list.back();
        free_list.pop_back();
        this->mask[id] = 0;
        return id;
      }
      uint32_t id = (uint32_t)this->mask.size();
      this->mask.push_back(0);
      gen.push_back(0);
      (std::vector<comps_t>::emplace_back(), ...);
      return id;
    }

    void destroy(uint32_t id) {
      if (this->mask[id] == 0) { return; }
      for (auto& hook : on_destroy_hooks) { hook(id); }
      ((pool<comps_t>()[id] = comps_t{}), ...);
      this->mask[id] = 0;
      gen[id]++;
      free_list.push_back(id);
    }

    template <typename T, typename... args_t>
    void add(uint32_t id, args_t&&... args) {
      this->mask[id] |= bit<T>();
      pool<T>()[id] = T{std::forward<args_t>(args)...};
    }

    template <typename T>
    void remove(uint32_t id) {
      this->mask[id] &= ~bit<T>();
    }

    template <typename... Ts>
    bool has(uint32_t id) {
      uint64_t m = mask_of<Ts...>();
      return (this->mask[id] & m) == m;
    }

    template <typename T>
    T& get(uint32_t id) {
      return pool<T>()[id];
    }

    template <typename... Ts, typename func_t>
    requires (sizeof...(Ts) > 0)
    void each(func_t&& func) {
      uint64_t m = mask_of<Ts...>();
      for (uint32_t i = 0; i < this->mask.size(); ++i) {
        if ((this->mask[i] & m) == m) {
          if constexpr (std::is_invocable_v<func_t, uint32_t, Ts&...>) {
            func(i, pool<Ts>()[i]...);
          } else {
            func(pool<Ts>()[i]...);
          }
        }
      }
    }

    template <typename func_t>
    void each(func_t&& func) {
      using args = typename fan::detail::fn_traits<decltype(&std::decay_t<func_t>::operator())>::args;
      _each(std::forward<func_t>(func), args{});
    }

    template <typename func_t, typename... Ts>
    void _each(func_t&& func, std::tuple<uint32_t, Ts...>) {
      each<Ts...>(std::forward<func_t>(func));
    }

    template <typename func_t, typename... Ts>
    void _each(func_t&& func, std::tuple<Ts...>) {
      each<Ts...>(std::forward<func_t>(func));
    }

    template <typename... Ts>
    uint32_t create_with(Ts&&... comps) {
      uint32_t id = this->create();
      (this->template add<std::decay_t<Ts>>(id, std::forward<Ts>(comps)), ...);
      return id;
    }

    template <typename... Ts, typename func_t>
    requires (sizeof...(Ts) > 0)
    bool each_breakable(func_t&& func) {
      uint64_t m = mask_of<Ts...>();
      for (uint32_t i = 0; i < this->mask.size(); ++i) {
        if ((this->mask[i] & m) == m) {
          if constexpr (std::is_invocable_v<func_t, uint32_t, Ts&...>) {
            if (!func(i, pool<Ts>()[i]...)) return false;
          } else {
            if (!func(pool<Ts>()[i]...)) return false;
          }
        }
      }
      return true;
    }

    template <typename func_t>
    bool each_breakable(func_t&& func) {
      using args = typename fan::detail::fn_traits<decltype(&std::decay_t<func_t>::operator())>::args;
      return _each_breakable(std::forward<func_t>(func), args{});
    }

    template <typename func_t, typename... Ts>
    bool _each_breakable(func_t&& func, std::tuple<uint32_t, Ts...>) {
      return each_breakable<Ts...>(std::forward<func_t>(func));
    }

    template <typename func_t, typename... Ts>
    bool _each_breakable(func_t&& func, std::tuple<Ts...>) {
      return each_breakable<Ts...>(std::forward<func_t>(func));
    }

    void clear() {
      this->mask.clear();
      gen.clear();
      free_list.clear();
      (pool<comps_t>().clear(), ...);
    }

    template <typename... Ts, typename func_t>
    requires (sizeof...(Ts) > 0)
    constexpr void destroy_if(func_t&& pred) {
      std::vector<uint32_t> dead;
      each<Ts...>([&](uint32_t i, Ts&... args) {
        if constexpr (std::is_invocable_v<func_t, uint32_t, Ts&...>) {
          if (pred(i, args...)) dead.push_back(i);
        } else {
          if (pred(args...)) dead.push_back(i);
        }
      });
      for (uint32_t id : dead) destroy(id);
    }

    template <typename func_t>
    constexpr void destroy_if(func_t&& pred) {
      using args = typename fan::detail::fn_traits<decltype(&std::decay_t<func_t>::operator())>::args;
      _destroy_if(std::forward<func_t>(pred), args{});
    }

    template <typename func_t, typename... Ts>
    constexpr void _destroy_if(func_t&& pred, std::tuple<uint32_t, Ts...>) {
      destroy_if<Ts...>(std::forward<func_t>(pred));
    }

    template <typename func_t, typename... Ts>
    constexpr void _destroy_if(func_t&& pred, std::tuple<Ts...>) {
      destroy_if<Ts...>(std::forward<func_t>(pred));
    }

    template <typename... Tags, typename F>
    constexpr bool destroy_dead(F&& cb) {
      bool destroyed = false;
      each<fan::ecs::c_hp, Tags...>([&](uint32_t id, fan::ecs::c_hp& hp, Tags&... args) {
        if (hp.current <= 0) {
          cb(args...);
          destroy(id);
          destroyed = true;
        }
      });
      return destroyed;
    }

    template <typename... Tags, typename F>
    void destroy_at(vec2 pos, F&& cb) {
      bool hit = false;
      (destroy_if<fan::ecs::c_pos, Tags>([&](fan::ecs::c_pos& p, Tags&) {
        return (hit |= p.v == pos), p.v == pos;
      }), ...);
      if (hit) cb();
    }

    template <typename... Tags>
    bool any() {
      bool found = false;
      each_breakable<Tags...>([&](Tags&...) { return !(found = true); });
      return found;
    }

    std::vector<std::function<void(uint32_t)>> on_destroy_hooks;
  };
}

export namespace fan::ecs::systems {
  template <typename vel_t, typename tag_t, typename registry_t>
  constexpr void apply_drag(registry_t& reg, f32_t drag_multiplier) {
    reg.template each<vel_t, tag_t>([&](vel_t& vel, tag_t&) {
      vel.v *= drag_multiplier;
    });
  }

  template <typename pos_t, typename vel_t, typename registry_t>
  constexpr void kinematics(registry_t& reg, f32_t dt) {
    reg.template each<pos_t, vel_t>([dt](pos_t& pos, vel_t& vel) {
      pos.v += vel.v * dt;
    });
  }

  template <typename life_t, typename registry_t>
  constexpr void lifetimes(registry_t& reg, f32_t dt) {
    std::vector<uint32_t> dead;
    reg.template each<life_t>([&](uint32_t e, life_t& life) {
      life.timer -= dt;
      if (life.timer <= 0.f) { dead.push_back(e); }
    });
    for (uint32_t id : dead) { reg.destroy(id); }
  }
}