module;

#include <vector>
#include <limits>
#include <cstdint>
#include <utility>
#include <optional>

export module fan.types.slot_map;

export namespace fan {
  template <typename T>
  struct slot_map_t {
    struct slot_t {
      uint32_t generation = 0;
      uint32_t next_free = 0;
      bool occupied = false;
      std::optional<T> data;
    };

    struct id_t {
      uint32_t index = std::numeric_limits<uint32_t>::max();
      uint32_t generation = 0;
      
      constexpr bool is_valid() const { return index != std::numeric_limits<uint32_t>::max(); }
      constexpr bool operator==(const id_t&) const = default;
    };

    std::vector<slot_t> slots;
    uint32_t free_head = std::numeric_limits<uint32_t>::max();
    uint32_t active_count = 0;

    template <typename... args_t>
    id_t emplace(args_t&&... args) {
      active_count++;
      if (free_head == std::numeric_limits<uint32_t>::max()) {
        uint32_t idx = (uint32_t)slots.size();
        slots.push_back({0, std::numeric_limits<uint32_t>::max(), true, std::optional<T>(std::in_place, std::forward<args_t>(args)...)});
        return {idx, 0};
      }
      uint32_t idx = free_head;
      free_head = slots[idx].next_free;
      slots[idx].occupied = true;
      slots[idx].data.emplace(std::forward<args_t>(args)...);
      return {idx, slots[idx].generation};
    }

    void erase(id_t id) {
      if (!is_valid(id)) { return; }
      slots[id.index].generation++;
      slots[id.index].occupied = false;
      slots[id.index].next_free = free_head;
      free_head = id.index;
      slots[id.index].data.reset();
      active_count--;
    }

    T* find(id_t id) {
      if (!is_valid(id)) { return nullptr; }
      return &*slots[id.index].data;
    }

    const T* find(id_t id) const {
      if (!is_valid(id)) { return nullptr; }
      return &*slots[id.index].data;
    }

    bool is_valid(id_t id) const {
      if (id.index >= slots.size()) { return false; }
      return slots[id.index].occupied && slots[id.index].generation == id.generation;
    }

    void clear() {
      slots.clear();
      free_head = std::numeric_limits<uint32_t>::max();
      active_count = 0;
    }

    struct iterator {
      slot_map_t* map;
      uint32_t idx;
      
      void advance() {
        while (idx < map->slots.size() && !map->slots[idx].occupied) {
          idx++;
        }
      }
      iterator& operator++() { idx++; advance(); return *this; }
      bool operator!=(const iterator& other) const { return idx != other.idx; }
      T& operator*() { return *map->slots[idx].data; }
    };
    
    iterator begin() { iterator it{this, 0}; it.advance(); return it; }
    iterator end() { return {this, (uint32_t)slots.size()}; }
  };
}