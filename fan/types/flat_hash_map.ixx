module;

export module fan.types.flat_hash_map;

import std;
import fan.types.vector;

export namespace fan {
  template <typename K, typename V>
  struct flat_map_t {
    enum class state_t : std::uint8_t { empty, occupied, deleted };

    struct entry_t {
      K key;
      V value;
      state_t state = state_t::empty;
    };

    struct iterator {
      flat_map_t* map;
      std::uint32_t index;
      V& operator*() { return map->table[index].value; }
      V* operator->() { return &map->table[index].value; }
    };

    flat_map_t(std::uint32_t initial_size = 1024) {
      table.resize(initial_size);
    }

    std::uint32_t find_slot(const K& key) const {
      std::uint32_t idx = fan::get_hash_fast(key) & (table.size() - 1);
      while (table[idx].state != state_t::empty) {
        if (table[idx].state == state_t::occupied && table[idx].key == key) {
          return idx;
        }
        idx = (idx + 1) & (table.size() - 1);
      }
      return ~0u;
    }

    std::uint32_t find_insert_slot(const K& key) {
      std::uint32_t idx = fan::get_hash_fast(key) & (table.size() - 1);
      std::uint32_t first_deleted = ~0u;
      while (table[idx].state != state_t::empty) {
        if (table[idx].state == state_t::occupied && table[idx].key == key) {
          return idx;
        }
        if (table[idx].state == state_t::deleted && first_deleted == ~0u) {
          first_deleted = idx;
        }
        idx = (idx + 1) & (table.size() - 1);
      }
      return first_deleted != ~0u ? first_deleted : idx;
    }

    void rehash() {
      auto old = std::move(table);
      table.assign(old.size() * 2, entry_t{});
      count = 0;
      deleted_count = 0;
      for (auto& e : old) {
        if (e.state == state_t::occupied) {
          std::uint32_t idx = find_insert_slot(e.key);
          table[idx].key = std::move(e.key);
          table[idx].value = std::move(e.value);
          table[idx].state = state_t::occupied;
          count++;
        }
      }
    }

    void maybe_rehash() {
      if ((count + deleted_count) * 2 >= table.size()) {
        rehash();
      }
    }

    V& operator[](const K& key) {
      maybe_rehash();
      std::uint32_t idx = find_insert_slot(key);
      if (table[idx].state != state_t::occupied) {
        table[idx].key = key;
        table[idx].state = state_t::occupied;
        count++;
      }
      return table[idx].value;
    }

    template <typename... args_t>
    std::pair<iterator, bool> try_emplace(const K& key, args_t&&... args) {
      maybe_rehash();
      std::uint32_t idx = find_insert_slot(key);
      if (table[idx].state == state_t::occupied) {
        return {iterator{this, idx}, false};
      }
      if (table[idx].state == state_t::deleted) {
        deleted_count--;
      }
      table[idx].key = key;
      table[idx].value = V(std::forward<args_t>(args)...);
      table[idx].state = state_t::occupied;
      count++;
      return {iterator{this, idx}, true};
    }

    bool contains(const K& key) const {
      return find_slot(key) != ~0u;
    }

    void erase(const K& key) {
      std::uint32_t idx = find_slot(key);
      if (idx == ~0u) { return; }
      table[idx].state = state_t::deleted;
      count--;
      deleted_count++;
      if (deleted_count * 2 >= table.size()) {
        rehash();
      }
    }

    void clear() {
      table.assign(table.size(), entry_t{});
      count = 0;
      deleted_count = 0;
    }

    std::vector<entry_t> table;
    std::uint32_t count = 0;
    std::uint32_t deleted_count = 0;
  };
}