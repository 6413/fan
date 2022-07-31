#pragma once

#include <unordered_set>
#include <set>

template <typename T>
struct ks_t {

  using value_type_t = T;
  using set_t = std::multiset<T, T>;

  void open() {
    set = new set_t;
  }
  void close() {
    delete set;
  }

  void insert(const T& value) {
    set->insert(value);
  }
  void insert(T&& value) {
    set->insert(std::move(value));
  }

  auto begin() const {
    return set->begin();
  }
  auto end() const {
    return set->end();
  }

  set_t * set;
};