#pragma once

#include <utility>
#include <cstdint>

namespace fan {
  template <typename T, typename T2>
  struct pair_t : std::pair<T, T2> {
    using std::pair<T, T2>::pair;


    template <typename dummy_t = T, typename dummy2_t = T2>
      requires(std::is_same_v<dummy_t, dummy2_t>)
    T& operator[](std::uint8_t i) {
      return i == 0 ? std::pair<T, T2>::first : std::pair<T, T2>::second;
    }
    template <typename dummy_t = T, typename dummy2_t = T2>
      requires(std::is_same_v<dummy_t, dummy2_t>)
    T operator[](std::uint8_t i) const {
      return i == 0 ? std::pair<T, T2>::first : std::pair<T, T2>::second;
    }
  };

}
