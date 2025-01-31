#pragma once

// implements some missing c++ standard functions and or classes from some compilers
template <typename It>
class enumerate_iterator {
  It _iter;
  std::size_t _index;
public:
  using value_type = std::tuple<std::size_t, typename std::iterator_traits<It>::reference>;
  using reference = value_type;
  using pointer = void;

  enumerate_iterator(It iter, std::size_t index) : _iter(iter), _index(index) {}

  reference operator*() const {
    return { _index, *_iter };
  }

  enumerate_iterator& operator++() {
    ++_iter;
    ++_index;
    return *this;
  }

  bool operator!=(const enumerate_iterator& other) const {
    return _iter != other._iter;
  }
};
template <typename Container>
class enumerate_view {
  Container& _container;
public:
  using iterator = enumerate_iterator<typename std::conditional<
    std::is_const_v<Container>,
    typename Container::const_iterator,
    typename Container::iterator
  >::type>;

  enumerate_view(Container& container) : _container(container) {}

  iterator begin() {
    return { std::begin(_container), 0 };
  }

  iterator end() {
    return { std::end(_container), std::size(_container) };
  }
};

namespace fan {
  struct enumerate_fn {
    template <typename Container>
    auto operator()(Container& container) const {
      return enumerate_view<Container>{container};
    }
  };

  inline constexpr enumerate_fn enumerate;
}

template <typename Container>
auto operator|(Container& container, const fan::enumerate_fn& view) {
  return view(container);
}