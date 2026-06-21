module;

export module fan.types.bitset;

import std;

export namespace fan {
  /*bitset will round up to the nearest full byte*/
  template <std::size_t n_bits>
  struct bitset_t {
    static_assert(n_bits > 0, "bitset_t requires at least 1 bit");
    static constexpr auto nbits = n_bits;

    struct reference_t {
      constexpr reference_t& operator=(bool v) noexcept {
        if (v) {
          byte |= mask;
        } else {
          byte &= ~mask;
        }
        return *this;
      }

      constexpr reference_t& operator=(const reference_t& o) noexcept {
        return *this = bool(o);
      }

      constexpr operator bool() const noexcept {
        return (byte & mask) != 0;
      }

      constexpr bool operator~() const noexcept {
        return (byte & mask) == 0;
      }

      constexpr reference_t& flip() noexcept {
        byte ^= mask;
        return *this;
      }

      std::uint8_t& byte;
      std::uint8_t mask;
    };

    constexpr bitset_t() noexcept = default;

    constexpr bitset_t(unsigned long long val) noexcept {
      const std::size_t max_b = std::min(num_bytes, sizeof(unsigned long long));
      for (std::size_t i = 0; i < max_b; ++i) {
        data[i] = static_cast<std::uint8_t>((val >> (i * 8)) & 0xFF);
      }
      sanitize();
    }

    template <typename char_t = char, typename traits_t = std::char_traits<char_t>>
    explicit constexpr bitset_t(
      std::basic_string_view<char_t, traits_t> str,
      std::size_t pos = 0,
      std::size_t n = std::basic_string_view<char_t, traits_t>::npos,
      char_t zero = char_t('0'),
      char_t one = char_t('1')
    ) {
      if (pos > str.size()) {
        throw std::out_of_range("bitset_t");
      }
      std::size_t rlen = std::min(n, str.size() - pos);
      std::size_t m = std::min(n_bits, rlen);
      for (std::size_t i = 0; i < m; ++i) {
        char_t c = str[pos + rlen - 1 - i];
        if (c == one) {
          set(i);
        } else if (c != zero) {
          throw std::invalid_argument("bitset_t");
        }
      }
    }

    constexpr bool operator==(const bitset_t& o) const noexcept {
      for (std::size_t i = 0; i < num_bytes; ++i) {
        if (data[i] != o.data[i]) {
          return false;
        }
      }
      return true;
    }

    constexpr bool operator[](std::size_t i) const noexcept {
      return (data[i / 8] & (1u << (i % 8))) != 0;
    }

    constexpr reference_t operator[](std::size_t i) noexcept {
      return {data[i / 8], static_cast<std::uint8_t>(1u << (i % 8))};
    }

    constexpr bool test(std::size_t i) const {
      if (i >= n_bits) {
        throw std::out_of_range("bitset_t");
      }
      return (*this)[i];
    }

    constexpr bool all() const noexcept {
      for (std::size_t i = 0; i < num_bytes - 1; ++i) {
        if (data[i] != 0xFF) {
          return false;
        }
      }
      return (data[num_bytes - 1] & clean_mask) == clean_mask;
    }

    constexpr bool any() const noexcept {
      for (std::size_t i = 0; i < num_bytes; ++i) {
        if (data[i] != 0) {
          return true;
        }
      }
      return false;
    }

    constexpr bool none() const noexcept {
      return !any();
    }

    constexpr std::size_t count() const noexcept {
      std::size_t c = 0;
      for (std::size_t i = 0; i < num_bytes; ++i) {
        c += std::popcount(data[i]);
      }
      return c;
    }

    constexpr std::size_t size() const noexcept {
      return n_bits;
    }

    constexpr bitset_t& operator&=(const bitset_t& o) noexcept {
      for (std::size_t i = 0; i < num_bytes; ++i) {
        data[i] &= o.data[i];
      }
      return *this;
    }

    constexpr bitset_t& operator|=(const bitset_t& o) noexcept {
      for (std::size_t i = 0; i < num_bytes; ++i) {
        data[i] |= o.data[i];
      }
      return *this;
    }

    constexpr bitset_t& operator^=(const bitset_t& o) noexcept {
      for (std::size_t i = 0; i < num_bytes; ++i) {
        data[i] ^= o.data[i];
      }
      return *this;
    }

    constexpr bitset_t operator~() const noexcept {
      bitset_t res = *this;
      return res.flip();
    }

    constexpr bitset_t& operator<<=(std::size_t shift) noexcept {
      if (shift >= n_bits) {
        return reset();
      }
      if (shift == 0) {
        return *this;
      }
      const std::size_t b_shift = shift / 8;
      const std::size_t bit_s = shift % 8;
      for (std::size_t i = num_bytes - 1; i < num_bytes; --i) {
        if (i < b_shift) {
          data[i] = 0;
        } else {
          std::uint8_t src = data[i - b_shift];
          std::uint8_t sub = (bit_s == 0 || i == b_shift) ? 0 : (data[i - b_shift - 1] >> (8 - bit_s));
          data[i] = static_cast<std::uint8_t>((src << bit_s) | sub);
        }
      }
      sanitize();
      return *this;
    }

    constexpr bitset_t& operator>>=(std::size_t shift) noexcept {
      if (shift >= n_bits) {
        return reset();
      }
      if (shift == 0) {
        return *this;
      }
      const std::size_t b_shift = shift / 8;
      const std::size_t bit_s = shift % 8;
      for (std::size_t i = 0; i < num_bytes; ++i) {
        if (i + b_shift >= num_bytes) {
          data[i] = 0;
        } else {
          std::uint8_t src = data[i + b_shift];
          std::uint8_t sup = (bit_s == 0 || i + b_shift + 1 >= num_bytes) ? 0 : (data[i + b_shift + 1] << (8 - bit_s));
          data[i] = static_cast<std::uint8_t>((src >> bit_s) | sup);
        }
      }
      return *this;
    }

    constexpr bitset_t operator<<(std::size_t shift) const noexcept {
      bitset_t res = *this;
      return res <<= shift;
    }

    constexpr bitset_t operator>>(std::size_t shift) const noexcept {
      bitset_t res = *this;
      return res >>= shift;
    }

    constexpr bitset_t& set() noexcept {
      for (std::size_t i = 0; i < num_bytes; ++i) {
        data[i] = 0xFF;
      }
      sanitize();
      return *this;
    }

    constexpr bitset_t& set(std::size_t i, bool val = true) {
      if (i >= n_bits) {
        throw std::out_of_range("bitset_t");
      }
      if (val) {
        data[i / 8] |= static_cast<std::uint8_t>(1u << (i % 8));
      } else {
        data[i / 8] &= static_cast<std::uint8_t>(~(1u << (i % 8)));
      }
      return *this;
    }

    constexpr bitset_t& reset() noexcept {
      for (std::size_t i = 0; i < num_bytes; ++i) {
        data[i] = 0;
      }
      return *this;
    }

    constexpr bitset_t& reset(std::size_t i) {
      return set(i, false);
    }

    constexpr bitset_t& flip() noexcept {
      for (std::size_t i = 0; i < num_bytes; ++i) {
        data[i] = static_cast<std::uint8_t>(~data[i]);
      }
      sanitize();
      return *this;
    }

    constexpr bitset_t& flip(std::size_t i) {
      if (i >= n_bits) {
        throw std::out_of_range("bitset_t");
      }
      data[i / 8] ^= static_cast<std::uint8_t>(1u << (i % 8));
      return *this;
    }

    template <typename char_t = char, typename traits_t = std::char_traits<char_t>, typename alloc_t = std::allocator<char_t>>
    constexpr std::basic_string<char_t, traits_t, alloc_t> to_string(char_t zero = char_t('0'), char_t one = char_t('1')) const {
      std::basic_string<char_t, traits_t, alloc_t> s(n_bits, zero);
      for (std::size_t i = 0; i < n_bits; ++i) {
        if (test(n_bits - 1 - i)) {
          s[i] = one;
        }
      }
      return s;
    }

    constexpr unsigned long to_ulong() const {
      const unsigned long long v = to_ullong();
      if (v > std::numeric_limits<unsigned long>::max()) {
        throw std::overflow_error("bitset_t");
      }
      return static_cast<unsigned long>(v);
    }

    constexpr unsigned long long to_ullong() const {
      if constexpr (num_bytes > sizeof(unsigned long long)) {
        for (std::size_t i = sizeof(unsigned long long); i < num_bytes; ++i) {
          if (data[i] != 0) {
            throw std::overflow_error("bitset_t");
          }
        }
      }
      unsigned long long res = 0;
      const std::size_t max_b = std::min(num_bytes, sizeof(unsigned long long));
      for (std::size_t i = 0; i < max_b; ++i) {
        res |= static_cast<unsigned long long>(data[i]) << (i * 8);
      }
      return res;
    }

    constexpr void sanitize() noexcept {
      if constexpr (n_bits % 8 != 0) {
        data[num_bytes - 1] &= clean_mask;
      }
    }

    static constexpr std::size_t num_bytes = (n_bits + 7) / 8;
    static constexpr std::uint8_t clean_mask = (n_bits % 8 == 0) ? 0xFF : static_cast<std::uint8_t>((1u << (n_bits % 8)) - 1);
    std::uint8_t data[num_bytes]{};
  };

  template <std::size_t n_bits>
  constexpr bitset_t<n_bits> operator&(const bitset_t<n_bits>& lhs, const bitset_t<n_bits>& rhs) noexcept {
    bitset_t<n_bits> res = lhs;
    return res &= rhs;
  }

  template <std::size_t n_bits>
  constexpr bitset_t<n_bits> operator|(const bitset_t<n_bits>& lhs, const bitset_t<n_bits>& rhs) noexcept {
    bitset_t<n_bits> res = lhs;
    return res |= rhs;
  }

  template <std::size_t n_bits>
  constexpr bitset_t<n_bits> operator^(const bitset_t<n_bits>& lhs, const bitset_t<n_bits>& rhs) noexcept {
    bitset_t<n_bits> res = lhs;
    return res ^= rhs;
  }

  template <typename char_t, typename traits_t, std::size_t n_bits>
  std::basic_istream<char_t, traits_t>& operator>>(std::basic_istream<char_t, traits_t>& is, bitset_t<n_bits>& x) {
    std::basic_string<char_t, traits_t> str;
    str.reserve(n_bits);
    char_t c;
    while (str.size() < n_bits && is.get(c)) {
      if (c == is.widen('0') || c == is.widen('1')) {
        str.push_back(c);
      } else {
        is.unget();
        break;
      }
    }
    if (str.empty()) {
      is.setstate(std::ios_base::failbit);
    } else {
      x = bitset_t<n_bits>(str);
    }
    return is;
  }

  template <typename char_t, typename traits_t, std::size_t n_bits>
  std::basic_ostream<char_t, traits_t>& operator<<(std::basic_ostream<char_t, traits_t>& os, const bitset_t<n_bits>& x) {
    return os << x.template to_string<char_t, traits_t>();
  }
}