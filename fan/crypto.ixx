module;

#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <charconv>
#include <bit>

export module fan.crypto;

import fan.types;

export namespace fan {
  constexpr std::size_t hamming_distance(std::string_view a, std::string_view b) {
    std::size_t n = 0;
    if (a.size() != b.size()) return n;
    for (std::size_t i = 0; i < a.size(); ++i) {
      n += std::popcount(uint8_t(a[i] ^ b[i]));
    }
    return n;
  }

  std::string base64_encode(const bytes_t& data) {
    static constexpr char chars[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string result;
    result.reserve(((data.size() + 2) / 3) * 4);

    for (size_t i = 0; i < data.size(); i += 3) {
      uint32_t val = static_cast<uint32_t>(data[i]) << 16;
      if (i + 1 < data.size()) val |= static_cast<uint32_t>(data[i + 1]) << 8;
      if (i + 2 < data.size()) val |= static_cast<uint32_t>(data[i + 2]);

      result.push_back(chars[(val >> 18) & 0x3F]);
      result.push_back(chars[(val >> 12) & 0x3F]);
      result.push_back((i + 1 < data.size()) ? chars[(val >> 6) & 0x3F] : '=');
      result.push_back((i + 2 < data.size()) ? chars[val & 0x3F] : '=');
    }

    return result;
  }

  template <typename range_t>
  std::string xor2hex(const range_t& a, const range_t& b) {
    if (b.empty()) return "";
    std::ostringstream r;
    r << std::hex << std::setfill('0');
    for (std::size_t i = 0; i < a.size(); ++i) {
      r << std::setw(2) << +uint8_t(a[i] ^ b[i % b.size()]);
    }
    return r.str();
  }

  bytes_t hex2bytes(std::string_view s) {
    bytes_t r(s.size() / 2);
    for (int i = 0; i < (int)r.size(); ++i)
      std::from_chars(s.data() + i * 2, s.data() + i * 2 + 2, r[i], 16);
    return r;
  }

  bytes_t xor_key(const bytes_t& a, const bytes_t& b) {
    bytes_t r(a.size());
    for (std::size_t i = 0; i < a.size(); ++i)
      r[i] = a[i] ^ b[i % b.size()];
    return r;
  }

  bytes_t xor_key(const bytes_t& a, uint8_t key) {
    return xor_key(a, bytes_t{key});
  }
}
