module;

#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <charconv>
#include <bit>
#include <array>
#include <cctype>
#include <span>
#include <algorithm>

export module fan.crypto;

import fan.utility;
import fan.types;
import fan.types.fstring;
import fan.print.error;

export namespace fan {
  constexpr std::size_t hamming_distance(std::string_view a, std::string_view b) {
    if (a.size() != b.size()) {
      throw std::invalid_argument("Hamming distance requires strings of equal length.");
    }
    std::size_t n = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      n += std::popcount(uint8_t(a[i] ^ b[i]));
    }
    return n;
  }
  static_assert(fan::hamming_distance("this is a test", "wokka wokka!!!") == 37);
  static_assert(fan::hamming_distance("aaaa", "aaaa") == 0);
  static_assert(fan::hamming_distance("aaaa", "bbbb") == 8);
  static_assert(fan::hamming_distance("", "") == 0);

  inline constexpr char enc_table[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

  inline constexpr auto dec_table = [] {
    std::array<uint8_t, 256> t {};
    for (uint8_t i = 0; i < 26; i++) t['A' + i] = i;
    for (uint8_t i = 0; i < 26; i++) t['a' + i] = 26 + i;
    for (uint8_t i = 0; i < 10; i++) t['0' + i] = 52 + i;
    t['+'] = 62; t['/'] = 63;
    return t;
  }();

  std::string base64_encode(const bytes_t& data) {
    std::string result;
    result.reserve(((data.size() + 2) / 3) * 4);

    for (size_t i = 0; i < data.size(); i += 3) {
      uint32_t val = static_cast<uint32_t>(data[i]) << 16;
      if (i + 1 < data.size()) val |= static_cast<uint32_t>(data[i + 1]) << 8;
      if (i + 2 < data.size()) val |= static_cast<uint32_t>(data[i + 2]);

      result.push_back(enc_table[(val >> 18) & 0x3F]);
      result.push_back(enc_table[(val >> 12) & 0x3F]);
      result.push_back((i + 1 < data.size()) ? enc_table[(val >> 6) & 0x3F] : '=');
      result.push_back((i + 2 < data.size()) ? enc_table[val & 0x3F] : '=');
    }

    return result;
  }

  bytes_t base64_decode(std::string data) {
    fan::strip_newlines(data);
    bytes_t result;
    result.reserve((data.size() / 4) * 3);

    for (size_t i = 0; i < data.size(); i += 4) {
      uint32_t val = static_cast<uint32_t>(dec_table[(uint8_t)data[i]]) << 18
        | static_cast<uint32_t>(dec_table[(uint8_t)data[i + 1]]) << 12
        | static_cast<uint32_t>(dec_table[(uint8_t)data[i + 2]]) << 6
        | static_cast<uint32_t>(dec_table[(uint8_t)data[i + 3]]);

      result.push_back(static_cast<uint8_t>((val >> 16) & 0xFF));
      if (data[i + 2] != '=') result.push_back(static_cast<uint8_t>((val >> 8) & 0xFF));
      if (data[i + 3] != '=') result.push_back(static_cast<uint8_t>(val & 0xFF));
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
  std::string bytes2hex(const bytes_t& b, bool pad = false) {
    std::string r(pad ? b.size() * 2 : 0, '0');
    for (int i = 0; i < (int)b.size(); ++i) {
      char buf[2];
      auto [ptr, ec] = std::to_chars(buf, buf + 2, b[i], 16);
      if (pad) {
        r[i * 2 + (ptr == buf + 1)] = buf[0];
        r[i * 2 + 1] = *(ptr - 1);
      }
      else {
        r.append(buf, ptr);
      }
    }
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

  // https://en.wikipedia.org/wiki/Letter_frequency
  #if 1
    constexpr f32_t freq_t[] = { 
      11.7f, 4.4f, 5.2f, 3.2f, 2.8f, 4.0f, 1.6f, 4.2f, 7.3f, 
      0.51f, 0.86f, 2.4f, 3.8f, 2.3f, 7.6f, 4.3f, 0.22f, 2.8f, 
      6.7f, 16.0f, 1.2f, 0.82f, 5.5f, 0.045f, 0.76f, 0.045f
    };
    constexpr f32_t freq_d[] = { 
      5.7f, 6.0f, 9.4f, 6.1f, 3.9f, 4.1f, 3.3f, 3.7f, 3.9f, 
      1.1f, 1.0f, 3.1f, 5.6f, 2.2f, 2.5f, 7.7f, 0.49f, 6.0f, 
      11.0f, 5.0f, 2.9f, 1.5f, 2.7f, 0.05f, 0.36f, 0.24f 
    };
  #else
    constexpr f32_t freq_t[] = {
      8.2f, 1.5f, 2.8f, 4.3f, 12.7f, 2.2f, 2.0f, 6.1f, 7.0f,
      0.16f, 0.77f, 4.0f, 2.4f, 6.7f, 7.5f, 1.9f, 0.12f, 6.0f,
      6.3f, 9.1f, 2.8f, 0.98f, 2.4f, 0.15f, 2.0f, 0.074f
    };
    constexpr f32_t freq_d[] = {
      7.8f, 2.0f, 4.0f, 3.8f, 11.0f, 1.4f, 3.0f, 2.3f, 8.6f,
      0.25f, 0.97f, 5.3f, 2.7f, 7.2f, 6.1f, 2.8f, 0.19f, 7.3f,
      8.7f, 6.7f, 3.3f, 1.0f, 0.91f, 0.27f, 1.6f, 0.44f
    };
  #endif

  struct single_byte_xor_t {
    void update(int ascii_char, f32_t s, const std::string& t) {
      if (s > score) { ascii_character = ascii_char, score = s; decrypted_text = t; }
    }
    int ascii_character = -1;
    f32_t score = 0.f;
    std::string decrypted_text;
  };

  void crack_single_byte_xor(
    const bytes_t& a_bytes,
    single_byte_xor_t& tbest,
    single_byte_xor_t& dbest)
  {
    constexpr int n = 256;
    f32_t st[n]{}, sd[n]{};
    std::string texts[n]{};

    for (int i = 0; i < n; ++i) {
      auto keyd = xor_key(a_bytes, uint8_t(i));
      texts[i] = as_chars(keyd);
      for (auto c : keyd) {
        if (std::isblank(c)) { st[i] += 10.f; sd[i] += 10.f; continue; }
        if (!std::isprint(c)) { st[i] -= 10.f; sd[i] -= 10.f; continue; }
        if (!std::isalpha(c)) continue;
        uint8_t p = (std::toupper(c) - 'A') % std::size(freq_d);
        st[i] += freq_t[p];
        sd[i] += freq_d[p];
      }
    }

    auto pick = [&](f32_t* s, single_byte_xor_t& best) {
      int d = std::max_element(s, s + n) - s;
      best.update(d, s[d], texts[d]);
    };
    pick(st, tbest);
    pick(sd, dbest);
  }
  single_byte_xor_t crack_single_byte_xor(const bytes_t& a_bytes) {
    single_byte_xor_t text, dict;
    crack_single_byte_xor(a_bytes, text, dict);
    return text.score > dict.score ? text : dict;
  }

  namespace repeating_key_xor {
    struct key_size_candidate_t {
      std::vector<std::string> spans;
      f32_t result;
      bool operator<(const key_size_candidate_t& d) const {
        return result < d.result;
      }
      void add(std::span<const uint8_t> key_x) {
        spans.emplace_back(fan::as_chars(key_x));
        if (spans.size() % 2 == 0) {
          calc_distance();
        }
      }
      void calc_distance() {
        if (!(spans.size() % 2 == 0) || spans.empty()) {
          fan::throw_error_impl();
        }
        f32_t distance = 0.f;
        for (int i = 0; i < spans.size() - 1; i += 2) {
          distance += fan::hamming_distance(spans[i], spans[i + 1]);
        }
        result = distance / (spans.front().size() * (spans.size() / 2));
      }
    };

    struct score_t {
      std::string text;
      f32_t value;
      bool operator<(const score_t& s) const {
        return value < s.value;
      }
    };

    void crack(fan::bytes_t& bytes, int key_size, std::vector<score_t>& scores) {
      std::string key;
      f32_t overall_score = 0.f;
      for (int x = 0; x < key_size; ++x) {
        fan::bytes_t block(bytes.size() / key_size);
        for (int y = 0; y < block.size(); y++) {
          block[y] = bytes[x + y * key_size];
        }
        auto result = fan::crack_single_byte_xor(block);
        key += result.ascii_character;
        overall_score += result.score;
      }
      scores.push_back({
        .text = fan::as_chars(fan::xor_key(bytes, fan::as_bytes(key))),
        .value = overall_score
      });
    }

    std::vector<key_size_candidate_t> get_key_size_candidates(const fan::bytes_t& bytes) {
      constexpr int key_size_min = 2;
      constexpr int key_size_max = 40;
      std::vector<key_size_candidate_t> distances(key_size_max - key_size_min + 1);
      for (int key_size = key_size_min; key_size <= key_size_max; ++key_size) {
        for (int i = 0; i < 4; ++i) {
          distances[key_size - key_size_min].add(
            fan::subspan(bytes, key_size * i, key_size)
          );
        }
      }
      std::sort(distances.begin(), distances.end());
      return distances;
    }
  }
}
