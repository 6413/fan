module;

#include <vector>
#include <cstring>
#include <memory>
#include <string>
#include <iomanip> // std::quoted
#include <charconv>
#include <ranges>

#include <ios>
#include <sstream>

export module fan.types.fstring;

import fan.types;
import fan.print.error; // for throw_error with msg
import fan.types.vector;
import fan.types.compile_time_string;

export namespace fan {

  using bytes_t = std::vector<uint8_t>;

  template <typename T>
  auto to_string(const T a_value, const int n = 2) {
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
  }

  inline constexpr uint64_t get_hash(const std::string_view& str) {
    uint64_t result = 0xcbf29ce484222325; // FNV offset basis

    uint32_t i = 0;

    while (i < str.size()) {
      result ^= (uint64_t)str[i];
      result *= 1099511628211; // FNV prime
      i++;
    }

    return result;
  }

  // static constexpr uint64_t get_hash(const std::string& str) {
  //   uint64_t result = 0xcbf29ce484222325; // FNV offset basis

  //   uint32_t i = 0;

  //   while (str[i] != 0) {
  //     result ^= (uint64_t)str[i];
  //     result *= 1099511628211; // FNV prime
  //     i++;
  //   }

  //   return result;
  // }
}

export namespace fan {
  struct string : public std::string {

    using type_t = std::string;
    using type_t::basic_string;

    string() = default;

    string(const std::string& str) : type_t(str) {}
    template <typename T>
    string(const std::vector<T>& vector) {
      for (uint32_t i = 0; i < vector.size() * sizeof(T); ++i) {
        append((uint8_t*)&vector[i], (uint8_t*)&vector[i] + sizeof(T));
      }
    }

    using char_type = std::string::value_type;

    static constexpr uint8_t UTF8_SizeOfCharacter(uint8_t byte) {
      if (byte < 0x80) {
        return 1;
      }
      if (byte < 0xc0) {
        fan::throw_error("invalid byte");
        /* error */
        return 1;
      }
      if (byte < 0xe0) {
        return 2;
      }
      if (byte < 0xf0) {
        return 3;
      }
      if (byte <= 0xf7) {
        return 4;
      }
      fan::throw_error("invalid byte");
      /* error */
      return 1;
    }

    uint32_t get_utf8_character(uintptr_t offset, uint8_t size) const {
      uint32_t code = 0;
      for (int j = 0; j < size; j++) {
        code <<= 8;
        code |= (*this)[offset + j];
      }
      return code;
    }
    void replace_all(const std::string& from, const std::string& to) {
      if(from.empty())
          return;
      size_t start_pos = 0;
      while((start_pos = find(from, start_pos)) != std::string::npos) {
          replace(start_pos, from.length(), to);
          start_pos += to.length();
      }
    }

  };

  template<typename>
  struct is_std_vector : std::false_type {};

  template<typename T, typename A>
  struct is_std_vector<std::vector<T, A>> : std::true_type {};

  template <typename T>
  inline T string_read_data(auto& f, auto& off) {
    if constexpr (std::is_same<std::string, T>::value || 
      std::is_same<std::string, T>::value) {
      uint64_t len = string_read_data<uint64_t>(f, off);
      std::string str;
      str.resize(len);
      memcpy(str.data(), &f[off], len);
      off += len;
      return str;
    }
    else if constexpr (is_std_vector<T>::value) {
      uint64_t len = string_read_data<uint64_t>(f, off);
      std::vector<typename T::value_type> vec(len);
      if (len > 0) {
        memcpy(vec.data(), &f[off], len * sizeof(typename T::value_type));
        off += len * sizeof(typename T::value_type);
      }
      return vec;
    }
    else {
      auto obj = &f[off];
      off += sizeof(T);
      return *(T*)obj;
    }
  }

  template <typename T>
  void write_to_string(auto& f, const T& o) {
    if constexpr (std::is_same<std::string, T>::value ||
      std::is_same<std::string, T>::value) {
      uint64_t len = o.size();
      f.append((char*)&len, sizeof(len));
      f.append(o.data(), len);
    }
    else if constexpr (is_std_vector<T>::value) {
      uint64_t len = o.size();
      f.append((char*)&len, sizeof(len));
      f.append((char*)o.data(), len * sizeof(T));
    }
    else {
      f.append((char*)&o, sizeof(o));
    }
  }
  template <typename T>
  T vector_read_data(const bytes_t& vec, size_t& off) {
    if constexpr (std::is_same<std::string, T>::value) {
      uint64_t len = vector_read_data<uint64_t>(vec, off);
      std::string str(len, '\0');
      std::memcpy(str.data(), vec.data() + off, len);
      off += len;
      return str;
    }
    else if constexpr (is_std_vector<T>::value) {
      uint64_t len = vector_read_data<uint64_t>(vec, off);
      std::vector<typename T::value_type> vec(len);
      if (len > 0) {
        std::memcpy(vec.data(), vec.data() + off, len * sizeof(typename T::value_type));
        off += len * sizeof(typename T::value_type);
      }
      return vec;
    }
    else {
      T obj;
      std::memcpy(&obj, vec.data() + off, sizeof(T));
      off += sizeof(T);
      return obj;
    }
  }

  std::string wstring_to_utf8(const std::wstring& w) {
    std::string out;
    out.reserve(w.size());

    for (wchar_t wc : w) {
      uint32_t c = (uint32_t)wc;

      if (c <= 0x7F) {
        out.push_back((char)c);
      }
      else if (c <= 0x7FF) {
        out.push_back((char)(0xC0 | ((c >> 6) & 0x1F)));
        out.push_back((char)(0x80 | (c & 0x3F)));
      }
      else if (c <= 0xFFFF) {
        out.push_back((char)(0xE0 | ((c >> 12) & 0x0F)));
        out.push_back((char)(0x80 | ((c >> 6) & 0x3F)));
        out.push_back((char)(0x80 | (c & 0x3F)));
      }
      else {
        out.push_back((char)(0xF0 | ((c >> 18) & 0x07)));
        out.push_back((char)(0x80 | ((c >> 12) & 0x3F)));
        out.push_back((char)(0x80 | ((c >> 6) & 0x3F)));
        out.push_back((char)(0x80 | (c & 0x3F)));
      }
    }

    return out;
  }

  template <typename T>
  void read_from_string(auto& f, auto& off, T& data) {
    data = fan::string_read_data<T>(f, off);
  }

  template <typename T>
  T string_to(const std::string& fstring) {
    T out;
    out.from_string(fstring);
    return out;
  }


  template <typename T>
  void write_to_vector(bytes_t& vec, const T& o) {
    if constexpr (std::is_same<std::string, T>::value) {
      uint64_t len = o.size();
      vec.insert(vec.end(), reinterpret_cast<const uint8_t*>(&len), reinterpret_cast<const uint8_t*>(&len + 1));
      vec.insert(vec.end(), o.begin(), o.end());
    }
    else if constexpr (is_std_vector<T>::value) {
      uint64_t len = o.size();
      vec.insert(vec.end(), reinterpret_cast<const uint8_t*>(&len), reinterpret_cast<const uint8_t*>(&len + 1));
      vec.insert(vec.end(), reinterpret_cast<const uint8_t*>(o.data()), reinterpret_cast<const uint8_t*>(o.data() + len));
    }
    else {
      vec.insert(vec.end(), reinterpret_cast<const uint8_t*>(&o), reinterpret_cast<const uint8_t*>(&o + 1));
    }
  }

  template <typename T>
  void read_from_vector(const bytes_t& vec, size_t& off, T& data) {
    data = vector_read_data<T>(vec, off);
  }

  std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(' ');
    if (std::string::npos == first) {
      return str;
    }
    size_t last = str.find_last_not_of(' ');
    return str.substr(first, (last - first + 1));
  }
  std::vector<std::string> split(const std::string& str, std::string_view token = "\n") {
    std::vector<std::string> result;
    std::size_t start = 0;
    std::size_t pos = 0;

    while ((pos = str.find(token, start)) != std::string::npos) {
      result.push_back(str.substr(start, pos - start));
      start = pos + token.size();
    }
    result.push_back(str.substr(start));

    return result;
  }

  std::vector<std::string> split_quoted(const std::string& input) {
    std::vector<std::string> args;
    std::istringstream stream(input);
    std::string arg;

    while (stream >> std::quoted(arg)) {
      args.push_back(arg);
    }

    return args;
  }

  /// <summary>
  /// Formats a number with thousands separators for improved readability.
  /// </summary>
  /// <param name="number">The numeric value to format (any numeric type).</param>
  /// <param name="separator">The string to use as separator (defaults to comma).</param>
  /// <param name="group_size">Number of digits between separators (defaults to 3).</param>
  /// <returns>Formatted string with separators inserted at specified intervals.</returns>
  std::string number_separator(auto number, const std::string& separator = ",", uint32_t group_size = 3) {
    std::string result = std::to_string(number);

    if (result.length() <= group_size || group_size == 0) {
      return result;
    }

    size_t digit_start = (result[0] == '-') ? 1 : 0;
    size_t digit_count = result.length() - digit_start;

    size_t separator_count = (digit_count - 1) / group_size;
    result.reserve(result.length() + separator_count * separator.length());

    for (int64_t insert_pos = static_cast<int64_t>(result.length() - group_size);
      insert_pos > static_cast<int64_t>(digit_start);
      insert_pos -= group_size) {
      result.insert(static_cast<size_t>(insert_pos), separator);
    }

    return result;
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

  template <typename T>
  std::string format_number(T v) {
    std::string s = std::to_string(v);
    while (s.back() == '0' && s.contains('.')) s.pop_back();
    if (s.back() == '.') s.pop_back();
    return s;
  }

  f64_t parse_f64(fan::str_view_t str) {
    f64_t result = 0;
    auto [ptr, ec] = std::from_chars(str.data(), str.data() + str.size(), result);
    return ec == std::errc {} ? result : 0;
  }

  inline std::string as_chars(const bytes_t& v) {
    return std::string(v.begin(), v.end());
  }

  inline std::string as_chars(std::string_view v) {
    return std::string(v);
  }

  template <typename range_t>
  inline std::string as_bytes(const range_t& v) {
    std::string s;
    for (int i = 0; i < (int)v.size(); ++i) {
      s += std::to_string((uint8_t)v[i]) + (i + 1 < (int)v.size() ? ", " : "");
    }
    return s;
  }

  std::string xor2hex(const bytes_t& a, const bytes_t& b) {
    std::ostringstream r;
    r << std::hex << std::setfill('0');
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
      r << std::setw(2) << static_cast<int>(a[i] ^ b[i]);
    }
    return r.str();
  }

  std::string xor_bytes(const bytes_t& a, const bytes_t& b) {
    std::string r;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
      r += static_cast<char>(a[i] ^ b[i]);
    }
    return r;
  }

  bytes_t hex2bytes(std::string_view s) {
    bytes_t r(s.size() / 2);
    for (int i = 0; i < (int)r.size(); ++i) {
      std::from_chars(s.data() + i * 2, s.data() + i * 2 + 2, r[i], 16);
    }
    return r;
  }

  bytes_t xor_key(const bytes_t& a, uint8_t key) {
    bytes_t r(a.size());
    for (int i = 0; i < (int)a.size(); ++i) {
      r[i] = a[i] ^ key;
    }
    return r;
  }

  std::vector<std::string_view> split_every_n(std::string_view s, size_t n) {
    if (n == 0) return {};
    return s | std::views::chunk(n)
      | std::views::transform([](auto c) { return std::string_view(c.begin(), c.end()); })
      | std::ranges::to<std::vector>();
  }

  inline std::size_t strip_newlines(std::string& str) {
    std::size_t pos = str.find_first_of("\n\r");
    if (pos == std::string::npos) {
      return str.size();
    }
    std::erase_if(str, [](char c) { return c == '\n' || c == '\r'; });
    return pos;
  }

  inline std::vector<std::string_view> split_lines(std::string_view str) {
    std::vector<std::string_view> lines;
    std::size_t start = 0;
    while (start < str.size()) {
      std::size_t end = str.find_first_of("\r\n", start);
      if (end > start) {
        lines.push_back(str.substr(start, end - start));
      }
      if (end == std::string_view::npos) {
        break;
      }
      start = end + 1;
    }
    return lines;
  }

  #define fan_enum_string_runtime(m_name, ...) \
    enum m_name { __VA_ARGS__ }; \
    inline std::vector<std::string> m_name##_strings = fan::split(#__VA_ARGS__)
}