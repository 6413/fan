module;

#include <iomanip>

#include <fan/utility.h>

export module fan.types.fstring;

import std;

import fan.types;
import fan.print.error; // for throw_error with msg
import fan.types.vector;
import fan.types.compile_time_string;

export namespace fan {

  template <typename T>
  auto to_string(const T a_value, const int n = 2) {
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
  }

  inline constexpr std::uint64_t get_hash(const std::string_view& str) {
    std::uint64_t result = 0xcbf29ce484222325; // FNV offset basis

    std::uint32_t i = 0;

    while (i < str.size()) {
      result ^= (std::uint64_t)str[i];
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
      for (std::uint32_t i = 0; i < vector.size() * sizeof(T); ++i) {
        append((std::uint8_t*)&vector[i], (std::uint8_t*)&vector[i] + sizeof(T));
      }
    }

    using char_type = std::string::value_type;

    static constexpr std::uint8_t UTF8_SizeOfCharacter(std::uint8_t byte) {
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

    std::uint32_t get_utf8_character(std::uintptr_t offset, std::uint8_t size) const {
      std::uint32_t code = 0;
      for (int j = 0; j < size; j++) {
        code <<= 8;
        code |= (*this)[offset + j];
      }
      return code;
    }
    void replace_all(const std::string& from, const std::string& to) {
      if(from.empty())
          return;
      std::size_t start_pos = 0;
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
      std::uint64_t len = string_read_data<std::uint64_t>(f, off);
      std::string str;
      str.resize(len);
      std::memcpy(str.data(), &f[off], len);
      off += len;
      return str;
    }
    else if constexpr (is_std_vector<T>::value) {
      std::uint64_t len = string_read_data<std::uint64_t>(f, off);
      std::vector<typename T::value_type> vec(len);
      if (len > 0) {
        std::memcpy(vec.data(), &f[off], len * sizeof(typename T::value_type));
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
      std::uint64_t len = o.size();
      f.append((char*)&len, sizeof(len));
      f.append(o.data(), len);
    }
    else if constexpr (is_std_vector<T>::value) {
      std::uint64_t len = o.size();
      f.append((char*)&len, sizeof(len));
      f.append((char*)o.data(), len * sizeof(T));
    }
    else {
      f.append((char*)&o, sizeof(o));
    }
  }
  template <typename T>
  T vector_read_data(const bytes_t& vec, std::size_t& off) {
    if constexpr (std::is_same<std::string, T>::value) {
      std::uint64_t len = vector_read_data<std::uint64_t>(vec, off);
      std::string str(len, '\0');
      std::memcpy(str.data(), vec.data() + off, len);
      off += len;
      return str;
    }
    else if constexpr (is_std_vector<T>::value) {
      std::uint64_t len = vector_read_data<std::uint64_t>(vec, off);
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
      std::uint32_t c = (std::uint32_t)wc;

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
      std::uint64_t len = o.size();
      vec.insert(vec.end(), reinterpret_cast<const std::uint8_t*>(&len), reinterpret_cast<const std::uint8_t*>(&len + 1));
      vec.insert(vec.end(), o.begin(), o.end());
    }
    else if constexpr (is_std_vector<T>::value) {
      std::uint64_t len = o.size();
      vec.insert(vec.end(), reinterpret_cast<const std::uint8_t*>(&len), reinterpret_cast<const std::uint8_t*>(&len + 1));
      vec.insert(vec.end(), reinterpret_cast<const std::uint8_t*>(o.data()), reinterpret_cast<const std::uint8_t*>(o.data() + len));
    }
    else {
      vec.insert(vec.end(), reinterpret_cast<const std::uint8_t*>(&o), reinterpret_cast<const std::uint8_t*>(&o + 1));
    }
  }

  template <typename T>
  void read_from_vector(const bytes_t& vec, std::size_t& off, T& data) {
    data = vector_read_data<T>(vec, off);
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
  
  std::vector<std::string> lines(const std::string& str)  {
    return split(str, "\n");
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

  std::string format_thousands(auto number, const std::string& separator = ",", std::uint32_t group_size = 3) {
    std::string result = std::to_string(number);

    if (result.length() <= group_size || group_size == 0) {
      return result;
    }

    std::size_t digit_start = (result[0] == '-') ? 1 : 0;
    std::size_t digit_count = result.length() - digit_start;

    std::size_t separator_count = (digit_count - 1) / group_size;
    result.reserve(result.length() + separator_count * separator.length());

    for (std::int64_t insert_pos = static_cast<std::int64_t>(result.length() - group_size);
      insert_pos > static_cast<std::int64_t>(digit_start);
      insert_pos -= group_size) {
      result.insert(static_cast<std::size_t>(insert_pos), separator);
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

  template <std::ranges::contiguous_range R>
  requires std::same_as<std::ranges::range_value_t<R>, std::uint8_t>
        || std::same_as<std::ranges::range_value_t<R>, char>
  inline std::string as_chars(R&& v) {
    return std::string(std::ranges::begin(v), std::ranges::end(v));
  }

  template <typename range_t>
  inline bytes_t as_bytes(const range_t& v) {
    return bytes_t(v.begin(), v.end());
  }
  template <typename range_t>
  inline std::vector<bytes_t> as_bytes(const std::vector<range_t>& v) {
    std::vector<bytes_t> bytes(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
      bytes[i] = bytes_t(v[i].begin(), v[i].end());
    }
    return bytes;
  }

  std::vector<std::string_view> chunks(std::string_view s, std::size_t n) {
    if (n == 0) return {};
    std::vector<std::string_view> out;
    for (std::size_t i = 0; i < s.size(); i += n) {
      out.emplace_back(s.data() + i, std::min(n, s.size() - i));
    }
    return out;
  }

  inline std::size_t strip_newlines(std::string& str) {
    std::size_t pos = str.find_first_of("\n\r");
    if (pos == std::string::npos) {
      return str.size();
    }
    std::erase_if(str, [](char c) { return c == '\n' || c == '\r'; });
    return pos;
  }
  inline std::string strip_newlines(const std::string& str) {
    std::string result = str;
    strip_newlines(result);
    return result;
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

  inline std::string to_hex(std::uint64_t v, int width = 0) {
    char buf[32];
    auto [ptr, ec] = std::to_chars(buf, buf + sizeof(buf), v, 16);
    for (char* p = buf; p != ptr; ++p) { if (*p >= 'a') *p &= ~0x20; }

    int len = (int)(ptr - buf);
    if (width <= len) return std::string(buf, ptr);
    return std::string(width - len, '0').append(buf, ptr);
  }
  inline constexpr std::uint64_t from_hex(std::string_view s) {
    std::uint64_t v = 0;
    std::from_chars(s.data(), s.data() + s.size(), v, 16);
    return v;
  }
  inline std::uint8_t parse_hex_byte(std::string_view s) {
    if (s.empty()) return 0;
    if (s.size() > 2) s = s.substr(s.size() - 2);
    return (std::uint8_t)fan::from_hex(s);
  }

  inline std::string to_ascii(std::uint8_t b) {
    if (b >= 0x20 && b <= 0x7E) return std::string(1, (char)b);
    //if (b == 0x0A) return "";
    //if (b == 0x0D) return "";
    return "·";
  }

  inline void strip_whitespace(std::string& str) {
    while (!str.empty() && std::isspace((std::uint8_t)str.front())) str.erase(str.begin());
    while (!str.empty() && std::isspace((std::uint8_t)str.back())) str.pop_back();
  }
  inline std::string strip_whitespace(const std::string& str) {
    std::string r = str;
    strip_whitespace(r);
    return r;
  }

  inline std::string_view trim(std::string_view s) {
    s.remove_prefix(std::min(s.find_first_not_of(" \t\n\r\f\v"), s.size()));
    auto last = s.find_last_not_of(" \t\n\r\f\v");
    if (last != std::string_view::npos) s.remove_suffix(s.size() - last - 1);
    return s;
  }

  std::vector<std::uint8_t> parse_hex_buffer(std::string_view hex_str) {
    std::vector<std::uint8_t> bytes;
    bytes.reserve(hex_str.size() / 2);
  
    std::string clean;
    clean.reserve(hex_str.size());
  
    for (std::size_t i = 0; i < hex_str.size(); ++i) {
      if (hex_str[i] == '0' && i + 1 < hex_str.size() && (hex_str[i+1] | 0x20) == 'x') {
        ++i; continue;
      }
      if (std::isxdigit((std::uint8_t)hex_str[i])) clean.push_back(hex_str[i]);
    }

    for (std::size_t i = 0; i + 1 < clean.size(); i += 2) {
      bytes.push_back(parse_hex_byte({clean.data() + i, 2}));
    }
    return bytes;
  }

  inline std::optional<std::uint8_t> extract_typed_char(std::string_view buf, std::string_view old_buf) {
    if (buf.empty()) return std::nullopt;
    if (buf.size() == 1) return (std::uint8_t)buf[0];
    if (buf.size() == 2) return (std::uint8_t)((buf[0] == old_buf[0]) ? buf[1] : buf[0]);
    return std::nullopt;
  }

  inline std::string format_scientific(double v) {
    char b[64];
    std::snprintf(b, sizeof(b), "%.6e", v);
    return b;
  }

  #define fan_enum_string_runtime(m_name, ...) \
    enum m_name { __VA_ARGS__ }; \
    inline std::vector<std::string> m_name##_strings = fan::split(#__VA_ARGS__)

  struct args_t {
    args_t(int argc, char** argv) {
      data.reserve(argc);
      for (int i = 0; i < argc; ++i) {
        data.emplace_back(argv[i]);
      }
    }
    auto operator[](size_t i) const {
      return data[i];
    }
    auto size() const {
      return data.size();
    }

    std::vector<std::string_view> data;
  };
} // namespace fan