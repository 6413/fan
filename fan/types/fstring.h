#pragma once

#include <vector>
#include <cstring>
#include <memory>
#include <string>

#include <fan/types/print.h> // for throw_error with msg
#include <fan/types/vector.h>


namespace fan {

  template <typename T>
  auto to_string(const T a_value, const int n) {
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
  }

  static constexpr uint64_t get_hash(const std::string_view& str) {
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

namespace fan {
  struct string : public std::string {

    using type_t = std::string;
    using type_t::basic_string;

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
    constexpr uint32_t get_utf8(std::size_t i) const {
      const std::u8string_view& sv = (char8_t*)c_str();
      uint32_t code = 0;
      uint32_t offset = 0;
      for (uint32_t k = 0; k <= i; k++) {
        code = 0;
        int len = 1;
        if ((sv[offset] & 0xF8) == 0xF0) { len = 4; }
        else if ((sv[offset] & 0xF0) == 0xE0) { len = 3; }
        else if ((sv[offset] & 0xE0) == 0xC0) { len = 2; }
        for (int j = 0; j < len; j++) {
          code <<= 8;
          code |= sv[offset];
          offset++;
        }
      }
      return code;
    }

    std::size_t utf8_size(std::size_t i) const {
      return UTF8_SizeOfCharacter((*this)[i]);
    }
    std::size_t utf8_size() const {
      std::size_t count = 0;
      for (auto i = begin(); i != end(); i++){
        if ((*i & 0xC0) != 0x80) {
          count++;
        }
      }
      return count;
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
  T read_data(auto& f, auto& off) {
    if constexpr (std::is_same<fan::string, T>::value || 
      std::is_same<std::string, T>::value) {
      uint64_t len = read_data<uint64_t>(f, off);
      fan::string str;
      str.resize(len);
      memcpy(str.data(), &f[off], len);
      off += len;
      return str;
    }
    else if constexpr (is_std_vector<T>::value) {
      uint64_t len = read_data<uint64_t>(f, off);
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
    if constexpr (std::is_same<fan::string, T>::value ||
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
  void read_from_string(auto& f, auto& off, T& data) {
    data = fan::read_data<T>(f, off);
  }

  template <typename T>
  T string_to(const fan::string& fstring) {
    T out;
    out.from_string(fstring);
    return out;
  }

  std::string trim(const std::string& str);

  std::vector<std::string> split(const std::string& s);

  #define fan_enum_string_runtime(m_name, ...) \
    enum m_name { __VA_ARGS__ }; \
    inline std::vector<std::string> m_name##_strings = fan::split(#__VA_ARGS__)
}