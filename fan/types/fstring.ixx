module;

#include <fan/types/types.h>

#include <vector>
#include <cstring>
#include <memory>
#include <string>

#include <ios>
#include <sstream>

export module fan.types.fstring;

import fan.print; // for throw_error with msg
import fan.types.vector;

export namespace fan {

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
  T vector_read_data(const std::vector<uint8_t>& vec, size_t& off) {
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
  void write_to_vector(std::vector<uint8_t>& vec, const T& o) {
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
  void read_from_vector(const std::vector<uint8_t>& vec, size_t& off, T& data) {
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

  std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, ',')) {
      std::istringstream tokenStream2(token);
      std::getline(tokenStream2, token, '=');
      tokens.push_back(fan::trim(token));
    }
    return tokens;
  }

  std::vector<std::string> split(const std::string& str, std::string_view token) {
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

  #define fan_enum_string_runtime(m_name, ...) \
    enum m_name { __VA_ARGS__ }; \
    inline std::vector<std::string> m_name##_strings = fan::split(#__VA_ARGS__)
}