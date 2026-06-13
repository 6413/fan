module;

#include <algorithm>
#include <cstdint>
#include <exception>
#include <ios>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

export module fan.types.json;

import fan.types;
import fan.types.vector;
import fan.types.color;
import fan.types.compile_time_string;
import fan.io.file;

export {
  namespace fan {
    struct json;

    struct json_iterator {
      json_iterator();
      ~json_iterator();
      json_iterator(const json_iterator& other);
      json_iterator& operator=(const json_iterator& other);
      json_iterator& operator++();
      json_iterator operator++(int);
      bool operator!=(const json_iterator& other) const;
      bool operator==(const json_iterator& other) const;
      int operator-(const json_iterator& other) const;
      json_iterator operator-(int n) const;
      json_iterator operator+(int n) const;
      
      std::string key() const;
      json& value();
      json& operator*();
      json* operator->();

      void* m_it = nullptr;
      json* m_current = nullptr;
    };

    struct json {
      using iterator = json_iterator;
      using const_iterator = json_iterator;
      using value_type = json;
      using reference = json&;
      using const_reference = const json&;
      using size_type = std::size_t;
      using array_t = std::vector<json>;
      using object_t = std::unordered_map<std::string, json>;

      enum class value_t : std::uint8_t {
        null, object, array, string, boolean, number_integer, number_unsigned, number_float, binary, discarded
      };

      json();
      ~json();
      json(const json& other);
      json(json&& other) noexcept;
      
      json(int v);
      json(std::uint32_t v);
      json(std::int64_t v);
      json(std::uint64_t v);
      json(f32_t v);
      json(f64_t v);
      json(bool v);
      json(char v);
      json(const char* v);
      json(const std::string& v);

      json& operator=(const json& other);
      json& operator=(json&& other) noexcept;
      
      static json parse(const std::string& raw);
      static json load_file(fan::str_view_t path);
      bool save(fan::str_view_t path, int indent = 2) const;
      static bool save_file(fan::str_view_t path, const json& j, int indent = 2);
      static json object();
      static json array();

      json& operator=(int v);
      json& operator=(std::uint32_t v);
      json& operator=(std::int64_t v);
      json& operator=(std::uint64_t v);
      json& operator=(f32_t v);
      json& operator=(f64_t v);
      json& operator=(bool v);
      json& operator=(char v);
      json& operator=(const char* v);
      json& operator=(const std::string& v);
      json& operator=(const fan::color& v);
      template <typename t_type> json& operator=(const fan::vec2_wrap_t<t_type>& v);
      template <typename t_type> json& operator=(const fan::vec3_wrap_t<t_type>& v);
      template <typename t_type> json& operator=(const fan::vec4_wrap_t<t_type>& v);
      template <typename t_type> json& operator=(const std::vector<t_type>& v) {
        *this = json::array();
        for (const auto& item : v) {
          this->push_back(item);
        }
        return *this;
      }

      json& operator+=(const json& val);

      template<typename T, typename std::enable_if_t<std::is_enum_v<T>, int> = 0>
      json& operator=(T v) {
        return *this = static_cast<std::underlying_type_t<T>>(v);
      }
      template<typename T, typename std::enable_if_t<std::is_enum_v<T>, int> = 0>
      operator T() const {
        return static_cast<T>(get<std::underlying_type_t<T>>());
      }

      json operator[](const char* key);
      json operator[](const std::string& key);
      json operator[](std::size_t index);
      json operator[](int index);
      const json operator[](const char* key) const;
      const json operator[](const std::string& key) const;
      const json operator[](std::size_t index) const;
      const json operator[](int index) const;

      json at(const char* key);
      json at(std::size_t index);
      const json at(const char* key) const;
      const json at(std::size_t index) const;

      void update(const json& other, bool merge_objects = false);

      bool operator==(int val) const;
      bool operator!=(int val) const;
      bool operator==(const json& val) const;
      bool operator!=(const json& val) const;

      template<typename T, typename std::enable_if_t<std::is_enum_v<T>, int> = 0>
      bool operator==(T val) const { 
        if (!m_ptr) return false;
        return get<std::underlying_type_t<T>>() == static_cast<std::underlying_type_t<T>>(val);
      }
      template<typename T, typename std::enable_if_t<std::is_enum_v<T>, int> = 0>
      bool operator!=(T val) const { 
        return !(*this == val);
      }

      value_t type() const;
      bool is_string() const;
      bool is_number() const;
      bool is_boolean() const;
      bool contains(const char* key) const;
      bool is_object() const;
      bool is_array() const;
      bool is_null() const;
      std::size_t size() const;
      bool empty() const;
      
      void push_back(const json& val);
      void push_back(const std::string& val);
      void push_back(const char* val);
      std::string dump(int indent = -1, const char* indent_char = " ", bool ensure_ascii = false) const;

      void reserve(std::size_t n);

      template <typename t_type> t_type _get_impl() const;

      template <typename t_type> t_type get() const {
        if constexpr (std::is_enum_v<t_type>) {
          return static_cast<t_type>(_get_impl<std::underlying_type_t<t_type>>());
        } else {
          return _get_impl<t_type>();
        }
      }
      
      template <typename t_type> t_type value(const char* key, const t_type& default_value) const {
        if (contains(key)) { return (*this)[key].get<t_type>(); }
        return default_value;
      }
      template <typename t_type> t_type value(const std::string& key, const t_type& default_value) const {
        return value(key.c_str(), default_value);
      }
      std::string value(const char* key, const char* default_value) const {
        if (contains(key)) { return (*this)[key].get<std::string>(); }
        return std::string(default_value);
      }
      std::string value(const std::string& key, const char* default_value) const {
        return value(key.c_str(), default_value);
      }

      operator int() const { return get<int>(); }
      operator std::uint32_t() const { return get<std::uint32_t>(); }
      operator std::int64_t() const { return get<std::int64_t>(); }
      operator std::uint64_t() const { return get<std::uint64_t>(); }
      operator std::uint16_t() const { return get<std::uint16_t>(); }
      operator std::int16_t() const { return get<std::int16_t>(); }
      operator std::uint8_t() const { return get<std::uint8_t>(); }
      operator std::int8_t() const { return get<std::int8_t>(); }
      operator char() const { return get<char>(); }
      operator f32_t() const { return get<f32_t>(); }
      operator f64_t() const { return get<f64_t>(); }
      operator bool() const { return get<bool>(); }
      operator std::string() const { return get<std::string>(); }
      operator std::string_view() const;
      operator fan::str_view_t() const;
      operator fan::color() const { return get<fan::color>(); }

      template <typename t_type> operator fan::vec2_wrap_t<t_type>() const { return get<fan::vec2_wrap_t<t_type>>(); }
      template <typename t_type> operator fan::vec3_wrap_t<t_type>() const { return get<fan::vec3_wrap_t<t_type>>(); }
      template <typename t_type> operator fan::vec4_wrap_t<t_type>() const { return get<fan::vec4_wrap_t<t_type>>(); }

      template <typename T>
      operator std::vector<T>() const {
        std::vector<T> result;
        for (auto it = begin(); it != end(); ++it) {
          result.push_back(static_cast<T>(*it));
        }
        return result;
      }

      void* get_internal_ptr() const { return m_ptr; }

      json_iterator begin();
      json_iterator end();
      json_iterator begin() const;
      json_iterator end() const;
      json_iterator cbegin() const;
      json_iterator cend() const;

      template<typename func_t> void find_and_iterate(const std::string& search_string, func_t&& callback) {
        if (this->is_object()) {
          for (auto it = this->begin(); it != this->end(); ++it) {
            if (it.key() == search_string) { callback(it.value()); }
            it.value().find_and_iterate(search_string, callback);
          }
        }
        else if (this->is_array()) {
          for (auto& item : *this) { item.find_and_iterate(search_string, callback); }
        }
      }

      template<typename t_type> void get_if(const char* key, t_type& out) const {
        if (contains(key)) {
          if constexpr (std::is_enum_v<t_type>) {
            out = static_cast<t_type>((*this)[key].get<std::underlying_type_t<t_type>>());
          } else {
            out = (*this)[key].get<t_type>();
          }
        }
      }

      template<typename t_type> void set(const char* key, const t_type& val) {
        (*this)[key] = val;
      }
      template<typename t_type> static void load_struct(fan::str_view_t path, t_type& obj) {
        auto j = load_file(path);
        obj.json_read(j);
      }
      template<typename t_type> static void save_struct(fan::str_view_t path, const t_type& obj) {
        fan::json j = json::object();
        obj.json_write(j);
        j.save(path);
      }
      json(void* ptr, bool is_ref);

      void preserve_unknown(const fan::json& source);

      void* m_ptr = nullptr;
      bool m_is_ref = false;
    };

    template <> int json::_get_impl<int>() const;
    template <> std::uint32_t json::_get_impl<std::uint32_t>() const;
    template <> std::int64_t json::_get_impl<std::int64_t>() const;
    template <> std::uint64_t json::_get_impl<std::uint64_t>() const;
    template <> std::uint16_t json::_get_impl<std::uint16_t>() const;
    template <> std::int16_t json::_get_impl<std::int16_t>() const;
    template <> std::uint8_t json::_get_impl<std::uint8_t>() const;
    template <> std::int8_t json::_get_impl<std::int8_t>() const;
    template <> char json::_get_impl<char>() const;
    template <> f32_t json::_get_impl<f32_t>() const;
    template <> f64_t json::_get_impl<f64_t>() const;
    template <> bool json::_get_impl<bool>() const;
    template <> std::string json::_get_impl<std::string>() const;
    template <> fan::color json::_get_impl<fan::color>() const;
    template <> fan::vec2_wrap_t<f32_t> json::_get_impl<fan::vec2_wrap_t<f32_t>>() const;
    template <> fan::vec2_wrap_t<int> json::_get_impl<fan::vec2_wrap_t<int>>() const;
    template <> fan::vec3_wrap_t<f32_t> json::_get_impl<fan::vec3_wrap_t<f32_t>>() const;
    template <> fan::vec3_wrap_t<int> json::_get_impl<fan::vec3_wrap_t<int>>() const;
    template <> fan::vec4_wrap_t<f32_t> json::_get_impl<fan::vec4_wrap_t<f32_t>>() const;
    template <> fan::vec4_wrap_t<int> json::_get_impl<fan::vec4_wrap_t<int>>() const;
    template <> std::vector<std::string> json::_get_impl<std::vector<std::string>>() const;

    struct json_stream_parser_t {
      struct parsed_result {
        fan::json value;
        std::string error;
        bool success;
      };
      std::pair<std::size_t, std::size_t> find_next_json_bounds(std::string_view s, std::size_t pos = 0) const noexcept;
      std::vector<parsed_result> process(std::string_view chunk);
      void clear() noexcept;

      std::string buf;
    };
  }
}