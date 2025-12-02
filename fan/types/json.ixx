module;

#include <fan/utility.h>

// With windows clang build there can be msvc and clang both defined
#if defined(fan_compiler_msvc) && !defined(fan_compiler_clang)
  import <fan/types/json_impl.h>;
#else
  #include <fan/types/json_impl.h>
#endif

export module fan.types.json;

import fan.types.vector;
import fan.types.color;

export {
  namespace fan {
    struct json : nlohmann::json {
      using base = nlohmann::json;
      using base::base;
      json() = default;
      json(const base& j) : base(j) {}
      json(base&& j) : base(std::move(j)) {}

    // nice pain

      template<typename... Args>
      static json parse(Args&&... args) {
        return json(base::parse(std::forward<Args>(args)...));
      }

      json& operator[](const std::string& key) {
        return static_cast<json&>(base::operator[](key));
      }
      const json& operator[](const std::string& key) const {
        return static_cast<const json&>(base::operator[](key));
      }
      json& operator[](std::string&& key) {
        return static_cast<json&>(base::operator[](std::move(key)));
      }
      json& operator[](const char* key) {
        return static_cast<json&>(base::operator[](key));
      }
      const json& operator[](const char* key) const {
        return static_cast<const json&>(base::operator[](key));
      }
      json& operator[](size_t idx) {
        return static_cast<json&>(base::operator[](idx));
      }
      const json& operator[](size_t idx) const {
        return static_cast<const json&>(base::operator[](idx));
      }

      class iterator : public base::iterator {
      public:
        using base_iter = base::iterator;

        iterator() = default;
        iterator(base::iterator it) : base_iter(it) {}

        json& operator*() const {
          return static_cast<json&>(base_iter::operator*());
        }
        json* operator->() const {
          return static_cast<json*>(&base_iter::operator*());
        }

        bool operator==(const iterator& other) const {
          return base_iter::operator==(static_cast<const base_iter&>(other));
        }
        bool operator!=(const iterator& other) const {
          return base_iter::operator!=(static_cast<const base_iter&>(other));
        }
        bool operator<(const iterator& other) const {
          return base_iter::operator<(static_cast<const base_iter&>(other));
        }
        bool operator<=(const iterator& other) const {
          return base_iter::operator<=(static_cast<const base_iter&>(other));
        }
        bool operator>(const iterator& other) const {
          return base_iter::operator>(static_cast<const base_iter&>(other));
        }
        bool operator>=(const iterator& other) const {
          return base_iter::operator>=(static_cast<const base_iter&>(other));
        }

        iterator& operator++() {
          base_iter::operator++();
          return *this;
        }
        iterator operator++(int) {
          iterator tmp = *this;
          base_iter::operator++();
          return tmp;
        }
        iterator& operator--() {
          base_iter::operator--();
          return *this;
        }
        iterator operator--(int) {
          iterator tmp = *this;
          base_iter::operator--();
          return tmp;
        }
        iterator& operator+=(typename base_iter::difference_type n) {
          base_iter::operator+=(n);
          return *this;
        }
        iterator operator+(typename base_iter::difference_type n) const {
          iterator tmp = *this;
          tmp += n;
          return tmp;
        }
        iterator& operator-=(typename base_iter::difference_type n) {
          base_iter::operator-=(n);
          return *this;
        }
        iterator operator-(typename base_iter::difference_type n) const {
          iterator tmp = *this;
          tmp -= n;
          return tmp;
        }
        typename base_iter::difference_type operator-(const iterator& other) const {
          return base_iter::operator-(static_cast<const base_iter&>(other));
        }
      };

      class const_iterator : public base::const_iterator {
      public:
        using base_iter = base::const_iterator;

        const_iterator() = default;
        const_iterator(base::const_iterator it) : base_iter(it) {}

        const json& operator*() const {
          return static_cast<const json&>(base_iter::operator*());
        }
        const json* operator->() const {
          return static_cast<const json*>(&base_iter::operator*());
        }

        bool operator==(const const_iterator& other) const {
          return base_iter::operator==(static_cast<const base_iter&>(other));
        }
        bool operator!=(const const_iterator& other) const {
          return base_iter::operator!=(static_cast<const base_iter&>(other));
        }
        bool operator<(const const_iterator& other) const {
          return base_iter::operator<(static_cast<const base_iter&>(other));
        }
        bool operator<=(const const_iterator& other) const {
          return base_iter::operator<=(static_cast<const base_iter&>(other));
        }
        bool operator>(const const_iterator& other) const {
          return base_iter::operator>(static_cast<const base_iter&>(other));
        }
        bool operator>=(const const_iterator& other) const {
          return base_iter::operator>=(static_cast<const base_iter&>(other));
        }

        const_iterator& operator++() {
          base_iter::operator++();
          return *this;
        }
        const_iterator operator++(int) {
          const_iterator tmp = *this;
          base_iter::operator++();
          return tmp;
        }
        const_iterator& operator--() {
          base_iter::operator--();
          return *this;
        }
        const_iterator operator--(int) {
          const_iterator tmp = *this;
          base_iter::operator--();
          return tmp;
        }
        const_iterator& operator+=(typename base_iter::difference_type n) {
          base_iter::operator+=(n);
          return *this;
        }
        const_iterator operator+(typename base_iter::difference_type n) const {
          const_iterator tmp = *this;
          tmp += n;
          return tmp;
        }
        const_iterator& operator-=(typename base_iter::difference_type n) {
          base_iter::operator-=(n);
          return *this;
        }
        const_iterator operator-(typename base_iter::difference_type n) const {
          const_iterator tmp = *this;
          tmp -= n;
          return tmp;
        }
        typename base_iter::difference_type operator-(const const_iterator& other) const {
          return base_iter::operator-(static_cast<const base_iter&>(other));
        }
      };

      iterator begin() {
        return iterator(base::begin());
      }
      iterator end() {
        return iterator(base::end());
      }
      const_iterator begin() const {
        return const_iterator(base::begin());
      }
      const_iterator end() const {
        return const_iterator(base::end());
      }
      const_iterator cbegin() const {
        return const_iterator(base::cbegin());
      }
      const_iterator cend() const {
        return const_iterator(base::cend());
      }

      template<typename func_t>
      void find_and_iterate(const std::string& search_string, func_t&& callback) {
        if (this->is_object()) {
          for (auto it = this->begin(); it != this->end(); ++it) {
            if (it.key() == search_string) {
              callback(static_cast<json&>(it.value()));
            }
            static_cast<json&>(it.value()).find_and_iterate(search_string, callback);
          }
        }
        else if (this->is_array()) {
          for (auto& item : *this) {
            item.find_and_iterate(search_string, callback);
          }
        }
      }
    };
  }



  template <typename T>
  struct nlohmann::adl_serializer<fan::vec2_wrap_t<T>> {
    static void to_json(nlohmann::json& j, const fan::vec2_wrap_t<T>& v) {
      j = nlohmann::json{ v.x, v.y };
    }
    static void from_json(const nlohmann::json& j, fan::vec2_wrap_t<T>& v) {
      v.x = j[0].get<T>();
      v.y = j[1].get<T>();
    }
  };

  template <typename T>
  struct nlohmann::adl_serializer<fan::vec3_wrap_t<T>> {
    static void to_json(nlohmann::json& j, const fan::vec3_wrap_t<T>& v) {
      j = nlohmann::json{ v.x, v.y, v.z };
    }
    static void from_json(const nlohmann::json& j, fan::vec3_wrap_t<T>& v) {
      v.x = j[0].get<T>();
      v.y = j[1].get<T>();
      v.z = j[2].get<T>();
    }
  };

  template <typename T>
  struct nlohmann::adl_serializer<fan::vec4_wrap_t<T>> {
    static void to_json(nlohmann::json& j, const fan::vec4_wrap_t<T>& v) {
      j = nlohmann::json{ v.x, v.y, v.z, v.w };
    }
    static void from_json(const nlohmann::json& j, fan::vec4_wrap_t<T>& v) {
      v.x = j[0].get<T>();
      v.y = j[1].get<T>();
      v.z = j[2].get<T>();
      v.w = j[3].get<T>();
    }
  };

  template <typename T> 
  struct nlohmann::adl_serializer<fan::color_<T>> {
    static void to_json(nlohmann::json& j, const fan::color_<T>& c) {
      j = nlohmann::json{ c[0], c[1], c[2], c[3]};
    }
    static void from_json(const nlohmann::json& j, fan::color_<T>& c) {
      c.r = j[0];
      c.g = j[1];
      c.b = j[2];
      c.a = j[3];
    }
  };
}