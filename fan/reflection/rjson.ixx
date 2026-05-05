module;

export module fan.rjson;

import std;

import fan.reflection;

export namespace fan {
  struct rjson_t {
    using array_t  = std::vector<rjson_t>;
    using object_t = std::vector<std::pair<std::string, rjson_t>>;
    using value_t  = std::variant<std::nullptr_t, bool, std::int64_t, double,
                                  std::string, array_t, object_t>;
    value_t v;

    rjson_t()                         : v(nullptr) {}
    rjson_t(std::nullptr_t)           : v(nullptr) {}
    rjson_t(bool b)                   : v(b) {}
    rjson_t(std::int64_t i)           : v(i) {}
    rjson_t(int i)                    : v((std::int64_t)i) {}
    rjson_t(double d)                 : v(d) {}
    rjson_t(std::string s)            : v(std::move(s)) {}
    rjson_t(std::string_view s)       : v(std::string(s)) {}
    rjson_t(const char* s)            : v(std::string(s)) {}
    rjson_t(array_t a)                : v(std::move(a)) {}
    rjson_t(object_t o)               : v(std::move(o)) {}

    bool is_null()   const { return std::holds_alternative<std::nullptr_t>(v); }
    bool is_bool()   const { return std::holds_alternative<bool>(v); }
    bool is_int()    const { return std::holds_alternative<std::int64_t>(v); }
    bool is_double() const { return std::holds_alternative<double>(v); }
    bool is_number() const { return is_int() || is_double(); }
    bool is_string() const { return std::holds_alternative<std::string>(v); }
    bool is_array()  const { return std::holds_alternative<array_t>(v); }
    bool is_object() const { return std::holds_alternative<object_t>(v); }

    template<typename T> T& get()             { return std::get<T>(v); }
    template<typename T> const T& get() const { return std::get<T>(v); }

    double as_double() const {
      if (is_int())    return (double)std::get<std::int64_t>(v);
      if (is_double()) return std::get<double>(v);
      throw std::runtime_error("rjson: not a number");
    }
    std::int64_t as_int() const {
      if (is_int())    return std::get<std::int64_t>(v);
      if (is_double()) return (std::int64_t)std::get<double>(v);
      throw std::runtime_error("rjson: not a number");
    }

    rjson_t& operator[](std::string_view key) {
      if (!is_object()) v = object_t{};
      auto& obj = get<object_t>();
      for (auto& [k, val] : obj) if (k == key) return val;
      obj.emplace_back(std::string(key), rjson_t{});
      return obj.back().second;
    }
    const rjson_t& operator[](std::string_view key) const {
      auto& obj = get<object_t>();
      for (auto& [k, val] : obj) if (k == key) return val;
      throw std::runtime_error("rjson: key not found: " + std::string(key));
    }
    rjson_t& operator[](std::size_t i) {
      if (!is_array()) v = array_t{};
      auto& arr = get<array_t>();
      if (i >= arr.size()) arr.resize(i + 1);
      return arr[i];
    }
    const rjson_t& operator[](std::size_t i) const { return get<array_t>()[i]; }

    bool contains(std::string_view key) const {
      if (!is_object()) return false;
      for (auto& [k, _] : get<object_t>()) if (k == key) return true;
      return false;
    }

    std::size_t size() const {
      if (is_array())  return get<array_t>().size();
      if (is_object()) return get<object_t>().size();
      if (is_string()) return get<std::string>().size();
      return 0;
    }

    void push_back(rjson_t val) {
      if (!is_array()) v = array_t{};
      get<array_t>().push_back(std::move(val));
    }

    auto begin()       { return get<array_t>().begin(); }
    auto end()         { return get<array_t>().end(); }
    auto begin() const { return get<array_t>().begin(); }
    auto end()   const { return get<array_t>().end(); }
  };

  template<typename T>
  struct is_vector_like : std::false_type {};

  template<typename T, typename A>
  struct is_vector_like<std::vector<T, A>> : std::true_type {};

  namespace rjson_detail {
    inline void escape_string(std::string& out, std::string_view s) {
      out += '"';
      for (char c : s) {
        switch (c) {
          case '"':  out += "\\\""; break;
          case '\\': out += "\\\\"; break;
          case '\n': out += "\\n";  break;
          case '\r': out += "\\r";  break;
          case '\t': out += "\\t";  break;
          default:
            if ((unsigned char)c < 0x20) {
              char buf[8]; std::snprintf(buf, sizeof(buf), "\\u%04x", (unsigned char)c);
              out += buf;
            } else out += c;
        }
      }
      out += '"';
    }

    inline void dump(const rjson_t& j, std::string& out, int indent, int depth) {
      std::visit([&](auto& val) {
        using T = std::decay_t<decltype(val)>;
        if constexpr (std::is_same_v<T, std::nullptr_t>) out += "null";
        else if constexpr (std::is_same_v<T, bool>) out += val ? "true" : "false";
        else if constexpr (std::is_same_v<T, std::int64_t>) out += std::to_string(val);
        else if constexpr (std::is_same_v<T, double>) {
          char buf[32]; std::snprintf(buf, sizeof(buf), "%g", val); out += buf;
        }
        else if constexpr (std::is_same_v<T, std::string>) escape_string(out, val);
        else if constexpr (std::is_same_v<T, rjson_t::array_t>) {
          if (val.empty()) { out += "[]"; return; }
          std::string nl = indent ? "\n" + std::string((depth + 1) * indent, ' ') : "";
          std::string cl = indent ? "\n" + std::string(depth * indent, ' ') : "";
          out += '[';
          for (std::size_t i = 0; i < val.size(); ++i) {
            if (i) out += ",";
            out += nl;
            dump(val[i], out, indent, depth + 1);
          }
          out += cl + ']';
        }
        else if constexpr (std::is_same_v<T, rjson_t::object_t>) {
          if (val.empty()) { out += "{}"; return; }
          std::string nl = indent ? "\n" + std::string((depth + 1) * indent, ' ') : "";
          std::string cl = indent ? "\n" + std::string(depth * indent, ' ') : "";
          std::string sep = indent ? ": " : ":";
          out += '{';
          bool first = true;
          for (auto& [k, v2] : val) {
            if (!first) out += ",";
            first = false;
            out += nl;
            escape_string(out, k);
            out += sep;
            dump(v2, out, indent, depth + 1);
          }
          out += cl + '}';
        }
      }, j.v);
    }

    struct parser_t {
      std::string_view src;
      std::size_t pos = 0;

      char peek() const { return pos < src.size() ? src[pos] : '\0'; }
      char next()       { return pos < src.size() ? src[pos++] : '\0'; }
      void skip_ws() {
        while (pos < src.size() && (src[pos]==' '||src[pos]=='\t'||src[pos]=='\n'||src[pos]=='\r')) ++pos;
      }
      void require(char c) {
        skip_ws();
        if (peek() != c) throw std::runtime_error("rjson: expected");
        ++pos;
      }

      std::string parse_string() {
        require('"');
        std::string s;
        while (pos < src.size()) {
          char c = next();
          if (c == '"') return s;
          if (c != '\\') { s += c; continue; }
          char e = next();
          switch (e) {
            case '"': s += '"'; break;
            case '\\': s += '\\'; break;
            case 'n': s += '\n'; break;
            case 'r': s += '\r'; break;
            case 't': s += '\t'; break;
            default: s += e;
          }
        }
        throw std::runtime_error("rjson: string");
      }

      rjson_t parse_number() {
        std::size_t start = pos;
        bool f = false;
        if (peek() == '-') ++pos;
        while (pos < src.size() && std::isdigit((unsigned char)src[pos])) ++pos;
        if (pos < src.size() && src[pos] == '.') { f = true; ++pos; while (pos < src.size() && std::isdigit((unsigned char)src[pos])) ++pos; }
        std::string_view num = src.substr(start, pos - start);
        if (f) return std::stod(std::string(num));
        return (std::int64_t)std::stoll(std::string(num));
      }

      rjson_t parse_value() {
        skip_ws();
        char c = peek();
        if (c == '"') return parse_string();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == 't') { pos += 4; return true; }
        if (c == 'f') { pos += 5; return false; }
        if (c == 'n') { pos += 4; return nullptr; }
        return parse_number();
      }

      rjson_t parse_object() {
        require('{');
        rjson_t::object_t o;
        skip_ws();
        if (peek() == '}') { ++pos; return o; }
        while (true) {
          auto k = parse_string();
          require(':');
          o.emplace_back(std::move(k), parse_value());
          skip_ws();
          if (peek() == '}') { ++pos; break; }
          require(',');
        }
        return o;
      }

      rjson_t parse_array() {
        require('[');
        rjson_t::array_t a;
        skip_ws();
        if (peek() == ']') { ++pos; return a; }
        while (true) {
          a.push_back(parse_value());
          skip_ws();
          if (peek() == ']') { ++pos; break; }
          require(',');
        }
        return a;
      }
    };
  }

  inline std::string rjson_dump(const rjson_t& j, int indent = 0) {
    std::string out;
    rjson_detail::dump(j, out, indent, 0);
    return out;
  }

  inline rjson_t rjson_parse(std::string_view s) {
    return rjson_detail::parser_t{s}.parse_value();
  }

  template<typename T>
  rjson_t to_rjson(const T& obj) {
    if constexpr (std::is_same_v<T, bool>) return obj;
    else if constexpr (std::is_integral_v<T>) return (std::int64_t)obj;
    else if constexpr (std::is_floating_point_v<T>) return (double)obj;
    else if constexpr (std::is_same_v<T, std::string> ||
                       std::is_same_v<T, std::string_view> ||
                       std::is_same_v<T, const char*>) return obj;
    else if constexpr (is_vector_like<T>::value) {
      rjson_t::array_t arr; arr.reserve(obj.size());
      for (auto& e : obj) arr.push_back(to_rjson(e));
      return arr;
    }
    else if constexpr (std::meta::is_class_type(^^T)) {
      rjson_t::object_t o;
      fan::refl::iterate_members<T>(obj, [&](std::string_view n, auto& v) {
        o.emplace_back(n, to_rjson(v));
      });
      return o;
    }
    else return nullptr;
  }

  template<typename T>
  void from_rjson(T& obj, const rjson_t& j) {
    if constexpr (std::is_same_v<T, bool>) { if (j.is_bool()) obj = j.get<bool>(); }
    else if constexpr (std::is_integral_v<T>) { if (j.is_number()) obj = (T)j.as_int(); }
    else if constexpr (std::is_floating_point_v<T>) { if (j.is_number()) obj = (T)j.as_double(); }
    else if constexpr (std::is_same_v<T, std::string>) { if (j.is_string()) obj = j.get<std::string>(); }
    else if constexpr (is_vector_like<T>::value) {
      if (!j.is_array()) return;
      auto& a = j.get<rjson_t::array_t>();
      obj.clear(); obj.reserve(a.size());
      for (auto& e : a) {
        typename T::value_type v;
        from_rjson(v, e);
        obj.push_back(std::move(v));
      }
    }
    else if constexpr (std::meta::is_class_type(^^T)) {
      if (!j.is_object()) return;
      fan::refl::for_each_member<T>([&]<std::meta::info m>{
        constexpr auto n = std::meta::identifier_of(m);
        if (j.contains(n)) from_rjson(obj.[:m:], j[n]);
      });
    }
  }

  template<typename T>
  std::string to_rjson_string(const T& obj, int indent = 0) {
    return rjson_dump(to_rjson(obj), indent);
  }

  template<typename T>
  T from_rjson_string(std::string_view s) {
    T obj{};
    from_rjson(obj, rjson_parse(s));
    return obj;
  }

  inline std::ostream& operator<<(std::ostream& os, const rjson_t& j) {
    return os << rjson_dump(j, 2);
  }
}