module;

export module fan.formatter;

import std;

export namespace fan {

  template<typename>
  struct is_std_vector : std::false_type {};

  template<typename T, typename A>
  struct is_std_vector<std::vector<T, A>> : std::true_type {};

  template<typename T>
  concept streamable = requires(std::ostream& os, T v) {
    os << v;
  };

  template <typename T>
  std::string to_str(const T& v) {
    if constexpr (std::is_same_v<std::decay_t<T>, std::string>)               return v;
    else if constexpr (std::is_same_v<std::decay_t<T>, std::string_view>)     return std::string(v);
    else if constexpr (std::is_same_v<std::decay_t<T>, const char*>)          return v ? v : "";
    else if constexpr (std::is_same_v<std::decay_t<T>, char>)                 return std::string(1, v);
    else if constexpr (std::is_arithmetic_v<std::decay_t<T>>)                 return std::to_string(v);
    else if constexpr (std::is_same_v<std::decay_t<T>, const unsigned char*>) return v ? reinterpret_cast<const char*>(v) : "";
    else if constexpr (std::is_same_v<std::decay_t<T>, unsigned char*>)       return v ? reinterpret_cast<const char*>(v) : "";
    else if constexpr (std::is_convertible_v<T, std::string_view>)            return std::string(static_cast<std::string_view>(v));
    else if constexpr (std::is_convertible_v<T, std::string>)                 return static_cast<std::string>(v);
    else if constexpr (streamable<T>) { std::ostringstream ss; ss << v; return ss.str(); }
    else if constexpr (is_std_vector<T>::value) { 
      std::string s; for (int i = 0; i < v.size(); ++i) s += std::to_string(v[i]) + (i + 1 < v.size() ? ", " : "");
                                                                              return s;
    }
    else                                                                      return v.to_string();
  }

  template <typename ...Args>
  std::string format_join(const char* sep, const Args&... args) {
    std::string result;
    int idx = 0;
    ((result += fan::to_str(args), result += (++idx == (int)sizeof...(args) ? "" : sep)), ...);
    return result;
  }

  template <typename ...Args> std::string format_args(const Args&... args) { return format_join(" ", args...); }
  template <typename ...Args> std::string format_args_raw(const Args&... args) { return format_join("", args...); }
  template <typename ...Args> std::string format_args_comma(const Args&... args) { return format_join(", ", args...); }
  template <typename ...Args> std::string format_args_no_space(const Args&... args) { return format_join("", args...); }
  template <typename ...Args> std::string format_args_with_space(const Args&... args) { return format_join(" ", args...); }
  template <typename ...Args> std::string format_error_args(const Args&... args) { return format_join(" ", args...); }

  template<typename T>
  auto convert_uint8(T value) {
    if constexpr (std::is_same_v<T, std::uint8_t>) return static_cast<int>(value);
    else if constexpr (std::is_same_v<T, std::string_view>) return std::string(value);
    else return value;
  }

  template <typename ...Args>
  std::string format_args_n8(const Args&... args) {
    std::string result;
    int idx = 0;
    ((result += fan::to_str(convert_uint8(args)) + (++idx == (int)sizeof...(args) ? "" : ", ")), ...);
    return result;
  }
}