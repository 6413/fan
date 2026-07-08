module fan.types.vector;

import std;

namespace fan {
  template <typename T>
  static auto val_to_string(const T a_value, const int n) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*f", n, (double)a_value);
    return std::string(buf);
  }

  template <typename Derived, access_type_t N, typename T>
  std::string vec_base<Derived, N, T>::to_string(int precision) const {
    std::string out("{");
    for (access_type_t i = 0; i < N - 1; ++i) { out += val_to_string(derived()[i], precision) + ", "; }
    if constexpr (N > 0) {
      out += val_to_string(derived()[N - 1], precision);
    }
    out += '}';
    return out;
  }

  template <typename Derived, access_type_t N, typename T>
  Derived vec_base<Derived, N, T>::from_string(const std::string& str) {
    Derived vec{};
    std::string s = str;
    s.erase(std::remove_if(s.begin(), s.end(),
      [](char c) { return c == '{' || c == '}' || c == ' '; }), s.end());

    access_type_t i = 0;
    std::size_t pos = 0;
    while (i < N) {
      std::size_t comma = s.find(',', pos);
      std::string item = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
      vec[i] = item.empty() ? T{} : static_cast<T>(std::stof(item));
      ++i;
      if (comma == std::string::npos) break;
      pos = comma + 1;
    }
    return vec;
  }

  #define INSTANTIATE_VEC(TYPE) \
    template struct vec_base<vec1_wrap_t<TYPE>, 1, TYPE>; \
    template struct vec_base<vec2_wrap_t<TYPE>, 2, TYPE>; \
    template struct vec_base<vec3_wrap_t<TYPE>, 3, TYPE>; \
    template struct vec_base<vec4_wrap_t<TYPE>, 4, TYPE>;

  INSTANTIATE_VEC(bool)
  INSTANTIATE_VEC(std::int8_t)
  INSTANTIATE_VEC(int)
  INSTANTIATE_VEC(long long)
  INSTANTIATE_VEC(std::uint32_t)
  INSTANTIATE_VEC(unsigned long long)
  INSTANTIATE_VEC(f32_t)
  INSTANTIATE_VEC(f64_t)
}