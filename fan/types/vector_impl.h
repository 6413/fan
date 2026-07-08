using value_type = value_type_t;

constexpr vec_t() = default;

template <typename U> requires std::is_arithmetic_v<U>
constexpr vec_t(U single_init) { for (access_type_t i = 0; i < vec_n; ++i) operator[](i) = single_init; } 

template<typename... Args>
requires ((std::is_arithmetic_v<std::remove_reference_t<Args>> && ...) && sizeof...(Args) == vec_n)
constexpr vec_t(Args&&...args) {
  access_type_t i = 0;
  ((this->operator[](i++) = static_cast<value_type_t>(args)), ...);
}

template <typename U> requires std::is_convertible_v<U, value_type_t>
constexpr vec_t(std::initializer_list<U> init) {
  access_type_t i = 0;
  for (auto&& e : init) {
    if (i >= vec_n) break;
    (*this)[i++] = static_cast<value_type_t>(e);
  }
  for (; i < vec_n; ++i) (*this)[i] = value_type_t{};
}

template<typename... Args>
requires((std::is_same_v<value_type_t, std::remove_reference_t<Args>> && ...) && (!std::is_arithmetic_v<std::remove_reference_t<Args>> && ...) && sizeof...(Args) == vec_n)
constexpr vec_t(Args&&...args) {
  access_type_t i = 0;
  ((this->operator[](i++) = args), ...);
}

#ifndef fan_vector_array
  template <typename U>
  constexpr vec_t(const vec_t<U>& test0) { for (access_type_t i = 0; i < vec_n; ++i) operator[](i) = test0[i]; } 
#else
  template <typename U>
  constexpr vec_t(const vec_t<vec_n, U>& test0) { for (access_type_t i = 0; i < vec_n; ++i) operator[](i) = test0[i]; } 
#endif

vec_t(const std::string& str) {
  *this = fan::vec_base<vec_t, vec_n, value_type_t>::from_string(str);
}

#define __FAN_SWITCH_IDX(x, idx) case idx: return x

constexpr value_type_t& operator[](access_type_t idx) { 
#ifndef fan_vector_array
  switch (idx) {
    #if vec_n
    __FAN__FOREACH(__FAN_SWITCH_IDX, fan_coordinate(vec_n);)
    #else
    default: return *(value_type_t*)nullptr;
    #endif
  }
#else
  return fan_coordinate(idx);
#endif
  return operator[](idx);
}
constexpr const value_type_t& operator[](access_type_t idx) const {
#ifndef fan_vector_array
  switch (idx) { 
    #if vec_n
    __FAN__FOREACH(__FAN_SWITCH_IDX, fan_coordinate(vec_n);)
    #else
    default: return *(value_type_t*)nullptr;
    #endif
  }
#else
  return fan_coordinate(idx);
#endif
  return operator[](idx);
}
#undef __FAN_SWITCH_IDX

constexpr auto begin() const { 
  return
#if !defined(fan_vector_array) && vec_n == 0 
    nullptr;
#else
#ifndef fan_vector_array
    &x;
#else
    &fan_coordinate(0);
#endif
#endif
}
constexpr auto end() const { return begin() + vec_n; }
constexpr auto data() const { return begin(); }

constexpr auto begin() { return &operator[](0); }
constexpr auto end() { return begin() + vec_n; }
constexpr auto data() { return begin(); }

#ifndef fan_vector_array
constexpr vec_t<int> grid_cell(value_type_t grid_size) const {
  return vec_t<int>((*this / grid_size).floor());
}
constexpr vec_t<int> grid_cell(const vec_t& grid_size) const {
  return vec_t<int>((*this / grid_size).floor());
}
#endif

#if !defined(fan_vector_array) && vec_n
value_type_t fan_coordinate(vec_n);
#elif defined(fan_vector_array)
  value_type_t fan_coordinate(vec_n){};
#endif

#undef vec_n
#undef vec_t
#undef fan_vector_array