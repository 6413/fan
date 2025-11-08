#pragma pack(push, 1)
#define make_for_all(todo) \
	vec_t ret = 0; for (access_type_t i = 0; i < size(); ++i) { todo; } return ret
#define make_for_all_test1_noret(todo) for (access_type_t i = 0; i < size() && i < test0.size(); ++i) { todo; }
#define make_for_all_test1(todo) \
	vec_t ret = 0; for (access_type_t i = 0; i < size() && i < test0.size(); ++i) { todo; } return ret
#define make_for_all_test2(todo) \
	vec_t ret = 0; for (access_type_t i = 0; i < size() && i < test0.size() && i < test1.size(); ++i) { todo; } return ret

#define make_operator_const(arithmetic) \
template <typename T> \
requires (!std::is_arithmetic_v<T>) \
constexpr vec_t operator arithmetic(const T& test0) const \
{ \
	make_for_all_test1(ret[i] = (*this)[i] arithmetic test0[i]); \
} \
\
template <typename T> \
requires (std::is_arithmetic_v<T>)\
constexpr vec_t operator arithmetic(T v0) const \
{ \
	make_for_all(ret[i] = (*this)[i] arithmetic v0); \
}
#define make_operator_assign(arithmetic) \
template <typename T> \
requires (!std::is_arithmetic_v<T>) \
constexpr vec_t& operator CONCAT(arithmetic,=) (const T& test0) \
{ \
	make_for_all_test1_noret((*this)[i] CONCAT(arithmetic,=) test0[i]); \
  return *this; \
} \
\
template <typename T> \
requires (std::is_arithmetic_v<T>)\
constexpr vec_t operator CONCAT(arithmetic,=)(T v0) \
{ \
	make_for_all((*this)[i] CONCAT(arithmetic,=) v0); \
}

using value_type = value_type_t;

static constexpr access_type_t size() { return vec_n; }

constexpr vec_t() = default;
template <typename T>
requires std::is_arithmetic_v<T>
constexpr vec_t(T single_init) { for (access_type_t i = 0; i < vec_n; ++i) operator[](i) = single_init; } 
template<typename... Args>
requires ((std::is_arithmetic_v<std::remove_reference_t<Args>> && ...) &&
          sizeof...(Args) == size())
constexpr vec_t(Args&&...args) {
  access_type_t i = 0;
  ((this->operator[](i++) = args), ...);
}
template<typename... Args>
requires(
  (std::is_same_v<value_type_t, std::remove_reference_t<Args>> && ...) &&
  (!std::is_arithmetic_v<std::remove_reference_t<Args>> && ...) &&
  sizeof...(Args) == size()
  )
constexpr vec_t(Args&&...args) {
  access_type_t i = 0;
  ((this->operator[](i++) = args), ...);
}
#ifndef fan_vector_array
  template <typename T>
  constexpr vec_t(const vec_t<T>& test0) { for (int i = 0; i < size(); ++i) operator[](i) = test0[i]; } 
#else
  template <typename T>
  constexpr vec_t(const vec_t<vec_n, T>& test0) { for (int i = 0; i < size(); ++i) operator[](i) = test0[i]; } 
#endif

#define make_operators(arithmetic) \
  make_operator_const(arithmetic); \
  make_operator_assign(arithmetic)

constexpr vec_t operator-() const { make_for_all(ret[i] = -(*this)[i]); }
constexpr vec_t operator+() const { make_for_all(ret[i] = +(*this)[i]); }
make_operators(-);  //make_operator_comparison(==);
make_operators(+);  //make_operator_comparison(!=);
make_operators(*);
make_operators(/);  
make_operators(%);  

template <typename T>
  requires (!std::is_arithmetic_v<T>)
constexpr bool operator ==(const T& rhs) const {
  for (access_type_t i = 0; i < size() && i < rhs.size(); ++i) {
    if ((*this)[i] != rhs[i]) {
      return false;
    }
  }
  
  return true;
}

template <typename T>
  requires (std::is_arithmetic_v<T>)
constexpr bool operator ==(const T& rhs) const {
  return (*this)[0] == rhs;
}

template <typename T>
  requires (!std::is_arithmetic_v<T>)
constexpr bool operator !=(const T& rhs) const {
  return !(*this == rhs);
}

template <typename T>
  requires (std::is_arithmetic_v<T>)
constexpr bool operator !=(const T& rhs) const {
  return !(*this == rhs);
}

explicit constexpr operator bool() const {
  return (*this != value_type_t(0));
}
                    
#define __FAN_SWITCH_IDX(x, idx) case size() - (idx + 1): return x

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
    // force crash with stackoverflow or gives error if idx is knowable at compiletime
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
    // force crash with stackoverflow or gives error if idx is knowable at compiletime
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
constexpr auto end() const { return begin() + size(); }
constexpr auto data() const { return begin(); }

constexpr auto begin() { return &operator[](0); }
constexpr auto end() { return begin() + size(); }
constexpr auto data() { return begin(); }

constexpr auto plus() const { return std::accumulate(begin(), end(), value_type_t{}, std::plus<value_type_t>()); }
constexpr auto minus() const { return std::accumulate(begin(), end(), value_type_t{}, std::minus<value_type_t>()); }
constexpr auto multiply() const { return std::accumulate(begin(), end(), value_type_t{1}, std::multiplies<value_type_t>()); }
constexpr auto sign() const { make_for_all(ret[i] = fan::math::sgn((*this)[i])); }
constexpr auto floor() const { make_for_all(ret[i] = std::floor((*this)[i])); }
constexpr auto floor(auto value) const { make_for_all(ret[i] = std::floor((*this)[i] / value)); }
constexpr auto ceil() const { make_for_all(ret[i] = std::ceil((*this)[i])); }
constexpr auto round() const { make_for_all(ret[i] = std::round((*this)[i])); }
constexpr auto abs() const { make_for_all(ret[i] = std::abs((*this)[i])); }
constexpr auto min() const { return *std::min_element(begin(), end()); }
template <typename T>
requires(std::is_arithmetic_v<T>)
constexpr auto min(const T& test0) const { make_for_all(ret[i] = std::min((*this)[i], test0)); }
constexpr auto min(const auto& test0) const { make_for_all_test1(ret[i] = std::min((*this)[i], test0[i])); }
constexpr auto max() const { return *std::max_element(begin(), end()); }
template <typename T>
requires(std::is_arithmetic_v<T>)
constexpr auto max(const T& test0) const { make_for_all(ret[i] = std::max((*this)[i], test0)); }
constexpr auto max(const auto& test0) const { make_for_all_test1(ret[i] = std::max((*this)[i], test0[i])); }
constexpr auto clamp(value_type mi, value_type ma) const { make_for_all(ret[i] = std::clamp((*this)[i], mi, ma)); }
constexpr auto clamp(const vec_t& test0, const vec_t& test1) const { make_for_all_test2(ret[i] = std::clamp((*this)[i], test0[i], test1[i])); }
constexpr auto reflect(const vec_t& normal) { return *this - normal * 2 * dot(normal); }
constexpr auto tangential_reflect(const vec_t& normal) { return *this - normal * dot(normal); }

// gives number furthest away from 0
constexpr auto abs_max() const { 
  auto v0 = min();
  auto v1 = max();
  return std::abs(v0) < std::abs(v1) ? v1 : v0;
}

constexpr auto dot(const auto& test0) const { return fan::math::dot(*this, test0); }
template <typename... Ts>
constexpr auto cross(Ts... args) const { return fan::math::cross(*this, args...); }
constexpr auto length() const { return sqrt(dot(*this)); }
constexpr auto normalized() const { auto l = length(); if (l == 0) return vec_t(0); make_for_all(ret[i] = (*this)[i] / l); }
template <typename T>
requires(!std::is_arithmetic_v<T>)
constexpr auto distance(const T& other) const { return (*this - other).length(); }

constexpr vec_t square_normalize() const { 
  auto max_val = abs().max();
  if (max_val == 0) {
    return vec_t{};
  }
  return *this / max_val; 
}

constexpr vec_t rotate(value_type_t angle) const {
  if constexpr (size() >= 2) {
    vec_t ret = *this;
    value_type_t cos_angle = std::cos(angle);
    value_type_t sin_angle = std::sin(angle);
    value_type_t x = (*this)[0];
    value_type_t y = (*this)[1];
    ret[0] = x * cos_angle - y * sin_angle;
    ret[1] = x * sin_angle + y * cos_angle;
    return ret;
  }
  else {
    return *this;
  }
}

constexpr vec_t snap_to_grid(f32_t grid_size) const {
  return (*this / grid_size + 0.5f).floor() * grid_size;
}

constexpr vec_t snap_to_grid(const vec_t& grid_size) const {
  return (*this / grid_size + 0.5f).floor() * grid_size;
}

template <typename T>
static auto val_to_string(const T a_value, const int n = 2) {
  std::ostringstream out;
  out.precision(n);
  out << std::fixed << a_value;
  return out.str();
}

std::string to_string(int precision = 4) const {
  std::string out("{");
  for (access_type_t i = 0; i < size() - 1; ++i) { out += val_to_string((*this)[i], precision) + ", "; }
  if constexpr (size()) {
    out += val_to_string((*this)[size() - 1], precision);
  }
  out += '}';
  return out;
}

void from_string(const std::string& str) {
  std::string s = str;
  // remove braces and spaces
  s.erase(std::remove_if(s.begin(), s.end(),
    [](char c) { return c == '{' || c == '}' || c == ' '; }), s.end());

  std::stringstream ss(s);
  std::string item;
  for (access_type_t i = 0; i < size(); ++i) {
    if (!std::getline(ss, item, ',')) {
      (*this)[i] = value_type_t{};
    }
    else {
      (*this)[i] = static_cast<value_type_t>(std::stof(item));
    }
  }
}

static vec_t parse(const std::string& str) {
  vec_t out{};
  std::string s = str;
  s.erase(std::remove_if(s.begin(), s.end(),
    [](char c) { return c == '{' || c == '}' || c == ' '; }), s.end());

  std::stringstream ss(s);
  std::string item;
  for (access_type_t i = 0; i < out.size(); ++i) {
    if (!std::getline(ss, item, ',')) {
      out[i] = value_type_t{}; // default
    }
    else {
      out[i] = static_cast<value_type_t>(std::stof(item));
    }
  }

  return out;
}

bool is_near(const vec_t& test0, value_type_t epsilon) const { 
  make_for_all_test1_noret(if (!fan::math::is_near((*this)[i], test0[i], epsilon)) return false;);
  return true;
} 

friend std::ostream& operator<<(std::ostream& os, const vec_t& test0) { os << test0.to_string(); return os; }
operator std::string const() const {
  return to_string();
}

#if !defined(fan_vector_array) && vec_n
value_type_t fan_coordinate(vec_n);
#elif defined(fan_vector_array)
  value_type_t fan_coordinate(vec_n){};
#endif

#undef make_operator_comparison
#undef make_operator_const
#undef make_operators
#undef make_operator_assign
#undef vec_n
#undef vec_t
#undef make_for_all
#undef make_for_all_test1
#undef make_for_all_test2
#undef __FAN_PTR_EACH
#undef fan_vector_array

#pragma pack(pop)