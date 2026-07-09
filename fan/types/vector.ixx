module;

#include <fan/utility.h>

export module fan.types.vector;
import std;

import fan.types;
import fan.math;

#define fan_coordinate_letters0
#define fan_coordinate_letters1 x
#define fan_coordinate_letters2 x, y
#define fan_coordinate_letters3 x, y, z
#define fan_coordinate_letters4 x, y, z, w

#define fan_coordinate(x) CONCAT(fan_coordinate_letters, x)

export namespace fan {
  using access_type_t = std::uint16_t;

  template <typename T> struct vec2_wrap_t;
  template <typename T> struct vec3_wrap_t;
  template <typename T> struct vec4_wrap_t;
  template <int N, typename T> struct vec_wrap_t;

  template <typename Derived, access_type_t N, typename T>
  struct vec_base {
    using value_type = T;
    static constexpr access_type_t size() { return N; }
    constexpr Derived& derived() { return *static_cast<Derived*>(this); }
    constexpr const Derived& derived() const { return *static_cast<const Derived*>(this); }

    template <access_type_t... Indices>
    constexpr auto swizzle() const {
      if constexpr (sizeof...(Indices) == 2) return vec2_wrap_t<T>(derived()[Indices]...);
      else if constexpr (sizeof...(Indices) == 3) return vec3_wrap_t<T>(derived()[Indices]...);
      else if constexpr (sizeof...(Indices) == 4) return vec4_wrap_t<T>(derived()[Indices]...);
    }

    constexpr vec2_wrap_t<T> xy() const { return swizzle<0, 1>(); }
    constexpr vec2_wrap_t<T> yx() const { return swizzle<1, 0>(); }
    constexpr vec3_wrap_t<T> xyz() const { return swizzle<0, 1, 2>(); }
    constexpr vec3_wrap_t<T> rgb() const { return swizzle<0, 1, 2>(); }
    constexpr vec4_wrap_t<T> rgba() const { return swizzle<0, 1, 2, 3>(); }
    constexpr vec4_wrap_t<T> xyxy() const { return swizzle<0, 1, 0, 1>(); }
    constexpr vec4_wrap_t<T> xxyy() const { return swizzle<0, 0, 1, 1>(); }
    constexpr vec4_wrap_t<T> ywyw() const { return swizzle<1, 3, 1, 3>(); }

    constexpr Derived operator-() const { Derived r{}; for (access_type_t i=0; i<N; ++i) r[i] = -derived()[i]; return r; }
    constexpr Derived operator+() const { Derived r{}; for (access_type_t i=0; i<N; ++i) r[i] = +derived()[i]; return r; }

    #define VEC_BIN_OP(op) \
    template <typename U> requires (!std::is_arithmetic_v<U>) \
    constexpr Derived operator op(const U& rhs) const { Derived r{}; for (access_type_t i=0; i<N && i<rhs.size(); ++i) r[i] = derived()[i] op rhs[i]; return r; } \
    template <typename U> requires std::is_arithmetic_v<U> \
    constexpr Derived operator op(U v) const { Derived r{}; for (access_type_t i=0; i<N; ++i) r[i] = derived()[i] op (T)v; return r; } \
    template <typename U> requires (!std::is_arithmetic_v<U>) \
    constexpr Derived& operator op##=(const U& rhs) { for (access_type_t i=0; i<N && i<rhs.size(); ++i) derived()[i] op##= rhs[i]; return derived(); } \
    template <typename U> requires std::is_arithmetic_v<U> \
    constexpr Derived& operator op##=(U v) { for (access_type_t i=0; i<N; ++i) derived()[i] op##= (T)v; return derived(); } \
    template <typename U> requires std::is_arithmetic_v<U> \
    friend constexpr Derived operator op(U lhs, const Derived& rhs) { Derived r{}; for (access_type_t i=0; i<N; ++i) r[i] = (T)lhs op rhs[i]; return r; }

    VEC_BIN_OP(+) VEC_BIN_OP(-) VEC_BIN_OP(*) VEC_BIN_OP(/)
    #undef VEC_BIN_OP

    #define VEC_REL_OP(op) \
    template <typename U> requires (!std::is_arithmetic_v<U>) \
    constexpr bool operator op(const U& rhs) const { for (access_type_t i=0; i<N && i<rhs.size(); ++i) { if (!(derived()[i] op rhs[i])) return false; } return true; } \
    template <typename U> requires std::is_arithmetic_v<U> \
    constexpr bool operator op(U v) const { for (access_type_t i=0; i<N; ++i) { if (!(derived()[i] op (T)v)) return false; } return true; }

    VEC_REL_OP(==) VEC_REL_OP(<) VEC_REL_OP(>) VEC_REL_OP(<=) VEC_REL_OP(>=)
    #undef VEC_REL_OP

    template <typename U> requires (!std::is_arithmetic_v<U>)
    constexpr bool operator!=(const U& rhs) const { for (access_type_t i=0; i<N && i<rhs.size(); ++i) { if (derived()[i] != rhs[i]) return true; } return false; }
    template <typename U> requires std::is_arithmetic_v<U>
    constexpr bool operator!=(U v) const { for (access_type_t i=0; i<N; ++i) { if (derived()[i] != (T)v) return true; } return false; }

    explicit constexpr operator bool() const {
      for (access_type_t i = 0; i < N; ++i) { if (derived()[i] != T(0)) return true; } return false;
    }

    constexpr T sum() const { T r{}; for (access_type_t i=0; i<N; ++i) r += derived()[i]; return r; }
    constexpr T multiply() const { T r{1}; for (access_type_t i=0; i<N; ++i) r *= derived()[i]; return r; }
    
    constexpr Derived sign() const                { Derived r{}; for (access_type_t i=0; i<N; ++i) r[i] = fan::math::sgn(derived()[i]); return r; }
    constexpr Derived floor() const requires (!std::same_as<T, bool>) { Derived r{}; for (access_type_t i=0; i<N; ++i) r[i] = std::floor(derived()[i]); return r; }
    constexpr Derived floor_div(auto value) const requires (!std::same_as<T, bool>) { Derived r{}; for (access_type_t i=0; i<N; ++i) r[i] = std::floor(derived()[i] / value); return r; }
    constexpr Derived ceil() const requires (!std::same_as<T, bool>) { Derived r{}; for (access_type_t i=0; i<N; ++i) r[i] = std::ceil(derived()[i]); return r; }
    constexpr Derived round() const requires (!std::same_as<T, bool>) { Derived r{}; for (access_type_t i=0; i<N; ++i) r[i] = std::round(derived()[i]); return r; }
    constexpr Derived abs() const                 {
      Derived r {};
      for (access_type_t i = 0; i < N; ++i) {
        if constexpr (std::is_unsigned_v<T>) {
          r[i] = derived()[i];
        }
        else {
          r[i] = std::abs(derived()[i]);
        }
      }
      return r;
    }

    constexpr T min() const { return *std::min_element(derived().begin(), derived().end()); }
    template <typename U> requires std::is_arithmetic_v<U>
    constexpr Derived min(U v) const { Derived r{}; for (access_type_t i=0; i<N; ++i) r[i] = std::min(derived()[i], (T)v); return r; }
    constexpr Derived min(const auto& v) const { Derived r{}; for (access_type_t i=0; i<N && i<v.size(); ++i) r[i] = std::min(derived()[i], (T)v[i]); return r; }

    constexpr T max() const { return *std::max_element(derived().begin(), derived().end()); }
    template <typename U> requires std::is_arithmetic_v<U>
    constexpr Derived max(U v) const { Derived r{}; for (access_type_t i=0; i<N; ++i) r[i] = std::max(derived()[i], (T)v); return r; }
    constexpr Derived max(const auto& v) const { Derived r{}; for (access_type_t i=0; i<N && i<v.size(); ++i) r[i] = std::max(derived()[i], (T)v[i]); return r; }

    constexpr Derived clamp(T mi, T ma) const { Derived r{}; for (access_type_t i=0; i<N; ++i) r[i] = fan::math::clamp(derived()[i], mi, ma); return r; }
    constexpr Derived clamp(const Derived& mi, const Derived& ma) const { Derived r{}; for (access_type_t i=0; i<N; ++i) r[i] = fan::math::clamp(derived()[i], mi[i], ma[i]); return r; }

    constexpr Derived reflect(const Derived& normal) const { return derived() - normal * 2 * dot(normal); }
    constexpr Derived reflect_tangent(const Derived& normal) const { return derived() - normal * dot(normal); }

    static constexpr T vmax() { return std::numeric_limits<T>::max(); }

    constexpr T max_abs() const {
      auto v0 = min();
      auto v1 = max();

      if constexpr (std::is_unsigned_v<T>) {
        return v1;
      }
      else {
        return (v0 < 0 ? -v0 : v0) < (v1 < 0 ? -v1 : v1) ? v1 : v0;
      }
    }

    constexpr T dot(const auto& v) const { return fan::math::dot(derived(), v); }
    template <typename... Ts> constexpr Derived cross(Ts... args) const { return fan::math::cross(derived(), args...); }

    constexpr auto length() const { return std::sqrt((double)dot(derived())); }
    constexpr T length_squared() const { return dot(derived()); }
    
    constexpr Derived normalize() const { auto l = length(); if (l == 0) return Derived(0); Derived r{}; for (access_type_t i=0; i<N; ++i) r[i] = derived()[i] / l; return r; }
    template <typename U> requires (!std::is_arithmetic_v<U>)
    constexpr auto distance(const U& other) const { return (derived() - other).length(); }

    constexpr Derived square_normalize() const { auto max_val = abs().max(); if (max_val == 0) return Derived{}; return derived() / max_val; }

    constexpr Derived rotate(T angle) const {
      if constexpr (N >= 2) {
        Derived ret = derived();
        T cos_angle = std::cos(angle), sin_angle = std::sin(angle);
        T x = derived()[0], y = derived()[1];
        ret[0] = x * cos_angle - y * sin_angle;
        ret[1] = x * sin_angle + y * cos_angle;
        return ret;
      } else return derived();
    }

    constexpr Derived grid_round(T grid_size) const requires (!std::same_as<T, bool>) { return (derived() / grid_size).round() * grid_size; }
    constexpr Derived grid_round(const Derived& grid_size) const requires (!std::same_as<T, bool>) { return (derived() / grid_size).round() * grid_size; }
    
    constexpr Derived grid_floor(T grid_size, T offset = 0) const
    requires (!std::same_as<T, bool>) {
      return (derived() / grid_size).floor() * grid_size + offset;
    }

    constexpr Derived grid_floor(const Derived& grid_size, const Derived& offset = Derived(0)) const
    requires (!std::same_as<T, bool>) {
      return (derived() / grid_size).floor() * grid_size + offset;
    }

    constexpr Derived fmod(T divisor) const requires (!std::same_as<T, bool>) { Derived r{}; for(access_type_t i=0; i<N; ++i) r[i] = std::fmod(derived()[i], divisor); return r; }
    constexpr Derived fmod(const Derived& v) const requires (!std::same_as<T, bool>) { Derived r{}; for(access_type_t i=0; i<N; ++i) r[i] = std::fmod(derived()[i], v[i]); return r; }

    constexpr bool in_range(const Derived& lo, const Derived& hi) const {
      for (access_type_t i=0; i<N; ++i) if (derived()[i] < lo[i] || derived()[i] > hi[i]) return false;
      return true;
    }

    bool is_near(const Derived& v, T epsilon) const {
      for (access_type_t i=0; i<N; ++i) if (!fan::math::is_near(derived()[i], v[i], epsilon)) return false;
      return true;
    }

    constexpr Derived approach(const Derived& target, T step) const {
      if (step <= T(0)) return derived();
      Derived d = target - derived();
      auto l2 = d.length_squared();
      if (l2 <= step * step) return target;
      return derived() + d.normalize() * step;
    }

    std::string to_string(int precision = 4) const;
    explicit operator std::string() const { return to_string(); }
    static Derived from_string(const std::string& str);
    friend std::ostream& operator<<(std::ostream& os, const Derived& v) { os << (std::string)v; return os; }

    inline void iterate_row(access_type_t y, access_type_t x0, access_type_t x1, auto&& fn) const {
      for (access_type_t xx = x0; xx <= x1; ++xx) fn(xx, y);
    }
  };

  struct vec_lexi_comp {
    template<typename vec_T>
    constexpr bool operator()(const vec_T& a, const vec_T& b) const {
      auto min_size = a.size() < b.size() ? a.size() : b.size();
      for (decltype(min_size) i = 0; i < min_size; ++i) {
        if (a[i] < b[i]) return true;
        if (a[i] > b[i]) return false;
      }
      return a.size() < b.size();
    }
  };

  template <typename vec_t, typename value_type_t>
  struct vec_ref2 {
    value_type_t& a; value_type_t& b;
    constexpr operator vec_t() const { return {a, b}; }
    constexpr vec_ref2& operator=(const vec_t& v) { a = v.x; b = v.y; return *this; }
    constexpr vec_t operator+(const vec_ref2& v) const { return vec_t(*this) + vec_t(v); }
    constexpr vec_t operator+(const vec_t& v) const { return vec_t(*this) + v; }
    constexpr vec_t operator+(value_type_t v) const { return vec_t(*this) + v; }
    constexpr vec_t operator-(const vec_ref2& v) const { return vec_t(*this) - vec_t(v); }
    constexpr vec_t operator-(const vec_t& v) const { return vec_t(*this) - v; }
    constexpr vec_t operator-(value_type_t v) const { return vec_t(*this) - v; }
    constexpr vec_t operator*(const vec_ref2& v) const { return vec_t(*this) * vec_t(v); }
    constexpr vec_t operator*(const vec_t& v) const { return vec_t(*this) * v; }
    constexpr vec_t operator*(value_type_t v) const { return vec_t(*this) * v; }
    constexpr vec_t operator/(const vec_ref2& v) const { return vec_t(*this) / vec_t(v); }
    constexpr vec_t operator/(const vec_t& v) const { return vec_t(*this) / v; }
    constexpr vec_t operator/(value_type_t v) const { return vec_t(*this) / v; }
  };

  template <typename vec_t, typename value_type_t>
  struct vec_ref3 {
    value_type_t& a; value_type_t& b; value_type_t& c;
    constexpr operator vec_t() const { return {a, b, c}; }
    constexpr vec_ref3& operator=(const vec_t& v) { a = v.x; b = v.y; c = v.z; return *this; }
    constexpr vec_t operator+(const vec_ref3& v) const { return vec_t(*this) + vec_t(v); }
    constexpr vec_t operator+(const vec_t& v) const { return vec_t(*this) + v; }
    constexpr vec_t operator+(value_type_t v) const { return vec_t(*this) + v; }
    constexpr vec_t operator-(const vec_ref3& v) const { return vec_t(*this) - vec_t(v); }
    constexpr vec_t operator-(const vec_t& v) const { return vec_t(*this) - v; }
    constexpr vec_t operator-(value_type_t v) const { return vec_t(*this) - v; }
    constexpr vec_t operator*(const vec_ref3& v) const { return vec_t(*this) * vec_t(v); }
    constexpr vec_t operator*(const vec_t& v) const { return vec_t(*this) * v; }
    constexpr vec_t operator*(value_type_t v) const { return vec_t(*this) * v; }
    constexpr vec_t operator/(const vec_ref3& v) const { return vec_t(*this) / vec_t(v); }
    constexpr vec_t operator/(const vec_t& v) const { return vec_t(*this) / v; }
    constexpr vec_t operator/(value_type_t v) const { return vec_t(*this) / v; }
  };

  template <typename vec_t, typename value_type_t>
  struct vec_ref4 {
    value_type_t& a; value_type_t& b; value_type_t& c; value_type_t& d;
    constexpr operator vec_t() const { return {a, b, c, d}; }
    constexpr vec_ref4& operator=(const vec_t& v) { a = v.x; b = v.y; c = v.z; d = v.w; return *this; }
    constexpr vec_t operator+(const vec_ref4& v) const { return vec_t(*this) + vec_t(v); }
    constexpr vec_t operator+(const vec_t& v) const { return vec_t(*this) + v; }
    constexpr vec_t operator+(value_type_t v) const { return vec_t(*this) + v; }
    constexpr vec_t operator-(const vec_ref4& v) const { return vec_t(*this) - vec_t(v); }
    constexpr vec_t operator-(const vec_t& v) const { return vec_t(*this) - v; }
    constexpr vec_t operator-(value_type_t v) const { return vec_t(*this) - v; }
    constexpr vec_t operator*(const vec_ref4& v) const { return vec_t(*this) * vec_t(v); }
    constexpr vec_t operator*(const vec_t& v) const { return vec_t(*this) * v; }
    constexpr vec_t operator*(value_type_t v) const { return vec_t(*this) * v; }
    constexpr vec_t operator/(const vec_ref4& v) const { return vec_t(*this) / vec_t(v); }
    constexpr vec_t operator/(const vec_t& v) const { return vec_t(*this) / v; }
    constexpr vec_t operator/(value_type_t v) const { return vec_t(*this) / v; }
  };

  #pragma pack(push, 1)

  template <typename value_type_t>
  struct vec0_wrap_t : vec_base<vec0_wrap_t<value_type_t>, 0, value_type_t> {
    #define vec_t vec0_wrap_t
    #define vec_n 0
    #include "vector_impl.h"
  };

  template <typename value_type_t>
  struct vec1_wrap_t : vec_base<vec1_wrap_t<value_type_t>, 1, value_type_t> {
    #define vec_t vec1_wrap_t
    #define vec_n 1
    #include "vector_impl.h"
  };

  template <typename value_type_t>
  struct vec2_wrap_t : vec_base<vec2_wrap_t<value_type_t>, 2, value_type_t> {
  #define vec_t vec2_wrap_t
  #define vec_n 2
  #include "vector_impl.h"

    template <typename T> constexpr vec2_wrap_t(const vec3_wrap_t<T>& test0) : vec2_wrap_t(test0.x, test0.y) { }

    constexpr auto copysign(const auto& test0) const { return vec2_wrap_t(fan::math::copysign(this->x, test0.x), fan::math::copysign(this->y, test0.y)); }

    template <typename Dest> requires (!requires { std::declval<Dest>().derived(); } && std::is_default_constructible_v<Dest> && requires(Dest d) { d.x; d.y; })
    constexpr operator Dest() const { Dest d{}; d.x = static_cast<std::remove_reference_t<decltype(d.x)>>(this->x); d.y = static_cast<std::remove_reference_t<decltype(d.y)>>(this->y); return d; }

    template <typename Src> requires (!requires { std::declval<Src>().derived(); } && requires(Src v) { v.x; v.y; })
    constexpr vec2_wrap_t(const Src& v) { this->x = static_cast<value_type_t>(v.x); this->y = static_cast<value_type_t>(v.y); }

    template <typename Dest> requires (!requires { std::declval<Dest>().derived(); } && std::is_default_constructible_v<Dest> && requires(Dest d) { d.width; d.height; })
    constexpr operator Dest() const { Dest d{}; d.width = static_cast<std::remove_reference_t<decltype(d.width)>>(this->x); d.height = static_cast<std::remove_reference_t<decltype(d.height)>>(this->y); return d; }

    template <typename Src> requires (!requires { std::declval<Src>().derived(); } && requires(Src v) { v.width; v.height; })
    constexpr vec2_wrap_t(const Src& v) { this->x = static_cast<value_type_t>(v.width); this->y = static_cast<value_type_t>(v.height); }

    constexpr auto csangle() const { return std::atan2(this->x, -this->y); }
    constexpr auto angle() const { return std::atan2(this->y, this->x); }
    constexpr value_type_t dir_angle() const {
      if (fan::math::abs(this->x) > fan::math::abs(this->y)) { return this->x > 0 ? 0 : fan::math::pi; }
      return this->y > 0 ? fan::math::half_pi : fan::math::pi * 1.5f;
    }
    constexpr value_type_t corner_angle(const vec2_wrap_t& to) const { return dir_angle() + (this->cross(to) > 0 ? 0 : fan::math::half_pi); }

    static vec2_wrap_t<value_type_t> from_angle(f32_t angle, f32_t length) { return vec2_wrap_t<value_type_t>(std::cos(angle), std::sin(angle)) * length; }

    template <typename T> bool is_collinear(const vec2_wrap_t<T>& a) { return a.x == this->x || a.y == this->y; }
    template <typename T> vec2_wrap_t<T> get_corner(const vec2_wrap_t<T>& a) { return {a.x, this->y}; }

    template <typename T> vec2_wrap_t<T> lerp(const vec2_wrap_t<T>& dst, T t) const { return { this->x + t * (dst.x - this->x), this->y + t * (dst.y - this->y) }; }

    constexpr vec2_wrap_t<value_type_t> perpendicular() const { return { -this->y, this->x }; }
    constexpr value_type_t cross(const vec2_wrap_t<value_type_t>& b) { return this->x * b.y - this->y * b.x; }

    constexpr vec2_wrap_t offset_x(value_type_t dx) const { return { this->x + dx, this->y }; }
    constexpr vec2_wrap_t offset_y(value_type_t dy) const { return { this->x, this->y + dy }; }
    constexpr vec2_wrap_t offset(value_type_t dx, value_type_t dy) const { return { this->x + dx, this->y + dy }; }

    constexpr void iterate_to(const vec2_wrap_t& max_, auto&& fn) const {
      for (int yy = this->y; yy <= max_.y; ++yy) {
        for (int xx = this->x; xx <= max_.x; ++xx) { fn(xx, yy); }
      }
    }
    constexpr void rect(const vec2_wrap_t& max_, auto&& fn) const {
      if (this->x <= max_.x && this->y <= max_.y) { iterate_to(max_, fn); }
    }
    constexpr void iterate_col(int col_x, int y0, int y1, auto&& fn) const {
      for (int yy = y0; yy <= y1; ++yy) { fn(col_x, yy); }
    }

    f32_t fit_scale(const vec2_wrap_t& container) const { return std::min(container.x / this->x, container.y / this->y); }
    vec2_wrap_t fit(const vec2_wrap_t& container) const { return *this * fit_scale(container); }
  };

  template <typename value_type_t>
  struct vec3_wrap_t : vec_base<vec3_wrap_t<value_type_t>, 3, value_type_t> {
  #define vec_t vec3_wrap_t
  #define vec_n 3
  #include "vector_impl.h"

    template <typename T> constexpr vec3_wrap_t(const vec2_wrap_t<T>& test0) : vec3_wrap_t(test0.x, test0.y, 0) { }
    template <typename T> constexpr vec3_wrap_t(const vec2_wrap_t<T>& test0, auto value) : vec3_wrap_t(test0.x, test0.y, value) { }
    template <typename T> constexpr vec3_wrap_t(const vec4_wrap_t<T>& test0) : vec3_wrap_t(test0.x, test0.y, test0.z) { }
    template <typename T> constexpr vec3_wrap_t(const vec3_wrap_t<T>& test0, auto value) : vec3_wrap_t(test0.x, test0.y, value) { }

  #if defined(FAN_3D)
    template <typename Dest> requires (!requires { std::declval<Dest>().derived(); } && std::is_default_constructible_v<Dest> && requires(Dest d) { d.x; d.y; d.z; })
    constexpr operator Dest() const { Dest d{}; d.x = static_cast<std::remove_reference_t<decltype(d.x)>>(this->x); d.y = static_cast<std::remove_reference_t<decltype(d.y)>>(this->y); d.z = static_cast<std::remove_reference_t<decltype(d.z)>>(this->z); return d; }

    template <typename Src> requires (!requires { std::declval<Src>().derived(); } && requires(Src v) { v.x; v.y; v.z; })
    constexpr vec3_wrap_t(const Src& v) { this->x = static_cast<value_type_t>(v.x); this->y = static_cast<value_type_t>(v.y); this->z = static_cast<value_type_t>(v.z); }
  #endif

    template <typename T>
    vec3_wrap_t& operator=(const vec2_wrap_t<T>& test0) { this->x = test0.x; this->y = test0.y; return *this; }

    template <typename T>
    constexpr auto cross(const fan::vec3_wrap_t<T>& vector) const { return fan::math::cross<vec3_wrap_t<T>>(*this, vector); }

    template <typename T>
    vec3_wrap_t<T> lerp(const vec3_wrap_t<T>& dst, T t) const { return { this->x + t * (dst.x - this->x), this->y + t * (dst.y - this->y), this->z + t * (dst.z - this->z) }; }

    constexpr vec3_wrap_t offset_x(value_type_t dx) const { return { this->x + dx, this->y, this->z }; }
    constexpr vec3_wrap_t offset_y(value_type_t dy) const { return { this->x, this->y + dy, this->z }; }
    constexpr vec3_wrap_t offset_z(value_type_t dz) const { return { this->x, this->y, this->z + dz }; }
    constexpr vec3_wrap_t offset(value_type_t dx, value_type_t dy, value_type_t dz) const { return { this->x + dx, this->y + dy, this->z + dz }; }
    constexpr vec3_wrap_t xz0() const { return {this->x, 0, this->z}; }
    constexpr vec3_wrap_t normalized_xz() const { vec3_wrap_t v = xz0(); return v.length_squared() > 0 ? v.normalize() : v; }
  };

  template <typename value_type_t>
  struct vec4_wrap_t : vec_base<vec4_wrap_t<value_type_t>, 4, value_type_t> {
  #define vec_t vec4_wrap_t
  #define vec_n 4
  #include "vector_impl.h"

    template <typename T> constexpr vec4_wrap_t(const vec2_wrap_t<T>& test0, auto third, auto fourth) : vec4_wrap_t(test0.x, test0.y, third, fourth) { }
    template <typename T> constexpr vec4_wrap_t(const vec2_wrap_t<T>& test0, const vec2_wrap_t<T>& test1) : vec4_wrap_t(test0.x, test0.y, test1.x, test1.y) { }
    template <typename T> constexpr vec4_wrap_t(const vec3_wrap_t<T>& test0) : vec4_wrap_t(test0.x, test0.y, test0.z, 0) { }
    template <typename T> constexpr vec4_wrap_t(const vec3_wrap_t<T>& test0, auto value) : vec4_wrap_t(test0.x, test0.y, test0.z, value) { }

  #if defined(FAN_GUI)
    template <typename Dest> requires (!requires { std::declval<Dest>().derived(); } && std::is_default_constructible_v<Dest> && requires(Dest d) { d.x; d.y; d.z; d.w; })
    constexpr operator Dest() const { Dest d{}; d.x = static_cast<std::remove_reference_t<decltype(d.x)>>(this->x); d.y = static_cast<std::remove_reference_t<decltype(d.y)>>(this->y); d.z = static_cast<std::remove_reference_t<decltype(d.z)>>(this->z); d.w = static_cast<std::remove_reference_t<decltype(d.w)>>(this->w); return d; }

    template <typename Src> requires (!requires { std::declval<Src>().derived(); } && requires(Src v) { v.x; v.y; v.z; v.w; })
    constexpr vec4_wrap_t(const Src& v) { this->x = static_cast<value_type_t>(v.x); this->y = static_cast<value_type_t>(v.y); this->z = static_cast<value_type_t>(v.z); this->w = static_cast<value_type_t>(v.w); }
  #endif

    template <typename T>
    constexpr operator vec2_wrap_t<T>() const { return {this->x, this->y}; }
  };

  #define fan_vector_types(X) \
    X(b,   bool) \
    X(i8,  std::int8_t) \
    X(i,   int) \
    X(ll,  long long) \
    X(ui,  std::uint32_t) \
    X(ull, unsigned long long) \
    X(f,   f32_t) \
    X(d,   f64_t)

  #define fan_gen_vec_aliases(suffix, type) \
    using vec1##suffix = vec1_wrap_t<type>; \
    using vec2##suffix = vec2_wrap_t<type>; \
    using vec3##suffix = vec3_wrap_t<type>; \
    using vec4##suffix = vec4_wrap_t<type>;

  fan_vector_types(fan_gen_vec_aliases);

  using vec1 = vec1f;
  using vec2 = vec2f;
  using vec3 = vec3f;
  using vec4 = vec4f;

  using vec2si = vec2_wrap_t<signed int>;

  template <typename casted_t, template<typename> typename vec_t, typename old_t>
  constexpr vec_t<casted_t> cast(const vec_t<old_t>& v) { return vec_t<casted_t>(v); }

  #define fan_vector_array
  #undef fan_coordinate
  #define fan_coordinate(x) arr[x]

  template <int vector_n, typename value_type_t>
  struct vec_wrap_t : vec_base<vec_wrap_t<vector_n, value_type_t>, vector_n, value_type_t> {
    #define vec_t vec_wrap_t
    #define vec_n vector_n
    #include "vector_impl.h"
    
    template <typename T> requires(vector_n >= 2)
    constexpr vec_wrap_t(const vec2_wrap_t<T>& test0) : vec_wrap_t(test0.x, test0.y) { } 

    template <typename T> requires(vector_n >= 2)
    constexpr vec_wrap_t(const vec3_wrap_t<T>& test0) : vec_wrap_t(test0.x, test0.y) { } 

    template <typename T> requires(vector_n >= 3)
    constexpr vec_wrap_t(const vec3_wrap_t<T>& test0) : vec_wrap_t(test0.x, test0.y, test0.z) { } 

    template <typename T> requires(vector_n >= 2)
    operator vec2_wrap_t<T>() const { return vec2_wrap_t<T>(this->operator[](0), this->operator[](1)); }

    template <typename T> requires(vector_n >= 3)
    operator vec3_wrap_t<T>() const { return vec3_wrap_t<T>(this->operator[](0), this->operator[](1), this->operator[](2)); }
  };

  #pragma pack(pop)

  struct ray3_t {
    fan::vec3 origin;
    fan::vec3 direction;
    constexpr ray3_t() = default;
    constexpr ray3_t(const fan::vec3& origin_, fan::vec3& direction_) : origin(origin_), direction(direction_){}
  };

  struct triangle_t { fan::vec3 v0, v1, v2; };

  #undef fan_coordinate_letters0
  #undef fan_coordinate_letters1
  #undef fan_coordinate_letters2
  #undef fan_coordinate_letters3
  #undef fan_coordinate_letters4
  #undef fan_coordinate
}

export namespace fan {
  void rect_diff(
    const fan::vec2i& old_min,
    const fan::vec2i& old_max,
    const fan::vec2i& new_min,
    const fan::vec2i& new_max,
    auto&& add_fn,
    auto&& remove_fn
  ) {
    new_min.rect({old_min.x - 1, new_max.y}, add_fn);
    fan::vec2i(old_max.x + 1, new_min.y).rect({new_max.x, new_max.y}, add_fn);
    new_min.rect({new_max.x, old_min.y - 1}, add_fn);
    fan::vec2i(new_min.x, old_max.y + 1).rect({new_max.x, new_max.y}, add_fn);
    old_min.rect({new_min.x - 1, old_max.y}, remove_fn);
    fan::vec2i(new_max.x + 1, old_min.y).rect({old_max.x, old_max.y}, remove_fn);
    old_min.rect({old_max.x, new_min.y - 1}, remove_fn);
    fan::vec2i(old_min.x, new_max.y + 1).rect({old_max.x, old_max.y}, remove_fn);
  }
  
  template <typename> inline constexpr bool is_vector_type_v = false;
  template <template <typename> typename V, typename T>
  inline constexpr bool is_vector_type_v<V<T>> =
    std::is_same_v<V<T>, vec0_wrap_t<T>> ||
    std::is_same_v<V<T>, vec1_wrap_t<T>> ||
    std::is_same_v<V<T>, vec2_wrap_t<T>> ||
    std::is_same_v<V<T>, vec3_wrap_t<T>> ||
    std::is_same_v<V<T>, vec4_wrap_t<T>>;
  template <int N, typename T>
  inline constexpr bool is_vector_type_v<vec_wrap_t<N, T>> = true;

  template <typename T> concept is_vector = is_vector_type_v<std::remove_cvref_t<T>>;
  template <typename T> concept is_not_vector = !is_vector_type_v<std::remove_cvref_t<T>>;

  template <class K>
  std::uint32_t get_hash_fast(const K& v) noexcept {
    if constexpr (requires { v.x; v.y; v.z; }) {
      return static_cast<std::uint32_t>(v.x) * 73856093u ^ static_cast<std::uint32_t>(v.y) * 19349663u ^ static_cast<std::uint32_t>(v.z) * 83492791u;
    }
    else if constexpr (requires { v.x; v.y; }) {
      return static_cast<std::uint32_t>(v.x) * 73856093u ^ static_cast<std::uint32_t>(v.y) * 19349663u;
    }
    else {
      return static_cast<std::uint32_t>(v);
    }
  }
}

namespace fan {
  constexpr std::size_t hash_combine(std::size_t seed, std::size_t h) {
    return seed ^ (h + 0x9e3779b9 + (seed << 6) + (seed >> 2));
  }
}

export namespace std {
  template <fan::is_vector V>
  struct hash<V> {
    std::size_t operator()(const V& v) const noexcept {
      std::size_t s = 0;
      for (std::size_t i = 0; i < v.size(); ++i) {
        s = fan::hash_combine(s, std::hash<typename std::remove_cvref_t<V>::value_type>{}(v[i]));
      }
      return s;
    }
  };
}

export namespace fan::math {
  template <typename T>
  constexpr fan::vec2_wrap_t<T> angle_to_vector(const T& angle_radians) {
    return fan::vec2_wrap_t<T>(std::cos(angle_radians), std::sin(angle_radians));
  }

  constexpr fan::vec3 centroid(const fan::vec3& v0, const fan::vec3& v1, const fan::vec3& v2) {
    return (v0 + v1 + v2) / 3.f;
  }
  
  fan::vec2 launch_to_target(fan::vec2 start, fan::vec2 target, f32_t gravity) {
    f32_t dy = target.y - start.y;
    f32_t vy = -std::sqrt(std::max(0.f, 2.f * gravity * std::abs(dy)));
    f32_t t = -vy / gravity;
    return {t > 0.f ? (target.x - start.x) / t : 0.f, vy};
  }
}