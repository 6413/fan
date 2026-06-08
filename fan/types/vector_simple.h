#pragma once

namespace fan {
  using access_type_t = uint8_t;

  template <typename T>
  struct vec0_wrap_t {

  };
  template <typename T>
  struct vec1_wrap_t {
    T x = 0;
    constexpr T& operator[](access_type_t idx) { return x; }
    constexpr const T& operator[](access_type_t idx) const { return x; }
  };
  template <typename T>
  struct vec2_wrap_t {
    T x = 0, y = 0;
    constexpr T& operator[](access_type_t idx) { return (idx == 0) ? x : y; }
    constexpr const T& operator[](access_type_t idx) const { return (idx == 0) ? x : y; }
  };
  template <typename T>
  struct vec3_wrap_t {
    T x = 0, y = 0, z = 0;
    constexpr T& operator[](access_type_t idx) { return (&x)[idx]; }
    constexpr const T& operator[](access_type_t idx) const { return (&x)[idx]; }
  };
  template <typename T>
  struct vec4_wrap_t {
    T x = 0, y = 0, z = 0, w = 0;
    constexpr T& operator[](access_type_t idx) { return (&x)[idx]; }
    constexpr const T& operator[](access_type_t idx) const { return (&x)[idx]; }
  };

  using vec1b = vec1_wrap_t<bool>;
  using vec2b = vec2_wrap_t<bool>;
  using vec3b = vec3_wrap_t<bool>;
  using vec4b = vec4_wrap_t<bool>;

  using vec1i = vec1_wrap_t<int>;
  using vec2i = vec2_wrap_t<int>;
  using vec3i = vec3_wrap_t<int>;
  using vec4i = vec4_wrap_t<int>;

  using vec1si = vec1i;
  using vec2si = vec2i;
  using vec3si = vec3i;
  using vec4si = vec4i;

  using vec1ui = vec1_wrap_t<uint32_t>;
  using vec2ui = vec2_wrap_t<uint32_t>;
  using vec3ui = vec3_wrap_t<uint32_t>;
  using vec4ui = vec4_wrap_t<uint32_t>;

  using vec1f = vec1_wrap_t<f32_t>;
  using vec2f = vec2_wrap_t<f32_t>;
  using vec3f = vec3_wrap_t<f32_t>;
  using vec4f = vec4_wrap_t<f32_t>;

  using vec1d = vec1_wrap_t<f64_t>;
  using vec2d = vec2_wrap_t<f64_t>;
  using vec3d = vec3_wrap_t<f64_t>;
  using vec4d = vec4_wrap_t<f64_t>;

  using vec1 = vec1_wrap_t<f32_t>;
  using vec2 = vec2_wrap_t<f32_t>;
  using vec3 = vec3_wrap_t<f32_t>;
  using vec4 = vec4_wrap_t<f32_t>;
}