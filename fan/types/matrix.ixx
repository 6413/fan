module;
#include <fan/utility.h>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <algorithm>

export module fan.types.matrix;

import fan.types;
import fan.types.vector;
import fan.types.quaternion;
import fan.print.error;
import fan.random;
import fan.math;

export namespace fan {

  template <typename T>
  struct _matrix4x4;

  template <typename T>
  struct _matrix2x2 {
    fan::vec2_wrap_t<T> m[2];
    using value_type = fan::vec2_wrap_t<T>;
    _matrix2x2() = default;
    constexpr _matrix2x2(T x, T y, T z, T w) : m{ {x, y}, {z, w} } {}
    constexpr _matrix2x2(const fan::vec2_wrap_t<T>& v1, const fan::vec2_wrap_t<T>& v2) : m{ v1, v2 } {}
    constexpr _matrix2x2 operator+(const fan::vec2_wrap_t<T>& v) const { return {m[0] + v, m[1] + v}; }
    constexpr fan::vec2_wrap_t<T> operator*(const fan::vec2_wrap_t<T>& v) const {
      return {m[0][0] * v.x + m[0][1] * v.y, m[1][0] * v.x + m[1][1] * v.y};
    }
    static fan::vec2 rotate(const fan::vec2& v, f32_t a) {
      f32_t c = std::cos(a), s = std::sin(a);
      return {v.x * c - v.y * s, v.x * s + v.y * c};
    }
    constexpr fan::vec2& operator[](uintptr_t i) { return m[i]; }
    constexpr fan::vec2 operator[](uintptr_t i) const { return m[i]; }
  };

  template <typename T>
  struct _matrix3x3 {
    fan::vec3_wrap_t<T> m[3];
    using value_type = fan::vec3_wrap_t<T>;
    _matrix3x3() = default;
    constexpr _matrix3x3(T x1, T y1, T z1, T x2, T y2, T z2, T x3, T y3, T z3) : m{ {x1, y1, z1}, {x2, y2, z2}, {x3, y3, z3} } {}
    constexpr _matrix3x3(const fan::vec3_wrap_t<T>& v1, const fan::vec3_wrap_t<T>& v2, const fan::vec3_wrap_t<T>& v3) : m{ v1, v2, v3 } {}
    _matrix3x3(const _matrix4x4<T>& m4);
    operator _matrix4x4<T>() const;
    constexpr _matrix3x3 operator+(const fan::vec3_wrap_t<T>& v) const { return {m[0] + v, m[1] + v, m[2] + v}; }
    constexpr fan::vec3_wrap_t<T> operator*(const fan::vec3_wrap_t<T>& v) const {
      return {m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z, m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z, m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z};
    }
    static fan::vec3 rotate_z(const fan::vec3& v, f32_t a) {
      f32_t c = std::cos(a), s = std::sin(a);
      return {v.x * c - v.y * s, v.x * s + v.y * c, v.z};
    }
    constexpr fan::vec3& operator[](uintptr_t i) { return m[i]; }
    constexpr fan::vec3 operator[](uintptr_t i) const { return m[i]; }
  };

  template <typename T>
  struct _matrix4x4 {
    fan::vec4_wrap_t<T> m[4];
    using value_type = fan::vec4_wrap_t<T>;
    _matrix4x4() = default;
    constexpr _matrix4x4(T v) : m{ {v, 0, 0, 0}, {0, v, 0, 0}, {0, 0, v, 0}, {0, 0, 0, v} } {}
    constexpr _matrix4x4(T x0, T y0, T z0, T w0, T x1, T y1, T z1, T w1, T x2, T y2, T z2, T w2, T x3, T y3, T z3, T w3)
      : m{ {x0, x1, x2, x3}, {y0, y1, y2, y3}, {z0, z1, z2, z3}, {w0, w1, w2, w3} } {}
    constexpr _matrix4x4(const fan::quaternion<T>& q) : _matrix4x4<T>(1) {
      T xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z, xz = q.x * q.z, xy = q.x * q.y, yz = q.y * q.z, wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;
      m[0][0] = 1 - 2 * (yy + zz); m[0][1] = 2 * (xy + wz); m[0][2] = 2 * (xz - wy);
      m[1][0] = 2 * (xy - wz);     m[1][1] = 1 - 2 * (xx + zz); m[1][2] = 2 * (yz + wx);
      m[2][0] = 2 * (xz + wy);     m[2][1] = 2 * (yz - wx);     m[2][2] = 1 - 2 * (xx + yy);
    }

    static constexpr _matrix4x4 identity() { return _matrix4x4(1); }
    constexpr bool is_identity() const {
      return m[0][0] == 1 && m[1][1] == 1 && m[2][2] == 1 && m[0][1] == 0 && m[1][0] == 0 && m[2][1] == 0 && m[0][2] == 0 && m[1][2] == 0 && m[2][0] == 0 && m[0][3] == 0 && m[1][3] == 0 && m[2][3] == 0;
    }
    static constexpr size_t size() { return 16; }

    constexpr _matrix4x4 operator*(const _matrix4x4& rhs) const {
      _matrix4x4 r{};
      for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) for (int k = 0; k < 4; k++) r[i][j] += m[k][j] * rhs[i][k];
      return r;
    }
    constexpr fan::vec4_wrap_t<T> operator*(const fan::vec4_wrap_t<T>& r) const {
      fan::vec4_wrap_t<T> res{};
      for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) res[i] += m[j][i] * r[j];
      return res;
    }
    constexpr _matrix4x4 operator*(T r) const {
      _matrix4x4 res;
      for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) res[i][j] = m[i][j] * r;
      return res;
    }
    constexpr _matrix4x4& operator+=(const _matrix4x4& r) {
      for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) m[i][j] += r[i][j];
      return *this;
    }
    constexpr fan::vec3 operator*(const fan::vec3& r) const {
      fan::vec3 res{};
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) res[i] += m[j][i] * r[j];
        res[i] += m[3][i];
      }
      return res;
    }

    constexpr _matrix4x4 inverse() const {
      _matrix4x4 r = *this, id = 1;
      for (int i = 0; i < 4; ++i) {
        int p = i;
        for (int j = i + 1; j < 4; ++j) if (std::abs(r[j][i]) > std::abs(r[p][i])) p = j;
        std::swap(r[i], r[p]); std::swap(id[i], id[p]);
        T d = r[i][i];
        if (d == 0) continue; 
        for (int j = 0; j < 4; ++j) { r[i][j] /= d; id[i][j] /= d; }
        for (int k = 0; k < 4; ++k) if (k != i) {
          T f = r[k][i];
          for (int j = 0; j < 4; ++j) { r[k][j] -= f * r[i][j]; id[k][j] -= f * id[i][j]; }
        }
      }
      return id;
    }
    constexpr _matrix4x4 translate(const fan::vec3& v) const {
      _matrix4x4 r(*this);
      r[3][0] += v[0]; r[3][1] += v[1]; r[3][2] += v[2];
      return r;
    }
    constexpr _matrix4x4 scale(const fan::vec3& v) const {
      _matrix4x4 r(*this);
      for(int i=0; i<3; ++i) for(int j=0; j<3; ++j) r[i][j] *= v[i];
      return r;
    }

    constexpr fan::vec4& operator[](uintptr_t i) { return m[i]; }
    constexpr fan::vec4 operator[](uintptr_t i) const { return m[i]; }
    T* data() { return &m[0][0]; }
    const T* data() const { return &m[0][0]; }

    constexpr fan::vec3 get_translation() const { return {m[3][0], m[3][1], m[3][2]}; }
    constexpr fan::quat get_rotation() const { return to_quat(*this); }
    constexpr fan::vec3 get_scale() const { return {fan::vec3(m[0][0], m[0][1], m[0][2]).length(), fan::vec3(m[1][0], m[1][1], m[1][2]).length(), fan::vec3(m[2][0], m[2][1], m[2][2]).length()}; }

    _matrix4x4 skew(const fan::vec3& s);
    _matrix4x4 perspective(const fan::vec4& p);
    void compose(const fan::vec3& p, const fan::quat& r, const fan::vec3& s, const fan::vec3& sk, const fan::vec4& pr);
    void decompose(fan::vec3& p, fan::quat& r, fan::vec3& s, fan::vec3& sk, fan::vec4& pr) const;
    
    _matrix4x4 rotation_set(f32_t a, const fan::vec3& v) const;
    _matrix4x4 rotate(f32_t a, const fan::vec3& v) const;
    _matrix4x4 rotate(const fan::vec3& a) const;
    _matrix4x4 rotate(const fan::quat& q) const;
    fan::vec3 get_euler_angles() const;

    static fan::quaternion<T> to_quat(const _matrix4x4& m);
    template <typename U> operator fan::quaternion<U>() const { return _matrix4x4<U>::to_quat(*(const _matrix4x4<U>*)this); }
  };

  template <uint32_t R, uint32_t C, typename T = f32_t>
  struct matrix2d {
    T m[R][C];
    constexpr matrix2d() : m{} {}
    template <typename... Args> requires (sizeof...(Args) > 1)
    constexpr matrix2d(Args&&... a) : m{ static_cast<T>(a)... } {}
    constexpr matrix2d(T v) : matrix2d() { for (uint32_t i = 0; i < R; ++i) m[i][i] = v; }
    constexpr T* operator[](uintptr_t i) { return m[i]; }
    constexpr const T* operator[](uintptr_t i) const { return m[i]; }
    constexpr matrix2d operator+(const matrix2d& o) const { matrix2d r; for (uint32_t i = 0; i < R * C; ++i) r.data()[i] = data()[i] + o.data()[i]; return r; }
    constexpr void operator+=(const matrix2d& o) { *this = *this + o; }
    constexpr matrix2d operator-(const matrix2d& o) const { matrix2d r; for (uint32_t i = 0; i < R * C; ++i) r.data()[i] = data()[i] - o.data()[i]; return r; }
    template <uint32_t C2> constexpr matrix2d<R, C2, T> operator*(const matrix2d<C, C2, T>& o) const {
      matrix2d<R, C2, T> r;
      for (uint32_t i = 0; i < R; ++i) for (uint32_t j = 0; j < C2; ++j) for (uint32_t k = 0; k < C; ++k) r[i][j] += m[i][k] * o[k][j];
      return r;
    }
    constexpr matrix2d<C, R, T> transpose() const { matrix2d<C, R, T> r; for (uint32_t i = 0; i < R; ++i) for (uint32_t j = 0; j < C; ++j) r[j][i] = m[i][j]; return r; }
    void randomize() { for (uint32_t i = 0; i < R * C; ++i) data()[i] = fan::random::value_f32(-1, 1); }
    constexpr matrix2d hadamard(const matrix2d& o) const { matrix2d r; for (uint32_t i = 0; i < R * C; ++i) r.data()[i] = data()[i] * o.data()[i]; return r; }
    constexpr matrix2d sigmoid() const { matrix2d r; for (uint32_t i = 0; i < R * C; ++i) r.data()[i] = fan::math::sigmoid(data()[i]); return r; }
    constexpr matrix2d sigmoid_derivative() const { matrix2d r; for (uint32_t i = 0; i < R * C; ++i) r.data()[i] = fan::math::sigmoid_derivative(data()[i]); return r; }
    void zero() { std::memset(m, 0, sizeof(m)); }
    T* data() { return &m[0][0]; }
    const T* data() const { return &m[0][0]; }
  };

  template <typename T = f32_t>
  struct runtime_matrix2d {
    uint32_t rows, columns;
    T** m = nullptr;
    runtime_matrix2d(uint32_t r, uint32_t c);
    runtime_matrix2d(const runtime_matrix2d& o);
    runtime_matrix2d& operator=(const runtime_matrix2d& o);
    ~runtime_matrix2d();
    T*& operator[](uint32_t i) { return m[i]; }
    const T* operator[](uint32_t i) const { return m[i]; }
    runtime_matrix2d operator+(const runtime_matrix2d& o) const;
    runtime_matrix2d operator-(const runtime_matrix2d& o) const;
    runtime_matrix2d operator*(const runtime_matrix2d& o) const;
    runtime_matrix2d transpose() const;
    void randomize();
    void zero();
  };

  template <typename T> fan::quaternion<T> to_quat(const _matrix4x4<T>& m) { return _matrix4x4<T>::to_quat(m); }
  template <typename T> constexpr _matrix4x4<T> translation_matrix(const fan::vec3_wrap_t<T>& v) { return _matrix4x4<T>(1).translate(v); }
  template <typename T> constexpr _matrix4x4<T> scaling_matrix(const fan::vec3_wrap_t<T>& v) { return _matrix4x4<T>(1).scale(v); }
  template <typename T> constexpr _matrix4x4<T> inverse(const _matrix4x4<T>& m) { return m.inverse(); }

  using mat2 = _matrix2x2<cf_t>;
  using mat3 = _matrix3x3<cf_t>;
  using mat4 = _matrix4x4<cf_t>;

  struct basis {
    fan::vec3 right, forward, up;
    operator fan::mat3() const { return {right.x, right.y, right.z, forward.x, forward.y, forward.z, up.x, up.y, up.z}; }
    fan::vec3 operator*(const fan::vec3& v) const { return (fan::mat3(*this)) * v; }
  };
}