module;
#include <fan/utility.h>
#include <cstdint>
#include <cstring>
#include <cmath>

module fan.types.matrix;

namespace fan {

  template <typename T>
  _matrix3x3<T>::_matrix3x3(const _matrix4x4<T>& m4) {
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) m[i][j] = m4[i][j];
  }

  template <typename T>
  _matrix3x3<T>::operator _matrix4x4<T>() const {
    _matrix4x4<T> r = _matrix4x4<T>::identity();
    for (int i = 0; i < 3; ++i) { r[i][0] = m[i][0]; r[i][1] = m[i][1]; r[i][2] = m[i][2]; }
    return r;
  }

  template <typename T>
  fan::quaternion<T> _matrix4x4<T>::to_quat(const _matrix4x4<T>& m) {
    fan::quaternion<T> q;
    T tr = m[0][0] + m[1][1] + m[2][2];
    if (tr > 0) {
      T s = 0.5f / std::sqrt(tr + 1);
      q.w = 0.25f / s; q.x = (m[1][2] - m[2][1]) * s; q.y = (m[2][0] - m[0][2]) * s; q.z = (m[0][1] - m[1][0]) * s;
    } else {
      if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
        T s = 2 * std::sqrt(1 + m[0][0] - m[1][1] - m[2][2]);
        q.w = (m[1][2] - m[2][1]) / s; q.x = 0.25f * s; q.y = (m[0][1] + m[1][0]) / s; q.z = (m[0][2] + m[2][0]) / s;
      } else if (m[1][1] > m[2][2]) {
        T s = 2 * std::sqrt(1 + m[1][1] - m[0][0] - m[2][2]);
        q.w = (m[2][0] - m[0][2]) / s; q.x = (m[0][1] + m[1][0]) / s; q.y = 0.25f * s; q.z = (m[1][2] + m[2][1]) / s;
      } else {
        T s = 2 * std::sqrt(1 + m[2][2] - m[0][0] - m[1][1]);
        q.w = (m[0][1] - m[1][0]) / s; q.x = (m[0][2] + m[2][0]) / s; q.y = (m[1][2] + m[2][1]) / s; q.z = 0.25f * s;
      }
    }
    return q.normalize();
  }

  template <typename T>
  runtime_matrix2d<T>::runtime_matrix2d(uint32_t r, uint32_t c) : rows(r), columns(c) {
    m = new T*[r];
    for (uint32_t i = 0; i < r; ++i) { m[i] = new T[c]; std::memset(m[i], 0, c * sizeof(T)); }
  }

  template <typename T>
  runtime_matrix2d<T>::~runtime_matrix2d() {
    if (m) { for (uint32_t i = 0; i < rows; ++i) delete[] m[i]; delete[] m; }
  }
  template struct _matrix4x4<f32_t>;
  template struct _matrix3x3<f32_t>;
  template struct runtime_matrix2d<f32_t>;
}