module;

module fan.types.matrix;

import std;

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

  template <typename T> _matrix4x4<T> _matrix4x4<T>::skew(const fan::vec3& s) {
    _matrix4x4 r(1); r[1][0] = s.x; r[0][1] = s.y; r[0][2] = s.z; return r;
  }

  template <typename T> _matrix4x4<T> _matrix4x4<T>::perspective(const fan::vec4& p) {
    _matrix4x4 r(1); r[3] = p; return r;
  }

  template <typename T> void _matrix4x4<T>::compose(const fan::vec3& p, const fan::quat& r, const fan::vec3& s, const fan::vec3& sk, const fan::vec4& pr) {
    *this = _matrix4x4(1).translate(p) * _matrix4x4(1).rotate(r) * _matrix4x4(1).scale(s) * _matrix4x4(1).skew(sk) * _matrix4x4(1).perspective(pr);
  }

  template <typename T> void _matrix4x4<T>::decompose(fan::vec3& p, fan::quat& r, fan::vec3& s, fan::vec3& sk, fan::vec4& pr) const {
    auto t = *this;
    p = {t[3][0], t[3][1], t[3][2]};
    s = {fan::vec3(t[0][0], t[0][1], t[0][2]).length(), fan::vec3(t[1][0], t[1][1], t[1][2]).length(), fan::vec3(t[2][0], t[2][1], t[2][2]).length()};
    if (std::abs(s.x) > 1e-6) t[0] = t[0] * (1.f / s.x);
    if (std::abs(s.y) > 1e-6) t[1] = t[1] * (1.f / s.y);
    if (std::abs(s.z) > 1e-6) t[2] = t[2] * (1.f / s.z);
    r = to_quat(t);
    sk.x = fan::math::dot(t[0], t[1]); t[1] = t[1] + t[0] * -sk.x;
    sk.y = fan::math::dot(t[0], t[2]); t[2] = t[2] + t[0] * -sk.y;
    sk.z = fan::math::dot(t[1], t[2]); t[2] = t[2] + t[1] * -sk.z;
    pr = t[3];
  }

  template <typename T> _matrix4x4<T> _matrix4x4<T>::rotation_set(f32_t a, const fan::vec3& v) const {
    f32_t c = std::cos(a), s = std::sin(a);
    fan::vec3 ax = v.normalize(), t = ax * (1.0f - c);
    _matrix4x4 rt{};
    rt[0][0] = c + t[0]*ax[0]; rt[0][1] = t[0]*ax[1] + s*ax[2]; rt[0][2] = t[0]*ax[2] - s*ax[1];
    rt[1][0] = t[1]*ax[0] - s*ax[2]; rt[1][1] = c + t[1]*ax[1]; rt[1][2] = t[1]*ax[2] + s*ax[0];
    rt[2][0] = t[2]*ax[0] + s*ax[1]; rt[2][1] = t[2]*ax[1] - s*ax[0]; rt[2][2] = c + t[2]*ax[2];
    _matrix4x4 r{};
    for(int i=0; i<3; ++i) for(int j=0; j<3; ++j) r[i][j] = m[0][j]*rt[i][0] + m[1][j]*rt[i][1] + m[2][j]*rt[i][2];
    r[3] = m[3];
    return r;
  }

  template <typename T> _matrix4x4<T> _matrix4x4<T>::rotate(f32_t a, const fan::vec3& v) const {
    f32_t c = std::cos(a), s = std::sin(a);
    fan::vec3 ax = v.normalize(), t = ax * (1.0f - c);
    _matrix4x4 rt{};
    rt[0][0] = c + t[0]*ax[0]; rt[0][1] = t[0]*ax[1] + s*ax[2]; rt[0][2] = t[0]*ax[2] - s*ax[1];
    rt[1][0] = t[1]*ax[0] - s*ax[2]; rt[1][1] = c + t[1]*ax[1]; rt[1][2] = t[1]*ax[2] + s*ax[0];
    rt[2][0] = t[2]*ax[0] + s*ax[1]; rt[2][1] = t[2]*ax[1] - s*ax[0]; rt[2][2] = c + t[2]*ax[2];
    _matrix4x4 r{};
    r[0] = m[0]*rt[0][0] + m[1]*rt[0][1] + m[2]*rt[0][2];
    r[1] = m[0]*rt[1][0] + m[1]*rt[1][1] + m[2]*rt[1][2];
    r[2] = m[0]*rt[2][0] + m[1]*rt[2][1] + m[2]*rt[2][2];
    r[3] = m[3];
    return r;
  }

  template <typename T> _matrix4x4<T> _matrix4x4<T>::rotate(const fan::vec3& a) const {
    f32_t cx = std::cos(a.x), sx = std::sin(a.x), cy = std::cos(a.y), sy = std::sin(a.y), cz = std::cos(a.z), sz = std::sin(a.z);
    _matrix4x4 rx{1,0,0,0, 0,cx,-sx,0, 0,sx,cx,0, 0,0,0,1};
    _matrix4x4 ry{cy,0,sy,0, 0,1,0,0, -sy,0,cy,0, 0,0,0,1};
    _matrix4x4 rz{cz,-sz,0,0, sz,cz,0,0, 0,0,1,0, 0,0,0,1};
    return *this * rx * ry * rz;
  }

  template <typename T> _matrix4x4<T> _matrix4x4<T>::rotate(const fan::quat& q) const {
    f32_t xx = q.x*q.x, yy = q.y*q.y, zz = q.z*q.z, xy = q.x*q.y, xz = q.x*q.z, yz = q.y*q.z, wx = q.w*q.x, wy = q.w*q.y, wz = q.w*q.z;
    _matrix4x4 rt{};
    rt[0][0] = 1.0f - 2.0f*(yy + zz); rt[0][1] = 2.0f*(xy - wz); rt[0][2] = 2.0f*(xz + wy);
    rt[1][0] = 2.0f*(xy + wz); rt[1][1] = 1.0f - 2.0f*(xx + zz); rt[1][2] = 2.0f*(yz - wx);
    rt[2][0] = 2.0f*(xz - wy); rt[2][1] = 2.0f*(yz + wx); rt[2][2] = 1.0f - 2.0f*(xx + yy);
    _matrix4x4 r{};
    for (int i=0; i<3; ++i) for (int j=0; j<3; ++j) r[i][j] = m[0][j]*rt[i][0] + m[1][j]*rt[i][1] + m[2][j]*rt[i][2];
    r[3] = m[3];
    return r;
  }

  template <typename T> fan::vec3 _matrix4x4<T>::get_euler_angles() const {
    fan::vec3 r;
    r.z = std::atan2(m[1][0], m[0][0]);
    f32_t sp = -m[2][0];
    if (sp <= -1.0f) r.y = -fan::math::two_pi;
    else if (sp >= 1.0f) r.y = fan::math::two_pi;
    else r.y = std::asin(sp);
    r.x = std::atan2(m[2][1], m[2][2]);
    return r;
  }

  template <typename T> runtime_matrix2d<T>::runtime_matrix2d(std::uint32_t r, std::uint32_t c) : rows(r), columns(c) {
    m = new T*[r];
    for (std::uint32_t i = 0; i < r; ++i) { m[i] = new T[c]; std::memset(m[i], 0, c * sizeof(T)); }
  }

  template <typename T> runtime_matrix2d<T>::runtime_matrix2d(const runtime_matrix2d& o) : rows(o.rows), columns(o.columns) {
    m = new T*[rows];
    for (std::uint32_t i = 0; i < rows; ++i) { m[i] = new T[columns]; std::copy(o.m[i], o.m[i] + columns, m[i]); }
  }

  template <typename T> runtime_matrix2d<T>& runtime_matrix2d<T>::operator=(const runtime_matrix2d& o) {
    if (this != &o) {
      if (m) { for (std::uint32_t i = 0; i < rows; ++i) delete[] m[i]; delete[] m; }
      rows = o.rows; columns = o.columns;
      m = new T*[rows];
      for (std::uint32_t i = 0; i < rows; ++i) { m[i] = new T[columns]; std::copy(o.m[i], o.m[i] + columns, m[i]); }
    }
    return *this;
  }

  template <typename T> runtime_matrix2d<T>::~runtime_matrix2d() {
    if (m) { for (std::uint32_t i = 0; i < rows; ++i) delete[] m[i]; delete[] m; }
  }

  template <typename T> runtime_matrix2d<T> runtime_matrix2d<T>::operator+(const runtime_matrix2d& o) const {
    runtime_matrix2d r(rows, columns);
    for (std::uint32_t i=0; i<rows; ++i) for (std::uint32_t j=0; j<columns; ++j) r[i][j] = m[i][j] + o[i][j];
    return r;
  }

  template <typename T> runtime_matrix2d<T> runtime_matrix2d<T>::operator-(const runtime_matrix2d& o) const {
    runtime_matrix2d r(rows, columns);
    for (std::uint32_t i=0; i<rows; ++i) for (std::uint32_t j=0; j<columns; ++j) r[i][j] = m[i][j] - o[i][j];
    return r;
  }

  template <typename T> runtime_matrix2d<T> runtime_matrix2d<T>::operator*(const runtime_matrix2d& o) const {
    runtime_matrix2d r(rows, o.columns);
    for (std::uint32_t i=0; i<rows; ++i) for (std::uint32_t j=0; j<o.columns; ++j) for (std::uint32_t k=0; k<columns; ++k) r[i][j] += m[i][k] * o[k][j];
    return r;
  }

  template <typename T> runtime_matrix2d<T> runtime_matrix2d<T>::transpose() const {
    runtime_matrix2d r(columns, rows);
    for (std::uint32_t i=0; i<rows; ++i) for (std::uint32_t j=0; j<columns; ++j) r[j][i] = m[i][j];
    return r;
  }

  template <typename T> void runtime_matrix2d<T>::randomize() {
    for (std::uint32_t i=0; i<rows; ++i) for (std::uint32_t j=0; j<columns; ++j) m[i][j] = fan::random::value_f32(-1, 1);
  }

  template <typename T> void runtime_matrix2d<T>::zero() {
    for (std::uint32_t i=0; i<rows; ++i) std::memset(m[i], 0, columns * sizeof(T));
  }

  template struct _matrix4x4<f32_t>;
  template struct _matrix3x3<f32_t>;
  template struct runtime_matrix2d<f32_t>;
}