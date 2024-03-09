#pragma once

#include _FAN_PATH(types/vector.h)
#include _FAN_PATH(types/quaternion.h)
#include _FAN_PATH(math/random.h)

#include <type_traits>

#include <iostream>
#include <exception>
#include <cstring>

namespace fan {

  template <typename type_t>
  struct _matrix2x2 {

    using value_type = fan::vec2_wrap_t<type_t>;

    _matrix2x2() = default;

    template <typename T>
    constexpr _matrix2x2(T x, T y, T z, T w) : m_array{ fan::vec2{x, y}, fan::vec2{z, w} } {}

    template <typename T, typename T2>
    constexpr _matrix2x2(const fan::vec2_wrap_t<T>& v, const fan::vec2_wrap_t<T2>& v2) : m_array{ v.x, v.y, v2.x, v2.y } {}

    constexpr _matrix2x2 operator+(const fan::vec2_wrap_t<type_t>& v) const {
      return _matrix2x2(m_array[0][0] + v[0], m_array[0][1] + v[1], m_array[1][0] + v[0], m_array[1][1] + v[1]);
    }

    template <typename T>
    constexpr fan::vec2_wrap_t<T> operator*(const fan::vec2_wrap_t<T>& v) const {
      return fan::vec2_wrap_t<T>(m_array[0][0] * v.x + m_array[0][1] * v.y, m_array[1][0] * v.x + m_array[1][1] * v.y);
    }

    static fan::vec2 rotate(const fan::vec2& v, f32_t angle) {
      f32_t c = cos(angle);
      f32_t s = sin(angle);
      return fan::vec2(v.x * c - v.y * s, v.x * s + v.y * c);
    }

    constexpr fan::vec2 operator[](const uintptr_t i) const {
      return m_array[i];
    }

    constexpr fan::vec2& operator[](const uintptr_t i) {
      return m_array[i];
    }

    template <typename T>
    friend std::ostream& operator<<(std::ostream& os, const _matrix2x2<T>& matrix)
    {
      for (uintptr_t i = 0; i < 2; i++) {
        for (uintptr_t j = 0; j < 2; j++) {
          os << matrix[j][i] << ' ';
        }
        os << '\n';
      }
      return os;
    }
  protected:
    std::array<fan::vec2, 2> m_array;
  };

  template <typename type_t>
  struct _matrix4x4 {

    using value_type = fan::vec4_wrap_t<type_t>;

    constexpr _matrix4x4() = default;

    constexpr _matrix4x4(type_t value) {
      std::memset(m_array.data(), 0, sizeof(type_t) * 4 * 4);
      for (int i = 0; i < 4; i++) {
        m_array[i][i] = value;
      }
    }
    constexpr _matrix4x4(
      type_t x0, type_t y0, type_t z0, type_t w0,
      type_t x1, type_t y1, type_t z1, type_t w1,
      type_t x2, type_t y2, type_t z2, type_t w2,
      type_t x3, type_t y3, type_t z3, type_t w3
    ) {
      m_array[0][0] = x0;
      m_array[1][0] = y0;
      m_array[2][0] = z0;
      m_array[3][0] = w0;

      m_array[0][1] = x1;
      m_array[1][1] = y1;
      m_array[2][1] = z1;
      m_array[3][1] = w1;

      m_array[0][2] = x2;
      m_array[1][2] = y2;
      m_array[2][2] = z2;
      m_array[3][2] = w2;

      m_array[0][3] = x3;
      m_array[1][3] = y3;
      m_array[2][3] = z3;
      m_array[3][3] = w3;
    }

    #if defined(loco_assimp)
    _matrix4x4(const aiMatrix4x4& mat) {
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          m_array[i][j] = mat[j][i];
        }
      }
    }
    operator aiMatrix4x4() const {
      aiMatrix4x4 mat;
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          mat[j][i] = m_array[i][j];
        }
      }
      return mat;
    }
    #endif

    _matrix4x4 skew(const fan::vec3& skew) {
      _matrix4x4 result(1);
      result[1][0] = skew.x;  // x-skew factor
      result[0][1] = skew.y;  // y-skew factor
      result[0][2] = skew.z;  // z-skew factor
      return result;
    }

    // Perspective transformation
    _matrix4x4 perspective(const fan::vec4& perspective) {
      _matrix4x4 result(1);
      result[3][0] = perspective.x;  // x-perspective factor
      result[3][1] = perspective.y;  // y-perspective factor
      result[3][2] = perspective.z;  // z-perspective factor
      result[3][3] = perspective.w;  // w-perspective factor (usually 1)
      return result;
    }

    void compose(const fan::vec3& position, const fan::quat& rotation, const fan::vec3& scale, const fan::vec3& skew, const fan::vec4& perspective) {
      const _matrix4x4 translation_matrix = _matrix4x4(1).translate(position);
      const _matrix4x4 rotation_matrix = _matrix4x4(1).rotate(rotation);
      const _matrix4x4 scale_matrix = _matrix4x4(1).scale(scale);
      const _matrix4x4 skew_matrix = _matrix4x4(1).skew(skew);
      const _matrix4x4 perspective_matrix = _matrix4x4(1).perspective(perspective);

      *this = translation_matrix * rotation_matrix * scale_matrix * skew_matrix * perspective_matrix;

      //*this = translation_matrix * rotation_matrix * scale_matrix;

    }

    fan::vec4 combine(const fan::vec4& a, const fan::vec4& b, float ascl, float bscl) const {
      return a * ascl + b * bscl;
    }

    void decompose(fan::vec3& position, fan::quat& rotation, fan::vec3& scale, fan::vec3& skew, fan::vec4& perspective) const {
      
      auto temp = *this;

      position = fan::vec3(temp[3][0], temp[3][1], temp[3][2]);

      scale = fan::vec3(
        fan::vec3(temp[0][0], temp[0][1], temp[0][2]).length(),
        fan::vec3(temp[1][0], temp[1][1], temp[1][2]).length(),
        fan::vec3(temp[2][0], temp[2][1], temp[2][2]).length()
      );

      const double epsilon = 1e-6;

      if (std::abs(scale.x) > epsilon) temp[0] /= scale.x;
      if (std::abs(scale.y) > epsilon) temp[1] /= scale.y;
      if (std::abs(scale.z) > epsilon) temp[2] /= scale.z;

      rotation = fan::quat().from_matrix(temp);

      skew.x = fan::math::dot(temp[0], temp[1]);
      temp[1] = combine(temp[1], temp[0], 1, -skew.x);

      skew.y = fan::math::dot(temp[0], temp[2]);
      temp[2] = combine(temp[2], temp[0], 1, -skew.y);

      skew.z = fan::math::dot(temp[1], temp[2]);
      temp[2] = combine(temp[2], temp[1], 1, -skew.z);

      // Calculate perspective
      perspective = temp[3];
    }

    template <typename T>
    constexpr _matrix4x4(const fan::quaternion<T>& quat) : _matrix4x4<type_t>(1) {
      f32_t qxx(quat[0] * quat[0]);
      f32_t qyy(quat[1] * quat[1]);
      f32_t qzz(quat[2] * quat[2]);
      f32_t qxz(quat[0] * quat[2]);
      f32_t qxy(quat[0] * quat[1]);
      f32_t qyz(quat[1] * quat[2]);
      f32_t qwx(quat[3] * quat[0]);
      f32_t qwy(quat[3] * quat[1]);
      f32_t qwz(quat[3] * quat[2]);

      m_array[0][0] = f32_t(1) - f32_t(2) * (qyy + qzz);
      m_array[0][1] = f32_t(2) * (qxy + qwz);
      m_array[0][2] = f32_t(2) * (qxz - qwy);

      m_array[1][0] = f32_t(2) * (qxy - qwz);
      m_array[1][1] = f32_t(1) - f32_t(2) * (qxx + qzz);
      m_array[1][2] = f32_t(2) * (qyz + qwx);

      m_array[2][0] = f32_t(2) * (qxz + qwy);
      m_array[2][1] = f32_t(2) * (qyz - qwx);
      m_array[2][2] = f32_t(1) - f32_t(2) * (qxx + qyy);
    }

    template <typename T>
    friend std::ostream& operator<<(std::ostream& os, const _matrix4x4& matrix)
    {
      for (uintptr_t i = 0; i < 4; i++) {
        for (uintptr_t j = 0; j < 4; j++) {
          os << matrix[j][i] << ' ';
        }
        os << '\n';
      }
      return os;
    }

    constexpr fan::vec4& operator[](const uintptr_t i) {
      return m_array[i];
    }

    constexpr fan::vec4 operator[](const uintptr_t i) const {
      return m_array[i];
    }

    constexpr _matrix4x4<type_t> operator*(const _matrix4x4<type_t>& rhs) const {
      _matrix4x4<type_t> result{};

      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
          for (int k = 0; k < 4; k++) {
            result[i][j] += m_array[k][j] * rhs[i][k];
          }
        }
      }
      return result;
    }

    constexpr fan::vec4_wrap_t<type_t> operator*(const fan::vec4_wrap_t<type_t>& rhs) const {
      fan::vec4_wrap_t<type_t> result{};

      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
          result[i] += m_array[j][i] * rhs[j];
        }
      }

      return result;
    }

    constexpr _matrix4x4<type_t> operator*(const type_t& rhs) const {
      _matrix4x4<type_t> result{};

      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
          result[i][j] = m_array[i][j] * rhs;
        }
      }

      return result;
    }

    constexpr _matrix4x4<type_t> operator+=(const _matrix4x4<type_t>& rhs) {
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
          (*this)[i][j] += rhs[i][j];
        }
      }
      return *this;
    }

    constexpr _matrix4x4 translate(const fan::vec3& v) const {
			_matrix4x4 matrix((*this));
			matrix[3][0] = (*this)[0][0] * v[0] + (*this)[1][0] * v[1] + (v.size() < 3 ? + 0 : ((*this)[2][0] * v[2])) + (*this)[3][0];
			matrix[3][1] = (*this)[0][1] * v[0] + (*this)[1][1] * v[1] + (v.size() < 3 ? + 0 : ((*this)[2][1] * v[2])) + (*this)[3][1];
			matrix[3][2] = (*this)[0][2] * v[0] + (*this)[1][2] * v[1] + (v.size() < 3 ? + 0 : ((*this)[2][2] * v[2])) + (*this)[3][2];
			matrix[3][3] = (*this)[0][3] * v[0] + (*this)[1][3] * v[1] + (v.size() < 3 ? + 0 : ((*this)[2][3] * v[2])) + (*this)[3][3];
			return matrix;
		}

    constexpr fan::vec3 get_translation() const {
			return fan::vec3((*this)[3][0], (*this)[3][1], (*this)[3][2]);
		}

		constexpr fan::vec3 get_scale() const {
      return fan::vec3(
        fan::vec3((*this)[0][0], (*this)[0][1], (*this)[0][2]).length(),
        fan::vec3((*this)[1][0], (*this)[1][1], (*this)[1][2]).length(),
        fan::vec3((*this)[2][0], (*this)[2][1], (*this)[2][2]).length()
      );
		}

		constexpr _matrix4x4 scale(const fan::vec3& v) const {
			_matrix4x4 matrix{};

			matrix[0][0] = (*this)[0][0] * v[0];
			matrix[0][1] = (*this)[0][1] * v[0];
			matrix[0][2] = (*this)[0][2] * v[0];

			matrix[1][0] = (*this)[1][0] * v[1];
			matrix[1][1] = (*this)[1][1] * v[1];
			matrix[1][2] = (*this)[1][2] * v[1];

			matrix[2][0] = (v.size() < 3 ? 0 : (*this)[2][0] * v[2]);
			matrix[2][1] = (v.size() < 3 ? 0 : (*this)[2][1] * v[2]);
			matrix[2][2] = (v.size() < 3 ? 0 : (*this)[2][2] * v[2]);

			matrix[3][0] = (*this)[3][0];
			matrix[3][1] = (*this)[3][1];
			matrix[3][2] = (*this)[3][2];

			matrix[3] = (*this)[3];
			return matrix;
		}

    // sets rotation to specific angle
		constexpr _matrix4x4 rotation_set(f32_t angle, const fan::vec3& v) const {
			const f32_t a = angle;
			const f32_t c = cos(a);
			const f32_t s = sin(a);
			fan::vec3 axis(fan_3d::math::normalize(v));
			fan::vec3 temp(axis * (1.0f - c));

			_matrix4x4 rotation{};
			rotation[0][0] = c + temp[0] * axis[0];
			rotation[0][1] = temp[0] * axis[1] + s * axis[2];
			rotation[0][2] = temp[0] * axis[2] - s * axis[1];

			rotation[1][0] = temp[1] * axis[0] - s * axis[2];
			rotation[1][1] = c + temp[1] * axis[1];
			rotation[1][2] = temp[1] * axis[2] + s * axis[0];

			rotation[2][0] = temp[2] * axis[0] + s * axis[1];
			rotation[2][1] = temp[2] * axis[1] - s * axis[0];
			rotation[2][2] = c + temp[2] * axis[2];

			_matrix4x4 matrix{};
			matrix[0][0] = ((*this)[0][0] * rotation[0][0]) + ((*this)[1][0] * rotation[0][1]) + ((*this)[2][0] * rotation[0][2]);
			matrix[1][0] = ((*this)[0][1] * rotation[0][0]) + ((*this)[1][1] * rotation[0][1]) + ((*this)[2][1] * rotation[0][2]);
			matrix[2][0] = ((*this)[0][2] * rotation[0][0]) + ((*this)[1][2] * rotation[0][1]) + ((*this)[2][2] * rotation[0][2]);

			matrix[0][1] = ((*this)[0][0] * rotation[1][0]) + ((*this)[1][0] * rotation[1][1]) + ((*this)[2][0] * rotation[1][2]);
			matrix[1][1] = ((*this)[0][1] * rotation[1][0]) + ((*this)[1][1] * rotation[1][1]) + ((*this)[2][1] * rotation[1][2]);
			matrix[2][1] = ((*this)[0][2] * rotation[1][0]) + ((*this)[1][2] * rotation[1][1]) + ((*this)[2][2] * rotation[1][2]);

			matrix[0][2] = ((*this)[0][0] * rotation[2][0]) + ((*this)[1][0] * rotation[2][1]) + ((*this)[2][0] * rotation[2][2]);
			matrix[1][2] = ((*this)[0][1] * rotation[2][0]) + ((*this)[1][1] * rotation[2][1]) + ((*this)[2][1] * rotation[2][2]);
			matrix[2][2] = ((*this)[0][2] * rotation[2][0]) + ((*this)[1][2] * rotation[2][1]) + ((*this)[2][2] * rotation[2][2]);

			matrix[3] = (*this)[3];

			return matrix;
		}
    constexpr _matrix4x4 rotate(f32_t angle, const fan::vec3& v) const {
      const f32_t a = angle;
      const f32_t c = cos(a);
      const f32_t s = sin(a);
      fan::vec3 axis(fan_3d::math::normalize(v));
      fan::vec3 temp(axis * (1.0f - c));

      _matrix4x4 rotation{};
      rotation[0][0] = c + temp[0] * axis[0];
      rotation[0][1] = temp[0] * axis[1] + s * axis[2];
      rotation[0][2] = temp[0] * axis[2] - s * axis[1];

      rotation[1][0] = temp[1] * axis[0] - s * axis[2];
      rotation[1][1] = c + temp[1] * axis[1];
      rotation[1][2] = temp[1] * axis[2] + s * axis[0];

      rotation[2][0] = temp[2] * axis[0] + s * axis[1];
      rotation[2][1] = temp[2] * axis[1] - s * axis[0];
      rotation[2][2] = c + temp[2] * axis[2];

      _matrix4x4 matrix{};
      matrix[0] = (*this)[0] * rotation[0][0] + (*this)[1] * rotation[0][1] + (*this)[2] * rotation[0][2];
      matrix[1] = (*this)[0] * rotation[1][0] + (*this)[1] * rotation[1][1] + (*this)[2] * rotation[1][2];
      matrix[2] = (*this)[0] * rotation[2][0] + (*this)[1] * rotation[2][1] + (*this)[2] * rotation[2][2];

      // Copy the unchanged part of the matrix
      matrix[3] = (*this)[3];

      return matrix;
    }
    constexpr _matrix4x4 rotate(const fan::vec3& angles) const {
      const f32_t angle_x = angles.x;
      const f32_t angle_y = angles.y;
      const f32_t angle_z = angles.z;

      const f32_t cos_x = cos(angle_x);
      const f32_t sin_x = sin(angle_x);
      const f32_t cos_y = cos(angle_y);
      const f32_t sin_y = sin(angle_y);
      const f32_t cos_z = cos(angle_z);
      const f32_t sin_z = sin(angle_z);

      const _matrix4x4 rotation_x{
          1.0f, 0.0f, 0.0f, 0.0f,
          0.0f, cos_x, -sin_x, 0.0f,
          0.0f, sin_x, cos_x, 0.0f,
          0.0f, 0.0f, 0.0f, 1.0f
      };

      const _matrix4x4 rotation_y{
          cos_y, 0.0f, sin_y, 0.0f,
          0.0f, 1.0f, 0.0f, 0.0f,
          -sin_y, 0.0f, cos_y, 0.0f,
          0.0f, 0.0f, 0.0f, 1.0f
      };

      const _matrix4x4 rotation_z{
          cos_z, -sin_z, 0.0f, 0.0f,
          sin_z, cos_z, 0.0f, 0.0f,
          0.0f, 0.0f, 1.0f, 0.0f,
          0.0f, 0.0f, 0.0f, 1.0f
      };

      return *this * rotation_x * rotation_y * rotation_z;
    }


    fan::vec3 get_euler_angles() const{
      _matrix4x4 matrix = *this;
      fan::vec3 euler_angles;

      // Yaw
      euler_angles.z = atan2(matrix[1][0], matrix[0][0]);

      // Pitch
      f32_t sp = -matrix[2][0];
      if (sp <= -1.0f)
        euler_angles.y = -fan::math::two_pi;
      else if (sp >= 1.0f)
        euler_angles.y = fan::math::two_pi;
      else
        euler_angles.y = asin(sp);

      // Roll
      euler_angles.x = atan2(matrix[2][1], matrix[2][2]);

      return euler_angles;
    }

    constexpr _matrix4x4<type_t> inverse() const {
      _matrix4x4<type_t> result = *this;
      _matrix4x4<type_t> identity = _matrix4x4<type_t>(1);

      for (int i = 0; i < 4; ++i) {
        int pivot_row = i;
        for (int j = i + 1; j < 4; ++j) {
          if (std::abs(result[j][i]) > std::abs(result[pivot_row][i])) {
            pivot_row = j;
          }
        }

        std::swap(result.m_array[i], result.m_array[pivot_row]);
        std::swap(identity.m_array[i], identity.m_array[pivot_row]);

        type_t diag_element = result[i][i];
        if (diag_element == 0) {
          fan::throw_error("singular matrix");
        }

        for (int j = 0; j < 4; ++j) {
          result[i][j] /= diag_element;
          identity[i][j] /= diag_element;
        }
        for (int k = 0; k < 4; ++k) {
          if (k != i) {
            type_t factor = result[k][i];
            for (int j = 0; j < 4; ++j) {
              result[k][j] -= factor * result[i][j];
              identity[k][j] -= factor * identity[i][j];
            }
          }
        }
      }
      return identity;
    }
    constexpr _matrix4x4 rotate(const fan::quat& q) const {
      f32_t xx = q.x * q.x;
      f32_t yy = q.y * q.y;
      f32_t zz = q.z * q.z;
      f32_t xy = q.x * q.y;
      f32_t xz = q.x * q.z;
      f32_t yz = q.y * q.z;
      f32_t wx = q.w * q.x;
      f32_t wy = q.w * q.y;
      f32_t wz = q.w * q.z;

      _matrix4x4 rotation{};
      rotation[0][0] = 1.0f - 2.0f * (yy + zz);
      rotation[0][1] = 2.0f * (xy - wz);
      rotation[0][2] = 2.0f * (xz + wy);

      rotation[1][0] = 2.0f * (xy + wz);
      rotation[1][1] = 1.0f - 2.0f * (xx + zz);
      rotation[1][2] = 2.0f * (yz - wx);

      rotation[2][0] = 2.0f * (xz - wy);
      rotation[2][1] = 2.0f * (yz + wx);
      rotation[2][2] = 1.0f - 2.0f * (xx + yy);

      _matrix4x4 matrix{};
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          matrix[i][j] = ((*this)[0][j] * rotation[i][0]) + ((*this)[1][j] * rotation[i][1]) + ((*this)[2][j] * rotation[i][2]);
        }
      }

      matrix[3] = (*this)[3];

      return matrix;
    }

    constexpr fan::vec3 operator*(const fan::vec3& rhs) const {
      fan::vec3 result{};

      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          result[i] += m_array[j][i] * rhs[j];
        }
        result[i] += m_array[3][i];
      }

      return result;
    }

  protected:
    std::array<fan::vec4, 4> m_array;
  };

  template <uint32_t _i, uint32_t _j, typename data_t = f32_t>
  struct matrix2d {

    static constexpr auto rows = _i;
    static constexpr auto columns = _j;

    std::array<std::array<data_t, columns>, rows> data;

    constexpr matrix2d() : data{ 0 } {

    }

    template <typename... Args>
    requires (sizeof...(Args) > 1)
    constexpr matrix2d(Args&&... args) : data{ static_cast<data_t>(std::forward<Args>(args))... } {}

    constexpr matrix2d(data_t value) : matrix2d() {
      for (uint32_t i = 0; i < _i; ++i) {
        data[i][i] = value;
      }
    }

    constexpr auto& operator[](const uintptr_t i) {
      return data[i];
    }

    constexpr auto operator[](const uintptr_t i) const {
      return data[i];
    }

    friend std::ostream& operator<<(std::ostream& os, const matrix2d& matrix)
    {
      for (uintptr_t i = 0; i < matrix.rows; i++) {
        for (uintptr_t j = 0; j < matrix.columns; j++) {
          os << matrix[i][j] << ' ';
        }
        os << '\n';
      }
      return os;
    }

    constexpr matrix2d operator+(const matrix2d& matrix) const {
      matrix2d<_i, _j, data_t> ret;
      for (uint32_t i = 0; i < _i; ++i) {
        for (uint32_t j = 0; j < _j; ++j) {
          ret[i][j] = (*this)[i][j] + matrix[i][j];
        }
      }
      return ret;
    }

    constexpr void operator+=(const matrix2d& matrix) {
      *this = operator+(matrix);
    }

    constexpr matrix2d operator-(const matrix2d& matrix) const {
      matrix2d<_i, _j, data_t> ret;
      for (uint32_t i = 0; i < _i; ++i) {
        for (uint32_t j = 0; j < _j; ++j) {
          ret[i][j] = (*this)[i][j] - matrix[i][j];
        }
      }
      return ret;
    }

    template <uint32_t _i2, uint32_t _j2, typename T>
    constexpr matrix2d operator*(const matrix2d<_i2, _j2, T>& matrix) const {
      matrix2d ret;
      for (uint32_t i = 0; i < _i; ++i) {
        for (uint32_t j = 0; j < _j2; ++j) {
          for (uint32_t k = 0; k < _j; ++k) {
            ret[i][j] += (*this)[i][k] * matrix[k][j];
          }
        }
      }
      return ret;
    }

    constexpr matrix2d transpose() const {
      matrix2d ret;
      for (uint32_t i = 0; i < _i; ++i) {
        for (uint32_t j = 0; j < _j; ++j) {
          ret[j][i] = (*this)[i][j];
        }
      }
      return ret;
    }

    constexpr matrix2d randomize() {
      for (uint32_t i = 0; i < _i; ++i) {
        for (uint32_t j = 0; j < _j; ++j) {
          (*this)[i][j] = fan::random::value_f32(-1, 1);
        }
      }
    }

    // term by term
    constexpr matrix2d hadamard(const matrix2d& matrix) const {
      matrix2d ret;
      for (uint32_t i = 0; i < _i; ++i) {
        for (uint32_t j = 0; j < _j; ++j) {
          ret[i][j] = (*this)[i][j] * matrix[i][j];
        }
      }
      return ret;
    }

    constexpr matrix2d sigmoid() const {
      matrix2d ret;
      for (uint32_t i = 0; i < _i; ++i) {
        for (uint32_t j = 0; j < _j; ++j) {
          ret[i][j] = fan::math::sigmoid((*this)[i][j]);
        }
      }
      return ret;
    }
    constexpr matrix2d sigmoid_derivative() const {
      matrix2d ret;
      for (uint32_t i = 0; i < _i; ++i) {
        for (uint32_t j = 0; j < _j; ++j) {
          ret[i][j] = fan::math::sigmoid_derivative((*this)[i][j]);
        }
      }
      return ret;
    }

    constexpr void zero() {
      for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < columns; ++j) {
          (*this)[i][j] = 0;
        }
      }
    }
  };

  template <typename data_t = f32_t>
  struct runtime_matrix2d {

    using value_type = data_t;

    uint32_t rows;
    uint32_t columns;
    data_t** data = nullptr;

    runtime_matrix2d() = default;

    runtime_matrix2d(uint32_t rows, uint32_t columns) : rows(rows), columns(columns) {
      data = new data_t*[rows];
      for (uint32_t i = 0; i < rows; ++i) {
        data[i] = new data_t[columns];
        std::fill(data[i], data[i] + columns, 0);
      }
    }

    runtime_matrix2d(const runtime_matrix2d& other) : rows(other.rows), columns(other.columns) {
      data = new data_t * [rows];
      for (uint32_t i = 0; i < rows; ++i) {
        data[i] = new data_t[columns];
        std::copy(other.data[i], other.data[i] + columns, data[i]);
      }
    }

    runtime_matrix2d& operator=(const runtime_matrix2d& other) {
      if (this != &other) {
        if (data) {
          for (uint32_t i = 0; i < rows; ++i) {
            delete[] data[i];
          }
          delete[] data;
        }
        rows = other.rows;
        columns = other.columns;
        data = new data_t * [rows];
        for (uint32_t i = 0; i < rows; ++i) {
          data[i] = new data_t[columns];
          std::copy(other.data[i], other.data[i] + columns, data[i]);
        }
      }
      return *this;
    }

    ~runtime_matrix2d() {
      for (uint32_t i = 0; i < rows; ++i) {
        delete[] data[i];
      }
      delete[] data;
      data = nullptr;
    }

    runtime_matrix2d(uint32_t rows, uint32_t columns, data_t value) : runtime_matrix2d(rows, columns) {
      for (uint32_t i = 0; i < std::min(rows, columns); ++i) {
        (*this)[i][i] = value;
      }
    }

    data_t*& operator[](uint32_t i) {
      return data[i];
    }

    const data_t* operator[](uint32_t i) const {
      return data[i];
    }

    friend std::ostream& operator<<(std::ostream& os, const runtime_matrix2d& matrix)
    {
      for (uintptr_t i = 0; i < matrix.rows; i++) {
        for (uintptr_t j = 0; j < matrix.columns; j++) {
          os << matrix[i][j] << ' ';
        }
        os << '\n';
      }
      return os;
    }

    constexpr runtime_matrix2d operator+(const runtime_matrix2d& matrix) const {
      runtime_matrix2d<data_t> ret(rows, columns);
      for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < columns; ++j) {
          ret[i][j] = (*this)[i][j] + matrix[i][j];
        }
      }
      return ret;
    }

    constexpr void operator+=(const runtime_matrix2d& matrix) {
      *this = operator+(matrix);
    }

    constexpr runtime_matrix2d operator-(const runtime_matrix2d& matrix) const {
      runtime_matrix2d<data_t> ret(rows, columns);
      for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < columns; ++j) {
          ret[i][j] = (*this)[i][j] - matrix[i][j];
        }
      }
      return ret;
    }

    template <typename T>
    constexpr runtime_matrix2d operator*(const runtime_matrix2d<T>& matrix) const {
      runtime_matrix2d ret(rows, matrix.columns);
      for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < matrix.columns; ++j) {
          for (uint32_t k = 0; k < columns; ++k) {
            ret[i][j] += (*this)[i][k] * matrix[k][j];
          }
        }
      }
      return ret;
    }

    constexpr runtime_matrix2d transpose() const {
      runtime_matrix2d ret(columns, rows);
      for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < columns; ++j) {
          ret[j][i] = (*this)[i][j];
        }
      }
      return ret;
    }

    constexpr void randomize() {
      for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < columns; ++j) {
          (*this)[i][j] = fan::random::value_f32(-1, 1);
        }
      }
    }

    // term by term
    constexpr runtime_matrix2d hadamard(const runtime_matrix2d& matrix) const {
      runtime_matrix2d ret(rows, columns);
      for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < columns; ++j) {
          ret[i][j] = (*this)[i][j] * matrix[i][j];
        }
      }
      return ret;
    }

    constexpr runtime_matrix2d sigmoid() const {
      runtime_matrix2d ret(rows, columns);
      for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < columns; ++j) {
          ret[i][j] = fan::math::sigmoid((*this)[i][j]);
        }
      }
      return ret;
    }
    constexpr runtime_matrix2d sigmoid_derivative() const {
      runtime_matrix2d ret(rows, columns);
      for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < columns; ++j) {
          ret[i][j] = fan::math::sigmoid_derivative((*this)[i][j]);
        }
      }
      return ret;
    }

    constexpr void zero() {
      for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < columns; ++j) {
          (*this)[i][j] = 0;
        }
      }
    }
  };

  template <typename T>
  inline constexpr auto fan::quaternion<T>::from_matrix(const auto& m) const {

    fan::quaternion<T> q;

    T trace = m[0][0] + m[1][1] + m[2][2];
    if (trace > 0) {
      T s = 0.5 / sqrt(trace + 1.0);
      q.w = 0.25 / s;
      q.x = (m[2][1] - m[1][2]) * s;
      q.y = (m[0][2] - m[2][0]) * s;
      q.z = (m[1][0] - m[0][1]) * s;
    }
    else {
      if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
        T s = 2.0 * sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]);
        q.w = (m[2][1] - m[1][2]) / s;
        q.x = 0.25 * s;
        q.y = (m[0][1] + m[1][0]) / s;
        q.z = (m[0][2] + m[2][0]) / s;
      }
      else if (m[1][1] > m[2][2]) {
        T s = 2.0 * sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]);
        q.w = (m[0][2] - m[2][0]) / s;
        q.x = (m[0][1] + m[1][0]) / s;
        q.y = 0.25 * s;
        q.z = (m[1][2] + m[2][1]) / s;
      }
      else {
        T s = 2.0 * sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]);
        q.w = (m[1][0] - m[0][1]) / s;
        q.x = (m[0][2] + m[2][0]) / s;
        q.y = (m[1][2] + m[2][1]) / s;
        q.z = 0.25 * s;
      }
    }

    return q;
  }
  

  using matrix4x4 = _matrix4x4<cf_t>;
  using matrix4x4ui = _matrix4x4<uintptr_t>;
  using mat4x4 = matrix4x4;
  using mat4x4ui = matrix4x4ui;
  using mat4 = mat4x4;
  using mat2x2 = _matrix2x2<cf_t>;
  using mat2x2ui = _matrix2x2<uintptr_t>;
  using mat2 = mat2x2;
  using mat2ui = mat2x2ui;

}