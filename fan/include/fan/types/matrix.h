#pragma once

#include <fan/types/vector.h>
#include <fan/types/quaternion.h>

#include <type_traits>

#include <iostream>
#include <exception>
#include <cstring>

namespace fan {

  template <typename type_t>
  struct _matrix2x2 {

    using value_type = fan::_vec2<type_t>;

    _matrix2x2() = default;

    template <typename T>
    constexpr _matrix2x2(T x, T y, T z, T w) : m_array{ fan::vec2{x, y}, fan::vec2{z, w} } {}

    template <typename T, typename T2>
    constexpr _matrix2x2(const fan::_vec2<T>& v, const fan::_vec2<T2>& v2) : m_array{ v.x, v.y, v2.x, v2.y } {}

    constexpr _matrix2x2 operator+(const fan::_vec2<type_t>& v) const {
      return _matrix2x2(m_array[0][0] + v[0], m_array[0][1] + v[1], m_array[1][0] + v[0], m_array[1][1] + v[1]);
    }

    template <typename T>
    constexpr fan::_vec2<T> operator*(const fan::_vec2<T>& v) const {
      return fan::_vec2<T>(m_array[0][0] * v.x + m_array[0][1] * v.y, m_array[1][0] * v.x + m_array[1][1] * v.y);
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

    using value_type = fan::_vec4<type_t>;

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

    constexpr _matrix4x4<type_t> operator*(const _matrix4x4<type_t>& matrix) const {
      _matrix4x4<type_t> result;

      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
          int num = 0;
          for (int k = 0; k < 4; k++) {
            num += m_array[i][k] * matrix[k][j];
          }
          result[i][j] = num;
        }
      }
      return result;
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
			return fan::vec3((*this)[0][0], (*this)[1][1], (*this)[2][2]);
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

  protected:
    std::array<fan::vec4, 4> m_array;
  };

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