#pragma once

#include <fan/types/types.h>

#include <fan/math/math.h>

#if defined(fan_3d)
  #include <assimp/quaternion.h>
#endif

#include <cmath>

import fan.types.vector;

namespace fan {

  template <typename type_t>
  struct _matrix4x4;

	template <typename T>
	struct quaternion : public fan::vec4_wrap_t<T> {

		using value_type = T;

		using inherited_type = fan::vec4_wrap_t<T>;

		template <typename _Ty>
		using inherited_type_t = fan::vec4_wrap_t<_Ty>;

		constexpr quaternion() : fan::quaternion<T>(1, T{ 0 }, T{ 0 }, T{ 0 }) {}

  #if defined(fan_3d)
    constexpr quaternion(const aiQuaternion& aq) {
      this->w = aq.w;
      this->x = aq.x;
      this->y = aq.y;
      this->z = aq.z;
    }
    operator aiQuaternion() const {
      return aiQuaternion(this->w, this->x, this->y, this->z);
    }
  #endif  

    quaternion(const _matrix4x4<T>& m);

		constexpr quaternion(auto scalar, auto x_, auto y_, auto z_)
		{
			this->w = scalar;
			this->x = x_;
			this->y = y_;
			this->z = z_;
		}

		template <typename _Ty, typename _Ty2>
		constexpr quaternion(_Ty scalar, const fan::vec3_wrap_t<_Ty2>& vector)
		{
			this->w = scalar;
			this->x = vector.x;
			this->y = vector.y;
			this->z = vector.z;
		}

		template <typename _Ty>
		constexpr quaternion(const fan::vec4_wrap_t<_Ty>& vector) : fan::vec4_wrap_t<T>(vector.x, vector.y, vector.z, vector.w) {}

		template <typename _Ty>
		constexpr quaternion<value_type> operator+(const quaternion<_Ty>& quat) const
		{
      quaternion<value_type> result(this->w + quat.w, this->x + quat.x, this->y + quat.y, this->z + quat.z);
      double magnitude = sqrt(result.w * result.w + result.x * result.x + result.y * result.y + result.z * result.z);
      return quaternion<value_type>(result.w / magnitude, result.x / magnitude, result.y / magnitude, result.z / magnitude);
     // return inherited_type::operator+(inherited_type(quat.x, quat.y, quat.z, quat.w));
		}

		template <typename _Ty>
		constexpr quaternion<value_type>& operator+=(const quaternion<_Ty>& quat) 
		{
			inherited_type::operator+=(inherited_type_t<_Ty>((*this)[0], (*this)[1], (*this)[2], (*this)[3]));
			return *this;
		}

		constexpr quaternion<value_type> operator-() const {
			return quaternion<value_type>(-this->w, -this->x, -this->y, -this->z);
		}

		template <typename _Ty>
		constexpr quaternion<value_type> operator-(const quaternion<_Ty>& quat) const
		{
      return inherited_type::operator-(inherited_type(quat.x, quat.y, quat.z, quat.w));
      //return quaternion<value_type>();/*inherited_type::operator-(inherited_type_t<_Ty>(quat.begin(), quat.end()));*/
		}

		template <typename _Ty>
		constexpr quaternion<value_type>& operator-=(const quaternion<_Ty>& quat) 
		{
			inherited_type::operator-=(inherited_type_t<_Ty>(quat.begin(), quat.end()));
			return *this;
		}

		template <typename _Ty>
		constexpr quaternion<value_type> operator*(_Ty value) const
		{
			return fan::vec4_wrap_t<T>::operator*(value);
		}

		constexpr fan::vec3_wrap_t<value_type> operator*(const fan::vec3_wrap_t<value_type>& v) const {
      f32_t vx = v.x;
      f32_t vy = v.y;
      f32_t vz = v.z;
      return fan::vec3_wrap_t<value_type>(
        vx * (1.0f - 2.0f * (this->y * this->y + this->z * this->z)) +
        2.0f * (this->x * vy - this->w * vz),

        vy * (1.0f - 2.0f * (this->x * this->x + this->z * this->z)) +
        2.0f * (this->y * vx + this->w * vz),

        vz * (1.0f - 2.0f * (this->x * this->x + this->y * this->y)) +
        2.0f * (this->z * vx - this->w * vy)
      );
		}


		template <typename _Ty>
		constexpr quaternion<value_type> operator*(const quaternion<_Ty>& quat) const
		{
			fan::vec3 vector(this->x, this->y, this->z);
			fan::vec3 vector2(quat.x, quat.y, quat.z);

			return quaternion<value_type>(
				this->w * quat.w - vector.dot(vector2),
				vector2 * this->w + vector * quat.w + vector.cross(vector2)
			);
		}

		template <typename _Ty>
		constexpr quaternion<value_type>& operator*=(const quaternion<_Ty>& quat) 
		{
			return (*this) = this->operator*(quat);
		}


		template <typename _Ty>
		constexpr quaternion<value_type>& operator*=(_Ty value) 
		{
			return (*this) = this->operator*(value);
		}

		template <typename _Ty>
		constexpr quaternion<value_type> operator/(_Ty value) const
		{
			return fan::vec4_wrap_t<T>::operator/(value);
		}

		template <typename _Ty>
		constexpr quaternion<value_type> operator/(const quaternion<_Ty>& quat) const
		{
			return ((*this) * quat.conjugate()) / std::pow(quat.length(), 2);
		}

		template <typename _Ty>
		constexpr quaternion<value_type>& operator/=(_Ty value) 
		{
			return (*this) = this->operator/(value);
		}

		template <typename _Ty>
		constexpr quaternion<value_type>& operator/=(const quaternion<_Ty>& quat) 
		{
			return (*this) = this->operator/(quat);
		}

		constexpr quaternion<value_type> conjugate() const {
			return quaternion<value_type>(this->w, -this->x, -this->y, -this->z);
		}

		constexpr auto length() const {
			return std::sqrt(pow(this->w, 2) + pow(this->x, 2) + pow(this->y, 2) + pow(this->z, 2));
		}

		template <typename _Ty>
		constexpr auto dot(const quaternion<_Ty>& quat) const {
			return 
         this->w * quat.w +
				 this->x * quat.x +
				 this->y * quat.y +
				 this->z * quat.z;
		}

		constexpr auto normalize() const {
			value_type length = std::sqrt(this->dot(*this));

			return quaternion(
				this->w / length,
				this->x / length,
				this->y / length,
				this->z / length
			);
		}

    constexpr quaternion inverse() const {
      T magnitude_squared = this->w * this->w +
        this->x * this->x +
        this->y * this->y +
        this->z * this->z;

      if (magnitude_squared < std::numeric_limits<T>::epsilon()) {
        return quaternion();
      }

      return quaternion(
         this->w / magnitude_squared,
        -this->x / magnitude_squared,
        -this->y / magnitude_squared,
        -this->z / magnitude_squared
      );
    }

    template <typename _Ty>
    static constexpr quaternion<T> slerp(const quaternion<_Ty>& q0, const quaternion<_Ty>& q1, f_t t) {

      auto v0 = q0;
      auto v1 = q1;

      f_t dot = v0.dot(v1);

      if (dot < 0) {
        v1 = -v1;
        dot = -dot;
      }

      constexpr auto dot_threshold = 0.9995;

      if (dot > dot_threshold) {
        return quaternion<T>(v0 + (v1 - v0) * t);
      }

      f_t theta_0 = acos(dot);
      f_t theta = theta_0 * t;
      f_t sin_theta = sin(theta);
      f_t sin_theta_0 = sin(theta_0);

      f_t s0 = cos(theta) - dot * sin_theta / sin_theta_0;
      f_t s1 = sin_theta / sin_theta_0;

      return (v0 * s0) + (v1 * s1);
    }

    // radians
    static quaternion<T> from_axis_angle(const fan::vec3& axis, float angle) {
      quaternion<T> q;
      float sinHalfAngle = sin(angle / 2.0f);
      q.x = axis.x * sinHalfAngle;
      q.y = axis.y * sinHalfAngle;
      q.z = axis.z * sinHalfAngle;
      q.w = cos(angle / 2.0f);
      return q;
    }

    void to_axis_angle(fan::vec3& axis, value_type& angle) const {
      quaternion<T> qn = normalize();
      angle = 2.0f * acos(qn.w);

      float s = sqrt(1.0f - qn.w * qn.w);
      if (s < 0.001f) {
        axis.x = qn.x;
        axis.y = qn.y;
        axis.z = qn.z;
      }
      else {
        axis.x = qn.x / s;
        axis.y = qn.y / s;
        axis.z = qn.z / s;
      }
    }
    void to_angles(fan::vec3& angles) {
      const quaternion& q = *this;

      f32_t sinr_cosp = 2.0f * (q.w * q.x + q.y * q.z);
      f32_t cosr_cosp = 1.0f - 2.0f * (q.x * q.x + q.y * q.y);
      angles.x = std::atan2(sinr_cosp, cosr_cosp);

      f32_t sinp = 2.0f * (q.w * q.y - q.z * q.x);
      if (std::abs(sinp) >= 1.0f)
        angles.y = std::copysign(fan::math::pi / 2.0f, sinp);
      else
        angles.y = std::asin(sinp);

      f32_t siny_cosp = 2.0f * (q.w * q.z + q.x * q.y);
      f32_t cosy_cosp = 1.0f - 2.0f * (q.y * q.y + q.z * q.z);
      angles.z = std::atan2(siny_cosp, cosy_cosp);
    }

    static quaternion<T> from_angles(const fan::vec3& angles) {
      f32_t cx = cos(-angles.x * 0.5f);
      f32_t sx = sin(-angles.x * 0.5f);
      f32_t cy = cos(-angles.y * 0.5f);
      f32_t sy = sin(-angles.y * 0.5f);
      f32_t cz = cos(-angles.z * 0.5f);
      f32_t sz = sin(-angles.z * 0.5f);

      quaternion<T> q;
      q.w = cx * cy * cz + sx * sy * sz;
      q.x = sx * cy * cz - cx * sy * sz;
      q.y = cx * sy * cz + sx * cy * sz;
      q.z = cx * cy * sz - sx * sy * cz;

      return q.normalize();
    }

    fan::vec3 to_euler() const {
      fan::vec3 angles;
      to_angles(angles);
      return angles;
    }

	};
  using quat = quaternion<f32_t>;

	template <typename T>
	static constexpr auto mix(T x, T y, T a) {
		return x * (1.f - a) + y * a;
	}

	static constexpr auto mix(const fan::vec3& x, const fan::vec3& y, f_t t) {
		return x * (1.f - t) + y * t;
	}
  static constexpr auto mix(const fan::vec4& a, const fan::vec4& b, f_t t) {
    return a * (1.f - t) + b * t;
  }
}