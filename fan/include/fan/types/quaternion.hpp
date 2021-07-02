#pragma once

#include <fan/types/matrix.hpp>
#include <fan/types/vector.hpp>

namespace fan {

	template <typename T>
	struct quaternion : public _vec4<T> {

		using value_type = T;

		using inherited_type = fan::_vec4<T>;

		template <typename _Ty>
		using inherited_type_t = fan::_vec4<_Ty>;

		constexpr quaternion() : fan::_vec4<T>() {}

		template <typename _Ty>
		constexpr quaternion(_Ty t, _Ty x_, _Ty y_, _Ty z_) 
		{
			this->x = t;
			this->y = x_;
			this->z = y_;
			this->w = z_;
		}

		template <typename _Ty, typename _Ty2>
		constexpr quaternion(_Ty scalar, const fan::_vec3<_Ty2>& vector)
		{
			this->x = scalar;
			this->y = vector.x;
			this->z = vector.y;
			this->w = vector.z;
		}

		template <typename _Ty, typename _Ty2>
		constexpr quaternion(const fan::vec4& vector) : fan::_vec4<T>(vector) {}

		template <typename _Ty>
		constexpr quaternion(const fan::_vec4<_Ty>& vector) : fan::_vec4<T>(vector) {}

		template <typename _Ty>
		constexpr quaternion(fan::_vec4<_Ty>&& vector) : fan::_vec4<T>(std::move(vector)) {}

	#ifdef ASSIMP_API

		constexpr quaternion(const aiQuaternion& quat) : fan::_vec4<T>(quat.x, quat.y, quat.z, quat.w) {}

	#endif

		template <typename _Ty>
		constexpr quaternion<value_type> operator+(const quaternion<_Ty>& quat) const
		{
			return inherited_type::operator+(inherited_type_t<_Ty>(quat.begin(), quat.end()));
		}

		template <typename _Ty>
		constexpr quaternion<value_type>& operator+=(const quaternion<_Ty>& quat) 
		{
			inherited_type::operator+=(inherited_type_t<_Ty>(quat.begin(), quat.end()));
			return *this;
		}

		constexpr quaternion<value_type> operator-() const {
			return quaternion<value_type>(-(*this)[0], -(*this)[1], -(*this)[2], -(*this)[3]);
		}

		template <typename _Ty>
		constexpr quaternion<value_type> operator-(const quaternion<_Ty>& quat) const
		{
			return inherited_type::operator-(inherited_type_t<_Ty>(quat.begin(), quat.end()));
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
			return fan::_vec4<T>::operator*(value);
		}

		template <typename _Ty>
		constexpr quaternion<value_type> operator*(const quaternion<_Ty>& quat) const
		{
			fan::vec3 vector(this->begin() + 1, this->end());
			fan::vec3 vector2(quat.begin() + 1, quat.end());

			return quaternion<value_type>(
				this->operator[](0) * quat[0] - vector.dot(vector2),
				vector2 * this->operator[](0) + vector * quat[0] + vector.cross(vector2)
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
			return fan::_vec4<T>::operator/(value);
		}

		template <typename _Ty>
		constexpr quaternion<value_type> operator/(const quaternion<_Ty>& quat) const
		{
			return ((*this) * quat.conjucate()) / std::pow(quat.length(), 2);
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

		constexpr quaternion<value_type> conjucate() const {
			return quaternion<value_type>(this->operator[](0), -fan::vec3(this->begin() + 1, this->end));
		}

		constexpr auto length() const {
			return std::sqrt(pow(this->operator[](0), 2) + pow(this->operator[](1), 2) + pow(this->operator[](2), 2) + pow(this->operator[](3), 2));
		}

		template <typename _Ty>
		constexpr auto dot(const quaternion<_Ty>& quat) const {
			return (*this)[0] * quat[0] +
				   (*this)[1] * quat[1] +
				   (*this)[2] * quat[2] +
				   (*this)[3] * quat[3];
		}

		constexpr auto normalize() const {
			value_type length = std::sqrt(this->dot(*this));

			return quaternion(
				(*this)[0] / length,
				(*this)[1] / length,
				(*this)[2] / length,
				(*this)[3] / length
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
			f_t theta = theta_0 * t ;
			f_t sin_theta = sin(theta);    
			f_t sin_theta_0 = sin(theta_0);

			f_t s0 = cos(theta) - dot * sin_theta / sin_theta_0;
			f_t s1 = sin_theta / sin_theta_0;

			return (v0 * s0) + (v1 * s1);
		}

	};

	template <typename T>
	static constexpr auto mix(T x, T y, T a) {
		return static_cast<T>(x) * (static_cast<T>(1) - a) + static_cast<T>(y) * a;
	}

	static constexpr auto mix(const fan::vec3& x, const fan::vec3& y, f_t t) {
		return x * (static_cast<f_t>(1) - t) + y * t;
	}

	using quat = quaternion<f32_t>;
}