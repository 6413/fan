#pragma once

#include <iostream>
#include <algorithm>
#include <numeric>
#include <array>
#include <string>

#include <fan/types/types.hpp>
#include <fan/math/math.hpp>

#include <box2d/b2_math.h>

namespace fan {

	template <class T, class... Ts>
	struct is_any : std::bool_constant<(std::is_same_v<T, Ts> || ...)> {};

	template <typename _Ty>
	class _vec3;

	template <typename _Ty>
	class _vec4;

	template <typename type_t, std::size_t n>
	struct list;

	template <typename _Ty>
	class _vec2 {
	public:
		_Ty x, y;

		using value_type = _Ty;
		using vec_t = _vec2<_Ty>;

		constexpr _vec2() : x(0), y(0) {}
		constexpr _vec2(_Ty value) : x(value), y(value) {}
		constexpr _vec2(_Ty x_, _Ty y_) : x(x_), y(y_) {}

		template <typename T>
		constexpr _vec2(const _vec2<T>& vec) : x(vec.x), y(vec.y) {}

		template <typename T>
		constexpr _vec2(_vec2<T>&& vec) : x(std::move(vec.x)), y(std::move(vec.y)) {}

		template <typename T>
		constexpr _vec2(const _vec3<T>& vec) : x(vec.x), y(vec.y) {}

		template <typename T>
		constexpr _vec2(_vec3<T>&& vec) : x(std::move(vec.x)), y(std::move(vec.y)) {}

		template <typename T>
		constexpr _vec2(const std::array<T, 2>& _array) : x(_array[0]), y(_array[1]) {}

		template <typename T>
		constexpr _vec2(std::array<T, 2>&& _array) : x(std::move(_array[0])), y(std::move(_array[1])) {}

		_vec2(b2Vec2 vector) : x(vector.x), y(vector.y) {}

		inline b2Vec2 b2() const {
			return b2Vec2(x, y);
		}

		template <typename type_t>
		constexpr _vec2(const type_t* begin, const type_t* end) 
		{
			std::copy(begin, end, this->begin());
		}

		template <typename T>
		constexpr _vec2<value_type>& operator=(const _vec2<T>& vector) {
			this->x = vector.x;
			this->y = vector.y;
			return *this;
		}

		constexpr _Ty& operator[](uint64_t idx) { return !idx ? x : y; }
		constexpr _Ty operator[](uint64_t idx) const { return !idx ? x : y; }


		template <typename _Type>
		constexpr _vec2<value_type> operator%(_Type single_value) const {
			return _vec2<value_type>(std::fmod(this->x, (decltype(this->x))single_value), std::fmod(this->y, (decltype(this->y))single_value));
		}

		template <typename _Type>
		constexpr _vec2<value_type> operator%(const _vec2<_Type>& vector) const {
			return _vec2<value_type>(std::fmod(this->x, vector.x), std::fmod(this->y, vector.y));
		}

		template <typename _Type>
		constexpr bool operator==(const _vec2<_Type>& vector) const {
			return this->x == vector.x && this->y == vector.y;
		}

		constexpr bool operator==(value_type single_value) const {
			return this->x == single_value && this->y == single_value;
		}

		constexpr bool operator!() const {
			return !this->x && !this->y;
		}

		template <typename _Type>
		constexpr bool operator!=(const _vec2<_Type>& vector) const {
			return this->x != vector.x || this->y != vector.y;
		}

		constexpr bool operator!=(value_type single_value) const {
			return this->x != single_value || this->y != single_value;
		}

		template <typename T>
		constexpr bool operator!=(const std::array<T, 2>& array) {
			return this->x != array[0] || this->y != array[1];
		}

		template <typename _Type>
		constexpr bool operator<(const _vec2<_Type>& vector) const {
			 return (x == vector.x) ? (y < vector.y) : (x < vector.x); // AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
		}

		template <typename _Type>
		constexpr bool operator<=(const _vec2<_Type>& vector) const {
			return this->x <= vector.x && this->y <= vector.y;
		}

		// math operators
		template <typename _Type>
		constexpr _vec2<value_type> operator+(const _vec2<_Type>& vector) const
		{
			return _vec2<value_type>(this->x + vector.x, this->y + vector.y);
		}

		template <typename T>
		constexpr _vec2<value_type> operator+(const list<T, 2>& list_) const
		{
			return _vec2<value_type>(this->x + list_[0], this->y + list_[1]);
		}

		template <typename _Type>
		constexpr _vec2<value_type> operator+(_Type single_value) const
		{
			return _vec2<value_type>(this->x + single_value, this->y + single_value);
		}

		template <typename _Type>
		constexpr _vec2<value_type>& operator+=(const _vec2<_Type>& vector)
		{
			this->x += vector.x;
			this->y += vector.y;
			return *this;
		}

		template <typename _Type>
		constexpr _vec2<value_type>& operator+=(_Type single_value)
		{
			this->x += single_value;
			this->y += single_value;
			return *this;
		}

		constexpr _vec2<value_type> operator-() const
		{
			return _vec2<value_type>(-this->x, -this->y);
		}

		template <typename _Type>
		constexpr _vec2<value_type> operator-(const _vec2<_Type>& vector) const
		{
			return _vec2<value_type>(this->x - vector.x, this->y - vector.y);
		}

		template <typename T>
		constexpr _vec2<value_type> operator-(const list<T, 2>& list_) const
		{
			return _vec2<value_type>(this->x - list_[0], this->y - list_[1]);
		}

		template <typename _Type>
		constexpr _vec2<value_type> operator-(_Type single_value) const
		{
			return _vec2<value_type>(this->x - single_value, this->y - single_value);
		}

		template <typename _Type>
		constexpr _vec2<value_type>& operator-=(const _vec2<_Type>& vector)
		{
			this->x -= vector.x;
			this->y -= vector.y;
			return *this;
		}

		template <typename _Type>
		constexpr _vec2<value_type>& operator-=(_Type single_value)
		{
			this->x -= single_value;
			this->y -= single_value;
			return *this;
		}


		template <typename _Type>
		constexpr _vec2<value_type> operator*(const _vec2<_Type>& vector) const
		{
			return _vec2<value_type>(this->x * vector.x, this->y * vector.y);
		}

		template <typename T>
		constexpr _vec2<value_type> operator*(const list<T, 2>& list_) const
		{
			return _vec2<value_type>(this->x * list_[0], this->y * list_[1]);
		}

		template <typename _Type>
		constexpr _vec2<value_type> operator*(_Type single_value) const
		{
			return _vec2<value_type>(this->x * single_value, this->y * single_value);
		}

		template <typename _Type>
		constexpr _vec2<value_type>& operator*=(const _vec2<_Type>& vector)
		{
			this->x *= vector.x;
			this->y *= vector.y;
			return *this;
		}

		template <typename _Type>
		constexpr _vec2<value_type>& operator*=(_Type single_value)
		{
			this->x *= single_value;
			this->y *= single_value;
			return *this;
		}


		template <typename _Type>
		constexpr _vec2<value_type> operator/(const _vec2<_Type>& vector) const
		{
			return _vec2<value_type>(this->x / vector.x, this->y / vector.y);
		}

		template <typename T>
		constexpr _vec2<value_type> operator/(const list<T, 2>& list_) const
		{
			return _vec2<value_type>(this->x / list_[0], this->y / list_[1]);
		}

		template <typename _Type>
		constexpr _vec2<value_type> operator/(_Type single_value) const
		{
			return _vec2<value_type>(this->x / single_value, this->y / single_value);
		}

		template <typename _Type>
		constexpr _vec2<value_type>& operator/=(const _vec2<_Type>& vector)
		{
			this->x /= vector.x;
			this->y /= vector.y;
			return *this;
		}

		template <typename _Type>
		constexpr _vec2<value_type>& operator/=(_Type single_value)
		{
			this->x /= single_value;
			this->y /= single_value;
			return *this;
		}

		constexpr auto gfne() const noexcept {
			for (uint_t i = 0; i < size(); i++) {
				if (this->operator[](i)) {
					return this->operator[](i);
				}
			}
			return _Ty();
		}

		constexpr vec_t floored() const { return vec_t(floor(x), floor(y)); }
		constexpr vec_t floored(_Ty value) const { return vec_t(floor(x / value), floor(y / value)); }

		constexpr vec_t rounded() const { return vec_t(std::round(x), std::round(y)); }

		constexpr vec_t ceiled() const { return vec_t(ceil(x), ceil(y)); }
		constexpr vec_t ceiled(_Ty value) const { return vec_t(ceil(x / value), ceil(y / value)); }

		constexpr _Ty min() const { return std::min(x, y); }
		constexpr _Ty max() const { return std::max(x, y); }
		constexpr vec_t abs() const { return vec_t(fan::abs(x), fan::abs(y)); }

		constexpr vec_t clamp(_Ty min, _Ty max) const {
			return vec_t(std::clamp(x, min, max), std::clamp(y, min, max));
		}

		template <fan::is_not_arithmetic_t T>
		constexpr vec_t clamp(const T min, T max) const {
			return vec_t(std::clamp(x, min[0], max[0]), std::clamp(y, min[1], max[1]));
		}

		constexpr bool isnan() const {
			return std::isnan(x) || std::isnan(y);
		}

		constexpr auto length() const {
			return fan_2d::vector_length(*this);
		}

		constexpr fan::_vec2<_Ty> normalize() const {
			return fan_2d::normalize(*this);
		}

		auto begin() const { return &x; }
		auto end() const { return begin() + size(); }
		auto data() const { return begin(); }

		auto begin() { return &x; }
		auto end() { return begin() + size(); }
		auto data() { return begin(); }

		static constexpr uint_t size() { return 2; }
		constexpr void print() const { std::cout << x << " " << y << std::endl; }

		template <typename T>
		constexpr auto dot(T vector) const {
			return fan_2d::dot(*this, vector);
		}

		template <typename T>
		friend std::ostream& operator<<(std::ostream& os, const _vec2<T>& vector);

		std::string to_string() const {
			return std::string(std::to_string(this->x) + ' '+ std::to_string(this->y));
		}

	};

	template <typename _Ty = f32_t>
	class _vec3 {
	public:
		_Ty x, y, z;

		using value_type = _Ty;
		using vec_t = _vec3<_Ty>;

		constexpr _vec3() : x(0), y(0), z(0) {}
		constexpr _vec3(_Ty x, _Ty y, _Ty z) : x(x), y(y), z(z) {}
		constexpr _vec3(_Ty value) : x(value), y(value), z(value) {}

		template <typename type_t>
		constexpr _vec3(const _vec3<type_t>& vec) : x(vec.x), y(vec.y), z(vec.z) {}

		template <typename type_t>
		constexpr _vec3(_vec3<type_t>&& vec) : x(std::move(vec.x)), y(std::move(vec.y)), z(std::move(vec.z)) {}

		template <typename type_t>
		constexpr _vec3(const _vec2<type_t>& vec) : x(vec.x), y(vec.y), z(0) {}

		template <typename type_t>
		constexpr _vec3(_vec2<type_t>&& vec) : x(std::move(vec.x)), y(std::move(vec.y)), z(0) {}

		template <typename type_t>
		constexpr _vec3(const std::array<type_t, 3>& array) : x(array[0]), y(array[1]), z(array[2]) {}

		template <typename type_t>
		constexpr _vec3(std::array<type_t, 3>&& array) : x(std::move(array[0])), y(std::move(array[1])), z(std::move(array[2])) {}

		template <typename type_t>
		constexpr _vec3(const _vec4<type_t>& vec) : x(vec.x), y(vec.y), z(vec.z) {}

		template <typename type_t>
		constexpr _vec3(_vec4<type_t>&& vec) : x(std::move(vec.x)), y(std::move(vec.y)), z(std::move(vec.z)) {}

		template <typename type_t>
		constexpr _vec3(const type_t* begin, const type_t* end) 
		{
			std::copy(begin, end, this->begin());
		}

	#ifdef ASSIMP_API
		constexpr _vec3(const aiVector3D& vector) : x(vector.x), y(vector.y), z(vector.z) {}
	#endif

		constexpr _Ty& operator[](uint64_t idx) { return !idx ? x : idx == 1 ? y : z; }
		constexpr _Ty operator[](uint64_t idx) const { return !idx ? x : idx == 1 ? y : z; }

		template <typename _Type>
		constexpr _vec3<value_type> operator%(_Type single_value) const {
			return _vec3<value_type>(fmod_dr(this->x, single_value), fmod_dr(this->y, single_value), fmod_dr(this->z, single_value));
		}

		template <typename _Type>
		constexpr bool operator==(const _vec3<_Type>& vector) const {
			return this->x == vector.x && this->y == vector.y && this->y == vector.z;
		}

		template <typename _Type>
		constexpr bool operator==(_Type single_value) const {
			return this->x == single_value && this->y == single_value && this->z == single_value;
		}

		constexpr bool operator!() const {
			return !this->x && !this->y && !this->z;
		}

		template <typename _Type>
		constexpr bool operator!=(const _vec3<_Type>& vector) const {
			return this->x != vector.x || this->y != vector.y || this->z != vector.z;
		}

		template <typename _Type>
		constexpr bool operator!=(_Type single_value) const {
			return this->x != single_value && this->y != single_value;
		}

		// math operators
		template <typename _Type>
		constexpr _vec3<value_type> operator+(const _vec3<_Type>& vector) const
		{
			return _vec3<value_type>(this->x + vector.x, this->y + vector.y, this->z + vector.z);
		}

		template <typename _Type>
		constexpr _vec3<value_type> operator+(_Type single_value) const
		{
			return _vec3<value_type>(this->x + single_value, this->y + single_value, this->z + single_value);
		}

		template <typename _Type>
		constexpr _vec3<value_type>& operator+=(const _vec3<_Type>& vector)
		{
			this->x += vector.x;
			this->y += vector.y;
			this->z += vector.z;
			return *this;
		}

		template <typename _Type>
		constexpr _vec3<value_type>& operator+=(_Type single_value)
		{
			this->x += single_value;
			this->y += single_value;
			this->z += single_value;
			return *this;
		}

		constexpr _vec3<value_type> operator-() const
		{
			return _vec3<value_type>(-this->x, -this->y, -this->z);
		}

		template <typename _Type>
		constexpr _vec3<value_type> operator-(const _vec3<_Type>& vector) const
		{
			return _vec3<value_type>(this->x - vector.x, this->y - vector.y, this->z - vector.z);
		}

		template <typename _Type>
		constexpr _vec3<value_type> operator-(_Type single_value) const
		{
			return _vec3<_Type>(this->x - single_value, this->y - single_value, this->z - single_value);
		}

		template <typename _Type>
		constexpr _vec3<value_type>& operator-=(const _vec3<_Type>& vector)
		{
			this->x -= vector.x;
			this->y -= vector.y;
			this->z -= vector.z;
			return *this;
		}

		template <typename _Type>
		constexpr _vec3<value_type>& operator-=(_Type single_value)
		{
			this->x -= single_value;
			this->y -= single_value;
			this->z -= single_value;
			return *this;
		}


		template <typename _Type>
		constexpr _vec3<value_type> operator*(const _vec3<_Type>& vector) const
		{
			return _vec3<value_type>(this->x * vector.x, this->y * vector.y, this->z * vector.z);
		}

		template <typename _Type>
		constexpr _vec3<value_type> operator*(_Type single_value) const
		{
			return _vec3<value_type>(this->x * single_value, this->y * single_value, this->z * single_value);
		}

		template <typename _Type>
		constexpr _vec3<value_type>& operator*=(const _vec3<_Type>& vector)
		{
			this->x *= vector.x;
			this->y *= vector.y;
			this->z *= vector.z;
			return *this;
		}

		template <typename _Type>
		constexpr _vec3<value_type>& operator*=(_Type single_value)
		{
			this->x *= single_value;
			this->y *= single_value;
			this->z *= single_value;
			return *this;
		}


		template <typename _Type>
		constexpr _vec3<value_type> operator/(const _vec3<_Type>& vector) const
		{
			return _vec3<value_type>(this->x / vector.x, this->y / vector.y, this->z / vector.z);
		}

		template <typename _Type>
		constexpr _vec3<value_type> operator/(_Type single_value) const
		{
			return _vec3<value_type>(this->x / single_value, this->y / single_value, this->z / single_value);
		}

		template <typename _Type>
		constexpr _vec3<value_type>& operator/=(const _vec3<_Type>& vector)
		{
			this->x /= vector.x;
			this->y /= vector.y;
			this->z /= vector.z;
			return *this;
		}

		template <typename _Type>
		constexpr _vec3<value_type>& operator/=(_Type single_value)
		{
			this->x /= single_value;
			this->y /= single_value;
			this->z /= single_value;
			return *this;
		}

		constexpr vec_t floored() const { return vec_t(floor(x), floor(y), floor(z)); }
		constexpr vec_t floored(_Ty value) const { return vec_t(floor(x / value), floor(y / value), floor(z / value)); }
		constexpr vec_t ceiled() const { return vec_t(ceil(x), ceil(y), ceil(z)); }
		constexpr vec_t rounded() const { return vec_t(round(x), round(y), round(z)); }

		constexpr _Ty min() const { return std::min({ x, y, z }); }
		constexpr _Ty max() const { return std::max({ x, y, z }); }
		constexpr vec_t abs() const { return vec_t(fan::abs(x), fan::abs(y), fan::abs(z)); }
		constexpr vec_t clamp(_Ty min, _Ty max) const {
			return vec_t(std::clamp(x, min, max), std::clamp(y, min, max), std::clamp(z, min, max));
		}

		template <fan::is_not_arithmetic_t T>
		constexpr vec_t clamp(const T min, T max) const {
			return vec_t(std::clamp(x, min[0], max[0]), std::clamp(y, min[1], max[1]), std::clamp(z, min[2], max[2]));
		}

		constexpr bool isnan() const {
			return std::isnan(x) || std::isnan(y) || std::isnan(z);
		}

		auto begin() const { return &x; }
		auto end() const { return begin() + size(); }
		auto data() const { return begin(); }

		auto begin() { return &x; }
		auto end() { return begin() + size(); }
		auto data() { return begin(); }


		static constexpr uint_t size() { return 3; }

		constexpr void print() const { std::cout << x << " " << y << " " << z << std::endl; }

		template <typename T>
		constexpr auto dot(const fan::_vec3<T>& vector) const {
			return fan_3d::dot(*this, vector);
		}

		template <typename T>
		constexpr auto cross(const fan::_vec3<T>& vector) const {
			return fan::cross(*this, vector);
		}

		constexpr f_t length() const {
			return fan_3d::vector_length(*this);
		}

		constexpr fan::_vec3<_Ty> normalize() const {
			return fan_3d::normalize(*this);
		}

		template <typename T>
		friend std::ostream& operator<<(std::ostream& os, const _vec3<T>& vector);

	};

	template <typename _Ty = f32_t>
	class _vec4 {
	public:
		_Ty x, y, z, w;

		using value_type = _Ty;
		using vec_t = _vec3<_Ty>;

		constexpr _vec4() : x(0), y(0), z(0), w(0) {}
		constexpr _vec4(_Ty x, _Ty y, _Ty z, _Ty w) : x(x), y(y), z(z), w(w) {}
		constexpr _vec4(_Ty value) : x(value), y(value), z(value), w(value) {}

		template <typename type_t, typename type2>
		constexpr _vec4(const _vec2<type_t>& a, const _vec2<type2>& b) : x(a.x), y(a.y), z(b.x), w(b.y) {}

		template <typename type_t, typename type2>
		constexpr _vec4(_vec2<type_t>&& a, _vec2<type2>&& b) : x(std::move(a.x)), y(std::move(a.y)), z(std::move(b.x)), w(std::move(b.y)) {}

		template <typename type_t, typename type2>
		constexpr _vec4(const _vec3<type_t> vector, type2 value) : x(vector.x), y(vector.y), z(vector.z), w(value) {}

		template <typename type_t, typename type2>
		constexpr _vec4(_vec3<type_t>&& vector, type2&& value) : x(std::move(vector.x)), y(std::move(vector.y)), z(std::move(vector.z)), w(std::move(value)) {}

		template <typename type_t>
		constexpr _vec4(const _vec4<type_t>& vector) : x(vector.x), y(vector.y), z(vector.z), w(vector.w) {}

		template <typename type_t>
		constexpr _vec4(_vec4<type_t>&& vector) : x(std::move(vector.x)), y(std::move(vector.y)), z(std::move(vector.z)), w(std::move(vector.w)) {}

		template <typename type_t>
		constexpr _vec4(const std::array<type_t, 4>& array) : x(array[0]), y(array[1]), z(array[2]), w(array[3]) {}

		template <typename type_t>
		constexpr _vec4(std::array<type_t, 4>&& array) : x(std::move(array[0])), y(std::move(array[1])), z(std::move(array[2])), w(std::move(array[3])) {}

		template <typename type_t>
		constexpr _vec4(const type_t* begin, const type_t* end) 
		{
			std::copy(begin, end, this->begin());
		}

		constexpr _Ty& operator[](uint64_t idx) { return !idx ? x : idx == 1 ? y : idx == 2 ? z : w; }
		constexpr _Ty operator[](uint64_t idx) const { return !idx ? x : idx == 1 ? y : idx == 2 ? z : w; }

		template <typename _Type>
		constexpr _vec3<value_type> operator%(_Type single_value) const {
			return _vec3<value_type>(fmod(this->x, single_value), fmod(this->y, single_value), fmod(this->z, single_value), fmod(this->w, single_value));
		}

		template <typename _Type>
		constexpr bool operator==(const _vec4<_Type>& vector) const {
			return this->x == vector.x && this->y == vector.y && this->y == vector.z && this->w == vector.w;
		}

		template <typename _Type>
		constexpr bool operator==(_Type single_value) const {
			return this->x == single_value && this->y == single_value && this->z == single_value && this->w == single_value;
		}

		constexpr bool operator!() const {
			return !this->x && !this->y && !this->z && !this->w;
		}

		template <typename _Type>
		constexpr bool operator!=(const _vec4<_Type>& vector) const {
			return this->x != vector.x || this->y != vector.y || this->z != vector.z || this->w != vector.w;
		}

		template <typename _Type>
		constexpr bool operator!=(_Type single_value) const {
			return this->x != single_value || this->y != single_value || this->w != single_value;
		}

		// math operators
		template <typename _Type>
		constexpr _vec4<value_type> operator+(const _vec4<_Type>& vector) const
		{
			return _vec4<value_type>(this->x + vector.x, this->y + vector.y, this->z + vector.z, this->w + vector.w);
		}

		template <typename _Type>
		constexpr _vec4<value_type> operator+(_Type single_value) const
		{
			return _vec4<_Type>(this->x + single_value, this->y + single_value, this->z + single_value, this->w + single_value);
		}

		template <typename _Type>
		constexpr _vec4<value_type>& operator+=(const _vec4<_Type>& vector)
		{
			this->x += vector.x;
			this->y += vector.y;
			this->z += vector.z;
			this->w += vector.w;
			return *this;
		}

		template <typename _Type>
		constexpr _vec4<value_type>& operator+=(_Type single_value)
		{
			this->x += single_value;
			this->y += single_value;
			this->z += single_value;
			this->w += single_value;
			return *this;
		}


		template <typename _Type>
		constexpr _vec4<value_type> operator-(const _vec4<_Type>& vector) const
		{
			return _vec4<value_type>(this->x - vector.x, this->y - vector.y, this->z - vector.z, this->w - vector.w);
		}

		template <typename _Type>
		constexpr _vec4<value_type> operator-(_Type single_value) const
		{
			return _vec4<_Type>(this->x - single_value, this->y - single_value, this->z - single_value, this->w - single_value);
		}

		template <typename _Type>
		constexpr _vec4<value_type>& operator-=(const _vec4<_Type>& vector)
		{
			this->x -= vector.x;
			this->y -= vector.y;
			this->z -= vector.z;
			this->z -= vector.w;
			return *this;
		}

		template <typename _Type>
		constexpr _vec4<value_type>& operator-=(_Type single_value)
		{
			this->x -= single_value;
			this->y -= single_value;
			this->z -= single_value;
			this->w -= single_value;
			return *this;
		}


		template <typename _Type>
		constexpr _vec4<value_type> operator*(const _vec4<_Type>& vector) const
		{
			return _vec4<value_type>(this->x * vector.x, this->y * vector.y, this->z * vector.z, this->w * vector.w);
		}

		template <typename _Type>
		constexpr _vec4<value_type> operator*(_Type single_value) const
		{
			return _vec4<_Type>(this->x * single_value, this->y * single_value, this->z * single_value, this->w * single_value);
		}

		template <typename _Type>
		constexpr _vec4<value_type>& operator*=(const _vec4<_Type>& vector)
		{
			this->x *= vector.x;
			this->y *= vector.y;
			this->z *= vector.z;
			this->w *= vector.w;
			return *this;
		}

		template <typename _Type>
		constexpr _vec4<value_type>& operator*=(_Type single_value)
		{
			this->x *= single_value;
			this->y *= single_value;
			this->z *= single_value;
			this->w *= single_value;
			return *this;
		}


		template <typename _Type>
		constexpr _vec4<value_type> operator/(const _vec4<_Type>& vector) const
		{
			return _vec4<value_type>(this->x / vector.x, this->y / vector.y, this->z / vector.z, this->w / vector.w);
		}

		template <typename _Type>
		constexpr _vec4<value_type> operator/(_Type single_value) const
		{
			return _vec4<_Type>(this->x / single_value, this->y / single_value, this->z / single_value, this->w / single_value);
		}

		template <typename _Type>
		constexpr _vec4<value_type>& operator/=(const _vec4<_Type>& vector)
		{
			this->x /= vector.x;
			this->y /= vector.y;
			this->z /= vector.z;
			this->w /= vector.w;
			return *this;
		}

		template <typename _Type>
		constexpr _vec4<value_type>& operator/=(_Type single_value)
		{
			this->x /= single_value;
			this->y /= single_value;
			this->z /= single_value;
			this->w /= single_value;
			return *this;
		}

		constexpr _vec4<_Ty> floored() const { return _vec4<_Ty>(floor(x), floor(y), floor(z), floor(w)); }
		constexpr _vec4<_Ty> floored(_Ty value) const { return _vec4<_Ty>(floor(x / value), floor(y / value), floor(z / value), floor(w / value)); }
		constexpr vec_t clamp(_Ty min, _Ty max) const {
			return vec_t(std::clamp(x, min, max), std::clamp(y, min, max), std::clamp(z, min, max), std::clamp(w, min, max));
		}

		template <fan::is_not_arithmetic_t T>
		constexpr vec_t clamp(const T min, T max) const {
			return vec_t(std::clamp(x, min[0], max[0]), std::clamp(y, min[1], max[1]), std::clamp(z, min[2], max[2]), std::clamp(w, min[3], max[3]));
		}

		constexpr bool isnan() const {
			return std::isnan(x) || std::isnan(y) || std::isnan(z) || std::isnan(w);
		}

		static constexpr uint_t size() { return 4; }

		auto begin() const { return &x; }
		auto end() const { return begin() + size(); }
		auto data() const { return begin(); }

		auto begin() { return &x; }
		auto end() { return begin() + size(); }
		auto data() { return begin(); }

		constexpr void print() const { std::cout << x << " " << y << " " << z << " " << w << std::endl; }

		template <typename T>
		friend std::ostream& operator<<(std::ostream& os, const _vec4<T>& vector);

	};

	template <typename T>
	std::ostream& operator<<(std::ostream& os, const _vec2<T>& vector)
	{
		os << vector.x << " " << vector.y;
		return os;
	}


	template <typename T>
	std::ostream& operator<<(std::ostream& os, const _vec3<T>& vector)
	{
		os << vector.x << " " << vector.y << " " << vector.z;
		return os;
	}

	template <typename T>
	std::ostream& operator<<(std::ostream& os, const _vec4<T>& vector)
	{
		os << vector.x << " " << vector.y << " " << vector.z << " " << vector.w;
		return os;
	}

	using vec2b = _vec2<bool>;
	using vec3b = _vec3<bool>;
	using vec4b = _vec4<bool>;

	using vec2i = _vec2<int>;
	using vec3i = _vec3<int>;
	using vec4i = _vec4<int>;

	using vec2ui = _vec2<uint_t>;
	using vec3ui = _vec3<uint_t>;
	using vec4ui = _vec4<uint_t>;

	using vec2f = _vec2<f32_t>;
	using vec3f = _vec3<f32_t>;
	using vec4f = _vec4<f32_t>;

	using vec2 = _vec2<cf_t>;
	using vec3 = _vec3<cf_t>;
	using vec4 = _vec4<cf_t>;

	template <typename _Casted, template<typename> typename _Vec_t, typename _Old>
	constexpr _Vec_t<_Casted> cast(const _Vec_t<_Old>& v)
	{
		return _Vec_t<_Casted>(v);
	}

	template <typename T>
	constexpr uint64_t vector_size(const std::vector<std::vector<T>>& vector) {
		uint64_t size = 0;
		for (const auto& i : vector) {
			size += i.size();
		}
		return size;
	}

	template <typename T>
	constexpr uint64_t vector_size(const std::vector<T>& vector) {
		return vector.size();
	}

	template <typename T = fan::vec2>
	static T random_vector(f_t min, f_t max) {
		if constexpr (std::is_same_v<T, fan::vec2>) {
			return T(fan::random<int64_t, int64_t>(min, max), fan::random<int64_t, int64_t>(min, max));
		}
		else {
			return T(fan::random<int64_t, int64_t>(min, max), fan::random<int64_t, int64_t>(min, max), fan::random<int64_t, int64_t>(min, max));
		}
	}

}