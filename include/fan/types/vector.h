#pragma once

#include <iostream>
#include <algorithm>
#include <numeric>
#include <array>
#include <string>

#include _FAN_PATH(math/math.h)


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

		constexpr _vec2() = default;
		constexpr _vec2(_Ty value) : x(value), y(value) {}
		constexpr _vec2(_Ty x_, _Ty y_) {
      x = x_;
      y = y_;
    }

    #if defined(loco_imgui)
    constexpr _vec2(const ImVec2& v) {
      x = v.x;
      y = v.y;
    }
    #endif

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
			for (uintptr_t i = 0; i < size(); i++) {
				if (this->operator[](i)) {
					return this->operator[](i);
				}
			}
			return _Ty();
		}

		constexpr vec_t floor() const { return vec_t(std::floor(x), std::floor(y)); }
		constexpr vec_t floor(_Ty value) const { return vec_t(std::floor(x / value), std::floor(y / value)); }

		constexpr vec_t round() const { return vec_t(std::round(x), std::round(y)); }

		constexpr vec_t ceil() const { return vec_t(std::ceil(x), std::ceil(y)); }
		constexpr vec_t ceil(_Ty value) const { return vec_t(std::ceil(x / value), std::ceil(y / value)); }

		constexpr _Ty min() const { return fan::min(x, y); }
		constexpr _Ty max() const { return fan::max(x, y); }
		constexpr _vec2<value_type> max(value_type max) const { return _vec2<value_type>(std::max(x, max), std::max(y, max)); }
		constexpr vec_t abs() const { return vec_t(fan::math::abs(x), fan::math::abs(y)); }

		constexpr vec_t clamp(_Ty min, _Ty max) const {
			return vec_t(std::clamp(x, min, max), std::clamp(y, min, max));
		}

		constexpr vec_t get_sign() const {
			return vec_t((_Ty(0) < x) - (x < _Ty(0)), (_Ty(0) < y) - (y < _Ty(0)));
		}

		template <typename T>
		constexpr vec_t clamp(const T min, T max) const {
			return vec_t(fan::clamp(x, min[0], max[0]), fan::clamp(y, min[1], max[1]));
		}

		constexpr bool isnan() const {
			return std::isnan(x) || std::isnan(y);
		}

		constexpr auto length() const {
			return fan_2d::math::vector_length(*this);
		}

		constexpr fan::_vec2<_Ty> normalize() const {
			return fan_2d::math::normalize(*this);
		}

    constexpr fan::_vec2<_Ty> square_normalize() const {
      return *this / abs().max();
    }

		auto begin() const { return &x; }
		auto end() const { return begin() + size(); }
		auto data() const { return begin(); }

		auto begin() { return &x; }
		auto end() { return begin() + size(); }
		auto data() { return begin(); }

		static constexpr uintptr_t size() { return 2; }
		constexpr void print() const { std::cout << x << " " << y << std::endl; }

		constexpr auto multiply() const {
			return x * y;
		}

		template <typename T>
		constexpr auto dot(T vector) const {
			return fan_2d::math::dot(*this, vector);
		}

		constexpr auto angle() const {
			return atan2(y, x);
		}
		// coordinate system angle. TODO need rename to something meaningful
		constexpr auto csangle() const {
			return atan2(x, -y);
		}

		template <typename T>
		friend std::ostream& operator<<(std::ostream& os, const _vec2<T>& vector);

		template <typename T>
		friend std::ofstream& operator<<(std::ofstream& os, const _vec2<T>& vector);

    void from_string(const fan::string& str) {
      std::sscanf(str.c_str(), "{%f, %f}", &x, &y);
    }
    std::string to_string(int precision = 2) const {
      return "{" + 
        fan::to_string(x, precision) + ", " + 
        fan::to_string(y, precision) + 
        "}";
    }

    std::string c_str(int precision = 2) const {
			return to_string();
		}

    auto hypotenuse() const {
      return std::sqrt(x * x +  y * y);
    }

    /*vec_t constrain(const vec_t& bounds) const {
      return fan::vec2(fan::clamp(x, 0, bounds.x), fan::clamp(y, 0, bounds.y));
    }*/

    // clamps between 
    void constrain(const vec_t& min_bounds) {
      x = fan::clamp(x, min_bounds.x, x);
      y = fan::clamp(y, min_bounds.y, y);
    }
    void constrain(const vec_t& min, const vec_t& max) {
      x = fan::clamp(x, min.x, max.x);
      y = fan::clamp(y, min.x, max.y);
    }

    #if defined(loco_imgui)
    operator ImVec2() const {
      return ImVec2(x, y);
    }
    #endif
	};

	template <typename _Ty = f32_t>
	class _vec3 {
	public:
		_Ty x, y, z;

		using value_type = _Ty;
		using vec_t = _vec3<_Ty>;

		_vec3() = default;
		constexpr _vec3(_Ty x, _Ty y, _Ty z) : x(x), y(y), z(z) {}
		constexpr _vec3(_Ty value) : x(value), y(value), z(value) {}
		template <typename T>
		constexpr _vec3(const fan::_vec2<T>& v, auto value) : x(v.x), y(v.y), z(value) {}

		template <typename type_t>
		constexpr _vec3(const _vec3<type_t>& vec) : x(vec.x), y(vec.y), z(vec.z) {}

		template <typename type_t>
		constexpr _vec3(_vec3<type_t>&& vec) : x(std::move(vec.x)), y(std::move(vec.y)), z(std::move(vec.z)) {}

		template <typename type_t>
		constexpr _vec3(const _vec2<type_t>& vec) : x(vec.x), y(vec.y), z(0) {}

    template <typename type_t>
    _vec3& operator=(const _vec2<type_t>& vec) {
      x = vec.x;
      y = vec.y;
      return *this;
    }

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

		template <typename _Type>
		constexpr _vec3<value_type>& operator+(const fan::_vec2<_Type>& v)
		{
			return _vec3<value_type>(x + v.x, y + v.y, z);
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
		constexpr _vec3<value_type>& operator+=(const fan::_vec2<_Type>& v)
		{
			this->x += v.x;
			this->y += v.y;
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

		// template <typename _Type>
		// constexpr _vec3<value_type>& operator-(const fan::_vec2<_Type>& v) const
		// {
			// return _vec3<value_type>(x - v.x, y - v.y, z);
		// }

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
		constexpr _vec3<value_type>& operator-=(const fan::_vec2<_Type>& v)
		{
			this->x -= v.x;
			this->y -= v.y;
			return *this;
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
		constexpr _vec3<value_type>& operator*(const fan::_vec2<_Type>& v)
		{
			return _vec3<value_type>(x * v.x, y * v.y, z);
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
		constexpr _vec3<value_type>& operator*=(const fan::_vec2<_Type>& v)
		{
			this->x *= v.x;
			this->y *= v.y;
			return *this;
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
		constexpr _vec3<value_type>& operator/(const fan::_vec2<_Type>& v)
		{
			return _vec3<value_type>(x / v.x, y / v.y, z);
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

		template <typename _Type>
		constexpr _vec3<value_type>& operator/=(const fan::_vec2<_Type>& v)
		{
			this->x /= v.x;
			this->y /= v.y;
			return *this;
		}

		constexpr bool operator<(const auto& vector) const {
			return x < vector.x && y < vector.y && z < vector.z;
		}

		constexpr bool operator<=(const auto& vector) const {
			return x <= vector.x && y <= vector.y && z <= vector.z;
		}

		constexpr vec_t floor() const { return vec_t(std::floor(x), std::floor(y), std::floor(z)); }
		constexpr vec_t floor(_Ty value) const { return vec_t(std::floor(x / value), std::floor(y / value), floor(z / value)); }
		constexpr vec_t ceil() const { return vec_t(std::ceil(x), std::ceil(y), std::ceil(z)); }
		constexpr vec_t round() const { return vec_t(std::round(x), std::round(y), std::round(z)); }

		constexpr _Ty min() const { return std::min({ x, y, z }); }
    constexpr _Ty min(const vec_t& v) const { return std::min({ std::min(x, v.x), std::min(y, v.y), std::min(z, v.z)}); }
		constexpr _Ty max() const { return std::max({ x, y, z }); }
    constexpr _Ty max(const vec_t& v) const {
      return std::max({ std::max(x, v.x), std::max(y, v.y), std::max(z, v.z) });
    }

		constexpr vec_t abs() const { return vec_t(fan::math::abs(x), fan::math::abs(y), fan::math::abs(z)); }
		constexpr vec_t clamp(_Ty min, _Ty max) const {
			return vec_t(std::clamp(x, min, max), std::clamp(y, min, max), std::clamp(z, min, max));
		}

		template <typename T>
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


		static constexpr uintptr_t size() { return 3; }

		constexpr void print() const { std::cout << x << " " << y << " " << z << std::endl; }

		template <typename T>
		constexpr auto dot(const fan::_vec3<T>& vector) const {
			return fan_3d::math::dot(*this, vector);
		}

		template <typename T>
		constexpr auto cross(const fan::_vec3<T>& vector) const {
			return fan::math::cross(*this, vector);
		}

		constexpr f_t length() const {
			return fan_3d::math::vector_length(*this);
		}

		constexpr fan::_vec3<_Ty> normalize() const {
			return fan_3d::math::normalize(*this);
		}

		template <typename T>
		friend std::ostream& operator<<(std::ostream& os, const _vec3<T>& vector);

    void from_string(const fan::string& str) {
      std::sscanf(str.c_str(), "{%f, %f, &f}", &x, &y, &z);
    }
    std::string to_string(int precision = 2) const {
      return "{" +
        fan::to_string(x, precision) + ", " +
        fan::to_string(y, precision) + ", " +
        fan::to_string(z, precision) +
      "}";
    }
    std::string c_str(int precision = 2) const {
			return to_string();
		}
	};

	template <typename _Ty = f32_t>
	class _vec4 {
	public:
		_Ty x, y, z, w;

		using value_type = _Ty;
		using vec_t = _vec3<_Ty>;

		constexpr _vec4() = default;
		constexpr _vec4(_Ty x, _Ty y, _Ty z, _Ty w) : x(x), y(y), z(z), w(w) {}
		constexpr _vec4(_Ty value) : x(value), y(value), z(value), w(value) {}

		template <typename type_t, typename type2>
		constexpr _vec4(const _vec2<type_t>& a, const _vec2<type2>& b) : x(a.x), y(a.y), z(b.x), w(b.y) {}

		template <typename type_t, typename type2>
		constexpr _vec4(const _vec3<type_t> vector, type2 value) : x(vector.x), y(vector.y), z(vector.z), w(value) {}

		template <typename type_t>
		constexpr _vec4(const _vec4<type_t>& vector) : x(vector.x), y(vector.y), z(vector.z), w(vector.w) {}

		template <typename type_t>
		constexpr _vec4(const std::array<type_t, 4>& array) : x(array[0]), y(array[1]), z(array[2]), w(array[3]) {}

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
		constexpr _vec4<value_type> operator*(const fan::_vec2<_Type>& v) const
		{
			return _vec4<_Type>(this->x * v.x, this->y * v.y, this->z * v.x, this->w * v.y);
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

		constexpr _vec4<_Ty> floor() const { return _vec4<_Ty>(std::floor(x), std::floor(y), std::floor(z), std::floor(w)); }
		constexpr _vec4<_Ty> floor(_Ty value) const { return _vec4<_Ty>(std::floor(x / value), std::floor(y / value), std::floor(z / value), std::floor(w / value)); }
		constexpr vec_t clamp(_Ty min, _Ty max) const {
			return vec_t(std::clamp(x, min, max), std::clamp(y, min, max), std::clamp(z, min, max), std::clamp(w, min, max));
		}

		template <typename T>
		constexpr vec_t clamp(const T min, T max) const {
			return vec_t(std::clamp(x, min[0], max[0]), std::clamp(y, min[1], max[1]), std::clamp(z, min[2], max[2]), std::clamp(w, min[3], max[3]));
		}

		constexpr bool isnan() const {
			return std::isnan(x) || std::isnan(y) || std::isnan(z) || std::isnan(w);
		}

		static constexpr uintptr_t size() { return 4; }

		auto begin() const { return &x; }
		auto end() const { return begin() + size(); }
		auto data() const { return begin(); }

		auto begin() { return &x; }
		auto end() { return begin() + size(); }
		auto data() { return begin(); }

		constexpr _Ty max() const { return std::max({ x, y, z, w }); }

    template <typename T>
		friend std::ostream& operator<<(std::ostream& os, const _vec4<T>& vector);

    void from_string(const fan::string& str) {
      std::sscanf(str.c_str(), "{%f, %f, %f, %f}", &x, &y, &z, &w);
    }
    std::string to_string(int precision = 2) const {
      return "{" +
        fan::to_string(x, precision) + ", " +
        fan::to_string(y, precision) + ", " +
        fan::to_string(z, precision) + ", " +
        fan::to_string(w, precision) +
        "}";
    }
	};

	template <typename T>
	std::ofstream& operator<<(std::ofstream& os, const _vec2<T>& vector)
	{
		os << vector.x << vector.y;
		return os;
	}


	template <typename T>
	std::ostream& operator<<(std::ostream& os, const _vec2<T>& vector)
	{
		os << vector.to_string();
		return os;
	}


	template <typename T>
	std::ostream& operator<<(std::ostream& os, const _vec3<T>& vector)
	{
		os << vector.to_string();
		return os;
	}

	template <typename T>
	std::ostream& operator<<(std::ostream& os, const _vec4<T>& vector)
	{
		os << vector.to_string();
		return os;
	}

	using vec2b = _vec2<bool>;
	using vec3b = _vec3<bool>;
	using vec4b = _vec4<bool>;

	using vec2i = _vec2<int>;
	using vec3i = _vec3<int>;
	using vec4i = _vec4<int>;

	using vec2si = vec2i;
	using vec3si = vec3i;
	using vec4si = vec4i;

	using vec2ui = _vec2<uint32_t>;
	using vec3ui = _vec3<uint32_t>;
	using vec4ui = _vec4<uint32_t>;

	using vec2f = _vec2<f32_t>;
	using vec3f = _vec3<f32_t>;
	using vec4f = _vec4<f32_t>;

	using vec2d = _vec2<f64_t>;
	using vec3d = _vec3<f64_t>;
	using vec4d = _vec4<f64_t>;

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
  template <typename T, typename T2>
  constexpr fan::_vec2<T> min(const fan::_vec2<T>& x, fan::_vec2<T2> y) {
    return { x.x < y.x ? x.x : y.x, x.y < y.y ? x.y : y.y };
  }
  template <typename T, typename T2>
  constexpr fan::_vec2<T> copysign(const fan::_vec2<T>& x, fan::_vec2<T2> y) {
    return { fan::math::copysign(x.x, y.x), fan::math::copysign(x.y, y.y) };
  }
  constexpr auto copysign(const auto& v, const auto v2) {
    return fan::math::copysign(v, v2);
  }
  namespace math {
    fan::vec2 reflect(const auto& Direction, const auto& Normal) {
      auto k = fan::math::cross(fan::vec3{ Normal.x, Normal.y, 0 }, fan::vec3{ 0, 0, -1 });
      f32_t multiplier = k.dot(fan::vec3{ Direction.x, Direction.y, 0 });
      return fan::vec2( k.x * multiplier, k.y * multiplier);
    }
  }
}

namespace fmt {
  template<typename T>
  struct fmt::formatter<fan::_vec2<T>> {
    auto parse(fmt::format_parse_context& ctx) {
      return ctx.end();
    }
    auto format(const fan::_vec2<T>& obj, fmt::format_context& ctx) {

      return fmt::format_to(ctx.out(), "{}", obj.to_string());
    }
  };
  template<typename T>
  struct fmt::formatter<fan::_vec3<T>> {
    auto parse(fmt::format_parse_context& ctx) {
      return ctx.end();
    }
    auto format(const fan::_vec3<T>& obj, fmt::format_context& ctx) {

      return fmt::format_to(ctx.out(), "{}", obj.to_string());
    }
  };
  template<typename T>
  struct fmt::formatter<fan::_vec4<T>> {
    auto parse(fmt::format_parse_context& ctx) {
      return ctx.end();
    }
    auto format(const fan::_vec4<T>& obj, fmt::format_context& ctx) {

      return fmt::format_to(ctx.out(), "{}", obj.to_string());
    }
  };
}