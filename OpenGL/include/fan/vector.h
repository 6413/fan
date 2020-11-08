#pragma once
#ifdef _MSC_VER
#pragma warning (disable : 4244)
#pragma warning (disable : 26451)
#endif

#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <array>
#include <functional>

namespace fan {

	template <typename T>
	constexpr auto abs(T value) {
		return value < 0 ? -value : value;
	}

	template <class T, class... Ts>
	struct is_any : std::bool_constant<(std::is_same_v<T, Ts> || ...)> {};

	constexpr f32_t INF = INFINITY;
	template <typename _Ty>
	class _vec3;

	template <typename type, std::size_t n>
	struct list;

	template <typename _Ty>
	class _vec2 {
	public:
		_Ty x, y;

		using type = _Ty;
		using vec_t = _vec2<_Ty>;

		constexpr _vec2() : x(0), y(0) { }
		constexpr _vec2(_Ty value) : x(value), y(value) { }
		constexpr _vec2(_Ty x_, _Ty y_) : x(x_), y(y_) { }
		template <typename T>
		constexpr _vec2(const _vec2<T>& vec) : x(vec.x), y(vec.y) { }
		template <typename T>
		constexpr _vec2(const _vec3<T>& vec) : x(vec.x), y(vec.y) { }
		template <typename T>
		constexpr _vec2(const list<T, 2>& _array) : x(_array[0]), y(_array[1]) {}

		template <typename T>
		constexpr _vec2<type>& operator=(const _vec2<T>& vector) {
			this->x = vector.x;
			this->y = vector.y;
			return *this;
		}

		constexpr _Ty& operator[](uint64_t idx) { return !idx ? x : y; }
		constexpr _Ty operator[](uint64_t idx) const { return !idx ? x : y; }

		template <typename _Type>
		constexpr _vec2<type> operator%(_Type single_value) const noexcept {
			return _vec2<type>(fmod_dr(this->x, single_value), fmod_dr(this->y, single_value));
		}

		template <typename _Type>
		constexpr bool operator==(const _vec2<_Type>& vector) const noexcept {
			return this->x == vector.x && this->y == vector.y;
		}

		constexpr bool operator==(type single_value) const noexcept {
			return this->x == single_value && this->y == single_value;
		}

		template <typename _Type>
		constexpr bool operator!=(const _vec2<_Type>& vector) const noexcept {
			return this->x != vector.x || this->y != vector.y;
		}

		constexpr bool operator!=(type single_value) const noexcept {
			return this->x != single_value || this->y != single_value;
		}

		template <typename T>
		constexpr bool operator!=(const std::array<T, 2>& array) {
			return this->x != array[0] || this->y != array[1];
		}

		template <typename _Type>
		constexpr bool operator<=(const _vec2<_Type>& vector) const noexcept {
			return this->x <= vector.x && this->y <= vector.y;
		}

		// math operators
		template <typename _Type>
		constexpr _vec2<type> operator+(const _vec2<_Type>& vector) const noexcept
		{
			return _vec2<type>(this->x + vector.x, this->y + vector.y);
		}

		template <typename T>
		constexpr _vec2<type> operator+(const list<T, 2>& list_) const noexcept
		{
			return _vec2<type>(this->x + list_[0], this->y + list_[1]);
		}

		template <typename _Type>
		constexpr _vec2<type> operator+(_Type single_value) const noexcept
		{
			return _vec2<_Type>(this->x + single_value, this->y + single_value);
		}

		template <typename _Type>
		constexpr _vec2<type> operator+=(const _vec2<_Type>& vector) noexcept
		{
			this->x += vector.x;
			this->y += vector.y;
			return *this;
		}

		template <typename _Type>
		constexpr _vec2<type> operator+=(_Type single_value) noexcept
		{
			this->x += single_value;
			this->y += single_value;
			return *this;
		}

		constexpr _vec2<type> operator-() const noexcept
		{
			return _vec2<type>(-this->x, -this->y);
		}

		template <typename _Type>
		constexpr _vec2<type> operator-(const _vec2<_Type>& vector) const noexcept
		{
			return _vec2<type>(this->x - vector.x, this->y - vector.y);
		}

		template <typename T>
		constexpr _vec2<type> operator-(const list<T, 2>& list_) const noexcept
		{
			return _vec2<type>(this->x - list_[0], this->y - list_[1]);
		}

		template <typename _Type>
		constexpr _vec2<type> operator-(_Type single_value) const noexcept
		{
			return _vec2<_Type>(this->x - single_value, this->y - single_value);
		}

		template <typename _Type>
		constexpr _vec2<type> operator-=(const _vec2<_Type>& vector) noexcept
		{
			this->x -= vector.x;
			this->y -= vector.y;
			return *this;
		}

		template <typename _Type>
		constexpr _vec2<type> operator-=(_Type single_value) noexcept
		{
			this->x -= single_value;
			this->y -= single_value;
			return *this;
		}


		template <typename _Type>
		constexpr _vec2<type> operator*(const _vec2<_Type>& vector) const noexcept
		{
			return _vec2<type>(this->x * vector.x, this->y * vector.y);
		}

		template <typename T>
		constexpr _vec2<type> operator*(const list<T, 2>& list_) const noexcept
		{
			return _vec2<type>(this->x * list_[0], this->y * list_[1]);
		}

		template <typename _Type>
		constexpr _vec2<type> operator*(_Type single_value) const noexcept
		{
			return _vec2<_Type>(this->x * single_value, this->y * single_value);
		}

		template <typename _Type>
		constexpr _vec2<type> operator*=(const _vec2<_Type>& vector) noexcept
		{
			this->x *= vector.x;
			this->y *= vector.y;
			return *this;
		}

		template <typename _Type>
		constexpr _vec2<type> operator*=(_Type single_value) noexcept
		{
			this->x *= single_value;
			this->y *= single_value;
			return *this;
		}


		template <typename _Type>
		constexpr _vec2<type> operator/(const _vec2<_Type>& vector) const noexcept
		{
			return _vec2<type>(this->x / vector.x, this->y / vector.y);
		}

		template <typename T>
		constexpr _vec2<type> operator/(const list<T, 2>& list_) const noexcept
		{
			return _vec2<type>(this->x / list_[0], this->y / list_[1]);
		}

		template <typename _Type>
		constexpr _vec2<type> operator/(_Type single_value) const noexcept
		{
			return _vec2<_Type>(this->x / (type)single_value, this->y / (type)single_value);
		}

		template <typename _Type>
		constexpr _vec2<type> operator/=(const _vec2<_Type>& vector) noexcept
		{
			this->x /= vector.x;
			this->y /= vector.y;
			return *this;
		}

		template <typename _Type>
		constexpr _vec2<type> operator/=(_Type single_value) noexcept
		{
			this->x /= single_value;
			this->y /= single_value;
			return *this;
		}

		constexpr vec_t floored() const { return vec_t(floor(x), floor(y)); }
		constexpr vec_t floored(_Ty value) const { return vec_t(floor(x / value), floor(y / value)); }

		constexpr vec_t ceiled() const { return vec_t(ceil(x), ceil(y)); }
		constexpr vec_t ceiled(_Ty value) const { return vec_t(ceil(x / value), ceil(y / value)); }

		constexpr _Ty min() const { return std::min(x, y); }
		constexpr _Ty max() const { return std::max(x, y); }
		constexpr vec_t abs() const { return vec_t(fan::abs(x), fan::abs(y)); }

		auto begin() const { return &x; }
		auto end() const { return begin() + size(); }
		auto data() const { return begin(); }

		auto begin() { return &x; }
		auto end() { return begin() + size(); }
		auto data() { return begin(); }

		static constexpr uint64_t size() { return 2; }
		constexpr void print() const { std::cout << x << " " << y << std::endl; }

		template <typename T>
		friend std::ostream& operator<<(std::ostream& os, const _vec2<T>& vector) noexcept;

	};

	template <typename _Ty = f32_t>
	class _vec3 {
	public:
		_Ty x, y, z;

		using type = _Ty;
		using vec_t = _vec3<_Ty>;

		constexpr _vec3() : x(0), y(0), z(0) { }
		constexpr _vec3(_Ty x, _Ty y, _Ty z) : x(x), y(y), z(z) { }
		constexpr _vec3(_Ty value) : x(value), y(value), z(value) { }

		template <typename type>
		constexpr _vec3(const _vec3<type>& vec) : x(vec.x), y(vec.y), z(vec.z) { }

		template <typename type>
		constexpr _vec3(const _vec2<type>& vec) : x(vec.x), y(vec.y), z(0) { }

		template <typename type>
		constexpr _vec3(const std::array<type, 3>& array) : x(array[0]), y(array[1]), z(array[2]) {}

		constexpr _Ty& operator[](uint64_t idx) { return !idx ? x : idx == 1 ? y : z; }
		constexpr _Ty operator[](uint64_t idx) const { return !idx ? x : idx == 1 ? y : z; }

		template <typename _Type>
		constexpr _vec3<type> operator%(_Type single_value) const noexcept {
			return _vec3<type>(fmod_dr(this->x, single_value), fmod_dr(this->y, single_value), fmod_dr(this->z, single_value));
		}

		template <typename _Type>
		constexpr bool operator==(const _vec3<_Type>& vector) const noexcept {
			return this->x == vector.x && this->y == vector.y && this->y == vector.z;
		}

		template <typename _Type>
		constexpr bool operator==(_Type single_value) const noexcept {
			return this->x == single_value && this->y == single_value && this->z == single_value;
		}

		template <typename _Type>
		constexpr bool operator!=(const _vec3<_Type>& vector) const noexcept {
			return this->x != vector.x || this->y != vector.y || this->z != vector.z;
		}

		template <typename _Type>
		constexpr bool operator!=(_Type single_value) const noexcept {
			return this->x != single_value && this->y != single_value;
		}

		// math operators
		template <typename _Type>
		constexpr _vec3<type> operator+(const _vec3<_Type>& vector) const noexcept
		{
			return _vec3<type>(this->x + vector.x, this->y + vector.y, this->z + vector.z);
		}

		template <typename _Type>
		constexpr _vec3<type> operator+(_Type single_value) const noexcept
		{
			return _vec3<_Type>(this->x + single_value, this->y + single_value, this->z + single_value);
		}

		template <typename _Type>
		constexpr _vec3<type> operator+=(const _vec3<_Type>& vector) noexcept
		{
			this->x += vector.x;
			this->y += vector.y;
			this->z += vector.z;
			return *this;
		}

		template <typename _Type>
		constexpr _vec3<type> operator+=(_Type single_value) noexcept
		{
			this->x += single_value;
			this->y += single_value;
			this->z += single_value;
			return *this;
		}

		constexpr _vec3<type> operator-() const noexcept
		{
			return _vec3<type>(-this->x, -this->y, -this->z);
		}

		template <typename _Type>
		constexpr _vec3<type> operator-(const _vec3<_Type>& vector) const noexcept
		{
			return _vec3<type>(this->x - vector.x, this->y - vector.y, this->z - vector.z);
		}

		template <typename _Type>
		constexpr _vec3<type> operator-(_Type single_value) const noexcept
		{
			return _vec3<_Type>(this->x - single_value, this->y - single_value, this->z - single_value);
		}

		template <typename _Type>
		constexpr _vec3<type> operator-=(const _vec3<_Type>& vector) noexcept
		{
			this->x -= vector.x;
			this->y -= vector.y;
			this->z -= vector.z;
			return *this;
		}

		template <typename _Type>
		constexpr _vec3<type> operator-=(_Type single_value) noexcept
		{
			this->x -= single_value;
			this->y -= single_value;
			this->z -= single_value;
			return *this;
		}


		template <typename _Type>
		constexpr _vec3<type> operator*(const _vec3<_Type>& vector) const noexcept
		{
			return _vec3<type>(this->x * vector.x, this->y * vector.y, this->z * vector.z);
		}

		template <typename _Type>
		constexpr _vec3<type> operator*(_Type single_value) const noexcept
		{
			return _vec3<_Type>(this->x * single_value, this->y * single_value, this->z * single_value);
		}

		template <typename _Type>
		constexpr _vec3<type> operator*=(const _vec3<_Type>& vector) noexcept
		{
			this->x *= vector.x;
			this->y *= vector.y;
			this->z *= vector.z;
			return *this;
		}

		template <typename _Type>
		constexpr _vec3<type> operator*=(_Type single_value) noexcept
		{
			this->x *= single_value;
			this->y *= single_value;
			this->z *= single_value;
			return *this;
		}


		template <typename _Type>
		constexpr _vec3<type> operator/(const _vec3<_Type>& vector) const noexcept
		{
			return _vec3<type>(this->x / vector.x, this->y / vector.y, this->z / vector.z);
		}

		template <typename _Type>
		constexpr _vec3<type> operator/(_Type single_value) const noexcept
		{
			return _vec3<_Type>(this->x / single_value, this->y / single_value, this->z / single_value);
		}

		template <typename _Type>
		constexpr _vec3<type> operator/=(const _vec3<_Type>& vector) noexcept
		{
			this->x /= vector.x;
			this->y /= vector.y;
			this->z /= vector.z;
			return *this;
		}

		template <typename _Type>
		constexpr _vec3<type> operator/=(_Type single_value) noexcept
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

		auto begin() const { return &x; }
		auto end() const { return begin() + size(); }
		auto data() const { return begin(); }

		auto begin() { return &x; }
		auto end() { return begin() + size(); }
		auto data() { return begin(); }


		static constexpr uint64_t size() { return 3; }

		constexpr void print() const { std::cout << x << " " << y << " " << z << std::endl; }

		template <typename T>
		friend std::ostream& operator<<(std::ostream& os, const _vec3<T>& vector) noexcept;

	};

	template <typename _Ty = f32_t>
	class _vec4 {
	public:
		_Ty x, y, z, w;

		using type = _Ty;
		using vec_t = _vec3<_Ty>;

		constexpr _vec4() : x(0), y(0), z(0), w(0) { }
		constexpr _vec4(_Ty x, _Ty y, _Ty z, _Ty w) : x(x), y(y), z(z), w(w) { }
		constexpr _vec4(_Ty value) : x(value), y(value), z(value), w(value) { }

		template <typename type, typename type2>
		constexpr _vec4(const _vec2<type>& a, const _vec2<type2>& b) : x(a.x), y(a.y), z(b.x), w(b.y) {}

		template <typename type, typename type2>
		constexpr _vec4(const _vec3<type> vec, type2 value) : x(vec.x), y(vec.y), z(vec.z), w(value) { }

		template <typename type>
		constexpr _vec4(const _vec4<type>& vec) : x(vec.x), y(vec.y), z(vec.z), w(vec.w) { }

		template <typename type>
		constexpr _vec4(const std::array<type, 4>& array) : x(array[0]), y(array[1]), z(array[2]), w(array[3]) {}

		constexpr _Ty& operator[](uint64_t idx) { return !idx ? x : idx == 1 ? y : idx == 2 ? z : w; }
		constexpr _Ty operator[](uint64_t idx) const { return !idx ? x : idx == 1 ? y : idx == 2 ? z : w; }

		template <typename _Type>
		constexpr _vec3<type> operator%(_Type single_value) const noexcept {
			return _vec3<type>(fmod(this->x, single_value), fmod(this->y, single_value), fmod(this->z, single_value), fmod(this->w, single_value));
		}

		template <typename _Type>
		constexpr bool operator==(const _vec4<_Type>& vector) const noexcept {
			return this->x == vector.x && this->y == vector.y && this->y == vector.z && this->w == vector.w;
		}

		template <typename _Type>
		constexpr bool operator==(_Type single_value) const noexcept {
			return this->x == single_value && this->y == single_value && this->z == single_value && this->w == single_value;
		}

		template <typename _Type>
		constexpr bool operator!=(const _vec4<_Type>& vector) const noexcept {
			return this->x != vector.x || this->y != vector.y || this->z != vector.z || this->w != vector.w;
		}

		template <typename _Type>
		constexpr bool operator!=(_Type single_value) const noexcept {
			return this->x != single_value || this->y != single_value || this->w != single_value;
		}

		// math operators
		template <typename _Type>
		constexpr _vec4<type> operator+(const _vec4<_Type>& vector) const noexcept
		{
			return _vec4<type>(this->x + vector.x, this->y + vector.y, this->z + vector.z, this->w + vector.w);
		}

		template <typename _Type>
		constexpr _vec4<type> operator+(_Type single_value) const noexcept
		{
			return _vec4<_Type>(this->x + single_value, this->y + single_value, this->z + single_value, this->w + single_value);
		}

		template <typename _Type>
		constexpr _vec4<type> operator+=(const _vec4<_Type>& vector) noexcept
		{
			this->x += vector.x;
			this->y += vector.y;
			this->z += vector.z;
			this->w += vector.w;
			return *this;
		}

		template <typename _Type>
		constexpr _vec4<type> operator+=(_Type single_value) noexcept
		{
			this->x += single_value;
			this->y += single_value;
			this->z += single_value;
			this->w += single_value;
			return *this;
		}


		template <typename _Type>
		constexpr _vec4<type> operator-(const _vec4<_Type>& vector) const noexcept
		{
			return _vec4<type>(this->x - vector.x, this->y - vector.y, this->z - vector.z, this->w - vector.w);
		}

		template <typename _Type>
		constexpr _vec4<type> operator-(_Type single_value) const noexcept
		{
			return _vec4<_Type>(this->x - single_value, this->y - single_value, this->z - single_value, this->w - single_value);
		}

		template <typename _Type>
		constexpr _vec4<type> operator-=(const _vec4<_Type>& vector) noexcept
		{
			this->x -= vector.x;
			this->y -= vector.y;
			this->z -= vector.z;
			this->z -= vector.w;
			return *this;
		}

		template <typename _Type>
		constexpr _vec4<type> operator-=(_Type single_value) noexcept
		{
			this->x -= single_value;
			this->y -= single_value;
			this->z -= single_value;
			this->w -= single_value;
			return *this;
		}


		template <typename _Type>
		constexpr _vec4<type> operator*(const _vec4<_Type>& vector) const noexcept
		{
			return _vec4<type>(this->x * vector.x, this->y * vector.y, this->z * vector.z, this->w * vector.w);
		}

		template <typename _Type>
		constexpr _vec4<type> operator*(_Type single_value) const noexcept
		{
			return _vec4<_Type>(this->x * single_value, this->y * single_value, this->z * single_value, this->w * single_value);
		}

		template <typename _Type>
		constexpr _vec4<type> operator*=(const _vec4<_Type>& vector) noexcept
		{
			this->x *= vector.x;
			this->y *= vector.y;
			this->z *= vector.z;
			this->w *= vector.w;
			return *this;
		}

		template <typename _Type>
		constexpr _vec4<type> operator*=(_Type single_value) noexcept
		{
			this->x *= single_value;
			this->y *= single_value;
			this->z *= single_value;
			this->w *= single_value;
			return *this;
		}


		template <typename _Type>
		constexpr _vec4<type> operator/(const _vec4<_Type>& vector) const noexcept
		{
			return _vec4<type>(this->x / vector.x, this->y / vector.y, this->z / vector.z, this->w / vector.w);
		}

		template <typename _Type>
		constexpr _vec4<type> operator/(_Type single_value) const noexcept
		{
			return _vec4<_Type>(this->x / single_value, this->y / single_value, this->z / single_value, this->w / single_value);
		}

		template <typename _Type>
		constexpr _vec4<type> operator/=(const _vec4<_Type>& vector) noexcept
		{
			this->x /= vector.x;
			this->y /= vector.y;
			this->z /= vector.z;
			this->w /= vector.w;
			return *this;
		}

		template <typename _Type>
		constexpr _vec4<type> operator/=(_Type single_value) noexcept
		{
			this->x /= single_value;
			this->y /= single_value;
			this->z /= single_value;
			this->w /= single_value;
			return *this;
		}

		constexpr _vec4<_Ty> floored() const { return _vec4<_Ty>(floor(x), floor(y), floor(z), floor(w)); }
		constexpr _vec4<_Ty> floored(_Ty value) const { return _vec4<_Ty>(floor(x / value), floor(y / value), floor(z / value), floor(w / value)); }

		static constexpr uint64_t size() { return 4; }

		constexpr void print() const { std::cout << x << " " << y << " " << z << " " << w << std::endl; }

		template <typename T>
		friend std::ostream& operator<<(std::ostream& os, const _vec4<T>& vector) noexcept;

	};

	template <typename T>
	std::ostream& operator<<(std::ostream& os, const _vec2<T>& vector) noexcept
	{
		os << vector.x << " " << vector.y;
		return os;
	}


	template <typename T>
	std::ostream& operator<<(std::ostream& os, const _vec3<T>& vector) noexcept
	{
		os << vector.x << " " << vector.y << " " << vector.z;
		return os;
	}

	template <typename T>
	std::ostream& operator<<(std::ostream& os, const _vec4<T>& vector) noexcept
	{
		os << vector.x << " " << vector.y << " " << vector.z << " " << vector.w;
		return os;
	}

	using vec2i = _vec2<int>;
	using vec3i = _vec3<int>;
	using vec4i = _vec4<int>;

	using vec2ui = _vec2<unsigned int>;
	using vec3ui = _vec3<unsigned int>;
	using vec4ui = _vec4<unsigned int>;

	using vec2f = _vec2<f32_t>;
	using vec3f = _vec3<f32_t>;
	using vec4f = _vec4<f32_t>;

	using vec2 = vec2f;
	using vec3 = vec3f;
	using vec4 = vec4f;

	template <typename _Casted, template<typename> typename _Vec_t, typename _Old>
	constexpr _Vec_t<_Casted> cast(_Vec_t<_Old> v) noexcept
	{
		return _Vec_t<_Casted>(v);
	}

	template <typename T>
	constexpr uint64_t vector_size(const std::vector<std::vector<T>>& vector) {
		uint64_t size = 0;
		for (auto i : vector) {
			size += i.size();
		}
		return size;
	}

	template <typename T>
	constexpr uint64_t vector_size(const std::vector<T>& vector) {
		uint64_t size = 0;
		for (auto i : vector) {
			size += i.size();
		}
		return size;
	}
}