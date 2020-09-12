#pragma once
#ifdef _MSC_VER
#pragma warning (disable : 4244)
#pragma warning (disable : 26451)
#endif

#include <iostream>
#include <cmath>
#include <algorithm>
#include <array>
#include <functional>

template <class T, class... Ts>
struct is_any : std::bool_constant<(std::is_same_v<T, Ts> || ...)> {};

using f32_t = float;
using f64_t = double;
using f_t = f32_t;

#define GL_FLOAT_T GL_FLOAT

using uint_t = unsigned int;
constexpr f_t INF = INFINITY;

template <typename _Ty>
class _vec3;

template <typename type, std::size_t n>
class list;

template <typename _Ty>
class _vec2 {
public:
	_Ty x, y;

	using type = _Ty;
	using vec_t = _vec2<_Ty>;

	constexpr _vec2() : x(0), y(0) { }
	constexpr _vec2(_Ty value) : x(value), y(value) { }
	constexpr _vec2(_Ty x, _Ty y) : x(x), y(y) { }
	template <typename type>
	constexpr _vec2(const _vec2<type>& vec) : x(vec.x), y(vec.y) { }
	template <typename type>
	constexpr _vec2(const _vec3<type>& vec) : x(vec.x), y(vec.y) { }
	template <typename type>
	constexpr _vec2(const list<type, 2>& _array) : x(_array[0]), y(_array[1]) {}

	constexpr _Ty& operator[](uint64_t idx) { return !idx ? x : y; }
	constexpr _Ty operator[](uint64_t idx) const { return !idx ? x : y; }

	template <typename _Type>
	constexpr _vec2<type> operator%(_Type single_value) const noexcept {
		return _vec2<type>(fmod(this->x, single_value), fmod(this->y, single_value));
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
	constexpr vec_t abs() const { return vec_t(std::abs(x), std::abs(y)); }

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

template <typename _Ty = f_t>
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
		return _vec3<type>(fmod(this->x, single_value), fmod(this->y, single_value), fmod(this->z, single_value));
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
		return this->x != vector.x && this->y != vector.y && this->z != vector.z;
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
	constexpr vec_t abs() const { return vec_t(std::abs(x), std::abs(y), std::abs(z)); }

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

template <typename _Ty = f_t>
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

template <typename T, std::uint64_t N>
std::ostream& operator<<(std::ostream& os, const std::array<T, N>& array) noexcept
{
	for (int i = 0; i < N; i++) {
		if (!(i - 1 == N)) {
			os << array[i] << " ";
		}
		else {
			os << array[i];
		}
	}
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

template <typename ...Args>
constexpr void LOG(const Args&... args) {
	((std::cout << args << " "), ...) << '\n';
}

#ifndef _MSC_VER

template <class>
inline constexpr bool is_pointer_v = false; // determine whether _Ty is a pointer

template <class _Ty>
inline constexpr bool is_pointer_v<_Ty*> = true;

template <class _Ty>
inline constexpr bool is_pointer_v<_Ty* const> = true;

template <class _Ty>
inline constexpr bool is_pointer_v<_Ty* volatile> = true;

template <class _Ty>
inline constexpr bool is_pointer_v<_Ty* const volatile> = true;

template <class, class>
inline constexpr bool is_same_v = false; // determine whether arguments are the same type
template <class _Ty>
inline constexpr bool is_same_v<_Ty, _Ty> = true;

template <class _Iter, class = void>
inline constexpr bool _Allow_inheriting_unwrap_v = true;

template <class _Iter>
inline constexpr bool _Allow_inheriting_unwrap_v<_Iter, std::void_t<typename _Iter::_Prevent_inheriting_unwrap>> =
is_same_v<_Iter, typename _Iter::_Prevent_inheriting_unwrap>;

template <class _Iter, class = void>
inline constexpr bool _Unwrappable_v = false;

template <class _Iter>
inline constexpr bool _Unwrappable_v<_Iter,
	std::void_t<decltype(std::declval<std::__remove_cvref_t<_Iter>&>()._Seek_to(std::declval<_Iter>()._Unwrapped()))>> =
	_Allow_inheriting_unwrap_v<std::__remove_cvref_t<_Iter>>;

template <class _Iter>
[[nodiscard]] constexpr decltype(auto) _Get_unwrapped(_Iter&& _It) {
	if constexpr (is_pointer_v<std::decay_t<_Iter>>) {
		return _It + 0;
	}
	else if constexpr (_Unwrappable_v<_Iter>) {
		return static_cast<_Iter&&>(_It)._Unwrapped();
	}
	else {
		return static_cast<_Iter&&>(_It);
	}
}
#endif

template <typename T>
constexpr auto average(T begin, T end) {
	auto it = _Get_unwrapped(begin);
	std::uint64_t cols = 0;
	auto sum = typename std::remove_pointer<decltype(_Get_unwrapped(begin))>::type(0);
	for (; it != _Get_unwrapped(end); ++it, ++cols) {
		sum += *it;
	}
	return sum / cols;
}

template <typename type, std::size_t Rows, std::size_t Cols>
struct matrix;

template <typename type, std::size_t Rows>
struct list;

template <typename type, std::size_t Rows, std::size_t Cols = 1>
using da_t = std::conditional_t<Cols == 1, list<type, Rows>, matrix<type, Rows, Cols>>;

template <typename T, typename T2, typename _Func>
constexpr void foreach(T src_begin, T src_end, T2 dst_begin, _Func function) {
	auto dst = dst_begin;
	for (auto it = src_begin; it != src_end; ++it, ++dst) {
		*dst = function(*it);
	}
}

template <typename type, std::size_t rows>
struct list : public std::array<type, rows> {

	using array_type = std::array<type, rows>;

	constexpr list(const _vec2<f_t>& vector) {
		for (int i = 0; i < std::min(rows, vector.size()); i++) {
			this->operator[](i) = vector[i];
		}
	}

	template <typename ...T>
	constexpr list(T... x) : std::array<type, rows>{ (type)x... } {}

	constexpr list(const _vec3<f_t>& vector) {
		for (int i = 0; i < std::min(rows, vector.size()); i++) {
			this->operator[](i) = vector[i];
		}
	}

	template <typename T>
	constexpr list(T value) {
		for (int i = 0; i < rows; i++) {
			this->operator[](i) = value;
		}
	}

	template <typename T, std::size_t array_n>
	constexpr list(const list<T, array_n>& list) {
		std::copy(list.begin(), list.end(), this->begin());
	}

	constexpr auto operator++() noexcept {
		return &this->_Elems + 1;
	}

	template <typename T, std::size_t list_n>
	constexpr list operator+(const list<T, list_n>& _list) const noexcept {
		static_assert(rows >= list_n, "second list is bigger than first");
		list calculation_list;
		for (int i = 0; i < rows; i++) {
			calculation_list[i] = this->operator[](i) + _list[i];
		}
		return calculation_list;
	}

	template <typename T>
	constexpr list operator+(T value) const noexcept {
		list list;
		for (int i = 0; i < rows; i++) {
			list[i] = this->operator[](i) + value;
		}
		return list;
	}

	template <typename T, std::size_t rows_>
	constexpr list operator+=(const list<T, rows_>& value) noexcept {
		//static_assert(rows >= list_n, "second list is bigger than first");
		for (int i = 0; i < rows; i++) {
			this->operator[](i) += value[i];
		}
		return *this;
	}


	/*template <typename T>
	constexpr list operator+=(T value) noexcept {
		for (int i = 0; i < rows; i++) {
			this->operator[](i) += value;
		}
		return *this;
	}*/

	constexpr list operator-() const noexcept {
		list l;
		for (int i = 0; i < rows; i++) {
			l[i] = -this->operator[](i);
		}
		return l;
	}

	template <typename T, std::size_t list_n>
	constexpr list operator-(const list<T, list_n>& _list) const noexcept {
		static_assert(rows >= list_n, "second list is bigger than first");
		list calculation_list;
		for (int i = 0; i < rows; i++) {
			calculation_list[i] = this->operator[](i) - _list[i];
		}
		return calculation_list;
	}

	/*template <typename T>
	constexpr list operator-(T value) const noexcept {
		list list;
		for (int i = 0; i < rows; i++) {
			list[i] = this->operator[](i) - value;
		}
		return list;
	}*/

	template <typename T, std::size_t list_n>
	constexpr list operator-=(const list<T, list_n>& value) noexcept {
		static_assert(rows >= list_n, "second list is bigger than first");
		for (int i = 0; i < rows; i++) {
			this->operator[](i) -= value[i];
		}
		return *this;
	}

	template <typename T>
	constexpr list operator-=(T value) noexcept {
		for (int i = 0; i < rows; i++) {
			this->operator[](i) -= value;
		}
		return *this;
	}

	template <typename T, std::size_t list_n>
	constexpr list operator*(const list<T, list_n>& _list) const noexcept {
		static_assert(rows >= list_n, "second list is bigger than first");
		list calculation_list;
		for (int i = 0; i < rows; i++) {
			calculation_list[i] = this->operator[](i) * _list[i];
		}
		return calculation_list;
	}

	constexpr list operator*(type value) const noexcept {
		list list;
		for (int i = 0; i < rows; i++) {
			list[i] = this->operator[](i) * value;
		}
		return list;
	}

	template <typename T, std::size_t list_n>
	constexpr list operator*=(const list<T, list_n>& value) noexcept {
		static_assert(rows >= list_n, "second list is bigger than first");
		for (int i = 0; i < rows; i++) {
			this->operator[](i) *= value[i];
		}
		return *this;
	}

	template <typename T>
	constexpr list operator*=(T value) noexcept {
		for (int i = 0; i < rows; i++) {
			this->operator[](i) *= value;
		}
		return *this;
	}


	template <typename T, std::size_t list_n>
	constexpr list operator/(const list<T, list_n>& _list) const noexcept {
		static_assert(rows >= list_n, "second list is bigger than first");
		list calculation_list;
		for (int i = 0; i < rows; i++) {
			calculation_list[i] = this->operator[](i) / _list[i];
		}
		return calculation_list;
	}

	template <typename T>
	constexpr list operator/(T value) const noexcept {
		list list;
		for (int i = 0; i < rows; i++) {
			list[i] = this->operator[](i) / value;
		}
		return list;
	}

	template <typename T, std::size_t list_n>
	constexpr list operator/=(const list<T, list_n>& value) noexcept {
		static_assert(rows >= list_n, "second list is bigger than first");
		for (int i = 0; i < rows; i++) {
			this->operator[](i) /= value[i];
		}
		return *this;
	}

	template <typename T>
	constexpr list operator/=(T value) noexcept {
		for (int i = 0; i < rows; i++) {
			this->operator[](i) /= value;
		}
		return *this;
	}

	template <typename T>
	constexpr auto operator%(T value) {
		list l;
		for (int i = 0; i < rows; i++) {
			l = fmodf(this->operator[](i), value);
		}
		return l;
	}

	constexpr bool operator<(const list<type, rows>& list_) {
		for (int i = 0; i < rows; i++) {
			if (this->operator[](i) < list_[i]) {
				return true;
			}
		}
		return false;
	}

	constexpr bool operator<=(const list<type, rows>& list_) {
		for (int i = 0; i < rows; i++) {
			if (this->operator[](i) <= list_[i]) {
				return true;
			}
		}
		return false;
	}

	template <typename T>
	constexpr bool operator==(T value) {
		for (int i = 0; i < rows; i++) {
			if (this->operator[](i) != value) {
				return false;
			}
		}
		return true;
	}

	constexpr bool operator==(const list<type, rows>& list_) {
		for (int i = 0; i < rows; i++) {
			if (this->operator[](i) != list_[i]) {
				return false;
			}
		}
		return true;
	}

	template <typename T>
	constexpr bool operator!=(T value) {
		for (int i = 0; i < rows; i++) {
			if (this->operator[](i) == value) {
				return false;
			}
		}
		return true;
	}

	template <typename T>
	constexpr bool operator!=(const list<T, rows>& list_) {
		for (int i = 0; i < rows; i++) {
			if (this->operator[](i) == list_[i]) {
				return false;
			}
		}
		return true;
	}

	constexpr auto& operator*() {
		return *this->begin();
	}

	constexpr void print() const {
		for (int i = 0; i < rows; i++) {
			std::cout << this->operator[](i) << ((i + 1 != rows) ? " " : "\rows");
		}
	}

	constexpr auto u() const noexcept {
		return this->operator-();
	}

	constexpr auto min() const noexcept {
		return *std::min_element(this->begin(), this->end());
	}

	constexpr auto max() const noexcept {
		return *std::max_element(this->begin(), this->end());
	}

	constexpr auto avg() const noexcept {
		return average(this->begin(), this->end());
	}

	constexpr auto abs() const noexcept {
		list l;
		for (int i = 0; i < rows; i++) {
			l[i] = std::abs(this->operator[](i));
		}
		return l;
	}

	constexpr list<f_t, 2> floor() const noexcept {
		list l;
		for (int i = 0; i < rows; i++) {
			l[i] = std::floor(this->operator[](i));
		}
		return l;
	}

	constexpr list<f_t, 2> ceil() const noexcept {
		list l;
		for (int i = 0; i < rows; i++) {
			l[i] = (this->operator[](i) < 0 ? -std::ceil(-this->operator[](i)) : std::ceil(this->operator[](i)));
		}
		return l;
	}

	constexpr list<f_t, 2> round() const noexcept {
		list l;
		for (int i = 0; i < rows; i++) {
			l[i] = std::round(-this->operator[](i));
		}
		return l;
	}

	constexpr f_t pmax() const noexcept {
		decltype(*this) list = this->abs();
		auto biggest = std::max_element(list.begin(), list.end());
		if (this->operator[](biggest - list.begin()) < 0) {
			return -*biggest;
		}
		return *biggest;
	}

	constexpr auto gfne() const noexcept {
		for (int i = 0; i < rows; i++) {
			if (this->operator[](i)) {
				return this->operator[](i);
			}
		}
	}
};

template <typename T>
constexpr bool dcom_fr(uint_t n, T x, T y) noexcept {
	switch (n) {
	case 0: {
		return x < y;
	}
	case 1: {
		return x > y;
	}
	}
}

template <typename type, std::size_t Rows, std::size_t Cols>
struct matrix {

	list<type, Cols> m[Rows];

	using matrix_type = matrix<type, Rows, Cols>;
	using value_type = list<type, Cols>;

	static constexpr std::size_t rows = Rows;
	static constexpr std::size_t cols = Cols;

	constexpr matrix() : m{ 0 } { }

	template <typename _Type, std::size_t cols, template <typename, std::size_t> typename... _List>
	constexpr matrix(_List<_Type, cols>... list_) : m{ 0 } {
		static_assert(sizeof...(list_) >= cols, "too many initializers");
		int i = 0;
		auto auto_type = std::get<0>(std::forward_as_tuple(list_...));
		((((decltype(auto_type)*)m)[i++] = list_), ...);
	}

	template <typename..._Type>
	constexpr matrix(_Type... value) {
		if constexpr (sizeof...(value) == cols * rows) {
			int init = 0;
			((((type*)m)[init++] = value), ...);
		}
	}

	template <typename T>
	constexpr matrix(T value) : m{ 0 } {
		for (int i = 0; i < rows && i < cols; i++) {
			m[i][i] = value;
		}
	}

	template <template <typename> typename... _Vec_t, typename _Type>
	matrix(const _Vec_t<_Type>&... vector) : m{ 0 } {
		int i = 0;
		auto auto_type = std::get<0>(std::forward_as_tuple(vector...));
		((((decltype(auto_type)*)m)[i++] = vector), ...);
	}

	template <typename T, std::size_t Rows2, std::size_t Cols2>
	constexpr matrix_type operator+(const matrix<T, Rows2, Cols2>& matrix) const noexcept { // matrix
		matrix_type _matrix;
		static_assert(Cols2 <= Cols, "Colums of the second matrix is bigger than first");
		for (int i = 0; i < rows; i++) {
			_matrix[i] = m[i] + matrix[i];
		}
		return _matrix;
	}

	template <typename T, std::size_t da_t_n>
	constexpr matrix_type operator+(const list<T, da_t_n>& da_t) const noexcept { // list
		matrix_type _matrix;
		static_assert(cols >= da_t_n, "list is bigger than the matrice's Rows");
		for (int i = 0; i < rows; i++) {
			_matrix[i] = m[i] + da_t;
		}
		return _matrix;
	}

	template <typename T>
	constexpr matrix_type operator+(T value) const noexcept { // basic value
		matrix_type _matrix;
		for (int i = 0; i < rows; i++) {
			_matrix[i] = m[i] + value;
		}
		return _matrix;
	}

	template <typename T, std::size_t Rows2, std::size_t Cols2>
	constexpr auto operator+=(const matrix<T, Rows2, Cols2>& matrix) noexcept { // matrix
		matrix_type _matrix;
		static_assert(Cols2 <= Cols, "Colums of the second matrix is bigger than first");
		for (int i = 0; i < rows; i++) {
			_matrix[i] = m[i] += matrix[i];
		}
		return _matrix;
	}

	template <typename T, std::size_t da_t_n>
	constexpr matrix_type operator+=(const list<T, da_t_n>& da_t) noexcept { // list
		matrix_type _matrix;
		static_assert(cols <= da_t_n, "list is bigger than the matrice's Rows");
		for (int i = 0; i < rows; i++) {
			_matrix[i] = m[i] += da_t;
		}
		return _matrix;
	}

	template <typename T>
	constexpr matrix_type operator+=(T value) noexcept { // basic value
		matrix_type _matrix;
		for (int i = 0; i < rows; i++) {
			_matrix[i] = m[i] += value;
		}
		return _matrix;
	}

	template <typename T, std::size_t Rows2, std::size_t Cols2>
	constexpr matrix_type operator-(const matrix<T, Rows2, Cols2>& matrix) const noexcept { // matrix
		matrix_type _matrix;
		static_assert(Cols2 <= Cols, "Colums of the second matrix is bigger than first");
		for (int i = 0; i < rows; i++) {
			_matrix[i] = m[i] - matrix[i];
		}
		return _matrix;
	}

	template <typename T, std::size_t da_t_n>
	constexpr matrix_type operator-(const list<T, da_t_n>& da_t) const noexcept { // list
		matrix_type _matrix;
		static_assert(cols <= da_t_n, "list is bigger than the matrice's Rows");
		for (int i = 0; i < rows; i++) {
			_matrix[i] = m[i] - da_t;
		}
		return _matrix;
	}

	template <typename T>
	constexpr matrix_type operator-(T value) const noexcept { // basic value
		matrix_type _matrix;
		for (int i = 0; i < rows; i++) {
			_matrix[i] = m[i] - value;
		}
		return _matrix;
	}

	template <typename T, std::size_t Rows2, std::size_t Cols2>
	constexpr matrix_type operator-=(const matrix<T, Rows2, Cols2>& matrix) noexcept { // matrix
		matrix_type _matrix;
		static_assert(Cols2 <= Cols, "Colums of the second matrix is bigger than first");
		for (int i = 0; i < rows; i++) {
			_matrix[i] = m[i] -= matrix[i];
		}
		return _matrix;
	}

	template <typename T, std::size_t da_t_n>
	constexpr matrix_type operator-=(const list<T, da_t_n>& da_t) noexcept { // list
		matrix_type _matrix;
		static_assert(cols <= da_t_n, "list is bigger than the matrice's Rows");
		for (int i = 0; i < rows; i++) {
			_matrix[i] = m[i] -= da_t;
		}
		return _matrix;
	}

	template <typename T>
	constexpr matrix_type operator-=(T value) noexcept { // basic value
		matrix_type _matrix;
		for (int i = 0; i < rows; i++) {
			_matrix[i] = m[i] -= value;
		}
		return _matrix;
	}

	constexpr matrix_type operator-() noexcept {
		for (int _I = 0; _I < Rows; _I++) {
			for (int _J = 0; _J < Cols; _J++) {
				m[_I][_J] = -m[_I][_J];
			}
		}
		return *this;
	}

	constexpr matrix_type operator*(const matrix<type, Rows, Cols>& _Lhs) noexcept {
		if (Rows != Cols) {
			throw("first matrix rows must be same as second's colums");
		}
		for (int _I = 0; _I < Rows; _I++) {
			for (int _J = 0; _J < Cols; _J++) {
				type _Value = 0;
				for (int _K = 0; _K < Cols; _K++) {
					_Value += m[_I][_K] * _Lhs[_K][_J];
				}
				m[_I][_J] = _Value;
			}
		}
		return *this;
	}

	template <typename T>
	constexpr auto operator*=(T value) {
		return this->operator[]<true>(0) *= value;
	}

	template <typename T>
	constexpr auto operator/(T value) const noexcept {
		return this->operator[]<true>(0) / value;
	}

	template <typename T>
	constexpr auto operator/=(T value) {
		return this->operator[]<true>(0) /= value;
	}

	template <typename T>
	constexpr auto operator%(T value) {
		for (int i = 0; i < Cols; i++) {
			for (int j = 0; j < Rows; j++) {
				m[i][j] = fmodf(m[i][j], value);
			}
		}
		return *this;
	}

	constexpr bool operator==(const matrix<type, rows, cols>& list_) {
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				if (m[i][j] != list_[i][j]) {
					return false;
				}
			}
		}
		return true;
	}

	template <bool return_array = false>
	constexpr auto operator[](std::size_t i) const {
		return m[i];
	}

	template <bool return_array = false>
	constexpr auto& operator[](std::size_t i) {
		return m[i];
	}

	constexpr void print() const {
		for (int i = 0; i < rows; i++) {
			m[i].print();
		}
	}

	list<type, Cols>* begin() noexcept {
		return &m[0];
	}

	list<type, Cols>* end() noexcept {
		return &m[rows];
	}

	constexpr auto data() noexcept {
		return begin();
	}

	constexpr auto u() noexcept {
		return this->operator-();
	}

	constexpr auto min() noexcept {
		return *std::min_element(begin(), end());
	}

	constexpr auto max() noexcept {
		return *std::max_element(begin(), end());
	}

	constexpr auto avg() noexcept {
		return average(this->begin(), this->end());
	}

	constexpr auto vector() noexcept {
		return std::vector<da_t<type, cols>>(this->begin(), this->end());
	}

	constexpr auto size() const noexcept {
		return rows;
	}
};

template <typename T, std::size_t rows>
std::ostream& operator<<(std::ostream& os, const list<T, rows> list_) noexcept
{
	for (int i = 0; i < rows; i++) {
		os << list_[i] << ' ';
	}
	return os;
}

template <
	template <typename, std::size_t, std::size_t> typename da_t_t,
	typename T, std::size_t rows, std::size_t cols
>
std::ostream& operator<<(std::ostream& os, const da_t_t<T, rows, cols>& da_t_) noexcept
{
	for (int i = 0; i < cols; i++) {
		for (int j = 0; j < rows; j++) {
			os << da_t_[i][j] << ' ';
		}
	}
	return os;
}

using mat2x2 = matrix<f_t, 2, 2>;
using mat2x3 = matrix<f_t, 2, 3>;
using mat3x2 = matrix<f_t, 3, 2>;
using mat4x2 = matrix<f_t, 4, 2>;
using mat3x3 = matrix<f_t, 3, 3>;
using mat4x4 = matrix<f_t, 4, 4>;

using mat2 = mat2x2;
using mat3 = mat3x3;
using mat4 = mat4x4;

using vec2 = _vec2<f_t>;
using vec3 = _vec3<f_t>;
using vec4 = _vec4<f_t>;
using vec2i = _vec2<int>;
using vec3i = _vec3<int>;
using vec4i = _vec4<int>;

class Color {
public:

	static constexpr Color rgb(f_t r, f_t g, f_t b, f_t a = 255) {
		return Color(r / 255.f, g / 255.f, b / 255.f, a / 255.f);
	}

	static constexpr Color hex(unsigned int hex) {
		if (hex <= 0xffffff) {
			return Color::rgb(
				(hex >> 16) & 0xff,
				(hex >> 8) & 0xff,
				(hex >> 0) & 0xff
			);
		}
		return Color::rgb(
			(hex >> 24) & 0xff,
			(hex >> 16) & 0xff,
			(hex >> 8) & 0xff,
			(hex >> 0) & 0xff
		);
	}

	f_t r, g, b, a;
	Color() : r(0), g(0), b(0), a(1) {}

	constexpr Color(f_t r, f_t g, f_t b, f_t a = 1) : r(r), g(g), b(b), a(a) {
		this->r = r;
		this->g = g;
		this->b = b;
		this->a = a;
	}
	constexpr Color(f_t value) : r(0), g(0), b(0), a(0) {
		this->r = value;
		this->g = value;
		this->b = value;
		this->a = 1;
	}
	Color& operator&=(const Color& color) {
		Color ret;
		ret.r = (unsigned int)r & (unsigned int)color.r;
		ret.g = (unsigned int)g & (unsigned int)color.g;
		ret.b = (unsigned int)b & (unsigned int)color.b;
		ret.a = (unsigned int)a & (unsigned int)color.a;
		return *this;
	}
	Color operator^=(const Color& color) {
		r = (int)r ^ (int)color.r;
		g = (int)g ^ (int)color.g;
		b = (int)b ^ (int)color.b;
		return *this;
	}
	bool operator!=(const Color& color) const {
		return r != color.r || g != color.g || b != color.b;
	}
	constexpr f_t operator[](size_t x) const {
		return !x ? this->r : x == 1 ? this->g : x == 2 ? this->b : x == 3 ? this->a : this->a;
	}
	constexpr Color operator-=(const Color& color) {
		return Color(r -= color.r, g -= color.g, b -= color.b, a -= color.a);
	}
	constexpr Color operator-(const Color& color) const {
		return Color(r - color.r, g - color.g, b - color.b, a);
	}
	constexpr Color operator+(const Color& color) const {
		return Color(r + color.r, g + color.g, b + color.b);
	}
	void print() const {
		std::cout << r << " " << g << " " << b << " " << a << std::endl;
	}
	void* data() {
		return &r;
	}
};

static Color random_color() {
	return Color::rgb(std::rand() % 255, std::rand() % 255, std::rand() % 255, 255);
}

template <typename _Casted, template<typename> typename _Vec_t, typename _Old>
constexpr _Vec_t<_Casted> Cast(_Vec_t<_Old> v) noexcept
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