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

using float_t = float;
using uint_t = unsigned int;
constexpr float_t INF = INFINITY;

template <typename _Ty>
class _vec3;

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
	constexpr _vec2(const std::array<type, 2>& _array) : x(_array[0]), y(_array[1]) {}

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

	template <typename _Type>
	constexpr bool operator==(_Type single_value) const noexcept {
		return this->x == single_value && this->y == single_value;
	}

	template <typename _Type>
	constexpr bool operator!=(const _vec2<_Type>& vector) const noexcept {
		return this->x != vector.x || this->y != vector.y;
	}

	template <typename _Type>
	constexpr bool operator!=(_Type single_value) const noexcept {
		return this->x != single_value || this->y != single_value;
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
		return _vec2<_Type>(this->x / single_value, this->y / single_value);
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

template <typename _Ty = float_t>
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

template <typename _Ty = float_t>
class _vec4 {
public:
	_Ty x, y, z, w;

	using type = _Ty;
	using vec_t = _vec3<_Ty>;

	constexpr _vec4() : x(0), y(0), z(0), w(0) { }
	constexpr _vec4(_Ty x, _Ty y, _Ty z, _Ty w) : x(x), y(y), z(z), w(w) { }
	constexpr _vec4(_Ty value) : x(value), y(value), z(value), w(value) { }

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

template <typename Arg, typename ...Args>
void LOG(const Arg& arg, const Args&... args) {
	std::cout << arg;
	((std::cout << " " << args), ...) << '\n';
}

template <int Cols, int Rows, typename _Ty = float>
class matrix {
public:
	std::array<_Ty, Cols> m[Rows];
	using type = _Ty;

	constexpr matrix() : m{ 0 } { }

	template <template<typename> typename _Vec_t, typename _Type, typename _Type2>
	constexpr matrix(const _Vec_t<_Type>& lhs, const _Vec_t<_Type2>& rhs) : m{ 0 } {
		static_assert(Rows == 2, "No support for more or less than 2x size matrix");
		static_assert(Cols == _Vec_t<_Type>::size(), "vector size and rows do not match");
		for (int i = 0; i < lhs.size(); i++) {
			m[0][i] = lhs[i];
		}
		for (int i = 0; i < rhs.size(); i++) {
			m[1][i] = rhs[i];
		}
	}

	template <typename ..._Type>
	constexpr matrix(_Ty _Val, _Type&&... _array) : m{ 0 } {
		if constexpr (!sizeof...(_array)) {
			for (int _I = 0; _I < Rows; _I++) {
				m[_I][_I] = _Val;
			}
		}
		else {
			static_assert(sizeof...(_array) > Rows * Cols - 2, "too few initializer values");
			static_assert(sizeof...(_array) + 1 <= Rows * Cols, "too many initializer values");
			int init = 0;
			((((_Ty*)m)[init++] = _array), ...);
		}
	}

	template <int m_cols, int m_rows>
	constexpr matrix(const matrix<m_cols, m_rows>& matrix) : m{ 0 } {
		for (int i = 0; i < matrix.cols(); i++) {
			for (int j = 0; j < matrix.rows(); j++) {
				if (i >= Cols || j >= Rows) {
					break;
				}
				this->m[i][j] = matrix[i][j];
			}
		}
	}

	std::array<_Ty, Rows>& operator[](uint64_t _Idx) {
		return m[_Idx];
	}

	auto operator[](uint64_t _Idx) const {
		return m[_Idx];
	}

	constexpr uint64_t rows() const {
		return Rows;
	}

	constexpr uint64_t cols() const {
		return Cols;
	}

	constexpr uint64_t size() const {
		return Rows * Cols;
	}

	matrix<Cols, Rows> operator-(const matrix<Cols, Rows>& m) {
		matrix<Cols, Rows> ret;
		for (int _I = 0; _I < Rows; _I++) {
			for (int _J = 0; _J < Cols; _J++) {
				ret[_I][_J] = -m[_I][_J];
			}
		}
		return ret;
	}

	void print() const {
		for (int _I = 0; _I < Rows; _I++) {
			for (int _J = 0; _J < Cols; _J++) {
				std::cout << m[_I][_J] << std::endl;
			}
		}
	}

	constexpr matrix<Cols, Rows, _Ty> operator*(const matrix<Cols, Rows, _Ty>& _Lhs) {
		if (rows() != _Lhs.cols()) {
			throw("first matrix rows must be same as second's colums");
		}
		for (int _I = 0; _I < Rows; _I++) {
			for (int _J = 0; _J < Cols; _J++) {
				_Ty _Value = 0;
				for (int _K = 0; _K < Cols; _K++) {
					_Value += m[_I][_K] * _Lhs[_K][_J];
				}
				m[_I][_J] = _Value;
			}
		}
		return *this;
	}

	_Ty* data() {
		return &m[0][0];
	}
};

using mat2x2 = matrix<2, 2>;
using mat2x3 = matrix<3, 2>;
using mat2x4 = matrix<4, 2>;
using mat3x3 = matrix<3, 3>;
using mat2 = mat2x2;
using mat3 = mat3x3;
using mat4 = matrix<4, 4>;
using vec2 = _vec2<float_t>;
using vec3 = _vec3<float_t>;
using vec4 = _vec4<float_t>;
using vec2i = _vec2<int>;
using vec3i = _vec3<int>;
using vec4i = _vec4<int>;

class Color {
public:
	float_t r, g, b, a;
	Color() : r(0), g(0), b(0), a(1) {}

	constexpr Color(float r, float g, float b, float a = 1, bool rgb = false) : r(r), g(g), b(b), a(a) {
		if (rgb) {
			this->r = r / 255.f;
			this->g = g / 255.f;
			this->b = b / 255.f;
			this->a = a / 255.f;
		}
		else {
			this->r = r;
			this->g = g;
			this->b = b;
			this->a = a;
		}
	}
	constexpr Color(float value, bool index = false) : r(0), g(0), b(0), a(0) {
		if (index) {
			this->r = r / value;
			this->g = g / value;
			this->b = b / value;
			this->a = a / value;
		}
		else {
			this->r = value;
			this->g = value;
			this->b = value;
			this->a = 1;
		}
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
	constexpr float_t operator[](size_t x) const {
		return !x ? this->r : x == 1 ? this->g : x == 2 ? this->b : x == 3 ? this->a : this->a;
	}
	constexpr Color operator-(const Color& color) const {
		return Color(r - color.r, g - color.g, b - color.b, a - color.a);
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

template <typename _Type, uint64_t _Size>
constexpr std::array<_Type, _Size> operator+=(std::array<_Type, _Size>& _Lhs, const std::array<_Type, _Size>& _Rhs) noexcept
{
	std::transform(_Lhs.begin(), _Lhs.end(), _Rhs.cbegin(), _Lhs.begin(), std::plus<void>());
	return _Lhs;
}

template <typename _Type, uint64_t _Size, typename _Type2>
constexpr std::array<_Type, _Size> operator+=(const std::array<_Type, _Size>& _Lhs, _Type2 _Rhs) noexcept
{
	std::array<_Type, _Size> _Multiplication;
	_Multiplication.fill(_Rhs);
	std::transform(_Lhs.begin(), _Lhs.end(), _Multiplication.cbegin(), _Lhs.begin(), std::plus<void>());
	return _Lhs;
}

template <typename _Type, uint64_t _Size>
constexpr std::array<_Type, _Size> operator+(const std::array<_Type, _Size>& _Lhs, const std::array<_Type, _Size>& _Rhs) noexcept
{
	std::array<_Type, _Size> _Array(_Lhs);
	std::transform(_Array.begin(), _Array.end(), _Rhs.cbegin(), _Array.begin(), std::plus<void>());
	return _Array;
}

template <typename _Type, uint64_t _Size, typename _Type2>
constexpr std::array<_Type, _Size> operator+(const std::array<_Type, _Size>& _Lhs, _Type2 _Rhs) noexcept  
{
	std::array<_Type, _Size> _Array(_Lhs);
	std::array<_Type, _Size> _Addition;
	_Addition.fill(_Rhs);
	std::transform(_Array.begin(), _Array.end(), _Addition.cbegin(), _Array.begin(), std::plus<void>());
	return _Array;
}

template <typename _Type, uint64_t _Size>
constexpr std::array<_Type, _Size> operator-=(std::array<_Type, _Size>& _Lhs, const std::array<_Type, _Size>& _Rhs) noexcept
{
	std::transform(_Lhs.begin(), _Lhs.end(), _Rhs.cbegin(), _Lhs.begin(), std::minus<void>());
	return _Lhs;
}

template <typename _Type, uint64_t _Size, typename _Type2>
constexpr std::array<_Type, _Size> operator-=(const std::array<_Type, _Size>& _Lhs, _Type2 _Rhs) noexcept
{
	std::array<_Type, _Size> _Multiplication;
	_Multiplication.fill(_Rhs);
	std::transform(_Lhs.begin(), _Lhs.end(), _Multiplication.cbegin(), _Lhs.begin(), std::minus<void>());
	return _Lhs;
}

template <typename _Type, uint64_t _Size>
constexpr std::array<_Type, _Size> operator-(const std::array<_Type, _Size>& _Lhs, const std::array<_Type, _Size>& _Rhs) noexcept
{
	std::array<_Type, _Size> _Array(_Lhs);
	std::transform(_Array.begin(), _Array.end(), _Rhs.cbegin(), _Array.begin(), std::minus<void>());
	return _Array;
}

template <typename _Type, uint64_t _Size, typename _Type2>
constexpr std::array<_Type, _Size> operator-(const std::array<_Type, _Size>& _Lhs, _Type2 _Rhs) noexcept
{
	std::array<_Type, _Size> _Array(_Lhs);
	std::array<_Type, _Size> _Subtraction;
	_Subtraction.fill(_Rhs);
	std::transform(_Array.begin(), _Array.end(), _Subtraction.cbegin(), _Array.begin(), std::minus<void>());
	return _Array;
}

template <typename _Type, uint64_t _Size>
constexpr std::array<_Type, _Size> operator*=(std::array<_Type, _Size>& _Lhs, const std::array<_Type, _Size>& _Rhs) noexcept
{
	std::transform(_Lhs.begin(), _Lhs.end(), _Rhs.cbegin(), _Lhs.begin(), std::multiplies<void>());
	return _Lhs;
}

template <typename _Type, uint64_t _Size, typename _Type2>
constexpr std::array<_Type, _Size> operator*=(std::array<_Type, _Size>& _Lhs, _Type2 _Rhs) noexcept
{
	std::array<_Type, _Size> _Multiplication;
	_Multiplication.fill(_Rhs);
	std::transform(_Lhs.begin(), _Lhs.end(), _Multiplication.cbegin(), _Lhs.begin(), std::multiplies<void>());
	return _Lhs;
}

template <typename _Type, uint64_t _Size>
constexpr std::array<_Type, _Size> operator*(const std::array<_Type, _Size>& _Lhs, const std::array<_Type, _Size>& _Rhs) noexcept
{
	std::array<_Type, _Size> _Array(_Lhs);
	std::transform(_Array.begin(), _Array.end(), _Rhs.cbegin(), _Array.begin(), std::multiplies<void>());
	return _Array;
}

template <typename _Type, uint64_t _Size, typename _Type2>
constexpr std::array<_Type, _Size> operator*(const std::array<_Type, _Size>& _Lhs, _Type2 _Rhs) noexcept 
{
	std::array<_Type, _Size> _Array(_Lhs);
	std::array<_Type, _Size> _Multiplication{ 0 }; // gcc note implicitly-defined constructor does not initialize - fix {0}
	_Multiplication.fill(_Rhs);
	std::transform(_Array.begin(), _Array.end(), _Multiplication.cbegin(), _Array.begin(), std::multiplies<void>());
	return _Array;
}

template <typename _Type, uint64_t _Size>
constexpr std::array<_Type, _Size> operator/=(std::array<_Type, _Size>& _Lhs, const std::array<_Type, _Size>& _Rhs) noexcept
{
	std::transform(_Lhs.begin(), _Lhs.end(), _Rhs.cbegin(), _Lhs.begin(), std::divides<void>());
	return _Lhs;
}

template <typename _Type, uint64_t _Size, typename _Type2>
constexpr std::array<_Type, _Size> operator/=(const std::array<_Type, _Size>& _Lhs, _Type2 _Rhs) noexcept
{
	std::array<_Type, _Size> _Division;
	_Division.fill(_Rhs);
	std::transform(_Lhs.begin(), _Lhs.end(), _Division.cbegin(), _Lhs.begin(), std::divides<void>());
	return _Lhs;
}

template <typename _Type, uint64_t _Size>
constexpr std::array<_Type, _Size> operator/(const std::array<_Type, _Size>& _Lhs, const std::array<_Type, _Size>& _Rhs) noexcept
{
	std::array<_Type, _Size> _Array(_Lhs);
	std::transform(_Array.begin(), _Array.end(), _Rhs.cbegin(), _Array.begin(), std::divides<void>());
	return _Array;
}

template <typename _Type, uint64_t _Size, typename _Type2>
constexpr std::array<_Type, _Size> operator/(const std::array<_Type, _Size>& _Lhs, _Type2 _Rhs) noexcept
{
	std::array<_Type, _Size> _Array(_Lhs);
	std::array<_Type, _Size> _Division;
	_Division.fill(_Rhs);
	std::transform(_Array.begin(), _Array.end(), _Division.cbegin(), _Array.begin(), std::divides<void>());
	return _Array;
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