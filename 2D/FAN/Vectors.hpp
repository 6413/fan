#pragma once
#ifdef _MSC_VER
#pragma warning (disable : 4244)
#pragma warning (disable : 26451)
#endif

#include <iostream>
#include <cmath>
#include <algorithm>

template <typename _Ty, uint64_t N>
class array {
public:
	_Ty _array[N];

	constexpr array<_Ty, N> operator*(_Ty value) {
		for (int i = 0; i < N; i++) {
			_array[i] *= value;
		}
		return *this;
	}
	constexpr array<_Ty, N> operator+(const array<_Ty, N>& lhs) {
		for (int i = 0; i < N; i++) {
			_array[i] += lhs[i];
		}
		return *this;
	}
	constexpr array<_Ty, N> operator+(_Ty value) {
		for (int i = 0; i < N; i++) {
			_array[i] += value;
		}
		return *this;
	}
	constexpr auto& operator[](uint64_t idx) {
		return _array[idx];
	}
	constexpr auto operator[](uint64_t idx) const {
		return _array[idx];
	}
};

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
	constexpr _vec2(const _vec3<type>& vec, bool init) : x(vec.x), y(vec.y) { }

	constexpr _Ty& operator[](uint64_t idx) { return !idx ? x : y; }
	constexpr _Ty operator[](uint64_t idx) const { return !idx ? x : y; }
	//constexpr vec_t operator*(_Ty val) const { return vec_t(x * val, y * val); }

	constexpr vec_t floored() const { return vec_t(floor(x), floor(y)); }
	constexpr vec_t floored(_Ty value) const { return vec_t(floor(x / value), floor(y / value)); }

	static constexpr uint64_t size() { return 2; }
	constexpr vec_t abs() const { return vec_t(std::abs(x), std::abs(y)); }
	constexpr _Ty min() const { return std::min(x, y); }
	constexpr _Ty max() const { return std::max(x, y); }

	constexpr void print() const { std::cout << x << " " << y << std::endl; }
};

template <typename _Ty = float>
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

	constexpr _Ty& operator[](uint64_t idx) { return !idx ? x : idx == 1 ? y : z; }
	constexpr _Ty operator[](uint64_t idx) const { return !idx ? x : idx == 1 ? y : z; }
	//constexpr vec_t operator*(_Ty val) const { return vec_t(x * val, y * val, z * val); }

	constexpr vec_t floored() const { return vec_t(floor(x), floor(y), floor(z)); }
	constexpr vec_t floored(_Ty value) const { return vec_t(floor(x / value), floor(y / value), floor(z / value)); }
	constexpr vec_t ceiled() const { return vec_t(ceil(x), ceil(y), ceil(z)); }
	constexpr vec_t rounded() const { return vec_t(round(x), round(y), round(z)); }

	constexpr _Ty min()const { return std::min({ x, y, z }); }
	constexpr _Ty max()const { return std::max({ x, y, z }); }
	constexpr vec_t abs() const { return vec_t(std::abs(x), std::abs(y), std::abs(z)); }

	auto begin() { return &operator[](0); }
	auto end() { return &operator[](size() - 1); }
	auto data() { return begin(); }
	static constexpr uint64_t size() { return 3; }

	constexpr void print() const { std::cout << x << " " << y << " " << z << std::endl; }
};

template <typename _Ty = float>
class _vec4 {
public:
	_Ty x, y, z, w;

	constexpr _vec4() : x(0), y(0), z(0), w(0) { }
	constexpr _vec4(_Ty x, _Ty y, _Ty z, _Ty w) : x(x), y(y), z(z), w(w) { }
	constexpr _vec4(_Ty value) : x(value), y(value), z(value), w(value) { }
	template <typename type, typename type2>
	constexpr _vec4(const _vec3<type> vec, type2 value) : x(vec.x), y(vec.y), z(vec.z), w(value) { }
	template <typename type>
	constexpr _vec4(const _vec4<type>& vec) : x(vec.x), y(vec.y), z(vec.z), w(vec.w) { }

	constexpr _Ty& operator[](uint64_t idx) { return !idx ? x : idx == 1 ? y : idx == 2 ? z : w; }
	constexpr _Ty operator[](uint64_t idx) const { return !idx ? x : idx == 1 ? y : idx == 2 ? z : w; }

	constexpr _vec4<_Ty> floored() const { return _vec4<_Ty>(floor(x), floor(y), floor(z), floor(w)); }
	constexpr _vec4<_Ty> floored(_Ty value) const { return _vec4<_Ty>(floor(x / value), floor(y / value), floor(z / value), floor(w / value)); }

	static constexpr uint64_t size() { return 4; }

	constexpr void print() const { std::cout << x << " " << y << " " << z << " " << w << std::endl; }
};

template <int Cols, int Rows, typename _Ty = float>
class matrix {
public:
	array<_Ty, Cols> m[Rows];
	using type = _Ty;

	constexpr matrix() : m{ } {
		for (int _I = 0; _I < Cols; _I++) {
			for (int _J = 0; _J < Rows; _J++) {
				m[_I][_J] = 0;
			}
		}
	}

	template <template<typename> typename _Vec_t, typename _Type>
	matrix(const _Vec_t<_Type>& lhs, const _Vec_t<_Type>& rhs) {
		static_assert(Rows == 2, "No support for more or less than 2x size matrix");
		static_assert(Cols == lhs.size(), "vector size and rows do not match");
		for (int i = 0; i < lhs.size(); i++) {
			m[0][i] = lhs[i];
		}
		for (int i = 0; i < rhs.size(); i++) {
			m[1][i] = rhs[i];
		}
	}

	template <typename ..._Type>
	matrix(_Ty _Val, _Type&&... _array) {
		if constexpr (!sizeof...(_array)) {
			for (int _I = 0; _I < Rows; _I++) {
				for (int _J = 0; _J < Cols; _J++) {
					m[_I][_J] = 0;
				}
			}
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

	array<_Ty, Rows>& operator[](uint64_t _Idx) {
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

	_Ty* data() const {
		return &m[0][0];
	}
};

template <typename _Ty>
class _mat2x2 {
public:
	constexpr _vec2<_Ty>& operator[](uint64_t idx) {
		return v[idx];
	}
	constexpr _vec2<_Ty> operator[](uint64_t idx) const {
		return v[idx];
	}

	constexpr _mat2x2() : v{ 0 } {}

	constexpr _mat2x2(_Ty x, _Ty y, _Ty z, _Ty a) : init_index(0) {
		v[0][0] = x;
		v[0][1] = y;
		v[1][0] = z;
		v[1][1] = a;
	}

	constexpr _mat2x2(const _vec2<_Ty>& first, const _vec2<_Ty> second) {
		v[0] = first;
		v[1] = second;
	}

private:
	_vec2<_Ty> v[2];
	uint64_t init_index;
};

template <typename _Ty>
class _mat2x4 {
public:
	_mat2x4() : v{} {}

	template <typename T, typename ...type>
	constexpr _mat2x4(T value, type&&... args) : init_index(0) {
		initialize(value, args...);
	}

	constexpr _vec2<_Ty>& operator[](uint64_t idx) {
		return v[idx];
	}
	constexpr _vec2<_Ty> operator[](uint64_t idx) const {
		return v[idx];
	}

	void initialize() {}

	template <typename T, typename ...type>
	constexpr void initialize(T value, type&&... args) {
		((T*)v)[init_index] = value;
		init_index++;
		initialize(args...);
	}

private:
	_vec2<_Ty> v[4];
	uint64_t init_index;
};

using mat2x2 = _mat2x2<float>;
using mat2x3 = _vec3<float>[2];
using mat2x4 = _mat2x4<float>;
using vec2 = _vec2<float>;
using vec3 = _vec3<float>;
using vec4 = _vec4<float>;
using vec2i = _vec2<int>;
using vec3i = _vec3<int>;
using vec4i = _vec4<int>;

class Color {
public:
	float r, g, b, a;
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
		return r != color.r && g != color.g && b != color.b;
	}
	constexpr float operator[](size_t x) const {
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
};

inline auto operator*(const matrix<4, 4>& _Mat, const vec4& _Vec) {
	vec4 return_vector;
	return_vector.x = _Mat[0][0] * _Vec.x + _Mat[0][1] * _Vec.y + _Mat[0][2] * _Vec.z + _Mat[0][3] * _Vec.w;
	return_vector.y = _Mat[1][0] * _Vec.x + _Mat[1][1] * _Vec.y + _Mat[1][2] * _Vec.z + _Mat[1][3] * _Vec.w;
	return_vector.z = _Mat[2][0] * _Vec.x + _Mat[2][1] * _Vec.y + _Mat[2][2] * _Vec.z + _Mat[2][3] * _Vec.w;
	return_vector.w = _Mat[3][0] * _Vec.x + _Mat[3][1] * _Vec.y + _Mat[3][2] * _Vec.z + _Mat[3][3] * _Vec.w;
	return return_vector;
}

template <typename _Casted, template<typename> typename _Vec_t, typename _Old>
constexpr _Vec_t<_Casted> Cast(_Vec_t<_Old> v) {
	return _Vec_t<_Casted>(v);
}

template <class, class>
constexpr bool is_same_v = false;
template <class _Ty>
constexpr bool is_same_v<_Ty, _Ty> = true;

template<typename T, typename... Rest>
struct is_any : std::false_type {};

template<typename T, typename First>
struct is_any<T, First> : std::is_same<T, First> {};

template<typename T, typename First, typename... Rest>
struct is_any<T, First, Rest...>
	: std::integral_constant<bool, std::is_same<T, First>::value || is_any<T, Rest...>::value>
{};

template<typename T, typename... Rest>
struct are_same : std::false_type {};

template<typename T, typename First>
struct are_same<T, First> : std::is_same<T, First> {};

template<typename T, typename First, typename... Rest>
struct are_same<T, First, Rest...>
	: std::integral_constant<bool, std::is_same<T, First>::value && is_any<T, Rest...>::value>
{};

template <template<typename> typename _Vec_t, typename _Type,
	std::enable_if_t<is_any<_Vec_t<_Type>, _vec2<_Type>, _vec3<_Type>, _vec4<_Type>>::value>* = nullptr>
constexpr std::ostream& operator<<(std::ostream& _Os, const _Vec_t<_Type>& _Lhs) { 
	for (int i = 0; i < _Lhs.size(); i++) {
		if (i - 1 != _Lhs.size()) {
			_Os << _Lhs[i] << ' ';
		}
		else {
			_Os << _Lhs[i];
		}
	}
	return _Os;
}

template <template<typename> typename _Vec_t, typename _Type, typename _Type2>
constexpr _Vec_t<_Type> operator+(const _Vec_t<_Type>& _Lhs, _Vec_t<_Type2> _Rhs) {
	_Vec_t<_Type> _Vec;
	for (int _I = 0; _I < _Vec.size(); _I++) {
		_Vec[_I] = _Lhs[_I] + _Rhs[_I];
	}
	return _Vec;
}

template <template<typename> typename _Vec_t, typename _Type, typename _Type2,
	std::enable_if_t<is_any<_Vec_t<_Type>, _vec2<_Type>, _vec3<_Type>, _vec4<_Type>>::value>* = nullptr>
constexpr _Vec_t<_Type> operator+(const _Vec_t<_Type>& _Lhs, _Type2 _Rhs) {
	_Vec_t<_Type> _Vec;
	for (int _I = 0; _I < _Vec.size(); _I++) {
		_Vec[_I] = _Lhs[_I] + _Rhs;
	}
	return _Vec;
}

template <template<typename> typename _Vec_t, typename _Type, typename _Type2,
	std::enable_if_t<is_any<_Vec_t<_Type>, _vec2<_Type>, _vec3<_Type>, _vec4<_Type>>::value>* = nullptr>
constexpr _Vec_t<_Type>& operator+=(_Vec_t<_Type>& _Lhs, const _Vec_t<_Type2>& _Rhs) {
	for (int _I = 0; _I < _Lhs.size(); _I++) {
		_Lhs[_I] = _Lhs[_I] + _Rhs[_I];
	}
	return _Lhs;
}

template <template<typename> typename _Vec_t, typename _Type, typename _Type2,
	std::enable_if_t<is_any<_Vec_t<_Type>, _vec2<_Type>, _vec3<_Type>, _vec4<_Type>>::value>* = nullptr>
	constexpr _Vec_t<_Type>& operator+=(_Vec_t<_Type>& _Lhs, _Type2 _Rhs) {
	for (int _I = 0; _I < _Lhs.size(); _I++) {
		_Lhs[_I] = _Lhs[_I] + _Rhs;
	}
	return _Lhs;
}

template <template<typename> typename _Vec_t, typename _Type, typename _Type2,
	std::enable_if_t<is_any<_Vec_t<_Type>, _vec2<_Type>, _vec3<_Type>, _vec4<_Type>>::value>* = nullptr>
constexpr _Vec_t<_Type>& operator-=(_Vec_t<_Type>& _Lhs, const _Vec_t<_Type2>& _Rhs) {
	for (int _I = 0; _I < _Lhs.size(); _I++) {
		_Lhs[_I] = _Lhs[_I] - _Rhs[_I];
	}
	return _Lhs;
}

template <template<typename> typename _Vec_t, typename _Type, typename _Type2,
	std::enable_if_t<is_any<_Vec_t<_Type>, _vec2<_Type>, _vec3<_Type>, _vec4<_Type>>::value>* = nullptr>
constexpr _Vec_t<_Type> operator-(const _Vec_t<_Type>& _Lhs, _Vec_t<_Type2> _Rhs) {
	_Vec_t<_Type> _Vec;
	for (int _I = 0; _I < _Vec.size(); _I++) {
		_Vec[_I] = _Lhs[_I] - _Rhs[_I];
	}
	return _Vec;
}

//template <template<typename> typename _Vec_t, typename _Type, typename _Type2,
//	std::enable_if_t<is_any<_Vec_t<_Type>, _vec2<_Type>, _vec3<_Type>, _vec4<_Type>>::value &&
//	!is_any<_Type2, _vec2<_Type2>, _vec3<_Type2>, _vec4<_Type2>>::value>* = nullptr>
//constexpr _Vec_t<_Type> operator-(const _Vec_t<_Type>& _Lhs, _Type2 _Rhs) {
//	_Vec_t<_Type> _Vec;
//	for (int _I = 0; _I < _Vec.size(); _I++) {
//		_Vec[_I] = _Lhs[_I] - _Rhs;
//	}
//	return _Vec;
//}

template <template<typename> typename _Vec_t, typename _Type,
	std::enable_if_t<is_any<_Vec_t<_Type>, _vec2<_Type>, _vec3<_Type>, _vec4<_Type>>::value>* = nullptr>
	constexpr _Vec_t<_Type> operator-(const _Vec_t<_Type>& _Lhs, double _Rhs) {
	_Vec_t<_Type> _Vec;
	for (int _I = 0; _I < _Vec.size(); _I++) {
		_Vec[_I] = _Lhs[_I] - _Rhs;
	}
	return _Vec;
}

template <template<typename> typename _Vec_t, typename _Type,
	std::enable_if_t<is_any<_Vec_t<_Type>, _vec2<_Type>, _vec3<_Type>, _vec4<_Type>>::value>* = nullptr>
constexpr _Vec_t<_Type> operator-(_Vec_t<_Type> _Vec) {
	for (int _I = 0; _I < _Vec.size(); _I++) {
		_Vec[_I] = -_Vec[_I];
	}
	return _Vec;
}

template <template<typename> typename _Vec_t, typename _Type, typename _Type2,
	std::enable_if_t<is_any<_Vec_t<_Type>, _vec2<_Type>, _vec3<_Type>, _vec4<_Type>>::value>* = nullptr>
constexpr _Vec_t<_Type> operator*(const _Vec_t<_Type>& _Lhs, const _Vec_t<_Type2>& _Rhs) {
	_Vec_t<_Type> _Vec;
	for (int _I = 0; _I < _Vec.size(); _I++) {
		_Vec[_I] = _Lhs[_I] * _Rhs[_I];
	}
	return _Vec;
}

template <template<typename> typename _Vec_t, typename _Type, typename _Type2,
	std::enable_if_t<is_any<_Vec_t<_Type>, _vec2<_Type>, _vec3<_Type>, _vec4<_Type>>::value &&
	!is_any<_Type2, _vec2<_Type2>, _vec3<_Type2>, _vec4<_Type2>>::value>* = nullptr>
constexpr _Vec_t<_Type> operator*(const _Vec_t<_Type>& _Lhs, _Type2 _Rhs) {
	_Vec_t<_Type> _Vec;
	for (int _I = 0; _I < _Vec.size(); _I++) {
		_Vec[_I] = _Lhs[_I] * _Rhs;
	}
	return _Vec;
}

template <template<typename> typename _Vec_t, typename _Type, typename _Type2,
	std::enable_if_t<is_any<_Vec_t<_Type>, _vec2<_Type>, _vec3<_Type>, _vec4<_Type>>::value>* = nullptr>
constexpr _Vec_t<_Type> operator/(const _Vec_t<_Type>& _Lhs, const _Vec_t<_Type2>& _Rhs) {
	_Vec_t<_Type> _Vec;
	for (int _I = 0; _I < _Vec.size(); _I++) {
		_Vec[_I] = _Lhs[_I] / _Rhs[_I];
	}
	return _Vec;
}

template <template<typename> typename _Vec_t, typename _Type, typename _Type2,
	std::enable_if_t<is_any<_Vec_t<_Type>, _vec2<_Type>, _vec3<_Type>, _vec4<_Type>>::value &&
	!is_any<_Type2, _vec2<_Type2>, _vec3<_Type2>, _vec4<_Type2>>::value>* = nullptr>
	constexpr _Vec_t<_Type> operator/=(_Vec_t<_Type>& _Lhs, _Type2 _Rhs) {
	for (int _I = 0; _I < _Lhs.size(); _I++) {
		_Lhs[_I] /= _Rhs;
	}
	return _Lhs;
}

template <template<typename> typename _Vec_t, typename _Type, typename _Type2,
	std::enable_if_t<is_any<_Vec_t<_Type>, _vec2<_Type>, _vec3<_Type>, _vec4<_Type>>::value &&
	!is_any<_Type2, _vec2<_Type2>, _vec3<_Type2>, _vec4<_Type2>>::value>* = nullptr>
constexpr _Vec_t<_Type> operator/(const _Vec_t<_Type>& _Lhs, _Type2 _Rhs) {
	_Vec_t<_Type> _Vec;
	for (int _I = 0; _I < _Vec.size(); _I++) {
		_Vec[_I] = _Lhs[_I] / _Rhs;
	}
	return _Vec;
}

template <template<typename> typename _Vec_t, typename _Type, typename _Type2,
	std::enable_if_t<is_any<_Vec_t<_Type>, _vec2<_Type>, _vec3<_Type>, _vec4<_Type>>::value &&
	!is_any<_Type2, _vec2<_Type2>, _vec3<_Type2>, _vec4<_Type2>>::value>* = nullptr>
constexpr _Vec_t<_Type> operator%(const _Vec_t<_Type>& _Lhs, _Type2 _Rhs) {
	_Vec_t<_Type> _Vec;
	for (int _I = 0; _I < _Lhs.size(); _I++) {
		_Vec[_I] = fmod(_Lhs[_I], _Rhs);
	}
	return _Vec;
}

template <template<typename> typename _Vec_t, typename _Type, 
	std::enable_if_t<is_any<_Vec_t<_Type>, _vec2<_Type>, _vec3<_Type>>::value>* = nullptr>
constexpr bool operator!(const _Vec_t<_Type>& _Lhs) {
	for (int _I = 0; _I < _Lhs.size() - 1; _I+=2) {
		if (_Lhs[_I] || _Lhs[_I + 1]) {
			return true;
		}
	}
	return false;
}

template <template<typename> typename _Vec_t = _vec3, typename _Type, typename _Type2,
	std::enable_if_t<is_any<_Vec_t<_Type>, _vec2<_Type>, _vec3<_Type>, _vec4<_Type>>::value && 
	!is_any<_Type2, _vec2<_Type>, _vec3<_Type>, _vec4<_Type>>::value>* = nullptr>
constexpr bool operator!=(const _Vec_t<_Type>& _Lhs, _Type2 _Rhs) {
	for (int _I = 0; _I < _Lhs.size(); _I++) {
		if (_Lhs[_I] == _Rhs) {
			return false;
		}	
	}
	return true;
}

template <template<typename> typename _Vec_t, typename _Type, typename _Type2,
	std::enable_if_t<is_any<_Vec_t<_Type>, _vec2<_Type>, _vec3<_Type>, _vec4<_Type>>::value &&
	!is_any<_Type2, _vec2<_Type2>, _vec3<_Type2>, _vec4<_Type2>>::value>* = nullptr >
constexpr bool operator==(const _Vec_t<_Type>& _Lhs, _Type2 _Rhs) {
	for (int _I = 0; _I < _Lhs.size(); _I++) {
		if (_Lhs[_I] != _Rhs) {
			return false;
		}
	}
	return true;
}

template <template<typename> typename _Vec_t, typename _Type, typename _Type2,
	std::enable_if_t<is_any<_Vec_t<_Type>, _vec2<_Type>, _vec3<_Type>, _vec4<_Type>>::value>* = nullptr>
constexpr bool operator==(const _Vec_t<_Type>& _Lhs, const _Vec_t<_Type2>& _Rhs) {
	for (int _I = 0; _I < _Lhs.size(); _I++) {
		if (_Lhs[_I] != _Rhs[_I]) {
			return false;
		}
	}
	return true;
}

template <template<typename> typename _Vec_t, typename _Type, typename _Type2,
	std::enable_if_t<is_any<_Vec_t<_Type>, _vec2<_Type>, _vec3<_Type>, _vec4<_Type>>::value>* = nullptr>
	constexpr bool operator<(const _Vec_t<_Type>& _Lhs, const _Vec_t<_Type2>& _Rhs) {
	for (int _I = 0; _I < _Lhs.size(); _I++) {
		if (_Lhs[_I] >= _Rhs[_I]) {
			return false;
		}
	}
	return true;
}

template <template<typename> typename _Vec_t, typename _Type, typename _Type2,
	std::enable_if_t<is_any<_Vec_t<_Type>, _vec2<_Type>, _vec3<_Type>, _vec4<_Type>>::value>* = nullptr>
	constexpr bool operator>(const _Vec_t<_Type>& _Lhs, const _Vec_t<_Type2>& _Rhs) {
	for (int _I = 0; _I < _Lhs.size(); _I++) {
		if (_Lhs[_I] <= _Rhs[_I]) {
			return false;
		}
	}
	return true;
}

template <template<typename> typename _Vec_t, typename _Type, typename _Type2,
	std::enable_if_t<is_any<_Vec_t<_Type>, _vec2<_Type>, _vec3<_Type>, _vec4<_Type>>::value>* = nullptr>
	constexpr bool operator>=(const _Vec_t<_Type>& _Lhs, const _Vec_t<_Type2>& _Rhs) {
	for (int _I = 0; _I < _Lhs.size(); _I++) {
		if (_Lhs[_I] < _Rhs[_I]) {
			return false;
		}
	}
	return true;
}

template <template<typename> typename _Vec_t, typename _Type, typename _Type2,
	std::enable_if_t<is_any<_Vec_t<_Type>, _vec2<_Type>, _vec3<_Type>, _vec4<_Type>>::value>* = nullptr>
	constexpr bool operator<=(const _Vec_t<_Type>& _Lhs, const _Vec_t<_Type2>& _Rhs) {
	for (int _I = 0; _I < _Lhs.size(); _I++) {
		if (_Lhs[_I] > _Rhs[_I]) {
			return false;
		}
	}
	return true;
}