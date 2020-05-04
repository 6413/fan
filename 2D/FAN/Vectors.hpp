#pragma once
#ifdef _MSC_VER
#pragma warning (disable : 4244)
#endif

#include <iostream>
#include <cmath>

template <typename _Ty, std::size_t N>
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
	constexpr auto& operator[](std::size_t idx) {
		return _array[idx];
	}
	constexpr auto operator[](std::size_t idx) const {
		return _array[idx];
	}
};

template <typename _Ty>
class _vec2 {
public:
	_Ty x, y;

	constexpr _vec2() : x(0), y(0) { }
	constexpr _vec2(_Ty value) : x(value), y(value) { }
	constexpr _vec2(_Ty x, _Ty y) : x(x), y(y) { }
	template <typename type>
	constexpr _vec2(const _vec2<type>& vec) : x(vec.x), y(vec.y) { }

	constexpr _Ty& operator[](std::size_t idx) { return !idx ? x : y; }
	constexpr _Ty operator[](std::size_t idx) const { return !idx ? x : y; }
	constexpr _vec2<_Ty> operator-() const { return _vec2<_Ty>(-x, -y); }
	constexpr _vec2<_Ty> operator-(_Ty val) const { return _vec2<_Ty>(x - val, y - val); }
	constexpr _vec2<_Ty> operator-(const _vec2<_Ty>& val) const { return _vec2<_Ty>(x - val.x, y - val.y); }
	constexpr _vec2<_Ty> operator+(const _vec2<_Ty>& val) const { return _vec2<_Ty>(x + val.x, y + val.y); }
	constexpr _vec2<_Ty> operator/=(_Ty val) { x /= val; y /= val; return *this; }

	constexpr bool operator!=(const _vec2<_Ty> vec) const { return x != vec.x && y != vec.y; }
	constexpr bool operator==(const _vec2<_Ty> vec) const { return x == vec.x && y == vec.y; }

	constexpr _vec2<_Ty> floored() const { return _vec2<_Ty>(floor(x), floor(y)); }
	constexpr _vec2<_Ty> floored(_Ty value) const { return _vec2<_Ty>(floor(x / value), floor(y / value)); }

	constexpr std::size_t size() const { return 2; }

	constexpr void print() const { std::cout << x << " " << y << std::endl; }
};

template <typename _Ty = float>
class _vec3  {
public:
	_Ty x, y, z;

	constexpr _vec3() : x(0), y(0), z(0) { }
	template <typename type>
	constexpr _vec3(const _vec3<type>& vec) : x(vec.x), y(vec.y), z(vec.z) { }
	constexpr _vec3(_Ty x, _Ty y, _Ty z) : x(x), y(y), z(z) { }

	constexpr _Ty& operator[](std::size_t idx) { return !idx ? x : idx == 1 ? y : z; }
	constexpr _Ty operator[](std::size_t idx) const { return !idx ? x : idx == 1 ? y : z; }
	constexpr _vec3<_Ty> operator-(const _vec3<_Ty>& val) const { return _vec3<_Ty>(x - val.x, y - val.y, z - val.z); }

	constexpr std::size_t size() const { return 3; }
};

template <int Cols, int Rows, typename _Ty = float>
class matrix {
public:
	array<_Ty, Cols> m[Rows];
	using type = _Ty;

	constexpr matrix() : m{ }, _InIt_I(0) {
		for (int _I = 0; _I < Cols; _I++) {
			for (int _J = 0; _J < Rows; _J++) {
				m[_I][_J] = 0;
			}
		}
	}

	void initialize() {}

	template <typename T, typename ..._Type>
	constexpr void initialize(T _Val, _Type&&... _array) {
		((_Ty*)m)[_InIt_I] = _Val;
		_InIt_I++;
		initialize(_array...);
	}

	template <typename ..._Type>
	matrix(_Ty _Val, _Type&&... _array) : _InIt_I(0) {
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
			static_assert(sizeof...(_array) > Rows* Cols - 2, "too few initializer values");
			static_assert(sizeof...(_array) + 1 <= Rows * Cols, "too many initializer values");
			initialize(_Val, _array...);
		}
	}

	array<_Ty, Rows>& operator[](std::size_t _Idx) {
		return m[_Idx];
	}

	array<_Ty, Rows> operator[](std::size_t _Idx) const {
		return m[_Idx];
	}

	constexpr std::size_t rows() const {
		return Rows;
	}

	constexpr std::size_t cols() const {
		return Cols;
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
private:
	std::size_t _InIt_I;
};

template <typename _Ty>
class _mat2x2 {
public:
	constexpr _vec2<_Ty>& operator[](std::size_t idx) {
		return v[idx];
	}
	constexpr _vec2<_Ty> operator[](std::size_t idx) const {
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
	std::size_t init_index;
};

template <typename _Ty>
class _mat2x4 {
public:
	_mat2x4() : v {} {}

	template <typename T, typename ...type>
	constexpr _mat2x4(T value, type&&... args) : init_index(0) {
		initialize(value, args...);
	}

	constexpr _vec2<_Ty>& operator[](std::size_t idx) {
		return v[idx];
	}
	constexpr _vec2<_Ty> operator[](std::size_t idx) const {
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
	std::size_t init_index;
};

using mat2x2 = _mat2x2<float>;
using mat2x4 = _mat2x4<float>;
using vec2 = _vec2<float>;
using vec3 = _vec3<float>;

class Color {
public:
	float r, g, b, a;
	Color() : r(0), g(0), b(0), a(1) {}

	constexpr Color(float r, float g, float b, float a = 1, bool index = false) : r(0), g(0), b(0), a(0) {
		if (index) {
			this->r = r / 255.f;
			this->g = g / 255.f;
			this->b = b / 255.f;
			this->a = a / 255.f;
			return;
		}
		this->r = r;
		this->g = g;
		this->b = b;
		this->a = a;
	}
	constexpr Color(float value, bool index = false) : r(0), g(0), b(0), a(0) {
		if (index) {
			this->r = r / value;
			this->g = g / value;
			this->b = b / value;
			this->a = a / value;
			return;
		}
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
};

template <typename color_t, typename _Type>
constexpr color_t operator/(const color_t& c, _Type value) {
	return color_t(c.r / value, c.g / value, c.b / value, c.a / value);
}

template <typename _Vec2 = vec2, typename _Vec3 = vec3>
constexpr _Vec3 Vec2ToVec3(const _Vec2& _Vec) {
	return _Vec3(_Vec.x, _Vec.y, 0);
}

template <typename _Casted, template<typename> typename _Vec_t, typename _Old>
constexpr _Vec_t<_Casted> Cast(_Vec_t<_Old> v) {
	return _Vec_t<_Casted>(v);
}

template <template<typename> typename _Vec_t, typename _Type>
constexpr _Vec_t<_Type> Round(const _Vec_t<_Type>& v) {
	return _Vec_t<_Type>(round(v.x), round(v.y), round(v.z));
}

template <template<typename> typename _Vec_t, typename _Type, typename _Type2>
constexpr _Vec_t<_Type> operator+(_Vec_t<_Type> _Lhs, _Vec_t<_Type2> _Rhs) {
	_Vec_t<_Type> _Vec;
	for (int _I = 0; _I < _Vec.size(); _I++) {
		_Vec[_I] = _Lhs[_I] + _Rhs[_I];
	}
	return _Vec;
}

template <template<typename> typename _Vec_t, typename _Type, typename _Type2>
constexpr _Vec_t<_Type>& operator+=(_Vec_t<_Type>& _Lhs, const _Vec_t<_Type2>& _Rhs) {
	for (int _I = 0; _I < _Lhs.size(); _I++) {
		_Lhs[_I] = _Lhs[_I] + _Rhs[_I];
	}
	return _Lhs;
}

template <template<typename> typename _Vec_t, typename _Type, typename _Type2>
constexpr _Vec_t<_Type>& operator-=(_Vec_t<_Type>& _Lhs, const _Vec_t<_Type2>& _Rhs) {
	for (int _I = 0; _I < _Lhs.size(); _I++) {
		_Lhs[_I] = _Lhs[_I] - _Rhs[_I];
	}
	return _Lhs;
}

//template <template<typename> typename _Vec_t, typename _Type>
//constexpr _Vec_t<_Type> operator-(const _Vec_t<_Type>& _Lhs, const _Vec_t<_Type>& _Rhs) {
//	_Vec_t<_Type> _Vec;
//	for (int _I = 0; _I < _Lhs.size(); _I++) {
//		_Vec[_I] = _Lhs[_I] - _Rhs[_I];
//	}
//	return _Vec;
//}

template <template<typename> typename _Vec_t, typename _Type, typename _Type2>
constexpr _Vec_t<_Type> operator-(const _Vec_t<_Type>& _Vec) {
	for (int _I = 0; _I < _Vec.size(); _I++) {
		_Vec[_I] = -_Vec[_I];
	}
	return _Vec;
}

template <template<typename> typename _Vec_t, typename _Type, typename _Type2>
constexpr _Vec_t<_Type> operator*(_Vec_t<_Type> _Lhs, _Vec_t<_Type2> _Rhs) {
	_Vec_t<_Type> _Vec;
	for (int _I = 0; _I < _Vec.size(); _I++) {
		_Vec[_I] = _Lhs[_I] * _Rhs[_I];
	}
	return _Vec;
}

template <template<typename> typename _Vec_t, typename _Type, typename _Type2>
constexpr _Vec_t<_Type> operator*(_Vec_t<_Type> _Lhs, _Type2 _Rhs) {
	_Vec_t<_Type> _Vec;
	for (int _I = 0; _I < _Vec.size(); _I++) {
		_Vec[_I] = _Lhs[_I] * _Rhs;
	}
	return _Vec;
}

template <template<typename> typename _Vec_t, typename _Type, typename _Type2>
constexpr _Vec_t<_Type> operator/(_Vec_t<_Type> _Lhs, _Vec_t<_Type2> _Rhs) {
	_Vec_t<_Type> _Vec;
	for (int _I = 0; _I < _Vec.size(); _I++) {
		_Vec[_I] = _Lhs[_I] / _Rhs[_I];
	}
	return _Vec;
}

template <template<typename> typename _Vec_t, typename _Type, typename _Type2>
constexpr _Vec_t<_Type> operator/(_Vec_t<_Type> _Lhs, _Type2 _Rhs) {
	_Vec_t<_Type> _Vec;
	for (int _I = 0; _I < _Vec.size(); _I++) {
		_Vec[_I] = _Lhs[_I] / _Rhs;
	}
	return _Vec;
}

template <template<typename> typename _Vec_t, typename _Type, typename _Type2>
constexpr _Vec_t<_Type> operator%(const _Vec_t<_Type>& _Lhs, _Type2 _Rhs) {
	_Vec_t<_Type> _Vec;
	for (int _I = 0; _I < _Lhs.size(); _I++) {
		_Vec[_I] = fmodf(_Lhs[_I], _Rhs);
	}
	return _Vec;
}