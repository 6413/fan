#pragma once
#ifdef _MSC_VER
#pragma warning (disable : 4244)
#endif

#include <iostream>
#include <cmath>
#include "Alloc.hpp"

class Color;

template <typename _Type>
class __Vec2 :public iterator<_Type> {
public:
	_Type x, y;

	__Vec2() : x(0), y(0) {}

	__Vec2(const _Type& _Value) : x(_Value), y(_Value) {}

	template <typename _InIt>
	__Vec2(const __Vec2<_InIt>& _Value) : x(_Value.x), y(_Value.y) {}

	__Vec2(_Type _x, _Type _y) : x(_x), y(_y) {}

	constexpr bool Empty() const {
		return !((int)this->x & (int)this->y);
	}

	constexpr _Type& operator[](const int x) {
		return !x ? this->x : this->y;
	}

	constexpr _Type operator[](const int x) const {
		return !x ? this->x : this->y;
	}

	constexpr __Vec2<_Type>& operator+=(const __Vec2<_Type>& v) {
		x += v.x;
		y += v.y;
		return *this;
	}

	template <typename _Typepe>
	constexpr __Vec2<_Typepe>& operator-=(const __Vec2<_Typepe>& v) {
		x -= v.x;
		y -= v.y;
		return *this;
	}

	constexpr __Vec2<_Type>& operator*=(const __Vec2<_Type>& v) {
		x *= v.x;
		y *= v.y;
		return *this;
	}

	constexpr bool operator!=(const __Vec2<_Type>& v) const {
		return v.x != x && v.y != y;
	}

	template <typename _Typepe>
	constexpr bool operator==(const __Vec2<_Typepe>& v) const {
		return v.x == x && v.y == y;
	}

	static constexpr size_t Size() {
		return 2;
	}

	constexpr auto Abs() {
		return __Vec2<_Type>(abs(x), abs(y));
	}

	constexpr void Print() const {
		std::cout << this->x << " " << this->y << std::endl;
	}
};

using Vec2 = __Vec2<float>;

template <typename _Type> 
class __Vec3 {
public:
	_Type x, y, z;
	__Vec3(const _Type& _x) {
		x = _x; y = _x; z = _x;
	}
	__Vec3(const Color& _Value) {
		this->x = _Value.r;
		this->y = _Value.g;
		this->z = _Value.b;
	}

	__Vec3(const __Vec2<_Type> _x) {
		x = _x.x; y = _x.y; z = 0;
	}
	__Vec3(const __Vec3<_Type>& v) {
		x = v.x; y = v.y; z = v.z;
	}
	__Vec3(const _Type& _x, const _Type& _y, const _Type& _z) {
		x = _x; y = _y; z = _z;
	}

	__Vec3() : x(0), y(0), z(0) {}
	constexpr auto operator-(const __Vec3<_Type>& v) {
		this->x = -v.x;
		this->y = -v.y;
		this->z = -v.z;
		return *this;
	}

	constexpr void operator+=(const __Vec3<_Type>& v) {
		this->x += v.x;
		this->y += v.y;
		this->z += v.z;
	}

	constexpr void operator-=(const __Vec3<_Type>& v) {
		this->x -= v.x;
		this->y -= v.y;
		this->z -= v.z;
	}
	
	constexpr void operator/=(_Type f) {
		this->x /= f;
		this->y /= f;
		this->z /= f;
	}

	constexpr void operator/=(const __Vec3<_Type>& v) {
		this->x /= v.x;
		this->y /= v.y;
		this->z /= v.z;
	}
	
	constexpr _Type at(size_t x) {
		return !x ? this->x : x == 1 ? this->y : this->z;
	}

	constexpr _Type operator[](size_t x) const {
		return !x ? this->x : x == 1 ? this->y : this->z;
	}

	constexpr _Type& operator[](size_t x) {
		return !x ? this->x : x == 1 ? this->y : this->z;
	}

	constexpr size_t Size() const {
		return 3;
	}
};

using Vec3 = __Vec3<float>;

template <typename _Type>
class __Vec4 {
public:

	_Type x, y, z, a;

	__Vec4(const _Type& _x) {
		x = _x; y = _x; z = _x; a = _x;
	}

	__Vec4(const __Vec4& v) {
		x = v.x; y = v.y; z = v.z; a = v.a;
	}

	__Vec4(const _Type& _x, const _Type& _y, const _Type& _z, const _Type& _a) {
		x = _x; y = _y; z = _z; a = _a;
	}

	__Vec4() : x(0), y(0), z(0), a(0) {}

	constexpr _Type operator[](const int x) const {
		return !x ? this->x : x == 1 ? this->y : x == 2 ? this->z : x == 3 ? this->a : this->a;
	}

	constexpr _Type& operator[](const int x) {
		return !x ? this->x : x == 1 ? this->y : x == 2 ? this->z : x == 3 ? this->a : this->a;
	}

	static constexpr size_t Size() {
		return 4;
	}
};

using Vec4 = __Vec4<float>;

template <typename _Ty>
struct __Mat2x2 {
	Vec2 vec[2];
	__Mat2x2(_Ty x) : vec{ Vec2(x, 0, 0, 0),
__Vec2<_Ty>(0, x, 0, 0) } {};
	__Mat2x2() : vec{ Vec2(), Vec2() } {};
	__Mat2x2(const __Mat2x2& m) {
		vec[0] = m.vec[0];
		vec[1] = m.vec[1];
		vec[2] = m.vec[2];
		vec[3] = m.vec[3];
	}
	template <typename _Type>
	__Mat2x2(const __Vec2<_Ty>& x, const __Vec2< _Type >& y) {
		vec[0] = x; vec[1] = y;
	}
	
	constexpr __Vec2<_Ty> operator[](size_t _Where) const {
		return vec[_Where];
	}

	constexpr size_t size() const {
		return 4;
	}
};

using Mat2x2 = __Mat2x2<float>;

template <typename _Ty>
struct __Mat2x3 {
	Vec2 vec[3];
	__Mat2x3(_Ty x) : vec{ 
	__Vec2<_Ty>(x, 0, 0, 0),
	__Vec2<_Ty>(0, x, 0, 0),
	__Vec2<_Ty>(0, 0, x, 0) } {};
	__Mat2x3() : vec{ Vec2(), Vec2(), Vec2() } {};

	__Mat2x3(const __Mat2x3& m) {
		vec[0] = m.vec[0];
		vec[1] = m.vec[1];
		vec[2] = m.vec[2];
	}
	__Mat2x3(const Vec2& m0, const Vec2& m1, const Vec2& m2) {
		vec[0] = m0;
		vec[1] = m1;
		vec[2] = m2;
	}
};

using Mat2x3 = __Mat2x3<float>;

template <typename _Ty>
struct __Mat2x4 {
	Vec2 vec[4];
	__Mat2x4(const __Mat2x4& m) {
		vec[0] = m.vec[0];
		vec[1] = m.vec[1];
		vec[2] = m.vec[2];
		vec[3] = m.vec[3];
	}
	__Mat2x4(const Vec2& x, const Vec2& y, const Vec2& z, const Vec2& a) {
		vec[0] = x; vec[1] = y; vec[2] = z; vec[3] = a;
	}
	__Mat2x4(_Ty x) : vec{ Vec2(x, 0, 0, 0),
	__Vec2<_Ty>(0, x, 0, 0),
	__Vec2<_Ty>(0, 0, x, 0),
	__Vec2<_Ty>(0, 0, 0, x) } {};
	__Mat2x4() : vec{ Vec2(), Vec2(), Vec2(), Vec2() } {};
};

using Mat2x4 = __Mat2x4<float>;

template <typename _Ty> 
struct __Mat4x4 {
	__Vec4<_Ty> vec[4];

	__Mat4x4(const __Vec4<_Ty>& x, const __Vec4<_Ty>& y, const __Vec4<_Ty>& z, const __Vec4<_Ty>& a) {
		vec[0] = x; vec[1] = y; vec[2] = z; vec[3] = a;
	}
	__Mat4x4(_Ty x) : vec{ Vec4(x, 0, 0, 0), 
	__Vec4<_Ty>(0, x, 0, 0), 
	__Vec4<_Ty>(0, 0, x, 0), 
	__Vec4<_Ty>(0, 0, 0, x) } {};
	__Mat4x4() : vec{ Vec4(), Vec4(), Vec4(), Vec4() } {};
};

using Mat4x4 = __Mat4x4<float>;

class Color {
public:
	float r, g, b, a;
	Color() : r(0), g(0), b(0), a(0) {}

	Color(float r, float g, float b, float a) {
		this->r = r;
		this->g = g;
		this->b = b;
		this->a = a;
	}

	constexpr float operator[](size_t x) const {
		return !x ? this->r : x == 1 ? this->g : x == 2 ? this->b : x == 3 ? this->a : this->a;
	}
};

template <typename color_t, typename _Type>
constexpr color_t operator/(const color_t& c, _Type value) {
	return color_t(c.r / value, c.g / value, c.b / value, c.a / value);
}

template <typename _Casted, template<typename> typename _Vec_t, typename _Old>
constexpr _Vec_t<_Casted> Cast(_Vec_t<_Old> v) {
	return _Vec_t<_Casted>(v);
}

template <template<typename> typename _Vec_t, typename _Type>
constexpr _Vec_t<_Type> Round(const _Vec_t<_Type>& v) {
	return _Vec_t<_Type>(round(v.x), round(v.y), round(v.z));
}

template <template<typename> typename _Vec_t, typename _Type>
constexpr _Vec_t<_Type> operator+(_Vec_t<_Type> _Lhs, _Vec_t<_Type> _Rhs) {
	_Vec_t<_Type> _Vec;
	for (int _I = 0; _I < _Vec.Size(); _I++) {
		_Vec[_I] = _Lhs[_I] + _Rhs[_I];
	}
	return _Vec;
}


template <template<typename> typename _Vec_t, typename _Type>
constexpr _Vec_t<_Type> operator+(_Vec_t<_Type> _Lhs, float _Rhs) {
	_Vec_t<_Type> _Vec;
	for (int _I = 0; _I < _Vec.Size(); _I++) {
		_Vec[_I] = _Lhs[_I] + _Rhs;
	}
	return _Vec;
}

template <template<typename> typename _Vec_t, typename _Type>
constexpr _Vec_t<_Type> operator*(_Vec_t<_Type> _Lhs, float _Rhs) {
	_Vec_t<_Type> _Vec;
	for (int _I = 0; _I < _Vec.Size(); _I++) {
		_Vec[_I] = _Lhs[_I] * _Rhs;
	}
	return _Vec;
}

template <template<typename> typename _Vec_t, typename _Type> 
constexpr _Vec_t<_Type> operator-(const _Vec_t<_Type>& _Lhs, const _Vec_t<_Type>& _Rhs) { 
	_Vec_t<_Type> _Vec;
	for (int _I = 0; _I < _Lhs.Size(); _I++) {
		_Vec[_I] = _Lhs[_I] - _Rhs[_I];
	}
	return _Vec;
}


template <template<typename> typename _Vec_t, typename _Type>
constexpr _Vec_t<_Type> operator*(const _Vec_t<_Type>& _Lhs, const _Vec_t<_Type>& _Rhs) {
	_Vec_t<_Type> _Vec;
	for (int _I = 0; _I < _Lhs.Size(); _I++) {
		_Vec[_I] = _Lhs[_I] * _Rhs[_I];
	}
	return _Vec;
}

template <template<typename> typename _Vec_t, typename _Type>
constexpr _Vec_t<_Type> operator-(const _Vec_t<_Type>& _Lhs, float _Rhs) {
	_Vec_t<_Type> _Vec;
	for (int _I = 0; _I < _Lhs.Size(); _I++) {
		_Vec[_I] = _Lhs[_I] - _Rhs;
	}
	return _Vec;
}

template <template<typename> typename _Vec_t, typename _Type>
constexpr _Vec_t<_Type> operator%(const _Vec_t<_Type>& _Lhs, float _Rhs) {
	_Vec_t<_Type> _Vec;
	for (int _I = 0; _I < _Lhs.Size(); _I++) {
		_Vec[_I] = fmodf(_Lhs[_I], _Rhs);
	}
	return _Vec;
}


//template <template<typename> typename _Vec_t, typename _Type, typename _Val>
//constexpr _Vec_t<_Type> operator/(const _Vec_t<_Type>& _Lhs, _Val _Value) {
//	_Vec_t<_Type> _Vec;
//	for (int _I = 0; _I < _Lhs.Size(); _I++) {
//		_Vec[_I] = _Lhs[_I] / _Value;
//	}
//	return _Vec;
//}


//template <template<typename> typename _Vec_t, typename _Type>
//constexpr _Vec_t<_Type> operator*(const _Vec_t<_Type>& _Lhs, const _Vec_t<_Type>& _Rhs) {
//	_Vec_t<_Type> _Vec;
//	for (int _I = 0; _I < _Lhs.Size(); _I++) {
//		_Vec[_I] = _Lhs[_I] * _Rhs[_I];
//	}
//	return _Vec;
//}
//
//template <typename _Ty>
//constexpr auto operator/(const __Vec2<_Ty>& lhs, const __Vec2<_Ty>& rhs) {
//	return __Vec2<_Ty>(lhs.x / rhs.x, lhs.y / rhs.y);
//}

//template <typename _Ty, typename _Ty2>
//constexpr auto operator+(const __Vec2<_Ty>& lhs, const _Ty2 rhs) {
//	return __Vec2<_Ty>(lhs.x + rhs, lhs.y + rhs);
//}

//template <typename _Ty, typename _Ty2>
//constexpr auto operator*(const __Vec2<_Ty>& lhs, const _Ty2 rhs) {
//	return __Vec2<_Ty>(lhs.x * rhs, lhs.y * rhs);
//}

template <typename _Ty, typename _Ty2>
constexpr auto operator/(const __Vec2<_Ty>& lhs, const _Ty2 rhs) {
	return __Vec2<_Ty>(lhs.x / rhs, lhs.y / rhs); 
}
//
//template <typename _Ty>
//constexpr auto operator*(const __Vec3<_Ty>& lhs, const float rhs) {
//	return __Vec3<_Ty>(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs);
//}

//template <typename _Ty>
//constexpr auto operator+(const __Vec4<_Ty>& lhs, const __Vec4<_Ty>& rhs) {
//	return __Vec4<_Ty>(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.a + rhs.a);
//}

//template <typename _Ty, typename _Ty2>
//constexpr auto operator*(const __Vec4<_Ty>& lhs, const _Ty2 rhs) {
//	return __Vec4<_Ty>(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs, lhs.a * rhs);
//}

template <typename _Ty>
constexpr auto operator+(const __Vec4<_Ty>& lhs, const _Ty rhs) {
	return __Vec4<_Ty>(lhs.x + rhs, lhs.y + rhs, lhs.z + rhs, lhs.a + rhs);
}