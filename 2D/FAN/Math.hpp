#pragma once
#ifdef _MSC_VER
#pragma warning (disable : 4244)
#endif
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cfloat>
#include "Vectors.hpp"
#include <assert.h>
constexpr float PI = 3.1415926535f;

template <typename T>
constexpr auto IntersectionPoint(const T& p1Start, const T& p1End, const T& p2Start, const T& p2End) {
	float den = (p1Start.x - p1End.x) * (p2Start.y - p2End.y) - (p1Start.y - p1End.y) * (p2Start.x - p2End.x);
	if (!den) {
		return Vec2(-1, -1);
	}
	float t = ((p1Start.x - p2Start.x) * (p2Start.y - p2End.y) - (p1Start.y - p2Start.y) * (p2Start.x - p2End.x)) / den;
	float u = -((p1Start.x - p1End.x) * (p1Start.y - p2Start.y) - (p1Start.y - p1End.y) * (p1Start.x - p2Start.x)) / den;
	if (t > 0 && t < 1 && u > 0 && u < 1) {
		return Vec2(p1Start.x + t * (p1End.x - p1Start.x), p1Start.y + t * (p1End.y - p1Start.y));
	}
	return Vec2(-1, -1);
}


template<std::size_t N, class T>
constexpr std::size_t ArrLen(T(&)[N]) { return N; }

// converts degrees to radians
template<typename T>
constexpr auto Radians(T x) { return (x * PI / 180.0f); } 

 // converts radians to degrees
template<typename T>
constexpr auto Degrees(T x) { return (x * 180.0f / PI); }

template <typename T> int sgn(T val) {
	return (T(0) < val) - (val < T(0));
}

template <typename T>
constexpr auto Cross(const T& x, const T& y) {
	return Vec3(
		x.y * y.z - y.y * x.z,
		x.z * y.x - y.z * x.x,
		x.x * y.y - y.x * x.y
	);
}

template <typename T>
constexpr auto Dot(const T& x, const T& y) {
	return (x.x * y.x) + (x.y * y.y) + (x.z * y.z);
}

template <typename T>
constexpr auto Normalize(const T& x) {
	float length = sqrtf(x.x * x.x + x.y * x.y + x.z * x.z);
	return Vec3(x.x / length, x.y / length, x.z / length);
}

template <typename _Ty, typename _Ty2>
constexpr auto AimAngle(const __Vec2<_Ty>& src, const __Vec2<_Ty2>& dst) {
	return Degrees(atan2f(dst.y - src.y, dst.x - src.x));
}

//template <typename _Ty>
//constexpr auto DirectionVector(const _Ty& aimAngle)
//{
//	return __Vec2<_Ty>(cos(aimAngle), sin(aimAngle));
//}

//template <typename _Ty>
//constexpr auto DirectionVector(const _Ty& aimAngle)
//{
//	return __Vec2<_Ty>(cos(aimAngle.x), sin(aimAngle.y));
//}

template <typename T>
constexpr auto AbsAngle(const T& src, const T& dst, float deltaTime) {
	return V3ToV2(Normalize(Vec3(src.x - dst.x, src.y - dst.y, 0))) * deltaTime;
}

template <typename _Ty>
constexpr auto Grid(const __Vec2<_Ty>& world, const __Vec2<_Ty>& offSet, float blockSize) {
	return __Vec2<_Ty>(floor((world.x + offSet.x) / blockSize), floor(((world.y + offSet.y) / blockSize)));
}

template <typename _Ty>
constexpr auto Distance(const __Vec2<_Ty>& src, const __Vec2<_Ty>& dst) {
	return sqrtf(powf((src.x - dst.x), 2) + powf(((src.y - dst.y)), 2));
}

template <typename _Type>
constexpr auto Abs(const _Type _Value) {
	if (_Value < 0) {
		return -_Value;
	}
	return _Value;
}

template <typename _Ty>
constexpr auto ManhattanDistance(const __Vec2<_Ty>& src, const __Vec2<_Ty>& dst) {
	return Abs(src.x - dst.x) + Abs(src.y - dst.y);
}

template <typename T1, typename T2>
constexpr auto Translate(T1& m, T2 v) {
	T1 Result(m);
	Result.vec[3] =
		(m.vec[0] * v.x) +
		(m.vec[1] * v.y) +
		(m.vec[2] * v.z) +
		m.vec[3];
	return Result;
}

template <typename T1, typename T2> constexpr 
auto Scale(const T1& m, const T2& v) {
	T1 Result;
	Result.vec[0] = m.vec[0] * v.x;
	Result.vec[1] = m.vec[1] * v.y;
	Result.vec[2] = m.vec[2] * v.z;
	Result.vec[3] = m.vec[3];
	return Result;
}

template <typename _Ty> constexpr auto PixelsToSomething(_Ty pixels, _Ty screen) {
	return _Ty((2.0f / screen.x * pixels.x - 1.0f), -(2.0f / screen.y * pixels.y - 1.0f));
}

template <typename T = bool, typename T2 = Mat4x4> constexpr
auto Ortho(float left, float right, float bottom, float top, float zNear, float zFar) {
	T2 Result(1);
	Result.vec[0].x = static_cast<float>(2) / (right - left);
	Result.vec[1].y = static_cast<float>(2) / (top - bottom);
	Result.vec[2].z = static_cast<float>(1) / (zFar - zNear);
	Result.vec[3].x = -(right + left) / (right - left);
	Result.vec[3].y = -(top + bottom) / (top - bottom);
	Result.vec[3].z = -zNear / (zFar - zNear);
	return Result;
}

template <typename T, typename T2 = Mat4x4> 
constexpr T2 perspectiveRH_NO(T fovy, T aspect, T zNear, T zFar) {
	abs(aspect - std::numeric_limits<T>::epsilon()) > static_cast<T>(0);
	T const tanHalfFovy = tan(fovy / static_cast<T>(2));
	T2 Result(static_cast<T>(1));
	Result.vec[0].x = static_cast<T>(1) / (aspect * tanHalfFovy);
	Result.vec[1].y = static_cast<T>(1) / (tanHalfFovy);
	Result.vec[2].z = -(zFar + zNear) / (zFar - zNear);
	Result.vec[2].a = -static_cast<T>(1);
	Result.vec[3].z = -(static_cast<T>(2)* zFar* zNear) / (zFar - zNear);
	return Result;
}

template <typename _Ty, typename _Ty2 = Mat4x4>
constexpr auto LookAt(const _Ty& eye, const _Ty& center, const _Ty& up) {
	_Ty const f(Normalize(center - eye));
	_Ty const s(Normalize(Cross(f, up)));
	_Ty const u(Cross(s, f));

	_Ty2 Result(1);
	Result.vec[0].x = s.x;
	Result.vec[1].x = s.y;
	Result.vec[2].x = s.z;
	Result.vec[0].y = u.x;
	Result.vec[1].y = u.y;
	Result.vec[2].y = u.z;
	Result.vec[0].z = -f.x;
	Result.vec[1].z = -f.y;
	Result.vec[2].z = -f.z;
	float x = -Dot(s, eye);
	float y = -Dot(u, eye);
	float z = Dot(f, eye);
	Result.vec[3].x = x;
	Result.vec[3].y = y;
	Result.vec[3].z = z;
	return Result;
}

template <typename _Ty, typename _Ty2>
constexpr auto Rotate(_Ty m, float angle, _Ty2 v) {
	const float a = angle;
	const float c = cos(a);
	const float s = sin(a);
	_Ty2 axis(Normalize(v));
	_Ty2 temp(axis * (1.0f - c));

	_Ty Rotate;
	Rotate.vec[0].x = c + temp.x * axis.x;
	Rotate.vec[0].y = temp.x * axis.y + s * axis.z;
	Rotate.vec[0].z = temp.x * axis.z - s * axis.y;

	Rotate.vec[1].x = temp.y * axis.x - s * axis.z;
	Rotate.vec[1].y = c + temp.y * axis.y;
	Rotate.vec[1].z = temp.y * axis.z + s * axis.x;

	Rotate.vec[2].x = temp.z * axis.x + s * axis.y;
	Rotate.vec[2].y = temp.z * axis.y - s * axis.x;
	Rotate.vec[2].z = c + temp.z * axis.z;

	_Ty Result;
	Result.vec[0] = (m.vec[0] * Rotate.vec[0].x) + (m.vec[1] * Rotate.vec[0].y) + (m.vec[2] * Rotate.vec[0].z);
	Result.vec[1] = (m.vec[0] * Rotate.vec[1].x) + (m.vec[1] * Rotate.vec[1].y) + (m.vec[2] * Rotate.vec[1].z);
	Result.vec[2] = (m.vec[0] * Rotate.vec[2].x) + (m.vec[1] * Rotate.vec[2].y) + (m.vec[2] * Rotate.vec[2].z);
	Result.vec[3] = m.vec[3];
	return Result;
}

Mat4x4 operator*(const Mat4x4& lhs, const Mat4x4& rhs);