#pragma once
#ifdef _MSC_VER
#pragma warning (disable : 4244)
#endif
#include "Vectors.hpp"

#include <cfloat>
#include <random>
#include <functional>


template <typename T>
void debugger(std::function<T> functionPtr) {
	printf("start\n");
	functionPtr();
	printf("end\n");
}

constexpr float PI = 3.1415926535f;
constexpr float HALF_PI = PI / 2;

constexpr bool ray_hit(const vec2& point) {
	return point != -1;
}

constexpr bool on_hit(const vec2& point, std::function<void()>&& lambda) {
	if (ray_hit(point)) {
		lambda();
		return true;
	}
	return false;
}

template <typename T>
constexpr auto IntersectionPoint(const T& p1Start, const T& p1End, const T& p2Start, const T& p2End) {
	float den = (p1Start.x - p1End.x) * (p2Start.y - p2End.y) - (p1Start.y - p1End.y) * (p2Start.x - p2End.x);
	if (!den) {
		return vec2(-1, -1);
	}
	float t = ((p1Start.x - p2Start.x) * (p2Start.y - p2End.y) - (p1Start.y - p2Start.y) * (p2Start.x - p2End.x)) / den;
	float u = -((p1Start.x - p1End.x) * (p1Start.y - p2Start.y) - (p1Start.y - p1End.y) * (p1Start.x - p2Start.x)) / den;
	if (t > 0 && t < 1 && u > 0 && u < 1) {
		return vec2(p1Start.x + t * (p1End.x - p1Start.x), p1Start.y + t * (p1End.y - p1Start.y));
	}
	return vec2(-1, -1);
}

template <typename first, typename second>
auto random(first min, second max) {
	static std::random_device device;
	static std::mt19937_64 random(device());
	std::uniform_int_distribution<first> distance(min, max);
	return distance(random);
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
	return vec3(
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
	return vec3(x.x / length, x.y / length, x.z / length);
}

template <typename _Ty, typename _Ty2>
constexpr auto AimAngle(const _vec2<_Ty>& src, const _vec2<_Ty2>& dst) {
	return atan2f(dst.y - src.y, dst.x - src.x);
}

//template <typename _Ty>
//constexpr auto DirectionVector(const _Ty& aimAngle)
//{
//	return _vec2<_Ty>(cos(aimAngle), sin(aimAngle));
//}

//template <typename _Ty>
//constexpr auto DirectionVector(const _Ty& aimAngle)
//{
//	return _vec2<_Ty>(cos(aimAngle.x), sin(aimAngle.y));
//}

//template <typename T>
//constexpr auto AbsAngle(const T& src, const T& dst, float delta_time) {
//	return V3ToV2(Normalize(vec3(src.x - dst.x, src.y - dst.y, 0))) * delta_time;
//}

template <typename _Ty>
constexpr auto Grid(const _vec2<_Ty>& world, const _vec2<_Ty>& offSet, float block_size) {
	return _vec2<_Ty>(floor((world.x + offSet.x) / block_size), floor(((world.y + offSet.y) / block_size)));
}

template <typename _Ty>
constexpr auto Distance(const _vec2<_Ty>& src, const _vec2<_Ty>& dst) {
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
constexpr auto ManhattanDistance(const _vec2<_Ty>& src, const _vec2<_Ty>& dst) {
	return Abs(src.x - dst.x) + Abs(src.y - dst.y);
}

template <typename T1, typename T2>
constexpr matrix<4, 4> Translate(T1& m, T2 v) {
	T1 Result(m);
	Result[3] =
		(m[0] * v.x) +
		(m[1] * v.y) +
		(m[2] * v.z) +
		 m[3];
	return Result;
}

template <typename T1, typename T2> constexpr 
auto Scale(const T1& m, const T2& v) {
	T1 Result;
	Result[0] = m[0] * v.x;
	Result[1] = m[1] * v.y;
	Result[2] = m[2] * v.z;
	
	Result[3] = m[3];
	return Result;
}

template <typename _Ty> constexpr auto PixelsToSomething(_Ty pixels, _Ty screen) {
	return _Ty((2.0f / screen.x * pixels.x - 1.0f), -(2.0f / screen.y * pixels.y - 1.0f));
}

template<typename T = matrix<4, 4>>
auto Ortho(float left, float right, float bottom, float top) {
	T Result(static_cast<T>(1));
	Result.vec[0][0] = static_cast<float>(2) / (right - left);
	Result.vec[1][1] = static_cast<float>(2) / (top - bottom);
	Result.vec[2][2] = -static_cast<float>(1);
	Result.vec[3][0] = -(right + left) / (right - left);
	Result.vec[3][1] = -(top + bottom) / (top - bottom);
	return Result;
}


template <typename T = matrix<4, 4>> constexpr
auto Ortho(float left, float right, float bottom, float top, float zNear, float zFar) {
	T Result(1);
	Result.m[0][0] = 2.f / (right - left);
	Result.m[1][1] = 2.f / (top - bottom);
	Result.m[2][2] = 1.f / (zFar - zNear);
	Result.m[3][0] = -(right + left) / (right - left);
	Result.m[3][1] = -(top + bottom) / (top - bottom);
	Result.m[3][2] = -zNear / (zFar - zNear);
	return Result;
}

template <typename T, typename T2 = matrix<4, 4>> 
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

template <typename _Ty = vec3, typename _Ty2 = matrix<4, 4>>
constexpr auto LookAt(const _Ty& eye, const _Ty& center, const _Ty& up) {

	//vec3 ne();
	//debugger<void()>([center, eye]() { Normalize((center - eye)).print(); });

	vec3 f(Normalize(center - eye));
	vec3 s(Normalize(Cross(f, up)));
	vec3 u(Cross(s, f));

	_Ty2 Result(1);
	Result.m[0][0] = s[0];
	Result.m[1][0] = s[1];
	Result.m[2][0] = s[2];
	Result.m[0][1] = u[0];
	Result.m[1][1] = u[1];
	Result.m[2][1] = u[2];
	Result.m[0][2] = -f[0];
	Result.m[1][2] = -f[1];
	Result.m[2][2] = -f[2];
	float x = -Dot(s, eye);
	float y = -Dot(u, eye);
	float z = Dot(f, eye);
	Result.m[3][0] = x;
	Result.m[3][1] = y;
	Result.m[3][2] = z;
	return Result;
}


static auto Rotate(const matrix<4, 4>& m, float angle, const vec3& v) {
	const float a = angle;
	const float c = cos(a);
	const float s = sin(a);
	vec3 axis(Normalize(v));
	vec3 temp(axis * (1.0f - c));

	matrix<4, 4> Rotate;
	Rotate[0][0] = c + temp[0] * axis[0];
	Rotate[0][1] = temp[0] * axis[1] + s * axis[2];
	Rotate[0][2] = temp[0] * axis[2] - s * axis[1];

	Rotate[1][0] = temp[1] * axis[0] - s * axis[2];
	Rotate[1][1] = c + temp[1] * axis[1];
	Rotate[1][2] = temp[1] * axis[2] + s * axis[0];

	Rotate[2][0] = temp[2] * axis[0] + s * axis[1];
	Rotate[2][1] = temp[2] * axis[1] - s * axis[0];
	Rotate[2][2] = c + temp[2] * axis[2];

	matrix<4, 4> Result;
	Result[0] = (m[0] * Rotate[0][0]) + (m[1] * Rotate[0][1]) + (m[2] * Rotate[0][2]);
	Result[1] = (m[0] * Rotate[1][0]) + (m[1] * Rotate[1][1]) + (m[2] * Rotate[1][2]);
	Result[2] = (m[0] * Rotate[2][0]) + (m[1] * Rotate[2][1]) + (m[2] * Rotate[2][2]);
	Result[3] = m[3];
	return Result;
}
