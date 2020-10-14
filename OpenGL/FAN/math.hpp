#pragma once
#ifdef _MSC_VER
#pragma warning (disable : 4244)
#endif

#include <FAN/types.h>

#include <cfloat>
#include <random>
#include <functional>
#include <cmath>

namespace fan {
	template <typename T>
	void debugger(std::function<T> functionPtr) {
		printf("start\n");
		functionPtr();
		printf("end\n");
	}

	constexpr float PI = 3.1415926535f;
	constexpr float HALF_PI = PI / 2;

	constexpr int RAY_DID_NOT_HIT = -1;

	constexpr bool ray_hit(const vec2& point) {
		return point != RAY_DID_NOT_HIT;
	}

	constexpr bool ray_hit(const vec3& point) {
		return point != RAY_DID_NOT_HIT;
	}

	constexpr bool on_hit(const vec2& point, std::function<void()>&& lambda) {
		if (ray_hit(point)) {
			lambda();
			return true;
		}
		return false;
	}

	template <typename T>
	constexpr vec2 intersection_point(const T& p1Start, const T& p1End, const T& p2Start, const T& p2End, bool infinite_long_ray) {
		f_t den = (p1Start.x - p1End.x) * (p2Start.y - p2End.y) - (p1Start.y - p1End.y) * (p2Start.x - p2End.x);
		if (!den) {
			return RAY_DID_NOT_HIT;
		}
		f_t t = ((p1Start.x - p2Start.x) * (p2Start.y - p2End.y) - (p1Start.y - p2Start.y) * (p2Start.x - p2End.x)) / den;
		f_t u = -((p1Start.x - p1End.x) * (p1Start.y - p2Start.y) - (p1Start.y - p1End.y) * (p1Start.x - p2Start.x)) / den;
		if (!infinite_long_ray) {
			if (t > 0 && t < 1 && u > 0 && u < 1) {
				return vec2(p1Start.x + t * (p1End.x - p1Start.x), p1Start.y + t * (p1End.y - p1Start.y));
			}
		}
		else {
			if (t > 0 && u > 0 && u < 1) {
				return vec2(p1Start.x + t * (p1End.x - p1Start.x), p1Start.y + t * (p1End.y - p1Start.y));
			}
		}

		return RAY_DID_NOT_HIT;
	}

	template <typename first, typename second>
	auto random(first min, second max) {
		static std::random_device device;
		static std::mt19937_64 random(device());
		std::uniform_int_distribution<first> distance(min, max);
		return distance(random);
	}

	// converts degrees to radians
	template<typename T>
	constexpr auto radians(T x) { return (x * PI / 180.0f); }

	// converts radians to degrees
	template<typename T>
	constexpr auto degrees(T x) { return (x * 180.0f / PI); }

	template <typename T>
	constexpr T sign(T val) {
		return (T(0) < val) - (val < T(0));
	}

	template <typename T>
	constexpr auto cross(const T& a, const T& b) {
		return T(
			a[1] * b[2] - b[1] * a[2],
			a[2] * b[0] - b[2] * a[0],
			a[0] * b[1] - b[0] * a[1]
		);
	}

	constexpr float dot(const vec2& x, const vec2& y) {
		return (x.x * y.x) + (x.y * y.y);
	}

	constexpr auto dot(const vec3& x, const vec3& y) {
		return (x.x * y.x) + (x.y * y.y) + (x.z * y.z);
	}

	inline vec2 normalize(const vec2& x) {
		float length = sqrt(dot(x, x));
		return vec2(x.x / length, x.y / length);
	}

	inline vec3 normalize(const vec3& x) {
		auto length = sqrt(dot(x, x));
		return vec3(x.x / length, x.y / length, x.z / length);
	}

	template <typename _Ty, typename _Ty2>
	constexpr auto aim_angle(const _vec2<_Ty>& src, const _vec2<_Ty2>& dst) {
		return atan2f(dst.y - src.y, dst.x - src.x);
	}

	inline vec2 direction_vector(float angle)
	{
		return vec2(
			cos(angle),
			sin(angle)
		);
	}

	// depends about world rotation
	inline vec3 direction_vector(float alpha, float beta)
	{
		return vec3(
			cos(radians(alpha)) * cos(radians(beta)),
			sin(radians(alpha)) * cos(radians(beta)),
			sin(radians(beta))
		);
	}

	inline f_t distance(f_t src, f_t dst) {
		return std::abs(std::abs(dst) - std::abs(src));
	}

	template <typename _Ty>
	constexpr auto distance(const _Ty& src, const _Ty& dst) {
		return sqrtf(powf((src[0] - dst[0]), 2) + powf(((src[1] - dst[1])), 2));
	}

	template <typename _Ty, typename _Ty2>
	constexpr auto distance(const _Ty& src, const _Ty2& dst) {
		return sqrtf(powf((src[0] - dst[0]), 2) + powf(((src[1] - dst[1])), 2) + powf(((src[2] - dst[2])), 2));
	}

	static auto ManhattanDistance(const vec3& src, const vec3& dst) {
		return std::abs(src.x - dst.x) + std::abs(src.y - dst.y) + std::abs(src.z - dst.z);
	}

	template <typename _Matrix>
	constexpr _Matrix translate(const _Matrix& m, const vec3& v) {
		_Matrix Result(m);
		Result[3][0] = m[0][0] * v.x + m[1][0] * v.y + m[2][0] * v.z + m[3][0];
		Result[3][1] = m[0][1] * v.x + m[1][1] * v.y + m[2][1] * v.z + m[3][1];
		Result[3][2] = m[0][2] * v.x + m[1][2] * v.y + m[2][2] * v.z + m[3][2];
		Result[3][3] = m[0][3] * v.x + m[1][3] * v.y + m[2][3] * v.z + m[3][3];
		return Result;
	}

	inline auto scale(const mat4& m, const vec3& v) {
		mat4 Result;
		Result[0] = m[0] * v.x;
		Result[1] = m[1] * v.y;
		Result[2] = m[2] * v.z;

		Result[3] = m[3];
		return Result;
	}

	template <typename _Ty> constexpr auto PixelsToSomething(_Ty pixels, _Ty screen) {
		return _Ty((2.0f / screen.x * pixels.x - 1.0f), -(2.0f / screen.y * pixels.y - 1.0f));
	}

	template<typename T = mat4>
	auto ortho(float left, float right, float bottom, float top) {
		T Result(static_cast<T>(1));
		Result[0][0] = static_cast<float>(2) / (right - left);
		Result[1][1] = static_cast<float>(2) / (top - bottom);
		Result[2][2] = -static_cast<float>(1);
		Result[3][0] = -(right + left) / (right - left);
		Result[3][1] = -(top + bottom) / (top - bottom);
		return Result;
	}


	template <typename T = mat4>
	constexpr auto ortho(f_t left, f_t right, f_t bottom, f_t top, f_t zNear, f_t zFar) {
		T Result(1);
		Result.m[0][0] = 2.f / (right - left);
		Result.m[1][1] = 2.f / (top - bottom);
		Result.m[2][2] = 1.f / (zFar - zNear);
		Result.m[3][0] = -(right + left) / (right - left);
		Result.m[3][1] = -(top + bottom) / (top - bottom);
		Result.m[3][2] = -zNear / (zFar - zNear);
		return Result;
	}

	template <typename T = mat4>
	constexpr T perspective(f_t fovy, f_t aspect, f_t zNear, f_t zFar) {
		f_t const tanHalfFovy = tan(fovy / static_cast<f_t>(2));
		T Result;
		Result[0][0] = static_cast<f_t>(1) / (aspect * tanHalfFovy);
		Result[1][1] = static_cast<f_t>(1) / (tanHalfFovy);
		Result[2][2] = -(zFar + zNear) / (zFar - zNear);
		Result[2][3] = -static_cast<f_t>(1);
		Result[3][2] = -(static_cast<f_t>(2) * zFar * zNear) / (zFar - zNear);
		return Result;
	}

	template <typename _Ty = vec3, typename _Ty2 = mat4>
	constexpr auto look_at(const _Ty& eye, const _Ty& center, const _Ty& up) {
		vec3 f(normalize(center - eye));
		vec3 s(normalize(cross(f, up)));
		vec3 u(cross(s, f));

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
		float x = -dot(s, eye);
		float y = -dot(u, eye);
		float z = dot(f, eye);
		Result.m[3][0] = x;
		Result.m[3][1] = y;
		Result.m[3][2] = z;
		return Result;
	}


	static auto rotate(const mat4& m, float angle, const vec3& v) {
		const float a = angle;
		const float c = cos(a);
		const float s = sin(a);
		vec3 axis(normalize(v));
		vec3 temp(axis * (1.0f - c));

		mat4 Rotate;
		Rotate[0][0] = c + temp[0] * axis[0];
		Rotate[0][1] = temp[0] * axis[1] + s * axis[2];
		Rotate[0][2] = temp[0] * axis[2] - s * axis[1];

		Rotate[1][0] = temp[1] * axis[0] - s * axis[2];
		Rotate[1][1] = c + temp[1] * axis[1];
		Rotate[1][2] = temp[1] * axis[2] + s * axis[0];

		Rotate[2][0] = temp[2] * axis[0] + s * axis[1];
		Rotate[2][1] = temp[2] * axis[1] - s * axis[0];
		Rotate[2][2] = c + temp[2] * axis[2];

		mat4 Result;
		Result[0] = (m[0] * Rotate[0][0]) + (m[1] * Rotate[0][1]) + (m[2] * Rotate[0][2]);
		Result[1] = (m[0] * Rotate[1][0]) + (m[1] * Rotate[1][1]) + (m[2] * Rotate[1][2]);
		Result[2] = (m[0] * Rotate[2][0]) + (m[1] * Rotate[2][1]) + (m[2] * Rotate[2][2]);
		Result[3] = m[3];
		return Result;
	}
}