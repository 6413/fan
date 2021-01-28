#pragma once
#ifdef _MSC_VER
#pragma warning (disable : 4244)
#endif

#include <cfloat>
#include <random>
#include <functional>
#include <cmath>

namespace fan_2d {

	template <typename vector_t>
	constexpr auto dot(const vector_t& x, const vector_t& y) {
		return x[0] * y[0] + x[1] * y[1];
	}

	template <typename _Ty, typename _Ty2>
	constexpr f_t distance(const _Ty& src, const _Ty2& dst) {
		const auto x = src[0] - dst[0];
		const auto y = src[1] - dst[1];

		return std::sqrt((x * x) + (y * y));
	}

	template <typename _Ty>
	constexpr f_t vector_length(const _Ty& vector) {
		return distance<_Ty>(_Ty(), vector);
	}

	template <typename T>
	inline T normalize(const T& vector) {
		f_t length = sqrt(fan_2d::dot(vector, vector));
		if (!length) {
			return T();
		}
		return T(vector.x / length, vector.y / length);
	}

}

namespace fan_3d {

	template <typename vector_t>
	constexpr auto dot(const vector_t& x, const vector_t& y) {
		return (x[0] * y[0]) + (x[1] * y[1]) + (x[2] * y[2]);
	}

	template <typename _Ty, typename _Ty2>
	constexpr f_t distance(const _Ty& src, const _Ty2& dst) {
		const auto x = src[0] - dst[0];
		const auto y = src[1] - dst[1];
		const auto z = src[2] - dst[2];

		return std::sqrt((x * x) + (y * y) + (z * z));
	}

	template <typename _Ty>
	constexpr f_t vector_length(const _Ty& vector) {
		return distance<_Ty>(_Ty(), vector);
	}

	template <typename vector_t>
	inline vector_t normalize(const vector_t& vector) {
		auto length = sqrt(dot(vector, vector));
		return vector_t(vector[0] / length, vector[1] / length, vector[2] / length);
	}

}

namespace fan {

	template <typename T>
	constexpr int64_t ceil(T num)
	{
		return (static_cast<float>(static_cast<int64_t>(num)) == num)
			? static_cast<int64_t>(num)
			: static_cast<int64_t>(num) + ((num > 0) ? 1 : 0);
	}

	constexpr f_t inf = INFINITY;
	constexpr f_t infinite = inf;
	constexpr f_t infinity = infinite;

	template <typename T>
	void debugger(std::function<T> functionPtr) {
		printf("start\n");
		functionPtr();
		printf("end\n");
	}

	constexpr f_t pi = 3.14159265358979323846264338327950288419716939937510;
	constexpr f_t half_pi = pi / 2;
	constexpr f_t two_pi = pi * 2;

	constexpr auto RAY_DID_NOT_HIT = fan::inf;

	template <typename T>
	constexpr bool ray_hit(const T& point) {
		return point != RAY_DID_NOT_HIT;
	}

	template <typename T>
	constexpr bool on_hit(const T& point, std::function<void()>&& lambda) {
		if (ray_hit(point)) {
			lambda();
			return true;
		}
		return false;
	}

	template <typename T>
	constexpr T intersection_point(const T& p1Start, const T& p1End, const T& p2Start, const T& p2End, bool infinite_long_ray) {
		f_t den = (p1Start[0] - p1End[0]) * (p2Start[1] - p2End[1]) - (p1Start[1] - p1End[1]) * (p2Start[0] - p2End[0]);
		if (!den) {
			return RAY_DID_NOT_HIT;
		}
		f_t t = ((p1Start[0] - p2Start[0]) * (p2Start[1] - p2End[1]) - (p1Start[1] - p2Start[1]) * (p2Start[0] - p2End[0])) / den;
		f_t u = -((p1Start[0] - p1End[0]) * (p1Start[1] - p2Start[1]) - (p1Start[1] - p1End[1]) * (p1Start[0] - p2Start[0])) / den;
		if (!infinite_long_ray) {
			if (t >= 0 && t <= 1 && u >= 0 && u <= 1) {
				return T(p1Start[0] + t * (p1End[0] - p1Start[0]), p1Start[1] + t * (p1End[1] - p1Start[1]));
			}
		}
		else {
			if (t >= 0 && u >= 0 && u <= 1) {
				return T(p1Start[0] + t * (p1End[0] - p1Start[0]), p1Start[1] + t * (p1End[1] - p1Start[1]));
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
	constexpr auto radians(T x) { return (x * pi / 180.0f); }

	// converts radians to degrees
	template<typename T>
	constexpr auto degrees(T x) { return (x * 180.0f / pi); }

	template <typename T>
	constexpr bool sign(T a, T b) {
		return a * b >= 0.0;
	}

	template <typename vector_t>
	constexpr auto cross(const vector_t& a, const vector_t& b) {
		return vector_t(
			a[1] * b[2] - b[1] * a[2],
			a[2] * b[0] - b[2] * a[0],
			a[0] * b[1] - b[0] * a[1]
		);
	}

	template <typename vector_t>
	constexpr vector_t normalize_no_sqrt(const vector_t& vector) {
		return vector / fan_3d::dot(vector, vector);
	}

	#define PI_FLOAT     3.14159265f
	#define PIBY2_FLOAT  1.5707963f
	// |error| < 0.005
	constexpr float atan2_approximation2( float y, float x )
	{
		if ( x == 0.0f )
		{
			if ( y > 0.0f ) return PIBY2_FLOAT;
			if ( y == 0.0f ) return 0.0f;
			return -PIBY2_FLOAT;
		}
		float atan;
		float z = y/x;
		if ( fabs( z ) < 1.0f )
		{
			atan = z/(1.0f + 0.28f*z*z);
			if ( x < 0.0f )
			{
				if ( y < 0.0f ) return atan - PI_FLOAT;
				return atan + PI_FLOAT;
			}
		}
		else
		{
			atan = PIBY2_FLOAT - z/(z*z + 0.28f);
			if ( y < 0.0f ) return atan - PI_FLOAT;
		}
		return atan;
	}

	template <typename T, typename T2>
	constexpr auto DiamondAngle(T y, T2 x)
	{
		if (y >= 0)
			return (x >= 0 ? y/(x+y) : 1-x/(-x+y)); 
		else
			return (x < 0 ? 2-y/(-x-y) : 3+x/(x-y)); 
	}

	template <typename _Ty, typename _Ty2>
	constexpr auto aim_angle(const _Ty& src, const _Ty2& dst) {
		return std::atan2(dst[1] - src[1], dst[0] - src[0]);
	}

	template <typename vector_t>
	inline vector_t direction_vector(f32_t angle)
	{
		return vector_t(
			sin(angle),
			-cos(angle)
		);
	}

	// depends about world rotation

	template <typename vector_t>
	inline vector_t direction_vector(f32_t alpha, f32_t beta)
	{
		return vector_t(
			(sin(fan::radians(alpha)) * cos(fan::radians(beta))),
			 sin(radians(beta)),
			(cos(fan::radians(alpha)) * cos(fan::radians(beta)))
		);
	}

	// z up
	//inline vec3 direction_vector(f32_t alpha, f32_t beta)
	//{
	//	return vec3(
	//		-(cos(fan::radians(alpha + 90)) * cos(fan::radians(beta))),
	//		-(sin(fan::radians(alpha + 90)) * cos(fan::radians(beta))),
	//		  sin(radians(beta))
	//	);
	//}


	inline f_t distance(f_t src, f_t dst) {
		return std::abs(std::abs(dst) - std::abs(src));
	}

	template <typename _Ty, typename _Ty2>
	constexpr f_t custom_pythagorean_no_sqrt(const _Ty& src, const _Ty2& dst) {
		return std::abs(src[0] - dst[0]) + std::abs(src[1] - dst[1]);
	}

	template <typename vector_t>
	inline auto distance(const vector_t& src, const vector_t& dst) {
		return std::abs(src[0] - dst[0]) + std::abs(src[1] - dst[1]) + std::abs(src[2] - dst[2]);
	}

	template <typename matrix_t, typename vector_t>
	constexpr matrix_t translate(const matrix_t& m, const vector_t& v) {
		matrix_t matrix(m);
		matrix[3][0] = m[0][0] * v[0] + m[1][0] * v[1] + (v.size() < 3 ? + 0 : (m[2][0] * v[2])) + m[3][0];
		matrix[3][1] = m[0][1] * v[0] + m[1][1] * v[1] + (v.size() < 3 ? + 0 : (m[2][1] * v[2])) + m[3][1];
		matrix[3][2] = m[0][2] * v[0] + m[1][2] * v[1] + (v.size() < 3 ? + 0 : (m[2][2] * v[2])) + m[3][2];
		matrix[3][3] = m[0][3] * v[0] + m[1][3] * v[1] + (v.size() < 3 ? + 0 : (m[2][3] * v[2])) + m[3][3];
		return matrix;
	}

	template <typename matrix_t, typename vector_t>
	inline auto scale(const matrix_t& m, const vector_t& v) {
		matrix_t matrix;

		matrix[0][0] = m[0][0] * v[0];
		matrix[0][1] = m[0][1] * v[0];
		matrix[0][2] = m[0][2] * v[0];

		matrix[1][0] = m[1][0] * v[1];
		matrix[1][1] = m[1][1] * v[1];
		matrix[1][2] = m[1][2] * v[1];

		matrix[2][0] = (v.size() < 3 ? 0 : m[2][0] * v[2]);
		matrix[2][1] = (v.size() < 3 ? 0 : m[2][1] * v[2]);
		matrix[2][2] = (v.size() < 3 ? 0 : m[2][2] * v[2]);

		matrix[3][0] = m[3][0];
		matrix[3][1] = m[3][1];
		matrix[3][2] = m[3][2];

		matrix[3] = m[3];
		return matrix;
	}

	template <typename matrix_t>
	auto ortho(float left, float right, float bottom, float top) {
		matrix_t matrix(1);
		matrix[0][0] = static_cast<float>(2) / (right - left);
		matrix[1][1] = static_cast<float>(2) / (top - bottom);
		matrix[2][2] = -static_cast<float>(1);
		matrix[3][0] = -(right + left) / (right - left);
		matrix[3][1] = -(top + bottom) / (top - bottom);
		return matrix;
	}


	template <typename matrix_t>
	constexpr auto ortho(f_t left, f_t right, f_t bottom, f_t top, f_t zNear, f_t zFar) {
		matrix_t matrix(1);
		matrix[0][0] = 2.f / (right - left);
		matrix[1][1] = 2.f / (top - bottom);
		matrix[2][2] = 1.f / (zFar - zNear);
		matrix[3][0] = -(right + left) / (right - left);
		matrix[3][1] = -(top + bottom) / (top - bottom);
		matrix[3][2] = -zNear / (zFar - zNear);
		return matrix;
	}

	template <typename matrix_t>
	constexpr matrix_t perspective(f_t fovy, f_t aspect, f_t zNear, f_t zFar) {
		f_t const tanHalfFovy = tan(fovy / static_cast<f_t>(2));
		matrix_t matrix;
		matrix[0][0] = static_cast<f_t>(1) / (aspect * tanHalfFovy);
		matrix[1][1] = static_cast<f_t>(1) / (tanHalfFovy);
		matrix[2][2] = -(zFar + zNear) / (zFar - zNear);
		matrix[2][3] = -static_cast<f_t>(1);
		matrix[3][2] = -(static_cast<f_t>(2) * zFar * zNear) / (zFar - zNear);
		return matrix;
	}

	template <typename matrix_t, typename vector_t>
	constexpr auto look_at_left(const vector_t& eye, const vector_t& center, const vector_t& up)
	{
		const vector_t f(fan_3d::normalize(eye - center));
		const vector_t s(fan_3d::normalize(fan::cross(f, up)));
		const vector_t u(fan::cross(s, f));

		matrix_t matrix(1);
		matrix[0][0] = s[0];
		matrix[1][0] = s[1];
		matrix[2][0] = s[2];
		matrix[0][1] = u[0];
		matrix[1][1] = u[1];
		matrix[2][1] = u[2];
		matrix[0][2] = f[0];
		matrix[1][2] = f[1];
		matrix[2][2] = f[2];
		matrix[3][0] = -fan_3d::dot(s, eye);
		matrix[3][1] = -fan_3d::dot(u, eye);
		matrix[3][2] = -fan_3d::dot(f, eye);
		return matrix;
	}

	//default
	template <typename matrix_t, typename vector_t>
	constexpr auto look_at_right(const vector_t& eye, const vector_t& center, const vector_t& up) {
		vector_t f(fan_3d::normalize(center - eye));
		vector_t s(fan_3d::normalize(cross(f, up)));
		vector_t u(fan::cross(s, f));

		matrix_t matrix(1);
		matrix[0][0] = s[0];
		matrix[1][0] = s[1];
		matrix[2][0] = s[2];
		matrix[0][1] = u[0];
		matrix[1][1] = u[1];
		matrix[2][1] = u[2];
		matrix[0][2] = -f[0];
		matrix[1][2] = -f[1];
		matrix[2][2] = -f[2];
		f_t x = -dot(s, eye);
		f_t y = -dot(u, eye);
		f_t z = dot(f, eye);
		matrix[3][0] = x;
		matrix[3][1] = y;
		matrix[3][2] = z;
		return matrix;
	}

	template <typename matrix_t, typename vector_t>
	static matrix_t rotate(const matrix_t& m, f_t angle, const vector_t& v) {
		const f_t a = angle;
		const f_t c = cos(a);
		const f_t s = sin(a);
		vector_t axis(fan_3d::normalize(v));
		vector_t temp(axis * (1.0f - c));

		matrix_t rotation;
		rotation[0][0] = c + temp[0] * axis[0];
		rotation[0][1] = temp[0] * axis[1] + s * axis[2];
		rotation[0][2] = temp[0] * axis[2] - s * axis[1];

		rotation[1][0] = temp[1] * axis[0] - s * axis[2];
		rotation[1][1] = c + temp[1] * axis[1];
		rotation[1][2] = temp[1] * axis[2] + s * axis[0];

		rotation[2][0] = temp[2] * axis[0] + s * axis[1];
		rotation[2][1] = temp[2] * axis[1] - s * axis[0];
		rotation[2][2] = c + temp[2] * axis[2];

		matrix_t matrix;
		matrix[0][0] = (m[0][0] * rotation[0][0]) + (m[1][0] * rotation[0][1]) + (m[2][0] * rotation[0][2]);
		matrix[1][0] = (m[0][1] * rotation[0][0]) + (m[1][1] * rotation[0][1]) + (m[2][1] * rotation[0][2]);
		matrix[2][0] = (m[0][2] * rotation[0][0]) + (m[1][2] * rotation[0][1]) + (m[2][2] * rotation[0][2]);

		matrix[0][1] = (m[0][0] * rotation[1][0]) + (m[1][0] * rotation[1][1]) + (m[2][0] * rotation[1][2]);
		matrix[1][1] = (m[0][1] * rotation[1][0]) + (m[1][1] * rotation[1][1]) + (m[2][1] * rotation[1][2]);
		matrix[2][1] = (m[0][2] * rotation[1][0]) + (m[1][2] * rotation[1][1]) + (m[2][2] * rotation[1][2]);

		matrix[0][2] = (m[0][0] * rotation[2][0]) + (m[1][0] * rotation[2][1]) + (m[2][0] * rotation[2][2]);
		matrix[1][2] = (m[0][1] * rotation[2][0]) + (m[1][1] * rotation[2][1]) + (m[2][1] * rotation[2][2]);
		matrix[2][2] = (m[0][2] * rotation[2][0]) + (m[1][2] * rotation[2][1]) + (m[2][2] * rotation[2][2]);

		matrix[3] = m[3];

		return matrix;
	}
}