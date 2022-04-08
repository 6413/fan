#pragma once
#ifdef _MSC_VER
#pragma warning (disable : 4244)
#endif

#include <cfloat>

#include <functional>
#include <cmath>

namespace fan { 
	namespace math {
		constexpr f_t pi = 3.14159265358979323846264338327950288419716939937510;
		constexpr f_t half_pi = pi / 2;
		constexpr f_t two_pi = pi * 2;
	}
}

namespace fan_2d {
	
	namespace math {

		// for reverse y coordinates
		template <typename vector2d_t>
		vector2d_t velocity_resolve_in_collision(const vector2d_t& velocity_src, const vector2d_t& velocity_dst, const vector2d_t& normal) {
			vector2d_t direction = velocity_dst - velocity_src;
			f32_t angle = fan::math::pi - 2 * atan2(-normal.y, -normal.x);
			f32_t cos_ = cos(angle);
			f32_t sin_ = sin(angle);
			return velocity_dst + vector2d_t(direction.x * cos_ - direction.y * sin_, -(direction.x * sin_ + direction.y * cos_));
		}

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
			f_t length = sqrt(fan_2d::math::dot(vector, vector));
			if (!length) {
				return T();
			}
			return T(vector.x / length, vector.y / length);
		}

		template <typename T>
		constexpr auto manhattan_distance(const T& src, const T& dst) {
			return std::abs(src.x - dst.x) + std::abs(src.y - dst.y);
		}

		template <typename T>
		inline auto pythagorean(const T& vector) {
			return std::sqrt((vector.x * vector.x) + (vector.y * vector.y));
		}

	}

}

namespace fan_3d {

	namespace math {

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

		template <typename T>
		constexpr auto manhattan_distance(const T& src, const T& dst) {
			return std::abs(src.x - dst.x) + std::abs(src.y - dst.y) + std::abs(src.z - dst.z);
		}

	}

	
}

namespace fan {

	namespace math {

		constexpr f_t inf = INFINITY;
		constexpr f_t infinite = inf;
		constexpr f_t infinity = infinite;

		constexpr f32_t fast_trunc(f32_t d) {
      unsigned constexpr MANTISSA_BITS = 52,
        HI_MANTISSA_BITS = 20,
        EXP_BIAS = 0x3FF,
        INF_NAN_BASE = 0x7FF;
      uint32_t constexpr EXP_MASK = (uint32_t)0x7FFu << HI_MANTISSA_BITS,
        SIGN_MASK = (uint32_t)0x800u << HI_MANTISSA_BITS,
        MIN_INTEGRAL_DIGITS_EXP = (uint32_t)EXP_BIAS << HI_MANTISSA_BITS,
        MAX_INTEGRAL32_EXP = (uint32_t)(EXP_BIAS + HI_MANTISSA_BITS) << HI_MANTISSA_BITS,
        MIN_INTEGRAL_ONLY_EXP = (uint32_t)(EXP_BIAS + MANTISSA_BITS) << HI_MANTISSA_BITS,
        INF_NAN_EXP = (uint32_t)INF_NAN_BASE << HI_MANTISSA_BITS,
        NEG_HI_MANTISSA_MASK = 0x000FFFFFu,
        NEG_LO_MANTISSA_MASK = 0xFFFFFFFFu;
      union
      {
        double du;
        struct
        {
          uint32_t dxLo;
          uint32_t dxHi;
        }dx;
      };
      du = d;
      uint32_t exp = dx.dxHi & EXP_MASK;
      if (exp >= MIN_INTEGRAL_DIGITS_EXP)
        if (exp < MIN_INTEGRAL_ONLY_EXP)
          if (exp <= MAX_INTEGRAL32_EXP)
          {
            unsigned shift = (unsigned)(exp >> HI_MANTISSA_BITS) - EXP_BIAS;
            dx.dxHi &= ~(NEG_HI_MANTISSA_MASK >> shift);
            dx.dxLo = 0;
            return du;
          }
          else
          {
            unsigned shift = (unsigned)(exp >> HI_MANTISSA_BITS) - EXP_BIAS - HI_MANTISSA_BITS;
            dx.dxLo &= ~(NEG_LO_MANTISSA_MASK >> shift);
            return du;
          }
        else
          if (exp < INF_NAN_EXP)
            return du;
          else
            return du + du;
      else
      {
        dx.dxHi &= SIGN_MASK;
        dx.dxLo = 0;
        return du;
      }
		}

		constexpr f32_t fast_fmod(f32_t v, f32_t m)
    {
        return v - fast_trunc(v / m ) * m ;
    }

		static int solve_quadratic(f32_t a, f32_t b, f32_t c, f32_t& root1, f32_t& root2) {
			f32_t discriminant = b * b - 4 * a * c;
			if (discriminant < 0) {
				root1 = fan::math::inf;
				root2 = -root1;
				return 0;
			}

			root1 = (-b + sqrt(discriminant)) / (2 * a);
			root2 = (-b - sqrt(discriminant)) / (2 * a);

			return discriminant > 0 ? 2 : 1;
		}

		template <typename T>
		static bool interception_direction(const T& a, const T& b, const T& v_a, f32_t s_b, T& result) {
			T a_to_b = b - a;
			f32_t d_c = a_to_b.length();

			T x = a_to_b;
			T y = v_a;

			f32_t alpha = atan2(y.y, y.x) - atan2(x.y, x.x);
			f32_t s_a = v_a.length();
			f32_t r = s_a / s_b;

			f32_t root1;
			f32_t root2;

			if (solve_quadratic(1 - r * r, 2 * r * d_c * cos(alpha), -(d_c * d_c), root1, root2) == 0) {
				result = 0;
				return false;
			}

			f32_t d_a = std::max(root1, root2);
			f32_t t = d_a / s_b;
			T c = a + v_a * t;
			result = (c - b).normalize();
			return true;

		}

		// no delta
		template <typename T>
		T aimbot(f32_t bullet_speed, const T& start_position, const T& target_position, const T& target_vel) {
			T direction;
			if (interception_direction(target_position, start_position, target_vel, 2 * (bullet_speed / 10), direction)) {
				return direction;
			}
			return fan::math::inf;
		}

		inline auto pythagorean(f_t a, f_t b) {
			return std::sqrt((a * a) + (b * b));
		}

		template <typename T>
		constexpr auto abs(T value) {
			return value < 0 ? -value : value;
		}

		template <typename T>
		constexpr auto modi(T first, T second) {
			return (first % second + second) % second;
		}

		// could be improved with look up table
		template <typename T>
		constexpr auto number_of_digits(T x)
		{
			return x > 0 ? static_cast<int>(std::log10(x)) + 1 : 1;
		}

		template <typename T>
		constexpr int64_t ceil(T num)
		{
			return (static_cast<f32_t>(static_cast<int64_t>(num)) == num)
				? static_cast<int64_t>(num)
				: static_cast<int64_t>(num) + ((num > 0) ? 1 : 0);
		}

		template <typename T>
		void debugger(std::function<T> functionPtr) {
			printf("start\n");
			functionPtr();
			printf("end\n");
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

		template <typename T>
		constexpr typename T::value_type sign (const T& p1, const T& p2, const T& p3)
		{
			return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
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
			return vector / fan_3d::math::dot(vector, vector);
		}

		#define PI_f32_t     3.14159265f
		#define PIBY2_f32_t  1.5707963f
		// |error| < 0.005
		constexpr f32_t atan2_approximation2( f32_t y, f32_t x )
		{
			if ( x == 0.0f )
			{
				if ( y > 0.0f ) return PIBY2_f32_t;
				if ( y == 0.0f ) return 0.0f;
				return -PIBY2_f32_t;
			}
			f32_t atan;
			f32_t z = y/x;
			if ( fabs( z ) < 1.0f )
			{
				atan = z/(1.0f + 0.28f*z*z);
				if ( x < 0.0f )
				{
					if ( y < 0.0f ) return atan - PI_f32_t;
					return atan + PI_f32_t;
				}
			}
			else
			{
				atan = PIBY2_f32_t - z/(z*z + 0.28f);
				if ( y < 0.0f ) return atan - PI_f32_t;
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
				(sin(fan::math::radians(alpha)) * cos(fan::math::radians(beta))),
				sin(radians(beta)),
				(cos(fan::math::radians(alpha)) * cos(fan::math::radians(beta)))
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

		template <typename matrix_t>
		auto ortho(f32_t left, f32_t right, f32_t bottom, f32_t top) {
			matrix_t matrix(1);
			matrix[0][0] = static_cast<f32_t>(2) / (right - left);
			matrix[1][1] = static_cast<f32_t>(2) / (top - bottom);
			matrix[2][2] = -static_cast<f32_t>(1);
			matrix[3][0] = -(right + left) / (right - left);
			matrix[3][1] = -(top + bottom) / (top - bottom);
			return matrix;
		}

		// left
		template <typename matrix_t>
		constexpr auto ortho(f_t left, f_t right, f_t bottom, f_t top, f_t zNear, f_t zFar) {
			matrix_t matrix(1);
			matrix[0][0] = 2.0 / (right - left);
			matrix[1][1] = 2.0 / (top - bottom);
			matrix[2][2] = 1.0 / (zFar - zNear);
			matrix[3][0] = -(right + left) / (right - left);
			matrix[3][1] = -(top + bottom) / (top - bottom);
			matrix[3][2] = -zNear / (zFar - zNear);
			return matrix;
		}

		template <typename matrix_t>
		constexpr auto ortho_right(f_t left, f_t right, f_t bottom, f_t top, f_t zNear, f_t zFar) {
			matrix_t matrix(1);
			matrix[0][0] = 2.0 / (right - left);
			matrix[1][1] = 2.0 / (top - bottom);
			matrix[2][2] = - 2.0 / (zFar - zNear);
			matrix[3][0] = -(right + left) / (right - left);
			matrix[3][1] = -(top + bottom) / (top - bottom);
			matrix[3][2] = -(zFar + zNear) / (zFar - zNear);
			return matrix;
		}

		template <typename matrix_t>
		constexpr matrix_t perspective(f_t fovy, f_t aspect, f_t zNear, f_t zFar) {
			f_t const tanHalfFovy = tan(fovy / static_cast<f_t>(2));
			matrix_t matrix{};
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
			const vector_t f(fan_3d::math::normalize(eye - center));
			const vector_t s(fan_3d::math::normalize(fan::math::cross(f, up)));
			const vector_t u(fan::math::cross(s, f));

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
			matrix[3][0] = -fan_3d::math::dot(s, eye);
			matrix[3][1] = -fan_3d::math::dot(u, eye);
			matrix[3][2] = -fan_3d::math::dot(f, eye);
 			return matrix;
		}

		//default
		template <typename matrix_t, typename vector_t>
		constexpr auto look_at_right(const vector_t& eye, const vector_t& center, const vector_t& up) {
			vector_t f(fan_3d::math::normalize(center - eye));
			vector_t s(fan_3d::math::normalize(cross(f, up)));
			vector_t u(fan::math::cross(s, f));

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
			f_t x = -fan_3d::math::dot(s, eye);
			f_t y = -fan_3d::math::dot(u, eye);
			f_t z = fan_3d::math::dot(f, eye);
			matrix[3][0] = x;
			matrix[3][1] = y;
			matrix[3][2] = z;
			return matrix;
		}

		constexpr auto RAY_DID_NOT_HIT = fan::math::inf;

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
		constexpr bool dcom_fr(uintptr_t n, T x, T y) noexcept {
			switch (n) {
				case 0: {
					return x < y;
				}
				case 1: {
					return x > y;
				}
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
					return T(p1Start[0] + t * (p1End[0] - p1Start[0]), p1Start[1] + t * (p1End[1] - p1Start[1]));;
				}
			}
			else {
				if (t >= 0 && u >= 0 && u <= 1) {
					return T(p1Start[0] + t * (p1End[0] - p1Start[0]), p1Start[1] + t * (p1End[1] - p1Start[1]));
				}
			}

			return RAY_DID_NOT_HIT;
		}

	}

}