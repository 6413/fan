#pragma once
#ifdef _MSC_VER
#pragma warning (disable : 4244)
#endif

#include <cfloat>

#include <cmath>

#include <fan/types/types.h>

namespace fan { 
	namespace math {
		constexpr f32_t pi = 3.1415927f;
		constexpr f32_t half_pi = pi / 2;
		constexpr f32_t two_pi = pi * 2;

		template <typename vector_t>
		constexpr typename vector_t::value_type dot(const vector_t& x, const vector_t& y) {
		  typename vector_t::value_type ret = 0;
			for (uintptr_t i = 0; i < vector_t::size(); ++i) {
				ret += x[i] * y[i];
			}
			return ret;
		}
    template <typename T> 
    int sgn(T val) {
      return (T(0) < val) - (val < T(0));
    }
    template <typename T>
    constexpr auto is_near(T a, T b, f64_t epsilon) {
      return std::abs(a - b) < epsilon;
    }
	}
}

namespace fan_2d {
	
  	namespace math {

		// for reverse y coordinates
		template <typename vector2d_t>
		vector2d_t velocity_resolve_in_collision(const vector2d_t& velocity_src, const vector2d_t& velocity_dst, const vector2d_t& normal) {
			vector2d_t direction = velocity_dst - velocity_src;
			f32_t angle = fan::math::pi - 2 * atan2(-normal.y, -normal.x);
			f32_t cos_ = std::cos(angle);
			f32_t sin_ = std::sin(angle);
			return velocity_dst + vector2d_t(direction.x * cos_ - direction.y * sin_, -(direction.x * sin_ + direction.y * cos_));
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
			f_t length = sqrt(fan::math::dot(vector, vector));
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

    template <typename T>
    T lerp(T src, T dst, T t) {
      return { src + t * (dst - src) };
    }

    template <typename T>
    T normalize(T val, T min, T max) {
      return (val - min) / (max - min);
    }

    static double sigmoid(double x) {
      return 1.0 / (1 + exp(-x));
    }

    static constexpr double sigmoid_derivative(double x) {
      return x * (1 - x);
    }

    constexpr f32_t map(f32_t value, f32_t start1, f32_t stop1, f32_t start2, f32_t stop2) {
      return start2 + (stop2 - start2) * ((value - start1) / (stop1 - start1));
    }

		constexpr f_t inf = INFINITY;
		constexpr f_t infinite = inf;
		constexpr f_t infinity = infinite;

		static double fast_trunc(f32_t d) {
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

		inline f32_t fast_fmod(f32_t v, f32_t m)
    {
        return v - (f32_t)fast_trunc(v / m ) * m ;
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

    template <typename val_type, uintptr_t n>
		val_type cross_matrix_determinant(const auto &mat) {
			if constexpr(n == 1) {
				return mat[0][0];
			}
			if constexpr(n == 2) {
				return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0];
			}
			val_type det = 0;
			if constexpr(n != 1){
				for (uintptr_t j = 0; j < n; ++j) {
					val_type submat[n - 1][n - 1];
					for (uintptr_t row = 1; row < n; ++row) {
						uintptr_t colIdx = 0;
						for (uintptr_t col = 0; col < n; ++col) {
							if (col != j) {
								submat[row - 1][colIdx] = mat[row][col];
								++colIdx;
							}
						}
					}
					
					det += (j % 2 ? -1 : +1) * mat[0][j] * cross_matrix_determinant<val_type, n - 1>(submat);
				}
			}
			
      return det;
    }

    template <typename vec_t, typename... vecs_t>
    constexpr auto cross(const vec_t& first, const vecs_t&... rest) {
      constexpr size_t n = sizeof...(vecs_t) + 1;
      using value_type = typename vec_t::value_type;
      vec_t result;
      const vec_t* vectors[] = { &first, &rest... };
      for (size_t i = 0; i < vec_t::size(); ++i) {
        value_type submat[n][n];
        for (size_t row = 0; row < n; ++row) {
          size_t colIdx = 0;
          for (size_t col = 0; col < vec_t::size(); ++col) {
            if (col != i) {
              submat[row][colIdx++] = (*vectors[row])[col];
            }
          }
        }
        value_type det = cross_matrix_determinant<value_type, n>(submat);
        result[i] = (i % 2 == 0) ? det : -det;
      }
      return result;
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
			f32_t atan = 0;
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
				std::sin(angle),
				-std::cos(angle)
			);
		}

		// depends about world rotation

		inline f32_t sin(f32_t x) {
			float res=0, pow=x, fact=1;
			for(int i=0; i<5; ++i)
			{
				res+=pow/fact;
				pow*=-1*x*x;
				fact *= float((2 * (i + 1)) * (2 * (i + 1) + 1));
			}

			return res;
		}

		inline f32_t cos(f32_t x) {
			f32_t t, s ;
			int p;
			p = 0;
			s = 1.0;
			t = 1.0;
			while(fabs(t/s) > .0001f)
			{
				p++;
				t = f32_t((f32_t)(-t * x * x) / (f32_t)((2 * p - 1) * (2 * p)));
				s += t;
			}
			return s;
		}

		template <typename vector_t>
		inline vector_t direction_vector(f32_t alpha, f32_t beta)
		{
			return vector_t(
				(sin(fan::math::radians(alpha)) * cos(fan::math::radians(beta))),
				sin(radians(beta)),
				(cos(fan::math::radians(alpha)) * cos(fan::math::radians(beta)))
			);
		}

    // calculate reflection velocity from wall from a point
    // takes half size of rect
    template <typename T>
    auto reflection_no_rot(const T& velocity, const T& point, const T& wall, const T& wall_size) {
      T vector = {wall.x - point.x, wall.y - point.y};
      T normal;
      if (point.y < wall.y - wall_size.y) {
        normal = {0, -1};
      }
      else if (point.y > wall.y + wall_size.y) {
        normal = {0, 1};
      }
      else if (point.x < wall.x - wall_size.x) {
        normal = {-1, 0};
      }
      else if (point.x > wall.x + wall_size.x) {
        normal = {1, 0};
      }

      auto dot = velocity.dot(normal);

      T reflection;
      reflection.x = velocity.x - 2 * dot * normal.x;
      reflection.y = velocity.y - 2 * dot * normal.y;
      return reflection;
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
		constexpr bool on_hit(const T& point, auto lambda) {
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

    auto hypotenuse(const auto& vector) {
      return std::sqrt((vector.x * vector.x) + (vector.y * vector.y));
    }
    template <typename T>
    auto copysign(const T& mag, const T& sgn) {
      return T(
        std::copysign(mag.x, sgn.x),
        std::copysign(mag.y, sgn.y)
      );
    }
	}
}

#ifndef __compile_time_64log2
  #define __compile_time_64log2(v) ( \
    (v) >= 0x8000000000000000 ? 0x3f : \
    (v) >= 0x4000000000000000 ? 0x3e : \
    (v) >= 0x2000000000000000 ? 0x3d : \
    (v) >= 0x1000000000000000 ? 0x3c : \
    (v) >= 0x0800000000000000 ? 0x3b : \
    (v) >= 0x0400000000000000 ? 0x3a : \
    (v) >= 0x0200000000000000 ? 0x39 : \
    (v) >= 0x0100000000000000 ? 0x38 : \
    (v) >= 0x0080000000000000 ? 0x37 : \
    (v) >= 0x0040000000000000 ? 0x36 : \
    (v) >= 0x0020000000000000 ? 0x35 : \
    (v) >= 0x0010000000000000 ? 0x34 : \
    (v) >= 0x0008000000000000 ? 0x33 : \
    (v) >= 0x0004000000000000 ? 0x32 : \
    (v) >= 0x0002000000000000 ? 0x31 : \
    (v) >= 0x0001000000000000 ? 0x30 : \
    (v) >= 0x0000800000000000 ? 0x2f : \
    (v) >= 0x0000400000000000 ? 0x2e : \
    (v) >= 0x0000200000000000 ? 0x2d : \
    (v) >= 0x0000100000000000 ? 0x2c : \
    (v) >= 0x0000080000000000 ? 0x2b : \
    (v) >= 0x0000040000000000 ? 0x2a : \
    (v) >= 0x0000020000000000 ? 0x29 : \
    (v) >= 0x0000010000000000 ? 0x28 : \
    (v) >= 0x0000008000000000 ? 0x27 : \
    (v) >= 0x0000004000000000 ? 0x26 : \
    (v) >= 0x0000002000000000 ? 0x25 : \
    (v) >= 0x0000001000000000 ? 0x24 : \
    (v) >= 0x0000000800000000 ? 0x23 : \
    (v) >= 0x0000000400000000 ? 0x22 : \
    (v) >= 0x0000000200000000 ? 0x21 : \
    (v) >= 0x0000000100000000 ? 0x20 : \
    (v) >= 0x0000000080000000 ? 0x1f : \
    (v) >= 0x0000000040000000 ? 0x1e : \
    (v) >= 0x0000000020000000 ? 0x1d : \
    (v) >= 0x0000000010000000 ? 0x1c : \
    (v) >= 0x0000000008000000 ? 0x1b : \
    (v) >= 0x0000000004000000 ? 0x1a : \
    (v) >= 0x0000000002000000 ? 0x19 : \
    (v) >= 0x0000000001000000 ? 0x18 : \
    (v) >= 0x0000000000800000 ? 0x17 : \
    (v) >= 0x0000000000400000 ? 0x16 : \
    (v) >= 0x0000000000200000 ? 0x15 : \
    (v) >= 0x0000000000100000 ? 0x14 : \
    (v) >= 0x0000000000080000 ? 0x13 : \
    (v) >= 0x0000000000040000 ? 0x12 : \
    (v) >= 0x0000000000020000 ? 0x11 : \
    (v) >= 0x0000000000010000 ? 0x10 : \
    (v) >= 0x0000000000008000 ? 0x0f : \
    (v) >= 0x0000000000004000 ? 0x0e : \
    (v) >= 0x0000000000002000 ? 0x0d : \
    (v) >= 0x0000000000001000 ? 0x0c : \
    (v) >= 0x0000000000000800 ? 0x0b : \
    (v) >= 0x0000000000000400 ? 0x0a : \
    (v) >= 0x0000000000000200 ? 0x09 : \
    (v) >= 0x0000000000000100 ? 0x08 : \
    (v) >= 0x0000000000000080 ? 0x07 : \
    (v) >= 0x0000000000000040 ? 0x06 : \
    (v) >= 0x0000000000000020 ? 0x05 : \
    (v) >= 0x0000000000000010 ? 0x04 : \
    (v) >= 0x0000000000000008 ? 0x03 : \
    (v) >= 0x0000000000000004 ? 0x02 : \
    (v) >= 0x0000000000000002 ? 0x01 : \
    0 \
  )
#endif

#ifndef __compile_time_32log2
  #define __compile_time_32log2(v) ( \
    (v) >= 0x80000000 ? 0x1f : \
    (v) >= 0x40000000 ? 0x1e : \
    (v) >= 0x20000000 ? 0x1d : \
    (v) >= 0x10000000 ? 0x1c : \
    (v) >= 0x08000000 ? 0x1b : \
    (v) >= 0x04000000 ? 0x1a : \
    (v) >= 0x02000000 ? 0x19 : \
    (v) >= 0x01000000 ? 0x18 : \
    (v) >= 0x00800000 ? 0x17 : \
    (v) >= 0x00400000 ? 0x16 : \
    (v) >= 0x00200000 ? 0x15 : \
    (v) >= 0x00100000 ? 0x14 : \
    (v) >= 0x00080000 ? 0x13 : \
    (v) >= 0x00040000 ? 0x12 : \
    (v) >= 0x00020000 ? 0x11 : \
    (v) >= 0x00010000 ? 0x10 : \
    (v) >= 0x00008000 ? 0x0f : \
    (v) >= 0x00004000 ? 0x0e : \
    (v) >= 0x00002000 ? 0x0d : \
    (v) >= 0x00001000 ? 0x0c : \
    (v) >= 0x00000800 ? 0x0b : \
    (v) >= 0x00000400 ? 0x0a : \
    (v) >= 0x00000200 ? 0x09 : \
    (v) >= 0x00000100 ? 0x08 : \
    (v) >= 0x00000080 ? 0x07 : \
    (v) >= 0x00000040 ? 0x06 : \
    (v) >= 0x00000020 ? 0x05 : \
    (v) >= 0x00000010 ? 0x04 : \
    (v) >= 0x00000008 ? 0x03 : \
    (v) >= 0x00000004 ? 0x02 : \
    (v) >= 0x00000002 ? 0x01 : \
    0 \
  )
#endif

#ifndef __compile_time_log2
  #define __compile_time_log2 CONCAT3(__compile_time_,SYSTEM_BIT,log2)
#endif

#ifndef __fast_8log2
  #define __fast_8log2 __fast_8log2
  __forceinline static uint8_t __fast_8log2(uint8_t v){
    return 31 - __clz32(v);
  }
#endif
#ifndef __fast_16log2
  #define __fast_16log2 __fast_16log2
  __forceinline static uint8_t __fast_16log2(uint16_t v){
    return 31 - __clz32(v);
  }
#endif
#ifndef __fast_32log2
  #define __fast_32log2 __fast_32log2
  __forceinline static uint8_t __fast_32log2(uint32_t v){
    return 31 - __clz32(v);
  }
#endif
#ifndef __fast_64log2
  #define __fast_64log2 __fast_64log2
  __forceinline static uint8_t __fast_64log2(uint64_t v){
    return 63 - __clz64(v);
  }
#endif
#ifndef __fast_log2
  #define __fast_log2 CONCAT3(__fast_,SYSTEM_BIT,log2)
#endif