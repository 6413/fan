module;

#include <cstdint>
#include <cmath>
#include <limits>
#include <initializer_list>
#include <string_view>
#include <charconv>
#include <expected>
#include <optional>

export module fan.math;

import fan.types;

import fan.types.compile_time_string;

export namespace fan {
  namespace math {
    constexpr f32_t pi = 3.14159265358979323846;
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
    constexpr int sign(T x) {
      return (T(0) < x) - (x < T(0));
    }
    template <typename T>
    constexpr auto is_near(T a, T b, f64_t epsilon) {
      return std::abs(a - b) < epsilon;
    }
  }
}

export namespace fan {
  namespace math {

    template<typename T>
    const T& min(const T& a, const T& b) {
      return (b < a) ? b : a;
    }
    template<typename T> 
    const T& max(const T& a, const T& b) {
      return (a < b) ? b : a;
    }

    template <typename T>
    T lerp(T src, T dst, double t) {
      return T{ (T)(src + (dst - src) * t) };
    }

    template <typename T>
    requires (std::is_arithmetic_v<T>)
    T normalize(T val, T min, T max) {
      return (val - min) / (max - min);
    }

    double sigmoid(double x) {
      return 1.0 / (1 + std::exp(-x));
    }
    constexpr double sigmoid_derivative(double x) {
      return x * (1 - x);
    }

    template <typename T>
    T tanh_activation(T x) {
      return tanh(x);
    }
    template <typename T>
    T tanh_derivative(T x) {
      T t = tanh(x);
      return T{ 1 } - t * t;
    }

    constexpr f32_t map(f32_t value, f32_t start1, f32_t stop1, f32_t start2, f32_t stop2) {
      return start2 + (stop2 - start2) * ((value - start1) / (stop1 - start1));
    }

    constexpr f_t inf = std::numeric_limits<float>::infinity();

    constexpr f_t infinite = inf;
    constexpr f_t infinity = infinite;

    double fast_trunc(f32_t d) {
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
      return v - (f32_t)fast_trunc(v / m) * m;
    }

    int solve_quadratic(f32_t a, f32_t b, f32_t c, f32_t& root1, f32_t& root2) {
      if (std::abs(a) < 1e-6f) {
        if (std::abs(b) < 1e-6f) {
          return 0;
        }
        root1 = root2 = -c / b;
        return 1;
      }

      f32_t discriminant = b * b - 4 * a * c;
      if (discriminant < 0) {
        root1 = fan::math::inf;
        root2 = -root1;
        return 0;
      }

      f32_t q = -0.5f * (b + std::copysign(std::sqrt(discriminant), b));
      root1 = q / a;
      root2 = c / q;

      return discriminant > 0 ? 2 : 1;
    }

    template <typename T>
    bool interception_direction(const T& a, const T& b, const T& v_a, f32_t s_b, T& result) {
      T d = a - b;
      f32_t a_coef = v_a.x * v_a.x + v_a.y * v_a.y - s_b * s_b;
      f32_t b_coef = 2 * (d.x * v_a.x + d.y * v_a.y);
      f32_t c_coef = d.x * d.x + d.y * d.y;

      f32_t t1;
      f32_t t2;

      if (solve_quadratic(a_coef, b_coef, c_coef, t1, t2) == 0) {
        result = T(0);
        return false;
      }

      f32_t t = t1 > 0 && t2 > 0 ? fan::math::min(t1, t2) : fan::math::max(t1, t2);
      if (t < 0) {
        result = T(0);
        return false;
      }

      result = (d + v_a * t).normalize();
      return true;
    }

    template <typename T>
    std::optional<T> aimbot(f32_t bullet_speed, const T& start_position, const T& target_position, const T& target_vel) {
      T direction;
      if (interception_direction(target_position, start_position, target_vel, bullet_speed, direction)) {
        return direction;
      }
      return std::nullopt;
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
    constexpr std::int64_t ceil(T num)
    {
      return (static_cast<f32_t>(static_cast<std::int64_t>(num)) == num)
        ? static_cast<std::int64_t>(num)
        : static_cast<std::int64_t>(num) + ((num > 0) ? 1 : 0);
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
    constexpr typename T::value_type sign(const T& p1, const T& p2, const T& p3)
    {
      return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
    }

    template <typename val_type, std::uintptr_t n>
    val_type cross_matrix_determinant(const auto& mat) {
      if constexpr (n == 1) {
        return mat[0][0];
      }
      if constexpr (n == 2) {
        return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0];
      }
      val_type det = 0;
      if constexpr (n != 1) {
        for (std::uintptr_t j = 0; j < n; ++j) {
          val_type submat[n - 1][n - 1];
          for (std::uintptr_t row = 1; row < n; ++row) {
            std::uintptr_t colIdx = 0;
            for (std::uintptr_t col = 0; col < n; ++col) {
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
      return vector / fan::math::dot(vector, vector);
    }

    template <typename vector_t>
    vector_t snap_line_to_angle(const vector_t& start, const vector_t& end, f32_t snap_increment = 45.0f) {
      vector_t direction = end - start;
      f32_t length = direction.length();

      if (length < 1.0f) {
        return end;
      }

      f32_t current_angle = fan::math::degrees(direction.angle());
      f32_t snapped_angle = std::round(current_angle / snap_increment) * snap_increment;
      f32_t snapped_radians = fan::math::radians(snapped_angle);
      vector_t snapped_direction(std::cos(snapped_radians), std::sin(snapped_radians));

      return start + snapped_direction * length;
    }

#define PI_f32_t     3.14159265f
#define PIBY2_f32_t  1.5707963f
    // |error| < 0.005
    constexpr f32_t atan2_approximation2(f32_t y, f32_t x)
    {
      if (x == 0.0f)
      {
        if (y > 0.0f) return PIBY2_f32_t;
        if (y == 0.0f) return 0.0f;
        return -PIBY2_f32_t;
      }
      f32_t atan = 0;
      f32_t z = y / x;
      if (std::abs(z) < 1.0f)
      {
        atan = z / (1.0f + 0.28f * z * z);
        if (x < 0.0f)
        {
          if (y < 0.0f) return atan - PI_f32_t;
          return atan + PI_f32_t;
        }
      }
      else
      {
        atan = PIBY2_f32_t - z / (z * z + 0.28f);
        if (y < 0.0f) return atan - PI_f32_t;
      }
      return atan;
    }

    template <typename T, typename T2>
    constexpr auto DiamondAngle(T y, T2 x)
    {
      if (y >= 0)
        return (x >= 0 ? y / (x + y) : 1 - x / (-x + y));
      else
        return (x < 0 ? 2 - y / (-x - y) : 3 + x / (x - y));
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
      float res = 0, pow = x, fact = 1;
      for (int i = 0; i < 5; ++i)
      {
        res += pow / fact;
        pow *= -1 * x * x;
        fact *= float((2 * (i + 1)) * (2 * (i + 1) + 1));
      }

      return res;
    }

    inline f32_t cos(f32_t x) {
      f32_t t, s;
      int p;
      p = 0;
      s = 1.0;
      t = 1.0;
      while (std::abs(t / s) > .0001f)
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
      T vector = { wall.x - point.x, wall.y - point.y };
      T normal;
      if (point.y < wall.y - wall_size.y) {
        normal = { 0, -1 };
      }
      else if (point.y > wall.y + wall_size.y) {
        normal = { 0, 1 };
      }
      else if (point.x < wall.x - wall_size.x) {
        normal = { -1, 0 };
      }
      else if (point.x > wall.x + wall_size.x) {
        normal = { 1, 0 };
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
      matrix[2][2] = -2.0 / (zFar - zNear);
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
      const vector_t f((eye - center).normalize());
      const vector_t s((fan::math::cross(f, up)).normalize());
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
      matrix[3][0] = -fan::math::dot(s, eye);
      matrix[3][1] = -fan::math::dot(u, eye);
      matrix[3][2] = -fan::math::dot(f, eye);
      return matrix;
    }

    //default
    template <typename matrix_t, typename vector_t>
    constexpr auto look_at_right(const vector_t& eye, const vector_t& center, const vector_t& up) {
      vector_t f(fan::math::normalize(center - eye));
      vector_t s(fan::math::normalize(cross(f, up)));
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
      f_t x = -fan::math::dot(s, eye);
      f_t y = -fan::math::dot(u, eye);
      f_t z = fan::math::dot(f, eye);
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
    constexpr bool dcom_fr(std::uintptr_t n, T x, T y) noexcept {
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

    template <typename T>
    struct less {
      constexpr bool operator()(const T& lhs, const T& rhs) const {
        return lhs < rhs;
      }
    };
    template<class T, class Compare>
    constexpr const T& clamp(const T& v, const T& lo, const T& hi, Compare comp) {
      return comp(v, lo) ? lo : comp(hi, v) ? hi : v;
    }
    template<class T>
    constexpr const T& clamp(const T& v, const T& lo, const T& hi) {
      return clamp(v, lo, hi, fan::math::less<T>{});
    }

    template <typename T>
    constexpr T round(const T& v) {
      return (v >= T(0)) ? static_cast<T>(static_cast<long long>(v + T(0.5)))
        : static_cast<T>(static_cast<long long>(v - T(0.5)));
    }

    enum class error { parse_left, parse_right, bad_op, div_by_zero, overflow };

    struct error_info {
      error code;
      fan::str_view_t detail;
    };

    constexpr bool parse_double(std::string_view s, double& out) noexcept {
      if (s.empty()) return false;
      double result = 0;
      int sign = 1;
      size_t i = 0;
      if (i < s.size() && (s[i] == '-' || s[i] == '+')) sign = (s[i++] == '-') ? -1 : 1;
      if (i >= s.size()) return false;
      bool any = false;
      while (i < s.size() && s[i] >= '0' && s[i] <= '9') { result = result * 10 + (s[i++] - '0'); any = true; }
      if (i < s.size() && s[i] == '.') {
        ++i;
        double frac = 0.1;
        while (i < s.size() && s[i] >= '0' && s[i] <= '9') { result += (s[i++] - '0') * frac; frac *= 0.1; any = true; }
      }
      if (!any) return false;
      out = result * sign;
      return true;
    }

    constexpr std::expected<double, error> compute_from_strings(std::string_view lhs, char op, std::string_view rhs) noexcept {
      double a {}, b {};
      if (!parse_double(lhs, a)) return std::unexpected(error::parse_left);
      if (!parse_double(rhs, b)) return std::unexpected(error::parse_right);
      if (op == '/' && fan::math::abs(b) < 1e-12) return std::unexpected(error::div_by_zero);
      switch (op) {
      case '+': return a + b;
      case '-': return a - b;
      case '*': return a * b;
      case '/': return a / b;
      default:  return std::unexpected(error::bad_op);
      }
    }

    constexpr std::expected<double, error_info> compute_from_strings_with_detail(std::string_view lhs, char op, std::string_view rhs) noexcept {
      double a {}, b {};
      if (!parse_double(lhs, a)) return std::unexpected(error_info {error::parse_left, fan::str_view_t(lhs)});
      if (!parse_double(rhs, b)) return std::unexpected(error_info {error::parse_right, fan::str_view_t(rhs)});
      if (op == '/' && fan::math::abs(b) < 1e-12) return std::unexpected(error_info {error::div_by_zero, fan::str_view_t(rhs)});
      switch (op) {
      case '+': return a + b;
      case '-': return a - b;
      case '*': return a * b;
      case '/': return a / b;
      default:  return std::unexpected(error_info {error::bad_op, fan::str_view_t(&op, 1)});
      }
    }

    constexpr std::expected<double, error> eval_simple_expr(std::string_view expr) noexcept {
      for (size_t i = 1; i < expr.size(); ++i) {
        char c = expr[i];
        if (c == '+' || c == '-' || c == '*' || c == '/') {
          auto lhs = expr.substr(0, i);
          auto rhs = expr.substr(i + 1);
          return compute_from_strings(lhs, c, rhs);
        }
      }
      return std::unexpected(error::bad_op);
    }

    constexpr std::expected<double, error_info> eval_simple_expr_with_detail(std::string_view expr) noexcept {
      for (size_t i = 1; i < expr.size(); ++i) {
        char c = expr[i];
        if (c == '+' || c == '-' || c == '*' || c == '/') {
          auto lhs = expr.substr(0, i);
          auto rhs = expr.substr(i + 1);
          return compute_from_strings_with_detail(lhs, c, rhs);
        }
      }
    }
  } // namespace fan::math
} // namespace fan