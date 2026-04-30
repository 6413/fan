module;

export module fan.random;

import std;

import fan.types;
import fan.math;
import fan.types.vector;

export namespace fan {

  namespace random {

    inline int c_i(std::uint32_t min, std::uint32_t max) {
      if (max == 0) {
        return 0;
      }
      return std::rand() % max + min;
    }

    template <typename T>
    T value(const T& vmin, const T& vmax) {
      static std::random_device device;
      static std::mt19937_64 random(device());

      if constexpr (std::is_integral_v<T>) {
        std::uniform_int_distribution<T> distribution(vmin, vmax);
        return distribution(random);
      }
      else if constexpr (std::is_floating_point_v<T>) {
        std::uniform_real_distribution<T> distribution(vmin, vmax);
        return distribution(random);
      }
      else {
        static_assert(std::is_arithmetic_v<T>, "Arguments must be arithmetic");
      }
    }
    // legacy
    inline std::int64_t value_i64(std::int64_t min, std::int64_t max) {
      return value<std::int64_t>(min, max);
    }
    inline std::int64_t i64(std::int64_t min, std::int64_t max) {
      return value_i64(min, max);
    }
    inline f32_t value_f32(f32_t min, f32_t max) {
      return value<f32_t>(min, max);
    }
    inline f32_t f32(f32_t min, f32_t max) {
      return value_f32(min, max);
    }

    std::string string(std::uint32_t len) {
      std::string str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
      std::string newstr;
      std::size_t pos;
      newstr.reserve(len);
      while (newstr.size() != len) {
        pos = fan::random::value_i64(0, str.size() - 1);
        newstr += str[pos];
      }
      return newstr;
    }

    // static fan::utf16_string utf_string(uint32_t len) {
    //   fan::utf16_string str = L"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    //   fan::utf16_string newstr;
    //   int pos;
    //   while(newstr.size() != len) {
    //     pos = fan::random::value_i64(0, str.size() - 1);
    //     newstr += str.substr(pos, 1);
    //   }
    //   return newstr;
    // }

    fan::vec2i vec2i(std::int64_t min, std::int64_t max) {
      return fan::vec2i(fan::random::value_i64(min, max), fan::random::value_i64(min, max));
    }

    inline fan::vec2 vec2(f32_t min, f32_t max) {
      return fan::vec2(fan::random::value_f32(min, max), fan::random::value_f32(min, max));
    }

    inline fan::vec2 vec2(const fan::vec2& v0, const fan::vec2& v1) {
      return fan::vec2(fan::random::value_f32(v0.x, v1.x), fan::random::value_f32(v0.y, v1.y));
    }

    inline fan::vec3 vec3(f32_t min, f32_t max) {
      return fan::vec3(fan::random::value_f32(min, max), fan::random::value_f32(min, max), fan::random::value_f32(min, max));
    }

    template<typename T_t>
    inline T_t vec(const T_t& v0, const T_t& v1) {
      return {fan::random::value(v0.x, v1.x), fan::random::value(v0.y, v1.y)};
    }

    struct percent_output_t {
      f32_t percent;
      std::uint32_t output;
    };

    std::uint32_t get_output_with_percent(const std::vector<percent_output_t>& po) {

      for (std::size_t i = 0; i < po.size(); i++) {
        if (!(1.0 / fan::random::value_i64(0, (std::uint32_t)~0) < 1.0 / (po[i].percent * (f32_t)~(std::uint32_t)0))) {
          return po[i].output;
        }
      }

      return -1;
    }

    inline fan::vec2 vec2_direction(f32_t min, f32_t max) {
      min = -min;
      max = -max;

      if (min > max) {
        std::swap(min, max);
      }

      f32_t r = fan::random::value_f32(min, max);
      return fan::vec2(std::cos(r), std::sin(r));
    }
    f32_t angle_45() {
      int i = fan::random::value_i64(-4, 4);
      return i * (fan::math::pi / 4);
    }
    f32_t angle_90() {
      int i = fan::random::value_i64(-2, 2);
      return i * (fan::math::pi / 2);
    }
    f32_t angle_180() {
      int i = fan::random::value_i64(-1, 1);
      return i * fan::math::pi;
    }
    f32_t angle() {
      return fan::random::value_f32(-fan::math::pi, fan::math::pi);
    }
    std::vector<std::uint8_t> pixels(std::size_t count) {
      std::vector<std::uint8_t> result(count);
      for (auto& p : result) p = value(0.f, 255.f);
      return result;
    }
    fan::vec2 border_pos(const fan::vec2& bounds, f32_t margin) {
      bool side = fan::random::value(0, 1);
      bool edge = fan::random::value(0, 1);
      if (side) { return {edge ? -margin : bounds.x + margin, fan::random::value(0.f, bounds.y)}; }
      return {fan::random::value(0.f, bounds.x), edge ? -margin : bounds.y + margin};
    }
  }
}