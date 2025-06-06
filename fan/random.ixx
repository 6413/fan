module;

#include <fan/types/types.h>
#include <fan/math/math.h>

#include <random>

export module fan.random;

export import fan.types.vector;
export import fan.types.color;

export namespace fan {

  namespace random {

    inline int c_i(uint32_t min, uint32_t max) {
      if (max == 0) {
        return 0;
      }
      return rand() % max + min;
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
    inline int64_t value_i64(int64_t min, int64_t max) {
      return value<int64_t>(min, max);
    }
    inline int64_t i64(int64_t min, int64_t max) {
      return value_i64(min, max);
    }
    inline f32_t value_f32(f32_t min, f32_t max) {
      return value<f32_t>(min, max);
    }
    inline f32_t f32(f32_t min, f32_t max) {
      return value_f32(min, max);
    }

    fan_module_api std::string string(uint32_t len) {
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

    fan_module_api fan::vec2i vec2i(int64_t min, int64_t max) {
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

    inline fan::color color() {
      return fan::color(
        fan::random::value_f32(0, 1),
        fan::random::value_f32(0, 1),
        fan::random::value_f32(0, 1),
        1
      );
    }
    // always makes one channel brightest and scales other channels accordingly
    inline fan::color bright_color() {
      fan::color rand_color = fan::random::color();
      f32_t max_channel = fan::max({rand_color.r, rand_color.g, rand_color.b});
      return rand_color / max_channel;
    }

    struct percent_output_t {
      f32_t percent;
      uint32_t output;
    };

    fan_module_api uint32_t get_output_with_percent(const std::vector<percent_output_t>& po) {

      for (std::size_t i = 0; i < po.size(); i++) {
        if (!(1.0 / fan::random::value_i64(0, (uint32_t)~0) < 1.0 / (po[i].percent * (f32_t)~(uint32_t)0))) {
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
      return fan::vec2(cos(r), sin(r));
    }
  }
}