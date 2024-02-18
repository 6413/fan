#pragma once

#include _FAN_PATH(types/vector.h)
#include _FAN_PATH(types/color.h)

#include <random>

namespace fan {

  namespace random {

    inline int c_i(uint32_t min, uint32_t max) {
      if (max == 0) {
        return 0;
      }
      return rand() % max + min;
    }

    template <typename type_t>
    struct fast_rand_t {

      fast_rand_t(type_t min, type_t max) : 
        m_random(m_device()),
        m_distance(min, max)
      { }

      type_t get() {
        return m_distance(m_random);
      }

      void set_min_max(type_t min, type_t max) {
        m_distance.param(std::uniform_int_distribution<type_t>::param_type(min, max));
      }

    protected:

      std::random_device m_device;
      std::mt19937_64 m_random;

      std::uniform_int_distribution<type_t> m_distance;

    };

   inline int64_t value_i64(int64_t min, int64_t max) {
      static std::random_device device;
      static std::mt19937_64 random(device());

      std::uniform_int_distribution<int64_t> distance(min, max);

      return distance(random);
    }
   inline int64_t i64(int64_t min, int64_t max) {
     return value_i64(min, max);
   }
    
    static constexpr auto float_accuracy = 1000000;

    inline f32_t value_f32(f32_t min, f32_t max) {
      return (f32_t)value_i64(min * float_accuracy, max * float_accuracy) / float_accuracy;
    }
    inline f32_t f32(f32_t min, f32_t max) {
      return value_f32(min, max);
    }

    static fan::string string(uint32_t len) {
      fan::string str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
      fan::string newstr;
      std::size_t pos;
      while(newstr.size() != len) {
        pos = fan::random::value_i64(0, str.size() - 1);
        newstr += str.substr(pos, 1);
      }
      return newstr;
    }

   /* static fan::utf16_string utf_string(uint32_t len) {
      fan::utf16_string str = L"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
      fan::utf16_string newstr;
      int pos;
      while(newstr.size() != len) {
        pos = fan::random::value_i64(0, str.size() - 1);
        newstr += str.substr(pos, 1);
      }
      return newstr;
    }*/

    inline fan::vec2i vec2i(int64_t min, int64_t max) {
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

    struct percent_output_t {
      f32_t percent;
      uint32_t output;
    };

    // percent 0-1
    static uint32_t get_output_with_percent(const std::vector<percent_output_t>& po) {

      for (int i = 0; i < po.size(); i++) {
        if (!(1.0 / fan::random::value_i64(0, (uint32_t)~0) < 1.0 / (po[i].percent * ~(uint32_t)0))) {
          return po[i].output;
        }
      }

      return -1;
    }

    // gives random angle between two angles
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