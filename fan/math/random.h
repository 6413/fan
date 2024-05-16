#pragma once

#include <fan/types/vector.h>
#include <fan/types/color.h>
#include <fan/types/fstring.h>
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

    /**
     * Generates a random 2D integer vector with components between min and max (inclusive).
     * 
     * @param min The minimum value of the vector components.
     * @param max The maximum value of the vector components.
     * @return A random 2D integer vector with components between min and max.
     */
    inline fan::vec2i vec2i(int64_t min, int64_t max) {
      return fan::vec2i(fan::random::value_i64(min, max), fan::random::value_i64(min, max));
    }

    /**
     * Generates a random 2D floating-point vector with components between min and max (inclusive).
     * 
     * @param min The minimum value of the vector components.
     * @param max The maximum value of the vector components.
     * @return A random 2D floating-point vector with components between min and max.
     */
    inline fan::vec2 vec2(f32_t min, f32_t max) {
      return fan::vec2(fan::random::value_f32(min, max), fan::random::value_f32(min, max));
    }

    /**
     * Generates a random 2D floating-point vector with components between the corresponding components of v0 and v1.
     * 
     * @param v0 The minimum values of the vector components.
     * @param v1 The maximum values of the vector components.
     * @return A random 2D floating-point vector with components between v0 and v1.
     */
    inline fan::vec2 vec2(const fan::vec2& v0, const fan::vec2& v1) {
      return fan::vec2(fan::random::value_f32(v0.x, v1.x), fan::random::value_f32(v0.y, v1.y));
    }

    /**
     * Generates a random 3D floating-point vector with components between min and max (inclusive).
     * 
     * @param min The minimum value of the vector components.
     * @param max The maximum value of the vector components.
     * @return A random 3D floating-point vector with components between min and max.
     */
    inline fan::vec3 vec3(f32_t min, f32_t max) {
      return fan::vec3(fan::random::value_f32(min, max), fan::random::value_f32(min, max), fan::random::value_f32(min, max));
    }

    /**
     * Generates a random color with RGB components between 0 and 1, and alpha component set to 1.
     * 
     * @return A random color.
     */
    inline fan::color color() {
      return fan::color(
        fan::random::value_f32(0, 1), 
        fan::random::value_f32(0, 1),
        fan::random::value_f32(0, 1), 
        1
      );
    }

    /**
     * Represents a percent-output pair.
     */
    struct percent_output_t {
      f32_t percent; /**< The percentage chance of the output. */
      uint32_t output; /**< The output value. */
    };

    /**
     * Returns an output value based on the specified percentage chances.
     * 
     * @param po A vector of percent-output pairs.
     * @return An output value based on the percentage chances.
     */
    static uint32_t get_output_with_percent(const std::vector<percent_output_t>& po) {

      for (std::size_t i = 0; i < po.size(); i++) {
        if (!(1.0 / fan::random::value_i64(0, (uint32_t)~0) < 1.0 / (po[i].percent * (f32_t)~(uint32_t)0))) {
          return po[i].output;
        }
      }

      return -1;
    }

    /**
     * Generates a random 2D vector representing a direction with an angle between min and max (inclusive).
     * 
     * @param min The minimum angle in radians.
     * @param max The maximum angle in radians.
     * @return A random 2D vector representing a direction with an angle between min and max.
     */
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