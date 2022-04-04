#pragma once

#include <fan/types/vector.h>

namespace fan_2d {
  namespace graphics {
    namespace lighting {
      struct light_t {

        void open() {
          instance.open();
        }
        void close() {
          instance.close();
        }

        struct properties_t {
          fan::vec2 position = 0;
          f32_t radius = 100;
          f32_t intensity = 1;
          f32_t ambient_strength = 0.1;
          fan::color color = fan::colors::white;
        };

        void push_back(const properties_t& p) {
          instance.push_back(p);
        }

        fan::vec2 get_position(uint32_t i) const {
          return instance[i].position;
        }
        void set_position(uint32_t i, const fan::vec2& position) {
          instance[i].position = position;
        }

        f32_t get_radius(uint32_t i) const {
          return instance[i].radius;
        }
        void set_radius(uint32_t i, f32_t radius) {
          instance[i].radius = radius;
        }

        f32_t get_intensity(uint32_t i) const {
          return instance[i].intensity;
        }
        void set_intensity(uint32_t i, f32_t intensity) {
          instance[i].intensity = intensity;
        }

        f32_t get_ambient_strength(uint32_t i) const {
          return instance[i].ambient_strength;
        }
        void set_ambient_strength(uint32_t i, f32_t ambient_strength) {
          instance[i].ambient_strength = ambient_strength;
        }

        fan::color get_color(uint32_t i) const {
          return instance[i].color;
        }
        void set_color(uint32_t i, const fan::color& color) {
          instance[i].color = color;
        }

        uint32_t size() const {
          return instance.size();
        }

      protected:

        fan::hector_t<properties_t> instance;
      };
    }
  }
}