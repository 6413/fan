#pragma once

#include <fan/types/vector.h>

namespace fan_2d {
  namespace graphics {
    namespace lighting {
      struct light_t {

        enum class light_e{
          light_added,
          light_removed
        };

        enum class light_update_e {
          position,
          size,
          radius,
          intensity,
          ambient_strength,
          color,
          rotation_point,
          angle
        };

        typedef void(*light_cb_t)(light_t*, light_e, uint32_t node_reference);
        typedef void(*light_update_cb_t)(light_t*, light_update_e, uint32_t node_reference);

        void open() {
          set_light_cb([](light_t*, light_e, uint32_t) {});
          set_light_update_cb([](light_t*, light_update_e, uint32_t) {});
          m_instance.open();
        }
        void close() {
          m_instance.close();
        }

        void set_light_cb(light_cb_t light_cb)
        {
          m_light_cb = light_cb;
        }
        void set_light_update_cb(light_update_cb_t light_update_cb) {
          m_light_update_cb = light_update_cb;
        }

        struct properties_t {
          fan::vec2 position = 0;
          f32_t radius = 100;
          f32_t intensity = 1;
          f32_t ambient_strength = 0.1;
          fan::color color = fan::colors::white;
          fan::vec2 rotation_point = fan::vec2(0, 0);
          f32_t angle = 0;
          uint8_t type = 0;
        };

        uint32_t push_back(const properties_t& p) {
          uint32_t node_reference = m_instance.push_back(p);
          m_light_cb(this, light_e::light_added, node_reference);
          return node_reference;
        }
        void erase(uint32_t node_reference) {
          m_light_cb(this, light_e::light_removed, node_reference);
          m_instance.erase(node_reference);
        }

        fan::vec2 get_position(uint32_t i) const {
          return m_instance[i].position;
        }
        void set_position(uint32_t i, const fan::vec2& position) {
          m_instance[i].position = position;
          m_light_update_cb(this, light_update_e::position, i);
        }

        f32_t get_radius(uint32_t i) const {
          return m_instance[i].radius;
        }
        void set_radius(uint32_t i, f32_t radius) {
          m_instance[i].radius = radius;
          m_light_update_cb(this, light_update_e::radius, i);
        }

        f32_t get_intensity(uint32_t i) const {
          return m_instance[i].intensity;
        }
        void set_intensity(uint32_t i, f32_t intensity) {
          m_instance[i].intensity = intensity;
          m_light_update_cb(this, light_update_e::intensity, i);
        }

        f32_t get_ambient_strength(uint32_t i) const {
          return m_instance[i].ambient_strength;
        }
        void set_ambient_strength(uint32_t i, f32_t ambient_strength) {
          m_instance[i].ambient_strength = ambient_strength;
          m_light_update_cb(this, light_update_e::ambient_strength, i);
        }

        fan::vec3 get_color(uint32_t i) const {
          return fan::vec3(m_instance[i].color.r, m_instance[i].color.g, m_instance[i].color.b);
        }
        void set_color(uint32_t i, const fan::color& color) {
          m_instance[i].color = color;
          m_light_update_cb(this, light_update_e::color, i);
        }

        fan::vec2 get_rotation_point(uint32_t i) const {
          return m_instance[i].rotation_point;
        }
        void set_rotation_point(uint32_t i, const fan::vec2& rotation_point) {
          m_instance[i].rotation_point = rotation_point;
          m_light_update_cb(this, light_update_e::rotation_point, i);
        }

        f32_t get_angle(uint32_t i) const {
          return m_instance[i].angle;
        }
        void set_angle(uint32_t i, f32_t angle) {
          m_instance[i].angle = angle;
          m_light_update_cb(this, light_update_e::angle, i);
        }

        uint8_t get_type(uint32_t i) const {
          return m_instance[i].type;
        }
        void set_type(uint32_t i, uint8_t type) {
          m_instance[i].type = type;
        }

        uint32_t size() const {
          return m_instance.size();
        }

        bll_t<properties_t> m_instance;

      protected:

        light_cb_t m_light_cb;
        light_update_cb_t m_light_update_cb;
      };
    }
  }
}