#pragma once

#include _FAN_PATH(types/matrix.h)

namespace fan {

  class camera {
  public:

    camera() {
      m_yaw = 0;
      m_pitch = 0;
      m_right = 0;
      m_up = 0;
      m_front = 0;
      this->update_view();
    }

    //void rotate_camera(bool when);

    fan::mat4 get_view_matrix() const {
      return fan::math::look_at_left<fan::mat4, fan::vec3>(fan::vec3(position), position + m_front, this->m_up);
    }

    fan::mat4 get_view_matrix(const fan::mat4& m) const {
      return m * fan::math::look_at_left<fan::mat4, fan::vec3>(fan::vec3(position), position + m_front, this->world_up);
    }

    fan::vec3 get_front() const {
      return this->m_front;
    }

    void set_front(const fan::vec3 front) {
      this->m_front = front;
    }


    fan::vec3 get_right() const {
      return m_right;
    }

    void set_right(const fan::vec3 right) {
      m_right = right;
    }

    f32_t get_yaw() const {
      return this->m_yaw;
    }
    void set_yaw(f32_t angle) {
      this->m_yaw = angle;
      if (m_yaw > fan::camera::max_yaw) {
        m_yaw = -fan::camera::max_yaw;
      }
      if (m_yaw < -fan::camera::max_yaw) {
        m_yaw = fan::camera::max_yaw;
      }
    }

    f32_t get_pitch() const {
      return this->m_pitch;
    }
    void set_pitch(f32_t angle) {
      this->m_pitch = angle;
      if (this->m_pitch > fan::camera::max_pitch) {
        this->m_pitch = fan::camera::max_pitch;
      }
      if (this->m_pitch < -fan::camera::max_pitch) {
        this->m_pitch = -fan::camera::max_pitch;
      }
    }

    bool first_movement = true;

    void update_view() {
      this->m_front = fan_3d::math::normalize(fan::math::direction_vector<fan::vec3>(this->m_yaw, this->m_pitch));
      this->m_right = fan_3d::math::normalize(fan::math::cross(this->world_up, this->m_front));
      this->m_up = fan_3d::math::normalize(fan::math::cross(this->m_front, this->m_right));
    }

    void rotate_camera(fan::vec2 offset) {
      offset *= sensitivity;

      this->set_yaw(this->get_yaw() + offset.x);
      this->set_pitch(this->get_pitch() - offset.y);

      this->update_view();
    }

    f32_t sensitivity = 0.1;

    static constexpr f32_t max_yaw = 180;
    static constexpr f32_t max_pitch = 89;

    static constexpr f32_t gravity = 500;
    static constexpr f32_t jump_force = 100;

    static constexpr fan::vec3 world_up = fan::vec3(0, 1, 0);

    void move(f32_t movement_speed, f32_t friction = 12);

 // protected:

    f32_t m_yaw;
    f32_t m_pitch;
    fan::vec3 m_right;
    fan::vec3 m_up;
    fan::vec3 m_front;
    fan::vec3 position = 0;
    fan::vec3 velocity = 0;
  };
}