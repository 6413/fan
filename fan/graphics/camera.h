#pragma once

#include <fan/types/matrix.h>

namespace fan {

  struct camera {

    camera();

    fan::mat4 get_view_matrix() const;

    fan::mat4 get_view_matrix(const fan::mat4& m) const;

    fan::vec3 get_front() const;

    void set_front(const fan::vec3 front);


    fan::vec3 get_right() const;

    void set_right(const fan::vec3 right);

    f32_t get_yaw() const;
    void set_yaw(f32_t angle);

    f32_t get_pitch() const;
    void set_pitch(f32_t angle);

    bool first_movement = true;

    void update_view();

    void rotate_camera(fan::vec2 offset);

    f32_t sensitivity = 0.1f;

    static constexpr f32_t max_yaw = 180;
    static constexpr f32_t max_pitch = 89;

    static constexpr f32_t gravity = 500;
    static constexpr f32_t jump_force = 100;

    static constexpr fan::vec3 world_up = fan::vec3(0, 1, 0);

    void move(f32_t movement_speed, f32_t friction = 12);

 // protected:

    f32_t m_yaw = 0;
    f32_t m_pitch = 0;
    fan::vec3 m_right;
    fan::vec3 m_up;
    fan::vec3 m_front;
    fan::vec3 position = 0;
    fan::vec3 velocity = 0;
  };
}