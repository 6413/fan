#pragma once

#include _FAN_PATH(types/vector.h)
#include _FAN_PATH(types/memory.h)

namespace fan_2d {
  namespace physics {
    typedef void(*set_position_cb_t)(void* userptr, uint8_t joint_type, uint32_t joint_id, const fan::vec2&);
    typedef void(*set_angle_cb_t)(void* userptr, uint8_t joint_type, uint32_t joint_id, const fan::vec2& p, f32_t angle);

    struct joint_tail_t{

      fan::vec2 m_position;
      fan::vec2 rotation_point;
      uint8_t joint_type;
      uint32_t joint_id;
      fan::hector_t<joint_tail_t> joint_tail;

      void set_position(void* userptr, set_position_cb_t cb, const fan::vec2& position) {
        cb(userptr, joint_type, joint_id, m_position + position);
        for (uint32_t i = 0; i < joint_tail.size(); i++) {
          joint_tail[i].set_position(userptr, cb, m_position + position);
        }
      }

      void set_angle(void* userptr, set_position_cb_t sp_cb, set_angle_cb_t sa_cb, const fan::vec2& position, f32_t angle) {
        fan::vec2 new_position = 0;
        f32_t c = cos(-angle);
        f32_t s = sin(-angle);

        new_position -= m_position + rotation_point;

        fan::vec2 p;
        p.x = new_position.x * c - new_position.y * s;
        p.y = new_position.x * s + new_position.y * c;

        new_position += p + m_position + position + rotation_point;

        sp_cb(userptr, joint_type, joint_id, new_position);
        sa_cb(userptr, joint_type, joint_id, new_position, angle);
        for (uint32_t i = 0; i < joint_tail.size(); i++) {
          joint_tail[i].set_position(userptr, sp_cb, new_position);
          joint_tail[i].set_angle(userptr, sp_cb, sa_cb, new_position, angle);
        }
      }
    };

    struct joint_head_t{

      uint8_t joint_type;
      uint32_t joint_id;
      fan::vec2 rotation_point;
      fan::hector_t<joint_tail_t> joint_tail;

      void set_position(void* userptr, set_position_cb_t set_position_cb, const fan::vec2& position) {
        set_position_cb(userptr, joint_type, joint_id, position);
        for (uint32_t i = 0; i < joint_tail.size(); i++) {
          joint_tail[i].set_position(userptr, set_position_cb, position);
        }
      }
      void set_angle(void* userptr, set_position_cb_t set_position_cb, set_angle_cb_t set_angle_cb, const fan::vec2& position, f32_t angle) {

        fan::vec2 new_position = 0;
        f32_t c = cos(-angle);
        f32_t s = sin(-angle);

        new_position += rotation_point;

        fan::vec2 p;
        p.x = new_position.x * c - new_position.y * s;
        p.y = new_position.x * s + new_position.y * c;

        new_position -= rotation_point;

        new_position += p + position;

        set_angle_cb(userptr, joint_type, joint_id, new_position, angle);

        for (uint32_t i = 0; i < joint_tail.size(); i++) {
          joint_tail[i].set_angle(userptr, set_position_cb, set_angle_cb, new_position, angle);
        }
      }
    };
  }
}