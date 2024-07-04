#include <pch.h>
#include "camera.h"

#include <fan/window/window_input.h>

fan::camera::camera() {
  this->update_view();
}

fan::mat4 fan::camera::get_view_matrix() const {
  return fan::math::look_at_left<fan::mat4, fan::vec3>(fan::vec3(position), position + m_front, this->m_up);
}

fan::mat4 fan::camera::get_view_matrix(const fan::mat4& m) const {
  return m * fan::math::look_at_left<fan::mat4, fan::vec3>(fan::vec3(position), position + m_front, this->world_up);
}

fan::vec3 fan::camera::get_front() const {
  return this->m_front;
}

void fan::camera::set_front(const fan::vec3 front) {
  this->m_front = front;
}

fan::vec3 fan::camera::get_right() const {
  return m_right;
}

void fan::camera::set_right(const fan::vec3 right) {
  m_right = right;
}

f32_t fan::camera::get_yaw() const {
  return this->m_yaw;
}

void fan::camera::set_yaw(f32_t angle) {
  this->m_yaw = angle;
  if (m_yaw > fan::camera::max_yaw) {
    m_yaw = -fan::camera::max_yaw;
  }
  if (m_yaw < -fan::camera::max_yaw) {
    m_yaw = fan::camera::max_yaw;
  }
}

f32_t fan::camera::get_pitch() const {
  return this->m_pitch;
}

void fan::camera::set_pitch(f32_t angle) {
  this->m_pitch = angle;
  if (this->m_pitch > fan::camera::max_pitch) {
    this->m_pitch = fan::camera::max_pitch;
  }
  if (this->m_pitch < -fan::camera::max_pitch) {
    this->m_pitch = -fan::camera::max_pitch;
  }
}

void fan::camera::update_view() {
  this->m_front = fan_3d::math::normalize(fan::math::direction_vector<fan::vec3>(this->m_yaw, this->m_pitch));
  this->m_right = fan_3d::math::normalize(fan::math::cross(this->world_up, this->m_front));
  this->m_up = fan_3d::math::normalize(fan::math::cross(this->m_front, this->m_right));
}

void fan::camera::rotate_camera(fan::vec2 offset) {
  offset *= sensitivity;

  this->set_yaw(this->get_yaw() + offset.x);
  this->set_pitch(this->get_pitch() - offset.y);

  this->update_view();
}