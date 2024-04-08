#include "camera.h"

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

//#define loco_imgui

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif

#ifndef FAN_INCLUDE_PATH
#define _FAN_PATH(p0) <fan/p0>
#else
#define FAN_INCLUDE_PATH_END fan/
#define _FAN_PATH(p0) <FAN_INCLUDE_PATH/fan/p0>
#define _FAN_PATH_QUOTE(p0) STRINGIFY_DEFINE(FAN_INCLUDE_PATH) "/fan/" STRINGIFY(p0)
#endif

#if defined(loco_imgui)
#define IMGUI_IMPL_OPENGL_LOADER_CUSTOM
#define IMGUI_DEFINE_MATH_OPERATORS
#include _FAN_PATH(imgui/imgui.h)
#include _FAN_PATH(imgui/imgui_impl_opengl3.h)
#include _FAN_PATH(imgui/imgui_impl_glfw.h)
#include _FAN_PATH(imgui/imgui_neo_sequencer.h)
#endif

#ifndef fan_verbose_print_level
#define fan_verbose_print_level 1
#endif
#ifndef fan_debug
#define fan_debug 0
#endif
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH,fan/types/types.h)

#include <fan/graphics/loco_settings.h>
#include <fan/graphics/loco.h>

fan::camera::camera() {
  m_yaw = 0;
  m_pitch = 0;
  m_right = 0;
  m_up = 0;
  m_front = 0;
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

void fan::camera::move(f32_t movement_speed, f32_t friction) {
  this->velocity /= friction * gloco->delta_time + 1;
  static constexpr auto minimum_velocity = 0.001;
  if (this->velocity.x < minimum_velocity && this->velocity.x > -minimum_velocity) {
    this->velocity.x = 0;
  }
  if (this->velocity.y < minimum_velocity && this->velocity.y > -minimum_velocity) {
    this->velocity.y = 0;
  }
  if (this->velocity.z < minimum_velocity && this->velocity.z > -minimum_velocity) {
    this->velocity.z = 0;
  }
  if (gloco->window.key_pressed(fan::input::key_w)) {
    this->velocity += this->m_front * (movement_speed * gloco->delta_time);
  }
  if (gloco->window.key_pressed(fan::input::key_s)) {
    this->velocity -= this->m_front * (movement_speed * gloco->delta_time);
  }
  if (gloco->window.key_pressed(fan::input::key_a)) {
    this->velocity -= this->m_right * (movement_speed * gloco->delta_time);
  }
  if (gloco->window.key_pressed(fan::input::key_d)) {
    this->velocity += this->m_right * (movement_speed * gloco->delta_time);
  }

  if (gloco->window.key_pressed(fan::input::key_space)) {
    this->velocity.y += movement_speed * gloco->delta_time;
  }
  if (gloco->window.key_pressed(fan::input::key_left_shift)) {
    this->velocity.y -= movement_speed * gloco->delta_time;
  }

  if (gloco->window.key_pressed(fan::input::key_left)) {
    this->set_yaw(this->get_yaw() - sensitivity * 100 * gloco->delta_time);
  }
  if (gloco->window.key_pressed(fan::input::key_right)) {
    this->set_yaw(this->get_yaw() + sensitivity * 100 * gloco->delta_time);
  }
  if (gloco->window.key_pressed(fan::input::key_up)) {
    this->set_pitch(this->get_pitch() + sensitivity * 100 * gloco->delta_time);
  }
  if (gloco->window.key_pressed(fan::input::key_down)) {
    this->set_pitch(this->get_pitch() - sensitivity * 100 * gloco->delta_time);
  }

  this->position += this->velocity * gloco->delta_time;
  this->update_view();
}