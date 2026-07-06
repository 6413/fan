module fan.camera;

import std;

fan::camera::camera() {
  update_view();
}
fan::mat4 fan::camera::get_view_matrix() const {
  return fan::math::look_at_right<fan::mat4, fan::vec3>(
    fan::vec3(position),
    position + front,
    up
  );
}
fan::mat4 fan::camera::get_view_matrix(const fan::mat4& m) const {
  return m * fan::math::look_at_right<fan::mat4, fan::vec3>(
    fan::vec3(position),
    position + front,
    world_up
  );
}
fan::vec3 fan::camera::get_front() const {
  return front;
}
void fan::camera::set_front(const fan::vec3 front) {
  this->front = front;
}
fan::vec3 fan::camera::get_right() const {
  return right;
}
void fan::camera::set_right(const fan::vec3 right) {
  this->right = right;
}
f32_t fan::camera::get_yaw() const {
  return yaw;
}
void fan::camera::set_yaw(f32_t angle) {
  yaw = angle;
  if (yaw > max_yaw) {
    yaw = -max_yaw;
  }
  if (yaw < -max_yaw) {
    yaw = max_yaw;
  }
}
f32_t fan::camera::get_pitch() const {
  return pitch;
}
void fan::camera::set_pitch(f32_t angle) {
  pitch = angle;
  if (pitch > max_pitch) {
    pitch = max_pitch;
  }
  if (pitch < -max_pitch) {
    pitch = -max_pitch;
  }
}
void fan::camera::update_view() {
  front = (fan::math::direction_vector<fan::vec3>(yaw, pitch)).normalize();
  front.z *= -1.f;
  right = (fan::math::cross(world_up, front)).normalize();
  up = (fan::math::cross(front, right)).normalize();
}
void fan::camera::rotate_camera(fan::vec2 offset) {
  offset *= sensitivity;
  set_yaw(get_yaw() + offset.x);
  set_pitch(get_pitch() - offset.y);
  update_view();
}