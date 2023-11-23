#pragma once

struct key_frame_t {
  f32_t time = 0;
  fan::vec3 position = 0;
  fan::vec2 size = 400;
  f32_t angle = 0;
  fan::vec3 rotation_vector = fan::vec3(0, 0, 1);
};

struct controls_t {
  bool playing = false;
  bool loop = true;
  f32_t time = 0;
  f32_t max_time = 0;
}inline controls;

struct animation_t {
  void push_frame(const key_frame_t& key_frame) {
    key_frames.push_back(key_frame);
  }

  void update_seek() {
    if (key_frames.empty()) {
      return;
    }
    if (frame_index + 1 >= key_frames.size()) {
      return;
    }
    auto& frame_src = key_frames[frame_index];
    auto& frame_dst = key_frames[frame_index + 1];
    // assuming controls.time is between src and dst
    f32_t offset = fan::math::normalize(controls.time, frame_src.time, frame_dst.time);
    current_frame.position = frame_src.position.lerp(frame_dst.position, offset);
    current_frame.size = frame_src.size.lerp(frame_dst.size, offset);
    current_frame.angle = fan::math::lerp(frame_src.angle, frame_dst.angle, offset);
    current_frame.rotation_vector = frame_src.rotation_vector.lerp(frame_dst.rotation_vector, offset);
  }

  bool update() {
    if (key_frames.empty()) {
      return 0;
    }
    if (frame_index + 1 >= key_frames.size()) {
      return 0;
    }
    if (controls.time < key_frames[frame_index].time) {
      return 0;
    }
    auto& frame_src = key_frames[frame_index];
    auto& frame_dst = key_frames[frame_index + 1];
    if (controls.time < frame_dst.time) {
      f32_t offset = fan::math::normalize(controls.time, frame_src.time, frame_dst.time);
      current_frame.position = frame_src.position.lerp(frame_dst.position, offset);
      current_frame.size = frame_src.size.lerp(frame_dst.size, offset);
      current_frame.angle = fan::math::lerp(frame_src.angle, frame_dst.angle, offset);
      current_frame.rotation_vector = frame_src.rotation_vector.lerp(frame_dst.rotation_vector, offset);
    }
    else {
      frame_index++;
      current_frame = key_frames[frame_index];
    }
    return 1;
  }
  int frame_index = 0;
  std::vector<key_frame_t> key_frames;
  key_frame_t current_frame;
  // can be either image or texturepack image name
  fan::string image_name;
  std::unique_ptr<fan::graphics::vfi_root_t> sprite;
  f32_t time = 0;
};

inline std::vector<animation_t> objects;

void play_from_begin() {
  controls.time = 0;
  for (auto& obj : objects) {
    obj.frame_index = 0;
  }
  controls.playing = true;
}

void load_image(loco_t::shape_t& shape, const fan::string& name, loco_t::image_t& image) {
  if (image.load(name)) {
    fan::print_warning("failed to load image:" + name);
  }
  else {
    shape.set_image(&image);
  }
}

void load_image(loco_t::shape_t& shape, const fan::string& name, loco_t::texturepack_t& texturepack) {
  loco_t::texturepack_t::ti_t ti;
  if (texturepack.qti(name, &ti)) {
    fan::print_warning("failed to load texturepack image:" + name);
  }
  else {
    shape.set_tp(&ti);
  }
}

bool is_finished() {
  return controls.max_time <= controls.time;
}

static void play_animation(const fan::vec3& origin, controls_t& controls, animation_t& animation, loco_t::shape_t& shape) {
  if (controls.loop && is_finished) {
    play_from_begin();
  }

  animation.update();
  key_frame_t kf = animation.current_frame;
  shape.set_position(origin + kf.position);
  shape.set_size(kf.size);
  shape.set_angle(kf.angle);
  shape.set_rotation_vector(kf.rotation_vector);
  controls.time += gloco->delta_time;
}