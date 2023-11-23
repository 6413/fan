#include fan_pch

struct controls_t {
  bool playing = false;
  bool loop = true;
  f32_t time = 0;
  f32_t max_time = 0;
}controls;

struct key_frame_t {
  f32_t time = 0;
  fan::vec3 position = 0;
  fan::vec2 size = 400;
  f32_t angle = 0;
  fan::vec3 rotation_vector = fan::vec3(0, 0, 1);
};

struct animation_t {
  void push_frame(const key_frame_t& key_frame) {
    key_frames.push_back(key_frame);
    update_key_frame_value();
  }

  void update_key_frame_render_value() {
    if (key_frames.empty()) {
      return;
    }
    if (frame_index >= key_frames.size()) {
      return;
    }
    current_frame = key_frames[frame_index];
    if (current_frame.time == 0) {
      current_frame.time = 1;
    }
  }

  void update_key_frame_value() {
    if (key_frames.empty()) {
      return;
    }
    if (frame_index + 1 >= key_frames.size()) {
      return;
    }
    current_frame = key_frames[frame_index];
    current_frame.time = key_frames[frame_index + 1].time - key_frames[frame_index].time;
    if (current_frame.time == 0) {
      current_frame.time = 1;
    }
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
    current_frame.position = frame_src.position.lerp(frame_dst.position, current_frame.time);
    current_frame.size = frame_src.size.lerp(frame_dst.size, current_frame.time);
    current_frame.angle = fan::math::lerp(frame_src.angle, frame_dst.angle, current_frame.time);
    current_frame.rotation_vector = frame_src.rotation_vector.lerp(frame_dst.rotation_vector, current_frame.time);
  }

  void update(f32_t dt) {
    if (key_frames.empty()) {
      return;
    }
    if (frame_index + 1 >= key_frames.size()) {
      return;
    }
    auto& frame_src = current_frame;
    auto& frame_dst = key_frames[frame_index + 1];
    if (current_frame.time >= 0) {
      if (current_frame.time == 0) {
        current_frame = key_frames[frame_index];
      }
      else {
        current_frame.position += (frame_dst.position - frame_src.position) / current_frame.time * dt;
        current_frame.size += (frame_dst.size - frame_src.size) / current_frame.time * dt;
        current_frame.angle += (frame_dst.angle - frame_src.angle) / current_frame.time * dt;
        current_frame.rotation_vector += (frame_dst.rotation_vector - frame_src.rotation_vector) / current_frame.time * dt;
        current_frame.time -= dt;
      }
    }
  }
  int frame_index = 0;
  std::vector<key_frame_t> key_frames;
  key_frame_t current_frame;
  // can be either image or texturepack image name
  fan::string image_name;
  std::unique_ptr<fan::graphics::vfi_root_t> sprite;
  f32_t time = 0;
};

std::vector<animation_t> objects;

void file_load(const fan::string& path) {
  fan::string istr;
  fan::io::file::read(path, &istr);
  uint32_t off = 0;
  fan::read_from_string(istr, off, controls.loop);
  fan::read_from_string(istr, off, controls.max_time);
  uint32_t obj_size = 0;
  fan::read_from_string(istr, off, obj_size);
  objects.resize(obj_size);
  for (auto& obj : objects) {
    fan::read_from_string(istr, off, obj.image_name);
    uint32_t keyframe_size = 0;
    fan::read_from_string(istr, off, keyframe_size);
    obj.key_frames.resize(keyframe_size);
    int frame_idx = 0;
    for (auto& frame : obj.key_frames) {
      frame = ((key_frame_t*)&istr[off])[frame_idx++];
      //timeline.frames.push_back(frame.time * time_divider);
    }
    memcpy(obj.key_frames.data(), &istr[off], sizeof(key_frame_t) * obj.key_frames.size());
    if (obj.key_frames.size()) {
      /*push_sprite(fan::graphics::sprite_t{ {
        .position = obj.key_frames[0].position,
        .size = obj.key_frames[0].size,
        .angle = obj.key_frames[0].angle,
        .rotation_vector = obj.key_frames[0].rotation_vector
      } });*/
    }
  }
}

int main() {
  loco_t loco;
  fan::vec2 vs = loco.window.get_size();
  loco.default_camera->camera.set_ortho(
    fan::vec2(-vs.x, vs.x),
    fan::vec2(-vs.y, vs.y)
  );

  file_load("0.fka");

  // assuming there is at least 1 obj and 2 keyframes in it

  animation_t& obj = objects[0];

  // initializing with first keyframe
  fan::graphics::sprite_t s{{
    .position = obj.key_frames[0].position,
    .size = obj.key_frames[0].size,
    .angle = obj.key_frames[0].angle,
    .rotation_vector = obj.key_frames[0].rotation_vector
  }};

  obj.current_frame = obj.key_frames[0];

  obj.current_frame.time = obj.key_frames[1].time - obj.key_frames[0].time;

  loco.loop([&] {
    
    if (obj.current_frame.time <= 0) {
      obj.current_frame = obj.key_frames[0];
      obj.current_frame.time = obj.key_frames[1].time - obj.key_frames[0].time;
    }

    obj.update(loco.delta_time);
    key_frame_t kf = obj.current_frame;
    s.set_position(kf.position);
    s.set_size(kf.size);
    s.set_angle(kf.angle);
    s.set_rotation_vector(kf.rotation_vector);

  });
}