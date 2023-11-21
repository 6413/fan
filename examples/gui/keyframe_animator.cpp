#include fan_pch

struct key_frame_t {
  f32_t time = 0;
  fan::vec3 position = 0;
  fan::vec2 size = 400;
  f32_t angle = 0;
  fan::vec3 rotation_vector = fan::vec3(0, 0, 1);
};

// per object
struct animation_t {
  void push_frame(const key_frame_t& key_frame) {
    key_frames.push_back(key_frame);
    update_key_frame_value();
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
      current_frame.position += (frame_dst.position - frame_src.position) / current_frame.time * dt;
      current_frame.size += (frame_dst.size - frame_src.size) / current_frame.time * dt;
      current_frame.angle += (frame_dst.angle - frame_src.angle) / current_frame.time * dt;
      current_frame.rotation_vector += (frame_dst.rotation_vector - frame_src.rotation_vector) / current_frame.time * dt;
      current_frame.time -= dt;
    }
    else {
      current_frame = key_frames[frame_index + 1];
      frame_index++;
      update_key_frame_value();
    }
  }
  int frame_index = 0;
  std::vector<key_frame_t> key_frames;
  key_frame_t current_frame;
};

std::vector<animation_t> objects;

void handle_imgui() {
  ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());

  ImGui::Begin("Editor");

  ImGui::End();

  ImGui::Begin("Key frames");
  if (ImGui::Button("+", ImVec2(50, 25))) {
    objects.resize(objects.size() + 1);
  }
  ImGui::End();

  ImGui::Begin("Controls");

  ImGui::End();
}

int main() {
  loco_t loco;

  fan::graphics::imgui_element_t main_view =
    fan::graphics::imgui_element_t([&] {handle_imgui(); });

  animation_t animation;
  key_frame_t start;
  start.time = 0;
  start.position = fan::vec2(0, 200);
  start.size = 50;
  animation.push_frame(start);
  key_frame_t end;
  end.angle = fan::math::pi * 2;
  end.time = 2;
  end.size = 50;
  end.position = fan::vec2(800, 600);
  animation.push_frame(end);

  key_frame_t end2;
  end2.angle = -fan::math::pi * 2;
  end2.time = 4;
  end2.size = 25;
  end2.position = fan::vec2(800, 200);
  animation.push_frame(end2);

  loco_t::image_t image;
  image.load("images/hi.webp");

  loco_t::shape_t sprite = fan::graphics::sprite_t{ {
    .position = fan::vec3(50, 50, 0),
    .size = 50,
    .image = &image
  }};

  loco.loop([&] {
    animation.update(loco.delta_time);
    key_frame_t kf = animation.current_frame;
    sprite.set_position(kf.position);
    sprite.set_size(kf.size);
    sprite.set_angle(kf.angle);
    sprite.set_rotation_vector(kf.rotation_vector);
  });
}