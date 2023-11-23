#include fan_pch

#include _FAN_PATH(graphics/gui/keyframe_animator/editor.h)

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

loco_t loco;

struct player_t {

  static constexpr fan::vec2 speed{ 200, 200 };

  void update() {
    f32_t dt = gloco->get_delta_time();
    f32_t multiplier = 1;
    if (gloco->window.key_pressed(fan::key_shift)) {
      multiplier = 3;
    }
    if (gloco->window.key_pressed(fan::key_d)) {
      velocity.x = speed.x * multiplier;
    }
    else if (gloco->window.key_pressed(fan::key_a)) {
      velocity.x = -speed.x * multiplier;
    }
    else {
      velocity.x = 0;
    }

    if (gloco->window.key_pressed(fan::key_w)) {
      velocity.y = -speed.y * multiplier;
    }
    else if (gloco->window.key_pressed(fan::key_s)) {
      velocity.y = speed.y * multiplier;
    }
    else {
      velocity.y = 0;
    }

    visual.set_velocity(velocity);
    visual.set_position(visual.get_collider_position());
  }

  fan::ev_timer_t::timer_t timer;
  bool jumping = false;

  fan::vec2 velocity = 0;
  fan::graphics::collider_dynamic_t visual;
}player;

int main() {

  fan::vec2 vs = loco.window.get_size();
  loco.default_camera->camera.set_ortho(
    fan::vec2(-vs.x, vs.x),
    fan::vec2(-vs.y, vs.y)
  );

  loco_t::texturepack_t texturepack;
  texturepack.open_compiled("texture_packs/tilemap.ftp");

  file_load("keyframe0.fka");

  // assuming there is at least 1 obj and 2 keyframes in it

  animation_t& obj = objects[0];

  // set to origin
  fan::vec2 off = -obj.key_frames[0].position;
  for (auto& i : obj.key_frames) {
    i.position += off;
  }

  // initializing with first keyframe
  player.visual = fan::graphics::sprite_t{{
    .position = obj.key_frames[0].position,
    .size = obj.key_frames[0].size,
    .angle = obj.key_frames[0].angle,
    .rotation_vector = obj.key_frames[0].rotation_vector
  }};

  obj.current_frame = obj.key_frames[0];

  load_image(player.visual, obj.image_name, texturepack);

  controls.loop = false;

  loco.window.add_keys_callback([&](const auto& d) {
    if (d.state != fan::keyboard_state::press) {
      return;
    }
    switch (d.key) {
      case fan::key_space: {
        if (player.jumping) {
          break;
        }
        player.jumping = true;
        play_from_begin();
        player.timer.cb = [&](const fan::ev_timer_t::cb_data_t&) {
          play_animation(player.visual.get_collider_position(), controls, obj, player.visual);
          if (!is_finished()) {
            gloco->ev_timer.start(&player.timer, 0);
          }
          else {
            player.jumping = false;
          }
        };
        loco.ev_timer.start(&player.timer, 0);
        break;
      }
    }
  });

  loco.loop([&] {
    player.update();
  });
}