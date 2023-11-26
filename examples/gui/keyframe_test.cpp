#include fan_pch

#include _FAN_PATH(graphics/gui/keyframe_animator/loader.h)

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

    collider.set_velocity(velocity);
    player.animation.set_position(0, player.collider.get_collider_position());
  }

  fan::ev_timer_t::timer_t timer;
  bool jumping = false;

  fan::vec2 velocity = 0;
  fan::graphics::collider_dynamic_hidden_t collider;
  fan::graphics::animation_t animation;
}player;

int main() {

  fan::vec2 vs = loco.window.get_size();
  loco.default_camera->camera.set_ortho(
    fan::vec2(-vs.x, vs.x),
    fan::vec2(-vs.y, vs.y)
  );

  loco_t::texturepack_t texturepack;
  texturepack.open_compiled("texture_packs/tilemap.ftp");
  player.animation = fan::graphics::animation_t(&texturepack);
  player.animation.file_load("keyframe0.fka");
  player.animation.set_origin();

  player.collider = fan::graphics::collider_dynamic_hidden_t(
    player.animation.objects[0].key_frames[0].position,
    player.animation.objects[0].key_frames[0].size
  );


  // initializing with first keyframe

  player.animation.objects[0].current_frame = player.animation.objects[0].key_frames[0];

  //player.animation.load_image(player.visual, player.animation.image_name, texturepack);

  player.animation.controls.loop = false;

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
        player.animation.play_from_begin();
        player.timer.cb = [&](const fan::ev_timer_t::cb_data_t&) {
          player.animation.play_animation();
          if (!player.animation.is_finished()) {
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
  loco.set_vsync(true);
  loco.loop([&] {
    loco.get_fps();
    player.update();
  });
}