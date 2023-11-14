#include fan_pch

#include _FAN_PATH(graphics/gui/tilemap_editor/loader.h)

loco_t::image_t ismot[4];

struct player_t {
  static constexpr fan::vec2 speed{200, 200};

  player_t() {
    visual = fan::graphics::sprite_t{{
      .position = fan::vec3(100, 100, 10),
      .size = 32,
      .blending = true
    }};
  }
  void update() {
    auto& window = *gloco->get_window();

    f32_t dt = gloco->get_delta_time();
    if (window.key_pressed(fan::key_d)) {
      velocity.x = speed.x;
      visual.set_image(&ismot[0]);
    }
    else if (window.key_pressed(fan::key_a)) {
      velocity.x = -speed.x;
      visual.set_image(&ismot[1]);
    }
    else {
      velocity.x = 0;
    }

    if (window.key_pressed(fan::key_w)) {
      velocity.y = -speed.y;
      visual.set_image(&ismot[3]);
    }
    else if (window.key_pressed(fan::key_s)) {
      velocity.y = speed.y;
      visual.set_image(&ismot[2]);
    }
    else {
      velocity.y = 0;
    }

    visual.set_velocity(velocity);
    visual.set_position(visual.get_collider_position());
  }
  fan::vec2 velocity = 0;
  fan::graphics::collider_dynamic_t visual;
};

f32_t zoom = 2;
void init_zoom() {
  auto& window = *gloco->get_window();
  auto update_ortho = []{
    fan::vec2 s = gloco->get_window()->get_size();
    gloco->default_camera->camera.set_ortho(
      fan::vec2(-s.x, s.x) / zoom,
      fan::vec2(-s.y, s.y) / zoom
    );;
  };

  update_ortho();

  window.add_buttons_callback([&](const auto&d) {
    if (d.button == fan::mouse_scroll_up) {
      zoom *= 1.2;
    }
    else if (d.button == fan::mouse_scroll_down) {
      zoom /= 1.2;
    }
    update_ortho();
  });
}

int main() {
  loco_t loco;
  loco_t::texturepack_t tp;
  tp.open_compiled("tile_map");

  fte_loader_t loader;
  loader.open(&tp);

  auto compiled_map = loader.compile("file.fte");

  fte_loader_t::properties_t p;

  p.position = fan::vec3(0, 0, 0);
  auto map_id0_t = loader.add(&compiled_map, p);

  init_zoom();

  player_t player;


  loco.loop([&] {
    gloco->get_fps();
    player.update();
    gloco->default_camera->camera.set_position(player.visual.get_position());
  });
}