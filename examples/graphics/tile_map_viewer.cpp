#include fan_pch

#include _FAN_PATH(graphics/gui/tile_map_editor/loader.h)

int main() {
  loco_t loco;
  loco_t::texturepack_t tp;
  tp.open_compiled("TexturePack");

  ftme_loader_t loader;
  loader.open(&tp);

  auto compiled_map = loader.compile("file.ftme");

  ftme_loader_t::properties_t p;

  p.position = fan::vec3(400, 400, 0);
  //p.size = 0.5;

  auto map_id0_t = loader.add(&compiled_map, p);

  std::vector<fan::graphics::collider_dynamic_t> balls;
  static constexpr int ball_count = 10;
  balls.reserve(ball_count);
  for (int i = 0; i < ball_count; ++i) {
    balls.push_back(fan::graphics::circle_t{{
        .position = fan::vec3(100, 100, i),
        .radius = 30,
        .color = fan::random::color(),
        .blending = true
      }});
    balls.back().set_velocity(fan::random::vec2_direction(-1, 1) * 2);
  }

  loco.loop([&] {
    for (auto& i : balls) {
      i.set_position(i.get_collider_position());
    }
  });
}