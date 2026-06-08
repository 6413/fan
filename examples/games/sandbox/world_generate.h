#include "crack_shader.h"

enum tile_type_t {
  tile_empty,
  tile_grass,
  tile_dirt,
  tile_stone,
};

struct cave_config_t {
  fan::vec2i map_tiles = fan::vec2 {8400, 2400} / 14.f;
  f32_t tile_size = 16.f;
  int iterations = 3;
  int neighbor_threshold = 10;
  fan::color empty_color = fan::colors::transparent;
  fan::color dirt_color = fan::color(0.5, 0.3, 0.1);
  fan::color stone_color = fan::color(0.3, 0.3, 0.3);
};

bool is_air(int x, int y) {
  if (x < 0 || y < 0 || x >= world.map_size.x || y >= world.map_size.y) {
    return false;
  }
  return !world.tiles[x + y * world.map_size.x];
}
bool is_empty(int x, int y) {
  if (x < 0 || y < 0 || x >= world.map_size.x || y >= world.map_size.y) {
    return true;
  }
  return !world.tiles[x + y * world.map_size.x];
}
void spawn_grass_leaf(int x, int y, f32_t ts) {
  fan::vec2 tile_pos = fan::vec2(x + fan::random::value(-ts / 2.f, ts / 2.f), y) * ts;
  fan::vec2 leaf_size = fan::vec2(ts) / fan::random::value(1.f, 2.f);
  fan::vec2 leaf_pos = tile_pos + fan::vec2(ts * 0.5f, ts - leaf_size.y / 1.1f);
  grass_leaves.push_back(fan::graphics::shader_shape_t {{
    .position = fan::vec3(leaf_pos, 0),
    .size = leaf_size,
    .rotation_point = fan::vec2(0, leaf_size.y),
    .shader = grass_shader,
    .image = fan::graphics::image_t("images/plant_03.png", fan::graphics::image_presets::smooth()),
    .enable_culling = false
  }});
}
void handle_grass_tile(auto& shape, int x, int y) {
  bool u = is_empty(x, y - 1), d = is_empty(x, y + 1);
  bool l = is_empty(x - 1, y), r = is_empty(x + 1, y);

  const char* img = "images/Grass0.webp";
  f32_t angle = 0;

  if (u && d && l && r) { img = "images/grass_corner_full.webp"; }
  else if (l && u && r) { img = "images/grass_corner_double.webp"; }
  else if (u && r && d) { img = "images/grass_corner_double.webp"; angle = 0.5f; }
  else if (r && d && l) { img = "images/grass_corner_double.webp"; angle = 1.f; }
  else if (d && l && u) { img = "images/grass_corner_double.webp"; angle = 1.5f; }
  else if (u && r) { img = "images/grass_corner.webp"; }
  else if (r && d) { img = "images/grass_corner.webp"; angle = 0.5f; }
  else if (d && l) { img = "images/grass_corner.webp"; angle = 1.f; }
  else if (l && u) { img = "images/grass_corner.webp"; angle = 1.5f; }
  else if (r) { angle = 0.5f; }
  else if (d) { angle = 1.f; }
  else if (l) { angle = 1.5f; }

  shape.set_image(img);
  shape.set_angle(fan::vec3(0, 0, fan::math::pi * angle));
}

fan::graphics::tilemap_t tilemap;
fan::graphics::tile_world_generator_t world;
std::vector<fan::physics::entity_t> collisions;
std::vector<fan::graphics::shader_shape_t> grass_leaves;
fan::graphics::render_view_t view;
fan::graphics::shader_t crack_shader;
fan::graphics::shader_t grass_shader;
cave_config_t cfg;