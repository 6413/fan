#include "crack_shader.h"

inline static const char* grass_shader_fragment = R"(#version 330

layout (location = 0) out vec4 o_attachment0;

in vec2 texture_coordinate;
in vec4 instance_color;
flat in float object_seed;

uniform sampler2D _t00;
uniform float grass_time;
uniform float grass_wind;

void main() {
  vec2 uv = texture_coordinate;
  float sway_factor = 1.0 - uv.y;
  float phase = object_seed * 10.0;
  float amp = mix(0.8, 1.2, fract(sin(object_seed * 123.456) * 999.0));
  float offset =
    sin(grass_time * 6.0 + uv.y * 12.0 + phase)
    * sway_factor
    * 0.05
    * grass_wind
    * amp;

  uv.x += offset;

  vec4 c = texture(_t00, uv) * instance_color;

  if (c.a < 0.1) {
    discard;
  }

  o_attachment0 = c;
}
)";

enum tile_type_t {
  tile_empty,
  tile_grass,
  tile_dirt,
  tile_stone,
};

struct cave_config_t {
  fan::vec2i map_tiles = fan::vec2{8400, 2400} / 14.f;
  f32_t tile_size = 64.f;
  int iterations = 3;
  int neighbor_threshold = 10;
  fan::color empty_color = fan::colors::transparent;
  fan::color dirt_color = fan::color(0.5, 0.3, 0.1);
  fan::color stone_color = fan::color(0.3, 0.3, 0.3);
};


void break_block(int x, int y) {
  int idx = x + y * world.map_size.x;
  if (!world.tiles[idx]) {
    return;
  }
  world.tiles[idx] = false;
  tilemap.set_tile_color({x, y}, cfg.empty_color);
  collisions[idx].destroy();
  //auto& shape = tilemap.get_tile({x, y});
  //fan::graphics::physics::sprite_t::properties_t props;
  //props.position = shape.get_position().offset_z(1);
  //props.size = shape.get_size() / 4.f;
  //props.image = shape.get_image();
  //props.body_type = fan::physics::body_type_e::dynamic_body;

  //pickupables.push_back(props);
}
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
  fan::vec2 tile_pos = fan::vec2(x + fan::random::value(-ts/2.f, ts/2.f), y) * ts;
  fan::vec2 leaf_size = fan::vec2(ts) / fan::random::value(4.f, 8.f);
  fan::vec2 leaf_pos = tile_pos + fan::vec2(ts * 0.5f, ts - leaf_size.y / 1.1f);
  grass_leaves.push_back(fan::graphics::shader_shape_t{{
    .position = fan::vec3(leaf_pos, 0),
    .size = leaf_size,
    .rotation_point = fan::vec2(0, leaf_size.y),
    .shader = grass_shader,
    .image = fan::graphics::image_t("images/grass_leaves.webp", fan::graphics::image_presets::smooth()),
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
void open(void* sod) {
  if (ready) {
    return;
  }

  f32_t ts = cfg.tile_size;

  tilemap = fan::graphics::tilemap_t(fan::vec2(ts), fan::colors::gray, fan::vec2(cfg.map_tiles) * ts, fan::vec3(0, 0, 100));
  crack_shader = fan::graphics::shader_t{gloco()->get_sprite_vertex_shader(crack_shader_fragment)};
  grass_shader = fan::graphics::shader_t{gloco()->get_sprite_vertex_shader(grass_shader_fragment)};

  world.initial_fill = 0.54f;
  world.map_size = tilemap.get_cell_count();
  world.init();

  for (int i = 0; i < cfg.iterations; i++) {
    world.iterate();
  }

  fan::vec2i cc = world.map_size;
  collisions.resize(cc.x * cc.y);
  grass_leaves.clear();

  for (int x = 0; x < cc.x; x++) {
    bool surface_leaf_spawned = false;
    for (int y = 0; y < cc.y; y++) {
      int idx = x + y * cc.x;
      bool solid = world.tiles[idx];
      bool air_up = is_air(x, y - 1), air_down = is_air(x, y + 1);
      bool air_left = is_air(x - 1, y), air_right = is_air(x + 1, y);

      int type = tile_empty;
      if (solid) {
        if (y == 0) {
          type = tile_grass;
        }
        else if (air_up || air_down || air_left || air_right) {
          type = tile_grass;
        }
        else if (world.count_neighbors(x, y) >= cfg.neighbor_threshold) {
          type = tile_dirt;
        }
        else {
          type = tile_stone;
        }
      }

      if (!surface_leaf_spawned && type == tile_grass && y == 0) {
        for (int gc = 0; gc < (fan::random::value(0, 4) == 0 ? fan::random::value(1, 4) : 0); ++gc) {
          spawn_grass_leaf(x, -1, ts);
          surface_leaf_spawned = true;
        }
      }

      auto& shape = tilemap.get_tile({x, y});
      shape.set_color(fan::colors::white);

      switch (type) {
      case tile_empty:
        tilemap.set_tile_color({x, y}, cfg.empty_color);
        break;
      case tile_grass:
        handle_grass_tile(shape, x, y);
        break;
      case tile_dirt:
        shape.set_image("images/dirt" + std::to_string(fan::random::value_i64(0, 3)) + ".webp");
        break;
      case tile_stone:
        shape.set_image("images/stone.png");
        break;
      }

      if (type != tile_empty) {
        collisions[idx] = gloco()->physics_context.create_rectangle(fan::vec2(x, y) * ts + ts / 2.f, ts / 2.f);
      }
    }
  }

  //crack_overlay = fan::graphics::shader_shape_t{{
  //  .render_view = &view,
  //  .position = fan::vec3(0, 0, 10000),
  //  .size = fan::vec2(ts) / 2.f,
  //  .shader = crack_shader,
  //  .image = pile.engine.default_texture,
  //  .enable_culling = false
  //}};
  ready = true;
}

void update_grass() {
  int seed = 0;
  for (auto& leaf : grass_leaves) {

    float sway = sin(grass_time * 7.f + seed++);

    float max_angle = 0.05f;
    float angle_z = sway * max_angle * grass_wind;

    leaf.set_angle({0, 0, angle_z});
  }

  grass_time += gloco()->delta_time;
  gloco()->shader_set_value(grass_shader, "grass_time", grass_time);
  gloco()->shader_set_value(grass_shader, "grass_wind", grass_wind);
}

void update_crack() {
  f32_t ts = cfg.tile_size;

  bool held = gloco()->is_mouse_down();
  fan::vec2 pos = fan::graphics::screen_to_world(gloco()->get_mouse_position());
  fan::vec2i cell = tilemap.to_grid(pos);

  if (gloco()->is_mouse_clicked()) {
  //  crack_overlay.set_color(fan::color(1, 1, 1, fan::random::value(0.f, 1.f)));
  }
 // crack_overlay.set_size(cracking ? fan::vec2(ts) / 2.f : fan::vec2(0));

  if (!held || !tilemap.in_bounds(cell)) {
    crack_progress = 0;
    cracking = false;
    return;
  }

  int idx = cell.x + cell.y * world.map_size.x;
  if (!world.tiles[idx]) {
    crack_progress = 0;
    cracking = false;
    return;
  }

  if (!cracking || crack_tile != cell) {
    cracking = true;
    crack_tile = cell;
    crack_progress = 0;
    auto& shape = tilemap.get_tile(cell);
  //  crack_overlay.set_image(shape.get_image());
  //  crack_overlay.set_position(fan::vec3(fan::vec2(cell) * cfg.tile_size + cfg.tile_size / 2.f, 10000));
  }

  crack_progress += gloco()->delta_time * 30.f;
  if (crack_progress > 1.f) {
    break_block(cell.x, cell.y);
    crack_progress = 0;
    cracking = false;
    return;
  }

  //gloco()->shader_set_value(crack_shader, "crack_strength", crack_progress);
}

void update() {

  update_grass();
  update_crack();
}
void close() {
  ready = false;
}

fan::graphics::tilemap_t tilemap;
fan::graphics::tile_world_generator_t world;
std::vector<fan::physics::entity_t> collisions;
std::vector<fan::graphics::shader_shape_t> grass_leaves;
std::vector<fan::graphics::physics::sprite_t> pickupables;
bool ready = false;
fan::graphics::render_view_t view;
fan::graphics::shader_t crack_shader;
fan::graphics::shader_t grass_shader;
fan::graphics::shader_shape_t crack_overlay;
fan::vec2i crack_tile;
f32_t crack_progress = 0;
bool cracking = false;
f32_t grass_time = 0.0f;
f32_t grass_wind = 0.6f;

cave_config_t cfg;