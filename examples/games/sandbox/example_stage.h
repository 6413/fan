#include "world_generate.h"
#include "grass_shader.h"

struct pickupable_t {
  fan::graphics::physics::sprite_t sprite;
  f32_t spawn_time;
};

struct digging_config_t {
  f32_t dig_range = 200.f;
  f32_t dig_speed = 30.f;
  f32_t pickup_range = 100.f;
};

void break_block(int x, int y) {
  int idx = x + y * world.map_size.x;
  if (!world.tiles[idx]) {
    return;
  }
  world.tiles[idx] = false;
  tilemap.set_tile_color({x, y}, cfg.empty_color);
  collisions[idx].destroy();

  auto& shape = tilemap.get_tile({x, y});
  fan::graphics::physics::sprite_t::properties_t props;
  props.position = shape.get_position().offset_z(1);
  props.size = shape.get_size() / 4.f;
  props.image = shape.get_image();
  props.body_type = fan::physics::body_type_e::dynamic_body;
  props.shape_properties.linear_damping = 2.f;
  props.shape_properties.angular_damping = 2.f;

  f32_t desired_velocity = 200.f;
  f32_t mass = props.shape_properties.density * (props.size.x / fan::physics::length_units_per_meter) * (props.size.y / fan::physics::length_units_per_meter);

  pickupables.push_back({
    fan::graphics::physics::sprite_t(props),
    (f32_t)fan::time::now()
    });

  auto& p = pickupables.back().sprite;
  fan::vec2 impulse = fan::vec2(
    fan::random::value(-desired_velocity, desired_velocity),
    -desired_velocity
  ) * mass;
  p.apply_linear_impulse_center(impulse);
}
void update_grass() {
  int seed = 0;
  for (auto& leaf : grass_leaves) {

    float sway = sin(grass_time * 4.f + seed++);

    float max_angle = 0.05f;
    float angle_z = sway * max_angle * grass_wind;

    leaf.set_angle({0, 0, angle_z});
  }

  grass_time += gloco()->delta_time;
  gloco()->shader_set_value(grass_shader, "grass_time", grass_time);
  gloco()->shader_set_value(grass_shader, "grass_wind", grass_wind);
}
fan::vec2i find_closest_tile(const fan::vec2& player_pos) {
  fan::vec2 mouse_pos = fan::graphics::screen_to_world(gloco()->get_mouse_position());
  fan::vec2 dir = (mouse_pos - player_pos).normalized();

  std::vector<fan::vec2i> cells = fan::graphics::algorithm::grid_raycast(
    {player_pos, player_pos + dir * dig_cfg.dig_range},
    fan::vec2(cfg.tile_size)
  );

  for (const auto& cell : cells) {
    if (!tilemap.in_bounds(cell)) {
      continue;
    }
    int idx = cell.x + cell.y * world.map_size.x;
    if (world.tiles[idx]) {
      return cell;
    }
  }
  return fan::vec2i(-1, -1);
}
void update_digging(const fan::vec2& player_pos) {
  bool held = gloco()->is_mouse_down();

  if (!held) {
    crack_progress = 0;
    cracking = false;
    return;
  }

  fan::vec2i cell = find_closest_tile(player_pos);

  if (cell.x == -1) {
    crack_progress = 0;
    cracking = false;
    return;
  }

  if (!cracking || crack_tile != cell) {
    cracking = true;
    crack_tile = cell;
    crack_progress = 0;
  }

  crack_progress += gloco()->delta_time * dig_cfg.dig_speed;
  if (crack_progress > 1.f) {
    break_block(cell.x, cell.y);
    crack_progress = 0;
    cracking = false;
  }
}
void update_pickupables(const fan::vec2& player_pos) {
  for (int i = pickupables.size() - 1; i >= 0; --i) {
    auto& p = pickupables[i];
    fan::vec2 pickup_pos = p.sprite.get_position();
    f32_t dist = (pickup_pos - player_pos).length();

    if (dist < dig_cfg.pickup_range) {
      p.sprite.erase();
      pickupables.erase(pickupables.begin() + i);
    }
  }
}

void gen_world() {
  f32_t ts = cfg.tile_size;

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
}

void open(void* sod) {
  cloud_system = new parallax_cloud_system_t();
  cloud_system->init(5);
  f32_t ts = cfg.tile_size;

  tilemap = fan::graphics::tilemap_t(fan::vec2(ts), fan::colors::gray, fan::vec2(cfg.map_tiles) * ts, fan::vec3(0, 0, 100));
  crack_shader = fan::graphics::shader_t {gloco()->get_sprite_vertex_shader(crack_shader_fragment)};
  grass_shader = fan::graphics::shader_t {gloco()->get_sprite_vertex_shader(grass_shader_fragment)};

  parallax_bg = fan::graphics::shapes_from_json("background.json");
  for (auto [i, shape] : fan::enumerate(parallax_bg)) {
    shape.remove_culling();
    shape.push_shaper();
    shape.set_position(fan::vec2(shape.get_position()) + fan::vec2(3000, 0));
    shape.set_parallax_factor(1.f - shape.get_position().z / 100.f);
  }

  gen_world();

  pile.engine.lighting.set_target(0.0);

}
void close() {
  for (auto& p : pickupables) {
    p.sprite.erase();
  }
  pickupables.clear();
}
void update() {
  //pile.engine.clear_color = fan::color::from_rgb(0x574a89) * fan::color(pile.engine.lighting.ambient);

  if (cloud_system) {
    cloud_system->update(gloco()->delta_time);
  }
  update_grass();
  update_digging(pile.player.body.get_center());
  update_pickupables(pile.player.body.get_center());
  auto sky_cs = sky_colors;
  for (auto& col : sky_cs) {
    col = col * fan::color(pile.engine.lighting.ambient);
  }
  sky.set_colors(sky_cs);
}

std::vector<pickupable_t> pickupables;
fan::graphics::shader_shape_t crack_overlay;
fan::vec2i crack_tile;
f32_t crack_progress = 0;
bool cracking = false;
f32_t grass_time = 0.0f;
f32_t grass_wind = 0.6f;
digging_config_t dig_cfg;
std::vector<fan::graphics::shape_t> parallax_bg;

static constexpr fan::vec2 world_size {1024 * 32, 32 * 32};

std::array<fan::color, 4> sky_colors {
  fan::color(0.40, 0.65, 0.95, 1.0),
  fan::color(0.46, 0.71, 0.97, 1.0),  // base #75b4f7
  fan::color(0.60, 0.80, 0.98, 1.0),
  fan::color(0.70, 0.85, 0.99, 1.0)
};

fan::graphics::gradient_t sky {{
  .position = fan::vec3(world_size.x * 0.5f, -world_size.y * 0.5f, 0),
  .size = fan::vec2(world_size.x * 0.5f, world_size.y * .5f),
  .color = sky_colors
}};


inline static const char* cloud_fragment_shader = R"(#version 330
layout (location = 0) out vec4 o_attachment0;

in vec2 texture_coordinate;
in vec4 instance_color;

uniform float _time;

float hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);

    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));

    float nx0 = mix(a, b, f.x);
    float nx1 = mix(c, d, f.x);
    return mix(nx0, nx1, f.y);
}

float fbm(vec2 p) {
    float value = 0.0;
    float amplitude = 0.55;
    float frequency = 1.0;

    for (int i = 0; i < 4; ++i) {
        value += amplitude * noise(p * frequency);
        frequency *= 2.0;
        amplitude *= 0.5;
    }
    return value;
}

void main() {
    vec2 uv = texture_coordinate;
    uv = uv * 2.0 - 1.0;

    vec2 cloud_seed = vec2(instance_color.r, instance_color.g) * 1000.0;
    float cloud_alpha = instance_color.b;

    vec2 p = uv * 2.0 + cloud_seed;

    float cloud = fbm(p + _time * 0.004);

    float seed_hash = hash(cloud_seed);
    cloud *= (0.85 + seed_hash * 0.3);

    float dist_x = abs(uv.x) * (0.7 * (0.85 + seed_hash * 0.3));
    float dist_y = abs(uv.y) * (1.2 / (0.85 + seed_hash * 0.3));

    float dist = dist_x * dist_x + dist_y * dist_y;
    cloud *= smoothstep(1.0, 0.04, dist);

    cloud = cloud * 0.9 + cloud * cloud * 0.1;

    float alpha = smoothstep(0.28, 0.55, cloud);

    float edge_low = smoothstep(0.16, 0.4, alpha);
    float edge_high = smoothstep(0.9, 0.6, alpha);
    float outline = edge_low * edge_high * 0.5;

    float brightness = cloud * 0.28 + 0.72;
    vec3 cloud_color = vec3(0.95, 0.97, 1.0) * brightness;

    vec3 shadow_color = vec3(0.7, 0.75, 0.85);
    cloud_color = mix(shadow_color, cloud_color, smoothstep(0.3, 0.7, cloud));

    vec3 outline_color = vec3(0.65, 0.7, 0.8);
    vec3 final_color = mix(cloud_color, outline_color, outline);

    float final_alpha = alpha * cloud_alpha;

    if (final_alpha < 0.04)
        discard;

    o_attachment0 = vec4(final_color, final_alpha*2.f);
})";

inline static fan::graphics::image_t g_cloud_white_image;

struct parallax_cloud_system_t {
  struct cloud_t {
    fan::graphics::shader_shape_t shape;
    fan::vec2 velocity;
    fan::vec2 seed;
    f32_t layer_depth;
  };

  fan::graphics::shader_t cloud_shader;
  std::vector<std::vector<cloud_t>> layers;

  void init(uint32_t num_layers = 5) {
    cloud_shader = gloco()->get_sprite_vertex_shader(cloud_fragment_shader);

    if (!g_cloud_white_image) {
      g_cloud_white_image = gloco()->image_create(fan::colors::white);
    }

    layers.clear();
    layers.resize(num_layers);

    for (uint32_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
      f32_t depth_ratio = (f32_t)layer_idx / fan::math::max<f32_t>(1.0f, (f32_t)(num_layers - 1));
      f32_t scroll_speed = 12.f + depth_ratio * 45.f;
      f32_t z_depth = 50.f + 1 * 8.f;

      uint32_t base_min = 22u;
      uint32_t base_max = 40u;
      f32_t density_scale = 0.6f + depth_ratio * 0.9f;

      uint32_t clouds_per_layer = (uint32_t)(fan::random::value(
        (uint32_t)(base_min * density_scale),
        (uint32_t)(base_max * density_scale)
      ));

      layers[layer_idx].reserve(clouds_per_layer);

      for (uint32_t i = 0; i < clouds_per_layer; ++i) {
        cloud_t cloud;

        f32_t width = fan::random::value(180.f, 420.f);
        f32_t height = width * fan::random::value(0.32f, 0.5f);
        fan::vec2 size(width, height);

        fan::vec2 position = fan::vec2(
          fan::random::value(0.f, world_size.x),
          fan::random::value(-world_size.y * 0.65f, -world_size.y * 0.18f)
        );

        cloud.seed = fan::random::vec2(
          fan::vec2(0, 0),
          fan::vec2(1000, 1000)
        );

        f32_t speed_variation = fan::random::value(0.8f, 1.25f);
        cloud.velocity = fan::vec2(scroll_speed * speed_variation, 0);
        cloud.layer_depth = z_depth;

        fan::color encoded_color = fan::color(
          cloud.seed.x / 1000.0f,
          cloud.seed.y / 1000.0f,
          0.75f - depth_ratio * 0.25f,
          1.0f
        );

        cloud.shape = fan::graphics::shader_shape_t {{
            .position = fan::vec3(position, z_depth),
            .size = size,
            .color = encoded_color,
            .shader = cloud_shader,
            .image = g_cloud_white_image
        }};

        layers[layer_idx].push_back(cloud);
      }
    }
  }

  void update(f32_t delta_time) {
    for (auto& layer : layers) {
      for (auto& cloud : layer) {
        fan::vec2 pos = cloud.shape.get_position();
        pos += cloud.velocity * delta_time;

        fan::vec2 cloud_size = cloud.shape.get_size();
        if (pos.x > world_size.x + cloud_size.x) {
          pos.x = -cloud_size.x;
          pos.y += fan::random::value(-80.f, 80.f);
          pos.y = fan::math::clamp(pos.y, -world_size.y * 0.7f, -world_size.y * 0.15f);
        }

        cloud.shape.set_position(pos);
      }
    }
  }

  void cleanup() {
    gloco()->shader_erase(cloud_shader);
  }
};

parallax_cloud_system_t* cloud_system = nullptr;
