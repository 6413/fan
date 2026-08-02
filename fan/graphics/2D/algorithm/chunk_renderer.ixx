module;

export module fan.graphics.algorithm.chunk_renderer;

import std;

import fan.types;
import fan.types.vector;
import fan.types.color;
import fan.graphics.common_context;
import fan.physics.types;
import fan.physics.b2_integration;
import fan.graphics.physics_shapes;
import fan.graphics;
import fan.noise;

export namespace fan::graphics::algorithm {

struct chunk_renderer_t {
  using solid_fn_t = std::function<bool(int gx, int gy)>;
  using image_fn_t = std::function<fan::graphics::image_t(int gx, int gy)>;

  struct config_t {
    f32_t cell_size = 16.f;
    int chunk_size = 32;
    solid_fn_t is_solid;
    image_fn_t get_image;

    fan::noise_t* hill_noise = nullptr;
    fan::noise_t* cave_noise = nullptr;
    fan::noise_t* detail_noise = nullptr;
    f32_t surface_base = 108.f;
    f32_t hill_freq = 0.002f;
    f32_t hill_amp = 60.f;
    f32_t detail_freq = 0.01f;
    f32_t detail_amp = 12.f;
    f32_t micro_freq = 0.05f;
    f32_t micro_amp = 4.f;
    f32_t mountain_freq = 0.001f;
    f32_t mountain_amp = 120.f;
    f32_t mountain_power = 3.f;
    fan::noise_t* sky_island_noise = nullptr;
    f32_t sky_island_freq = 0.04f;
    f32_t sky_island_threshold = 0.65f;
    f32_t sky_island_min = 20.f;
    f32_t sky_island_max = 60.f;
    fan::graphics::image_t img_sky_island;
    f32_t cave_freq = 0.015f;
    f32_t cave_depth_min = 5.f;
    f32_t cave_depth_max = 80.f;
    f32_t cave_blend = 80.f;
    f32_t cave_threshold = 0.85f;
    f32_t cave_deep_mult = 1.5f;
    f32_t cave_sharpness = 6.f;

    std::vector<std::pair<f32_t, fan::graphics::image_t>> tile_layers;
    fan::noise_t* scatter_noise = nullptr;
    fan::graphics::image_t scatter_img;
    f32_t scatter_threshold = 0.5f;
    f32_t terrain_z = 20.f;
    fan::physics::shape_properties_t shape_properties = {.friction = 0.6f};
  };

  chunk_renderer_t(config_t cfg);
  ~chunk_renderer_t();

  struct edit_t { int gx; int gy; int type; }; // type -1 = air (dug), else tile layer index

  void stream(fan::vec2 cam_pos, fan::vec2 viewport_size);
  std::vector<edit_t> dig(fan::vec2 world_pos, f32_t radius);
  void place(fan::vec2 world_pos, f32_t radius, int type);
  void set_solid(int gx, int gy, bool solid);
  fan::vec2 raycast(fan::vec2 start, fan::vec2 end, f32_t radius) const;
  f32_t cell_size() const { return m_cfg.cell_size; }
  int chunk_size() const { return m_cfg.chunk_size; }
  f32_t surface_height_at(f32_t world_x) const;
  bool get_solid(int gx, int gy) const;
  bool is_cave(int gx, int gy) const;
  int get_tile_type(int gx, int gy) const;
  std::vector<edit_t> get_edits() const;
  void load_edits(const std::vector<edit_t>& edits);

  struct occluder_rect_t { fan::vec2 center; fan::vec2 half_size; };
  std::vector<occluder_rect_t> build_occluders(fan::vec2 view_min, fan::vec2 view_max) const;
  const std::vector<fan::vec4>& shadow_occluders();

  bool raycast_visible = true;

private:
  struct chunk_t {
    std::unordered_map<fan::vec2i, fan::graphics::sprite_t> sprites;
    std::vector<fan::physics::entity_t> colliders;
    std::vector<fan::vec4> occluders;
  };

  config_t m_cfg;
  std::unordered_map<fan::vec2i, bool> m_solid_map;
  std::unordered_map<fan::vec2i, int> m_placed_map;
  std::unordered_map<fan::vec2i, chunk_t> m_chunks;
  fan::vec2i m_last_center{std::numeric_limits<int>::max(), std::numeric_limits<int>::max()};
  std::unordered_set<fan::vec2i> m_physics_dirty;

  mutable std::vector<fan::vec4> m_occluders_cache;
  mutable bool m_occluders_cache_dirty = true;

  fan::vec2i chunk_of(int gx, int gy) const;

  f32_t surface_height(int gx) const;
  fan::graphics::image_t tile_image(int gx, int gy) const;

  bool is_cave_with_surface(int gx, int gy, f32_t surface) const;
  fan::graphics::image_t tile_image_with_surface(int gx, int gy, f32_t surface) const;
  bool get_solid_with_surface(int gx, int gy, f32_t surface) const;

  void set_cell_sprite(chunk_t& chunk, fan::vec2i local, fan::vec2 world_pos, int gx, int gy);
  void remesh_chunk(fan::vec2i cc);
  void remesh_chunk_physics(fan::vec2i cc);
};

}
