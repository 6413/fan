#include <algorithm>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>
import fan;
import fan.graphics.vulkan.ray_tracing.hardware_renderer;

namespace rt = fan::graphics::vulkan::ray_tracing;

struct value_noise_2d_t {
  static f32_t fade(f32_t t) { return t * t * t * (t * (t * 6.f - 15.f) + 10.f); }
  static std::uint32_t hash(std::int32_t x, std::int32_t y, std::uint32_t seed) {
    std::uint32_t h = (std::uint32_t)x * 0x8da6b343u ^ (std::uint32_t)y * 0xd8163841u ^ seed * 0xcb1ab31fu;
    h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu; h ^= h >> 16;
    return h;
  }
  f32_t value(f32_t x, f32_t y) const {
    std::int32_t x0 = (std::int32_t)std::floor(x), y0 = (std::int32_t)std::floor(y);
    f32_t tx = fade(x - x0), ty = fade(y - y0);
    auto h = [&](std::int32_t px, std::int32_t py) {
      return (f32_t)(hash(px, py, seed) & 0x00ffffffu) / 16777215.f;
    };
    return std::lerp(std::lerp(h(x0,y0), h(x0+1,y0), tx), std::lerp(h(x0,y0+1), h(x0+1,y0+1), tx), ty);
  }
  f32_t fbm(f32_t x, f32_t y, std::uint32_t octaves, f32_t lacunarity = 2.f, f32_t gain = 0.5f) const {
    f32_t sum = 0, amp = 1, norm = 0, freq = 1;
    for (std::uint32_t i = 0; i < octaves; ++i) {
      sum += value(x * freq, y * freq) * amp; norm += amp; freq *= lacunarity; amp *= gain;
    }
    return sum / norm;
  }
  std::uint32_t seed = 1337;
};

struct chunk_coord_t {
  bool operator==(const chunk_coord_t& o) const { return x == o.x && z == o.z; }
  std::int32_t x = 0, z = 0;
};
struct chunk_coord_hash_t {
  std::size_t operator()(const chunk_coord_t& c) const {
    return (std::size_t)((std::uint64_t)(std::uint32_t)c.x | ((std::uint64_t)(std::uint32_t)c.z << 32));
  }
};
struct terrain_chunk_t {
  rt::context_t::object_handle_t handle;
  bool uploaded = false;
  bool pending  = false;
};
struct chunk_build_result_t {
  chunk_coord_t coord;
  rt::context_t::voxel_mesh_input_t mesh;
};

struct terrain_streamer_t {
  static constexpr std::uint32_t chunk_size         = 32;
  static constexpr std::uint32_t sy                 = 96;
  static constexpr std::uint32_t height_cache_width = chunk_size + 2;
  static constexpr std::uint32_t sea_level          = 28;
  static constexpr std::uint32_t snow_line          = 74;
  static constexpr std::uint32_t rock_line          = 62;
  static constexpr f32_t         voxel_size         = 1.5f;

  void init() {
    water_color = cv(palette.get(65));   sand_color  = cv(palette.get(115));
    dirt_color  = cv(palette.get(145));  grass_color = cv(palette.get(220));
    rock_color  = cv(palette.get(165));  snow_color  = cv(palette.get(300));
    worker_count = std::max<std::uint32_t>(1, std::thread::hardware_concurrency() - 1);
    for (std::uint32_t i = 0; i < worker_count; ++i) {
      workers.emplace_back([this] { worker_loop(); });
    }
  }
  void destroy() {
    { std::unique_lock lk(queue_mtx); stop = true; }
    queue_cv.notify_all();
    for (auto& t : workers) { if (t.joinable()) { t.join(); } }
  }
  // One chunk per frame — each call is O(1) regardless of scene size
  void update(rt::context_t& renderer, const fan::vec3& cam) {
    chunk_coord_t center = to_chunk(cam);
    queue_missing(center);
    upload_one(renderer, center);
  }

  static fan::vec4 cv(const fan::color& c) { return fan::vec4(c.r, c.g, c.b, 1.f); }
  chunk_coord_t to_chunk(const fan::vec3& p) const {
    f32_t s = (f32_t)chunk_size * voxel_size;
    return { (std::int32_t)std::floor(p.x / s), (std::int32_t)std::floor(p.z / s) };
  }
  std::uint32_t height_at(std::int32_t x, std::int32_t z) const {
    f32_t h = noise.fbm((f32_t)x * 0.0028f, (f32_t)z * 0.0028f, 5, 2.f, 0.55f) * 0.7f
            + noise.fbm((f32_t)x * 0.011f,  (f32_t)z * 0.011f,  5, 2.f, 0.5f)  * 0.25f
            + noise.fbm((f32_t)x * 0.045f,  (f32_t)z * 0.045f,  3, 2.f, 0.45f) * 0.05f;
    h = std::clamp((h - 0.25f) / 0.72f, 0.f, 1.f);
    h = h * h * (3.f - 2.f * h);
    return (std::uint32_t)std::clamp(8.f + h * ((f32_t)sy - 14.f), 1.f, (f32_t)sy - 2.f);
  }
  f32_t cave_at(std::int32_t x, std::int32_t y, std::int32_t z) const {
    return (noise.fbm((f32_t)x*0.035f+(f32_t)y*0.011f, (f32_t)z*0.035f-(f32_t)y*0.017f, 4)
          + noise.fbm((f32_t)x*0.021f-(f32_t)y*0.013f, (f32_t)z*0.021f+(f32_t)y*0.019f, 3)) * 0.5f;
  }
  fan::vec4 terrain_color(std::uint32_t y, std::uint32_t height, f32_t slope) const {
    if (height <= sea_level + 2) { return y + 2 >= height ? sand_color : dirt_color; }
    if (height >= snow_line)     { return y + 1 >= height ? snow_color : rock_color; }
    if (height >= rock_line || slope > 7.f) { return y + 1 >= height ? rock_color : dirt_color; }
    return y + 1 >= height ? grass_color : dirt_color;
  }
  chunk_build_result_t build_chunk(chunk_coord_t coord) const {
    rt::context_t::voxel_grid_t grid;
    grid.resize(chunk_size, sy, chunk_size);
    std::vector<std::uint32_t> hcache(height_cache_width * height_cache_width);
    auto hidx = [](std::uint32_t x, std::uint32_t z) { return z * height_cache_width + x; };
    for (std::uint32_t z = 0; z < height_cache_width; ++z) {
      for (std::uint32_t x = 0; x < height_cache_width; ++x) {
        hcache[hidx(x, z)] = height_at(
          coord.x * (std::int32_t)chunk_size + (std::int32_t)x - 1,
          coord.z * (std::int32_t)chunk_size + (std::int32_t)z - 1);
      }
    }
    for (std::uint32_t z = 0; z < chunk_size; ++z) {
      for (std::uint32_t x = 0; x < chunk_size; ++x) {
        std::int32_t wx = coord.x * (std::int32_t)chunk_size + (std::int32_t)x;
        std::int32_t wz = coord.z * (std::int32_t)chunk_size + (std::int32_t)z;
        std::uint32_t hx = x + 1, hz = z + 1;
        std::uint32_t height = hcache[hidx(hx, hz)];
        f32_t slope = std::abs((f32_t)height - (f32_t)hcache[hidx(hx+1,hz)])
                    + std::abs((f32_t)height - (f32_t)hcache[hidx(hx-1,hz)])
                    + std::abs((f32_t)height - (f32_t)hcache[hidx(hx,hz+1)])
                    + std::abs((f32_t)height - (f32_t)hcache[hidx(hx,hz-1)]);
        std::uint32_t col_limit = std::min<std::uint32_t>(std::max(height, sea_level), sy - 1);
        for (std::uint32_t y = 0; y <= col_limit; ++y) {
          if (y <= height) {
            if (enable_caves && y > 8 && y + 5 < height && ((x+y+z)&7) == 0 && cave_at(wx,(std::int32_t)y,wz) > 0.76f) { continue; }
            auto& v = grid.at(x,y,z); v.id = 1; v.color = terrain_color(y, height, slope);
          }
          else if (y <= sea_level) { auto& v = grid.at(x,y,z); v.id = 1; v.color = water_color; }
        }
      }
    }
    auto mesh = rt::context_t::greedy_mesh_grid(grid, voxel_size);
    mesh.transform = fan::translate(fan::vec3(
      (f32_t)(coord.x * (std::int32_t)chunk_size) * voxel_size,
      -(f32_t)sea_level * voxel_size,
      (f32_t)(coord.z * (std::int32_t)chunk_size) * voxel_size));
    return { .coord = coord, .mesh = std::move(mesh) };
  }

  void worker_loop() {
    while (true) {
      chunk_coord_t coord{};
      {
        std::unique_lock lk(queue_mtx);
        queue_cv.wait(lk, [&] { return stop || !work_queue.empty(); });
        if (stop && work_queue.empty()) { return; }
        coord = work_queue.front();
        work_queue.erase(work_queue.begin());
      }
      auto result = build_chunk(coord);
      { std::lock_guard lk(ready_mtx); ready.push_back(std::move(result)); }
    }
  }
  void queue_missing(chunk_coord_t center) {
    std::int32_t r = render_distance;
    std::vector<chunk_coord_t> wanted;
    for (std::int32_t z = center.z - r; z <= center.z + r; ++z) {
      for (std::int32_t x = center.x - r; x <= center.x + r; ++x) {
        std::int32_t dx = x - center.x, dz = z - center.z;
        if (dx*dx + dz*dz > r*r) { continue; }
        chunk_coord_t c{x, z};
        auto& ch = chunks[c];
        if (!ch.uploaded && !ch.pending) { wanted.push_back(c); }
      }
    }
    std::sort(wanted.begin(), wanted.end(), [&](const chunk_coord_t& a, const chunk_coord_t& b) {
      std::int32_t adx=a.x-center.x, adz=a.z-center.z, bdx=b.x-center.x, bdz=b.z-center.z;
      return adx*adx+adz*adz < bdx*bdx+bdz*bdz;
    });
    { std::lock_guard lk(queue_mtx);
      for (chunk_coord_t c : wanted) { chunks[c].pending = true; work_queue.push_back(c); } }
    queue_cv.notify_all();
  }
  // Picks the closest ready chunk and uploads it via add_mesh_incremental.
  // Cost per call: 5 buffer-grow ops + 1 BLAS build + 1 TLAS rebuild.
  // All constant-time relative to total chunk count.
  void upload_one(rt::context_t& renderer, chunk_coord_t center) {
    chunk_build_result_t best{};
    {
      std::lock_guard lk(ready_mtx);
      if (ready.empty()) { return; }
      std::size_t bi = 0;
      std::int32_t bd = std::numeric_limits<std::int32_t>::max();
      for (std::size_t i = 0; i < ready.size(); ++i) {
        std::int32_t dx = ready[i].coord.x - center.x, dz = ready[i].coord.z - center.z;
        std::int32_t d = dx*dx + dz*dz;
        if (d < bd) { bd = d; bi = i; }
      }
      best = std::move(ready[bi]);
      ready.erase(ready.begin() + bi);
    }
    auto& ch = chunks[best.coord];
    if (ch.uploaded) { ch.pending = false; return; }
    ch.handle   = renderer.add_mesh_incremental(best.mesh);
    ch.uploaded = true;
    ch.pending  = false;
  }
  // Used only for the initial synchronous load before the renderer opens
  void load_initial(rt::context_t& renderer, chunk_coord_t center, std::int32_t radius) {
    for (std::int32_t z = center.z - radius; z <= center.z + radius; ++z) {
      for (std::int32_t x = center.x - radius; x <= center.x + radius; ++x) {
        std::int32_t dx = x - center.x, dz = z - center.z;
        if (dx*dx + dz*dz > radius*radius) { continue; }
        chunk_coord_t coord{x, z};
        auto& ch = chunks[coord];
        if (ch.uploaded) { continue; }
        auto result  = build_chunk(coord);
        ch.handle    = renderer.add_mesh(result.mesh);  // deferred, no GPU work yet
        ch.uploaded  = true;
      }
    }
  }

  std::vector<std::thread> workers;
  std::vector<chunk_coord_t> work_queue;
  std::mutex queue_mtx;
  std::condition_variable queue_cv;
  bool stop = false;
  std::uint32_t worker_count = 1;

  std::vector<chunk_build_result_t> ready;
  std::mutex ready_mtx;

  std::unordered_map<chunk_coord_t, terrain_chunk_t, chunk_coord_hash_t> chunks;
  fan::graphics::terrain_palette_t palette;
  value_noise_2d_t noise;
  fan::vec4 water_color, sand_color, dirt_color, grass_color, rock_color, snow_color;

  std::int32_t render_distance     = 12;
  std::int32_t initial_load_radius = 3;
  bool         enable_caves        = false;
};

int main() {
  fan::graphics::engine_t engine{{
    .renderer = fan::graphics::renderer_t::vulkan,
  }};

  rt::context_t renderer(engine);
  renderer.set_light(fan::vec3(0.f, 220.f, -180.f), fan::vec3(1.f, 0.96f, 0.88f), 12.f);

  terrain_streamer_t terrain;
  terrain.init();

  auto cam = engine.perspective_render_view.camera;
  engine.camera_set_position(cam, fan::vec3(0, 90, -520));
  terrain.load_initial(renderer, terrain.to_chunk(engine.camera_get(cam).position), terrain.initial_load_radius);

  auto cb = engine.window.add_mouse_motion_callback([&](const auto& d) {
    if (engine.window.is_cursor_enabled()) { return; }
    engine.camera_get(cam).rotate_camera(d.motion);
  });

  engine.loop([&] {
    terrain.update(renderer, engine.camera_get(cam).position);

    if (engine.is_key_clicked(fan::key_r))         { renderer.reload_pipeline(); }
    if (engine.is_mouse_clicked(fan::mouse_right))  { engine.window.toggle_cursor(); }
    if (engine.is_mouse_released(fan::mouse_right)) { engine.window.toggle_cursor(); }

    if (auto h = fan::graphics::gui::hud_interactive{"##rt"}; h && engine.is_toggled(fan::key_t)) {
      fan::graphics::gui::camera_controls();
      fan::graphics::gui::drag("render distance", &terrain.render_distance);
      fan::graphics::gui::checkbox("caves",        &terrain.enable_caves);
      renderer.render_gui();
    }
    else {
      engine.camera_move();
    }
  });

  terrain.destroy();
}
