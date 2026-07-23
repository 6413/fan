module;

module fan.graphics.algorithm.chunk_renderer;

import std;

import fan.types.vector;
import fan.graphics;
import fan.graphics.physics_shapes;
import fan.noise;
import fan.graphics.algorithm.raymarch;

namespace fan::graphics::algorithm {

chunk_renderer_t::chunk_renderer_t(config_t cfg)
  : m_cfg(std::move(cfg)) {}

chunk_renderer_t::~chunk_renderer_t() {
  for (auto& [cc, chunk] : m_chunks) {
    for (auto& e : chunk.colliders) {
      e.destroy();
    }
  }
}

bool chunk_renderer_t::get_solid(int gx, int gy) const {
  auto it = m_solid_map.find({gx, gy});
  if (it != m_solid_map.end()) {
    return it->second;
  }
  if (m_cfg.is_solid) {
    return m_cfg.is_solid(gx, gy);
  }
  if (!m_cfg.hill_noise) {
    return false;
  }
  f32_t surface = surface_height(gx);
  if (gy < surface) return false;
  f32_t depth = gy - surface;
  if (depth > m_cfg.cave_depth_min && is_cave(gx, gy)) return false;
  return true;
}

f32_t chunk_renderer_t::surface_height(int gx) const {
  auto& hn = *m_cfg.hill_noise;
  auto& dn = *m_cfg.detail_noise;
  f32_t h0 = hn.simplex_fbm_norm(gx * m_cfg.hill_freq, 0) * m_cfg.hill_amp;
  f32_t h1 = dn.simplex_fbm_norm(gx * m_cfg.detail_freq, 123.f) * m_cfg.detail_amp;
  f32_t h2 = dn.simplex_fbm_norm(gx * m_cfg.micro_freq, 456.f) * m_cfg.micro_amp;
  return m_cfg.surface_base + h0 + h1 + h2;
}

bool chunk_renderer_t::is_cave(int gx, int gy) const {
  auto& cn = *m_cfg.cave_noise;
  f32_t n = cn.simplex_fbm_norm(gx * m_cfg.cave_freq, gy * m_cfg.cave_freq);
  f32_t n2 = cn.simplex_fbm_norm(gx * m_cfg.cave_freq * 2.f + 200.f, gy * m_cfg.cave_freq * 2.f + 300.f);
  f32_t c = n * 0.7f + n2 * 0.3f;
  f32_t tunnel = 1.f - std::abs(c - 0.5f) * 4.f;
  f32_t depth = gy - surface_height(gx);
  f32_t dscale = std::clamp(depth / m_cfg.cave_blend, 0.f, 1.f);
  f32_t deep_c = cn.simplex_fbm_norm(gx * 0.02f + 500.f, gy * 0.02f + 500.f);
  f32_t deep_tunnel = 1.f - std::abs(deep_c - 0.5f) * 4.f;
  f32_t mixed = tunnel * (1.f - dscale) + deep_tunnel * dscale;
  return mixed > 0.4f;
}

fan::graphics::image_t chunk_renderer_t::tile_image(int gx, int gy) const {
  f32_t depth = gy - surface_height(gx);
  for (auto& [threshold, img] : m_cfg.tile_layers) {
    if (depth < threshold) return img;
  }
  return fan::graphics::image_t{};
}

void chunk_renderer_t::set_solid(int gx, int gy, bool solid) {
  m_solid_map[{gx, gy}] = solid;
}

void chunk_renderer_t::set_cell_sprite(chunk_t& chunk, fan::vec2i local, fan::vec2 world_pos, int gx, int gy) {
  auto it = chunk.sprites.find(local);
  if (!get_solid(gx, gy)) {
    if (it != chunk.sprites.end()) {
      chunk.sprites.erase(it);
    }
    return;
  }
  fan::graphics::image_t img = m_cfg.get_image ? m_cfg.get_image(gx, gy) : tile_image(gx, gy);
  if (it != chunk.sprites.end()) {
    it->second.set_image(img);
    return;
  }
  auto& s = chunk.sprites[local];
  s = fan::graphics::sprite_t{{
    .position = fan::vec3(world_pos, 0),
    .size = fan::vec2(m_cfg.cell_size, m_cfg.cell_size) * 0.5f,
    .image = img,
  }};
}

void chunk_renderer_t::remesh_chunk(fan::vec2i cc) {
  f32_t cw = m_cfg.chunk_size * m_cfg.cell_size;
  fan::vec2 origin = fan::vec2(cc) * cw;
  auto& chunk = m_chunks[cc];

  for (int cy = 0; cy < m_cfg.chunk_size; ++cy) {
    for (int cx = 0; cx < m_cfg.chunk_size; ++cx) {
      int gx = cc.x * m_cfg.chunk_size + cx;
      int gy = cc.y * m_cfg.chunk_size + cy;
      fan::vec2 center = origin + (fan::vec2(cx, cy) + 0.5f) * m_cfg.cell_size;
      set_cell_sprite(chunk, {cx, cy}, center, gx, gy);
    }
  }
}

void chunk_renderer_t::remesh_chunk_physics(fan::vec2i cc) {
  auto& chunk = m_chunks[cc];
  for (auto& e : chunk.colliders) {
    e.destroy();
  }
  chunk.colliders.clear();

  int cs = m_cfg.chunk_size;
  std::vector<bool> solid(cs * cs);
  std::vector<bool> visited(cs * cs);

  for (int cy = 0; cy < cs; ++cy) {
    for (int cx = 0; cx < cs; ++cx) {
      solid[cy * cs + cx] = get_solid(cc.x * cs + cx, cc.y * cs + cy);
    }
  }

  fan::vec2 origin = fan::vec2(cc) * cs * m_cfg.cell_size;

  for (int cy = 0; cy < cs; ++cy) {
    for (int cx = 0; cx < cs; ++cx) {
      if (!solid[cy * cs + cx] || visited[cy * cs + cx]) continue;

      int rx = cx;
      while (rx + 1 < cs && solid[cy * cs + rx + 1] && !visited[cy * cs + rx + 1]) ++rx;

      int ry = cy;
      for (bool row_ok = true; row_ok && ry + 1 < cs; ) {
        for (int x = cx; x <= rx; ++x) {
          if (!solid[(ry + 1) * cs + x] || visited[(ry + 1) * cs + x]) { row_ok = false; break; }
        }
        if (row_ok) ++ry;
      }

      for (int y = cy; y <= ry; ++y) {
        for (int x = cx; x <= rx; ++x) {
          visited[y * cs + x] = true;
        }
      }

      fan::vec2 half = fan::vec2(rx - cx + 1, ry - cy + 1) * m_cfg.cell_size * 0.5f;
      fan::vec2 pos = origin + fan::vec2(cx, cy) * m_cfg.cell_size + half;
      chunk.colliders.push_back(fan::physics::gphysics()->create_box(pos, half, 0, fan::physics::body_type_e::static_body, m_cfg.shape_properties));
    }
  }
}

void chunk_renderer_t::stream(fan::vec2 cam_pos, fan::vec2 viewport_size) {
  f32_t chunk_world = m_cfg.chunk_size * m_cfg.cell_size;
  fan::vec2 half_ws = viewport_size * 0.5f;

  fan::vec2i min_cc{
    (int)std::floor((cam_pos.x - half_ws.x) / chunk_world) - 1,
    (int)std::floor((cam_pos.y - half_ws.y) / chunk_world) - 1
  };
  fan::vec2i max_cc{
    (int)std::floor((cam_pos.x + half_ws.x) / chunk_world) + 1,
    (int)std::floor((cam_pos.y + half_ws.y) / chunk_world) + 1
  };

  if (m_last_center == min_cc && !m_chunks.empty()) return;
  m_last_center = min_cc;

  for (int y = min_cc.y; y <= max_cc.y; ++y) {
    for (int x = min_cc.x; x <= max_cc.x; ++x) {
      fan::vec2i cc{x, y};
      if (!m_chunks.contains(cc)) {
        remesh_chunk(cc);
        remesh_chunk_physics(cc);
      }
    }
  }

  for (auto it = m_chunks.begin(); it != m_chunks.end(); ) {
    if (it->first.x < min_cc.x || it->first.x > max_cc.x ||
        it->first.y < min_cc.y || it->first.y > max_cc.y) {
      for (auto& e : it->second.colliders) e.destroy();
      it = m_chunks.erase(it);
    }
    else {
      ++it;
    }
  }
}

void chunk_renderer_t::dig(fan::vec2 world_pos, f32_t radius) {
  int gx0 = (int)std::floor((world_pos.x - radius) / m_cfg.cell_size);
  int gx1 = (int)std::ceil((world_pos.x + radius) / m_cfg.cell_size);
  int gy0 = (int)std::floor((world_pos.y - radius) / m_cfg.cell_size);
  int gy1 = (int)std::ceil((world_pos.y + radius) / m_cfg.cell_size);

  auto chunk_of = [this](int gx, int gy) {
    return fan::vec2i{(int)std::floor((f32_t)gx / m_cfg.chunk_size), (int)std::floor((f32_t)gy / m_cfg.chunk_size)};
  };

  std::unordered_set<fan::vec2i> touched;

  for (int gy = gy0; gy <= gy1; ++gy) {
    for (int gx = gx0; gx <= gx1; ++gx) {
      fan::vec2 cell_min = fan::vec2(gx, gy) * m_cfg.cell_size;
      fan::vec2 cell_max = cell_min + m_cfg.cell_size;
      f32_t cx = std::clamp(world_pos.x, cell_min.x, cell_max.x);
      f32_t cy = std::clamp(world_pos.y, cell_min.y, cell_max.y);
      if ((fan::vec2(cx, cy) - world_pos).length() > radius) continue;
      if (!get_solid(gx, gy)) continue;
      m_solid_map[{gx, gy}] = false;
      fan::vec2i cc = chunk_of(gx, gy);
      auto it = m_chunks.find(cc);
      if (it != m_chunks.end()) {
        fan::vec2 cell_center = cell_min + m_cfg.cell_size * 0.5f;
        set_cell_sprite(it->second, {gx - cc.x * m_cfg.chunk_size, gy - cc.y * m_cfg.chunk_size}, cell_center, gx, gy);
      }
      touched.insert(cc);
    }
  }

  for (auto& cc : touched) {
    remesh_chunk_physics(cc);
  }
}

fan::vec2 chunk_renderer_t::raycast(fan::vec2 start, fan::vec2 end, f32_t radius) const {
  return raymarch_thick(start, end, m_cfg.cell_size, radius, [this](int gx, int gy) {
    return get_solid(gx, gy);
  });
}

}
