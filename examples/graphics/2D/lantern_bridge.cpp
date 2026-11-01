#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

import fan;

using namespace fan::graphics;

// a knight resting on a stone bridge at night. the scene is painted on the cpu at a low
// resolution and relit per pixel, so glow, ripples and mist all stay on the pixel grid.
namespace lantern_bridge {
  constexpr int scene_w = 480;
  constexpr int scene_h = 270;
  constexpr f32_t tick_rate = 12.f;
  constexpr f32_t lantern_x = 280.f;
  constexpr f32_t lantern_y = 116.f;
  constexpr f32_t color_levels = 24.f;

  constexpr std::array<f32_t, 16> bayer = {
    0.f,  8.f,  2.f,  10.f,
    12.f, 4.f,  14.f, 6.f,
    3.f,  11.f, 1.f,  9.f,
    15.f, 7.f,  13.f, 5.f
  };

  struct rgb_t {
    f32_t r = 0, g = 0, b = 0;
  };
  rgb_t operator+(rgb_t a, rgb_t b) { return {a.r + b.r, a.g + b.g, a.b + b.b}; }
  rgb_t operator*(rgb_t a, rgb_t b) { return {a.r * b.r, a.g * b.g, a.b * b.b}; }
  rgb_t operator*(rgb_t a, f32_t s) { return {a.r * s, a.g * s, a.b * s}; }

  f32_t clamp01(f32_t v) { return std::clamp(v, 0.f, 1.f); }
  f32_t fract(f32_t v) { return v - std::floor(v); }
  f32_t mixf(f32_t a, f32_t b, f32_t t) { return a + (b - a) * t; }
  f32_t smooth01(f32_t a, f32_t b, f32_t v) {
    v = clamp01((v - a) / (b - a));
    return v * v * (3.f - 2.f * v);
  }
  rgb_t mix_rgb(rgb_t a, rgb_t b, f32_t t) {
    return {mixf(a.r, b.r, t), mixf(a.g, b.g, t), mixf(a.b, b.b, t)};
  }

  f32_t hash2(int x, int y) {
    std::uint32_t h = std::uint32_t(x) * 374761393u + std::uint32_t(y) * 668265263u;
    h = (h ^ (h >> 13)) * 1274126177u;
    return f32_t((h ^ (h >> 16)) & 0xffffffu) / 16777216.f;
  }
  f32_t value_noise(f32_t x, f32_t y) {
    int ix = int(std::floor(x));
    int iy = int(std::floor(y));
    f32_t fx = smooth01(0.f, 1.f, x - ix);
    f32_t fy = smooth01(0.f, 1.f, y - iy);
    return mixf(
      mixf(hash2(ix, iy), hash2(ix + 1, iy), fx),
      mixf(hash2(ix, iy + 1), hash2(ix + 1, iy + 1), fx),
      fy
    );
  }
  f32_t fbm(f32_t x, f32_t y) {
    f32_t sum = 0.f, amplitude = 0.5f, total = 0.f;
    for (int i = 0; i < 4; ++i) {
      sum += value_noise(x, y) * amplitude;
      total += amplitude;
      x = x * 2.03f + 17.f;
      y = y * 2.03f + 9.f;
      amplitude *= 0.5f;
    }
    return sum / total;
  }

  struct voronoi_t {
    f32_t d1 = 1e3f, d2 = 1e3f, dx = 0.f, dy = 0.f;
    int id_x = 0, id_y = 0;
  };
  voronoi_t voronoi(f32_t x, f32_t y) {
    voronoi_t out;
    int cx = int(std::floor(x));
    int cy = int(std::floor(y));
    for (int j = -1; j <= 1; ++j) {
      for (int i = -1; i <= 1; ++i) {
        int ix = cx + i;
        int iy = cy + j;
        f32_t dx = x - (ix + 0.15f + 0.7f * hash2(ix, iy));
        f32_t dy = y - (iy + 0.15f + 0.7f * hash2(iy + 91, ix + 17));
        f32_t d = std::sqrt(dx * dx + dy * dy);
        if (d < out.d1) {
          out.d2 = out.d1;
          out.d1 = d;
          out.dx = dx;
          out.dy = dy;
          out.id_x = ix;
          out.id_y = iy;
        }
        else if (d < out.d2) {
          out.d2 = d;
        }
      }
    }
    return out;
  }

  enum class kind_e : std::uint8_t {
    backdrop,
    water,
    stone,
    figure,
    glass
  };

  struct pixel_t {
    rgb_t albedo;
    f32_t nx = 0.f, ny = 0.f, nz = 1.f;
    f32_t gloss = 0.f;
    f32_t shade = 1.f;
    kind_e kind = kind_e::backdrop;
  };

  struct brush_t {
    rgb_t color;
    f32_t gloss = 0.f;
    int bands = 0;
    f32_t folds = 0.f;
  };

  pixel_t surface(rgb_t albedo, f32_t nx, f32_t ny, f32_t gloss, kind_e kind) {
    f32_t nz = std::sqrt(std::max(0.05f, 1.f - nx * nx - ny * ny));
    f32_t length = std::sqrt(nx * nx + ny * ny + nz * nz);
    return {albedo, nx / length, ny / length, nz / length, gloss, 1.f, kind};
  }

  struct painting_t {
    std::vector<pixel_t> pixels = std::vector<pixel_t>(scene_w * scene_h);
    std::vector<f32_t> occlusion = std::vector<f32_t>(scene_w * scene_h, 1.f);
    std::vector<f32_t> vignette = std::vector<f32_t>(scene_w * scene_h, 1.f);
    std::vector<std::uint8_t> frame = std::vector<std::uint8_t>(scene_w * scene_h * 4, 255);
    f32_t flicker = 1.f;

    painting_t() {
      paint_backdrop();
      paint_rocks();
      paint_bridge();
      paint_tree();
      paint_knight();
      paint_lantern();
      bake_occlusion();
      bake_vignette();
    }

    pixel_t& at(int x, int y) {
      return pixels[y * scene_w + x];
    }

    void paint_backdrop() {
      constexpr std::array<f32_t, 6> trunk_x = {64.f, 118.f, 176.f, 262.f, 336.f, 430.f};
      constexpr std::array<f32_t, 6> trunk_w = {7.f, 5.f, 9.f, 6.f, 8.f, 5.f};
      for (int y = 0; y < scene_h; ++y) {
        for (int x = 0; x < scene_w; ++x) {
          f32_t fx = f32_t(x), fy = f32_t(y);
          f32_t bank = 112.f + 7.f * value_noise(fx * 0.04f, 3.f);
          pixel_t p;
          if (fy >= bank) {
            p.kind = kind_e::water;
            at(x, y) = p;
            continue;
          }
          rgb_t color = mix_rgb({0.016f, 0.012f, 0.04f}, {0.05f, 0.024f, 0.09f}, clamp01(fy / bank));
          f32_t ridge = 34.f + 52.f * fbm(fx * 0.014f, 1.7f);
          if (fy > ridge) {
            color = mix_rgb(color, {0.06f, 0.03f, 0.1f}, 0.5f + 0.5f * fbm(fx * 0.05f, fy * 0.07f));
          }
          if (fbm(fx * 0.04f + 9.f, fy * 0.055f) > 0.5f) {
            color = color * 0.62f;
          }
          for (std::size_t i = 0; i < trunk_x.size(); ++i) {
            f32_t center = trunk_x[i] + 5.f * std::sin(fy * 0.03f + f32_t(i));
            if (std::abs(fx - center) < trunk_w[i]) {
              color = rgb_t{0.018f, 0.012f, 0.032f} * (0.8f + 0.5f * value_noise(fx * 0.5f, fy * 0.08f));
            }
          }
          color = mix_rgb(color, {0.03f, 0.03f, 0.05f}, clamp01((fy - bank + 9.f) / 9.f));

          int cell_x = x / 4, cell_y = y / 4;
          f32_t density = clamp01(1.f - fy / (58.f + 24.f * std::sin(fx * 0.019f))) * clamp01(1.25f - fx / 300.f);
          density = std::max(density, 0.6f * clamp01(1.f - fy / 28.f) * clamp01((fx - 300.f) / 120.f));
          density *= 0.55f + 0.9f * fbm(cell_x * 0.2f, cell_y * 0.2f);
          if (hash2(cell_x, cell_y) < density) {
            color = mix_rgb(
              {0.11f, 0.025f, 0.15f}, {0.36f, 0.07f, 0.37f},
              hash2(cell_y + 5, cell_x + 3) * clamp01(1.2f - fy / 70.f)
            );
          }
          p.albedo = color;
          at(x, y) = p;
        }
      }
    }

    pixel_t rubble(int x, int y, f32_t scale, rgb_t tint) {
      f32_t fx = f32_t(x), fy = f32_t(y);
      voronoi_t v = voronoi(fx / scale, fy / scale);
      f32_t edge = (v.d2 - v.d1) * scale;
      if (edge < 1.2f) {
        return surface({0.024f, 0.016f, 0.03f}, 0.f, 0.f, 0.f, kind_e::stone);
      }
      f32_t bevel = clamp01(1.f - edge / 4.5f);
      f32_t length = std::sqrt(v.dx * v.dx + v.dy * v.dy) + 0.001f;
      f32_t nx = v.dx / length * bevel * 0.75f;
      f32_t ny = v.dy / length * bevel * 0.75f;
      f32_t tone = 0.8f + 0.35f * hash2(v.id_x, v.id_y + 31) - 0.3f * ny;
      tone += 0.08f * (value_noise(fx * 0.35f, fy * 0.35f) - 0.5f);
      return surface(tint * tone, nx, ny, 0.f, kind_e::stone);
    }

    pixel_t block(int x, int y, f32_t u, f32_t v, f32_t width, f32_t height, rgb_t tint, f32_t tilt) {
      int row = int(std::floor(v / height));
      f32_t shift = hash2(row, 7) * width + (row & 1 ? width * 0.5f : 0.f);
      int column = int(std::floor((u + shift) / width));
      f32_t fx = fract((u + shift) / width) * width;
      f32_t fy = fract(v / height) * height;
      f32_t edge_x = std::min(fx, width - fx);
      f32_t edge_y = std::min(fy, height - fy);
      f32_t edge = std::min(edge_x, edge_y) + (hash2(x, y) - 0.5f) * 0.6f;
      if (edge < 1.1f) {
        return surface({0.024f, 0.016f, 0.03f}, 0.f, tilt * 0.5f, 0.f, kind_e::stone);
      }
      f32_t nx = 0.f, ny = tilt;
      if (edge < 3.2f) {
        if (edge_x < edge_y) {
          nx = fx < width * 0.5f ? -0.6f : 0.6f;
        }
        else {
          ny += fy < height * 0.5f ? -0.6f : 0.6f;
        }
      }
      f32_t tone = 0.82f + 0.35f * hash2(column, row + 53);
      tone += fy < 2.5f ? 0.22f : (fy > height - 2.5f ? -0.22f : 0.f);
      tone += 0.08f * (value_noise(x * 0.35f, y * 0.35f) - 0.5f);
      return surface(tint * tone, nx, std::clamp(ny, -0.9f, 0.9f), 0.f, kind_e::stone);
    }

    void paint_rocks() {
      for (int x = 300; x < scene_w; ++x) {
        f32_t t = (x - 300.f) / 180.f;
        f32_t top = 118.f - 96.f * std::pow(t, 1.15f) + 22.f * fbm(x * 0.03f, 8.f);
        for (int y = std::max(0, int(top)); y < 150; ++y) {
          at(x, y) = rubble(x, y, 16.f, rgb_t{0.15f, 0.1f, 0.17f});
        }
      }
    }

    f32_t deck_top(f32_t x) const {
      f32_t rough = 1.5f * value_noise(x * 0.12f, 2.f);
      if (x < 182.f) {
        return 134.f + 52.f * std::pow((182.f - x) / 182.f, 1.7f) + rough;
      }
      if (x < 352.f) {
        return 134.f + rough;
      }
      return 134.f - 12.f * smooth01(352.f, 420.f, x) + rough;
    }

    void paint_bridge() {
      for (int y = 0; y < scene_h; ++y) {
        for (int x = 0; x < scene_w; ++x) {
          f32_t fx = f32_t(x), fy = f32_t(y);
          f32_t top = deck_top(fx);
          if (fy < top) {
            continue;
          }
          f32_t ax = (fx - 272.f) / 92.f;
          f32_t ay = (fy - 286.f) / 80.f;
          f32_t radius = std::sqrt(ax * ax + ay * ay);
          if (radius < 1.f) {
            f32_t bx = (fx - 258.f) / 92.f;
            f32_t by = (fy - 298.f) / 80.f;
            if (std::sqrt(bx * bx + by * by) >= 1.f) {
              at(x, y) = surface({0.016f, 0.012f, 0.026f}, 0.f, 0.f, 0.f, kind_e::stone);
            }
            else {
              at(x, y).shade = mixf(0.3f, 1.f, clamp01((fy - 205.f) / 60.f));
            }
            continue;
          }
          if (x >= 348 && fy > 250.f + 12.f * value_noise(fx * 0.03f, 6.f)) {
            at(x, y).shade = 0.55f;
            continue;
          }
          if (radius < 1.17f) {
            f32_t angle = std::atan2(ay, ax);
            at(x, y) = block(x, y, angle * 78.f, (radius - 1.f) * 80.f, 13.f, 14.f, {0.2f, 0.135f, 0.2f}, 0.f);
            continue;
          }
          f32_t depth = fy - top;
          if (depth < 14.f) {
            at(x, y) = block(x, y, fx, depth, 19.f, 7.5f, {0.24f, 0.17f, 0.25f}, -0.75f);
          }
          else if (depth < 58.f) {
            at(x, y) = block(x, y, fx, depth, 24.f, 12.f, {0.17f, 0.115f, 0.19f}, 0.f);
          }
          else {
            at(x, y) = rubble(x, y, 13.f, rgb_t{0.14f, 0.095f, 0.16f});
          }
        }
      }
    }

    void paint_capsule(f32_t ax, f32_t ay, f32_t bx, f32_t by, f32_t ra, f32_t rb, const brush_t& brush, kind_e kind = kind_e::figure) {
      f32_t reach = std::max(ra, rb) + 1.f;
      int x0 = std::max(0, int(std::min(ax, bx) - reach));
      int x1 = std::min(scene_w - 1, int(std::max(ax, bx) + reach));
      int y0 = std::max(0, int(std::min(ay, by) - reach));
      int y1 = std::min(scene_h - 1, int(std::max(ay, by) + reach));
      f32_t bax = bx - ax, bay = by - ay;
      f32_t length2 = std::max(bax * bax + bay * bay, 1e-4f);
      for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
          f32_t px = x + 0.5f, py = y + 0.5f;
          f32_t t = clamp01(((px - ax) * bax + (py - ay) * bay) / length2);
          f32_t dx = px - (ax + bax * t);
          f32_t dy = py - (ay + bay * t);
          f32_t radius = mixf(ra, rb, t);
          f32_t distance = std::sqrt(dx * dx + dy * dy);
          if (distance > radius) {
            continue;
          }
          f32_t nx = dx / radius, ny = dy / radius;
          f32_t moon = 0.5f + 0.55f * clamp01(-0.5f * nx - 0.75f * ny);
          rgb_t color = brush.color * (std::floor(moon * 5.f) / 5.f);
          if (brush.bands && fract(t * brush.bands) < 0.16f) {
            color = color * 0.45f;
          }
          if (brush.folds != 0.f) {
            color = color * (1.f + brush.folds * std::sin(px * 0.55f + py * 0.22f));
          }
          if (radius - distance < 1.1f) {
            color = color * 0.28f;
          }
          at(x, y) = surface(color, nx, ny, brush.gloss, kind);
        }
      }
    }

    void paint_tree() {
      brush_t bark{{0.075f, 0.04f, 0.06f}, 0.f, 0, 0.35f};
      paint_capsule(9.f, -8.f, 13.f, 72.f, 12.f, 16.f, bark, kind_e::stone);
      paint_capsule(13.f, 72.f, 2.f, 138.f, 16.f, 25.f, bark, kind_e::stone);
      paint_capsule(14.f, 112.f, 38.f, 142.f, 9.f, 3.f, bark, kind_e::stone);
      paint_capsule(12.f, 34.f, 58.f, -6.f, 7.f, 3.f, bark, kind_e::stone);
    }

    void paint_knight() {
      brush_t steel{{0.3f, 0.3f, 0.38f}, 0.75f, 0};
      brush_t plated{{0.3f, 0.3f, 0.38f}, 0.75f, 5};
      brush_t dark{{0.16f, 0.16f, 0.22f}, 0.45f, 0};
      brush_t cloth{{0.62f, 0.11f, 0.09f}, 0.f, 0, 0.12f};
      brush_t cape{{0.12f, 0.21f, 0.31f}, 0.f, 0, 0.3f};
      brush_t plume{{0.8f, 0.12f, 0.08f}, 0.f, 0};

      paint_capsule(216.f, 84.f, 186.f, 128.f, 9.f, 20.f, cape);
      paint_capsule(190.f, 124.f, 170.f, 152.f, 20.f, 17.f, cape);

      paint_capsule(224.f, 120.f, 254.f, 142.f, 9.f, 7.5f, steel);
      paint_capsule(254.f, 142.f, 250.f, 176.f, 7.f, 5.5f, plated);
      paint_capsule(250.f, 178.f, 264.f, 187.f, 5.f, 4.f, dark);
      paint_capsule(254.f, 142.f, 254.f, 142.f, 7.f, 7.f, steel);

      paint_capsule(238.f, 92.f, 248.f, 114.f, 7.f, 6.5f, steel);
      paint_capsule(248.f, 114.f, 234.f, 132.f, 6.f, 5.f, plated);

      paint_capsule(226.f, 92.f, 212.f, 120.f, 16.f, 14.f, {{0.3f, 0.3f, 0.38f}, 0.75f, 4});
      paint_capsule(214.f, 112.f, 212.f, 124.f, 15.f, 14.f, cloth);
      paint_capsule(212.f, 122.f, 222.f, 130.f, 10.f, 7.f, cloth);

      paint_capsule(208.f, 122.f, 230.f, 146.f, 11.f, 8.5f, steel);
      paint_capsule(230.f, 146.f, 224.f, 182.f, 8.f, 6.f, plated);
      paint_capsule(224.f, 184.f, 238.f, 196.f, 6.f, 4.5f, dark);
      paint_capsule(230.f, 146.f, 230.f, 146.f, 7.5f, 7.5f, steel);

      paint_capsule(238.f, 88.f, 238.f, 88.f, 8.f, 8.f, steel);
      paint_capsule(228.f, 82.f, 228.f, 82.f, 9.f, 9.f, dark);
      paint_capsule(214.f, 86.f, 214.f, 86.f, 11.f, 11.f, steel);
      paint_capsule(212.f, 94.f, 218.f, 118.f, 7.5f, 7.f, steel);
      paint_capsule(218.f, 116.f, 218.f, 116.f, 7.5f, 7.5f, cloth);
      paint_capsule(218.f, 118.f, 234.f, 132.f, 6.5f, 5.f, plated);
      paint_capsule(234.f, 133.f, 234.f, 133.f, 6.f, 6.f, dark);

      paint_capsule(231.f, 66.f, 233.f, 76.f, 12.f, 11.f, steel);
      paint_capsule(226.f, 57.f, 238.f, 60.f, 2.f, 1.6f, dark);
      paint_capsule(236.f, 70.f, 244.f, 75.f, 1.6f, 1.4f, {{0.02f, 0.02f, 0.03f}});
      for (int i = 0; i < 3; ++i) {
        f32_t offset = f32_t(i) * 3.f;
        paint_capsule(235.f + offset, 78.f - offset * 0.2f, 235.f + offset, 80.f - offset * 0.2f, 0.7f, 0.7f, {{0.02f, 0.02f, 0.03f}});
      }

      f32_t last_x = 228.f, last_y = 58.f;
      for (int i = 1; i <= 24; ++i) {
        f32_t t = f32_t(i) / 24.f;
        f32_t x = 228.f - 36.f * t;
        f32_t y = 58.f - 9.f * std::sin(t * 3.14159f) + 8.f * t * t;
        paint_capsule(last_x, last_y, x, y, 3.8f * (1.f - (t - 0.04f) * 0.7f), 3.8f * (1.f - t * 0.7f), plume);
        last_x = x;
        last_y = y;
      }
    }

    void paint_lantern() {
      for (int y = 86; y < 137; ++y) {
        for (int x = 266; x < 295; ++x) {
          f32_t dx = std::abs(x - 280.f);
          f32_t dy = y - 93.f;
          f32_t ring = std::sqrt(dx * dx + dy * dy);
          bool frame_pixel = false;
          bool glass_pixel = false;
          if (y < 98) {
            frame_pixel = ring > 3.f && ring < 5.2f;
          }
          else if (y < 104) {
            frame_pixel = dx <= 4.f + (y - 98) * 1.7f;
          }
          else if (y < 130) {
            glass_pixel = dx < 8.f && y != 116;
            frame_pixel = dx <= 9.f && !glass_pixel;
          }
          else {
            frame_pixel = dx <= 11.f - (135 - y) * 0.4f;
          }
          if (glass_pixel) {
            at(x, y) = surface({1.f, 1.f, 1.f}, 0.f, 0.f, 0.f, kind_e::glass);
          }
          else if (frame_pixel) {
            at(x, y) = surface({0.24f, 0.13f, 0.09f}, (x - 280.f) / 11.f, 0.f, 0.5f, kind_e::figure);
          }
        }
      }
    }

    bool lantern_part(f32_t x, f32_t y) const {
      return x > 266.f && x < 295.f && y > 86.f && y < 137.f;
    }

    void bake_occlusion() {
      for (int y = 0; y < scene_h; ++y) {
        for (int x = 0; x < scene_w; ++x) {
          const pixel_t& p = pixels[y * scene_w + x];
          if (p.kind == kind_e::figure || p.kind == kind_e::glass) {
            continue;
          }
          bool stone_blocks = p.kind == kind_e::water || p.kind == kind_e::backdrop;
          for (int step = 1; step < 16; ++step) {
            f32_t t = f32_t(step) / 16.f;
            f32_t sx = mixf(f32_t(x), lantern_x, t);
            f32_t sy = mixf(f32_t(y), lantern_y, t);
            if (lantern_part(sx, sy)) {
              continue;
            }
            kind_e hit = pixels[int(sy) * scene_w + int(sx)].kind;
            if (hit == kind_e::figure || (stone_blocks && hit == kind_e::stone)) {
              occlusion[y * scene_w + x] = 0.16f;
              break;
            }
          }
        }
      }
    }

    void bake_vignette() {
      for (int y = 0; y < scene_h; ++y) {
        for (int x = 0; x < scene_w; ++x) {
          f32_t dx = (x - 270.f) / 300.f;
          f32_t dy = (y - 125.f) / 200.f;
          f32_t distance = clamp01(std::sqrt(dx * dx + dy * dy));
          vignette[y * scene_w + x] = 1.f - 0.62f * distance * distance;
        }
      }
    }

    rgb_t water_color(int x, int y, f32_t time, f32_t shade) const {
      f32_t fx = f32_t(x), fy = f32_t(y);
      f32_t perspective = clamp01((fy - 112.f) / 158.f);
      f32_t broad = value_noise(fx * 0.028f / (0.45f + perspective) + time * 0.22f, fy * (0.55f - 0.25f * perspective) - time * 0.08f);
      f32_t fine = value_noise(fx * 0.09f - time * 0.31f, fy * 0.9f + 40.f);
      f32_t wave = broad * 0.68f + fine * 0.32f;

      rgb_t color = mix_rgb({0.018f, 0.03f, 0.085f}, {0.05f, 0.06f, 0.19f}, wave);
      color = mix_rgb(color, {0.085f, 0.035f, 0.16f}, smooth01(0.55f, 1.f, perspective) * 0.7f);
      if (wave > 0.6f) {
        color = color + rgb_t{0.12f, 0.12f, 0.27f} * clamp01((wave - 0.6f) * 5.f);
      }
      if (wave > 0.7f) {
        color = rgb_t{0.3f, 0.29f, 0.55f};
      }
      color = color * shade;

      if (fy > 150.f) {
        f32_t glint = std::exp(-std::abs(fx - 286.f) / 34.f) * smooth01(0.58f, 0.68f, fine);
        color = color + rgb_t{0.75f, 0.22f, 0.5f} * (glint * flicker * 0.85f);
      }
      return color;
    }

    rgb_t relight(const pixel_t& p, rgb_t albedo, f32_t fx, f32_t fy, f32_t occluded) const {
      f32_t lx = lantern_x - fx;
      f32_t ly = lantern_y - fy;
      constexpr f32_t lz = 16.f;
      f32_t distance = std::sqrt(lx * lx + ly * ly + lz * lz);
      lx /= distance;
      ly /= distance;
      f32_t lz_n = lz / distance;

      f32_t diffuse = std::max(0.f, p.nx * lx + p.ny * ly + p.nz * lz_n);
      f32_t warm = std::exp(-distance / 24.f) * 1.3f;
      f32_t violet = std::exp(-distance / 75.f) * 0.75f;
      rgb_t light = rgb_t{1.f, 0.52f, 0.2f} * warm + rgb_t{0.85f, 0.2f, 0.6f} * violet;
      rgb_t color = albedo + albedo * light * (diffuse * occluded * flicker * 5.f);

      if (p.gloss > 0.f) {
        f32_t hx = lx, hy = ly, hz = lz_n + 1.f;
        f32_t h_length = std::sqrt(hx * hx + hy * hy + hz * hz);
        f32_t highlight = std::max(0.f, (p.nx * hx + p.ny * hy + p.nz * hz) / h_length);
        color = color + rgb_t{1.f, 0.7f, 0.42f} * (std::pow(highlight, 14.f) * p.gloss * occluded * flicker * (warm + violet));
      }
      return color;
    }

    void plot(int x, int y, rgb_t color, f32_t strength) {
      if (x < 0 || y < 0 || x >= scene_w || y >= scene_h) {
        return;
      }
      std::uint8_t* out = frame.data() + (y * scene_w + x) * 4;
      out[0] = std::uint8_t(std::min(255.f, out[0] + color.r * strength * 255.f));
      out[1] = std::uint8_t(std::min(255.f, out[1] + color.g * strength * 255.f));
      out[2] = std::uint8_t(std::min(255.f, out[2] + color.b * strength * 255.f));
    }

    void draw_motes(f32_t time) {
      for (int i = 0; i < 16; ++i) {
        f32_t seed = hash2(i, 3);
        f32_t base_x = 30.f + 420.f * hash2(i, 11);
        f32_t base_y = 40.f + 110.f * hash2(i, 23);
        int x = int(base_x + 16.f * std::sin(time * (0.21f + seed * 0.2f) + f32_t(i)));
        int y = int(base_y + 9.f * std::cos(time * (0.27f + seed * 0.2f) + f32_t(i) * 2.f));
        f32_t blink = std::pow(std::max(0.f, std::sin(time * (0.7f + seed) + seed * 40.f)), 4.f);
        rgb_t color{0.95f, 0.8f, 0.34f};
        plot(x, y, color, blink);
        plot(x + 1, y, color, blink * 0.35f);
        plot(x - 1, y, color, blink * 0.35f);
        plot(x, y + 1, color, blink * 0.35f);
        plot(x, y - 1, color, blink * 0.35f);
      }
      for (int i = 0; i < 7; ++i) {
        f32_t life = fract(time * 0.13f + f32_t(i) / 7.f);
        int x = int(lantern_x + 6.f * std::sin(life * 9.f + f32_t(i) * 1.7f) + (i - 3) * 1.5f);
        int y = int(96.f - life * 62.f);
        plot(x, y, {1.f, 0.5f, 0.2f}, (1.f - life) * (1.f - life));
      }
    }

    void composite(f32_t time) {
      int tick = int(time * tick_rate);
      flicker = 0.84f + 0.08f * std::sin(time * 7.3f) + 0.12f * value_noise(time * 3.1f, 0.5f) + 0.05f * hash2(tick, 19);

      for (int y = 0; y < scene_h; ++y) {
        for (int x = 0; x < scene_w; ++x) {
          int index = y * scene_w + x;
          const pixel_t& p = pixels[index];
          f32_t fx = f32_t(x), fy = f32_t(y);
          rgb_t color;

          if (p.kind == kind_e::glass) {
            f32_t dx = (fx - lantern_x) / 9.f;
            f32_t dy = (fy - lantern_y) / 14.f;
            f32_t falloff = clamp01(std::sqrt(dx * dx + dy * dy) + 0.14f * hash2(x, y + tick));
            color = mix_rgb({1.f, 0.92f, 0.55f}, {1.f, 0.42f, 0.1f}, falloff) * (0.72f + 0.32f * flicker);
          }
          else {
            rgb_t albedo = p.kind == kind_e::water ? water_color(x, y, time, p.shade) : p.albedo;
            color = relight(p, albedo, fx, fy, occlusion[index]);
            if ((p.kind == kind_e::water || p.kind == kind_e::backdrop) && fy > 92.f && fy < 172.f) {
              f32_t band = (fy - 118.f) / 26.f;
              f32_t mist = value_noise(fx * 0.012f + time * 0.06f, fy * 0.035f);
              mist += 0.5f * value_noise(fx * 0.03f - time * 0.09f, fy * 0.07f + 7.f);
              color = mix_rgb(color, {0.16f, 0.11f, 0.26f}, clamp01(std::exp(-band * band) * std::max(0.f, mist - 0.62f) * 0.95f));
            }
          }

          color = color * vignette[index];
          f32_t threshold = (bayer[(y & 3) * 4 + (x & 3)] + 0.5f) / 16.f;
          std::uint8_t* out = frame.data() + index * 4;
          out[0] = std::uint8_t(clamp01(std::floor(color.r * color_levels + threshold) / color_levels) * 255.f);
          out[1] = std::uint8_t(clamp01(std::floor(color.g * color_levels + threshold) / color_levels) * 255.f);
          out[2] = std::uint8_t(clamp01(std::floor(color.b * color_levels + threshold) / color_levels) * 255.f);
        }
      }
      draw_motes(time);
    }
  };
}

int main() {
  namespace lb = lantern_bridge;

  engine_t engine{{.window_size = fan::vec2(1440.f, 810.f)}};
  engine.set_window_name("lantern bridge");
  engine.set_settings({
    .clear_color = fan::color(0.012f, 0.008f, 0.024f),
    .bloom_strength = 0.06f,
    .bloom_threshold = 1.f,
    .bloom_knee = 0.1f,
    .bloom_filter_radius = 0.12f
  });

  lb::painting_t painting;
  painting.composite(0.f);

  // the frame is the same size on every reload, and passing the same load properties
  // keeps image_reload from recreating the gpu image on each tick
  image_load_properties_t load_properties = image_presets::pixel_art();
  fan::image::info_t info;
  info.data = painting.frame.data();
  info.size = fan::vec2i(lb::scene_w, lb::scene_h);
  info.channels = 4;
  image_t image = engine.image_load(info, load_properties);

  unlit_sprite_t canvas{{
    .position = fan::vec3(engine.viewport_get_size() / 2.f, 0.f),
    .size = fan::vec2(lb::scene_w, lb::scene_h) / 2.f,
    .image = image,
    .enable_culling = false
  }};

  fan::vec2 fitted_viewport = -1.f;
  fan::vec2 origin = 0.f;
  f32_t scale = 1.f;
  f32_t time = 0.f;
  f32_t accumulator = 0.f;

  engine.loop([&] {
    fan::vec2 viewport = engine.viewport_get_size();
    if (viewport.x != fitted_viewport.x || viewport.y != fitted_viewport.y) {
      fan::vec2 canvas_size = fan::vec2(lb::scene_w, lb::scene_h);
      scale = std::min(viewport.x / canvas_size.x, viewport.y / canvas_size.y);
      origin = viewport / 2.f - canvas_size * scale / 2.f;
      canvas.set_position(fan::vec3(viewport / 2.f, 0.f));
      canvas.set_size(canvas_size * scale / 2.f);
      fitted_viewport = viewport;
    }

    f32_t delta = f32_t(engine.get_delta_time());
    time += delta;
    accumulator += delta;
    if (accumulator >= 1.f / lb::tick_rate) {
      accumulator = std::fmod(accumulator, 1.f / lb::tick_rate);
      painting.composite(time);
      engine.image_reload(image, info, load_properties);
    }

    // brighter than 1.0, so this is what the engine's bloom pass picks up
    fan::vec2 flame = origin + fan::vec2(lb::lantern_x, lb::lantern_y) * scale;
    f32_t heat = painting.flicker;
    circle(fan::vec3(flame, 1.f), 3.f * scale, fan::color(2.8f * heat, 1.35f * heat, 0.4f * heat, 1.f));
  });
}
