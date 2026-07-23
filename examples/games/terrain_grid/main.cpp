import std;
import fan;

using namespace fan::graphics;

fan::noise_t hill_noise{42};
fan::noise_t cave_noise{137};
fan::noise_t detail_noise{999};

static image_t gen_texture(int seed, fan::color c1, fan::color c2, fan::color c3, f32_t freq, f32_t warp_str) {
  fan::noise_t n_fbm{seed};
  fan::noise_t n_cell{seed + 1337};
  fan::noise_t n_warp{seed + 42};
  int s = 128;
  f32_t scale = 32.f / s;
  std::vector<std::uint8_t> pixels(s * s * 4);
  for (int y = 0; y < s; ++y) {
    for (int x = 0; x < s; ++x) {
      f32_t nx = x * freq * scale, ny = y * freq * scale;
      f32_t sx = (x - s) * freq * scale, sy = (y - s) * freq * scale;

      f32_t u = fan::noise_t::fade(static_cast<f32_t>(x) / s);
      f32_t v_fade = fan::noise_t::fade(static_cast<f32_t>(y) / s);

      auto sample = [&](f32_t px, f32_t py) {
        f32_t wx = n_warp.sample_norm(fan::noise_t::base_t::open_simplex2, fan::noise_t::fractal_t::fbm, px, py, 4, 2.f, 0.5f) * warp_str;
        f32_t wy = n_warp.sample_norm(fan::noise_t::base_t::open_simplex2, fan::noise_t::fractal_t::fbm, px + 5.3f, py + 1.1f, 4, 2.f, 0.5f) * warp_str;
        
        f32_t base = n_fbm.sample_norm(fan::noise_t::base_t::open_simplex2, fan::noise_t::fractal_t::fbm, px + wx, py + wy, 6, 2.f, 0.5f);
        f32_t grit = n_cell.cellular_norm((px + wx) * 3.f, (py + wy) * 3.f);
        return std::lerp(base, grit, 0.25f);
      };

      f32_t v = std::lerp(
        std::lerp(sample(nx, ny), sample(sx, ny), u),
        std::lerp(sample(nx, sy), sample(sx, sy), u),
        v_fade
      );

      int i = (y * s + x) * 4;
      fan::color c = v < 0.5f ? c1.lerp(c2, v * 2.f) : c2.lerp(c3, (v - 0.5f) * 2.f);
      
      pixels[i + 0] = static_cast<std::uint8_t>(c.r * 255.f);
      pixels[i + 1] = static_cast<std::uint8_t>(c.g * 255.f);
      pixels[i + 2] = static_cast<std::uint8_t>(c.b * 255.f);
      pixels[i + 3] = 255;
    }
  }
  fan::image::info_t ii;
  ii.data = pixels.data();
  ii.size = fan::vec2ui(s, s);
  ii.channels = 4;
  return image_load(ii, image_presets::pixel_art());
}

int main() {
  engine_t engine;
  interactive_camera_t ic;
  engine.get_clear_color() = fan::color(0.4f, 0.6f, 0.85f, 1.f);
  engine.update_physics(true);

  auto img_grass      = gen_texture(101, fan::color(0.10f, 0.35f, 0.05f), fan::color(0.15f, 0.42f, 0.08f), fan::color(0.28f, 0.58f, 0.18f), 0.08f, 0.8f);
  auto img_dirt       = gen_texture(202, fan::color(0.25f, 0.15f, 0.08f), fan::color(0.35f, 0.22f, 0.10f), fan::color(0.45f, 0.30f, 0.15f), 0.10f, 1.5f);
  auto img_stone      = gen_texture(303, fan::color(0.12f, 0.14f, 0.18f), fan::color(0.18f, 0.20f, 0.24f), fan::color(0.28f, 0.30f, 0.35f), 0.25f, 0.3f);
  auto img_deep_stone = gen_texture(404, fan::color(0.05f, 0.06f, 0.08f), fan::color(0.08f, 0.10f, 0.12f), fan::color(0.12f, 0.14f, 0.18f), 0.35f, 0.1f);
  auto img_dark_grass = gen_texture(505, fan::color(0.08f, 0.25f, 0.03f), fan::color(0.10f, 0.30f, 0.05f), fan::color(0.12f, 0.35f, 0.06f), 0.12f, 0.6f);
  auto img_rock       = gen_texture(606, fan::color(0.35f, 0.32f, 0.28f), fan::color(0.45f, 0.42f, 0.38f), fan::color(0.52f, 0.48f, 0.42f), 0.15f, 0.4f);
  auto img_island     = gen_texture(707, fan::color(0.20f, 0.24f, 0.15f), fan::color(0.25f, 0.32f, 0.18f), fan::color(0.32f, 0.40f, 0.22f), 0.10f, 0.7f);

  algorithm::chunk_renderer_t terrain{{
    .cell_size = 16.f,
    .chunk_size = 32,
    .hill_noise = &hill_noise,
    .cave_noise = &cave_noise,
    .detail_noise = &detail_noise,
    .surface_base = -10.f,
    .sky_island_noise = &hill_noise,
    .sky_island_freq = 0.04f,
    .sky_island_threshold = 0.6f,
    .sky_island_min = 80.f,
    .sky_island_max = 150.f,
    .img_sky_island = img_island,
    .tile_layers = {
      {2.f, img_grass}, 
      {12.f, img_dirt}, 
      {45.f, img_stone}, 
      {std::numeric_limits<f32_t>::max(), img_deep_stone}
    },
    .scatter_noise = &detail_noise,
    .scatter_img = img_dark_grass,
    .scatter_threshold = 0.6f,
  }};

  physics::character2d_t player = physics::character_sprite({
    //.center0 = {0.f, -24.f},
    //.center1 = {0.f, 24.f},
    //.radius = 12.f,
    .size=12.f,
    //.shape_properties{.restitution=0.f}
  });
  auto& pctx = engine.get_physics_context();
  pctx.set_gravity(pctx.get_gravity() / 1.5f);
  player.set_mass(100.f);
  player.enable_default_movement(300.f, 200.f);

  engine.camera_set_target(player, 10.f);

  f32_t dig_radius = 12.f;
  fan::time::interval_t dig_interval{0.003f};

  engine.loop([&] {
    f64_t dt = engine.get_delta_time();
    fan::vec2 player_pos = player.get_position();

    if (fan::window::is_key_clicked(fan::key_r)) {
      player.set_physics_position({player_pos.x, 0});
    }

    terrain.stream(player_pos, engine.ws());

    fan::vec2 mouse_pos = engine.get_mouse_position();
    fan::vec2 hit_pos = terrain.raycast(player_pos, mouse_pos, dig_radius);

    line(player_pos, hit_pos, fan::color(1.f, 0.f, 0.f, 0.3f), dig_radius);
    circle(hit_pos, dig_radius, fan::color(1.f, 1.f, 0.f, 0.2f));
    circle(hit_pos, dig_radius * 0.15f, fan::colors::red);

    if (fan::window::is_mouse_down(fan::mouse_left) && dig_interval.tick(dt)) {
      terrain.dig(hit_pos, dig_radius);
    }
  });

  return 0;
}
