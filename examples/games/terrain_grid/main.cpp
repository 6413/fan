import std;
import fan;

using namespace fan::graphics;

auto seed = (int)std::chrono::steady_clock::now().time_since_epoch().count();
fan::noise_t hill_noise{seed};
fan::noise_t cave_noise{seed + 1};
fan::noise_t detail_noise{seed + 2};

int main() {
  engine_t engine;
  interactive_camera_t ic;
  engine.get_clear_color() = fan::colors::black;
  engine.update_physics(true);

  gradient_t bg_sky{fan::color(0.2f, 0.4f, 0.75f, 1.f)*.2, fan::color(0.6f, 0.75f, 0.9f, 1.f)*.2, fan::vec3(1.f), engine.whs()};
  sprite_t bg_below{{
    .position=fan::vec3(0, 0, 0.f),
    .size = engine.whs(),
    .image = "fossil_cave.jpg"
  }};

  auto pa = image_presets::pixel_art();
  image_t img_grass     {"images/Textures/Grass/cubeGreen_1.png",   pa};
  image_t img_dirt      {"images/Textures/Dirt/cubeDirt_1.png",     pa};
  image_t img_stone     {"images/Textures/Stone/cubeStone_1.png",   pa};
  image_t img_bedrock   {"images/Textures/Stone/cubeBedrock_1.png", pa};
  image_t img_island    {"images/Textures/Grass/cubeGreen_1.png",   pa};
  image_t img_dark_grass{"images/Textures/Grass/cubeGreen_2.png",   pa};

  algorithm::chunk_renderer_t terrain{{
    .cell_size = 16.f,
    .chunk_size = 32,
    .hill_noise = &hill_noise,
    .cave_noise = &cave_noise,
    .detail_noise = &detail_noise,
    .surface_base = -10.f,
    .img_sky_island = img_island,
    .tile_layers = {
      {3.f, img_grass},
      {12.f, img_dirt},
      {50.f, img_stone},
      {std::numeric_limits<f32_t>::max(), img_bedrock}
    },
    .scatter_noise = &detail_noise,
    .scatter_img = img_dark_grass,
    .scatter_threshold = 0.6f,
  }};

  auto player = physics::character_sprite({
    .position{0, 0, 6.f},
    .size=12.f,
    .image = "images/duck.webp"
  });
  auto& pctx = engine.get_physics_context();
  pctx.set_gravity(pctx.get_gravity() / 1.5f);
  player.set_mass(100.f);
  player.enable_default_movement(300.f, 200.f);

  engine.camera_set_target(player, 10.f);

  auto& lighting = fan::graphics::get_lighting();
  //lighting.ambient = fan::vec3(0.35f, 0.45f, 0.55f);

  auto sun = light_t(light_properties_t{
    .position = fan::vec3(0, 200.f, 5.f),
    .size = fan::vec2(900),
    .color = fan::color(1.f, 0.95f, 0.85f, 0.5f),
  });
  auto torch = light_t(light_properties_t{
    .position = fan::vec3(0, 0, 10.f),
    .size = fan::vec2(160),
    .color = fan::color(1.f, 0.7f, 0.3f, 0.9f),
  });
  light_t* lights[] = {&sun, &torch};

  struct shadow_slot_t { shadow_t instance; bool visible = true; };
  std::vector<shadow_slot_t> shadow_pool;

  fan::spatial::world_t<int> occluder_grid;
  occluder_grid.init(fan::vec2(-5000), fan::vec2(512), fan::vec2i(40, 40));
  int next_occluder_id = 0;
  std::vector<algorithm::chunk_renderer_t::occluder_rect_t> occluders;

  f32_t dig_radius = 64.f;
  fan::time::interval_t dig_interval{0.003f};

  gpu_particle_system_t<> dig_particles;
  image_t particle_img{fan::colors::white};
  struct break_effect_t { circle_t circle; f32_t timer = 0.f; };
  std::vector<break_effect_t> break_effects;

  engine.loop([&] {
    f64_t dt = engine.get_delta_time();
    fan::vec2 player_pos = player.get_position();
    fan::vec2 cam_center = ic.get_center();

    f32_t ground_y = 512.f;
    bg_sky.set_position(fan::vec2(cam_center.x, std::min(cam_center.y, ground_y)));
    bg_below.set_position(cam_center);

    f32_t surface_h = terrain.surface_height_at(player_pos.x);
    f32_t depth = std::max(0.f, surface_h - player_pos.y);
    f32_t cave_factor = std::min(depth / 40.f, 1.f);

    //lighting.ambient = fan::vec3(0.35f, 0.45f, 0.55f).lerp(fan::vec3(0.04f, 0.04f, 0.05f), cave_factor);

    sun.set_position(fan::vec3(cam_center.x, 200.f, 5.f));
    sun.set_color(fan::color(1.f, 0.95f, 0.85f, 0.5f).lerp(fan::color(1.f, 0.95f, 0.85f, 0.f), cave_factor));
    torch.set_position(fan::vec3(player_pos, 10.f));
    torch.set_color(fan::color(1.f, 0.7f, 0.3f, 0.9f).lerp(fan::color(1.f, 0.7f, 0.3f, 1.2f), cave_factor));

    fan::vec2 view_half = engine.whs();
    f32_t max_light_radius = 900.f;
    fan::vec2 region_min = cam_center - view_half - max_light_radius;
    fan::vec2 region_max = cam_center + view_half + max_light_radius;
    occluders = terrain.build_occluders(region_min, region_max);

    while (next_occluder_id > 0) {
      occluder_grid.remove(--next_occluder_id);
    }
    for (auto& occ : occluders) {
      occluder_grid.upsert(next_occluder_id, {occ.center - occ.half_size, occ.center + occ.half_size}, fan::spatial::movement_static);
      ++next_occluder_id;
    }

    for (auto& slot : shadow_pool) { slot.visible = false; }
    std::size_t pool_idx = 0;

    for (auto* light : lights) {
      fan::vec3 lpos = light->get_position();
      f32_t lradius = light->get_size().x;
      fan::color lcolor = light->get_color();
      if (lcolor.a <= 0.01f) continue;

      f32_t shadow_darkness = 0.6f * lcolor.a * (1.f - cave_factor);
      fan::color shadow_color = fan::color(0.f, 0.f, 0.f, shadow_darkness);

      occluder_grid.query_radius(lpos, lradius, [&](int id) {
        auto& aabb = occluder_grid.registry.aabb_cache[id];
        fan::vec2 center = (aabb.min + aabb.max) * 0.5f;
        fan::vec2 half = (aabb.max - aabb.min) * 0.5f + 0.5f;

        if (pool_idx >= shadow_pool.size()) {
          shadow_pool.push_back({shadow_t{shadow_properties_t{
            .position = fan::vec3(center, 25.f),
            .size = half,
            .color = shadow_color,
            .light_position = lpos,
            .light_radius = lradius,
          }}, true});
        } else {
          auto& sh = shadow_pool[pool_idx].instance;
          sh.set_position(fan::vec3(center, 25.f));
          sh.set_size(half);
          sh.set_color(shadow_color);
          sh.set_light_position(lpos);
          sh.set_light_radius(lradius);
        }
        shadow_pool[pool_idx].visible = true;
        ++pool_idx;
      });
    }

    for (auto& slot : shadow_pool) {
      slot.instance.set_visible(slot.visible);
    }

    if (fan::window::is_key_clicked(fan::key_r)) {
      player.set_physics_position({player_pos.x, 0});
    }

    terrain.stream(player_pos, engine.ws());

    fan::vec2 mouse_pos = engine.get_mouse_position();
    fan::vec2 hit_pos = terrain.raycast(player_pos, mouse_pos, dig_radius);

    if (!fan::graphics::gui::want_io() && fan::window::is_mouse_down(fan::mouse_left) && dig_interval.tick(dt)) {
      terrain.dig(hit_pos, dig_radius);

      break_effects.push_back({
        .circle = circle_t(circle_properties_t{
          .position = fan::vec3(hit_pos, 26.f),
          .radius = dig_radius * 0.5f,
          .color = fan::color(1.f, 1.f, 1.f, 0.6f),
          .outline_color = fan::color(1.f, 0.9f, 0.5f, 0.4f),
          .outline_width = 2.f,
        }),
        .timer = 0.25f,
      });

      dig_particles.spawn([&](auto& p) {
        p.loop = false;
        p.position = fan::vec3(hit_pos, 26.f);
        p.begin_color = fan::color(0.5f, 0.4f, 0.3f, 1.f);
        p.end_color = fan::color(0.3f, 0.2f, 0.1f, 0.f);
        p.start_size = fan::vec2(5);
        p.end_size = fan::vec2(1);
        p.alive_time = 0.35f;
        p.count = 14;
        p.start_velocity = fan::vec2(70, -90);
        p.end_velocity = fan::vec2(25, 50);
        p.start_spread = fan::vec2(70);
        p.end_spread = fan::vec2(110);
        p.expansion_power = 0.5f;
        p.shape = fan::graphics::shapes::particles_t::shapes_e::circle;
        p.image = particle_img;
      });
    }

    std::erase_if(break_effects, [&](auto& effect) {
    effect.timer -= dt;
    if (effect.timer <= 0.f) { return true; }
    f32_t t = effect.timer / 0.25f;
    effect.circle.set_radius(dig_radius * 0.5f + (1.f - t) * dig_radius * 1.5f);
    effect.circle.set_color(fan::color(1.f, 1.f, 1.f, t * 0.6f));
    effect.circle.set_outline_color(fan::color(1.f, 0.9f, 0.5f, t * 0.4f));
    return false;
  });

    dig_particles.update(dt);
  });

  return 0;
}
