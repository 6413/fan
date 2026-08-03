import std;
import fan;
import fan.graphics.gui.hotbar;
import fan.graphics.gui.inventory;
import fan.graphics.gameplay.types;

using namespace fan::graphics;

namespace {
  struct block_def_t {
    const char* name;
    const char* image_path;
    f32_t depth;
  };
  constexpr block_def_t block_defs[] = {
    {"Grass",      "images/Textures/Grass/cubeGreen_1.png",    3.f},
    {"Dirt",       "images/Textures/Dirt/cubeDirt_1.png",      30.f},
    {"Clay",       "images/Textures/Purple/cubePurple_1.png",  100.f},
    {"Sandstone",  "images/Textures/Desert/cubeDesert_1.png",  220.f},
    {"Stone",      "images/Textures/Stone/cubeStone_1.png",    550.f},
    {"Deep Stone", "images/Textures/Stone/cubeStone_5.png",    1000.f},
    {"Bedrock",    "images/Textures/Stone/cubeBedrock_1.png",  std::numeric_limits<f32_t>::max()},
  };
  constexpr int block_count = (int)std::size(block_defs);
}

int main() {
  engine_t engine;
  interactive_camera_t ic;
  ic.ignore_input = true;
  ic.set_zoom(5.f);
  engine.get_clear_color() = fan::colors::black;
  engine.update_physics(true);

  engine.shadow_enable_tile_mode();
  engine.shadow_set_darkness(0.6f);

  gameplay::gui_theme_t theme{
    .panel_bg = fan::color(0.08f, 0.08f, 0.10f, 0.90f),
    .panel_border = fan::color(0.5f, 0.5f, 0.6f, 1.f),
    .panel_corner_accent = fan::color(0.6f, 0.65f, 0.7f, 1.f),
    .slot_bg = fan::color(0.15f, 0.15f, 0.17f, 0.95f),
    .slot_bg_hover = fan::color(0.30f, 0.32f, 0.36f, 0.95f),
    .slot_border = fan::color(0.4f, 0.4f, 0.45f, 1.f),
    .selected_border_color = fan::colors::white
  };

  gui::inventory_t inventory_ui;
  inventory_ui.create(block_count, block_count);
  inventory_ui.visible = false;
  inventory_ui.style.theme = theme;

  gui::hotbar_t hotbar;
  hotbar.create(block_count);
  inventory_ui.secondary = &hotbar;

  const char* save_path = "world_save.json";
  int world_seed = (int)std::chrono::steady_clock::now().time_since_epoch().count();
  std::vector<algorithm::chunk_renderer_t::edit_t> world_edits;
  try { 
    auto j = fan::json::load_file(save_path);
    world_seed = j.value("seed", world_seed);
    for (auto& e : j["edits"]) {
      world_edits.push_back({e["gx"].get<int>(), e["gy"].get<int>(), e["type"].get<int>()});
    }
    for (auto& s : j["inventory"]) {
      hotbar.add_item(
        fan::graphics::gameplay::items::create((std::uint32_t)s["id"].get<int>()),
        (std::uint32_t)s["count"].get<int>());
    }
  } catch (...) {}

  fan::noise_t hill_noise{world_seed};
  fan::noise_t cave_noise{world_seed + 1};
  fan::noise_t detail_noise{world_seed + 2};

  gradient_t bg_sky{fan::color(0.2f, 0.4f, 0.75f, 1.f)*.2, fan::color(0.6f, 0.75f, 0.9f, 1.f)*.2, fan::vec3(1.f), engine.whs()};
  unlit_sprite_t bg_below{{
    .position=fan::vec3(0, 0, 0.f),
    .size = engine.whs(),
    .image = "fossil_cave.jpg"
  }};

  auto pa = image_presets::pixel_art();
  image_t img_island    {"images/Textures/Grass/cubeGreen_1.png",   pa};
  image_t img_dark_grass{"images/Textures/Grass/cubeGreen_2.png",   pa};

  auto& item_reg = fan::graphics::gameplay::items::get_registry();
  std::vector<std::pair<f32_t, image_t>> tile_layers;
  tile_layers.reserve(block_count);
  for (int i = 0; i < block_count; ++i) {
    image_t img{block_defs[i].image_path, pa};
    tile_layers.push_back({block_defs[i].depth, img});
    item_reg.register_item({
      .id = (std::uint32_t)i,
      .name = block_defs[i].name,
      .icon = img,
      .max_stack = 999,
      .description = block_defs[i].name,
    });
  }

  algorithm::chunk_renderer_t terrain{{
    .cell_size = 16.f,
    .chunk_size = 32,
    .hill_noise = &hill_noise,
    .cave_noise = &cave_noise,
    .detail_noise = &detail_noise,
    .surface_base = -10.f,
    .img_sky_island = img_island,
    .tile_layers = tile_layers,
    .scatter_noise = &detail_noise,
    .scatter_img = img_dark_grass,
    .scatter_threshold = 0.6f,
  }};
  terrain.load_edits(world_edits);

  auto player = physics::from_json({
    .json_path = "models/Base Character/base_character.json"
  });
  player.play_sprite_sheet("idle");
  player.setup_attack_properties({
    .max_health = 100.f,
    .health = 100.f,
    .damage = 0.f,
  });
  auto player_aabb = player.get_aabb();
  fan::vec2 spawn_position{
    0.f,
    std::ceil(terrain.surface_height_at(0.f)) * terrain.cell_size() -
      (player_aabb.max.y - player_aabb.min.y) / 2.f
  };
  player.set_physics_position(spawn_position);
  terrain.stream(spawn_position, engine.ws());
  auto& pctx = engine.get_physics_context();
  pctx.set_gravity(pctx.get_gravity() / 2.5f);
  player.set_mass(100.f);
  player.enable_default_movement(70.f, 120.f);

  engine.camera_set_target(player, 10.f);

  const f32_t cs = terrain.cell_size();
  constexpr f32_t objective_depth_blocks = 20.f;
  constexpr f32_t hazard_depth_blocks = 10.f;
  const f32_t hazard_x = spawn_position.x + cs * 3.f;
  const f32_t hazard_surface_y = std::ceil(terrain.surface_height_at(hazard_x)) * cs;
  const fan::vec2 hazard_position{hazard_x, hazard_surface_y + cs * hazard_depth_blocks};
  rectangle_t hazard_zone{{
    .position = fan::vec3(hazard_position, 21.f),
    .size = fan::vec2(cs * 1.5f, cs * 0.5f),
    .color = fan::color(0.8f, 0.08f, 0.02f, 0.75f),
    .outline_color = fan::color(1.f, 0.65f, 0.1f, 1.f),
  }};
  const std::uint32_t layout_seed = (std::uint32_t)world_seed;
  const f32_t core_distance_blocks = 24.f + (layout_seed % 25u);
  const f32_t core_direction = (layout_seed & 1u) ? 1.f : -1.f;
  const f32_t core_x = spawn_position.x + core_direction * cs * core_distance_blocks;
  const int core_gx = (int)std::round(core_x / cs);
  const int core_surface_gy = (int)std::ceil(terrain.surface_height_at(core_x));
  const int core_gy = core_surface_gy + (int)objective_depth_blocks;
  const fan::vec2 core_position = (fan::vec2(core_gx, core_gy) + 0.5f) * cs;
  // Keep the objective deep, but make the generated objective room reachable.
  terrain.dig(core_position, cs * 0.8f);
  circle_t deep_core{{
    .position = fan::vec3(core_position, 22.f),
    .radius = cs * 0.55f,
    .color = fan::color(0.2f, 0.9f, 1.f, 0.9f),
    .outline_color = fan::colors::white,
    .outline_width = 2.f,
  }};
  circle_t core_pickup_ring{{
    .position = fan::vec3(core_position, 23.f),
    .radius = cs,
    .color = fan::colors::transparent,
    .outline_color = fan::colors::transparent,
    .outline_width = 3.f,
  }};

  auto& lighting = fan::graphics::get_lighting();
  lighting.ambient = fan::vec3(.6f/255.);

  const f32_t dig_radius = cs * 0.25f;
  const f32_t dig_reach = cs * 5.f;
  fan::time::interval_t dig_interval{0.12f};
  fan::time::interval_t save_interval{10.f};
  fan::time::interval_t hazard_damage_interval{0.75f};
  f32_t hazard_pulse = 0.f;
  f32_t core_pulse = 0.f;
  constexpr f32_t core_pickup_duration = 0.7f;
  f32_t core_pickup_timer = 0.f;
  f32_t damage_flash = 0.f;
  f32_t feedback_timer = 0.f;
  f32_t feedback_duration = 0.f;
  f32_t feedback_strength = 0.f;
  fan::color feedback_color = fan::colors::transparent;
  bool core_collected = false;
  bool game_over = false;
  bool game_won = false;

  gpu_particle_system_t<> dig_particles;
  struct break_effect_t { circle_t circle; f32_t timer = 0.f; };
  std::vector<break_effect_t> break_effects;

  auto trigger_feedback = [&](const fan::color& color, f32_t duration, f32_t strength) {
    feedback_color = color;
    feedback_duration = duration;
    feedback_strength = strength;
    feedback_timer = duration;
  };

  auto reset_run = [&]() {
    player.set_physics_position(spawn_position);
    player.set_linear_velocity(fan::vec2(0.f));
    player.reset_health();
    player.movement_state.ignore_input = false;
    hazard_damage_interval.reset();
    damage_flash = 0.f;
    core_pickup_timer = 0.f;
    feedback_timer = 0.f;
    core_collected = false;
    game_over = false;
    game_won = false;
  };

  fan::color torch_color_hud{1.f, 1.f, 0.8f, 1.f};
  f32_t torch_size_hud = 600.f;
  f32_t torch_shadow_a_hud = 1.0f;
  f32_t torch_shadow_strength = 1.0f;
  f32_t torch_shadow_softness = 0.956f;
  f32_t torch_visual_a_hud = 0.200f;
  f32_t sun_shadow_softness = 0.956f;

  auto save_world = [&]() {
    fan::json j = fan::json::object();
    j.set("seed", world_seed);
    fan::json edits = fan::json::array();
    for (auto& e : terrain.get_edits()) {
      edits.push_back(fan::json{{"gx", e.gx}, {"gy", e.gy}, {"type", e.type}});
    }
    j.set("edits", edits);
    fan::json inv = fan::json::array();
    for (auto& s : hotbar.slots) {
      if (s.is_empty()) { continue; }
      inv.push_back(fan::json{{"id", (int)*s.id}, {"count", (int)*s.stack_size}});
    }
    j.set("inventory", inv);
    j.save(save_path);
  };
  engine.get_window().add_close_callback([&](const fan::window_t::close_data_t&) {
    save_world();
  });

  auto cell_of = [&](fan::vec2 p) -> fan::vec2i {
    return {(int)std::floor(p.x / cs), (int)std::floor(p.y / cs)};
  };
  auto cell_center = [&](int gx, int gy) -> fan::vec2 {
    return (fan::vec2((f32_t)gx, (f32_t)gy) + 0.5f) * cs;
  };
  auto try_place = [&](fan::vec2 hit_pos) -> bool {
    auto& sel = hotbar.slots[hotbar.selected_slot];
    if (sel.is_empty()) { return false; }
    int gx, gy;
    fan::vec2 mouse_pos = engine.get_mouse_position();
    if (hit_pos == mouse_pos) {
      auto c = cell_of(mouse_pos);
      if (terrain.get_solid(c.x, c.y)) { return false; }
      gx = c.x; gy = c.y;
    }
    else {
      fan::vec2 diff = mouse_pos - player.get_position();
      f32_t dist = diff.length();
      if (dist < 1e-4f) { return false; }
      fan::vec2 dir = diff / dist;
      bool found = false;
      for (f32_t d = cs * 0.5f; d <= cs * 3.f; d += cs * 0.5f) {
        fan::vec2 p = hit_pos - dir * d;
        auto c = cell_of(p);
        if (!terrain.get_solid(c.x, c.y)) { gx = c.x; gy = c.y; found = true; break; }
      }
      if (!found) { return false; }
    }
    fan::vec2 cc = cell_center(gx, gy);
    fan::physics::aabb_t aabb = player.get_aabb();
    if (cc.x + cs * 0.5f > aabb.min.x && cc.x - cs * 0.5f < aabb.max.x &&
        cc.y + cs * 0.5f > aabb.min.y && cc.y - cs * 0.5f < aabb.max.y) { return false; }
    int type = (int)*sel.id;
    if (!hotbar.consume_slot(hotbar.selected_slot, nullptr)) { return false; }
    terrain.place(cc, cs * 0.5f, type);
    return true;
  };

  engine.loop([&] {
    f64_t dt = engine.get_delta_time();
    feedback_timer = std::max(0.f, feedback_timer - (f32_t)dt);
    core_pickup_timer = std::max(0.f, core_pickup_timer - (f32_t)dt);
    if (fan::window::is_key_clicked(fan::key_r)) {
      if (game_over || game_won) {
        reset_run();
      }
      else {
        fan::vec2 reset_position = player.get_position();
        player.set_physics_position({reset_position.x, 0.f});
      }
    }
    bool game_locked = game_over || game_won;
    player.movement_state.ignore_input = game_locked;
    player.update_animations();
    fan::vec2 player_pos = player.get_position();
    fan::vec2 cam_center = ic.get_center();

    f32_t ground_y = 512.f;
    bg_sky.set_position(fan::vec2(cam_center.x, std::min(cam_center.y, ground_y)));
    bg_below.set_position(cam_center);
    bg_below.set_color(lighting.ambient);

    f32_t surface_h = terrain.surface_height_at(player_pos.x);
    f32_t depth_blocks = std::max(0.f, player_pos.y / cs - surface_h);
    f32_t cave_factor = std::min(depth_blocks / 40.f, 1.f);
    f32_t core_distance_from_player_blocks = std::abs(core_position.x - player_pos.x) / cs;

    core_pulse += (f32_t)dt;
    bool player_at_core = deep_core.get_aabb().intersects(player.get_aabb());
    if (!game_locked && !core_collected && player_at_core) {
      core_collected = true;
      core_pickup_timer = core_pickup_duration;
      trigger_feedback(fan::color(0.1f, 0.8f, 1.f, 1.f), 0.35f, 0.24f);
    }
    if (core_collected && !game_won && depth_blocks <= 1.f) {
      game_won = true;
      trigger_feedback(fan::color(0.1f, 1.f, 0.3f, 1.f), 0.6f, 0.28f);
    }
    game_locked = game_over || game_won;
    player.movement_state.ignore_input = game_locked;

    f32_t core_alpha = 0.65f + 0.2f * std::sin(core_pulse * 3.f);
    deep_core.set_color(core_collected ? fan::colors::transparent : fan::color(0.2f, 0.9f, 1.f, core_alpha));
    deep_core.set_outline_color(core_collected ? fan::colors::transparent : fan::colors::white);
    if (core_pickup_timer > 0.f) {
      f32_t progress = 1.f - core_pickup_timer / core_pickup_duration;
      core_pickup_ring.set_radius(cs * (0.65f + progress * 2.5f));
      core_pickup_ring.set_outline_color(fan::color(0.1f, 0.85f, 1.f, (1.f - progress) * 0.9f));
    }
    else {
      core_pickup_ring.set_outline_color(fan::colors::transparent);
    }

    hazard_pulse += (f32_t)dt;
    damage_flash = std::max(0.f, damage_flash - (f32_t)dt);
    bool player_in_hazard = hazard_zone.get_aabb().intersects(player.get_aabb());
    if (!game_locked && player_in_hazard) {
      if (hazard_damage_interval.tick((f32_t)dt)) {
        player.set_health(std::max(0.f, player.get_health() - 15.f));
        damage_flash = 0.2f;
        trigger_feedback(fan::color(1.f, 0.05f, 0.02f, 1.f), 0.18f, 0.2f);
        if (player.is_dead()) {
          game_over = true;
        }
      }
    }
    else if (!player_in_hazard) {
      hazard_damage_interval.reset();
    }
    f32_t hazard_alpha = 0.55f + 0.15f * std::sin(hazard_pulse * 4.f);
    if (damage_flash > 0.f) {
      hazard_alpha = 1.f;
    }
    hazard_zone.set_color(fan::color(0.8f, 0.08f, 0.02f, hazard_alpha));

    //lighting.ambient = fan::vec3(0.35f, 0.45f, 0.55f).lerp(fan::vec3(0.04f, 0.04f, 0.05f), cave_factor);

    fan::vec2 sun_lpos = fan::vec2(cam_center.x, 200.f);
    fan::vec2 torch_lpos = player_pos;
    fan::vec2 torch_shadow_lpos = player_pos.offset_y(-15.f);

    f32_t sun_visual_a = 0.5f * (1.f - cave_factor);
    f32_t sun_shadow_a = 0.4f * (1.f - cave_factor);
    f32_t torch_visual_a = torch_visual_a_hud * (1.f + cave_factor);
    f32_t torch_shadow_a = torch_shadow_a_hud * torch_shadow_strength * (1.f + cave_factor);

    fan::color sun_col(1.f, 0.95f, 0.85f, 1.f);

    engine.shadow_clear_lights();
    engine.shadow_add_light(sun_lpos, 90.f, 1.f, sun_shadow_softness);
    engine.shadow_add_light(torch_shadow_lpos, torch_size_hud, torch_color_hud.set_alpha(torch_shadow_a), torch_shadow_softness);

    fan::graphics::light(fan::vec3(sun_lpos, 5.f), fan::vec2(90), sun_col.set_alpha(sun_visual_a));
    fan::graphics::light(fan::vec3(torch_lpos, 10.f), fan::vec2(torch_size_hud), torch_color_hud.set_alpha(torch_visual_a));

    {
      fan::time::scope_profiler_t profiler{"Terrain: Shadow Occluders"};
      engine.shadow_set_tile_occluders(terrain.shadow_occluders());
    }

    hotbar.handle_input();
    if (fan::window::is_key_clicked(fan::key_i)) {
      inventory_ui.visible = !inventory_ui.visible;
    }

    {
      fan::time::scope_profiler_t profiler{"Terrain: Stream"};
      terrain.stream(player_pos, engine.ws());
    }

    fan::vec2 mouse_pos = engine.get_mouse_position();
    fan::vec2 ray_end = mouse_pos;
    fan::vec2 ray_diff = mouse_pos - player_pos;
    f32_t ray_distance = ray_diff.length();
    if (ray_distance > dig_reach) {
      ray_end = player_pos + ray_diff / ray_distance * dig_reach;
    }
    fan::vec2 hit_pos;
    fan::vec2 dig_hit_pos;
    {
      fan::time::scope_profiler_t profiler{"Terrain: Raycast"};
      hit_pos = terrain.raycast(player_pos, mouse_pos, dig_radius);
      dig_hit_pos = terrain.raycast(player_pos, ray_end, dig_radius);
    }

    bool can_dig = !game_locked && !fan::graphics::gui::want_io() && fan::window::is_mouse_down(fan::mouse_left);
    if (!can_dig) {
      dig_interval.reset();
    }
    if (can_dig && dig_interval.tick(dt)) {
      std::vector<algorithm::chunk_renderer_t::edit_t> broken;
      {
        fan::time::scope_profiler_t profiler{"Terrain: Dig"};
        broken = terrain.dig(dig_hit_pos, dig_radius);
      }
      for (auto& b : broken) {
        if (b.type >= 0) {
          hotbar.add_item(fan::graphics::gameplay::items::create((std::uint32_t)b.type), 1);
        }
      }
      if (!broken.empty()) {
        break_effects.push_back({
          .circle = circle_t(circle_properties_t{
            .position = fan::vec3(dig_hit_pos, 26.f),
            .radius = cs * 0.2f,
            .color = fan::color(1.f, 1.f, 1.f, 0.6f),
            .outline_color = fan::color(1.f, 0.9f, 0.5f, 0.4f),
            .outline_width = 2.f,
          }),
          .timer = 0.25f,
        });

        dig_particles.spawn_from_json("models/dig_particles.json", fan::vec3(dig_hit_pos, 26.f));
      }
    }

    if (!game_locked && !fan::graphics::gui::want_io() && engine.is_mouse_clicked(fan::mouse_right)) {
      try_place(hit_pos);
    }

    std::erase_if(break_effects, [&](auto& effect) {
      effect.timer -= dt;
      if (effect.timer <= 0.f) { return true; }
      f32_t t = effect.timer / 0.25f;
      effect.circle.set_radius(cs * 0.2f + (1.f - t) * cs * 0.6f);
      effect.circle.set_color(fan::color(1.f, 1.f, 1.f, t * 0.6f));
      effect.circle.set_outline_color(fan::color(1.f, 0.9f, 0.5f, t * 0.4f));
      return false;
    });

    if (engine.is_toggled(fan::key_t)) {
      if (auto hud = gui::hud_interactive("Torch")) {
        gui::color_edit3("Color", (fan::vec3*)&torch_color_hud);
        gui::drag("Size", &torch_size_hud);
        gui::drag("Alpha", &torch_visual_a_hud);
        gui::drag("Shadow A", &torch_shadow_a_hud);
        gui::drag("Shadow Strength", &torch_shadow_strength);
        gui::drag("Shadow Softness", &torch_shadow_softness);
        gui::drag("Sun Softness", &sun_shadow_softness);
      }
    }

    if (save_interval.tick((f32_t)dt)) {
      save_world();
    }

    dig_particles.update(dt);
    inventory_ui.render();
    hotbar.render(inventory_ui.style.theme, inventory_ui.drag_state, inventory_ui.hovered_secondary_slot);

    if (feedback_timer > 0.f) {
      f32_t feedback_alpha = feedback_strength * feedback_timer / feedback_duration;
      auto feedback_style = gui::style_scope_t{gui::col_window_bg, feedback_color};
      if (auto feedback = gui::hud_interactive("##terrain_feedback", feedback_alpha)) {
      }
    }

    if (auto hud = gui::hud("##terrain_game_hud")) {
      gui::set_cursor_screen_pos({20.f, 20.f});
      if (game_won) {
        gui::text(fan::colors::green, "EXPEDITION COMPLETE");
      }
      else if (core_collected) {
        gui::text(fan::colors::yellow, "OBJECTIVE: Return to the surface");
      }
      else {
        gui::text(fan::colors::white, "OBJECTIVE: Find the deep core at ", (int)objective_depth_blocks, " blocks");
      }
      if (!core_collected && !game_won) {
        gui::set_cursor_screen_pos({20.f, 48.f});
        gui::text(
          fan::colors::white,
          "CORE: ", core_position.x < player_pos.x ? "LEFT" : "RIGHT",
          " ", (int)core_distance_from_player_blocks, " blocks"
        );
      }
      gui::set_cursor_screen_pos({20.f, 76.f});
      gui::text(fan::colors::white, "DEPTH: ", (int)depth_blocks, " / ", (int)objective_depth_blocks, " blocks");
      gui::set_cursor_screen_pos({20.f, 104.f});
      int max_health = (int)std::ceil(player.get_max_health());
      int health = std::clamp((int)std::ceil(player.get_health()), 0, max_health);
      fan::color health_fill = game_over ? fan::colors::red : fan::colors::green;
      if (!game_over && player.get_health() <= player.get_max_health() * 0.25f) {
        health_fill = std::sin(hazard_pulse * 8.f) > 0.f ? fan::colors::red : fan::colors::yellow;
      }
      gui::healthbar_labeled(
        "HEALTH", health, max_health, {220.f, 18.f},
        fan::colors::white,
        health_fill
      );
      if (player_in_hazard && !game_locked) {
        gui::set_cursor_screen_pos({20.f, 140.f});
        gui::text(fan::colors::yellow, "WARNING: Hazard damage");
      }
    }

    if (game_over || game_won) {
      if (auto overlay = gui::hud_interactive("##terrain_game_over", 0.78f)) {
        gui::set_cursor_screen_pos({engine.ws().x * 0.5f - 110.f, engine.ws().y * 0.5f - 40.f});
        gui::text(game_won ? fan::colors::green : fan::colors::red, game_won ? "EXPEDITION COMPLETE" : "YOU WERE HURT");
        gui::set_cursor_screen_pos({engine.ws().x * 0.5f - 130.f, engine.ws().y * 0.5f + 5.f});
        gui::text(fan::colors::white, game_won ? "Press R to start again" : "Press R to respawn");
      }
    }
  });

  return 0;
}
