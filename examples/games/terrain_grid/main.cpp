import std;
import fan;
import fan.graphics.gui.hotbar;
import fan.graphics.gui.inventory;
import fan.graphics.gameplay.types;

using namespace fan::graphics;

namespace {
  constexpr const char* block_names[] = {
    "Grass", "Dirt", "Clay", "Sandstone", "Stone", "Deep Stone", "Bedrock"
  };
  constexpr int block_count = (int)std::size(block_names);
}

int main() {
  engine_t engine;
  interactive_camera_t ic;
  engine.get_clear_color() = fan::colors::black;
  engine.update_physics(true);

  engine.shadow_enable_tile_mode();
  engine.shadow_set_darkness(0.6f);

  gameplay::gui_theme_t theme;
  theme.panel_bg = fan::color(0.08f, 0.08f, 0.10f, 0.90f);
  theme.panel_border = fan::color(0.5f, 0.5f, 0.6f, 1.f);
  theme.panel_corner_accent = fan::color(0.6f, 0.65f, 0.7f, 1.f);
  theme.slot_bg = fan::color(0.15f, 0.15f, 0.17f, 0.95f);
  theme.slot_bg_hover = fan::color(0.30f, 0.32f, 0.36f, 0.95f);
  theme.slot_border = fan::color(0.4f, 0.4f, 0.45f, 1.f);
  theme.selected_border_color = fan::colors::white;

  gui::inventory_t inventory_ui;
  inventory_ui.create(block_count, block_count);
  inventory_ui.visible = false;
  inventory_ui.style.theme = theme;

  gui::hotbar_t hotbar;
  hotbar.create(block_count);

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
  image_t img_grass     {"images/Textures/Grass/cubeGreen_1.png",   pa};
  image_t img_dirt      {"images/Textures/Dirt/cubeDirt_1.png",     pa};
  image_t img_clay      {"images/Textures/Purple/cubePurple_1.png",  pa};
  image_t img_sandstone {"images/Textures/Desert/cubeDesert_1.png",  pa};
  image_t img_stone     {"images/Textures/Stone/cubeStone_1.png",   pa};
  image_t img_deep_stone{"images/Textures/Stone/cubeStone_5.png",   pa};
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
      {3.f,    img_grass},
      {30.f,   img_dirt},
      {100.f,  img_clay},
      {220.f,  img_sandstone},
      {550.f,  img_stone},
      {1000.f, img_deep_stone},
      {std::numeric_limits<f32_t>::max(), img_bedrock}
    },
    .scatter_noise = &detail_noise,
    .scatter_img = img_dark_grass,
    .scatter_threshold = 0.6f,
  }};
  terrain.load_edits(world_edits);

  auto& item_reg = fan::graphics::gameplay::items::get_registry();
  image_t block_icons[block_count] = {
    img_grass, img_dirt, img_clay, img_sandstone, img_stone, img_deep_stone, img_bedrock
  };
  for (int i = 0; i < block_count; ++i) {
    item_reg.register_item({
      .id = (std::uint32_t)i,
      .name = block_names[i],
      .icon = block_icons[i],
      .max_stack = 999,
      .description = block_names[i],
    });
  }

  auto player = physics::from_json({
    .json_path = "models/Base Character/base_character.json"
  });
  player.play_sprite_sheet("idle");
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

  engine.shadow_add_caster(&player, 0.05f);

  auto& lighting = fan::graphics::get_lighting();
  lighting.ambient = fan::vec3(.6f/255.);

  f32_t dig_radius = 2.f;
  fan::time::interval_t dig_interval{0.003f};
  fan::time::interval_t save_interval{10.f};

  gpu_particle_system_t<> dig_particles;
  struct break_effect_t { circle_t circle; f32_t timer = 0.f; };
  std::vector<break_effect_t> break_effects;

  fan::color torch_color_hud{1.f, 1.f, 0.8f, 1.f};
  f32_t torch_size_hud = 600.f;
  f32_t torch_shadow_a_hud = 1.0f;
  f32_t torch_shadow_strength = 1.0f;
  f32_t torch_shadow_softness = 15.956f;
  f32_t torch_visual_a_hud = 0.200f;
  f32_t sun_shadow_softness = 30.f;

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
      if (s.is_empty()) continue;
      fan::json entry = fan::json::object();
      entry.set("id", (int)*s.id);
      entry.set("count", (int)*s.stack_size);
      inv.push_back(entry);
    }
    j.set("inventory", inv);
    j.save(save_path);
  };
  engine.get_window().add_close_callback([&](const fan::window_t::close_data_t&) {
    save_world();
  });

  f32_t cs = terrain.cell_size();
  auto cell_of = [&](fan::vec2 p) -> fan::vec2i {
    return {(int)std::floor(p.x / cs), (int)std::floor(p.y / cs)};
  };
  auto cell_center = [&](int gx, int gy) -> fan::vec2 {
    return (fan::vec2((f32_t)gx, (f32_t)gy) + 0.5f) * cs;
  };
  auto try_place = [&](fan::vec2 hit_pos) -> bool {
    auto& sel = hotbar.slots[hotbar.selected_slot];
    if (sel.is_empty()) return false;
    int gx, gy;
    fan::vec2 mouse_pos = engine.get_mouse_position();
    if (hit_pos == mouse_pos) {
      auto c = cell_of(mouse_pos);
      if (terrain.get_solid(c.x, c.y)) return false;
      gx = c.x; gy = c.y;
    }
    else {
      fan::vec2 diff = mouse_pos - player.get_position();
      f32_t dist = diff.length();
      if (dist < 1e-4f) return false;
      fan::vec2 dir = diff / dist;
      bool found = false;
      for (f32_t d = cs * 0.5f; d <= cs * 3.f; d += cs * 0.5f) {
        fan::vec2 p = hit_pos - dir * d;
        auto c = cell_of(p);
        if (!terrain.get_solid(c.x, c.y)) { gx = c.x; gy = c.y; found = true; break; }
      }
      if (!found) return false;
    }
    fan::vec2 cc = cell_center(gx, gy);
    fan::physics::aabb_t aabb = player.get_aabb();
    if (cc.x + cs * 0.5f > aabb.min.x && cc.x - cs * 0.5f < aabb.max.x &&
        cc.y + cs * 0.5f > aabb.min.y && cc.y - cs * 0.5f < aabb.max.y) return false;
    int type = (int)*sel.id;
    if (!hotbar.consume_slot(hotbar.selected_slot, nullptr)) return false;
    terrain.place(cc, cs * 0.5f, type);
    return true;
  };

  engine.loop([&] {
    f64_t dt = engine.get_delta_time();
    player.update_animations();
    fan::vec2 player_pos = player.get_position();
    fan::vec2 cam_center = ic.get_center();

    f32_t ground_y = 512.f;
    bg_sky.set_position(fan::vec2(cam_center.x, std::min(cam_center.y, ground_y)));
    bg_below.set_position(cam_center);
    bg_below.set_color(lighting.ambient);

    f32_t surface_h = terrain.surface_height_at(player_pos.x);
    f32_t depth = std::max(0.f, surface_h - player_pos.y);
    f32_t cave_factor = std::min(depth / 40.f, 1.f);

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

    engine.shadow_set_tile_occluders(terrain.shadow_occluders());

    if (fan::window::is_key_clicked(fan::key_r)) {
      player.set_physics_position({player_pos.x, 0});
    }

    hotbar.handle_input();
    if (fan::window::is_key_clicked(fan::key_i)) {
      inventory_ui.visible = !inventory_ui.visible;
    }

    terrain.stream(player_pos, engine.ws());

    fan::vec2 mouse_pos = engine.get_mouse_position();
    fan::vec2 hit_pos = terrain.raycast(player_pos, mouse_pos, dig_radius);

    if (!fan::graphics::gui::want_io() && fan::window::is_mouse_down(fan::mouse_left) && dig_interval.tick(dt)) {
      auto broken = terrain.dig(hit_pos, dig_radius);
      for (auto& b : broken) {
        if (b.type < 0) continue;
        hotbar.add_item(fan::graphics::gameplay::items::create((std::uint32_t)b.type), 1);
      }
      if (!broken.empty()) {
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

        dig_particles.spawn_from_json("models/dig_particles.json", fan::vec3(hit_pos, 26.f));
      }
    }

    if (!fan::graphics::gui::want_io() && engine.is_mouse_clicked(fan::mouse_right)) {
      try_place(hit_pos);
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

    if (engine.is_toggled(fan::key_t))
    if (auto hud = gui::hud_interactive("Torch")) {
      gui::color_edit3("Color", (fan::vec3*)&torch_color_hud);
      gui::slider("Size", &torch_size_hud, 50.f, 600.f);
      gui::slider("Alpha", &torch_visual_a_hud, 0.f, 2.f);
      gui::slider("Shadow A", &torch_shadow_a_hud, 0.f, 1.f);
      gui::slider("Shadow Strength", &torch_shadow_strength, 0.f, 2.f);
      gui::slider("Shadow Softness", &torch_shadow_softness, 0.f, 500.f);
      gui::slider("Sun Softness", &sun_shadow_softness, 0.f, 500.f);
    }

    if (save_interval.tick((f32_t)dt)) {
      save_world();
    }

    dig_particles.update(dt);
    inventory_ui.render();
    hotbar.render(inventory_ui.style.theme, inventory_ui.drag_state, inventory_ui.hovered_secondary_slot);
  });

  return 0;
}
