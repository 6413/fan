static inline constexpr f32_t spike_height = 32.0f;
static inline constexpr f32_t base_half_width = 32.0f;

static inline constexpr std::array<fan::vec2, 3> get_spike_points(std::string_view dir) {
  static constexpr f32_t h = spike_height;
  static constexpr f32_t w = base_half_width;

  fan::vec2 a{0, -h}, b{-w, h}, c{w, h};
  if (dir == "down") {
    a.y = -a.y; b.y = -b.y; c.y = -c.y;
  }
  else if (dir == "left") {
    a = { h, 0 }; b = {-h, -w}; c = {-h,  w};
  }
  else if (dir == "right") {
    a = {-h, 0}; b = { h, -w}; c = { h,  w};
  }
  return {{a, b, c}};
}

void load_enemies() {
  pile->enemy_skeleton.Clear();
  // todo cache
  pile->renderer.iterate_visual(main_map_id, [&](fte_loader_t::tile_t& tile) ->bool {
    if (tile.id.contains("enemy_skeleton")) {
      auto nr = pile->enemy_skeleton.NewNodeLast();
      pile->enemy_skeleton[nr] = skeleton_t(pile->enemy_skeleton, nr, fan::vec3(fan::vec2(tile.position), 5));
    }
    return false;
  });
}

void load_map() {
  main_compiled_map = pile->renderer.compile("sample_level.fte");
  fan::vec2i render_size(16, 9);
  //render_size /= 0.01;
  fte_loader_t::properties_t p;
  p.size = render_size;
  p.position = pile->player.body.get_position();
  main_map_id = pile->renderer.add(&main_compiled_map, p);
  pile->engine.lighting.set_target(main_compiled_map.lighting.ambient, 0.01);

  checkpoint_flag = fan::graphics::sprite_sheet_from_json({
    .path = "effects/flag.json",
    .loop = true
  });

  checkpoint_flag.set_size(checkpoint_flag.get_size() / fan::vec2(1.5f, 1.0f));
  checkpoint_flag.set_position(pile->renderer.get_position(main_map_id, "checkpoint0") + fan::vec2(0, checkpoint_flag.get_size().y/2.0f));
  static auto axe_anim = fan::graphics::sprite_sheet_from_json({
    .path = "traps/axe/axe.json",
    .loop = true,
    .start = false
    });

  static auto lamp1_anim = fan::graphics::sprite_sheet_from_json({
    .path = "lights/lamp1/lamp.json",
    .loop = true
    });

  pile->renderer.iterate_visual(main_map_id, [&](fte_loader_t::tile_t& tile) -> bool {

    const std::string& id = tile.id;

    if (id.contains("checkpoint")) {
      player_checkpoints.emplace_back(
        fan::physics::create_sensor_rectangle(tile.position, tile.size)
      );
    }
    else if (tile.id.contains("roof_chain")) {}
    else if (id.contains("trap_axe")) {
      axes.emplace_back(axe_anim);
      axes.back().set_position(fan::vec3(fan::vec2(tile.position), 3));
    }
    else if (id.contains("sensor_health")) {
      health_sensors.emplace_back(
        fan::physics::create_sensor_rectangle(tile.position, tile.size / 1.2f)
      );
    }
    else if (id.contains("spikes")) {
      auto pts = get_spike_points(id.substr(std::strlen("spikes_")));
      spike_sensors.emplace_back(
        pile->engine.physics_context.create_polygon(
          tile.position,
          0.0f,
          pts.data(),
          pts.size(),
          fan::physics::body_type_e::static_body,
          {.is_sensor = true}
        )
      );
    }
    else if (id.contains("lamp1")) {
      lamps.emplace_back(lamp1_anim);
      auto& l = lamps.back();
      l.set_current_animation_frame(fan::random::value(0, l.get_current_animation_frame_count()));
      l.set_position(fan::vec3(fan::vec2(tile.position) + fan::vec2(1.f, -2.f), 1));
    }
    else if (tile.mesh_property == fte_loader_t::fte_t::mesh_property_t::none) {
      tile_collisions.emplace_back(
        pile->engine.physics_context.create_rectangle(
          tile.position,
          tile.size,
          0.0f,
          fan::physics::body_type_e::static_body,
          {.friction = 0.f, .fixed_rotation = true}
        )
      );
    }

    return false;
  });

  pile->player.respawn();
  pile->player.particles.set_color(0);
}

void open(void* sod) {
  pile->level_stage = this->stage_common.stage_id;
  load_map();
}

void close() {
  for (auto& i : health_sensors) {
    i.destroy();
  }
  for (auto& i : player_checkpoints) {
    i.destroy();
  }
  player_checkpoints.clear();
  for (auto& i : tile_collisions) {
    i.destroy();
  }
  tile_collisions.clear();
  for (auto& i : spike_sensors) {
    i.destroy();
  }
  pile->renderer.erase(main_map_id);
}

void reload_map() {
  pile->stage_loader.erase_stage(this->stage_common.stage_id);
  pile->stage_loader.open_stage<level_t>();
}

void update() {
  static fan::color c;
  static fan::vec2 p = background.get_position();
  static fan::vec2 s = background.get_size();
  fan::graphics::gui::begin("A");
  fan::graphics::gui::color_edit4("test",&c);
  fan::graphics::gui::drag("test2",&p);
  fan::graphics::gui::drag("test3",&s);
  fan::graphics::gui::end();
  background.set_color(c);
  background.set_position(p);
  background.set_size(s);

  lights.resize(lamps.size());
  
  for (auto [i, lamp] : fan::enumerate(lamps)) {
    lights[i].set_position(fan::vec2(lamp.get_position()));
    lights[i].set_size(512);
    auto tc_center = lamp.get_tc_position() + lamp.get_tc_size() * 0.5f;
    auto pixel_size = fan::vec2(1.0f) / image_get_data(lamp.get_image()).size;
    auto pixels = read_pixels_from_image(lamp.get_image(), tc_center, pixel_size);
    
    uint32_t ch = fan::graphics::get_channel_amount(image_get_settings(lamp.get_image()).format);
    fan::color current = lights[i].get_color() / 2.f;
    f32_t lerp_speed = std::min(pile->engine.delta_time * 10.0f, 1.0);
    fan::color new_color = current.lerp(fan::color(pixels.data(), pixels.data() + ch), lerp_speed);
    fan::color yellow_tint(0.9f, 0.9f, 0.2f, 1.0f);

    lights[i].set_color(fan::color(pixels.data(), pixels.data() + ch) * yellow_tint);
    pile->engine.lighting.set_target(fan::color(pixels.data(), pixels.data() + ch) / 5.f + 0.7, 0.1);
  }

  for (auto it = health_sensors.begin(); it != health_sensors.end(); ++it) {
    auto& sensor = *it;
    if (pile->player.body.get_health() < pile->player.body.get_max_health()) {
      if (fan::physics::is_on_sensor(pile->player.body, sensor)) {
        static constexpr f32_t health_restore = 10.f;
        pile->player.body.set_health(
          pile->player.body.get_health() + health_restore
        );
        fan::vec2 pos = sensor.get_position();
        it->destroy();
        health_sensors.erase(it);
        pile->renderer.remove_visual(
          pile->get_level().main_map_id,
          "sensor_health",
          pos
        );
        break;
      }
    }
    bool consumed = false;
    for (auto& enemy : pile->enemy_skeleton) {
      if (enemy.body.get_health() == enemy.body.get_max_health()) {
        continue;
      }
      if (!fan::physics::is_on_sensor(enemy.body, sensor)) {
        continue;
      }
      static constexpr f32_t health_restore = 10.f;
      enemy.body.set_health(
        enemy.body.get_health() + health_restore
      );
      fan::vec2 pos = sensor.get_position();
      it->destroy();
      health_sensors.erase(it);
      pile->renderer.remove_visual(
        pile->get_level().main_map_id,
        "sensor_health",
        pos
      );
      consumed = true;
      break;
    }
    if (consumed) {
      break;
    }
  }

  for (auto& spike : spike_sensors) {
    if (fan::physics::is_on_sensor(pile->player.body, spike)) {
      pile->player.respawn();
      load_enemies();
    }
    for (auto [nr, enemy] : fan::enumerate(pile->enemy_skeleton)) {
      if (fan::physics::is_on_sensor(enemy.body, spike)) {
        enemy.destroy();
      }
    }
  }

  if (!pile->engine.render_console) {
    if (fan::window::is_key_pressed(fan::key_e)) {
      pile->pause = !pile->pause;
    }
  
    if (fan::window::is_key_pressed(fan::key_r)) {
      reload_map();
      return;
    }
  }
  pile->renderer.update(main_map_id, pile->player.body.get_position());
  pile->update();
}

fte_loader_t::id_t main_map_id;
fte_loader_t::compiled_map_t main_compiled_map;

std::vector<fan::physics::entity_t> spike_sensors;
std::vector<fan::physics::body_id_t> health_sensors;
std::vector<fan::physics::entity_t> tile_collisions;

std::vector<fan::graphics::sprite_t> axes;
std::vector<fan::physics::entity_t> axe_collisions;

fan::graphics::sprite_t checkpoint_flag;
std::vector<fan::physics::entity_t> player_checkpoints;
//fan::graphics::sprite_t bg{{
//  .position = fan::vec3(10000, 10000, 0),
//  .size = 10000, 
//  .image = fan::graphics::image_create(fan::colors::black + 0.1)
//}};

std::vector<fan::graphics::sprite_t> lamps;
std::vector<fan::graphics::light_t> lights;

fan::graphics::sprite_t background {{
  .position = fan::vec3(10000, 6010, 0),
  .size = fan::vec2(9192, 10000),
  .color = fan::color(0.6, 0.576, 1),
  .image = fan::graphics::image_t("images/background.png"),
  .tc_size = 1 * 300.0,
}};