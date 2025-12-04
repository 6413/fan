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
  pile->renderer.iterate_visual(main_map_id, [&](fte_loader_t::tile_t& tile) {
    if (tile.id.contains("enemy_skeleton")) {
      auto nr = pile->enemy_skeleton.NewNodeLast();
      pile->enemy_skeleton[nr] = skeleton_t(pile->enemy_skeleton, nr, tile.position);
    }
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

  static auto lamp1 = fan::graphics::sprite_sheet_from_json({
    .path = "lights/lamp1/lamp.json",
    .loop = true
  });

  pile->renderer.iterate_visual(main_map_id, [&](fte_loader_t::tile_t& tile) {
    if (tile.id.contains("checkpoint")) {
      player_checkpoints.emplace_back(fan::physics::create_sensor_rectangle(tile.position, tile.size));
    }
    else if (tile.id.contains("spikes")) {
      auto points = get_spike_points(tile.id.substr(std::strlen("spikes_")));
      spike_sensors.emplace_back(pile->engine.physics_context.create_polygon(
        tile.position,
        0.0f,
        points.data(),
        points.size(),
        fan::physics::body_type_e::static_body,
        {.is_sensor = true}
      ));
    }
    else if (tile.id.contains("lamp1")) {
      lamps.push_back(lamp1);
      lamps.back().set_current_animation_frame(fan::random::value(0, lamps.back().get_current_animation_frame_count()));
      lamps.back().set_position(fan::vec3(fan::vec2(tile.position) + fan::vec2(1.f, -2.f), 1));
    }
    else if (tile.mesh_property == fte_loader_t::fte_t::mesh_property_t::none) {
      tile_collisions.emplace_back(pile->engine.physics_context.create_rectangle(
        tile.position,
        tile.size,
        0.0f,
        fan::physics::body_type_e::static_body,
        {
          .friction=0.f,
          .fixed_rotation = true
        }
      ));
    }
  });

  pile->player.respawn();
  pile->player.particles.set_color(0);
}

void open(void* sod) {
  pile->level_stage = this->stage_common.stage_id;
  load_map();
}

void close() {
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
    lights[i].set_color(fan::color(pixels.data(), pixels.data() + ch));
    pile->engine.lighting.set_target(fan::color(pixels.data(), pixels.data() + ch) / 5.f + 0.7, 0.1);
  }
  
  for (auto& spike : spike_sensors) {
    if (fan::physics::is_on_sensor(pile->player.body, spike)) {
      pile->player.respawn();
      load_enemies();
    }
    for (auto& enemy : pile->enemy_skeleton) {
      if (fan::physics::is_on_sensor(enemy.body, spike)) {
        enemy.destroy();
      }
    }
  }

  static bool pause = false;
  if (!pile->engine.render_console) {
    if (fan::window::is_key_pressed(fan::key_e)) {
      pause = !pause;
    }
  
    if (fan::window::is_key_pressed(fan::key_r)) {
      reload_map();
      return;
    }
  }
  pile->renderer.update(main_map_id, pile->player.body.get_position());
  if (pause) {
    return;
  }
  pile->update();
}



fte_loader_t::id_t main_map_id;
fte_loader_t::compiled_map_t main_compiled_map;

std::vector<fan::physics::entity_t> spike_sensors;
std::vector<fan::physics::entity_t> tile_collisions;

fan::graphics::sprite_t checkpoint_flag;
std::vector<fan::physics::entity_t> player_checkpoints;
fan::graphics::sprite_t bg{{
  .position = 10000, 
  .size = 10000, 
  .image = fan::graphics::image_create(fan::colors::black + 0.1)
}};

std::vector<fan::graphics::sprite_t> lamps;
std::vector<fan::graphics::light_t> lights;