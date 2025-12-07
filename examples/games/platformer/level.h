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
  pile->enemy_list.Clear();
  pile->renderer.iterate_visual(main_map_id, [&](fte_loader_t::tile_t& tile) ->bool {
    if (tile.id.contains("enemy_skeleton")) {
      auto nr = pile->enemy_list.NewNodeLast();
      pile->enemy_list[nr] = skeleton_t(pile->enemy_list, nr, fan::vec3(fan::vec2(tile.position), 5));
    }
    return false;
  });
  pile->renderer.iterate_visual(main_map_id, [&](fte_loader_t::tile_t& tile) ->bool {
    if (tile.id.contains("enemy_fly")) {
     // auto nr = pile->enemy_list.NewNodeLast();
      //pile->enemy_list[nr] = fly_t(pile->enemy_list, nr, fan::vec3(fan::vec2(tile.position), 5));
    }
    return false;
  });
}

// returns whether the object was picked up
template <typename T>
bool handle_pickupable(const std::string& id, T& who) {
  constexpr bool is_player = std::is_same_v<T, player_t>;
  switch (fan::get_hash(id)) {
  case fan::get_hash("pickupable_health"):
  {
    static constexpr f32_t health_restore = 10.f;
    who.get_body().set_health(
      who.get_body().get_health() + health_restore
    );
    break;
  }
  case fan::get_hash("pickupable_health_potion"):
  {
    if constexpr (!is_player) {
      return false;
    }
    else {
      ++who.potion_count;
    }
    break;
  }
  }
  return true;
}

void load_map() {
  main_compiled_map = pile->renderer.compile("sample_level.fte");
  fan::vec2i render_size(16, 9);
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
      int checkpoint_idx = std::stoi(id.substr(std::strlen("checkpoint")));
      if (player_checkpoints.size() < checkpoint_idx) {
        player_checkpoints.resize(checkpoint_idx + 1);
      }
      player_checkpoints[checkpoint_idx] = fan::physics::create_sensor_rectangle(tile.position, tile.size);
    }
    else if (id.contains("roof_chain")) {}
    else if (id.contains("trap_axe")) {
      axes.emplace_back(axe_anim);
      axes.back().set_position(fan::vec3(fan::vec2(tile.position), 3));
    }
    else if (id.contains("pickupable_")) {
      pickupables.push_back(
        {id, fan::physics::create_sensor_rectangle(tile.position, tile.size / 1.2f)}
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
  for (auto& i : pickupables) {
    i.second.destroy();
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

  for (auto it = pickupables.begin(); it != pickupables.end(); ++it) {
    auto& sensor = it->second;
    if (pile->player.body.get_health() < pile->player.body.get_max_health()) {
      if (fan::physics::is_on_sensor(pile->player.body, sensor)) {
        if (handle_pickupable(it->first, pile->player)) {
          fan::vec2 pos = sensor.get_position();
          pile->renderer.remove_visual(
            pile->get_level().main_map_id,
            it->first,
            pos
          );
          sensor.destroy();
          pickupables.erase(it);
        }
        break;
      }
    }
    bool consumed = false;
    for (auto enemy : pile->enemies()) {
      if (enemy.get_body().get_health() == enemy.get_body().get_max_health()) {
        continue;
      }
      if (!fan::physics::is_on_sensor(enemy.get_body(), sensor)) {
        continue;
      }
      if (handle_pickupable(it->first, enemy)) {
        fan::vec2 pos = sensor.get_position();
        it->second.destroy();
        pickupables.erase(it);
        pile->renderer.remove_visual(
          pile->get_level().main_map_id,
          it->first,
          pos
        );
        consumed = true;
      }
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
    for (auto enemy : pile->enemies()) {
      if (fan::physics::is_on_sensor(enemy.get_body(), spike)) {
        enemy.destroy();
      }
    }
  }

  if (!pile->engine.render_console) {
    if (fan::window::is_key_pressed(fan::key_e)) {
      pile->pause = !pile->pause;
    }

    if (fan::window::is_key_pressed(fan::key_t)) {
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
std::vector<std::pair<std::string, fan::physics::body_id_t>> pickupables;
std::vector<fan::physics::entity_t> tile_collisions;

std::vector<fan::graphics::sprite_t> axes;
std::vector<fan::physics::entity_t> axe_collisions;

fan::graphics::sprite_t checkpoint_flag;
std::vector<fan::physics::entity_t> player_checkpoints;

std::vector<fan::graphics::sprite_t> lamps;
std::vector<fan::graphics::light_t> lights;

fan::graphics::sprite_t background {{
    .position = fan::vec3(10000, 6010, 0),
    .size = fan::vec2(9192, 10000),
    .color = fan::color(0.6, 0.576, 1),
    .image = fan::graphics::image_t("images/background.png"),
    .tc_size = 1 * 300.0,
  }};