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
  pile->renderer.iterate_marks(main_map_id, [&](tilemap_loader_t::fte_t::spawn_mark_data_t &data) ->bool {
    const auto& id = data.id;
    if (id.contains("enemy_skeleton")) {
      auto nr = pile->enemy_list.NewNodeLast();
      pile->enemy_list[nr] = skeleton_t(pile->enemy_list, nr, fan::vec3(fan::vec2(data.position), 5));
    }
    else if (id.contains("enemy_fly")) {
      auto nr = pile->enemy_list.NewNodeLast();
      pile->enemy_list[nr] = fly_t(pile->enemy_list, nr, fan::vec3(fan::vec2(data.position), 5));
    }
    else if (id.contains("boss_skeleton")) {
      boss_position = data.position;
    }
    return false;
  });
}

// returns whether the object was picked up
template <typename T>
bool handle_pickupable(const std::string& id, T& who) {
  constexpr bool is_player = std::is_same_v<T, player_t>;
  
  auto pickup_item = [&] {
    switch (fan::get_hash(id)) {
    case fan::get_hash("pickupable_health"):
    {
      if (who.get_body().get_health() >= who.get_body().get_max_health()) {
        return false;
      }
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
  };

  bool ret = pickup_item();
  if (ret) {
    fan::audio::play(audio_pickup_item);
  }
  return ret;
}


std::vector<fan::auto_color_transition_t> light_lights;

void start_lights(uint32_t index) {
  static fan::color target_color = fan::color::from_rgb(0x114753) * 4.f;
  static auto torch_particles = fan::graphics::shape_from_json("effects/torch.json");
  torch_particles.set_position(fan::vec2(-0xfffff));

  static auto add_light_particles = [] (level_t* level, uint32_t index) {
    level->boss_torch_particles.emplace_back(torch_particles);
    auto& l = level->boss_torch_particles.back();
    l.start_particles();
    l.set_position(fan::vec3(fan::vec2(level->lights_boss[index].get_position()) + fan::vec2(0, 30.f), 1));
    l.set_static(true);
  };

  if (index + 1 >= lights_boss.size()) {
      auto* shape = pile->renderer.get_light_by_id(main_map_id, "boss_room_ambient_light");
    fan::color target_color = boss_room_target_color;
    boss_room_light.start_once(
      shape->get_color(),
      target_color,
      1.0f,
      [shape](fan::color c) {
        shape->set_color(c);
      }
    );
  }

  if (index >= lights_boss.size()) {
    typename decltype(pile->enemy_list)::nr_t nr;
    nr.gint() = boss_nr;
    std::visit([]<typename T>(T& v) {
      if constexpr (requires{ T::allow_move; }) {
        v.allow_move = true;
      }
    }, pile->enemy_list[nr]);

    for (auto [i, light] : fan::enumerate(light_lights)) {
      light_lights[i] = fan::auto_color_transition_t{};
      light_lights[i].start(
        lights_boss[i].get_color() * (fan::color(1.0f, 0.7f, 0.7f) / 1.0f),
        lights_boss[i].get_color() * (fan::color(1.0f, 1.0f, 1.0f) * 1.0f),
        0.2f + fan::random::value(0.0f, 0.5f),
        [this, i](fan::color c){
          lights_boss[i].set_color(c); 
        }
      );
    }
    return;
  }

  light_lights[index] = fan::auto_color_transition_t{};
  add_light_particles(this, index);
  light_lights[index].on_end = [this, index] {
    start_lights(index + 1);
  };
  light_lights[index].start_once(
    lights_boss[index].get_color(),
    target_color,
    1.0f,
    [this, index](fan::color c) {
      lights_boss[index].set_color(c);
    }
  );
}

void enter_boss() {
  boss_door_collision = pile->engine.physics_context.create_rectangle(
    boss_door_position,
    boss_door_size
  );
  is_entering_door = false;
  
  auto nr= pile->enemy_list.NewNodeLast();
  pile->enemy_list[nr] = boss_skeleton_t(pile->enemy_list, nr, fan::vec3(boss_position, 5));
  boss_nr = nr.gint();
  light_lights.resize(lights_boss.size());
  start_lights(0);
}

void load_map() {
  //pile->engine.culling_rebuild_grid();
  background.set_static();
  main_compiled_map = pile->renderer.compile("sample_level.fte");
  fan::vec2i render_size(16, 9);
  tilemap_loader_t::properties_t p;
  p.size = render_size * 1000;
  pile->engine.set_cull_padding(100);

  p.position = pile->player.body.get_position();
  main_map_id = pile->renderer.add(&main_compiled_map, p);
  pile->engine.lighting.set_target(main_compiled_map.lighting.ambient, 0.01);

  static auto checkpoint_flag = fan::graphics::sprite_sheet_from_json({
    .path = "effects/flag.json",
    .loop = true
  });

  static auto axe_anim = fan::graphics::sprite_sheet_from_json({
    .path = "traps/axe/axe.json",
    .loop = true,
    .start = false
  });

  static auto lamp1_anim = fan::graphics::sprite_sheet_from_json({
    .path = "lights/lamp1/lamp.json",
    .loop = true
  });


  checkpoint_flag.set_position(fan::vec2(-0xfffff));
  axe_anim.set_position(fan::vec2(-0xfffff));
  lamp1_anim.set_position(fan::vec2(-0xfffff));

  pile->renderer.iterate_physics_entities(main_map_id, [&](auto& data, auto& entity_visual) -> bool {
    const auto& id = data.id;
    if (id.contains("checkpoint")) {
      int checkpoint_idx = std::stoi(id.substr(std::strlen("checkpoint")));
      if (player_checkpoints.size() < checkpoint_idx) {
        player_checkpoints.resize(checkpoint_idx + 1);
      }
      auto& chkp = player_checkpoints[checkpoint_idx];
      checkpoint_flag.set_position(fan::vec3(entity_visual.get_position()));
      chkp.visual = checkpoint_flag;
      checkpoint_flag.set_position(fan::vec2(-0xfffff));
      chkp.visual.set_size(checkpoint_flag.get_size() / fan::vec2(1.5f, 1.0f));
      chkp.visual.start_sprite_sheet_animation();
      chkp.entity = entity_visual;
    }
    else if (id.contains("sensor_enter_boss")) {
      boss_sensor = entity_visual;
    }
    else if (id.contains("boss_door_collision")) {
      boss_door_position = entity_visual.get_position();
      boss_door_size = entity_visual.get_size();
      boss_door_particles = fan::graphics::shape_from_json("effects/boss_spawn.json");
      boss_door_particles.set_position(fan::vec3(fan::vec2(entity_visual.get_position()), 0xFAAA / 2 - 2 + boss_door_particles.get_position().z));
      boss_door_particles.set_static(true); // reset the static culling build
      boss_door_particles.start_particles();
    }
    return false;
  });

  pile->renderer.iterate_marks(main_map_id, [&](tilemap_loader_t::fte_t::spawn_mark_data_t& data) -> bool {
    const auto& id = data.id;
    if (id.contains("lamp1")) {
      lamp_sprites.emplace_back(lamp1_anim);
      auto& l = lamp_sprites.back();

      l.set_current_animation_frame(fan::random::value(0, l.get_current_animation_frame_count()));
      l.set_position(fan::vec3(fan::vec2(data.position) + fan::vec2(1.f, -2.f), 1));
      l.set_static(true);
      lights.emplace_back(fan::graphics::light_t {{
        .position = l.get_position(),
        .size = 512
      }});
    }
    else if (id.contains("lamp2")) {
      //boss_torch_particles.emplace_back(torch_particles);
      //auto& l = boss_torch_particles.back();
      //l.start_particles();
      //l.set_position(fan::vec3(fan::vec2(data.position) + fan::vec2(0, 30.f), 1));
      //l.set_static(true);
      lights_boss.emplace_back(fan::graphics::light_t {{
        .position = data.position,
        .size = 512,
        .color = fan::colors::black,
      }});
    }
    else if (id.contains("boss_elevator")) {
      fan::graphics::image_t image = fan::graphics::image_load("images/cage.png", fan::graphics::image_presets::pixel_art());
      fan::vec3 v = data.position;
      v.y += 26.f;

      fan::vec2 start_pos = fan::vec2(v.x, v.y - 1024.f);
      fan::vec2 end_pos = fan::vec2(v.x, v.y);
      f32_t elevator_duration = 10.f;
      cage_elevator.init(fan::graphics::sprite_t(fan::vec3(start_pos, v.z + 1), image.get_size() * 2.0f, image), start_pos, end_pos, elevator_duration);
      cage_elevator.on_end_cb = [pos = data.position, this] {
        fan::vec3 v = pos;
        fan::vec2 start_pos = fan::vec2(v.x, v.y);
        fan::vec2 end_pos = fan::vec2(v.x, v.y - 5000.f);
        cage_elevator.start_position = start_pos;
        cage_elevator.end_position = end_pos;
        cage_elevator.going_up = true;
      };
    }
    return false; // continue iterating all instances
  });

  pile->renderer.iterate_visual(main_map_id, [&](tilemap_loader_t::tile_t& tile) -> bool {
    const std::string& id = tile.id;

    if (id.contains("roof_chain")) {}
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
    else if (id.contains("no_collision")) {
      return false;
    }
    else if (tile.mesh_property == tilemap_loader_t::fte_t::mesh_property_t::none) {
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

  {
    auto* boss_room_light = pile->renderer.get_light_by_id(main_map_id, "boss_room_ambient_light");
    boss_room_target_color = boss_room_light->get_color();
    boss_room_light->set_color(fan::colors::black);
  }
}

void open(void* sod) {
  pile->level_stage = this->stage_common.stage_id;
  load_map();
  pile->engine.lighting.set_target(0, 0);
  is_entering_door = false;
}

void close() {
  cage_elevator.destroy();
  if (boss_door_collision) {
    boss_door_collision.destroy();
  }
  for (auto& i : pickupables) {
    i.second.destroy();
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
  pile->renderer.update(pile->get_level().main_map_id, pile->player.body.get_position());
  //                          ^ 'this' has been erased by erase stage, so query new pointer from get_level
}

fan::graphics::physics::elevator_t cage_elevator;
bool send_elevator_down_initially = true;

void update() {
  if (is_boss_dead && send_elevator_down_initially) {
    cage_elevator.start();
    send_elevator_down_initially = false;
  }

  cage_elevator.update(pile->player.body);

  for (auto [i, lamp] : fan::enumerate(lamp_sprites)) {
    if (i < lights.size()) {
      auto tc_center = lamp.get_tc_position() + lamp.get_tc_size() * 0.5f;
      auto pixel_size = fan::vec2(1.0f) / image_get_data(lamp.get_image()).size;
      auto pixels = read_pixels_from_image(lamp.get_image(), tc_center, pixel_size);
  
      uint32_t ch = fan::graphics::get_channel_amount(image_get_settings(lamp.get_image()).format);
      fan::color current = lights[i].get_color() / 2.f;
      f32_t lerp_speed = std::min(pile->engine.delta_time * 10.0f, 1.0);
      fan::color new_color = current.lerp(fan::color(pixels.data(), pixels.data() + ch), lerp_speed);
      fan::color yellow_tint(0.9f, 0.9f, 0.6f, 1.0f);

      lights[i].set_color(fan::color(pixels.data(), pixels.data() + ch) * yellow_tint * 2.f);
    }

    //pile->engine.lighting.set_target(fan::color(pixels.data(), pixels.data() + ch) / 5.f + 0.7, 0.1);
  }

  for (auto it = pickupables.begin(); it != pickupables.end(); ) {
    auto& sensor = it->second;
    if (fan::physics::is_on_sensor(pile->player.body, sensor)) {
      if (handle_pickupable(it->first, pile->player)) {
        fan::vec2 pos = sensor.get_position();

        pile->renderer.remove_visual(
          pile->get_level().main_map_id,
          it->first,
          pos
        );

        sensor.destroy();

        it = pickupables.erase(it);
        break;
      }
      else {
        ++it;
      }
    }
    else {
      ++it;
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
      enemy.get_body().update_dynamic();
    }
  }

  if (boss_sensor && 
    fan::physics::is_on_sensor(pile->player.body, boss_sensor)
  )
  {
    if (pile->engine.is_key_pressed(fan::key_e)) {
      pile->renderer.erase_physics_entity(main_map_id, "boss_door_collision");
      boss_sensor.destroy();
      is_entering_door = true;
      //boss_door_particles.stop_particles();
    }
    else {
      fan::vec2 window_size = fan::graphics::gui::get_window_size();
      static std::string enter_text("Press E to enter");
      static fan::vec2 text_size = fan::graphics::gui::get_text_size(enter_text);
      if (fan::graphics::gui::hud("Interact hud")) {
        fan::graphics::gui::text_box_at(enter_text, fan::vec2(window_size.x / 2.f - text_size.x / 2.f, window_size.y * 0.85f));
      }
    }
  }

  if (!pile->engine.render_console) {
    if (fan::window::is_key_pressed(fan::key_escape)) {
      pile->pause = !pile->pause;
    }

    if (fan::window::is_key_pressed(fan::key_t)) {
      reload_map();
      return;
    }
  }
  
  pile->update();
}

fan::physics::entity_t boss_sensor;
fan::graphics::shape_t boss_door_particles;
std::vector<fan::graphics::shape_t> boss_torch_particles;
fan::vec2 boss_position = 0;
fan::vec2 boss_door_position = 0, boss_door_size = 0;
fan::physics::entity_t boss_door_collision;
uint32_t boss_nr = (uint32_t)-1;
fan::auto_color_transition_t boss_room_light;
fan::color boss_room_target_color;


tilemap_loader_t::id_t main_map_id;
tilemap_loader_t::compiled_map_t main_compiled_map;

std::vector<fan::physics::entity_t> spike_sensors;
std::vector<std::pair<std::string, fan::physics::body_id_t>> pickupables;
std::vector<fan::physics::entity_t> tile_collisions;

std::vector<fan::graphics::sprite_t> axes;
std::vector<fan::physics::entity_t> axe_collisions;

struct checkpoint_t {
  fan::graphics::sprite_t visual;
  fan::physics::entity_t entity;
};

std::vector<checkpoint_t> player_checkpoints;

std::vector<fan::graphics::sprite_t> lamp_sprites;
std::vector<fan::graphics::light_t> lights, lights_boss;
std::vector<fan::auto_color_transition_t> flicker_anims;

fan::graphics::sprite_t background {{
  .position = fan::vec3(10000, 6010, 0),
  .size = fan::vec2(9192, 10000),
  .color = fan::color(0.6, 0.576, 1),
  .image = fan::graphics::image_t("images/background.png"),
  .tc_size = 1 * 300.0,
}};

fan::audio::piece_t
  audio_pickup_item{"audio/pickup.sac"};


bool is_entering_door = false;
bool is_boss_dead = false;