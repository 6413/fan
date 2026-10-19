void load_map() {
  fan::time::lap_timer_t laps("load_map", measure_load_time);

  spike_spatial.clear();
  laps.lap("spike_spatial_clear");

  pickupable_spatial.init(
    fan::vec2(0),
    pile->get_compiled(stage_name).map_size * pile->get_compiled(stage_name).tile_size * 2.f
  );
  laps.lap("pickupable_spatial_init");

  torch_particles.set_position(fan::vec2(-0xfffff));
  laps.lap("torch_particles_move");

  fan::vec2i render_size(16, 9);
  tilemap_loader_t::properties_t p;
  p.size = render_size;
  pile->engine.set_cull_padding(100);
  p.position = pile->player.body.get_position();
  laps.lap("engine_setup");

  main_map_id = pile->renderer.add(&pile->get_compiled(stage_name), p);
  laps.lap("renderer_add");

  pile->engine.get_lighting().set_target(pile->get_compiled(stage_name).lighting.ambient, 0.01);
  laps.lap("lighting_set_target");

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
  laps.lap("static_animations_setup");

  pile->checkpoint_system.load_from_map(pile->renderer, main_map_id, [](auto& visual, auto& entity) {
    checkpoint_flag.set_position(entity.get_position());
    visual = checkpoint_flag;
    checkpoint_flag.set_position(fan::vec2(-0xfffff));
    visual.set_size(checkpoint_flag.get_size() / fan::vec2(1.5f, 1.0f));
    visual.play_sprite_sheet();
  });
  laps.lap("checkpoint_system_load");

  reload_boss_door_collision();
  laps.lap("reload_boss_door");

  pile->renderer.iterate_marks(main_map_id, [&](tilemap_loader_t::fte_t::spawn_mark_data_t& data) -> bool {
    const auto& id = data.id;
    if (id.contains("lamp1")) {
      lamp_sprites.emplace_back(lamp1_anim);
      auto& l = lamp_sprites.back();

      l.set_current_sprite_sheet_frame(fan::random::value(0, l.get_current_sprite_sheet_frame_count()));
      l.set_position(fan::vec3(fan::vec2(data.position) + fan::vec2(1.f, -2.f), 1));
      lights.emplace_back(fan::graphics::light_t {{
        .position = l.get_position(),
        .size = light_size
      }});
    }
    else if (id.contains("lamp2")) {
      boss_torch_particles.emplace_back(torch_particles);
      auto& l = boss_torch_particles.back();
      l.start_particles();
      l.set_position(data.position.offset_y(boss_light_adjustment_y));
      static_lights.emplace_back(fan::graphics::light_t {{
        .position = l.get_position(),
        .size = light_size,
        .color = lamp2_color
      }});
    }
    else if (id.contains("boss_lamp")) {
      lights_boss.emplace_back(fan::graphics::light_t {{
        .position = data.position,
        .size = light_size,
        .color = fan::colors::black,
      }});
    }
    else if (id.contains("boss_elevator_begin")) {
      fan::graphics::image_t image = fan::graphics::image_load("images/cage.webp", fan::graphics::image_presets::pixel_art());
      fan::vec3 v = data.position;
      static constexpr f32_t elevator_landing_offset_y = 45.f;
      v.y += elevator_landing_offset_y;

      fan::vec2 start_pos = fan::vec2(v.x, v.y - 1024.f);
      fan::vec2 end_pos = fan::vec2(v.x, v.y);
      f32_t elevator_duration = 3.f;
      cage_elevator.init(fan::graphics::sprite_t(fan::vec3(start_pos, v.z + 1), image.get_size() * 1.5f, image), start_pos, end_pos, elevator_duration);
      fan::vec3 pos = cage_elevator.visual.get_position();
      fan::vec2 size = cage_elevator.visual.get_size();
      fan::graphics::image_t chain_image("images/chain.webp", fan::graphics::image_presets::pixel_art_repeat());
      cage_elevator_chain = fan::graphics::sprite_t(pos.offset_y(-size.y).offset_z(-1), fan::vec2(32, 512), chain_image);
      cage_elevator_chain.set_dynamic();
      size = cage_elevator_chain.get_size();
      cage_elevator_chain.set_tc_size(fan::vec2(1.f, size.y / chain_image.get_size().y));

      cage_elevator.on_start_cb = [this] {
        audio_elevator_chain.play_looped();
      };
      cage_elevator.on_end_cb = [init = true, this] mutable {
        audio_elevator_chain.stop();
        if (!init) return;
        init = false;
        fan::vec2 top = cage_elevator.visual.get_position();
        cage_elevator.start_position = top;
        cage_elevator.end_position = boss_elevator_end.offset_y(-elevator_landing_offset_y / 2.2f);
        cage_elevator.going_up = true;
        cage_elevator.duration = 20.f;
      };
    }
    else if (id.contains("boss_elevator_end")) {
      boss_elevator_end = data.position;
    }
    else if (id.contains("exit_world")) {
      fan::graphics::image_t image("images/portal.webp", fan::graphics::image_presets::pixel_art());

      fan::vec2 image_size = image.get_size();

      portal_sprite = fan::graphics::sprite_t(
        data.position,
        fan::vec2(256) * image_size.normalize(),
        image
      );

      fan::vec3 pos = portal_sprite.get_position();
      fan::vec2 size = portal_sprite.get_size();

      static_lights.emplace_back(fan::graphics::light_t {{
        .position = pos.offset_y(-size.y / 2.f),
        .size = fan::vec2(200, 200),
        .color = fan::color::from_rgb(0x008fbb) * 4.f
      }});

      static_lights.emplace_back(fan::graphics::light_t {{
        .position = pos.offset_y(size.y / 2),
        .size = fan::vec2(400, 256),
        .color = fan::color::from_rgb(0x008fbb) * 4.f,
        .flags = 1
      }});

      portal_light_flicker.start(
        fan::color::from_rgb(0x008fbb) * 4.f * (fan::color(1.0f, 0.7f, 0.7f) / 1.0f),
        fan::color::from_rgb(0x008fbb) * 4.f * (fan::color(1.0f, 1.0f, 1.0f) * 1.0f),
        3.f,
        [this, idx = static_lights.size() - 1](fan::color c) {
        static_lights[idx].set_color(c);
        static_lights[idx - 1].set_color(c);
      },
        fan::ease_e::pulse
      );

      portal_particles = fan::graphics::shape_from_json("effects/portal.json");

      portal_particles.set_position(pos.offset_z(1).offset_y(size.y / 4.f));
      portal_particles.start_particles();

      portal_sensor = pile->engine.get_physics_context().create_sensor_rectangle(
        pos,
        fan::vec2(size.x / 2.5f, size.y)
      );
    }
    return false;
  });
  laps.lap("iterate_marks");

  pile->renderer.iterate_visual(main_map_id, [&](tilemap_loader_t::tile_t& tile) -> bool {
    const std::string& id = tile.id;

    if (id.contains("roof_chain")) {}
    else if (id.contains("trap_axe")) {
      axes.emplace_back(axe_anim);
      axes.back().set_position(fan::vec3(fan::vec2(tile.position), 3));
    }
    else if (id.contains("pickupable_")) {
      if (collected_pickupables.count(fan::vec2i(tile.position))) {
        pile->renderer.remove_visual(main_map_id, id, tile.position);
        return false;
      }
      fan::physics::body_id_t sensor = fan::physics::create_sensor_rectangle(tile.position, tile.size / 1.2f);
      pickupable_spatial.add(id, sensor);
    }
    else if (id.contains("spikes")) {
      auto pts = spike_spatial_t::get_spike_points(id.substr(std::strlen("spikes_")));
      spike_sensors.emplace_back(
        pile->engine.get_physics_context().create_polygon(
          tile.position,
          0.0f,
          pts.data(),
          pts.size(),
          fan::physics::body_type_e::static_body,
          {.is_sensor = true}
        )
      );
      spike_spatial.add(spike_sensors.back());
    }
    else if (id.contains("no_collision")) {
      return false;
    }
    else if (tile.mesh_property == tilemap_loader_t::fte_t::mesh_property_t::none) {
      tile_collisions.emplace_back(
        pile->engine.get_physics_context().create_rectangle(
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
  laps.lap("iterate_visual");

  pile->player.respawn();
  laps.lap("player_respawn");

  {
    auto* boss_room_light = pile->renderer.get_light_by_id(main_map_id, "boss_room_ambient_light");
    boss_room_target_color = boss_room_light->get_color();
    boss_room_light->set_color(fan::colors::black);
  }
  laps.lap("boss_room_light_setup");
}

void open(void* sod) {
  fan::time::lap_timer_t laps("open", measure_load_time);
  torch_particles = fan::graphics::shape_from_json("effects/torch.json");
  bg = fan::graphics::shapes_from_json("images/background.json");
  audio_pickup_item = fan::audio::sound_t("audio/pickup.sac");
  audio_elevator_chain = fan::audio::sound_t("audio/chain.sac");
  audio_skeleton_lord = fan::audio::sound_t("audio/skeleton_lord_music.sac");
  laps.lap("assets");

  pile->level_stage = this->stage_common.stage_id;
  laps.lap("setup");

  load_map();
  laps.lap("load_map");

  pile->engine.get_lighting().set_target(0, 0);
  is_entering_door = false;
  laps.lap("lighting");

  physics_step_nr = fan::physics::add_physics_step_callback([this]() {

    std::string enter_text;
    auto keys = pile->engine.get_input_action().get_all_keys(actions::interact);

    enter_text = "Press '";

    for (int i = 0; i < keys.size(); i++) {
      enter_text += fan::get_key_name(keys[i]);
      if (i + 1 < keys.size()) {
        enter_text += " / ";
      }
    }

    enter_text += "' to interact";

    interact_prompt.type = interact_type::none;

    if (boss_sensor && fan::physics::is_on_sensor(pile->player.body, boss_sensor)) {
      interact_prompt.type = interact_type::boss_door;
    }

    fan::vec2 player_pos = pile->player.body.get_position();
    auto nearby_indices = pickupable_spatial.query_radius(player_pos, 128.f);

    for (auto idx : nearby_indices) {
      auto* pickup = pickupable_spatial.get(idx);
      if (!pickup) continue;

      if (fan::physics::is_on_sensor(pile->player.body, pickup->sensor)) {
        if (handle_pickupable(pickup->id, pile->player)) {
          fan::vec2 pos = pickup->sensor.get_position();
          collected_pickupables.insert(pos);

          pile->renderer.remove_visual(main_map_id, pickup->id, pos);

          auto found = dropped_pickupables.find(pos);
          if (found != dropped_pickupables.end()) {
            dropped_pickupables.erase(found);
          }

          pickup->sensor.destroy();
          pickupable_spatial.remove(idx);
          break;
        }
      }
    }

    if (spike_spatial.query(pile->player.body)) {
      reload_map();
      return;
    }

    for (auto& enemy : pile->enemies()) {
      if (spike_spatial.query(enemy.get_body())) {
        enemy.destroy();
      }
    }

    if (fan::physics::is_on_sensor(pile->player.body, portal_sensor)) {
      interact_prompt.type = interact_type::portal;
    }
  });
  laps.lap("physics_callback");

  pile->renderer.update(main_map_id, pile->player.body.get_position());
  laps.lap("renderer_update");
}

void close() {
  fan::time::lap_timer_t laps("close", measure_load_time);
  collected_pickupables.clear();

  pile->enemy_list.clear();
  laps.lap("enemy_clear");

  cage_elevator.destroy();
  laps.lap("cage_elevator");

  if (boss_door_collision) {
    boss_door_collision.destroy();
  }
  laps.lap("boss_door");
  if (boss_sensor) {
    boss_sensor.destroy();
  }
  if (portal_sensor) {
    portal_sensor.destroy();
  }
  
  for (auto& i : tile_collisions) {
    i.destroy();
  }
  tile_collisions.clear();
  laps.lap("tile_collisions");

  for (auto& i : spike_sensors) {
    i.destroy();
  }
  spike_sensors.clear();
  laps.lap("spike_sensors");

  pile->renderer.erase(main_map_id);
  laps.lap("renderer_erase");

  pickupable_spatial.clear();
  laps.lap("pickupable_clear");
}

void reload_map() {
  fan::time::lap_timer_t laps("reload_map", measure_load_time);

  pile->stage_loader.close_stage(stage_common.stage_id);
  laps.lap("close_stage");

  pile->level_stage = pile->stage_loader.open_stage<self_t>();
  laps.lap("open_stage");
}