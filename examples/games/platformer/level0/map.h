#define MEASURE_TIME 1

#if MEASURE_TIME
  #define TIMER_START(name) fan::time::timer timer_##name{true}
  #define TIMER_PRINT(name) fan::printcl("Load:", #name ":", fan::to_string(timer_##name.millis(), 3) + "ms")
#else
  #define TIMER_START(name)
  #define TIMER_PRINT(name)
#endif

#include "loader.h"
#include "../spikes.h"
spike_spatial_t spike_spatial;
#include "../lamp_colors.h"

void load_enemies() {
  pile->enemy_list.clear();
  pile->renderer.iterate_marks(main_map_id, [&](tilemap_loader_t::fte_t::spawn_mark_data_t& d) -> bool {
    const auto& id = d.id;
    if (id.contains("enemy_skeleton")) pile->spawn_enemy<skeleton_t>(d.position);
    else if (id.contains("enemy_fly")) pile->spawn_enemy<fly_t>(d.position);
    else if (id.contains("boss_skeleton")) boss_position = d.position;
    return false;
  });
}

template <typename T>
bool handle_pickupable(const std::string& id, T& who) {
  auto& body = who.get_body();
  switch (fan::get_hash(id)) {
    case fan::get_hash("pickupable_health"):
      if (body.get_health() >= body.get_max_health() || body.get_health() <= 0) return false;
      body.set_health(body.get_health() + 10.f);
      break;

    case fan::get_hash("pickupable_health_potion"):
    if constexpr (!std::is_same_v<T, player_t>) {
      return false;
    }
    {
      auto& reg = fan::graphics::gameplay::items::get_registry();
      auto item = reg.create_item(items::id_e::health_potion);
      pile->get_gui().inventory.add_item(item, 1);
    }
    break;

    default: return false;
  }

  audio_pickup_item.play();
  return true;
}

void start_lights(uint32_t index) {
  auto add_particles = [&](uint32_t i) {
    boss_torch_particles.emplace_back(torch_particles);
    auto& p = boss_torch_particles.back();
    p.start_particles();
    p.set_position(lights_boss[i].get_position().offset_y(boss_light_adjustment_y).offset_z(1));
  };

  if (index + 1 >= lights_boss.size()) {
    auto* shape = pile->renderer.get_light_by_id(main_map_id, "boss_room_ambient_light");
    boss_room_light.start_once(
      shape->get_color(),
      boss_room_target_color,
      1.f,
      [shape](fan::color c) { shape->set_color(c); }
    );
  }

  if (index >= lights_boss.size()) {
    typename decltype(pile->enemy_list)::nr_t nr; nr.gint() = boss_nr;
    std::visit([&](auto& e) {
      if constexpr (requires { e.allow_move; }) {
        e.allow_move = true;
        e.render_health_bar = true;
        fan::audio::stop(pile->audio_background_play_id);
        audio_skeleton_lord.play_looped();
      }
    }, pile->enemy_list[nr]);

    for (uint32_t i = 0; i < light_lights.size(); ++i) {
      auto base = lights_boss[i].get_color();
      light_lights[i] = fan::auto_color_transition_t{};
      light_lights[i].start(
        base * fan::color(1.f, 0.7f, 0.7f),
        base * fan::color(1.f, 1.f, 1.f),
        0.2f + fan::random::value(0.f, 0.5f),
        [&, i](fan::color c) { lights_boss[i].set_color(c); }
      );
    }
    return;
  }

  light_lights[index] = fan::auto_color_transition_t{};
  add_particles(index);

  light_lights[index].on_end = [&, index] {
    start_lights(index + 1);
  };

  light_lights[index].start_once(
    lights_boss[index].get_color(),
    lamp2_color,
    1.f,
    [&, index](fan::color c) { lights_boss[index].set_color(c); }
  );
}

void enter_boss() {
  boss_door_collision = pile->engine.physics_context.create_rectangle(boss_door_position, boss_door_size);
  is_entering_door = false;
  boss_nr = pile->spawn_enemy<boss_skeleton_t>(boss_position).gint();
  light_lights.resize(lights_boss.size());
  start_lights(0);
}

void reload_boss_door_collision() {
  if (boss_sensor) return;

  pile->renderer.iterate_physics_entities(main_map_id, [&](auto& data, auto& ent) -> bool {
    const auto& id = data.id;
    if (id.contains("sensor_enter_boss")) boss_sensor = ent;
    else if (id.contains("boss_door_collision")) {
      boss_door_position = ent.get_position();
      boss_door_size = ent.get_size();
      boss_door_particles = fan::graphics::shape_from_json("effects/boss_spawn.json");
      boss_door_particles.set_position(fan::vec3(boss_door_position, 0xFAAA / 2 - 2 + boss_door_particles.get_position().z));
      boss_door_particles.start_particles();
    }
    return false;
  });
}

void enter_portal() {
  static fan::auto_color_transition_t anim;
  pile->stage_transition = fan::graphics::rectangle_t(
    pile->player.body.get_position().offset_z(1),
    fan::graphics::gui::get_window_size(),
    fan::colors::black.set_alpha(0)
  );
  anim.on_end = [this] {
    pile->stage_transition.remove();
    pile->stage_loader.erase_stage(stage_common.stage_id);
    pile->level_stage.sic();
  };
  anim.start_once(
    fan::colors::black.set_alpha(0),
    fan::colors::black,
    1.f,
    [](fan::color c) { pile->stage_transition.set_color(c); }
  );
}

void update() {
  pile->renderer.update(main_map_id, pile->player.body.get_position());
  auto keys = pile->engine.input_action.get_all_keys(actions::interact);
  std::string text = "Press '" + fan::join_keys(keys, " / ") + "' to interact";

  if (interact_prompt.type != interact_type::none) {
    auto ws = fan::graphics::gui::get_window_size();
    auto ts = fan::graphics::gui::get_text_size(text);
    if (auto hud = fan::graphics::gui::hud("Interact hud"))
      fan::graphics::gui::text_box_at(text, fan::vec2(ws.x * 0.5f - ts.x * 0.5f, ws.y * 0.85f));
  }

  if (interact_prompt.type != interact_type::none &&
      fan::window::is_input_action_active(actions::interact))
  {
    switch (interact_prompt.type) {
      case interact_type::boss_door:
        pile->renderer.erase_physics_entity(main_map_id, "boss_door_collision");
        boss_sensor.destroy(); boss_sensor.invalidate();
        is_entering_door = true;
        break;
      case interact_type::portal:
        enter_portal();
        break;
      default: break;
    }
  }

  if (is_boss_dead && send_elevator_down_initially) {
    cage_elevator.start();
    send_elevator_down_initially = false;
    audio_elevator_chain.play_looped();
  }

  cage_elevator.update(pile->player.body);
  auto p = cage_elevator.visual.get_position();
  auto s = cage_elevator.visual.get_size();
  cage_elevator_chain.set_position(fan::vec2(p.x, p.y - s.y - cage_elevator_chain.get_size().y));

  for (auto [i, lamp] : fan::enumerate(lamp_sprites)) {
    if (i < lights.size()) {
      fan::color tint(0.9f, 0.9f, 0.6f, 1.f);
      lights[i].set_color(lamp_colors[lamp.get_current_animation_frame() % std::size(lamp_colors)] * tint * 2.f);
    }
  }

  if (!pile->engine.render_console) {
    if (pile->engine.input_action.is_clicked(fan::actions::toggle_settings))
      pile->pause = !pile->pause;

    if (fan::window::is_key_down(fan::key_left_control) &&
        fan::window::is_key_pressed(fan::key_t)) {
      reload_map();
      return;
    }
  }
}

std::vector<fan::auto_color_transition_t> light_lights;
fan::graphics::shape_t torch_particles = fan::graphics::shape_from_json("effects/torch.json");
inline static constexpr fan::color lamp2_color = fan::color::from_rgb(0x114753) * 4.f;
inline static constexpr f32_t boss_light_adjustment_y = 30.f;

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

std::vector<fan::physics::entity_t> spike_sensors;
std::unordered_map<fan::vec2i, fan::graphics::sprite_t> dropped_pickupables;
std::vector<fan::physics::entity_t> tile_collisions;

std::vector<fan::graphics::sprite_t> axes;
std::vector<fan::physics::entity_t> axe_collisions;

struct checkpoint_t { fan::graphics::sprite_t visual; fan::physics::entity_t entity; };

std::vector<fan::graphics::sprite_t> lamp_sprites;
std::vector<fan::graphics::light_t> lights, lights_boss, static_lights;
std::vector<fan::auto_color_transition_t> flicker_anims;

fan::graphics::sprite_t background {{
  .position = fan::vec3(10000, 6010, 0),
  .size = fan::vec2(9192, 10000),
  .color = fan::color(0.6, 0.576, 1),
  .image = fan::graphics::image_t("images/background.webp"),
  .tc_size = 300.0,
}};

fan::audio::sound_t
  audio_pickup_item{"audio/pickup.sac"},
  audio_elevator_chain{"audio/chain.sac"},
  audio_skeleton_lord{"audio/skeleton_lord_music.sac"};

fan::graphics::physics::elevator_t cage_elevator;
fan::graphics::sprite_t cage_elevator_chain;
fan::vec2 boss_elevator_end = 0;
bool send_elevator_down_initially = true;

fan::graphics::sprite_t portal_sprite;
fan::graphics::shape_t portal_particles;
fan::auto_color_transition_t portal_light_flicker;
fan::physics::entity_t portal_sensor;
fan::physics::step_callback_nr_t physics_step_nr;
fan::graphics::gameplay::pickupable_spatial_t pickupable_spatial;

inline static std::unordered_set<fan::vec2i> collected_pickupables;

enum class interact_type { none, boss_door, portal };

struct interact_prompt_t { interact_type type = interact_type::none; } interact_prompt;

bool is_entering_door = false;
bool is_boss_dead = false;