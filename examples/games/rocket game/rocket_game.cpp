import std;
import fan;

using namespace fan::graphics;

enum class game_state_e { menu, playing, win, game_over };

struct rocket_game_t : engine_t, fan::frame_task_t<rocket_game_t> {
  struct game_stage_t : fan::stage_t<game_stage_t> {
    struct config {
      static constexpr f32_t safe_landing_speed = 400.f;
      static constexpr f32_t safe_landing_angle = 0.5f;
      static constexpr f32_t landing_duration = 1.f;
      static constexpr f32_t thrust_power = 15.f;
      static constexpr f32_t turn_speed = 3.14159f;
    };

    void open(void* data) {
      state = (data && *(bool*)data) ? game_state_e::menu : game_state_e::playing;
      fan::vec2 view_size = pile.viewport_get_size();

      std::string json_str;
      if (fan::io::file::read("map0.json", &json_str)) {
        fan::json map_data = fan::json::parse(json_str);
        fan::graphics::shape_deserialize_t iterator;
        fan::graphics::shape_t shape;
        while (iterator.iterate(map_data["shapes"], &shape)) {
          fan::vec2 pos = shape.get_position().xy(), size = shape.get_size();
          if (iterator.current_json->value("id", "") == "landing_pad") {
            pad_shape = {fan::vec3(pos, 3500), size, fan::colors::transparent, fan::physics::body_type_e::static_body};
            fan::vec3 p = pad_shape.get_position();
            pad_visual = {fan::vec3(p.x, p.y, 2500), fan::vec2(size.x * 1.2f, size.y * 0.4f), fan::color(0.3f, 0.5f, 0.3f, 0.8f)};
            pad_stripe = {fan::vec3(p.x, p.y - 2.f, 2501), fan::vec2(size.x * 0.8f, size.y * 0.2f), fan::color(1.f, 1.f, 1.f, 0.3f)};
            pad_light = {fan::vec3(p.x, p.y + 2.f, 2499), 6.f, fan::color(0.5f, 1.f, 0.5f, 0.2f)};
          } else {
            terrain_shapes.emplace_back(physics::rectangle_t{fan::vec3(pos, 4000), size, shape.get_color(), fan::physics::body_type_e::static_body});
          }
        }
      }

      fan::vec2 start_pos = pad_shape.is_valid() ? fan::vec2(pad_shape.get_position().x, pad_shape.get_position().y - 1200.f) : fan::vec2(view_size.x / 2.f, view_size.y * 0.1f);
      rocket_shape = physics::capsule_t{fan::vec3(start_pos, 4000), fan::vec2(0, -15), fan::vec2(0, 15), 15.f, fan::colors::transparent, state == game_state_e::menu ? fan::physics::body_type_e::static_body : fan::physics::body_type_e::dynamic_body};
      
      ship_body = capsule_t{fan::vec3(start_pos, 5000), fan::vec2(0, -15), fan::vec2(0, 15), 15.f, fan::color(0.85f, 0.88f, 0.92f, 1.f) / 5.f};
      ship_window = circle_t{ship_body.get_position() + fan::vec3(0, -8, 10), 7.f, fan::color(0.3f, 0.7f, 1.f, 1.f)};
      ship_nozzle = rectangle_t{ship_body.get_position() + fan::vec3(0, 30, -10), fan::vec2(14.f, 6.f), fan::color(0.25f, 0.25f, 0.28f, 1.f)};
      ship_body.set_dynamic(); ship_window.set_dynamic(); ship_nozzle.set_dynamic();
      ship_body.add_child(ship_window); ship_body.add_child(ship_nozzle);

      auto handle_impact = [&](fan::physics::entity_t other, f32_t speed = 0.f) {
        if (state != game_state_e::playing) return;
        if (other == pad_shape && speed < config::safe_landing_speed && std::fmod(std::abs(rocket_shape.get_angle().z), fan::math::pi * 2.f) < config::safe_landing_angle) {
          is_landing = true; landing_timer = config::landing_duration; return;
        }
        particles.spawn_explosion(rocket_shape.get_position(), fan::colors::orange, 200, particle_image);
        state = game_state_e::game_over; rocket_shape.set_body_type(fan::physics::body_type_e::static_body);
      };

      collision_scope.on_enter(rocket_shape, [&](fan::physics::entity_t other) { handle_impact(other); });
      collision_scope.on_hit(rocket_shape, handle_impact);
      collision_scope.on_exit(rocket_shape, [&](fan::physics::entity_t other) { if (state == game_state_e::playing && other == pad_shape) is_landing = false; });
      pile.camera_follow(rocket_shape.get_position());
    }

    void draw_gui() {
      using namespace fan::graphics::gui;
      if (auto h = hud_interactive{"##game_ui"}) {
        fan::vec2 vs = pile.viewport_get_size();
        push_font(get_font(24, font::bold));
        
        f32_t speed = rocket_shape.is_valid() ? rocket_shape.get_linear_velocity().length() : 0.f;
        f32_t angle = rocket_shape.is_valid() ? std::fmod(std::abs(rocket_shape.get_angle().z), fan::math::pi * 2.f) : 0.f;
        
        set_cursor_screen_pos(fan::vec2(20.f, 20.f));
        text(speed < config::safe_landing_speed ? fan::colors::green : fan::colors::red, std::format("VELOCITY: {}", (int)speed));
        set_cursor_screen_pos(fan::vec2(20.f, 50.f));
        text(angle < config::safe_landing_angle ? fan::colors::green : fan::colors::red, std::format("ANGLE: {:.2f} rad", angle));
        pop_font();

        if (state != game_state_e::playing) {
          bool menu = state == game_state_e::menu, over = state == game_state_e::game_over;
          push_font(get_font(72, font::bold));
          set_cursor_screen_pos(fan::vec2(vs.x / 2.f, vs.y / 2.f - 150.f));
          text(menu ? "ROCKET GAME" : (over ? "CRASHED!" : "LANDED!"), {.color = menu ? fan::colors::white : (over ? fan::colors::red : fan::colors::green), .align = align_e::center});
          pop_font();

          set_cursor_screen_pos(fan::vec2(vs.x / 2.f - 100.f, vs.y / 2.f + 50.f));
          if (button(menu ? "Start" : (over ? "Restart" : "Play Again"), fan::vec2(200.f, 60.f)) || pile.is_key_clicked(fan::key_space)) {
            menu ? (state = game_state_e::playing, rocket_shape.set_body_type(fan::physics::body_type_e::dynamic_body)) : pile.stage_restart<game_stage_t>();
          }
        } else if (is_landing) {
          push_font(get_font(48, font::bold));
          set_cursor_screen_pos(fan::vec2(vs.x / 2.f, vs.y / 2.f - 200.f));
          text(std::format("LANDING: {:.1f}s", landing_timer), {.color = fan::colors::yellow, .align = align_e::center});
          pop_font();
        }
      }
    }

    void update() {
      f32_t dt = pile.get_delta_time();
      if (rocket_shape.is_valid()) {
        pile.camera_set_center(rocket_shape.get_position().xy());
        ship_body.set_position(fan::vec3(rocket_shape.get_position().xy(), 5000));
        ship_body.set_angle(fan::vec3(0, 0, rocket_shape.get_angle().z));
      }

      if (state == game_state_e::playing) {
        if (is_landing) {
          if (rocket_shape.get_linear_velocity().length() >= config::safe_landing_speed || std::fmod(std::abs(rocket_shape.get_angle().z), fan::math::pi * 2.f) >= config::safe_landing_angle) {
            particles.spawn_explosion(rocket_shape.get_position(), fan::colors::orange, 200, particle_image);
            state = game_state_e::game_over; rocket_shape.set_body_type(fan::physics::body_type_e::static_body); is_landing = false;
          } else if ((landing_timer -= dt) <= 0.f) {
            state = game_state_e::win; rocket_shape.set_body_type(fan::physics::body_type_e::static_body); is_landing = false;
          }
        }

        fan::vec2 input = pile.get_input_vector();
        rocket_shape.set_angular_velocity(input.x * config::turn_speed);
        if (input.y < 0.f) {
          fan::vec2 dir(std::sin(rocket_shape.get_angle().z), -std::cos(rocket_shape.get_angle().z));
          rocket_shape.apply_linear_impulse_center(dir * config::thrust_power * rocket_shape.get_mass() * dt);
          if (int fc = (fire_accumulator += dt * 180.f); fc > 0) { particles.spawn_fire(rocket_shape.get_position() - fan::vec3(dir * 35.f, 0), fc, fire_image); fire_accumulator -= fc; }
          if (int sc = (smoke_accumulator += dt * 60.f); sc > 0) { particles.spawn_smoke(rocket_shape.get_position() - fan::vec3(dir * 35.f, 0), sc, particle_image); smoke_accumulator -= sc; }
        }
      }

      particles.update(dt);
      draw_gui();
    }

    game_state_e state = game_state_e::menu;
    bool is_landing = false;
    f32_t landing_timer = 0.f, fire_accumulator = 0.f, smoke_accumulator = 0.f;

    gradient_t sky{fan::vec3(0.f, 0.f, 0), fan::vec2(100000.f, 100000.f), {fan::color(0.005f, 0.005f, 0.02f, 1.f), fan::color(0.005f, 0.005f, 0.02f, 1.f), fan::color(0.02f, 0.05f, 0.15f, 1.f), fan::color(0.15f, 0.08f, 0.12f, 1.f)}};

    std::vector<physics::rectangle_t> terrain_shapes;
    physics::rectangle_t pad_shape; rectangle_t pad_visual; rectangle_t pad_stripe; circle_t pad_light;

    physics::capsule_t rocket_shape;
    capsule_t ship_body; circle_t ship_window; rectangle_t ship_nozzle;
    
    gpu_particle_system_t<> particles;
    image_t particle_image{"images/smoke.webp"}; image_t fire_image{"images/circle.png"};
    physics::collision_scope_t collision_scope;
  };

  rocket_game_t() {
    update_physics(true);
    set_settings({.clear_color = fan::colors::black, .ambient_color = fan::colors::white, .mode = fan::graphics::post_process_mode_e::none, .bloom_strength = 1.0f, .bloom_threshold = 0.241f, .bloom_knee = 0.264f, .bloom_tint = fan::vec3(1.f, 1.f, 1.f), .bloom_filter_radius = 0.299f, .blur_amount = 0.049f, .blur_filter_radius = 0.027f, .blur_focus_enabled = false, .gamma = 1.0f, .exposure = 1.0f, .contrast = 1.0f});
    bool is_menu = true;
    stage_open<game_stage_t>(&is_menu);
  }
} pile;

int main() {
  pile.loop();
}