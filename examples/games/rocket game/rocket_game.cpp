import std;
import fan;
import fan.graphics.scene;
import fan.physics.vehicle_controller;

using namespace fan::graphics;
using namespace fan::color_literals;

enum class game_state_e { menu, playing, win, game_over };

struct rocket_game_t : engine_t, fan::frame_task_t<rocket_game_t> {
  struct game_stage_t : fan::stage_t<game_stage_t> {
    struct config {
      static constexpr f32_t safe_landing_speed = 500.f;
      //static constexpr f32_t safe_landing_angle = fan::math::pi / 4.f;
      static constexpr f32_t landing_duration = 1.f;
      static constexpr f32_t thrust_power = 18.f;
      static constexpr f32_t turn_speed = fan::math::pi*1.5f;
    };

    void open(void* data) {
      state = (data && *(bool*)data) ? game_state_e::menu : game_state_e::playing;
      
      fan::vec2 view_size = engine->viewport_get_size();
      engine->camera_set_ortho(
        engine->orthographic_render_view.camera, 
        fan::vec2(-view_size.x / 2.f, view_size.x / 2.f), 
        fan::vec2(-view_size.y / 2.f, view_size.y / 2.f)
      );

      scene.load(current_level == 0 ? "map0.json" : "map1.json");
      pad_shape = scene.get_physics_body<physics::sprite_t>("landing_pad");
      auto* start_pad = scene.get_physics_body<physics::sprite_t>("start_pad");
      
      f32_t ship_radius = 15.f;
      fan::vec2 nozzle_size(14.f, 6.f);
      fan::vec2 start_pos = start_pad->get_position().offset_y(-start_pad->get_size().y - ship_radius);
      

      rocket_shape = physics::capsule_t{
        fan::vec3(start_pos, 4000), 
        fan::vec2(0, -ship_radius), 
        fan::vec2(0, ship_radius), 
        ship_radius, 
        fan::color(0.85f, 0.88f, 0.92f, 1.f) / 5.f, 
        state == game_state_e::menu ? fan::physics::body_type_e::static_body : fan::physics::body_type_e::dynamic_body
      };
      
      ship_window = circle_t{fan::vec3(0), 7.f, fan::color(0.3f, 0.7f, 1.f, 1.f)};
      ship_nozzle = rectangle_t{fan::vec3(0), nozzle_size, fan::color(0.25f, 0.25f, 0.28f, 1.f)};
      
      rocket_shape.add_child(ship_window); 
      rocket_shape.add_child(ship_nozzle);
      
      ship_window.set_local_position(fan::vec3(0, -8, 10));
      ship_nozzle.set_local_position(fan::vec3(0, 30, -10));

      flame_light = light_t{
        fan::vec3(0), 
        fan::vec2(200.f), 
        (fan::colors::orange * 3.f).set_alpha(1.f)
      };
      rocket_shape.add_child(flame_light);
      flame_light.set_local_position(fan::vec3(0, 35, -20));

      flame_light_pulse.start(
        fan::vec2(80.f), fan::vec2(500.f),
        0.08f,
        [this](const fan::vec2& s) { flame_light.set_size(s); },
        fan::ease_e::pulse
      );
      flame_light_pulse.stop(fan::vec2(0.f));

      flame_color_pulse.start(
        (fan::colors::orange * 2.f).set_alpha(1.f),
        (fan::colors::orange * 5.f).set_alpha(1.f),
        0.06f,
        [this](const fan::color& c) { flame_light.set_color(c); },
        fan::ease_e::pulse
      );
      flame_color_pulse.stop(fan::colors::transparent);

      rocket_controller.bind(&rocket_shape);

      auto on_pad_touch = [this](fan::physics::entity_t other) {
        if (state != game_state_e::playing) return;
        if (!pad_shape || static_cast<fan::physics::body_id_t>(other) != static_cast<const fan::physics::entity_t&>(*pad_shape)) return;
        is_landing = true; 
        landing_timer = config::landing_duration; 
      };

      auto on_pad_hit = [this](fan::physics::entity_t other, f32_t speed) {
        if (state != game_state_e::playing) return;
        if (speed >= config::safe_landing_speed) {
          particles.spawn_explosion(rocket_shape.get_position(), fan::colors::orange, 200, particle_image);
          rocket_shape.set_visible(false);
          ship_window.set_visible(false);
          ship_nozzle.set_visible(false);
          flame_light.set_visible(false);
          state = game_state_e::game_over; 
          rocket_shape.set_body_type(fan::physics::body_type_e::static_body);
        }
      };

      collision_scope.on_enter(rocket_shape, on_pad_touch);
      collision_scope.on_hit(rocket_shape, on_pad_hit);
      collision_scope.on_exit(rocket_shape, [this](fan::physics::entity_t other) { 
        if (state == game_state_e::playing && pad_shape && static_cast<fan::physics::body_id_t>(other) == static_cast<const fan::physics::entity_t&>(*pad_shape)) {
          is_landing = false; 
        }
      });
      
      engine->camera_follow(rocket_shape.get_position());
    }

    void draw_gui() {
      using namespace fan::graphics::gui;
      if (auto h = hud_interactive{"##game_ui"}) {
        fan::vec2 vs = engine->viewport_get_size();
        push_font(get_font(24, font::bold));
        
        f32_t speed = rocket_shape.is_valid() ? rocket_shape.get_linear_velocity().length() : 0.f;
        f32_t angle = rocket_shape.is_valid() ? fan::math::wrap_angle(rocket_shape.get_angle().z): 0.f;
        
        set_cursor_screen_pos(fan::vec2(20.f, 20.f));
        text(speed < config::safe_landing_speed ? fan::colors::green : fan::colors::red, std::format("VELOCITY: {}", (int)speed));
        text(fan::colors::green, std::format("ANGLE: {:.2f} deg", fan::math::degrees(angle)));
        pop_font();

        if (state != game_state_e::playing) {
          bool menu = state == game_state_e::menu;
          bool over = state == game_state_e::game_over;
          bool won = state == game_state_e::win;
          bool last_level = current_level >= 1;
          
          push_font(get_font(72, font::bold));
          set_cursor_screen_pos(fan::vec2(vs.x / 2.f, vs.y / 2.f - 150.f));
          const char* title = menu ? "ROCKET GAME" : (over ? "CRASHED!" : (last_level ? "YOU WIN!" : "LANDED!"));
          text(title, {.color = menu ? fan::colors::white : (over ? fan::colors::red : fan::colors::green), .align = align_e::center});
          pop_font();

          const char* btn_text = menu ? "Start" : (over ? "Restart" : (last_level ? "Play Again" : "Next Level"));
          set_cursor_screen_pos(fan::vec2(vs.x / 2.f - 100.f, vs.y / 2.f + 50.f));
          if (button(btn_text, fan::vec2(200.f, 60.f)) || engine->is_key_clicked(fan::key_space)) {
            if (menu) {
              state = game_state_e::playing;
              rocket_shape.set_body_type(fan::physics::body_type_e::dynamic_body);
            } else if (won && !last_level) {
              current_level = 1;
              engine->stage_restart<game_stage_t>();
            } else {
              if (won) current_level = 0;
              engine->stage_restart<game_stage_t>();
            }
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
      f32_t dt = engine->get_delta_time();
      if (rocket_shape.is_valid()) {
        engine->camera_set_center(rocket_shape.get_position().xy());
      }

      if (engine->is_key_down(fan::key_left_control) && engine->is_key_down(fan::key_left_shift) && engine->is_key_clicked(fan::key_r)) {
        fan::image::async_cache().clear();
        engine->stage_restart<game_stage_t>();
      }

      if (state == game_state_e::playing) {
        if (is_landing) {
          if (rocket_shape.get_linear_velocity().length() >= config::safe_landing_speed) 
          {
            particles.spawn_explosion(rocket_shape.get_position(), fan::colors::orange, 200, particle_image);
            state = game_state_e::game_over; 
            rocket_shape.set_body_type(fan::physics::body_type_e::static_body); 
            is_landing = false;
          } 
          else if ((landing_timer -= dt) <= 0.f) 
          {
            state = game_state_e::win; 
            rocket_shape.set_body_type(fan::physics::body_type_e::static_body); 
            is_landing = false;
          }
        }

        fan::vec2 input = engine->get_input_vector();
        rocket_controller.apply_turn(input.x * config::turn_speed);

        if (input.y < 0.f) {
          rocket_controller.apply_thrust(config::thrust_power, dt);

          if (!flame_light_pulse.active) {
            flame_light_pulse.start(
              fan::vec2(8.f), fan::vec2(50.f),
              0.08f,
              [this](const fan::vec2& s) { flame_light.set_size(s); },
              fan::ease_e::pulse
            );
          }
          if (!flame_color_pulse.active) {
            flame_color_pulse.start(
              (fan::colors::orange * 2.f) / 50.f,
              (fan::colors::orange * 5.f) / 50.f,
              0.02f,
              [this](const fan::color& c) { flame_light.set_color(c); },
              fan::ease_e::pulse
            );
          }

          if (fan::time::every(5)) {
            particles.spawn_fire(rocket_controller.get_thrust_position(35.f), 1, fire_image);
          }
          if (fan::time::every(17)) {
            particles.spawn_smoke(rocket_controller.get_thrust_position(35.f), 1, particle_image); 
          }
        }
        else {
          if (flame_light_pulse.active) {
            flame_light_pulse.stop(fan::vec2(0.f));
          }
          if (flame_color_pulse.active) {
            flame_color_pulse.stop(fan::colors::transparent);
          }
        }
      }

      particles.update(dt);
      draw_gui();
    }

    inline static int current_level = 0;
    game_state_e state = game_state_e::menu;
    bool is_landing = false;
    f32_t landing_timer = 0.f;

    gradient_t sky{
      {0.f, 0.f, 0.f}, 
      {100000.f, 100000.f}, 
      std::array{0x010105_rgb, 0x010105_rgb, 0x050d26_rgb, 0x26141f_rgb} 
    };

    fan::graphics::scene_t scene;
    physics::sprite_t* pad_shape = nullptr;

    physics::capsule_t rocket_shape;
    circle_t ship_window; rectangle_t ship_nozzle;
    fan::physics::vehicle_controller_t rocket_controller;
    
    gpu_particle_system_t<> particles;
    image_t particle_image{"images/smoke.webp"}; 
    image_t fire_image{"images/circle.png"};

    light_t flame_light;
    fan::auto_vec2_transition_t flame_light_pulse;
    fan::auto_color_transition_t flame_color_pulse;

    physics::collision_scope_t collision_scope;
  };


  rocket_game_t() {
    update_physics(true);
    set_settings({
      .clear_color = fan::colors::black,
      .ambient_color = fan::colors::white,
      .mode = fan::graphics::post_process_mode_e::bloom_blur,
      .bloom_strength = 1.0f,
      .bloom_threshold = 0.241f,
      .bloom_knee = 0.264f,
      .bloom_tint = fan::vec3(1.f, 1.f, 1.f),
      .bloom_filter_radius = 0.299f,
      .blur_amount = 0.049f,
      .blur_filter_radius = 0.027f,
      .blur_focus_enabled = false,
      .gamma    = 1.0f,
      .exposure = 1.0f,
      .contrast = 1.0f
    });
    bool is_menu = true;
    stage_open<game_stage_t>(&is_menu);
  }
} pile;

int main() {
  pile.loop();
}