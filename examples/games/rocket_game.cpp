import std;
import fan;

using namespace fan::graphics;

enum class game_state_e { menu, playing, win, game_over };

struct ship_graphics_t {
  void init(fan::vec3 pos) {
    body = capsule_t(
      fan::vec3(pos.x, pos.y, 5000), 
      fan::vec2(0, -15), 
      fan::vec2(0, 15), 
      15.f, 
      fan::color(0.85f, 0.88f, 0.92f, 1.f) / 5.f
    );
    body.set_dynamic();
    pos = body.get_position();

    window = circle_t{
      pos + fan::vec3(0, -8, 10), 
      7.f, 
      fan::color(0.3f, 0.7f, 1.f, 1.f)
    };
    window.set_dynamic();

    nozzle = rectangle_t{
      pos + fan::vec3(0, 30, -10), 
      fan::vec2(14.f, 6.f), 
      fan::color(0.25f, 0.25f, 0.28f, 1.f)
    };
    nozzle.set_dynamic();

    body.add_child(window);
    body.add_child(nozzle);
  }

  void update(fan::vec3 pos, f32_t angle_z) {
    body.set_position(fan::vec3(pos.x, pos.y, 5000));
    body.set_angle(fan::vec3(0, 0, angle_z));
  }

  capsule_t body;
  circle_t window;
  rectangle_t nozzle;
};

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
      bool is_menu = data ? *(bool*)data : false;
      state = is_menu ? game_state_e::menu : game_state_e::playing;

      hills.clear();
      terrain_shapes.clear();
      stars.clear();
      far_mountains.clear();

      fan::vec2 view_size = pile.viewport_get_size();
      fan::vec2 start_pos(view_size.x / 2.f, view_size.y * 0.1f);

      rocket_shape = physics::capsule_t{
        fan::vec3(start_pos, 4000),
        fan::vec2(0, -15),
        fan::vec2(0, 15),
        15.f,
        fan::colors::transparent,
        is_menu ? fan::physics::body_type_e::static_body : fan::physics::body_type_e::dynamic_body
      };

      ship_graphics.init(rocket_shape.get_position());

      struct rect_t { fan::vec2 pos; fan::vec2 size; };
      std::vector<rect_t> terrain = {
        {fan::vec2(view_size.x / 2.f, view_size.y - 20.f), fan::vec2(view_size.x, 40.f)},
        {fan::vec2(20.f, view_size.y / 2.f), fan::vec2(40.f, view_size.y)},
        {fan::vec2(view_size.x - 20.f, view_size.y / 2.f), fan::vec2(40.f, view_size.y)},
        {fan::vec2(view_size.x * 0.7f, view_size.y * 0.6f), fan::vec2(200.f, 150.f)}
      };

      for (int i = 0; i < 30; ++i) {
        hills.emplace_back(circle_t{
          fan::vec3(fan::random::value(-1000.f, view_size.x + 1000.f), view_size.y + fan::random::value(330.f, 1200.f), 2500),
          fan::random::value(600.f, 1000.f),
          fan::color(0.06f, 0.08f, 0.12f, 1.f)
        });
      }

      for (const auto& t : terrain) {
        terrain_shapes.emplace_back(physics::rectangle_t{
          fan::vec3(t.pos, 4000),
          t.size / 2.f,
          fan::color(0.15f, 0.18f, 0.22f, 1.f),
          fan::physics::body_type_e::static_body
        });
      }

      pad_shape = physics::rectangle_t{
        fan::vec3(start_pos.x, view_size.y - 50.f, 3500),
        fan::vec2(50.f, 10.f),
        fan::colors::transparent,
        fan::physics::body_type_e::static_body
      };

      fan::vec3 pad_pos = pad_shape.get_position();

      pad_visual = rectangle_t{
        fan::vec3(pad_pos.x, pad_pos.y, 2500),
        fan::vec2(60.f, 4.f),
        fan::color(0.3f, 0.5f, 0.3f, 0.8f)
      };
      pad_stripe = rectangle_t{
        fan::vec3(pad_pos.x, pad_pos.y - 2.f, 2501),
        fan::vec2(40.f, 2.f),
        fan::color(1.f, 1.f, 1.f, 0.3f)
      };
      pad_light = circle_t{
        fan::vec3(pad_pos.x, pad_pos.y + 2.f, 2499),
        6.f,
        fan::color(0.5f, 1.f, 0.5f, 0.2f)
      };

      for (int i = 0; i < 500; ++i) {
        stars.emplace_back(circle_t{
          fan::vec3(fan::random::value(-3000.f, 5000.f), fan::random::value(-3000.f, 5000.f), 1000),
          fan::random::value(0.5f, 2.5f),
          fan::color(1.f, 1.f, 1.f, fan::random::value(0.2f, 0.9f))
        });
        stars.back().set_parallax_factor(0.1f);
      }

      for (int i = 0; i < 20; ++i) {
        far_mountains.emplace_back(circle_t{
          fan::vec3(fan::random::value(-2000.f, 5000.f), -fan::random::value(100.f, 500.f), 0.f),
          fan::random::value(200.f, 500.f),
          fan::color(0.03f, 0.04f, 0.08f, 0.6f)
        });
      }

      collision_scope.on_enter(rocket_shape, [&](fan::physics::entity_t other) {
        if (state != game_state_e::playing) return;
        if (other == pad_shape) {
          f32_t normalized_angle = std::fmod(std::abs(rocket_shape.get_angle().z), fan::math::pi * 2.f);
          if (normalized_angle < config::safe_landing_angle) {
            is_landing = true;
            landing_timer = config::landing_duration;
            return;
          }
        }
        particles.spawn_explosion(rocket_shape.get_position(), fan::colors::orange, 200, particle_image);
        state = game_state_e::game_over;
        rocket_shape.set_body_type(fan::physics::body_type_e::static_body);
      });

      collision_scope.on_hit(rocket_shape, [&](fan::physics::entity_t other, f32_t approach_speed) {
        if (state != game_state_e::playing) return;
        if (other == pad_shape) {
          f32_t normalized_angle = std::fmod(std::abs(rocket_shape.get_angle().z), fan::math::pi * 2.f);
          if (approach_speed < config::safe_landing_speed && normalized_angle < config::safe_landing_angle) {
            is_landing = true;
            landing_timer = config::landing_duration;
            return;
          }
        }
        particles.spawn_explosion(rocket_shape.get_position(), fan::colors::orange, 200, particle_image);
        state = game_state_e::game_over;
        rocket_shape.set_body_type(fan::physics::body_type_e::static_body);
      });

      collision_scope.on_exit(rocket_shape, [&](fan::physics::entity_t other) {
        if (state != game_state_e::playing) return;
        if (other == pad_shape) {
          is_landing = false;
        }
      });
      
      pile.camera_follow(rocket_shape.get_position());
    }

    void draw_gui() {
      using namespace fan::graphics::gui;
      if (auto h = hud_interactive{"##game_ui"}) {
        fan::vec2 view_size = pile.viewport_get_size();

        push_font(get_font(24, font::bold));
        set_cursor_screen_pos(fan::vec2(20.f, 20.f));

        fan::vec2 vel = rocket_shape.is_valid() ? rocket_shape.get_linear_velocity() : fan::vec2(0.f);
        f32_t speed = vel.length();
        fan::color vel_color = speed < config::safe_landing_speed ? fan::colors::green : fan::colors::red;
        text(vel_color, std::string("VELOCITY: ") + std::to_string((int)speed));

        f32_t angle = rocket_shape.is_valid() ? std::fmod(std::abs(rocket_shape.get_angle().z), fan::math::pi * 2.f) : 0.f;
        fan::color angle_color = angle < config::safe_landing_angle ? fan::colors::green : fan::colors::red;
        set_cursor_screen_pos(fan::vec2(20.f, 50.f));
        text(angle_color, std::format("ANGLE: {:.2f} rad", angle));
        pop_font();

        if (state != game_state_e::playing) {
          if (state == game_state_e::menu) {
            push_font(get_font(72, font::bold));
            set_cursor_screen_pos(fan::vec2(view_size.x / 2.f, view_size.y / 2.f - 150.f));
            text("ROCKET GAME", {
              .color = fan::colors::white,
              .align = align_e::center
            });
            pop_font();

            set_cursor_screen_pos(fan::vec2(view_size.x / 2.f - 100.f, view_size.y / 2.f + 50.f));
            if (button("Start", fan::vec2(200.f, 60.f)) || pile.is_key_clicked(fan::key_space)) {
              state = game_state_e::playing;
              rocket_shape.set_body_type(fan::physics::body_type_e::dynamic_body);
            }
          }
          else {
            bool crashed = (state == game_state_e::game_over);
            push_font(get_font(72, font::bold));
            set_cursor_screen_pos(fan::vec2(view_size.x / 2.f, view_size.y / 2.f - 150.f));
            text(crashed ? "CRASHED!" : "LANDED!", {
              .color = crashed ? fan::colors::red : fan::colors::green, 
              .align = align_e::center
            });
            pop_font();

            set_cursor_screen_pos(fan::vec2(view_size.x / 2.f - 100.f, view_size.y / 2.f + 50.f));
            if (button(crashed ? "Restart" : "Play Again", fan::vec2(200.f, 60.f)) || pile.is_key_clicked(fan::key_space)) {
              pile.stage_restart<game_stage_t>();
            }
          }
        } else if (is_landing) {
          push_font(get_font(48, font::bold));
          set_cursor_screen_pos(fan::vec2(view_size.x / 2.f, view_size.y / 2.f - 200.f));
          text(std::format("LANDING: {:.1f}s", landing_timer), {
            .color = fan::colors::yellow,
            .align = align_e::center
          });
          pop_font();
        }
      }
    }

    void update() {
      f32_t dt = pile.get_delta_time();
      if (rocket_shape.is_valid()) {
        pile.camera_set_center(rocket_shape.get_position().xy());
        ship_graphics.update(rocket_shape.get_position(), rocket_shape.get_angle().z);
      }

      bool thrusting = false;

      if (state == game_state_e::playing) {
        if (is_landing) {
          f32_t speed = rocket_shape.get_linear_velocity().length();
          f32_t normalized_angle = std::fmod(std::abs(rocket_shape.get_angle().z), fan::math::pi * 2.f);
          
          if (speed >= config::safe_landing_speed || normalized_angle >= config::safe_landing_angle) {
            particles.spawn_explosion(rocket_shape.get_position(), fan::colors::orange, 200, particle_image);
            state = game_state_e::game_over;
            rocket_shape.set_body_type(fan::physics::body_type_e::static_body);
            is_landing = false;
          } else {
            landing_timer -= dt;
            if (landing_timer <= 0.f) {
              state = game_state_e::win;
              rocket_shape.set_body_type(fan::physics::body_type_e::static_body);
              is_landing = false;
            }
          }
        }

        fan::vec2 input = pile.get_input_vector();
        rocket_shape.set_angular_velocity(input.x * config::turn_speed);

        thrusting = input.y < 0.f;

        if (thrusting) {
          f32_t current_angle = rocket_shape.get_angle().z;
          fan::vec2 thrust_dir(std::sin(current_angle), -std::cos(current_angle));

          rocket_shape.apply_linear_impulse_center(thrust_dir * config::thrust_power * rocket_shape.get_mass() * dt);

          fan::vec3 exhaust_pos = rocket_shape.get_position() - fan::vec3(thrust_dir * 35.f, 0);
          
          fire_accumulator += dt * 180.f; // 180 fire particles per second
          smoke_accumulator += dt * 60.f; // 60 smoke particles per second
          
          int fire_count = (int)fire_accumulator;
          int smoke_count = (int)smoke_accumulator;
          
          fire_accumulator -= fire_count;
          smoke_accumulator -= smoke_count;
          
          if (fire_count > 0) {
            particles.spawn_fire(exhaust_pos, fire_count, fire_image);
          }
          if (smoke_count > 0) {
            particles.spawn_smoke(exhaust_pos, smoke_count, particle_image);
          }
        }
      }

      if (rocket_shape.is_valid()) {
        f32_t pulse = thrusting ? (0.15f + 0.1f * std::sin(fan::time::now() / 1e8f)) : 0.05f;
      }

      particles.update(dt);
      draw_gui();
    }

    game_state_e state = game_state_e::menu;
    bool is_landing = false;
    f32_t landing_timer = 0.f;
    f32_t fire_accumulator = 0.f;
    f32_t smoke_accumulator = 0.f;

    gradient_t sky{
      fan::vec3(0.f, 0.f, 0),
      fan::vec2(100000.f, 100000.f),
      {
        fan::color(0.005f, 0.005f, 0.02f, 1.f),
        fan::color(0.005f, 0.005f, 0.02f, 1.f),
        fan::color(0.02f, 0.05f, 0.15f, 1.f),
        fan::color(0.15f, 0.08f, 0.12f, 1.f)
      }
    };

    physics::capsule_t rocket_shape;
    ship_graphics_t ship_graphics;

    gpu_particle_system_t<> particles;
    image_t particle_image{"images/smoke.webp"};
    image_t fire_image{"images/circle.png"};

    std::vector<physics::rectangle_t> terrain_shapes;
    std::vector<circle_t> hills;
    std::vector<circle_t> far_mountains;
    std::vector<circle_t> stars;
    physics::rectangle_t pad_shape;
    rectangle_t pad_visual;
    rectangle_t pad_stripe;
    circle_t pad_light;
    physics::collision_scope_t collision_scope;
  };

  rocket_game_t() {
    update_physics(true);

    set_settings({
      .clear_color = fan::colors::black,
      .ambient_color = fan::colors::white,
      .mode = fan::graphics::post_process_mode_e::none,
      .bloom_strength = 1.000f,
      .bloom_threshold = 0.241f,
      .bloom_knee = 0.264f,
      .bloom_tint = fan::vec3(1.f, 1.f, 1.f),
      .bloom_filter_radius = 0.299f,
      .blur_amount = 0.049f,
      .blur_filter_radius = 0.027f,
      .blur_focus_enabled = false,
      .gamma = 1.000f,
      .exposure = 1.000f,
      .contrast = 1.000f
    });

    bool is_menu = true;
    stage_open<game_stage_t>(&is_menu);
  }

} pile;

int main() {
  pile.loop();
}