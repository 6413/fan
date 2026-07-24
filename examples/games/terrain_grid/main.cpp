import std;
import fan;

using namespace fan::graphics;

auto seed = (int)std::chrono::steady_clock::now().time_since_epoch().count();
fan::noise_t hill_noise{seed};
fan::noise_t cave_noise{seed + 1};
fan::noise_t detail_noise{seed + 2};

int main() {
  engine_t engine;
  interactive_camera_t ic;
  engine.get_clear_color() = fan::colors::black;
  engine.update_physics(true);

  gradient_t bg_sky{fan::color(0.2f, 0.4f, 0.75f, 1.f), fan::color(0.6f, 0.75f, 0.9f, 1.f), fan::vec3(1.f), engine.whs()};
  //auto bg_below = gradient_t{fan::color(1.2f, 0.4f, 0.75f, 1.f), fan::color(1.6f, 0.75f, 0.9f, 1.f), fan::vec3(0), engine.whs()};
  sprite_t bg_below{{
    .position=fan::vec3(0, 0, 0.f),
    .size = engine.whs(),
    .image = "fossil_cave.jpg"
  }};

  auto pa = image_presets::pixel_art();
  image_t img_grass{"Textures/Grass/cubeGreen_1.png", pa};
  image_t img_dirt{"Textures/Dirt/cubeDirt_1.png", pa};
  image_t img_stone{"Textures/Stone/cubeStone_1.png", pa};
  image_t img_bedrock{"Textures/Stone/cubeBedrock_1.png", pa};
  image_t img_island{"Textures/Grass/cubeGreen_1.png", pa};
  image_t img_dark_grass{"Textures/Grass/cubeGreen_2.png", pa};

  algorithm::chunk_renderer_t terrain{{
    .cell_size = 16.f,
    .chunk_size = 32,
    .hill_noise = &hill_noise,
    .cave_noise = &cave_noise,
    .detail_noise = &detail_noise,
    .surface_base = -10.f,
    //.sky_island_noise = &hill_noise,
    //.sky_island_freq = 0.04f,
    //.sky_island_threshold = 0.6f,
    //.sky_island_min = 80.f,
    //.sky_island_max = 150.f,
    .img_sky_island = img_island,
    .tile_layers = {
      {3.f, img_grass}, 
      {12.f, img_dirt}, 
      {50.f, img_stone}, 
      {std::numeric_limits<f32_t>::max(), img_bedrock}
    },
    .scatter_noise = &detail_noise,
    .scatter_img = img_dark_grass,
    .scatter_threshold = 0.6f,
  }};

  auto player = physics::character_sprite({
    .position{0, 0, 6.f},
    .size=12.f,
    .image = "images/duck.webp"
  });
  auto& pctx = engine.get_physics_context();
  pctx.set_gravity(pctx.get_gravity() / 1.5f);
  player.set_mass(100.f);
  player.enable_default_movement(300.f, 200.f);

  engine.camera_set_target(player, 10.f);

  f32_t dig_radius = 64.f;
  fan::time::interval_t dig_interval{0.003f};

  gpu_particle_system_t<> dig_particles;
  image_t particle_img{fan::colors::white};
  struct break_effect_t { circle_t circle; f32_t timer = 0.f; };
  std::vector<break_effect_t> break_effects;

  engine.loop([&] {
    f64_t dt = engine.get_delta_time();
    fan::vec2 player_pos = player.get_position();
    fan::vec2 cam_center = ic.get_center();

    f32_t ground_y = 512.f;
    bg_sky.set_position(fan::vec2(cam_center.x, std::min(cam_center.y, ground_y)));
    bg_below.set_position(cam_center);
//    bg_below.set_position(fan::vec3(cam_center.x, std::max(cam_center.y, ground_y + bg_below.get_size().y*2.f), 0.f));

    if (fan::window::is_key_clicked(fan::key_r)) {
      player.set_physics_position({player_pos.x, 0});
    }

    terrain.stream(player_pos, engine.ws());

    fan::vec2 mouse_pos = engine.get_mouse_position();
    fan::vec2 hit_pos = terrain.raycast(player_pos, mouse_pos, dig_radius);

    //line(  fan::vec3(player_pos, 7.f), hit_pos, fan::color(1.f, 0.f, 0.f, 0.3f), dig_radius);
    //circle(fan::vec3(hit_pos, 7.f), dig_radius, fan::color(1.f, 1.f, 0.f, 0.2f));
    //circle(fan::vec3(hit_pos, 7.f), dig_radius * 0.15f, fan::colors::red);

    if (!fan::graphics::gui::want_io() && fan::window::is_mouse_down(fan::mouse_left) && dig_interval.tick(dt)) {
      terrain.dig(hit_pos, dig_radius);

      break_effects.push_back({
        .circle = circle_t(circle_properties_t{
          .position = fan::vec3(hit_pos, 22.f),
          .radius = dig_radius * 0.5f,
          .color = fan::color(1.f, 1.f, 1.f, 0.6f),
          .outline_color = fan::color(1.f, 0.9f, 0.5f, 0.4f),
          .outline_width = 2.f,
        }),
        .timer = 0.25f,
      });

      dig_particles.spawn([&](auto& p) {
        p.loop = false;
        p.position = fan::vec3(hit_pos, 21.f);
        p.begin_color = fan::color(0.5f, 0.4f, 0.3f, 1.f);
        p.end_color = fan::color(0.3f, 0.2f, 0.1f, 0.f);
        p.start_size = fan::vec2(5);
        p.end_size = fan::vec2(1);
        p.alive_time = 0.35f;
        p.count = 14;
        p.start_velocity = fan::vec2(70, -90);
        p.end_velocity = fan::vec2(25, 50);
        p.start_spread = fan::vec2(70);
        p.end_spread = fan::vec2(110);
        p.expansion_power = 0.5f;
        p.shape = fan::graphics::shapes::particles_t::shapes_e::circle;
        p.image = particle_img;
      });
    }

    for (std::size_t i = 0; i < break_effects.size(); ) {
      break_effects[i].timer -= dt;
      if (break_effects[i].timer <= 0.f) {
        break_effects.erase(break_effects.begin() + i);
      } else {
        f32_t t = break_effects[i].timer / 0.25f;
        break_effects[i].circle.set_radius(dig_radius * 0.5f + (1.f - t) * dig_radius * 1.5f);
        break_effects[i].circle.set_color(fan::color(1.f, 1.f, 1.f, t * 0.6f));
        break_effects[i].circle.set_outline_color(fan::color(1.f, 0.9f, 0.5f, t * 0.4f));
        ++i;
      }
    }
    dig_particles.update(dt);
  });

  return 0;
}