import std;
import fan;

using namespace fan::graphics;

fan::noise_t hill_noise{42};
fan::noise_t cave_noise{137};
fan::noise_t detail_noise{999};

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

  physics::character2d_t player = physics::character_sprite({
    .position{0, 0, 6.f},
    .size=12.f,
  });
  auto& pctx = engine.get_physics_context();
  pctx.set_gravity(pctx.get_gravity() / 1.5f);
  player.set_mass(100.f);
  player.enable_default_movement(300.f, 200.f);

  engine.camera_set_target(player, 10.f);

  f32_t dig_radius = 12.f;
  fan::time::interval_t dig_interval{2.3f};

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

    line(  fan::vec3(player_pos, 7.f), hit_pos, fan::color(1.f, 0.f, 0.f, 0.3f), dig_radius);
    circle(fan::vec3(hit_pos, 7.f), dig_radius, fan::color(1.f, 1.f, 0.f, 0.2f));
    circle(fan::vec3(hit_pos, 7.f), dig_radius * 0.15f, fan::colors::red);

    if (!fan::graphics::gui::want_io() && fan::window::is_mouse_down(fan::mouse_left) && dig_interval.tick(dt)) {
      terrain.dig(hit_pos, dig_radius);
    }
  });

  return 0;
}