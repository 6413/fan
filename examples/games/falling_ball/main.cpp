#include <cmath>
#include <vector>
import fan;

using namespace fan::graphics;

int main() {
  engine_t engine{{.renderer=renderer_t::vulkan}};
  engine.get_lighting().set_target(0.3f);

  interactive_camera_t ic;

  fan::vec2ui image_size(128, 128);

  auto make_image = [](fan::vec2ui size, auto&& fn) {
    std::vector<std::uint8_t> pixels(size.x * size.y * 4);
    for (std::uint32_t y = 0; y < size.y; ++y) {
      for (std::uint32_t x = 0; x < size.x; ++x) {
        fan::vec2 p = fan::vec2(x, y) / fan::vec2(size) * 2.f - 1.f;
        std::uint8_t a = fn(p) ? 255 : 0;
        std::size_t i = (y * size.x + x) * 4;
        pixels[i + 0] = pixels[i + 1] = pixels[i + 2] = 255;
        pixels[i + 3] = a;
      }
    }
    fan::image::info_t info;
    info.data = pixels.data();
    info.size = size;
    info.channels = 4;
    return image_load(info, image_presets::pixel_art());
  };

  image_t circle_img = make_image(image_size, [](fan::vec2 p) { return p.length() < 0.75f; });
  image_t cross_img = make_image(image_size, [](fan::vec2 p) { return std::abs(p.x) < 0.18f || std::abs(p.y) < 0.18f; });
  image_t ring_img = make_image(image_size, [](fan::vec2 p) { f32_t d = p.length(); return d > 0.42f && d < 0.78f; });
  image_t solid_img = make_image(image_size, [](fan::vec2) { return true; });

  image_load_properties_t smooth_lp = image_presets::smooth();
  smooth_lp.visual_output = image_sampler_address_mode_e::clamp_to_edge;
  image_t tree_img = image_load(std::string("images/content_browser/object.webp"), smooth_lp);

  f32_t half_max_y = 0.f;
  std::vector<physics::circle_t> bodies;
  std::vector<sprite_t> sprites;
  bodies.reserve(300);
  sprites.reserve(302);

  for (int i = 0; i < 300; ++i) {
    fan::vec2 p = fan::random::vec2(fan::vec2(0), fan::vec2(1024, 52000));
    f32_t r = fan::random::value(4.f, 32.f) * 3.f;

    bodies.emplace_back(physics::circle_t {{
      .position = fan::vec3(p, 0xfffa),
      .radius = r,
      .color = fan::random::bright_color(),
      .shape_properties{.restitution = 0.4f}
    }});

    auto& s = sprites.emplace_back(sprite_t {{
      .position = fan::vec3(p, 0xfffb),
      .size = fan::vec2(r) * 1.5f,
      .color = bodies.back().get_color(),
      .image = circle_img,
    }});
    gloco()->shadow_add_caster(&s, 0.001f);

    half_max_y = std::max(half_max_y, p.y);
  }
  half_max_y *= 0.5f;

  physics::rectangle_t wall_left {{
    .position = fan::vec3(fan::vec2(-100, half_max_y), 1),
    .size = fan::vec2(5, half_max_y),
    .shape_properties{.restitution = 1.f}
  }};
  physics::rectangle_t wall_right {{
    .position = fan::vec3(fan::vec2(1024 + 100, half_max_y), 1),
    .size = fan::vec2(5, half_max_y),
    .shape_properties{.restitution = 1.f}
  }};

  auto add_wall_caster = [&](const physics::rectangle_t& wall) {
    auto& s = sprites.emplace_back(sprite_t {{
      .position = fan::vec3(wall.get_position(), 0xfffb),
      .size = wall.get_size(),
      .color = fan::colors::transparent,
      .image = solid_img,
    }});
    gloco()->shadow_add_caster(&s, 0.001f);
  };

  add_wall_caster(wall_left);
  add_wall_caster(wall_right);

  gloco()->shadow_add_light({0, 0}, 1024.f, fan::color(1.f, 0.82f, 0.55f, 1) * 0.6f,  0.018f, 1.8f);
  gloco()->shadow_set_darkness(0.37f);

  physics::circle_t ball{{
    .position = fan::vec3(512, 0, 0xfffa), 
    .radius=14.f, 
    .body_type=fan::physics::body_type_e::dynamic_body
  }};

  engine.update_physics(true);
  engine.get_physics_context().set_gravity(fan::vec2(0, engine.get_physics_context().get_gravity().y / 2.4f));

  fan::time::timer stuck{0.5, false};
  f32_t prev_x = ball.get_position().x;

  rectangle_t rect(fan::vec3(fan::vec2(512), 0), 100000, fan::colors::gray);

  engine.loop([&](f32_t dt) {
    engine.camera_set_center(ball.get_position());
    if (fan::math::is_near(prev_x, ball.get_position().x, 0.01)) {
      if (!stuck.started()) {
        stuck.start();
      }
    }
    else {
      stuck.stop();
    }
    if (stuck) {
      stuck.stop();
      ball.apply_linear_impulse_center(fan::vec2(fan::random::value(-100.f, 100.f), 0));
    }
    prev_x = ball.get_position().x;

    gloco()->shadow_set_light_position(0, ball.get_position());
  });
}