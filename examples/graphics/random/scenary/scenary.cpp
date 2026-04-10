#include <string>

import fan;

using namespace fan::graphics;
using namespace fan::color_literals;

int main() {
  engine_t engine{{.window_size = fan::resolutions.r1440p}};
  engine.get_lighting().set_target(0.1f);
  engine.set_clear_color(0x0_rgb);

  interactive_camera_t ic;
  fan::vec2 ws = engine.window.get_size() / 2.f;

  light_t light(0, 1500.f, fan::colors::yellow);
  fan::vec2 cloud_size = fan::vec2(ws.x, ws.y * 0.6f);
  shader_shape_t clouds(engine.shaders.clouds, fan::vec3(0.f, 0.f, 1.f), cloud_size);
  
  fan::color sky_top(0.10f, 0.29f, 0.47f), sky_bot(0.42f, 0.69f, 0.90f);
  gradient_t sky(sky_top, sky_bot, fan::vec3(0.f), ws);

  auto shapes = shapes_from_json("tests/forest.json");
  fan::vec2 scaler = ws - shapes.front().get_size();
  
  for (auto& s : shapes) {
    s.set_tc_size(fan::vec2(10.f, 1.f));
    s.set_size((scaler + s.get_size()) * fan::vec2(10.f, 1.f));
    s.set_color(fan::color(0.5f));
    s.set_parallax_factor(fan::vec2(1.f - s.get_position().z / 15.f, 0.f));
  }

  fan::vec3 cloud_pos{0.f, -ws.y / 2.f, 0.f};
  fan::vec2 light_pos{-586.9f, -161.800}, light_sz{2000.f, 1500.f};
  fan::color light_col(1.56f, 0.61f, 0.5f, 1.0f);
  f32_t tc = 0.f;

  engine.loop([&] {
    engine.camera_set_center(fan::vec2(tc += engine.get_delta_time() * 700.f, 0.f));
    fan::vec2 cam_pos = engine.camera_get_center();

    for (auto& s : shapes) {
      fan::vec3 v = s.get_position();
      s.set_position(fan::vec3(v.x, v.z / 50.f * ws.y - 100.f, v.z));
    }

    clouds.set_position(fan::vec2(cam_pos) + cloud_pos);
    light.set_color(light_col);
    light.set_position(cam_pos + light_pos);
    light.set_size(light_sz);

    circle(fan::vec3(fan::vec2(light.get_position()), 0.f), 54.f, light_col*10.f);
    circle(fan::vec3(fan::vec2(light.get_position()), 1.f), 54.f, light_col.set_alpha(0.5));

    sky.set_position(cam_pos);
    fan::color t = sky_top.lerp(light_col, 0.01f), b = sky_bot.lerp(light_col, 0.1f);
    sky.set_colors({t, t, b, b});
  });
}