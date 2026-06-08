#include <cmath>
#include <vector>
import fan;

using namespace fan::graphics;

int main() {
  engine_t engine;
  engine.get_lighting().set_target(0.3f);

  interactive_camera_t ic;
  tilemap_renderer_t map_renderer;
  auto map_id = map_renderer.open_map("playground.json", {
    .position = fan::vec3(engine.viewport_get_size() / 2.f, 0),
    .size = fan::vec2i(30, 18),
  });

  fan::vec2ui image_size(128, 128);

  auto make_image = [](fan::vec2ui size, auto&& fn) {
    std::vector<std::uint8_t> pixels(size.x * size.y * 4);
    for (std::uint32_t y = 0; y < size.y; ++y) {
      for (std::uint32_t x = 0; x < size.x; ++x) {
        fan::vec2 p = fan::vec2(x, y) / fan::vec2(size) * 2.f - 1.f;
        std::uint8_t a = fn(p) ? 255 : 0;
        std::size_t i = (y * size.x + x) * 4;
        pixels[i+0] = pixels[i+1] = pixels[i+2] = 255;
        pixels[i+3] = a;
      }
    }
    fan::image::info_t info;
    info.data = pixels.data();
    info.size = size;
    info.channels = 4;
    return image_load(info, image_presets::pixel_art());
  };

  image_t circle_img = make_image(image_size, [](fan::vec2 p) { return p.length() < 0.75f; });
  image_t cross_img  = make_image(image_size, [](fan::vec2 p) { return std::abs(p.x) < 0.18f || std::abs(p.y) < 0.18f; });
  image_t ring_img   = make_image(image_size, [](fan::vec2 p) { f32_t d = p.length(); return d > 0.42f && d < 0.78f; });

  image_load_properties_t smooth_lp = image_presets::smooth();
  smooth_lp.visual_output = image_sampler_address_mode_e::clamp_to_edge;
  image_t tree_img = image_load(std::string("images/content_browser/object.webp"), smooth_lp);

  std::vector<sprite_t> sprites;
  sprites.reserve(12);

  auto add = [&](image_t img, fan::vec2 pos, fan::vec2 size, fan::color color, f32_t angle = 0, f32_t threshold = 0.05f) {
    sprites.emplace_back(sprite_t{{
      .position = fan::vec3(pos, 0xfffa),
      .size     = size,
      .angle    = fan::vec3(0, 0, angle),
      .color    = color,
      .image    = img,
    }});
    gloco()->shadow_add_caster(&sprites.back(), threshold);
  };

  add(circle_img, {420, 300},  {90, 90},   fan::color(0.75f, 0.25f, 0.2f,  1));
  add(cross_img,  {720, 380},  {110, 110}, fan::color(0.25f, 0.75f, 0.35f, 1), fan::math::pi / 8);
  add(ring_img,   {1040, 320}, {100, 100}, fan::color(0.25f, 0.45f, 1,     1));
  add(cross_img,  {560, 650},  {130, 70},  fan::color(1, 0.8f, 0.25f,      1), fan::math::pi / 5);
  add(circle_img, {960, 680},  {70, 130},  fan::color(0.8f, 0.3f, 1,       1));
  add(tree_img, engine.viewport_get_size() / 2.f, {256, 256}, fan::colors::white, 0, 0.1f);

  
  map_renderer.iterate_tiles(map_id, [&](auto& t) {
    if (t.id != "shadow") return;
    sprites.emplace_back(sprite_t{{
      .position = fan::vec3(t.position, 0xfffa),
      .size     = t.size,
      .color    = fan::colors::transparent,
      .texture_pack_unique_id = t.texture_pack_unique_id
    }});
    gloco()->shadow_add_caster(&sprites.back(), 0.1f);
  });

  gloco()->shadow_add_light({0, 0}, 2024.f, fan::color(1.f, 0.82f, 0.55f, 1) * 0.6f,  0.018f, 1.8f);
  gloco()->shadow_add_light({0, 0}, 1024.f, fan::color(0.35f, 0.55f, 1.f,  1) * 0.65f, 0.025f, 2.2f);
  gloco()->shadow_set_darkness(0.37f);

  f32_t time = 0;

  engine.loop([&](f32_t dt) {
    time += dt;

    fan::vec2 vs = engine.viewport_get_size();
    map_renderer.update(map_id, vs / 2.f);

    gloco()->shadow_set_light_position(0, engine.get_mouse_position());
    gloco()->shadow_set_light_position(1, vs / 2.f + fan::vec2(std::cos(time), std::sin(time * 1.2f)) * fan::vec2(360, 220));

    sprites[1].set_angle(fan::vec3(0, 0, time * 0.6f));
    sprites[3].set_angle(fan::vec3(0, 0, -time * 0.45f));
  });
}