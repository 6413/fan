#include fan_pch

int main() {

  loco_t loco;

  loco_t::image_t image;
  image.load("images/plant0.webp");
  std::vector<loco_t::shape_t> sprites;
  for (int i = 0; i < 25; ++i) {
    sprites.push_back(fan::graphics::sprite_t{ {
    .position = fan::vec3(64 + i * 64, 1175, i),
    .size = 32,
    .rotation_point = fan::vec2(0, 32),
    .image = &image,
    } });
  }
  loco.set_vsync(false);

  f32_t angle = 0;

  int direction = 1;

  f32_t threshold = fan::math::pi / 4;
  int intensity_index = 0;

  loco.add_fragment_shader_reload(
    fan::key_r,
    _FAN_PATH_QUOTE(graphics/glsl/opengl/2D/objects/sprite.vs),
    _FAN_PATH_QUOTE(graphics/glsl/opengl/2D/objects/sprite.fs)
  );

  loco.loop([&] {
    int index = 0;
    /*for (auto& i : sprites) {
      i.set_angle(fan::vec3(0, 0, angle));
      index += 1;
    }
    angle += loco.delta_time * direction;
    if (angle < -threshold) {
      direction *= -1;
    }
    if (angle > threshold) {
      direction *= -1;
    }*/
    loco.get_fps();
  });

  return 0;
}