#include fan_pch

int main() {
  loco_t loco;

  loco.default_camera->camera.set_ortho(
    fan::vec2(-1, 1),
    fan::vec2(-1, 1)
  );

  loco_t::image_t images[2];
  images[0].load("images/brick.webp");
  images[1].load("images/brick_inverted.webp");

  fan::graphics::sprite_t s0{{
    .position = fan::vec3(0, 0, 2),
    .size = 0.5,
    .color = {1, 1, 1, 0.5},
    .image = images,
    .blending = true
  }};

  fan::graphics::sprite_t s1{{
    .position = fan::vec3(-0.25, -0.25, 0),
    .size = 0.5,
    .color = {1, 1, 1, 0.5},
    .image = images + 1,
    .blending = true
  }};

  fan::graphics::rectangle_t r0{{
    .position = fan::vec3(0.25, 0.25, 1),
    .size = 0.5,
    .color = fan::colors::red,
    .blending = true
  }};
  fan::graphics::rectangle_t r1{{
    .position = fan::vec3(-0.25, 0.5, 3),
    .size = 0.5,
    .color = fan::colors::green - fan::color(0, 0, 0, .5),
    .blending = true
  }};

  loco.loop([&] {
    loco.get_fps();
  });

  return 0;
}