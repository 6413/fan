#include <fan/pch.h>


template <uintptr_t s>
struct ptrandsize_t {
  uint8_t p[s];
};
template<typename... Ts, uintptr_t s = (sizeof(Ts) + ...)>
constexpr ptrandsize_t<s> suffering(Ts... args) {
  ptrandsize_t<s> r;
  uintptr_t i = 0;
  ([&](auto arg) {
    __MemoryCopy(&arg, &r.p[i], sizeof(arg));
    }(args), ...);
  return r;
}


int main() {

  loco_t loco{ {.window_size = {1600, 900}} };

  loco.set_vsync(0);

  loco_t::image_t image;
  image = loco.image_load("images/tire.webp");
  std::vector<loco_t::shape_t> sprites;
  fan::vec2 window_size = loco.window.get_size();

  loco.loop([&] {
    sprites.clear();
    for (int i = 0; i < 3; ++i) {
      sprites.push_back(fan::graphics::sprite_t{ {
      .position = fan::vec3(fan::random::vec2(fan::vec2(0,1100), fan::vec2(0, 450)), 0),
      .size = 25,
      .rotation_point = fan::vec2(100, 0),
      .image = image,
      } });
    }
    loco.get_fps();
  });

  return 0;
}