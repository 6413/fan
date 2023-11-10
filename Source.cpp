#include fan_pch

struct a_t {
  fan::vec2 v = fan::vec2(1, 2);
  fan::vec2 t = fan::vec2(1, 2);
};

struct rectangle_properties_t {
  fan::graphics::camera_t* camera = gloco->default_camera;
  fan::vec3 position;
  fan::vec2 size;
  fan::color color = fan::color(1, 1, 1, 1);
  bool blending = false;
};

struct b_t : a_t{
  b_t(rectangle_properties_t a = rectangle_properties_t()) {

  }
};


int main() {
  b_t{{.position = fan::vec2(1, 2),
    .size = fan::vec2(1, 2),
    .color = fan::colors::red}};
}