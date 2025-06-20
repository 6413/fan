#include <fan/pch.h>

constexpr uint32_t count = 1;

struct pile_t {

  static constexpr fan::vec2 ortho_x = fan::vec2(-1, 1);
  static constexpr fan::vec2 ortho_y = fan::vec2(-1, 1);

  pile_t() {
   /* loco.camera_set_ortho(
      loco.orthographic_render_view.camera,
      ortho_x,
      ortho_y
    );*/
  }

  loco_t loco;
  loco_t::camera_t camera;
};

int main() {

  pile_t pile;

  loco_t::vfi_t::properties_t vfip;
  //vfip.mouse_button_cb = [](const loco_t::vfi_t::mouse_button_data_t& ii_d) -> int {
  //  fan::print("click always");
  //  return 0;
  //};
  //vfip.mouse_move_cb = [](const loco_t::vfi_t::mouse_move_data_t& ii_d) -> int {
  //    return 0;
  //};
  //vfip.keyboard_cb = [](const loco_t::vfi_t::keyboard_data_t& kd) -> int {
  //  return 0;
  //};


  //vfip.shape_type = loco_t::vfi_t::shape_t::always;
  //vfip.shape.always->z = 0;
  //loco_t::shape_t s0 = vfip;

  vfip.shape_type = loco_t::vfi_t::shape_t::rectangle;
  vfip.shape.rectangle->position = fan::vec3(500, 500, 1);
  vfip.shape.rectangle->size = fan::vec2(100, 100);
  vfip.shape.rectangle->size.x /= 2; // hitbox takes half size
  vfip.shape.rectangle->camera = gloco->orthographic_render_view.camera;
  vfip.shape.rectangle->viewport = gloco->orthographic_render_view.viewport;

  vfip.mouse_button_cb = [](const loco_t::vfi_t::mouse_button_data_t& ii_d) -> int {
    fan::print("click rectangle");
    return 0;
  };
  fan::graphics::rectangle_t r{ {
    .position = fan::vec3(*(fan::vec2*)&vfip.shape.rectangle->position, 0),
    .size = vfip.shape.rectangle->size,
    .color = fan::colors::red
} };

  //loco_t::shape_t s1 = r;

  fan::graphics::vfi_root_t root;

  /*typename loco_t::vfi_t::properties_t vfip;
  vfip.shape.rectangle->position = 0;
  vfip.shape.rectangle->position.z = 1;
  vfip.shape.rectangle->size = 200;
  vfip.shape.rectangle->angle = 0;
  vfip.shape.rectangle->rotation_point = 0;*/
  root.set_root(vfip);
  root.push_child(r);
  r.remove();


  pile.loco.loop([&] {
    pile.loco.get_fps();
  });

  return 0;
}