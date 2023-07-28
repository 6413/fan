#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

//#define loco_vulkan

#define loco_window
#define loco_context

#define loco_text
#include _FAN_PATH(graphics/loco.h)

constexpr uint32_t count = 1;

struct pile_t {

  static constexpr fan::vec2 ortho_x = fan::vec2(-1, 1);
  static constexpr fan::vec2 ortho_y = fan::vec2(-1, 1);

  pile_t() {
    loco.open_camera(
      &camera,
      ortho_x,
      ortho_y
    );
    loco.get_window()->add_resize_callback([&](const fan::window_t::resize_cb_data_t& d) {
      fan::vec2 window_size = d.size;
    fan::vec2 ratio = window_size / window_size.max();
    camera.set_ortho(
      &loco,
      ortho_x * ratio.x,
      ortho_y * ratio.y
    );
    viewport.set(loco.get_context(), 0, window_size, window_size);
      });
    viewport.open(loco.get_context());
    viewport.set(loco.get_context(), 0, loco.get_window()->get_size(), loco.get_window()->get_size());
  }

  loco_t loco;
  loco_t::camera_t camera;
  fan::graphics::viewport_t viewport;
  fan::graphics::cid_t cid[count];
};

int main() {

  pile_t* pile = new pile_t;

  loco_t::text_t::properties_t p;

  p.camera = &pile->camera;
  p.viewport = &pile->viewport;

  p.font_size = 0.05;
  p.text = "click me";
  p.position = fan::vec2(0, 0);
  //p.text = fan::random::string(5);
  pile->loco.text.push_back(p, &pile->cid[0]);

  loco_t::vfi_t::properties_t vfip;
  vfip.mouse_button_cb = [](const loco_t::mouse_button_data_t& ii_d) -> int {
    fan::print("click always");
    return 0;
  };
  vfip.mouse_move_cb = [](const loco_t::mouse_move_data_t& ii_d) -> int {
      return 0;
  };
  vfip.keyboard_cb = [](const loco_t::keyboard_data_t& kd) -> int {
    return 0;
  };

  loco_t::vfi_t::shape_id_t ids[2];

  vfip.shape_type = loco_t::vfi_t::shape_t::always;
  vfip.shape.always.z = 0;
  pile->loco.push_back_input_hitbox(&ids[0], vfip);

  vfip.shape_type = loco_t::vfi_t::shape_t::rectangle;
  vfip.shape.rectangle.position = fan::vec3(0, 0, 1);
  vfip.shape.rectangle.size = pile->loco.text.get_text_size(p.text, p.font_size);
  vfip.shape.rectangle.size.x /= 2; // hitbox takes half size
  vfip.shape.rectangle.camera = p.camera;
  vfip.shape.rectangle.viewport = p.viewport;

  vfip.mouse_button_cb = [](const loco_t::mouse_button_data_t& ii_d) -> int {
    fan::print("click rectangle");
    return 0;
  };

  pile->loco.push_back_input_hitbox(&ids[1], vfip);

  pile->loco.loop([&] {
    pile->loco.get_fps();
  });

  return 0;
}