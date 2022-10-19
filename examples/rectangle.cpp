// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#define loco_vulkan

#define loco_window
#define loco_context

//#define loco_post_process

#define loco_rectangle
#include _FAN_PATH(graphics/loco.h)

constexpr uint32_t count = 1.0e+1;

struct pile_t {

  static constexpr fan::vec2 ortho_x = fan::vec2(-1, 1);
  static constexpr fan::vec2 ortho_y = fan::vec2(-1, 1);

  void open() {
    loco.open(loco_t::properties_t());
    fan::graphics::open_matrices(
      loco.get_context(),
      &matrices,
      ortho_x,
      ortho_y
    );
    loco.get_window()->add_resize_callback([&](const fan::window_t::resize_cb_data_t& d) {
      //fan::vec2 window_size = window->get_size();
      viewport.set(loco.get_context(), 0, d.size, d.size);
     });
    viewport.open(loco.get_context());
    viewport.set(loco.get_context(), 0, loco.get_window()->get_size(), loco.get_window()->get_size());
  }

  loco_t loco;
  fan::graphics::matrices_t matrices;
  fan::graphics::viewport_t viewport;
  fan::graphics::cid_t cids[count];
};

int main() {

  pile_t* pile = new pile_t;
  pile->open();

  loco_t::rectangle_t::properties_t p;
  
  //p.block_properties.
  p.get_matrices() = &pile->matrices;
  p.get_viewport() = &pile->viewport;

  fan::time::clock c;
  c.start();
  p.size = fan::vec2(1.0 / 1920, 1.0 / 1920);
  for (f32_t j = 0; j < 1920; j++) {
    for (f32_t i = 0; i < 1920; i++) {
      p.position = fan::vec2(-1.0 + i / 1920 * 2, -1.0 + p.size.y * 2 * j);
      p.color = fan::random::color();
      pile->loco.rectangle.push_back(&pile->cids[1], p);
    }
  }
  fan::print("elapsed", c.elapsed());

  //p.position = fan::vec2(0.5, 0);
  //p.color = fan::random::color();
  //pile->loco.rectangle.push_back(&pile->cids[0], p);

  pile->loco.set_vsync(false);

  //VkPhysicalDeviceProperties pdp;
  //vkGetPhysicalDeviceProperties(pile->loco.get_context()->physicalDevice, &pdp);
  //fan::print(pdp.limits.maxMemoryAllocationCount);

  pile->loco.loop([&] {
    //pile->loco.rectangle.set(&pile->cids[0], &loco_t::rectangle_t::instance_t::position, pile->loco.get_mouse_position(pile->viewport));
    
    pile->loco.get_fps();
  });
  return 0;
}