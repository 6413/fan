// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#define fan_debug 3
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#define loco_vulkan

#define loco_window
#define loco_context

#define loco_sprite
#include _FAN_PATH(graphics/loco.h)

struct pile_t {

  static constexpr fan::vec2 ortho_x = fan::vec2(-1, 1);
  static constexpr fan::vec2 ortho_y = fan::vec2(-1, 1);

  void open() {
    loco.open(loco_t::properties_t());
    fan::vec2 window_size = loco.get_window()->get_size();
    fan::graphics::open_matrices(
      loco.get_context(),
      &matrices,
      ortho_x,
      ortho_y
    );
    loco.get_window()->add_resize_callback([&](const fan::window_t::resize_cb_data_t& d) {
      fan::vec2 window_size = d.size;
      fan::vec2 ratio = window_size / window_size.max();
      std::swap(ratio.x, ratio.y);
      matrices.set_ortho(
        ortho_x * ratio.x, 
        ortho_y * ratio.y
      );
      viewport.set(loco.get_context(), 0, d.size, d.size);
    });
    viewport.open(loco.get_context());
    viewport.set(loco.get_context(), 0, window_size, window_size);

  }

  loco_t loco;
  fan::graphics::matrices_t matrices;
  fan::graphics::viewport_t viewport;
  fan::graphics::cid_t cid;
};

int main() {

  pile_t* pile = new pile_t;
  pile->open();

  loco_t::sprite_t::properties_t p;

  p.size = fan::vec2(1, 1);
  //p.block_properties.
  p.get_matrices() = &pile->matrices;
  p.get_viewport() = &pile->viewport;

  fan::graphics::image_t image;
  image.load(pile->loco.get_context(), "images/test.webp");
  fan::graphics::image_t image2;
  image2.load(pile->loco.get_context(), "images/planet.webp");
  p.get_image() = &image;
  p.position = fan::vec2(0, 0);
  p.position.z = 0;
  // p.color = fan::color((f32_t)i / count, (f32_t)i / count + 00.1, (f32_t)i / count);
  p.position = fan::random::vec2(0, 0);
  pile->loco.sprite.push_back(&pile->cid, p);
  //p.position = fan::vec2(-0.1, -0.1);
  //p.get_image() = &image2;
  //p.position.z = 1;
  //pile->loco.sprite.push_back(&pile->cid, p);

  pile->loco.set_vsync(false);


  VkPhysicalDeviceProperties v;
  vkGetPhysicalDeviceProperties(pile->loco.get_context()->physicalDevice, &v);
  fan::print(v.limits.maxPerStageDescriptorSampledImages);
  pile->loco.loop([&] {
    pile->loco.get_fps();
  });

  return 0;
}