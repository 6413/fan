// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#define loco_vulkan

#define loco_window
#define loco_context

#define loco_rectangle
#define loco_sprite
#include _FAN_PATH(graphics/loco.h)

struct pile_t {

  static constexpr fan::vec2 ortho_x = fan::vec2(-1, 1);
  static constexpr fan::vec2 ortho_y = fan::vec2(-1, 1);

  pile_t() {
    fan::vec2 window_size = loco.get_window()->get_size();
    loco.open_matrices(
      &matrices[0],
      ortho_x,
      ortho_y
    );
    loco.open_matrices(
      &matrices[1],
      fan::vec2(0, 800),
      fan::vec2(0, 800)
    );
    loco.get_window()->add_resize_callback([&](const fan::window_t::resize_cb_data_t& d) {
      fan::vec2 window_size = d.size;
      fan::vec2 ratio = window_size / window_size.max();
      std::swap(ratio.x, ratio.y);
      //matrices.set_ortho(
      //  ortho_x * ratio.x, 
      //  ortho_y * ratio.y
      //);
      viewport.set(loco.get_context(), 0, d.size, d.size);
    });
    viewport.open(loco.get_context());
    viewport.set(loco.get_context(), 0, window_size, window_size);
  }

  loco_t loco;
  loco_t::matrices_t matrices[2];
  fan::graphics::viewport_t viewport;
  fan::graphics::cid_t cid;
};

int main() {

  pile_t* pile = new pile_t;

  loco_t::sprite_t::properties_t p;

  p.size = fan::vec2(0.2, 0.2);
  //p.block_properties.
  p.get_matrices() = &pile->matrices[0];
  p.get_viewport() = &pile->viewport;

  loco_t::image_t image;
  image.load(&pile->loco, "images/test.webp");
  p.get_image() = &image;
  p.position = fan::vec2(0, 0);
  p.position.z = 0;
  // p.color = fan::color((f32_t)i / count, (f32_t)i / count + 00.1, (f32_t)i / count);
  //p.position = fan::random::vec2(0, 0);
  pile->loco.sprite.push_back(&pile->cid, p);
  loco_t::image_t image2;
  image2.load(&pile->loco, "images/asteroid.webp");
  p.get_image() = &image2;
  p.get_matrices() = &pile->matrices[1];

  p.position = fan::vec2(300, 300);
  p.size = 100;
  for (uint32_t i = 0; i < 10000; i++) {
    p.position = fan::random::vec2(0, 800);
    //pile->loco.sprite.push_back(&pile->cid, p);
    pile->loco.sprite.push_back(&pile->cid, p);
    //pile->loco.sprite.push_back(&pile->cid, p);
  }
  /*for (uint32_t i = 0; i < 10000; i++) {
    pile->loco.sprite.push_back(&pile->cid, p);
  }*/

  pile->loco.set_vsync(false);

  loco_t::rectangle_t::properties_t rp;

  //p.block_properties.
  rp.get_matrices() = &pile->matrices[1];
  rp.get_viewport() = &pile->viewport;

  rp.size = fan::vec2(0.2, 0.2);
  rp.position = fan::vec2(0.3, 0.3);
  rp.color = fan::colors::blue;
  //for (uint32_t i = 0; i < 100000; i++)
   // pile->loco.rectangle.push_back(&pile->cid, rp);


  pile->loco.set_vsync(false);

  fan::vec2 suunta = fan::random::vec2(-1500, 1500);

  auto& rectangle = pile->loco.rectangle;

  auto& window = *pile->loco.get_window();


  //VkPhysicalDeviceProperties v;
  //vkGetPhysicalDeviceProperties(pile->loco.get_context()->physicalDevice, &v);
  //fan::print(v.limits.maxPerStageDescriptorSampledImages);
  pile->loco.loop([&] {
    pile->loco.get_fps();
  });

  return 0;
}