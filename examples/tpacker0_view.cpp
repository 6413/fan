// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(graphics/graphics.h)

#include _FAN_PATH(graphics/opengl/texture_pack.h)

#define loco_window
#define loco_context

#define loco_sprite
#include _FAN_PATH(graphics/loco.h)

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
    viewport.open(loco.get_context());
    viewport.set(loco.get_context(), 0, loco.get_window()->get_size(), loco.get_window()->get_size());
  }

  fan::opengl::matrices_t matrices;
  fan::opengl::viewport_t viewport;
  loco_t loco;
};

int main() {

  pile_t pile;
  pile.open();

  fan::opengl::texturepack texturepack;
  texturepack.open_compiled(pile.loco.get_context(), "TexturePackCompiled");

  loco_t::sprite_t::properties_t p;

  p.position = 0;
  p.get_matrices() = &pile.matrices;
  p.get_viewport() = &pile.viewport;
  auto image = texturepack.get_pixel_data(0).image;
  p.get_image() = &image;
  fan::tp::ti_t ti;
  if (texturepack.qti("test.webp", &ti)) {
    return 1;
  }
  
  p.tc_position = ti.position / fan::vec2(1024, 1024);
  p.tc_size = ti.size / fan::vec2(1024, 1024);
  p.size = 0.5;
  p.position = 0;
  fan::opengl::cid_t cid;
  pile.loco.sprite.push_back(&cid, p);
  
  pile.loco.loop([&] {

    });

  return 0;
}