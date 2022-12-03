// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#define loco_window
#define loco_context

#define loco_sprite
#include _FAN_PATH(graphics/loco.h)

struct pile_t {

  static constexpr fan::vec2 ortho_x = fan::vec2(-1, 1);
  static constexpr fan::vec2 ortho_y = fan::vec2(-1, 1);

  void open() {
    loco.open_matrices(
      &matrices,
      ortho_x,
      ortho_y
    );
    viewport.open(loco.get_context());
    viewport.set(loco.get_context(), 0, loco.get_window()->get_size(), loco.get_window()->get_size());
  }

  loco_t::matrices_t matrices;
  fan::graphics::viewport_t viewport;
  loco_t loco;
};

int main() {

  pile_t pile;
  pile.open();

  loco_t::texturepack texturepack;
  texturepack.open_compiled(&pile.loco, "TexturePackCompiled");

  loco_t::sprite_t::properties_t p;

  loco_t::texturepack::ti_t ti;
  if (texturepack.qti("test.webp", &ti)) {
    return 1;
  }
  p.load_tp(&texturepack, &ti);

  p.position = 0;
  p.matrices = &pile.matrices;
  p.viewport = &pile.viewport;
  p.size = 0.5;
  p.position = 0;
  fan::graphics::cid_t cid;
  pile.loco.sprite.push_back(&cid, p);
  
  pile.loco.loop([&] {

   });

  return 0;
}