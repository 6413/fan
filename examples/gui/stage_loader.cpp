#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

struct pile_t;

#ifndef FAN_INCLUDE_PATH
  #define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#define loco_opengl

#define loco_window
#define loco_context

#define loco_sprite
#define loco_button
#include _FAN_PATH(graphics/loco.h)

// in stagex.h getting pile from mouse cb
// pile_t* pile = OFFSETLESS(OFFSETLESS(mb.vfi, loco_t, vfi), pile_t, loco);

struct pile_t {
  loco_t loco;

  static constexpr fan::vec2 ortho_x = fan::vec2(-1, 1);
  static constexpr fan::vec2 ortho_y = fan::vec2(-1, 1);

  pile_t() {
    fan::vec2 window_size = loco.get_window()->get_size();
    loco.open_camera(
      &camera,
      ortho_x,
      ortho_y
    );
    loco.get_window()->add_resize_callback([&](const fan::window_t::resize_cb_data_t& d) {
      fan::vec2 window_size = d.size;
      // keep aspect ratio
      fan::vec2 ratio = window_size / window_size.max();
      camera.set_ortho(
        &loco,
        ortho_x * ratio.x,
        ortho_y * ratio.y
      );
      viewport.set(loco.get_context(), 0, window_size, window_size);
    });
    viewport.open(loco.get_context());
    viewport.set(loco.get_context(), 0, window_size, window_size);
    theme = loco_t::themes::deep_red();
    theme.open(loco.get_context());

    // requires manual open with compiled texture pack name
  }

  ~pile_t() {

  }

  loco_t::theme_t theme;
  loco_t::camera_t camera;
  fan::graphics::viewport_t viewport;

  #define loco_access &OFFSETLESS(this, pile_t, stage_loader)->loco
  #define stage_loader_path .
  #include _FAN_PATH(graphics/gui/stage_maker/loader.h)
  stage_loader_t stage_loader;
};

pile_t* pile = new pile_t;

int main(int argc, char** argv) {
  if (argc < 2) {
    fan::throw_error("usage: TexturePackCompiled");
  }

  loco_t::texturepack_t tp;
  tp.open_compiled(&pile->loco, argv[1]);
  pile->stage_loader.open(&tp);

	using sl = pile_t::stage_loader_t;
  
	sl::stage_open_properties_t op;
	op.camera = &pile->camera;
	op.viewport = &pile->viewport;
	op.theme = &pile->theme;

  auto it = pile->stage_loader.push_and_open_stage<sl::stage::stage0_t>(op);
  
	pile->loco.loop([&] {

	});
	
}