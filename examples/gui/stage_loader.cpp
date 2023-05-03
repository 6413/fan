#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

struct pile_t;

#ifndef FAN_INCLUDE_PATH
  #define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 2
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#define loco_opengl

#define loco_window
#define loco_context

#define loco_sprite
#define loco_button
#include _FAN_PATH(graphics/loco.h)

// in stagex.h getting pile from mouse cb
// pile_t* pile = OFFSETLESS(OFFSETLESS(mb.vfi, loco_t, vfi), pile_t, gloco);

#define stage_loader_path .
#include _FAN_PATH(graphics/gui/stage_maker/loader.h)

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
        ortho_x * ratio.x,
        ortho_y * ratio.y
      );
      viewport.set(0, window_size, window_size);
    });
    viewport.open();
    viewport.set(0, window_size, window_size);
    theme = loco_t::themes::deep_red();

    // requires manual open with compiled texture pack name
  }

  loco_t::theme_t theme;
  loco_t::camera_t camera;
  fan::graphics::viewport_t viewport;

  #define custom_stages fan::return_type_of_t<decltype([]{struct a: public test_t, stage_common_t_t<test_t>{ \
    using stage_common_t_t::stage_common_t_t; \
    const char* stage_name = ""; \
      void close(auto& loco){ \
		      test_t::close(loco);\
      } \
   }*v; return *v;})>*
  stage_loader_t stage_loader;
};

int main(int argc, char** argv) {
  
  if (argc < 2) {
    fan::throw_error("usage: TexturePackCompiled");
  }

  pile_t* pile = new pile_t;

  loco_t::texturepack_t tp;
  tp.open_compiled(&pile->loco, argv[1]);
  pile->stage_loader.open(&tp);

	stage_loader_t::stage_open_properties_t op;
	op.camera = &pile->camera;
	op.viewport = &pile->viewport;
	op.theme = &pile->theme;

  stage_loader_t::nr_t it2 = stage_loader_t::open_stage<stage_loader_t::stage::stage0_t>(op);
  //it2.erase();
  

	pile->loco.loop([&] {

	});
	
}