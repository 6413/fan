#include fan_pch

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
  loco_t::viewport_t viewport;

  stage_loader_t stage_loader;
};
pile_t* pile = new pile_t;

lstd_defstruct(custom_t)
  #include _FAN_PATH(graphics/gui/stage_maker/preset.h)
    
  static constexpr auto stage_name = "";
  loco_t::shape_t l;
  void open(auto sod) {
    loco_t::rectangle_t::properties_t p;
    p.camera = &pile->camera;
    p.viewport = &pile->viewport;
    p.position = 0;
    p.position.z = 10;
    p.size = 0.2;
    p.color = fan::random::color();
    l = p;
  }

  void close() {
		
  }

  void window_resize(){
		
  }

  void update(){
	
  }

};

int main(int argc, char** argv) {
  
  if (argc < 2) {
    fan::throw_error("usage: TexturePackCompiled");
  }


  loco_t::texturepack_t tp;
  tp.open_compiled(&pile->loco, argv[1]);
  pile->stage_loader.open(&tp);

	stage_loader_t::stage_open_properties_t op;
	op.camera = &pile->camera;
	op.viewport = &pile->viewport;
	op.theme = &pile->theme;

  //stage_loader_t::nr_t it3 = stage_loader_t::open_stage<custom_t>(op);
  //it3 = stage_loader_t::open_stage<custom_t>(op);
  stage_loader_t::nr_t it3 = stage_loader_t::open_stage<stage_loader_t::stage::stage_fuel_station_t>(op);
  //it2.erase();
  

	pile->loco.loop([&] {

	});
	
}