#include <fan/pch.h>

struct pile_t;

struct stage_loader_t;

struct pile_t {
  loco_t loco;
  stage_loader_t* stage_loader;
}pile;

#define stage_loader_path .
#include _FAN_PATH(graphics/gui/stage_maker/loader.h)


struct a_t {
  typedef a_t b_t;
  a_t() {

  }
};

// make custom stage
lstd_defstruct(custom_t)
  #include _FAN_PATH(graphics/gui/stage_maker/preset.h)

  static constexpr auto stage_name = "";

  fan::graphics::rectangle_t r;

  void open(void* sod) {
    fan::print("opened");
  }

  void close() {
    fan::print("closed");
  }

  void window_resize() {
    fan::print("resized");
  }

  void update() {
    //fan::print("update");
  }
};

int main() {

  fan::vec2 window_size = pile.loco.window.get_size();
  pile.loco.camera_set_ortho(
    pile.loco.orthographic_render_view.camera,
    fan::vec2(0, window_size.x),
    fan::vec2(0, window_size.y)
  );


  loco_t::texturepack_t tp;
  tp.open_compiled("TexturePack");
  pile.stage_loader = new stage_loader_t;

  stage_loader_t::stage_open_properties_t op;

  stage_loader_t::nr_t it_custom = stage_loader_t::open_stage<custom_t>(op);
  pile.stage_loader->erase_stage(it_custom);
  //auto it_stage0 = stage_loader_t::open_stage<stage_loader_t::stage::stage0_t>(op);

  pile.loco.loop([] {

  });
}