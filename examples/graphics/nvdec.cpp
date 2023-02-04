// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 1
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

//#define loco_vulkan


#define loco_window
#define loco_context

//#define loco_rectangle
#define loco_nv12
//#define loco_sprite
#include _FAN_PATH(graphics/loco.h)

struct pile_t {

  static constexpr fan::vec2 ortho_x = fan::vec2(-1, 1);
  static constexpr fan::vec2 ortho_y = fan::vec2(-1, 1);

  pile_t() {

    fan::vec2 window_size = loco.get_window()->get_size();
    loco.open_matrices(
      &matrices,
      ortho_x,
      ortho_y
    );
    loco.get_window()->add_resize_callback([&](const fan::window_t::resize_cb_data_t& d) {
      fan::vec2 window_size = d.size;
    //fan::vec2 ratio = window_size / window_size.max();
    //std::swap(ratio.x, ratio.y);
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
  loco_t::matrices_t matrices;
  fan::graphics::viewport_t viewport;
  fan::graphics::cid_t cid[5];
};

pile_t* pile = new pile_t;
loco_t::nv12_t::properties_t p;

#include _FAN_PATH(video/nvdec.h)

int main() {

  pile->loco.set_vsync(false);

  fan::cuda::nv_decoder_t nv;

  p.matrices = &pile->matrices;
  p.viewport = &pile->viewport;
  p.size = 1;
  p.y = &nv.image_y;
  p.vu = &nv.image_vu;
  pile->loco.nv12.push_back(&pile->cid[1], p);

  fan::string video_data;
  fan::io::file::read("encode", &video_data);

  nv.start_decoding(video_data);
  pile->loco.loop([] {});

   //fan::print(nv.timestamp.elapsed(), nv.current_frame, nv.timestamp.elapsed() / nv.current_frame);

  return 0;
}