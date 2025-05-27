#include <fan/time/timer.h>

#include "cuda_runtime.h"
#include <cuda.h>
#include <nvcuvid.h>


import fan;

//#define loco_vulkan

struct pile_t {

  static constexpr fan::vec2 ortho_x = fan::vec2(-1, 1);
  static constexpr fan::vec2 ortho_y = fan::vec2(-1, 1);

  pile_t() {

    fan::vec2 window_size = loco.window.get_size();
    camera = loco.camera_create(
      ortho_x,
      ortho_y
    );
    viewport = loco.viewport_create(0, window_size, window_size);
  }

  loco_t loco;
  loco_t::camera_t camera;
  fan::graphics::viewport_t viewport;
  fan::graphics::shape_t cid[5];
};

pile_t* pile = new pile_t;

#include <fan/video/nvdec.h>

int main() {
  pile->loco.set_vsync(false);

  fan::cuda::nv_decoder_t nv(&pile->loco);

  bool pushed = false;

  nv.sequence_cb = [&] {
    if (!pushed) {
      pile->cid[1].erase();
    }
    loco_t::universal_image_renderer_t::properties_t p;
    p.camera = pile->camera;
    p.viewport = pile->viewport;
    p.size = 1;
    pile->cid[1] = pile->loco.universal_image_renderer.push_back(p);
    pushed = true;
  };

  fan::string video_data;
  fan::io::file::read("videos/output.h264", &video_data);

  auto nr = pile->loco.m_update_callback.NewNodeLast();
  pile->loco.m_update_callback[nr] = [&nv] (loco_t* loco) {
    if (nv.images[0].iic() == false) {
      fan::graphics::image_t images[4]{};
      images[0] = nv.images[0];
      images[1] = nv.images[1];
      pile->cid[1].reload(fan::graphics::image_format::nv12, images);
    }
    fan::vec2 window_size = loco->window.get_size();
    loco->viewport_set(pile->viewport, 0, window_size, window_size);
  };

  nv.start_decoding(video_data);
 /* pile->loco.loop([&] {
    
    
  });*/

  //pile->loco.loop([] {});

   fan::print(nv.timestamp.elapsed(), nv.current_frame, nv.timestamp.elapsed() / nv.current_frame);

  return 0;
}