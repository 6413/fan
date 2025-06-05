#include <fan/types/types.h>
#include <fan/time/timer.h>
#include <fan/event/types.h>

import fan;
import fan.graphics.video.screen_codec;

using namespace fan::graphics;

int main() {
  engine_t engine;


  fan::graphics::universal_image_renderer_t screen_frame{ {
    .position = fan::vec3(gloco->window.get_size() / 2, 0),
    .size = gloco->window.get_size() / 2,
  } };
  screen_codec_t screen_codec;
  uint64_t bitrate = 0;

  engine.loop([&] {
    if (!screen_codec.screen_read()) {
      return;
    }

    if (screen_codec.encode_write()) {
      uintptr_t read = screen_codec.encode_read();
      bitrate += read;
      fan_ev_timer_loop(1000, {
        fan::print(bitrate);
        bitrate = 0;
      });
      screen_codec.decode(screen_frame);
    }
    screen_codec.sleep_thread();
  });
}