#include <fan/types/types.h>
#include <fan/time/time.h>
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
  screen_encode_t screen_encode;
  screen_decode_t screen_decode;
  uint64_t bitrate = 0;

  uint64_t encode_start = fan::time::clock::now();
  screen_encode.init(encode_start);

  screen_decode.init(encode_start);

  engine.loop([&] {
    if (!screen_encode.screen_read()) {
      return;
    }

    if (screen_encode.encode_write(screen_decode.FrameProcessStartTime)) {
      uintptr_t read = screen_encode.encode_read();
      bitrate += read;
      fan_ev_timer_loop(1000, {
        fan::print(bitrate);
        bitrate = 0;
      });
      screen_decode.decode(screen_encode.data, screen_encode.amount, screen_frame);
    }
    screen_decode.sleep_thread(screen_encode.settings.InputFrameRate);
  });
}