// this implementation is highly experimental

#include <emscripten/html5.h>
#include <emscripten.h>
#include <GLES3/gl3.h>
import fan;

using namespace fan::graphics;

EM_BOOL loop(double time, void* user_data) {
  engine_t* engine = static_cast<engine_t*>(user_data);

  engine->set_clear_color(fan::colors::blue);

  engine->process_frame([] {
    //glClearColor(0, 1, 0, 1);
    //glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  });

  return EM_TRUE;
}

int main() {
  auto dummy = []() {};
  emscripten_set_main_loop(dummy, 0, 0);
  emscripten_pause_main_loop();

  static engine_t engine; 

  emscripten_request_animation_frame_loop(loop, &engine);

  return 0;
}