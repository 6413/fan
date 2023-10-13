// rectangle text button using loco

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 1
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

//#define loco_vulkan

#define loco_window
#define loco_context

//#define loco_post_process
#define loco_button
#include _FAN_PATH(graphics/loco.h)

int main() {

  loco_t loco;

  loco_t::simple_button_t sb{{
    .position = {-0.85, 0.3, 0}, 
    .size = {0.1, 0.1},
    .mouse_button_cb = [] (const auto&){
      fan::print("hello");
      return 0;
    }
  }};

  loco_t::simple_text_t st;
  loco.loop([&] {
    st = { {.text = fan::to_string(rand())}};
  });

  return 0;
}
