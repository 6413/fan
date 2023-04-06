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

#define loco_no_inline

#define loco_sprite
#define loco_button

#include _FAN_PATH(graphics/loco.h)

struct pile_t {
  loco_t loco;
}*pile;

struct pile2_t {
  loco_t loco;
}*pile2;

#include _FAN_PATH(graphics/loco_define.h)

int main() {
  loco_t::id_t id;
	/*pile->loco.loop([&] {

	});
	*/
}