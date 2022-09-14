#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#define fan_debug 1
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#define loco_window
#define loco_context

#include _FAN_PATH(graphics/loco.h)

struct pile_t {

	#define stage_loader_path ../../
	#include _FAN_PATH(graphics/gui/stage_loader.h)

	stage_loader_t stage_loader;
	loco_t loco;

	void open() {
		loco_t::properties_t p;
		loco.open(p);
		stage_loader.open();
	}
	void close() {
		loco.close();
		stage_loader.close();
	}

};

int main() {

	pile_t pile;
	pile.open();

	pile.stage_loader.

	pile.loco.loop([&] {

	});
	
}