#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#define fan_debug 1
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#define loco_window
#define loco_context

#include _FAN_PATH(graphics/loco.h)


#define stage_loader_path ../../
#include _FAN_PATH(graphics/gui/stage_loader.h)

struct pile_t {

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

	auto stage0ptr = pile.stage_loader.get_stage<stage_loader_t::stage::stage0_t>(0);
	stage0ptr->open();
	stage0ptr->close();
	

	pile.loco.loop([&] {

		stage0ptr->update();

	});
	
}