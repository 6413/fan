#include <fan/graphics/graphics.hpp>

int main(){
	fan::window window;
	fan::camera camera(&window);

	fan_2d::gui::text_renderer tr(&camera, L"test", 0, 1, 64);

	window.loop(0, [&] {

		window.get_fps();

		tr.draw();

	});

	return 0;
}