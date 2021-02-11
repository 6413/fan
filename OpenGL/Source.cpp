#include <fan/graphics.hpp>

int main(){
	fan::window window;
	fan::camera camera(&window);

	window.add_key_callback(fan::key_escape, [&] {
		window.close();
	});

	fan_2d::gui::text_renderer tr(&camera, L"text", 10, fan::colors::white, 64);


	window.loop(0, [&] {
		tr.draw();
	});

	return 0;
}
