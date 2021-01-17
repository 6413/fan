#include <fan/graphics.hpp>

int main() {
	fan::window window;
	fan::camera camera(window);
	fan_2d::rectangle rectangle(camera, fan::vec2(100, 200), fan::vec2(25, 100), fan::colors::white);

	fan_2d::gui::text_renderer tr(camera, "hello", 0, fan::colors::red, 64);

	window.loop(0, [&] { 

		rectangle.draw(); 

		tr.draw();

	});
}