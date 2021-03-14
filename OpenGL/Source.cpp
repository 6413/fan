#include <fan/graphics/graphics.hpp>

int main() {

	fan::window window;

	fan::camera camera(&window);

	fan_2d::sprite s(&camera, "images/walltex.jpg", 0);

	window.loop(0, [&] {

		window.get_fps();

		s.draw();

	});

	return 0;
}