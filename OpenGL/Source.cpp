#include <fan/graphics.hpp>

int main() {

	fan::window window;

	fan::camera camera(window);

	fan_2d::gui::rounded_text_box rtb(camera, "test string", 64, 0, fan::colors::cyan, 20, 30);

	window.loop(0, [&] {

		rtb.draw();

	});

}