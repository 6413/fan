#include <fan/graphics/gui.hpp>

int main() {

	fan::window window;

	fan::camera camera(&window);

	fan_2d::graphics::gui::text_renderer tr(&camera);

	tr.push_back(L"text renderer", 64, 100, fan::colors::white);

	window.loop([&] {

		window.get_fps();

		tr.draw();

	});
}
