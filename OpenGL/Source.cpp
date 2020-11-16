#include <fan/graphics.hpp>

int main() {

	fan_2d::gui::text_renderer s;

	s.push_back("abc", 0, fan::color(1, 0, 0), 64, fan::color(0, 1, 1));
	s.push_back("def", fan::vec2(0, 200), fan::color(0, 0, 1), 64, fan::color(1, 1, 0));

	s.set_text(1, "hello");

	fan::window_loop(0, [&] {
		s.draw();
	});

	return 0;
}