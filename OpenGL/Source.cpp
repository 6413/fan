#include <fan/graphics.hpp>

#include <locale>

#include <codecvt>

int main() {

	fan::window window;

	fan::camera camera(window);

	fan_2d::gui::rounded_text_box s(camera, L"test string", 32, 0, fan::colors::purple - 0.4, 40, 30);

	s.set_input_callback(0);

	window.add_key_callback(fan::mouse_scroll_up, [&] {
		s.set_font_size(0, s.get_font_size(0) + 50);
	});

	window.add_key_callback(fan::mouse_scroll_down, [&] {
		s.set_font_size(0, s.get_font_size(0) - 50);
	});

	window.loop(0, [&] {

		window.get_fps();

		s.draw();

	});

}