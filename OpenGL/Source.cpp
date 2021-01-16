#include <fan/graphics.hpp>

int main() {

	fan::window window(fan::window::default_window_name, fan::window::resolutions::r_1360x768);

	window.vsync(0);

	fan::camera camera(window);

	window.add_key_callback(fan::key_escape, [&] {
		window.close();
	});

	fan_2d::line_vector lv(camera);

	lv.resize(4, fan::colors::white);

	window.loop(0, [&] {

		window.get_fps();

		lv.set_position(0, fan::mat2(fan::vec2(), window.get_mouse_position()));
		lv.set_position(1, fan::mat2(fan::vec2(window.get_size().x, 0), window.get_mouse_position()));
		lv.set_position(2, fan::mat2(fan::vec2(0, window.get_size().y), window.get_mouse_position()));
		lv.set_position(3, fan::mat2(window.get_size(), window.get_mouse_position()));

		lv.draw();

	});
	
}