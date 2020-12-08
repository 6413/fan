#include <fan/graphics.hpp>

int main() {

	fan::window window("window", fan::window::default_window_size);

	window.vsync(false);

	window.add_key_callback(fan::input::key_escape, [&] {
		window.close();
	});

	bool fullscreen = false;

	window.add_key_callback(fan::input::key_f, [&] {
		if (fullscreen = !fullscreen) {
			window.set_full_screen(fan::window::resolutions::r_1920x1080);
		}
		else {
			window.set_windowed();
		}
	});

	fan::camera c(window);

	fan_2d::square s(c, window.get_size() / 2, 100, fan::color(1, 0, 0, 1));

	fan_2d::gui::text_renderer tr(c, " ", 0, fan::colors::white, 16, fan::colors::blue);

	window.loop(fan::colors::black, [&] {
		
		if (uint_t fps = window.get_fps()) {
			tr.set_text(0, "fps: " + std::to_string(fps));
		}

		s.draw();
		tr.draw();

		s.move(100);

	});
}