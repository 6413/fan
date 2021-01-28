#include <fan/graphics.hpp>

int main() {

	fan::window window(fan::window::resolutions::r_800x600);

	window.set_max_fps(fan::get_screen_refresh_rate());

	fan::camera camera(window);

	const fan::vec2 border_size(40);
	const f_t radius = 30;

	fan_2d::gui::rounded_text_box rtb(camera, L"", 32, 0, fan::colors::purple - 0.4, border_size, radius);

	rtb.set_input_callback(0);

	window.loop(0, [&] {

		rtb.draw();

	});

}