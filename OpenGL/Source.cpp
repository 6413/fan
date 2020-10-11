#include <FAN/graphics.hpp>

int main() {

	fan_3d::add_camera_rotation_callback();

	fan_3d::terrain_generator tg("grass.jpg", 150, 10, 5);

	fan_3d::camera.set_position(fan::vec3(500, 20, 500));
	fan_gui::font::text_button_vector text(L"test", fan::window_size / 2, fan::color(1, 0, 0), 32);

	fan::window_loop(0, [&] {

		fan_3d::camera.move(true, 300);
		tg.draw();

		fan::gui_draw([&] {
			text.draw();
		});

	});

	return 0;
}