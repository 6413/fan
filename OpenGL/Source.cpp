#include <FAN/graphics.hpp>

int main() {
	fan_3d::add_camera_rotation_callback();

	fan_2d::sprite crosshair("crosshair.png", fan::window_size / 2 - 2, 4);

	fan_3d::square_vector s("sides_05.png", 32);
	s.push_back(fan::vec3(0, 0, 500), 10, fan::vec2(3, 1));

	fan::callback::window_resize.add([&]() {
		crosshair.set_position(fan::window_size / 2 - 2);
	});
	
	fan_3d::terrain_generator tg("grass.jpg", 100, 0, fan::vec2(100), 10, 5);

	fan::window_loop(fan::color::hex(0), [&] {
		fan_3d::camera.move(1000);

		tg.draw();

		s.set_position(0, s.get_position(0) + fan::vec3(fan::delta_time * 100, fan::delta_time * 100, 0));

		s.draw();

		fan::gui_draw([&] {
			crosshair.draw();
		});
	});

	return 0;
}