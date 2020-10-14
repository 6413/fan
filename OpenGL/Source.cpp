#include <FAN/graphics.hpp>

int main() {

	fan_3d::add_camera_rotation_callback();
	
	fan_3d::square_vector s("sides_05.png", 32);
	fan_3d::line_vector l(fan::mat2x3(fan::vec3(0), fan::vec3(100, 100, 50)), fan::color(1));

	s.push_back(fan::vec3(2, 0, 0), 2, fan::vec2(0, 1));

	fan::window_loop(fan::color::hex(0x87ceeb), [&] {

		fan_3d::camera.move(1000);

		s.draw();

		l.draw();

	});

	return 0;
}