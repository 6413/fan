#include <FAN/graphics.hpp>

int main() {

	fan_3d::add_camera_rotation_callback();

	fan_3d::terrain_generator tg("grass.jpg", 150, 10, 5);

	fan_3d::camera.set_position(fan::vec3(500, 20, 500));

	fan::window_loop(0, [&] {
		fan_3d::camera.move(true, 300);
		tg.draw();
	});

	return 0;
}