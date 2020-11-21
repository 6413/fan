#include <fan/graphics.hpp>

int main() {
	fan::window window(fan::window_flags::NO_MOUSE);
	fan::camera camera3d(&window);

	fan_3d::add_camera_rotation_callback(&window, camera3d);

	fan_3d::terrain_generator tg(&window, &camera3d, "images/grass.jpg", 10, 0, 100, 10, 5);

	fan::window_loop(&window, 0, [&] {
		camera3d.move(500);

		tg.draw();
	});
}