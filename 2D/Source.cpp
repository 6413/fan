#include <FAN/Graphics.hpp>

int main() {

	vec2 sizes[] = { 10, 50, 100, 200 };

	fan_2d::square player(window_size / 2, sizes[random(0, std::size(sizes) - 1)], Color::rgb(0, 0, 255));

	fan_2d::square_vector walls;
	for (int i = 0; i < 20; i++) {
		vec2 position(random(0, window_size.x), random(0, window_size.y));
		vec2 size(random(10, 200), random(10, 200));
		if (!rectangles_collide(player.get_position(), player.get_size(), position, size)) {
			walls.push_back(position, size, random_color());
		}
	}

	callbacks::key_callback.add(GLFW_KEY_R, true, [&] {
		while (!walls.empty()) {
			walls.erase(0);
		}
		player.set_size(sizes[random(0, std::size(sizes) - 1)]);
		for (int i = 0; i < 30; i++) {
			vec2 position(random(0, window_size.x), random(0, window_size.y));
			vec2 size(random(10, 200), random(10, 200));
			if (!rectangles_collide(player.get_position(), player.get_size(), position, size)) {
				walls.push_back(position, size, random_color());
			}
		}
	});

	fan_window_loop() {
		begin_render(Color::rgb(0, 0, 0));

		vec2 old_position = player.get_position();

		player.move(100, 0, 0, 10);

		collision_rectangle_2d(player, old_position, walls);

		player.draw();
		walls.draw();

		end_render();
	}

	return 0;
}