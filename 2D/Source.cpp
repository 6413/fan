#include <FAN/graphics.hpp>

int main() {
	fan_2d::square player(window_size / 2, 400, Color(0, 0, 1));

	constexpr f32_t movement_speed = 100;
	constexpr auto amount_of_walls = 50;

	vec2 sizes[] = { 10, 30, 80, 150 };

	fan_2d::square_vector walls;
	auto generate_world = [&] {
		player.set_size(sizes[random(0, std::size(sizes) - 1)]);
		for (int i = 0; i < amount_of_walls; i++) {
			vec2 position(random(0, window_size.x), random(0, window_size.y));
			vec2 size(random(10, 200), random(10, 200));
			if (!rectangles_collide(player.get_position(), player.get_size(), position, size)) {
				walls.push_back(position, size, random_color());
			}
		}
	};

	generate_world();

	callback::key.add(GLFW_KEY_R, true, [&] {
		while (!walls.empty()) {
			walls.erase(0);
		}
		generate_world();
	});

	window_loop(Color(0), [&] {
		player.move(movement_speed);
		rectangle_collision_2d(player, walls);
		walls.draw();
		player.draw();
	});

    return 0;
}