#include <fan/graphics/graphics.hpp>
#include <fan/physics/collision/rectangle.hpp>

int main() {

	fan::window window;

	fan::camera camera(&window);

	fan_2d::rectangle player(&camera);

	player.push_back(fan::vec2(500, 500), 50, fan::colors::blue);

	fan_2d::rectangle walls(&camera);

	walls.push_back(fan::vec2(400, 400), fan::vec2(400, 10), fan::colors::red);
	walls.push_back(fan::vec2(400, 400), fan::vec2(10, 400), fan::colors::red);


	window.loop(0, [&] {

		window.get_fps();

		player.draw();
		walls.draw();

		player.move(0, 100);

		fan_2d::collision::rectangle::resolve_collision(player, walls);

	});

	return 0;
}