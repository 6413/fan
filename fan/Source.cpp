#include <fan/graphics/graphics.hpp>

int main() {

	fan::window window;

	fan::camera camera(&window);

	fan_2d::graphics::rectangle r(&camera);

	window.add_key_callback(fan::key_escape, fan::key_state::press, [&] {
		window.close();
	});

	fan::begin_queue();

	for (int i = 0; i < 1000; i++) {
		r.push_back(fan::vec2(rand() % 800, rand() % 800), rand() % 50 + 10, fan::random::color());
		r.set_angle(i, fan::math::radians(rand() % 360));
	}
	
	fan::end_queue();

	r.release_queue();

	window.loop([&] {

		window.get_fps();

		r.draw();

		fan::begin_queue();

		for (int i = 0; i < r.size(); i++) {
			r.set_angle(i, r.get_angle(i) + window.get_delta_time());
		}

		fan::end_queue();

		r.release_queue();

	});
}