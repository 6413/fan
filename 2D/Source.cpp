#include <FAN/graphics.hpp>

int main() {

	fan_2d::square_vector square(window_size / 2, 100, Color(0, 0, 1));

	fan_2d::square s = square.construct(0);

	window_loop(Color(0), [&] {
		square.draw();
		s.move(100);
		square.move(0, 300);
		s.draw();
	});


	return 0;
}