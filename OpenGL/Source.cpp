#include <fan/graphics.hpp>
#include <ctime>

int main() {
	fan_2d::gui::text_renderer s;

	s.push_back("", 0, fan::color(1, 0, 0), 64, fan::color(1));

	fan::_timer<fan::milliseconds> timer(fan::timer::start());

	fan::window_loop(0, [&] {
		auto end = fan::system_clock::now();
		std::time_t end_time = fan::system_clock::to_time_t(end);

		s.set_text(0, std::ctime(&end_time), true);

		s.set_outline_color(0, fan::color(fabs(sin((f_t)timer.elapsed() / 1000))), true);

		s.free_queue();
		
		s.draw();
	});

	return 0;
}