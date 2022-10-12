#include <fan/types/types.h>
#include <fan/time/timer.h>

int main() {
	fan::ev_timer_t timer;
	int x = 0;
	fan::ev_timer_t::timer_t t([&] (const fan::ev_timer_t::cb_data_t& c) {
		fan::print("aaa");
		c.ev_timer->start(c.timer, 1e+9);
		if (x >= 2) {
			c.ev_timer->stop(c.timer);
		}
		x++;
	});
	timer.start(&t, 1e+9);

	while (1) {
		timer.process();
	}
}