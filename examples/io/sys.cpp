#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(system.h)
#include _FAN_PATH(time/time.h)
#include <thread>

int main() {
	printf("enter loop time in seconds:");
	uint64_t time;
	std::cin >> time;

	fan::sys::input input;

	std::vector<fan::vec2> click_position;

	bool direction = false;

	fan::print("press f2 to initialize clicking position");
	fan::print("press f4 start loop");
	fan::print("press f9 quit loop");

	std::vector<fan::vec2> p;

	volatile bool quit = 0;

	input.listen([&](uint16_t key, fan::key_state state, bool action, std::any data) {

		if (state != fan::key_state::press) {
			return;
		}

		if (!action) {
			return;
		}

		switch (key) {
			case fan::key_f2: {
				auto f = [&] {
					p.push_back(fan::sys::input::get_mouse_position());
					fan::print("initialized click:", p.size() - 1, "to:", *(p.end() - 1));
				};

				std::thread t(f);

				t.detach();

				break;
			}
			case fan::key_f4: {
				auto f = [&] {
					while (1) {
						for (uint32_t i = 0; i < p.size(); i++) {
							if (quit) {
								quit = 0;
								return;
							}
							fan::sys::input::set_mouse_position(p[i]);
							fan::sys::input::send_mouse_event(fan::mouse_left, fan::key_state::press);
							fan::delay(fan::time::nanoseconds(0.05e+9));
							fan::sys::input::send_mouse_event(fan::mouse_left, fan::key_state::release);
							fan::print("clicked position", p[i]);
							fan::delay(fan::time::nanoseconds(1.5e+9));
						}
						fan::delay(fan::time::nanoseconds(time * 1e+9));
					}
				};

				std::thread t(f);

				t.detach();

				break;
			}
			case fan::key_f9: {
				quit = 1;
				break;
			}
		}

	});
}