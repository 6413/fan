#include <fan/pch.h>

#include _FAN_PATH(system.h)
#include _FAN_PATH(time/time.h)
#include <thread>

int main() {

  std::vector<fan::vec2> p;

  fan::string filename;
  printf("type preset name (leave empty if none):");
  std::getline(std::cin, filename);
  uint64_t time;
  if (!filename.empty()) {
    std::fstream f(filename);
    fan::string line;
    while (std::getline(f, line)) {
      if (line.empty()) {
        continue;
      }
      p.push_back(fan::string_to<fan::vec2i>(line));
    }    
    fan::print("preset.txt loaded");
  }

  printf("enter loop time in seconds:");
  std::cin >> time;

	fan::sys::input input;

	bool direction = false;

	fan::print("press f2 to initialize clicking position");
	fan::print("press f4 start loop");
  fan::print("press f7 to save preset");
	fan::print("press f9 quit loop");

	volatile bool quit = 0;

	input.listen_keyboard([&](int key, fan::keyboard_state state, bool action) {

		if (state != fan::keyboard_state::press) {
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
							fan::sys::input::send_mouse_event(fan::mouse_left, fan::mouse_state::press);
							fan::delay(fan::time::nanoseconds(0.05e+9));
							fan::sys::input::send_mouse_event(fan::mouse_left, fan::mouse_state::release);
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
      case fan::key_f7: {
        auto f = [&] {
          if (fan::io::file::exists("preset.txt")) {
            fan::print("preset.txt already exists - rename or delete it");
            return;
          }
          std::ofstream f;
          f.open("preset.txt");
          for (int i = 0; i < p.size(); ++i) {
            f << p[i].x << ' ' << p[i].y << '\n';
          }
          fan::io::file::write("preset.txt", p);
          fan::print("preset saved to preset.txt");
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
  input.thread_loop([]{});
  std::cin.get();
}