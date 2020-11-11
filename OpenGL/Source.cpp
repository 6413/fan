#include <fan/graphics.hpp>

int main() {
	fan_2d::gui::text_renderer tr(32);

	std::wstring str;
	
	int index = 0;
	for (int i = 0; i < 50; i++) {
		for (int j = 0; j < 90; j++) {
			str.push_back((index = (index + 1) % 248));
		}
		str.push_back('\n');
	}
	tr.store_text(str, 0, 1);

	fan::window_loop(0, [&] {
		tr.draw();
	});

	return 0;
}