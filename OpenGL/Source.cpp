#include <fan/graphics.hpp>

int main() {
	fan::gui::font::text_button_vector_sprite v(L"this is test", fan::window_size / 2, "images/button.png", 32);

	fan::window_loop(fan::color(0), [&] {
		fan::gui_draw([&] {
			v.draw();
		});
	});
	return 0;
}