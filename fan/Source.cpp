#include <fan/graphics/gui.hpp>

int main() {

	fan::window window;

	window.set_vsync(false);

	fan::camera camera(&window);

	fan_2d::graphics::sprite s(&camera);

	fan_2d::graphics::sprite::properties_t properties;
	auto image = fan_2d::graphics::load_image(&window, "images/cursor.png");

	properties.position = 0;
	properties.size = image->size;
	properties.image = image;

	s.push_back(properties);

	s.write_data();

	window.loop([&] {

		s.draw();
	});

	delete image;
}