#include <fan/graphics/gui.hpp>

int main() {

	fan::window window;

	fan::camera camera(&window);

	fan_2d::graphics::gui::circle_text_slider<int> cs(&camera);

	fan_2d::graphics::gui::circle_slider<int>::property_t property;
	property.position = 100;
	property.box_size = fan::vec2(100, 5);
	property.box_radius = 50;
	property.button_radius = 10;
	property.current = 0;
	property.min = 0;
	property.max = 10;
	property.box_color = fan::colors::red;
	property.button_color = fan::colors::cyan;
	property.text_color = fan::colors::white;
	property.font_size = 32;

	cs.push_back(property);

	cs.on_drag([&](uint32_t i) {
		fan::print(cs.get_current_value(i));
	});

	cs.write_data();

	window.loop([&] {

		cs.draw();

	});

}