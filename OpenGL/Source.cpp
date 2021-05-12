#include <fan/engine/graphics/graphics.hpp>
#include <fan/graphics/gui.hpp>

int main() {

	fan::window::set_flag_value<fan::window::flags::anti_aliasing>(8);
	fan::window::set_flag_value<fan::window::flags::borderless>(true);

	fan_2d::engine::engine_t engine(fan::vec2(0, 10));

	engine.m_window.set_error_callback();

	engine.m_window.set_size(fan::get_resolution());

	f_t time_step = fan::get_screen_refresh_rate();

	engine.m_window.set_max_fps(time_step);

	const fan::vec2 window_size = engine.m_window.get_size();

	fan_2d::graphics::gui::rectangle gui_r(&engine.m_camera);

	gui_r.push_back(fan::vec2(window_size.x - 100, window_size.y - 500), 50, fan::colors::red);
	gui_r.push_back(0, 0, fan::colors::red);

	fan_2d::graphics::gui::circle gui_c(&engine.m_camera);

	gui_c.push_back(fan::vec2(window_size.x - 75, window_size.y - 400), 25, fan::colors::white);
	gui_c.push_back(0, 0, fan::colors::white);

	fan_2d::engine::circle c(&engine);

	fan_2d::engine::rectangle r(&engine);

	const fan::color active = fan::colors::green;
	const fan::color inactive = fan::colors::red;

	bool dynamic = false;

	fan_2d::graphics::gui::selectable_text_box stb(&engine.m_camera, L"static", 32, fan::vec2(window_size.x - 300, window_size.y - 100), active, 10);

	stb.push_back(L"dynamic", 32, fan::vec2(window_size.x - 200, window_size.y - 100), inactive, 10);

	stb.on_click([&](uint32_t i) {
		switch (i) {
			case 0:
			{
				stb.set_box_color(0, active);
				stb.set_box_color(1, inactive);
				dynamic = false;

				break;
			}
			case 1:
			{
				stb.set_box_color(0, inactive);
				stb.set_box_color(1, active);
				dynamic = true;

				break;
			}
		}
	});

	// 1 rectangle, 2 circle
	int mode = 0;
	bool pressing = false;
	int r_current = 0, c_current = 0;

	//block view properties
	fan::vec2 starting_position;
	fan::vec2 size;
	f32_t rotation = 0;

	engine.m_window.add_key_callback(fan::mouse_left, [&] {

		if (stb.inside(0) || stb.inside(1)) {
			return;
		}

		if (gui_r.inside(0)) {
			mode = 1;
		}
		else if (gui_c.inside(0)) {
			mode = 2;
		}
		else {
			starting_position = engine.m_window.get_mouse_position();

			switch (mode) {
				case 1: { gui_r.set_position(1, starting_position); gui_r.set_size(1, 1); gui_r.set_rotation(1, 0); break; }
				case 2: { gui_c.set_position(1, starting_position); gui_c.set_radius(1, 1); break; }
			}
			pressing = true;
		}
	});

	engine.m_window.add_key_callback(fan::mouse_left, [&] {

		if (!pressing) {
			return;
		}

		switch (mode) {
			case 1:
			{
				r.push_back(starting_position, size, fan::colors::red, !dynamic ? fan_2d::physics::body_type::static_body : fan_2d::physics::body_type::dynamic_body);

				gui_r.set_size(1, 0);
				gui_r.set_rotation(1, 0);


				r.set_rotation(r_current++, rotation);

				break;
			}
			case 2:
			{
				c.push_back(starting_position, size.x, fan::colors::white, !dynamic ? fan_2d::physics::body_type::static_body : fan_2d::physics::body_type::dynamic_body);
				
				gui_c.set_radius(1, 0);

				c_current++;

				break;
			}
		}

		rotation = 0;
		pressing = false;

	}, true);

	engine.m_window.add_mouse_move_callback([&] (fan::window* window) {
		if (pressing) {
			fan::vec2 new_size;

			f32_t xdiff = window->get_mouse_position().x - starting_position.x;
			f32_t ydiff = window->get_mouse_position().y - starting_position.y;

			if (xdiff < 1) {
				new_size.x = 1;
			}
			else {
				new_size.x = xdiff;
			}

			if (ydiff < 1) {
				new_size.y = 1;
			}
			else {
				new_size.y = ydiff;
			}

			switch (mode) {
				case 1:
				{
					gui_r.set_size(1, new_size);

					size = new_size;

					break;
				}
				case 2:
				{

					gui_c.set_radius(1, new_size.x);

					size.x = new_size.x;

					break;
				}

			}
			
		}
	});

	engine.m_window.add_key_callback(fan::key_escape, [&] {
		engine.m_window.close();
	});

	//engine.m_window.add_key_callback(fan::mouse_right, [&] {

	//	for (int i = r.size(); i--; ) {
	//		if (r.inside(i)) {
	//			r.erase(i);
	//			r_current--;
	//			break;
	//		}
	//	}

	//	for (int i = c.size(); i--; ) {
	//		if (c.inside(i)) {
	//			c.erase(i);
	//			c_current--;
	//			break;
	//		}
	//	}
	//});

	engine.m_window.add_scroll_callback([&](uint16_t key) {

		if (!pressing) {
			return;
		}

		if (key == fan::mouse_scroll_up) {
			gui_r.set_rotation(1, gui_r.get_rotation(1) + 0.1);
			rotation = gui_r.get_rotation(1);
		}
		else if (key == fan::mouse_scroll_down) {
			gui_r.set_rotation(1, gui_r.get_rotation(1) - 0.1);
			rotation = gui_r.get_rotation(1);
		}
	});

	engine.m_window.loop(0, [&] {

		engine.m_window.get_fps();

		c.update_position();

		r.update_position();

		r.draw();

		c.draw();
		
		gui_r.draw();
		gui_c.draw();

		stb.draw();

		engine.m_world->Step(1.0 / time_step, 6, 2);

	});

	return 0;
} 