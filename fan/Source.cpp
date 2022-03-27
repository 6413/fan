#define fan_debug 0
#include <fan/graphics/graphics.h>
#include <fan/graphics/gui.h>

int main() {
	//fan::window_t::set_flag_value<fan::window_t::flags::no_mouse>(true);
	fan::window_t w;
	w.open(fan::get_screen_resolution());

	fan::opengl::context_t context;
	context.init();
	context.bind_to_window(&w);
	context.set_viewport(0, w.get_size());
	w.add_resize_callback([&](fan::window_t*, fan::vec2) {
		context.set_viewport(0, w.get_size());
	});

	w.set_vsync(0);

	fan::opengl::image_t* image = fan::opengl::create_texture(1, fan::colors::red);

	fan_2d::opengl::particles_t pr;
	fan_2d::opengl::particles_t::properties_t p;
	pr.open(&context);
	p.image = image;
	p.size = image->size;
	p.count = 1000000;

	pr.set(p);

	pr.enable_draw(&context);

	while (1) {

		if (w.get_fps()) {
			printf("amount of triangles: %llu\n", pr.size(&context));
		}

		uint32_t window_event_flag = w.handle_events();
    if(window_event_flag & fan::window_t::events::close){
      w.close();
      break;
    }

		//if (w.key_press(fan::mouse_left)) {
		//	p.position = w.get_mouse_position();
		//	p.angle = fan::random::value_f32(0, fan::math::two_pi);
		//	p.angle_velocity = fan::random::value_f32(1, 10);//
		//	f32_t r = fan::random::value_f32(0, fan::math::pi * 2);
		//	p.position_velocity = fan::random::vec2_direction(r, r + fan::random::value_f32(0, fan::math::pi * 2 - r)) * 50;
		//	p.color = fan::random::color();
		//	p.timeout = 4e+9;
		//	pr.push_back(&context, p);
		//}

		pr.set_delta(&context, (f64_t)fan::time::clock::now() / 1e+9);

		context.process();
		context.render(&w);
	};

	return 0;
}