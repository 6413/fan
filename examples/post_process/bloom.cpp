#include <fan/pch.h>

int main() {
  loco_t loco;

	loco_t::sprite_t::properties_t p;

	loco_t::image_t image;
  image = loco.image_load("images/brick.webp");
	p.image = image;
	p.size = loco.image_get_data(image).size;
	p.position = 400;
	p.position.z = 0;
  p.flags = 0x2;
  loco_t::shape_t shape = p;

	//loco.post_process.push(&viewport, &camera);

	loco.set_vsync(false);

	/*loco.window.add_buttons_callback([&](const fan::window_t::mouse_buttons_cb_data_t& d) {

		if (d.state != fan::mouse_state::press) {
			return;
		}

		switch (d.button) {
		case fan::mouse_scroll_up: {
			loco.bloom.bloomamount += 0.01;
			break;
		}
		case fan::mouse_scroll_down: {
			loco.bloom.bloomamount -= 0.01;
			break;
		}
		}
	});*/

  auto f = loco_t::imgui_fs_var_t(gloco->m_fbo_final_shader, "bloom_strength", 0.04, 0.01);

	loco.loop([&] {
    fan::vec2 p = shape.get_position();
    shape.set_position(loco.get_mouse_position());
		loco.get_fps();
	});

	return 0;
}