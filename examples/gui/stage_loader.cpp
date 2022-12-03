#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#define fan_debug 1
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#define loco_opengl

#define loco_window
#define loco_context

#define loco_button
#include _FAN_PATH(graphics/loco.h)

struct pile_t {

	#define stage_loader_path ../../
	#include _FAN_PATH(graphics/gui/stage_maker/loader.h)

	stage_loader_t stage_loader;
	loco_t loco;

	static constexpr fan::vec2 ortho_x = fan::vec2(-1, 1);
	static constexpr fan::vec2 ortho_y = fan::vec2(-1, 1);

	void open() {
		fan::vec2 window_size = loco.get_window()->get_size();
		loco.open_matrices(
			&matrices,
			ortho_x,
			ortho_y
		);
		loco.get_window()->add_resize_callback([&](const fan::window_t::resize_cb_data_t& d) {
			fan::vec2 window_size = d.size;
			fan::vec2 ratio = window_size / window_size.max();
			std::swap(ratio.x, ratio.y);
			matrices.set_ortho(
				&loco,
				ortho_x * ratio.x,
				ortho_y * ratio.y
			);
			});
		viewport.open(loco.get_context());
		viewport.set(loco.get_context(), 0, window_size, window_size);
		theme = fan_2d::graphics::gui::themes::deep_red();
		theme.open(loco.get_context());
		stage_loader.open();
	}

	void close() {
		stage_loader.close();
	}

	fan_2d::graphics::gui::theme_t theme;
	loco_t::matrices_t matrices;
	fan::graphics::viewport_t viewport;
};

int main() {
	pile_t pile;
	pile.open();

	using sl = pile_t::stage_loader_t;

	sl::stage_common_t::open_properties_t op;
	op.matrices = &pile.matrices;
	op.viewport = &pile.viewport;
	op.theme = &pile.theme;
	pile.stage_loader.push_and_open_stage<sl::stage::stage0_t>(op);

	pile.loco.loop([&] {

		//stage0ptr->update();

	});
	
}