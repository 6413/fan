#define use_key_lambda(key, state) \
		if (mb.mouse_stage != loco_t::vfi_t::mouse_stage_e::inside) { \
			return 0; \
		} \
		if (mb.button != key) { \
			return 0; \
					} \
		if (mb.button_state != state) { \
			return 0; \
		}

struct fgm_t {

	#if defined(fgm_build_stage_maker)
		stage_maker_t* get_stage_maker() {
			return OFFSETLESS(this, stage_maker_t, fgm);
		}
	#endif

	struct viewport_area {
		static constexpr uint32_t global = 0;
		static constexpr uint32_t editor = 1;
		static constexpr uint32_t sidepanel = 2;
	};

	struct action {
		static constexpr uint32_t move = 1 << 0;
		static constexpr uint32_t resize = 1 << 1;
	};

  #include "common.h"

	static constexpr fan::vec2 button_size = fan::vec2(0.3, 0.08);
  static constexpr f32_t line_z_depth = 50;
  static constexpr f32_t right_click_z_depth = 11;

	void load() {
		resize_cb();
	}

	void open(const char* texturepack_name) {

		//editor_ratio = fan::vec2(1, 1);
		//move_offset = 0;
		//action_flag = 0;
		theme = loco_t::themes::deep_red();

		texturepack.open_compiled(gloco, texturepack_name);

    static constexpr fan::color image[3 * 3] = {
      fan::color(0, 1, 0, 0.3),
      fan::color(0, 1, 0, 0.3),
      fan::color(0, 1, 0, 0.3),
      fan::color(0, 1, 0, 0.3),
      fan::color(0, 0, 0, 0.3),
      fan::color(0, 1, 0, 0.3),
      fan::color(0, 1, 0, 0.3),
      fan::color(0, 1, 0, 0.3),
      fan::color(0, 1, 0, 0.3),
    };

    static constexpr fan::color mark_image_pixels[2] = {
      fan::color(1, 1, 1, 1),
      fan::color(1, 1, 1, 1)
    };

    hitbox_image.load((fan::color*)image, 3);
    mark_image.load((fan::color*)mark_image_pixels, 1);

		gloco->get_window()->add_resize_callback([this](const fan::window_t::resize_cb_data_t& d) {
			resize_cb();
		});
		//pile->loco.get_window()->add_mouse_move_callback([this](const auto& d) {
  //    if (view_action_flag & action::move) {
  //      fan::vec2 size = fan::vec2(
  //        camera[viewport_area::editor].coordinates.right,
  //        camera[viewport_area::editor].coordinates.down
  //      );
  //      *(fan::vec2*)& camera_position -= 
  //        ((gloco->get_mouse_position() - gloco->get_window()->get_previous_mouse_position()) * 
  //        (size / gloco->get_window()->get_size())) * 32;
  //      camera[viewport_area::editor].set_camera_position(camera_position);
  //    }
  //  });
		//gloco->get_window()->add_buttons_callback([this](const auto& d) {
  //    switch (d.button) {
  //      case fan::mouse_middle: {
  //        if (d.state == fan::mouse_state::press) {
  //          view_action_flag |= action::move;
  //        }
  //        else {
  //          view_action_flag &= ~action::move;
  //        }
  //        break;
  //      }
  //    }
  //  });

		gloco->get_window()->add_keys_callback([this](const auto& d) {
      switch (d.key) {
        case fan::key_f: {
          if (d.state != fan::keyboard_state::press) {
            return;
          }
          camera_position = fan::vec3(0, 0, 0);
          camera[viewport_area::editor].set_camera_position(camera_position);
          break;
        }
      }
    });

		//// half size
    sidepanel_line_position = fan::vec2(0.5, 0);
		editor_position = fan::vec2(-sidepanel_line_position.x / 2, 0);
		editor_size = editor_position.x + 0.9;

    for (auto& i : camera) {
      camera->open();
    }
    for (auto& i : viewport) {
      viewport->open();
    }

    editor_viewport = fan::vec4(-1, 1, -1, 1);

    loco_t::line_t::properties_t lp;
		lp.viewport = &viewport[viewport_area::global];
		lp.camera = &camera[viewport_area::global];
		lp.color = fan::colors::white;

    for (auto& i : lines) {
      i = lp;
    }

    resize_cb();
	}
	void close() {
		clear();
	}
	void clear() {

	}

	fan::string get_fgm_full_path(const fan::string& stage_name) {
		#if defined(fgm_build_stage_maker)
		return fan::string(stage_maker_t::stage_runtime_folder_name) + "/" + stage_name + ".fgm";
		#else
		fan::throw_error("");
		return "";
		#endif
	}
	#if defined(fgm_build_stage_maker)
  void read_from_file(const fan::string& stage_name) {
    fan::string path = get_fgm_full_path(stage_name);
    fan::string f;
    if (!fan::io::file::exists(path)) {
      return;
    }
    fan::io::file::read(path, &f);

    if (f.empty()) {
      return;
    }
    uint64_t off = 0;

    uint32_t file_version = fan::read_data<uint32_t>(f, off);

    switch (file_version) {
      // no need to put other versions than current because it would not compile
      case stage_maker_format_version: {
        while (off < f.size()) {
          iterate_masterpiece([&f, &off](auto& o) {
            off += o.from_string(f.substr(off));
          });
        }
        break;
      }
      default: {
        fan::throw_error("invalid version fgm version number", file_version);
        break;
      }
    }
  }
	void write_to_file(const fan::string& stage_name) {

		fan::string f;
    // header
    fan::write_to_string(f, stage_maker_format_version);

    iterate_masterpiece([&f](auto& shape) {
      f += shape.to_string();
    });
   
		fan::io::file::write(
			get_fgm_full_path(stage_name),
			f,
			std::ios_base::binary
		);

    auto offset = get_stage_maker()->stage_h_str.find(stage_name);

    if (offset == fan::string::npos) {
      fan::throw_error("corrupted stage.h");
    }
	}
	#endif

  fan::vec2 translate_viewport_position(const fan::vec2& value) {
		fan::vec2 window_size = gloco->get_window()->get_size();
		return (value + 1) / 2 * window_size;
	}
  fan::vec2 translate_to_global(const fan::vec2& position) const {
		return position / viewport[viewport_area::global].get_size() * 2 - 1;
	}

  void set_viewport_and_camera(uint32_t area, const fan::vec2& position, const fan::vec2& size) {
    fan::vec2 window_size = gloco->get_window()->get_size();
    fan::vec2 viewport_position = translate_viewport_position(position);
    fan::vec2 viewport_size = translate_viewport_position(size);
    fan::vec2 ratio = viewport_size / viewport_size.max();
    viewport[area].set(viewport_position, viewport_size, window_size);
    camera[area].set_ortho(fan::vec2(-1, 1) * ratio.x, fan::vec2(-1, 1) * ratio.y);
  }

  void create_lines() {
    fan::vec3 src(editor_position - editor_size, line_z_depth);
    fan::vec3 dst(editor_position.x + editor_size.x, src.y, line_z_depth);

    lines[0].set_line(src, dst);

    src = dst;
    dst.y = editor_position.y + editor_size.y;
    lines[1].set_line(src, dst);

    src = dst;
    dst.x = editor_position.x - editor_size.x;
    lines[2].set_line(src, dst);

    src = dst;
    dst.y = editor_position.y - editor_size.y;
    lines[3].set_line(src, dst);

    src = fan::vec3(translate_to_global(viewport[viewport_area::sidepanel].get_position()), line_z_depth);
    dst.x = src.x;
    dst.y = camera[viewport_area::global].coordinates.down;
    lines[4].set_line(src, dst);
  }

  void resize_cb() {
    set_viewport_and_camera(viewport_area::global, fan::vec2(-1), fan::vec2(1));
    set_viewport_and_camera(viewport_area::editor, 
      editor_position - editor_size, editor_size + fan::vec2(-sidepanel_line_position.x / 2 - 0.1)
    );
    set_viewport_and_camera(viewport_area::sidepanel, 
      fan::vec2(sidepanel_line_position.x, -1), 
      fan::vec2(1, 0) - translate_viewport_position(fan::vec2(sidepanel_line_position.x, -1))
    );

    create_lines();
  }

  #include "private.h"
	#include "shapes.h"

  std::array<loco_t::shape_t, 5> lines;

	loco_t::camera_t camera[3];
	fan::graphics::viewport_t viewport[3];

	loco_t::theme_t theme;

  fan::vec3 camera_position = 0;

  fan::vec2 editor_position;
	fan::vec2 editor_size;
	fan::vec2 editor_ratio;

  fan::vec2 sidepanel_line_position;
	f32_t line_y_offset_between_types_and_properties;

	loco_t::texturepack_t texturepack;

  loco_t::image_t hitbox_image;
  loco_t::image_t mark_image;

  fan::vec4 editor_viewport;
};

#undef use_key_lambda

#undef fgm_button
#undef fgm_sprite
#undef fgm_text
#undef fgm_hitbox
#undef fgm_mark