struct defaults {
  static inline fan::color text_color = fan::color(1);
  static inline fan::color text_color_place_holder = fan::color::hex(0x757575);
  static inline f32_t font_size = f32_t(0.1);
  static constexpr f32_t text_renderer_outline_size = 0.4;
};

enum class text_position_e {
  left,
  middle
};
enum class button_states_e {
  clickable = 1,
  locked = 2
};

struct src_dst_t {
  fan::vec2 src = 0;
  fan::vec2 dst = 0;
};
struct cursor_properties {
  static inline fan::color color = fan::colors::white;
  // nanoseconds
  static inline fan::time::nanoseconds blink_speed = 1e+8;
  static inline f32_t line_thickness = 0.002;
};


struct theme_t {

  struct mouse_move_data_t : loco_t::mouse_move_data_t {
    mouse_move_data_t(const loco_t::mouse_move_data_t& mm) : loco_t::mouse_move_data_t(mm) {

    }

    loco_t::theme_t* theme;
  };
  struct mouse_button_data_t : loco_t::mouse_button_data_t {
    mouse_button_data_t(const loco_t::mouse_button_data_t& mm) : loco_t::mouse_button_data_t(mm) {

    }

    loco_t::theme_t* theme;
  };
  struct keyboard_data_t : loco_t::keyboard_data_t {
    keyboard_data_t(const loco_t::keyboard_data_t& mm) : loco_t::keyboard_data_t(mm) {

    }

    loco_t::theme_t* theme;
  };

  struct text_data_t : loco_t::text_data_t {
    text_data_t(const loco_t::text_data_t& mm) : loco_t::text_data_t(mm) {

    }

    loco_t::theme_t* theme;
  };

  using mouse_move_cb_t = fan::function_t<int(const mouse_move_data_t&)>;
  using mouse_button_cb_t = fan::function_t<int(const mouse_button_data_t&)>;
  using keyboard_cb_t = fan::function_t<int(const keyboard_data_t&)>;
  using text_cb_t = fan::function_t<int(const text_data_t&)>;

  mouse_button_cb_t mouse_button_cb = [](const mouse_button_data_t&) -> int { return 0; };
  mouse_move_cb_t mouse_move_cb = [](const mouse_move_data_t&) -> int { return 0; };
  keyboard_cb_t keyboard_cb = [](const keyboard_data_t&) -> int { return 0; };
  text_cb_t text_cb = [](const text_data_t&) -> int { return 0; };

	#if defined(loco_opengl)
		using context_t = fan::opengl::context_t;
		#define ns fan::opengl
	#elif defined(loco_vulkan)
		using context_t = fan::vulkan::context_t;
		#define ns fan::vulkan
	#endif

  theme_t() {
    open();
  }
  theme_t(const theme_t& theme) {
    button = theme.button;
    open();
  }
  theme_t(theme_t&& theme) {
    this->button = theme.button;
    theme.theme_reference.NRI = -1;
  }
  theme_t& operator=(const theme_t& t) {
    button = t.button;
    open();
    return *this;
  }
  theme_t& operator=(theme_t&& t) {
    button = t.button;
    t.theme_reference.NRI = -1;
    return *this;
  }
  ~theme_t() {
    close();
  }

	void open(){
		theme_reference = gloco->get_context()->theme_list.NewNode();
		gloco->get_context()->theme_list[theme_reference].theme_id = this;
	}

	void close(){
		gloco->get_context()->theme_list.Recycle(theme_reference);
	}

	template <typename T>
	theme_t operator*(T value) const {
		theme_t t;
		t.theme_reference = theme_reference;
		t.button.color = button.color.mult_no_alpha(value);
		t.button.outline_color = button.outline_color.mult_no_alpha(value);
		t.button.text_outline_color = button.text_outline_color.mult_no_alpha(value);
		t.button.text_color = button.text_color;
		t.button.text_outline_size = button.text_outline_size;
		t.button.outline_size = button.outline_size;

		return t;
	}
	template <typename T>
	theme_t operator/(T value) const {
		theme_t t;
		t.theme_reference = theme_reference;
		t.button.color = button.color / value;
		t.button.outline_color = button.outline_color / value;
		t.button.text_outline_color = button.text_outline_color / value;
		t.button.text_color = button.text_color;
		t.button.text_outline_size = button.text_outline_size;
		t.button.outline_size = button.outline_size;

		return t;
	}

	struct button {

		button() = default;

		enum class states_e {
			outside,
			hovered,
			click
		};

		fan::color color;
		fan::color outline_color;
		fan::color text_color;
		fan::color text_outline_color;
		f32_t text_outline_size;

		f32_t outline_size;

	}button;

	ns::theme_list_NodeReference_t theme_reference;
};

using theme_ptr_t = fan::ptr_maker_t<theme_t>;



struct themes {

  struct empty : public theme_t {
    empty() {

      button.color = fan::color(0, 0, 0);
      button.outline_color = fan::color(0, 0, 0);
      button.text_color = defaults::text_color;
      button.text_outline_color = fan::colors::black;
      button.text_outline_size = defaults::text_renderer_outline_size;
      button.outline_size = 0.1;
    }
  };

  struct deep_blue : public theme_t {
    deep_blue(f32_t intensity = 1) {
      button.color = fan::color(0, 0, 0.3) * intensity;
      button.outline_color = fan::color(0, 0, 0.5) * intensity;
      button.text_color = defaults::text_color;
      button.text_outline_color = fan::colors::black * intensity;
      button.text_outline_size = defaults::text_renderer_outline_size;
      button.outline_size = 0.1;
    }
  };

  struct deep_red : public theme_t {

    deep_red(f32_t intensity = 1) {

      button.color = fan::color(0.3, 0, 0) * intensity;
      button.outline_color = fan::color(0.5, 0, 0) * intensity;
      button.text_color = defaults::text_color;
      button.text_outline_color = fan::colors::black;
      button.text_outline_size = defaults::text_renderer_outline_size;
      button.outline_size = 0.1;
    }
  };

  struct white : public theme_t {
    white() {

      button.color = fan::color(0.8, 0.8, 0.8);
      button.outline_color = fan::color(0.9, 0.9, 0.9);
      button.text_color = defaults::text_color;
      button.text_outline_color = fan::colors::black;
      button.text_outline_size = defaults::text_renderer_outline_size;
      button.outline_size = 0.1;
    }
  };

  struct locked : public theme_t {
    locked() {
      button.color = fan::color(0.2, 0.2, 0.2, 0.8);
      button.outline_color = fan::color(0.3, 0.3, 0.3, 0.8);
      button.text_color = defaults::text_color;
      button.text_outline_color = fan::colors::black;
      button.text_outline_size = defaults::text_renderer_outline_size;
      button.outline_size = 0.1;
    }
  };

  struct hidden : public theme_t {
    hidden() {
      button.color = fan::color(0.0, 0.0, 0.0, 0.3);
      button.outline_color = fan::color(0.0, 0.0, 0.0, 0.3);
      button.text_color = defaults::text_color;
      button.text_outline_color = fan::colors::black;
      button.text_outline_size = defaults::text_renderer_outline_size;
      button.outline_size = 0.1;
    }
  };

  struct transparent : public theme_t {
    transparent(f32_t intensity = 0.3) {
      button.color = fan::color(intensity, intensity, intensity, intensity);
      button.outline_color = fan::color(intensity + 0.2, intensity + 0.2, intensity + 0.2, intensity + 0.2);

      button.text_color = fan::colors::white;
      button.text_outline_color = fan::colors::black;
      button.text_outline_size = defaults::text_renderer_outline_size;
      button.outline_size = 0.1;
    }
  };

  struct gray : public theme_t {
    gray(f32_t intensity = 1) {

      button.color = fan::color(0.2, 0.2, 0.2, 1) * intensity;
      button.outline_color = fan::color(0.3, 0.3, 0.3, 1) * intensity;
      button.text_color = defaults::text_color;
      button.text_outline_color = fan::colors::black;
      button.text_outline_size = defaults::text_renderer_outline_size;
      button.outline_size = 0.1;
    }
  };

  struct custom : public theme_t {
    struct properties_t {
      fan::color color;
      fan::color outline_color;
    };

    custom(const properties_t& properties) {
      button.color = properties.color;
      button.outline_color = properties.outline_color;
      button.text_color = defaults::text_color;
      button.text_outline_color = fan::colors::black;
      button.text_outline_size = defaults::text_renderer_outline_size;
      button.outline_size = 2; // px
    }
  };
};