struct defaults {
  static inline fan::color text_color = fan::color(1);
  static inline fan::color text_color_place_holder = fan::color::hex(0x757575);
  static inline f32_t font_size = f32_t(0.1);
  static constexpr f32_t text_renderer_outline_size = 1;
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

#if defined(loco_imgui)

static inline ImVec4 ImLerp(const ImVec4& a, const ImVec4& b, float t) { return ImVec4(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t, a.w + (b.w - a.w) * t); }

struct imgui_themes {
  static void dark() {
    ImGuiStyle& style = ImGui::GetStyle();

    style.Alpha = 1.0f;
    style.DisabledAlpha = 0.6000000238418579f;
    style.WindowPadding = ImVec2(8.0f, 8.0f);
    style.WindowRounding = 0.0f;
    style.WindowBorderSize = 1.0f;
    style.WindowMinSize = ImVec2(32.0f, 32.0f);
    style.WindowTitleAlign = ImVec2(0.0f, 0.5f);
    style.WindowMenuButtonPosition = ImGuiDir_Left;
    style.ChildRounding = 0.0f;
    style.ChildBorderSize = 1.0f;
    style.PopupRounding = 0.0f;
    style.PopupBorderSize = 1.0f;
    style.FramePadding = ImVec2(4.0f, 3.0f);
    style.FrameRounding = 0.0f;
    style.FrameBorderSize = 0.0f;
    style.ItemSpacing = ImVec2(8.0f, 4.0f);
    style.ItemInnerSpacing = ImVec2(4.0f, 4.0f);
    style.CellPadding = ImVec2(4.0f, 2.0f);
    style.IndentSpacing = 21.0f;
    style.ColumnsMinSpacing = 6.0f;
    style.ScrollbarSize = 14.0f;
    style.ScrollbarRounding = 9.0f;
    style.GrabMinSize = 10.0f;
    style.GrabRounding = 0.0f;
    style.TabRounding = 4.0f;
    style.TabBorderSize = 0.0f;
    style.TabMinWidthForCloseButton = 0.0f;
    style.ColorButtonPosition = ImGuiDir_Right;
    style.ButtonTextAlign = ImVec2(0.5f, 0.5f);
    style.SelectableTextAlign = ImVec2(0.0f, 0.0f);

    style.Colors[ImGuiCol_Text] = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
    style.Colors[ImGuiCol_TextDisabled] = ImVec4(0.4980392158031464f, 0.4980392158031464f, 0.4980392158031464f, 1.0f);
    style.Colors[ImGuiCol_WindowBg] = ImVec4(0.15882352963089943f, 0.15882352963089943f, 0.15882352963089943f, 0.9399999976158142f);
    style.Colors[ImGuiCol_ChildBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
    style.Colors[ImGuiCol_PopupBg] = ImVec4(0.0784313753247261f, 0.0784313753247261f, 0.0784313753247261f, 0.9399999976158142f);
    style.Colors[ImGuiCol_Border] = ImVec4(0.4274509847164154f, 0.4274509847164154f, 0.4980392158031464f, 0.5f);
    style.Colors[ImGuiCol_BorderShadow] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
    style.Colors[ImGuiCol_FrameBg] = ImVec4(0.4392156898975372f, 0.4392156898975372f, 0.4392156898975372f, 0.6000000238418579f);
    style.Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.5686274766921997f, 0.5686274766921997f, 0.5686274766921997f, 0.699999988079071f);
    style.Colors[ImGuiCol_FrameBgActive] = ImVec4(0.7568627595901489f, 0.7568627595901489f, 0.7568627595901489f, 0.800000011920929f);
    style.Colors[ImGuiCol_TitleBg] = ImVec4(0.03921568766236305f, 0.03921568766236305f, 0.03921568766236305f, 1.0f);
    style.Colors[ImGuiCol_TitleBgActive] = ImVec4(0.1568627506494522f, 0.1568627506494522f, 0.1568627506494522f, 1.0f);
    style.Colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.0f, 0.0f, 0.0f, 0.6000000238418579f);
    style.Colors[ImGuiCol_MenuBarBg] = ImVec4(0.1372549086809158f, 0.1372549086809158f, 0.1372549086809158f, 1.0f);
    style.Colors[ImGuiCol_ScrollbarBg] = ImVec4(0.01960784383118153f, 0.01960784383118153f, 0.01960784383118153f, 0.5299999713897705f);
    style.Colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.3098039329051971f, 0.3098039329051971f, 0.3098039329051971f, 1.0f);
    style.Colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.407843142747879f, 0.407843142747879f, 0.407843142747879f, 1.0f);
    style.Colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.5098039507865906f, 0.5098039507865906f, 0.5098039507865906f, 1.0f);
    style.Colors[ImGuiCol_CheckMark] = ImVec4(0.1294117718935013f, 0.7490196228027344f, 0.5490196347236633f, 0.800000011920929f);
    style.Colors[ImGuiCol_SliderGrab] = ImVec4(0.1294117718935013f, 0.7490196228027344f, 0.7490196228027344f, 0.800000011920929f);
    style.Colors[ImGuiCol_SliderGrabActive] = ImVec4(0.1294117718935013f, 0.7490196228027344f, 1.0f, 0.800000011920929f);
    style.Colors[ImGuiCol_Button] = ImVec4(0.1294117718935013f, 0.7490196228027344f, 0.5490196347236633f, 0.4000000059604645f);
    style.Colors[ImGuiCol_ButtonHovered] = ImVec4(0.1294117718935013f, 0.7490196228027344f, 0.7490196228027344f, 0.6000000238418579f);
    style.Colors[ImGuiCol_ButtonActive] = ImVec4(0.1294117718935013f, 0.7490196228027344f, 1.0f, 0.800000011920929f);
    style.Colors[ImGuiCol_Header] = ImVec4(0.1294117718935013f, 0.7490196228027344f, 0.5490196347236633f, 0.4000000059604645f);
    style.Colors[ImGuiCol_HeaderHovered] = ImVec4(0.1294117718935013f, 0.7490196228027344f, 0.7490196228027344f, 0.6000000238418579f);
    style.Colors[ImGuiCol_HeaderActive] = ImVec4(0.1294117718935013f, 0.7490196228027344f, 1.0f, 0.800000011920929f);
    style.Colors[ImGuiCol_Separator] = ImVec4(0.1294117718935013f, 0.7490196228027344f, 0.5490196347236633f, 0.4000000059604645f);
    style.Colors[ImGuiCol_SeparatorHovered] = ImVec4(0.1294117718935013f, 0.7490196228027344f, 0.7490196228027344f, 0.6000000238418579f);
    style.Colors[ImGuiCol_SeparatorActive] = ImVec4(0.1294117718935013f, 0.7490196228027344f, 1.0f, 0.800000011920929f);
    style.Colors[ImGuiCol_ResizeGrip] = ImVec4(0.1294117718935013f, 0.7490196228027344f, 0.5490196347236633f, 0.4000000059604645f);
    style.Colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.1294117718935013f, 0.7490196228027344f, 0.7490196228027344f, 0.6000000238418579f);
    style.Colors[ImGuiCol_ResizeGripActive] = ImVec4(0.1294117718935013f, 0.7490196228027344f, 1.0f, 0.800000011920929f);
    style.Colors[ImGuiCol_Tab] = ImVec4(0.1294117718935013f, 0.7490196228027344f, 0.5490196347236633f, 0.800000011920929f);
    style.Colors[ImGuiCol_TabHovered] = ImVec4(0.1294117718935013f, 0.7490196228027344f, 0.7490196228027344f, 0.800000011920929f);
    style.Colors[ImGuiCol_TabActive] = ImVec4(0.1294117718935013f, 0.7490196228027344f, 1.0f, 0.800000011920929f);
    style.Colors[ImGuiCol_TabUnfocused] = ImVec4(0.1764705926179886f, 0.1764705926179886f, 0.1764705926179886f, 1.0f);
    style.Colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.3568627536296844f, 0.3568627536296844f, 0.3568627536296844f, 0.5400000214576721f);
    style.Colors[ImGuiCol_PlotLines] = ImVec4(0.6078431606292725f, 0.6078431606292725f, 0.6078431606292725f, 1.0f);
    style.Colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.0f, 0.4274509847164154f, 0.3490196168422699f, 1.0f);
    style.Colors[ImGuiCol_PlotHistogram] = ImVec4(0.8980392217636108f, 0.6980392336845398f, 0.0f, 1.0f);
    style.Colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.0f, 0.6000000238418579f, 0.0f, 1.0f);
    style.Colors[ImGuiCol_TableHeaderBg] = ImVec4(0.1882352977991104f, 0.1882352977991104f, 0.2000000029802322f, 1.0f);
    style.Colors[ImGuiCol_TableBorderStrong] = ImVec4(0.3098039329051971f, 0.3098039329051971f, 0.3490196168422699f, 1.0f);
    style.Colors[ImGuiCol_TableBorderLight] = ImVec4(0.2274509817361832f, 0.2274509817361832f, 0.2470588237047195f, 1.0f);
    style.Colors[ImGuiCol_TableRowBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
    style.Colors[ImGuiCol_TableRowBgAlt] = ImVec4(1.0f, 1.0f, 1.0f, 0.07000000029802322f);
    style.Colors[ImGuiCol_TextSelectedBg] = ImVec4(0.2588235437870026f, 0.5882353186607361f, 0.9764705896377563f, 0.3499999940395355f);
    style.Colors[ImGuiCol_DragDropTarget] = ImVec4(1.0f, 1.0f, 0.0f, 0.8999999761581421f);
    style.Colors[ImGuiCol_NavHighlight] = ImVec4(0.2588235437870026f, 0.5882353186607361f, 0.9764705896377563f, 1.0f);
    style.Colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.0f, 1.0f, 1.0f, 0.699999988079071f);
    style.Colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.800000011920929f, 0.800000011920929f, 0.800000011920929f, 0.2000000029802322f);
    style.Colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.800000011920929f, 0.800000011920929f, 0.800000011920929f, 0.3499999940395355f);
  }
};

#endif

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

  theme_t() {
    open();
  }
  theme_t(const theme_t& theme) {
    button = theme.button;
    open();
  }
  theme_t(theme_t&& theme) {
    this->button = theme.button;
    theme_reference = theme.theme_reference;
    gloco->get_context().theme_list[theme_reference].theme_id = this;
    theme.theme_reference.sic();
  }
  theme_t& operator=(const theme_t& t) {
    if (this != &t) {
      button = t.button;
      open();
    }
    return *this;
  }
  theme_t& operator=(theme_t&& t) {
    if (this != &t) {
      if (!theme_reference.iic()) {
        close();
      }
      theme_reference = t.theme_reference;
      button = t.button;
      gloco->get_context().theme_list[theme_reference].theme_id = this;
      t.theme_reference.sic();
    }
    return *this;
  }
  ~theme_t() {
    if (theme_reference.iic()) {
      return;
    }
    close();
  }

	void open(){
		theme_reference = gloco->get_context().theme_list.NewNode();
		gloco->get_context().theme_list[theme_reference].theme_id = this;
	}

	void close(){
		gloco->get_context().theme_list.Recycle(theme_reference);
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

	fan::opengl::theme_list_NodeReference_t theme_reference;
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
      button.outline_size = 1;
    }
  };

  struct deep_blue : public theme_t {
    deep_blue(f32_t intensity = 1) {
      button.color = fan::color(0, 0, 0.3) * intensity;
      button.outline_color = fan::color(0, 0, 0.5) * intensity;
      button.text_color = defaults::text_color;
      button.text_outline_color = fan::colors::black * intensity;
      button.text_outline_size = defaults::text_renderer_outline_size;
      button.outline_size = 1;
    }
  };

  struct deep_red : public theme_t {

    deep_red(f32_t intensity = 1) {

      button.color = fan::color(0.3, 0, 0) * intensity;
      button.outline_color = fan::color(0.5, 0, 0) * intensity;
      button.text_color = defaults::text_color;
      button.text_outline_color = fan::colors::black;
      button.text_outline_size = defaults::text_renderer_outline_size;
      button.outline_size = 1;
    }
  };

  struct white : public theme_t {
    white() {

      button.color = fan::color(0.8, 0.8, 0.8);
      button.outline_color = fan::color(0.9, 0.9, 0.9);
      button.text_color = defaults::text_color;
      button.text_outline_color = fan::colors::black;
      button.text_outline_size = defaults::text_renderer_outline_size;
      button.outline_size = 1;
    }
  };

  struct locked : public theme_t {
    locked() {
      button.color = fan::color(0.2, 0.2, 0.2, 0.8);
      button.outline_color = fan::color(0.3, 0.3, 0.3, 0.8);
      button.text_color = defaults::text_color;
      button.text_outline_color = fan::colors::black;
      button.text_outline_size = defaults::text_renderer_outline_size;
      button.outline_size = 1;
    }
  };

  struct hidden : public theme_t {
    hidden() {
      button.color = fan::color(0.0, 0.0, 0.0, 0.3);
      button.outline_color = fan::color(0.0, 0.0, 0.0, 0.3);
      button.text_color = defaults::text_color;
      button.text_outline_color = fan::colors::black;
      button.text_outline_size = defaults::text_renderer_outline_size;
      button.outline_size = 1;
    }
  };

  struct transparent : public theme_t {
    transparent(f32_t intensity = 0.3) {
      button.color = fan::color(intensity, intensity, intensity, intensity);
      button.outline_color = fan::color(intensity + 0.2, intensity + 0.2, intensity + 0.2, intensity + 0.2);

      button.text_color = fan::colors::white;
      button.text_outline_color = fan::colors::black;
      button.text_outline_size = defaults::text_renderer_outline_size;
      button.outline_size = 1;
    }
  };

  struct gray : public theme_t {
    gray(f32_t intensity = 1) {

      button.color = fan::color(0.2, 0.2, 0.2, 1) * intensity;
      button.outline_color = fan::color(0.3, 0.3, 0.3, 1) * intensity;
      button.text_color = defaults::text_color;
      button.text_outline_color = fan::colors::black;
      button.text_outline_size = defaults::text_renderer_outline_size;
      button.outline_size = 1;
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
      button.outline_size = 1; // px
    }
  };
};