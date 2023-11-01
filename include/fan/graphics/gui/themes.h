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

static inline ImVec4 ImLerp(const ImVec4& a, const ImVec4& b, float t) { return ImVec4(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t, a.w + (b.w - a.w) * t); }

struct imgui_themes {
  static void dark() {
    // Future Dark style by rewrking from ImThemes
    ImGuiStyle& style = ImGui::GetStyle();

    style.Alpha = 1.0f;
    style.DisabledAlpha = 1.0f;
    style.WindowPadding = ImVec2(12.0f, 12.0f);
    style.WindowRounding = 0.0f;
    style.WindowBorderSize = 0.0f;
    style.WindowMinSize = ImVec2(20.0f, 20.0f);
    style.WindowTitleAlign = ImVec2(0.5f, 0.5f);
    style.WindowMenuButtonPosition = ImGuiDir_None;
    style.ChildRounding = 0.0f;
    style.ChildBorderSize = 1.0f;
    style.PopupRounding = 0.0f;
    style.PopupBorderSize = 1.0f;
    style.FramePadding = ImVec2(6.0f, 6.0f);
    style.FrameRounding = 0.0f;
    style.FrameBorderSize = 0.0f;
    style.ItemSpacing = ImVec2(12.0f, 6.0f);
    style.ItemInnerSpacing = ImVec2(6.0f, 3.0f);
    style.CellPadding = ImVec2(12.0f, 6.0f);
    style.IndentSpacing = 20.0f;
    style.ColumnsMinSpacing = 6.0f;
    style.ScrollbarSize = 12.0f;
    style.ScrollbarRounding = 0.0f;
    style.GrabMinSize = 12.0f;
    style.GrabRounding = 0.0f;
    style.TabRounding = 0.0f;
    style.TabBorderSize = 0.0f;
    style.TabMinWidthForCloseButton = 0.0f;
    style.ColorButtonPosition = ImGuiDir_Right;
    style.ButtonTextAlign = ImVec2(0.5f, 0.5f);
    style.SelectableTextAlign = ImVec2(0.0f, 0.0f);

    style.Colors[ImGuiCol_Text] = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
    style.Colors[ImGuiCol_TextDisabled] = ImVec4(0.2745098173618317f, 0.3176470696926117f, 0.4509803950786591f, 1.0f);
    style.Colors[ImGuiCol_WindowBg] = ImVec4(0.0784313753247261f, 0.08627451211214066f, 0.1019607856869698f, 1.0f);
    style.Colors[ImGuiCol_ChildBg] = ImVec4(0.0784313753247261f, 0.08627451211214066f, 0.1019607856869698f, 1.0f);
    style.Colors[ImGuiCol_PopupBg] = ImVec4(0.0784313753247261f, 0.08627451211214066f, 0.1019607856869698f, 1.0f);
    style.Colors[ImGuiCol_Border] = ImVec4(0.1568627506494522f, 0.168627455830574f, 0.1921568661928177f, 1.0f);
    style.Colors[ImGuiCol_BorderShadow] = ImVec4(0.0784313753247261f, 0.08627451211214066f, 0.1019607856869698f, 1.0f);
    style.Colors[ImGuiCol_FrameBg] = ImVec4(0.1176470592617989f, 0.1333333402872086f, 0.1490196138620377f, 1.0f);
    style.Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.1568627506494522f, 0.168627455830574f, 0.1921568661928177f, 1.0f);
    style.Colors[ImGuiCol_FrameBgActive] = ImVec4(0.2352941185235977f, 0.2156862765550613f, 0.5960784554481506f, 1.0f);
    style.Colors[ImGuiCol_TitleBg] = ImVec4(0.0470588244497776f, 0.05490196123719215f, 0.07058823853731155f, 1.0f);
    style.Colors[ImGuiCol_TitleBgActive] = ImVec4(0.0470588244497776f, 0.05490196123719215f, 0.07058823853731155f, 1.0f);
    style.Colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.0784313753247261f, 0.08627451211214066f, 0.1019607856869698f, 1.0f);
    style.Colors[ImGuiCol_MenuBarBg] = ImVec4(0.15803921729326248f, 0.105882354080677f, 0.4215686276555061f, 1.0f);
    style.Colors[ImGuiCol_ScrollbarBg] = ImVec4(0.0470588244497776f, 0.05490196123719215f, 0.07058823853731155f, 1.0f);
    style.Colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.1176470592617989f, 0.1333333402872086f, 0.1490196138620377f, 1.0f);
    style.Colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.1568627506494522f, 0.168627455830574f, 0.1921568661928177f, 1.0f);
    style.Colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.1176470592617989f, 0.1333333402872086f, 0.1490196138620377f, 1.0f);
    style.Colors[ImGuiCol_CheckMark] = ImVec4(0.4980392158031464f, 0.5137255191802979f, 1.0f, 1.0f);
    style.Colors[ImGuiCol_SliderGrab] = ImVec4(0.4980392158031464f, 0.5137255191802979f, 1.0f, 1.0f);
    style.Colors[ImGuiCol_SliderGrabActive] = ImVec4(0.5372549295425415f, 0.5529412031173706f, 1.0f, 1.0f);
    style.Colors[ImGuiCol_Button] = ImVec4(0.1176470592617989f, 0.1333333402872086f, 0.1490196138620377f, 1.0f);
    style.Colors[ImGuiCol_ButtonHovered] = ImVec4(0.196078434586525f, 0.1764705926179886f, 0.5450980663299561f, 1.0f);
    style.Colors[ImGuiCol_ButtonActive] = ImVec4(0.2352941185235977f, 0.2156862765550613f, 0.5960784554481506f, 1.0f);
    style.Colors[ImGuiCol_Header] = ImVec4(0.1176470592617989f, 0.1333333402872086f, 0.1490196138620377f, 1.0f);
    style.Colors[ImGuiCol_HeaderHovered] = ImVec4(0.196078434586525f, 0.1764705926179886f, 0.5450980663299561f, 1.0f);
    style.Colors[ImGuiCol_HeaderActive] = ImVec4(0.2352941185235977f, 0.2156862765550613f, 0.5960784554481506f, 1.0f);
    style.Colors[ImGuiCol_Separator] = ImVec4(0.1568627506494522f, 0.1843137294054031f, 0.250980406999588f, 1.0f);
    style.Colors[ImGuiCol_SeparatorHovered] = ImVec4(0.1568627506494522f, 0.1843137294054031f, 0.250980406999588f, 1.0f);
    style.Colors[ImGuiCol_SeparatorActive] = ImVec4(0.1568627506494522f, 0.1843137294054031f, 0.250980406999588f, 1.0f);
    style.Colors[ImGuiCol_ResizeGrip] = ImVec4(0.1176470592617989f, 0.1333333402872086f, 0.1490196138620377f, 1.0f);
    style.Colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.196078434586525f, 0.1764705926179886f, 0.5450980663299561f, 1.0f);
    style.Colors[ImGuiCol_ResizeGripActive] = ImVec4(0.2352941185235977f, 0.2156862765550613f, 0.5960784554481506f, 1.0f);
    style.Colors[ImGuiCol_Tab] = ImVec4(0.25803921729326248f, 0.255882354080677f, 0.2515686276555061f, 1.0f);
    style.Colors[ImGuiCol_TabHovered] = ImVec4(0.1176470592617989f, 0.1333333402872086f, 0.1490196138620377f, 1.0f);
    style.Colors[ImGuiCol_TabActive] = ImVec4(0.44803921729326248f, 0.455882354080677f, 0.4715686276555061f, 1.0f);
    style.Colors[ImGuiCol_TabUnfocused] = ImVec4(0.3470588244497776f, 0.35490196123719215f, 0.37058823853731155f, 1.0f);
    style.Colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.30803921729326248f, 0.305882354080677f, 0.3015686276555061f, 1.0f);
    style.Colors[ImGuiCol_PlotLines] = ImVec4(0.5215686559677124f, 0.6000000238418579f, 0.7019608020782471f, 1.0f);
    style.Colors[ImGuiCol_PlotLinesHovered] = ImVec4(0.03921568766236305f, 0.9803921580314636f, 0.9803921580314636f, 1.0f);
    style.Colors[ImGuiCol_PlotHistogram] = ImVec4(1.0f, 0.2901960909366608f, 0.5960784554481506f, 1.0f);
    style.Colors[ImGuiCol_PlotHistogramHovered] = ImVec4(0.9960784316062927f, 0.4745098054409027f, 0.6980392336845398f, 1.0f);
    style.Colors[ImGuiCol_TableHeaderBg] = ImVec4(0.0470588244497776f, 0.05490196123719215f, 0.07058823853731155f, 1.0f);
    style.Colors[ImGuiCol_TableBorderStrong] = ImVec4(0.0470588244497776f, 0.05490196123719215f, 0.07058823853731155f, 1.0f);
    style.Colors[ImGuiCol_TableBorderLight] = ImVec4(0.0f, 0.0f, 0.0f, 1.0f);
    style.Colors[ImGuiCol_TableRowBg] = ImVec4(0.1176470592617989f, 0.1333333402872086f, 0.1490196138620377f, 1.0f);
    style.Colors[ImGuiCol_TableRowBgAlt] = ImVec4(0.09803921729326248f, 0.105882354080677f, 0.1215686276555061f, 1.0f);
    style.Colors[ImGuiCol_TextSelectedBg] = ImVec4(0.2352941185235977f, 0.2156862765550613f, 0.5960784554481506f, 1.0f);
    style.Colors[ImGuiCol_DragDropTarget] = ImVec4(0.4980392158031464f, 0.5137255191802979f, 1.0f, 1.0f);
    style.Colors[ImGuiCol_NavHighlight] = ImVec4(0.4980392158031464f, 0.5137255191802979f, 1.0f, 1.0f);
    style.Colors[ImGuiCol_NavWindowingHighlight] = ImVec4(0.4980392158031464f, 0.5137255191802979f, 1.0f, 1.0f);
    style.Colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.196078434586525f, 0.1764705926179886f, 0.5450980663299561f, 0.501960813999176f);
    style.Colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.196078434586525f, 0.1764705926179886f, 0.5450980663299561f, 0.501960813999176f);
  }
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
    theme_reference = theme.theme_reference;
    gloco->get_context()->theme_list[theme_reference].theme_id = this;
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
      gloco->get_context()->theme_list[theme_reference].theme_id = this;
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