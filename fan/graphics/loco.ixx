module;

#include <fan/graphics/opengl/init.h>

#define loco_framebuffer
#define loco_post_process
#define loco_vfi

#define loco_physics

/*
loco_line
loco_rectangle
loco_sprite
loco_light
loco_circle
loco_responsive_text
*/

#include <fan/types/bll_raii.h>

#define loco_opengl

#ifndef camera_list
	#define __fan_internal_camera_list (*(fan::graphics::camera_list_t*)fan::graphics::get_camera_list((uint8_t*)&gloco->context))
#endif

#ifndef shader_list
	#define __fan_internal_shader_list (*(fan::graphics::shader_list_t*)fan::graphics::get_shader_list((uint8_t*)&gloco->context))
#endif

#ifndef image_list
	#define __fan_internal_image_list (*(fan::graphics::image_list_t*)fan::graphics::get_image_list((uint8_t*)&gloco->context))
#endif

#ifndef viewport_list
	#define __fan_internal_viewport_list (*(fan::graphics::viewport_list_t*)fan::graphics::get_viewport_list((uint8_t*)&gloco->context))
#endif

// shaper

#if defined(fan_compiler_msvc)
	#ifndef fan_std23
		#define fan_std23
	#endif
#endif


#if defined(fan_gui)
	#include <fan/imgui/imgui.h>
	#include <fan/imgui/misc/freetype/imgui_freetype.h>
	#include <fan/imgui/imgui_impl_opengl3.h>
	#if defined(fan_vulkan)
		#include <fan/imgui/imgui_impl_vulkan.h>
	#endif
	#include <fan/imgui/imgui_impl_glfw.h>
	#include <fan/imgui/implot.h>
#endif

#if defined(fan_gui)
#include <fan/imgui/imgui_internal.h>
#include <fan/graphics/gui/imgui_themes.h>
#endif

#include <fan/event/types.h>
#include <uv.h>

// +cuda
#if __has_include("cuda.h")
	//#include "cuda_runtime.h"
	//#include <cuda.h>
	#include <nvcuvid.h>
	//#define loco_cuda
#endif

#undef min
#undef max

#include <source_location>
#include <deque>
#include <cstdlib>
#include <sstream>
#include <set>
#include <iostream>

#if defined(fan_std23)
  #include <stacktrace>
#endif

export module fan.graphics.loco;

import fan.utility;
#if defined(fan_gui)
  import fan.graphics.gui.text_logger;
#endif

export import fan.event;

export import fan.window;
export import fan.types.color;
export import fan.random;
export import fan.texture_pack.tp0;

export import fan.io.file;

#if defined(fan_physics)
	import fan.physics.b2_integration;
  import fan.physics.common_context;
#endif

#if defined(fan_audio)
	export import fan.audio;
#endif

#if defined(fan_gui)
	export import fan.console;
#endif

export import fan.graphics.opengl.core;

#if defined(fan_vulkan)
	export import fan.graphics.vulkan.core;
#endif

export import fan.graphics.shapes;

export import fan.physics.collision.rectangle;

export import fan.noise;

#if defined(fan_json)
	export import fan.types.json;
#endif

#include <fan/memory/memory.h>

#if defined(fan_json)
export namespace fan {
  struct json_stream_parser_t {
    std::string buf;

    struct parsed_result {
      bool success;
      fan::json value;
      std::string error;
    };

    [[nodiscard]]
    std::pair<size_t, size_t> find_next_json_bounds(std::string_view s, size_t pos = 0) const noexcept;

    std::vector<parsed_result> process(std::string_view chunk);

    void clear() noexcept;
  };
}
#endif

#if defined(fan_gui)
namespace fan {
	namespace graphics {
		namespace gui {
			void render_allocations_plot();
			void process_loop();
		}
	}
}
#endif

#if defined(loco_cuda)
export namespace fan {
	namespace cuda {
		void check_error(auto result) {
			if (result != CUDA_SUCCESS) {
				if constexpr (std::is_same_v<decltype(result), CUresult>) {
					const char* err_str = nullptr;
					cuGetErrorString(result, &err_str);
					fan::throw_error("function failed with:" + std::to_string(result) + ", " + err_str);
				}
				else {
					fan::throw_error("function failed with:" + std::to_string(result) + ", ");
				}
			}
		}
	}
}
export extern "C" {
	extern __host__ cudaError_t CUDARTAPI cudaGraphicsGLRegisterImage(struct cudaGraphicsResource** resource, GLuint image, GLenum target, unsigned int flags);
}
#endif

// -cuda

//#define debug_shape_t

export struct loco_t;

// to set new loco use gloco = new_loco;
struct global_loco_t {

	loco_t* loco = nullptr;

	operator loco_t* () {
		return loco;
	}

	global_loco_t& operator=(loco_t* l) {
		loco = l;
		return *this;
	}
	loco_t* operator->() {
		return loco;
	}
};

export inline thread_local global_loco_t gloco;

export namespace fan::graphics {
	struct engine_init_t {
#define BLL_set_SafeNext 1
#define BLL_set_AreWeInsideStruct 1
#define BLL_set_prefix init_callback
#include <fan/fan_bll_preset.h>
#define BLL_set_Link 1
#define BLL_set_type_node uint16_t
#define BLL_set_NodeDataType std::function<void(loco_t*)>
#define BLL_set_CPP_CopyAtPointerChange 1
#include <BLL/BLL.h>

		using init_callback_nr_t = init_callback_NodeReference_t;
	};

	// cbs called every time engine opens
#if !defined(fan_compiler_msvc)
  inline 
#endif
	engine_init_t::init_callback_t engine_init_cbs;

	uint32_t get_draw_mode(uint8_t internal_draw_mode);

  #if defined(fan_gui)
    namespace gui {
      bool render_blank_window(const std::string& name);
    }
  #endif
}

//#include <fan/graphics/vulkan/ssbo.h>
export struct loco_t {

  bool initialize_lists();
  uint8_t get_renderer();
  bool fan__init_list = initialize_lists();

// for shaper_get_* functions
private:
	using shader_t = fan::graphics::shader_nr_t;
	using image_t = fan::graphics::image_nr_t;
	using camera_t = fan::graphics::camera_nr_t;
	using viewport_t = fan::graphics::viewport_nr_t;
public:
	using image_load_properties_t = fan::graphics::image_load_properties_t;

	using image_sampler_address_mode = fan::graphics::image_sampler_address_mode;

  fan::graphics::shader_nr_t shader_create();
  fan::graphics::context_shader_t shader_get(fan::graphics::shader_nr_t nr);
  void shader_erase(fan::graphics::shader_nr_t nr);
  void shader_use(fan::graphics::shader_nr_t nr);
  void shader_set_vertex(fan::graphics::shader_nr_t nr, const std::string& vertex_code);
  void shader_set_fragment(fan::graphics::shader_nr_t nr, const std::string& fragment_code);
  bool shader_compile(fan::graphics::shader_nr_t nr);
	template <typename T>
	void shader_set_value(fan::graphics::shader_nr_t nr, const std::string& name, const T& val) {
		if (window.renderer == fan::window_t::renderer_t::opengl) {
			context.gl.shader_set_value(nr, name, val);
		}
		else if (window.renderer == fan::window_t::renderer_t::vulkan) {
			fan::throw_error("todo");
		}
	}
  void shader_set_camera(shader_t nr, camera_t camera_nr);
  fan::graphics::shader_nr_t shader_get_nr(uint16_t shape_type);
  fan::graphics::shader_list_t::nd_t& shader_get_data(uint16_t shape_type);

	fan::graphics::camera_list_t camera_list;
	fan::graphics::shader_list_t shader_list;
	fan::graphics::image_list_t image_list;
	fan::graphics::viewport_list_t viewport_list;

  std::vector<uint8_t> image_get_pixel_data(fan::graphics::image_nr_t nr, int image_format, fan::vec2 uvp = 0, fan::vec2 uvs = 1);
  fan::graphics::image_nr_t image_create();
  fan::graphics::context_image_t image_get(fan::graphics::image_nr_t nr);
  uint64_t image_get_handle(fan::graphics::image_nr_t nr);
  fan::graphics::image_data_t& image_get_data(fan::graphics::image_nr_t nr);
  void image_erase(fan::graphics::image_nr_t nr);
  void image_bind(fan::graphics::image_nr_t nr);
  void image_unbind(fan::graphics::image_nr_t nr);
  fan::graphics::image_load_properties_t& image_get_settings(fan::graphics::image_nr_t nr);
  void image_set_settings(fan::graphics::image_nr_t nr, const fan::graphics::image_load_properties_t& settings);
  fan::graphics::image_nr_t image_load(const fan::image::info_t& image_info);
  fan::graphics::image_nr_t image_load(const fan::image::info_t& image_info, const fan::graphics::image_load_properties_t& p);
  fan::graphics::image_nr_t image_load(const std::string& path, const std::source_location& callers_path = std::source_location::current());
  fan::graphics::image_nr_t image_load(const std::string& path, const fan::graphics::image_load_properties_t& p, const std::source_location& callers_path = std::source_location::current());
  fan::graphics::image_nr_t image_load(fan::color* colors, const fan::vec2ui& size);
  fan::graphics::image_nr_t image_load(fan::color* colors, const fan::vec2ui& size, const fan::graphics::image_load_properties_t& p);
  void image_unload(fan::graphics::image_nr_t nr);
  bool is_image_valid(fan::graphics::image_nr_t nr);
  fan::graphics::image_nr_t create_missing_texture();
  fan::graphics::image_nr_t create_transparent_texture();
  void image_reload(fan::graphics::image_nr_t nr, const fan::image::info_t& image_info);
  void image_reload(fan::graphics::image_nr_t nr, const fan::image::info_t& image_info, const fan::graphics::image_load_properties_t& p);
  void image_reload(fan::graphics::image_nr_t nr, const std::string& path, const std::source_location& callers_path = std::source_location::current());
  void image_reload(fan::graphics::image_nr_t nr, const std::string& path, const fan::graphics::image_load_properties_t& p, const std::source_location& callers_path = std::source_location::current());
  fan::graphics::image_nr_t image_create(const fan::color& color);
  fan::graphics::image_nr_t image_create(const fan::color& color, const fan::graphics::image_load_properties_t& p);

  fan::graphics::camera_nr_t camera_create();
  fan::graphics::context_camera_t& camera_get(fan::graphics::camera_nr_t nr);
  void camera_erase(fan::graphics::camera_nr_t nr);
  fan::graphics::camera_nr_t camera_create(const fan::vec2& x, const fan::vec2& y);
  fan::vec3 camera_get_position(fan::graphics::camera_nr_t nr);
  void camera_set_position(fan::graphics::camera_nr_t nr, const fan::vec3& cp);
  fan::vec2 camera_get_size(fan::graphics::camera_nr_t nr);
  f32_t camera_get_zoom(fan::graphics::camera_nr_t nr, fan::graphics::viewport_nr_t viewport);
  void camera_set_ortho(fan::graphics::camera_nr_t nr, fan::vec2 x, fan::vec2 y);
  void camera_set_perspective(fan::graphics::camera_nr_t nr, f32_t fov, const fan::vec2& window_size);
  void camera_rotate(fan::graphics::camera_nr_t nr, const fan::vec2& offset);
  void camera_set_target(fan::graphics::camera_nr_t nr, const fan::vec2& target, f32_t move_speed = 10);

  fan::graphics::viewport_nr_t viewport_create();
  fan::graphics::viewport_nr_t viewport_create(const fan::vec2& viewport_position, const fan::vec2& viewport_size);
  fan::graphics::context_viewport_t& viewport_get(fan::graphics::viewport_nr_t nr);
  void viewport_erase(fan::graphics::viewport_nr_t nr);
  fan::vec2 viewport_get_position(fan::graphics::viewport_nr_t nr);
  fan::vec2 viewport_get_size(fan::graphics::viewport_nr_t nr);
  void viewport_set(const fan::vec2& viewport_position, const fan::vec2& viewport_size);
  void viewport_set(fan::graphics::viewport_nr_t nr, const fan::vec2& viewport_position, const fan::vec2& viewport_size);
  void viewport_set_size(fan::graphics::viewport_nr_t nr, const fan::vec2& viewport_size);
  void viewport_set_position(fan::graphics::viewport_nr_t nr, const fan::vec2& viewport_position);
  void viewport_zero(fan::graphics::viewport_nr_t nr);
  bool inside(fan::graphics::viewport_nr_t nr, const fan::vec2& position);
  bool inside_wir(fan::graphics::viewport_nr_t nr, const fan::vec2& position);
  bool inside(const fan::graphics::render_view_t& render_view, const fan::vec2& position) const;
  bool is_mouse_inside(const fan::graphics::render_view_t& render_view) const;

  static std::string read_shader(const std::string& path, const std::source_location& callers_path = std::source_location::current());

  fan::graphics::context_functions_t context_functions;
  fan::graphics::context_t context;

	// unsafe
	loco_t(const loco_t&) = delete;
	loco_t& operator=(const loco_t&) = delete;
	loco_t(loco_t&&) = delete;
	loco_t& operator=(loco_t&&) = delete;

#if defined (fan_gui)
	using console_t = fan::console_t;
#endif

  void use();
  void camera_move(fan::graphics::context_camera_t& camera, f64_t dt, f32_t movement_speed, f32_t friction = 12);

#if defined(fan_opengl)
	// opengl namespace
	struct opengl {
#include <fan/graphics/opengl/engine_functions.h>
#include <fan/graphics/opengl/2D/effects/blur.h>

		blur_t blur;

    fan::window_t::resize_handle_t window_resize_handle;

		fan::opengl::core::framebuffer_t m_framebuffer;
		fan::opengl::core::renderbuffer_t m_rbo;
		fan::graphics::image_t color_buffers[4];
		fan::graphics::shader_t m_fbo_final_shader;

		GLenum blend_src_factor = GL_SRC_ALPHA;
		GLenum blend_dst_factor = GL_ONE_MINUS_SRC_ALPHA;

		uint32_t fb_vao;
		uint32_t fb_vbo;

#undef loco
	}gl;
#endif

#if defined(fan_vulkan)
	struct vulkan {
#include <fan/graphics/vulkan/engine_functions.h>

		fan::vulkan::context_t::descriptor_t d_attachments;
		fan::vulkan::context_t::pipeline_t post_process;
		VkResult image_error = VK_SUCCESS;
	}vk;
#endif

	template <typename T, typename T2>
	static T2& get_render_data(fan::graphics::shapes::shape_t* shape, T2 T::* attribute) {
		fan::graphics::shaper_t::ShapeRenderData_t* data = shape->GetRenderData(fan::graphics::g_shapes->shaper);
		return ((T*)data)->*attribute;
	}

	template <typename T, typename T2, typename T3, typename T4>
	static void modify_render_data_element_arr(fan::graphics::shapes::shape_t* shape, T2 T::* attribute, std::size_t i, auto T4::* arr_member, const T3& value) {
		fan::graphics::shaper_t::ShapeRenderData_t* data = shape->GetRenderData(fan::graphics::g_shapes->shaper);

		// remove gloco
		if (gloco->window.renderer == fan::window_t::renderer_t::opengl) {
			gloco->gl.modify_render_data_element_arr(shape, data, attribute, i, arr_member, value);
		}
#if defined(fan_vulkan)
		else if (gloco->window.renderer == fan::window_t::renderer_t::vulkan) {
			(((T*)data)->*attribute)[i].*arr_member = value;
			auto& data = fan::graphics::g_shapes->shaper.ShapeList[*shape];
			fan::graphics::g_shapes->shaper.ElementIsPartiallyEdited(
				data.sti,
				data.blid,
				data.ElementIndex,
				fan::member_offset(attribute),
				sizeof(T3)
			);
		}
#endif
	}

	template <typename T, typename T2, typename T3>
	static void modify_render_data_element(fan::graphics::shapes::shape_t* shape, T2 T::* attribute, const T3& value) {
		fan::graphics::shaper_t::ShapeRenderData_t* data = shape->GetRenderData(fan::graphics::g_shapes->shaper);

		// remove gloco
		if (gloco->window.renderer == fan::window_t::renderer_t::opengl) {
			gloco->gl.modify_render_data_element(shape, data, attribute, value);
		}
#if defined(fan_vulkan)
		else if (gloco->window.renderer == fan::window_t::renderer_t::vulkan) {
			((T*)data)->*attribute = value;
			auto& data = fan::graphics::g_shapes->shaper.ShapeList[*shape];
			fan::graphics::g_shapes->shaper.ElementIsPartiallyEdited(
				data.sti,
				data.blid,
				data.ElementIndex,
				fan::member_offset(attribute),
				sizeof(T3)
			);
		}
#endif
	}

public:

	std::vector<std::function<void()>> m_pre_draw;
	std::vector<std::function<void()>> m_post_draw;


	struct properties_t {
		bool render_shapes_top = false;
		bool vsync = true;
		fan::vec2 window_size = -1;
		uint64_t window_flags = 0;
		uint8_t renderer = fan::window_t::renderer_t::opengl;
		uint8_t samples = 0;
	};

	fan::time::timer start_time;
  f32_t time = 0;

  void add_shape_to_immediate_draw(fan::graphics::shapes::shape_t&& s);
  auto add_shape_to_static_draw(fan::graphics::shapes::shape_t&& s);
  void remove_static_shape_draw(const fan::graphics::shapes::shape_t& s);

  static void generate_commands(loco_t* loco);
	// -1 no reload, opengl = 0 etc
	uint8_t reload_renderer_to = -1;

#if defined(fan_vulkan)
  // todo move to vulkan context
  static void check_vk_result(VkResult err);
#endif

#if defined(fan_gui)
  void load_fonts(ImFont* (&fonts)[std::size(fan::graphics::gui::font_sizes)], const std::string& name, ImFontConfig* cfg = nullptr);
  void build_fonts();
  ImFont* get_font(f32_t font_size, bool bold = false);

  void init_imgui();

  void init_fonts();
  void load_emoticons();
  void destroy_imgui();
  bool enable_overlay = true;
#endif
  void init_framebuffer();

  loco_t();
  loco_t(const properties_t& p);
  ~loco_t();

  void destroy();
  void close();
  void setup_input_callbacks();


	// for renderer switch
	// input fan::window_t::renderer_t::
  void switch_renderer(std::uint8_t renderer);
  void draw_shapes();
  void process_shapes();
  void process_gui();

  struct time_monitor_t {
    static constexpr int buffer_size = 128;

    void update(f32_t value);
    void reset();

    struct stats_t {
      f32_t average;
      f32_t lowest;
      f32_t highest;
    };

    stats_t calculate_stats(f32_t last_value) const;

  #if defined(fan_gui)
    void plot(const char* label);
  #endif

    bool paused = false;
    int insert_index = 0;
    int valid_samples = 0;
    f32_t running_sum = 0.0f;
    f32_t running_min = std::numeric_limits<f32_t>::max();
    f32_t running_max = std::numeric_limits<f32_t>::min();
    fan::time::timer refresh_speed{ (uint64_t)0.05e9, true };
    std::array<f32_t, buffer_size> samples{};
  };
  time_monitor_t frame_monitor;
  time_monitor_t shape_monitor;
  time_monitor_t gui_monitor;

	std::vector<std::function<void()>> draw_end_cb;

  void process_frame();
  bool should_close();

  bool process_loop(const std::function<void()>& cb = [] {});
  void loop(const std::function<void()>& cb = [] {});
  camera_t open_camera(const fan::vec2& x, const fan::vec2& y);
  camera_t open_camera_perspective(f32_t fov = 90.0f);
  fan::graphics::viewport_t open_viewport(const fan::vec2& viewport_position, const fan::vec2& viewport_size);
  void set_viewport(fan::graphics::viewport_t viewport, const fan::vec2& viewport_position, const fan::vec2& viewport_size);
  fan::vec2 get_input_vector(
    const std::string& forward = "move_forward",
    const std::string& back = "move_back",
    const std::string& left = "move_left",
    const std::string& right = "move_right"
  );
  fan::vec2 transform_matrix(const fan::vec2& position);
  fan::vec2 screen_to_ndc(const fan::vec2& screen_pos);
  fan::vec2 ndc_to_screen(const fan::vec2& ndc_position);
  void set_vsync(bool flag);
  void start_timer();
  static void idle_cb(uv_idle_t* handle);
  void start_idle(bool start_idle = true);
  void update_timer_interval(bool idle = true);
  void set_target_fps(int32_t new_target_fps, bool idle = true);
  fan::graphics::context_t& get_context();
  fan::graphics::render_view_t render_view_create();
  fan::graphics::render_view_t render_view_create(
    const fan::vec2& ortho_x, const fan::vec2& ortho_y,
    const fan::vec2& viewport_position, const fan::vec2& viewport_size
  );

	fan::window::input_action_t input_action;

	fan::graphics::update_callback_t m_update_callback;

	std::vector<std::function<void()>> single_queue;

	#include "engine_images.h"

	fan::graphics::render_view_t orthographic_render_view;
	fan::graphics::render_view_t perspective_render_view;

	fan::window_t window;

  fan::graphics::shapes shapes;

  void set_window_name(const std::string& name);
  void set_window_icon(const fan::image::info_t& info);
  void set_window_icon(const fan::graphics::image_t& image);

	bool idle_init = false;
	uv_idle_t idle_handle;
	bool timer_init = false;
	uv_timer_t timer_handle{};

	int32_t target_fps = 165; // must be changed from function
	bool timer_enabled = target_fps > 0;
	bool vsync = false;

	std::function<void()> main_loop; // bad, but forced

	f64_t delta_time = window.m_delta_time;
  fan::time::timer shape_draw_timer;
  fan::time::timer gui_draw_timer;
  f64_t shape_draw_time_s = 0;
  f64_t gui_draw_time_s = 0;

#if defined(fan_gui)
	fan::graphics::gui_draw_cb_t gui_draw_cb;
#endif

  #define FORWARD_CB_TO_WINDOW(NAME, HANDLE, CBDATA_NAME) \
    HANDLE on_##NAME(int arg, CBDATA_NAME cb) {        \
      return window.on_##NAME(arg, std::move(cb));          \
    }

  #define FORWARD_CB_TO_WINDOW_NOARG(NAME, HANDLE, CBDATA_NAME) \
    using CBDATA_NAME = fan::window_t::CBDATA_NAME; \
    using NAME##_data_t = fan::window_t::NAME##_data_t; \
    HANDLE on_##NAME(CBDATA_NAME cb) { \
      return window.on_##NAME(std::move(cb)); \
    }

  using key_cb_t = fan::window_t::key_cb_t;
  using key_handle_t = fan::window_t::key_handle_t;
  using buttons_handle_t = fan::window_t::buttons_handle_t;
  using mouse_down_handle_t = fan::window_t::mouse_down_handle_t;
  using resize_handle_t = fan::window_t::resize_handle_t;
  using mouse_move_handle_t = fan::window_t::mouse_move_handle_t;

  using buttons_data_t = fan::window_t::buttons_data_t;
  using button_data_t = fan::window_t::button_data_t;
  using mouse_down_data_t = fan::window_t::mouse_down_data_t;
  using mouse_up_data_t = fan::window_t::mouse_up_data_t;
  using mouse_click_data_t = fan::window_t::mouse_click_data_t;
  using keys_data_t = fan::window_t::keys_data_t;
  using key_data_t = fan::window_t::key_data_t;
  using key_down_data_t = fan::window_t::key_down_data_t;
  using key_up_data_t = fan::window_t::key_up_data_t;
  using key_click_data_t = fan::window_t::key_click_data_t;

  using buttons_cb_t = fan::window_t::buttons_cb_t;

  FORWARD_CB_TO_WINDOW(mouse_click,  buttons_handle_t, buttons_cb_t);
  FORWARD_CB_TO_WINDOW(mouse_down,   mouse_down_handle_t, buttons_cb_t);
  FORWARD_CB_TO_WINDOW(mouse_up,     buttons_handle_t, buttons_cb_t);
  FORWARD_CB_TO_WINDOW(key_click,    key_handle_t, key_cb_t);
  FORWARD_CB_TO_WINDOW(key_down,     key_handle_t, key_cb_t);
  FORWARD_CB_TO_WINDOW(key_up,       key_handle_t, key_cb_t);

  FORWARD_CB_TO_WINDOW_NOARG(mouse_move, mouse_move_handle_t, mouse_move_cb_t);
  FORWARD_CB_TO_WINDOW_NOARG(resize, resize_handle_t, resize_cb_t);

	using push_back_cb = fan::graphics::shapes::shape_t(*)(void*);
	using set_position2_cb = void (*)(fan::graphics::shapes::shape_t*, const fan::vec2&);
	// depth
	using set_position3_cb = void (*)(fan::graphics::shapes::shape_t*, const fan::vec3&);
	using set_size_cb = void (*)(fan::graphics::shapes::shape_t*, const fan::vec2&);
	using set_size3_cb = void (*)(fan::graphics::shapes::shape_t*, const fan::vec3&);

	using get_position_cb = fan::vec3(*)(const fan::graphics::shapes::shape_t*);
	using get_size_cb = fan::vec2(*)(const fan::graphics::shapes::shape_t*);
	using get_size3_cb = fan::vec3(*)(const fan::graphics::shapes::shape_t*);

	using set_rotation_point_cb = void (*)(fan::graphics::shapes::shape_t*, const fan::vec2&);
	using get_rotation_point_cb = fan::vec2(*)(const fan::graphics::shapes::shape_t*);

	using set_color_cb = void (*)(fan::graphics::shapes::shape_t*, const fan::color&);
	using get_color_cb = fan::color(*)(const fan::graphics::shapes::shape_t*);

	using set_angle_cb = void (*)(fan::graphics::shapes::shape_t*, const fan::vec3&);
	using get_angle_cb = fan::vec3(*)(const fan::graphics::shapes::shape_t*);

	using get_tc_position_cb = fan::vec2(*)(fan::graphics::shapes::shape_t*);
	using set_tc_position_cb = void (*)(fan::graphics::shapes::shape_t*, const fan::vec2&);

	using get_tc_size_cb = fan::vec2(*)(fan::graphics::shapes::shape_t*);
	using set_tc_size_cb = void (*)(fan::graphics::shapes::shape_t*, const fan::vec2&);

	using load_tp_cb = bool(*)(fan::graphics::shapes::shape_t*, fan::graphics::texture_pack::ti_t*);

	using get_grid_size_cb = fan::vec2(*)(fan::graphics::shapes::shape_t*);
	using set_grid_size_cb = void (*)(fan::graphics::shapes::shape_t*, const fan::vec2&);

	using get_camera_cb = loco_t::camera_t(*)(const fan::graphics::shapes::shape_t*);
	using set_camera_cb = void (*)(fan::graphics::shapes::shape_t*, loco_t::camera_t);

	using get_viewport_cb = fan::graphics::viewport_t(*)(const fan::graphics::shapes::shape_t*);
	using set_viewport_cb = void (*)(fan::graphics::shapes::shape_t*, fan::graphics::viewport_t);


	using get_image_cb = fan::graphics::image_t(*)(fan::graphics::shapes::shape_t*);
	using set_image_cb = void (*)(fan::graphics::shapes::shape_t*, fan::graphics::image_t);

	using get_image_data_cb = fan::graphics::image_data_t& (*)(fan::graphics::shapes::shape_t*);

	using get_parallax_factor_cb = f32_t(*)(fan::graphics::shapes::shape_t*);
	using set_parallax_factor_cb = void (*)(fan::graphics::shapes::shape_t*, f32_t);
	using get_flags_cb = uint32_t(*)(fan::graphics::shapes::shape_t*);
	using set_flags_cb = void(*)(fan::graphics::shapes::shape_t*, uint32_t);
	//
	using get_radius_cb = f32_t(*)(fan::graphics::shapes::shape_t*);
	using get_src_cb = fan::vec3(*)(fan::graphics::shapes::shape_t*);
	using get_dst_cb = fan::vec3(*)(fan::graphics::shapes::shape_t*);
	using get_outline_size_cb = f32_t(*)(fan::graphics::shapes::shape_t*);
	using get_outline_color_cb = fan::color(*)(const fan::graphics::shapes::shape_t*);
	using set_outline_color_cb = void(*)(fan::graphics::shapes::shape_t*, const fan::color&);

	using reload_cb = void (*)(fan::graphics::shapes::shape_t*, uint8_t format, void** image_data, const fan::vec2& image_size, uint32_t filter);

	using draw_cb = void (*)(uint8_t draw_range);

	using set_line_cb = void (*)(fan::graphics::shapes::shape_t*, const fan::vec3&, const fan::vec3&);
	using set_line3_cb = void (*)(fan::graphics::shapes::shape_t*, const fan::vec3&, const fan::vec3&);

	struct functions_t {

		template<typename T>
		struct function_traits;

		template<typename R, typename... Args>
		struct function_traits<R(*)(Args...)> {
			using return_type = R;
			using args_tuple = std::tuple<Args...>;
			using function_type = R(*)(Args...);

			static constexpr function_type default_cb() {
				return [](Args... args) -> R {
					fan::print("default cb called, function did not exist");
					if constexpr (!std::is_void_v<R>) {
						if constexpr (std::is_reference_v<R>) {
							static std::remove_reference_t<R> dummy_object{};
							return dummy_object;
						}
						else {
							return R{};
						}
					}
					};
			}
		};

		template<typename CallbackType>
		static CallbackType make_dummy() {
			return function_traits<CallbackType>::default_cb();
		}

		template<typename CallbackType>
		void init_callback(CallbackType& callback) {
			callback = make_dummy<CallbackType>();
		}

		functions_t() {
			init_callback(push_back);
			init_callback(get_position);
			init_callback(set_position2);
			init_callback(set_position3);
			init_callback(get_size);
			init_callback(get_size3);
			init_callback(set_size);
			init_callback(set_size3);
			init_callback(get_rotation_point);
			init_callback(set_rotation_point);
			init_callback(get_color);
			init_callback(set_color);
			init_callback(get_angle);
			init_callback(set_angle);
			init_callback(get_tc_position);
			init_callback(set_tc_position);
			init_callback(get_tc_size);
			init_callback(set_tc_size);
			init_callback(load_tp);
			init_callback(get_grid_size);
			init_callback(set_grid_size);
			init_callback(get_camera);
			init_callback(set_camera);
			init_callback(get_viewport);
			init_callback(set_viewport);
			init_callback(get_image);
			init_callback(set_image);
			init_callback(get_image_data);
			init_callback(get_parallax_factor);
			init_callback(set_parallax_factor);
			init_callback(get_flags);
			init_callback(set_flags);
			init_callback(get_radius);
			init_callback(get_src);
			init_callback(get_dst);
			init_callback(get_outline_size);
			init_callback(get_outline_color);
			init_callback(set_outline_color);
			init_callback(reload);
			init_callback(draw);
			init_callback(set_line);
			init_callback(set_line3);
		}

		push_back_cb push_back;

		get_position_cb get_position;
		set_position2_cb set_position2;
		set_position3_cb set_position3;

		get_size_cb get_size;
		get_size3_cb get_size3;
		set_size_cb set_size;
		set_size3_cb set_size3;

		get_rotation_point_cb get_rotation_point;
		set_rotation_point_cb set_rotation_point;

		get_color_cb get_color;
		set_color_cb set_color;

		get_angle_cb get_angle;
		set_angle_cb set_angle;

		get_tc_position_cb get_tc_position;
		set_tc_position_cb set_tc_position;

		get_tc_size_cb get_tc_size;
		set_tc_size_cb set_tc_size;

		load_tp_cb load_tp;

		get_grid_size_cb get_grid_size;
		set_grid_size_cb set_grid_size;

		get_camera_cb get_camera;
		set_camera_cb set_camera;

		get_viewport_cb get_viewport;
		set_viewport_cb set_viewport;

		get_image_cb get_image;
		set_image_cb set_image;

		get_image_data_cb get_image_data;

		get_parallax_factor_cb get_parallax_factor;
		set_parallax_factor_cb set_parallax_factor;


		get_flags_cb get_flags;
		set_flags_cb set_flags;

		get_radius_cb get_radius;
		get_src_cb get_src;
		get_dst_cb get_dst;
		get_outline_size_cb get_outline_size;
		get_outline_color_cb get_outline_color;
		set_outline_color_cb set_outline_color;

		reload_cb reload;

		draw_cb draw;

		set_line_cb set_line;
		set_line3_cb set_line3;
	};

#if defined(fan_physics)
	fan::physics::context_t physics_context{ {} };
  void update_physics();
  fan::physics::physics_update_cbs_t::nr_t add_physics_update(const fan::physics::physics_update_data_t& cb_data);
  void remove_physics_update(fan::physics::physics_update_cbs_t::nr_t nr);
	
	fan::physics::physics_update_cbs_t shape_physics_update_cbs;
#endif

	// clears shapes after drawing, good for debug draw, not best for performance
	std::vector<fan::graphics::shapes::shape_t> immediate_render_list;
	std::unordered_map<uint32_t, fan::graphics::shapes::shape_t> static_render_list;

  fan::vec2 get_mouse_position(const camera_t& camera, const viewport_t& viewport) const;
  fan::vec2 get_mouse_position(const fan::graphics::render_view_t& render_view) const;
  fan::vec2 get_mouse_position() const;
  fan::vec2 translate_position(const fan::vec2& p, viewport_t viewport, camera_t camera) const;
  fan::vec2 translate_position(const fan::vec2& p) const;

  bool is_mouse_clicked(int button = fan::mouse_left);
  bool is_mouse_down(int button = fan::mouse_left);
  bool is_mouse_released(int button = fan::mouse_left);
  fan::vec2 get_mouse_drag(int button = fan::mouse_left);
  bool is_key_pressed(int key);
  bool is_key_down(int key);
  bool is_key_released(int key);

	// ShapeID_t must be at the beginning of fan::graphics::shapes::shape_t's memory since there are reinterpret_casts,
	// which assume that


	// pointer
	using shape_shader_locations_t = decltype(fan::graphics::shaper_t::BlockProperties_t::gl_t::locations);

  void shape_open(
    uint16_t shape_type,
    std::size_t sizeof_vi,
    std::size_t sizeof_ri,
    shape_shader_locations_t shape_shader_locations,
    const std::string& vertex,
    const std::string& fragment,
    fan::graphics::shaper_t::ShapeRenderDataSize_t instance_count = 1,
    bool instanced = true
  );

  fan::graphics::shader_t get_sprite_vertex_shader(const std::string& fragment);

	//#if defined(loco_texture_pack)
	//#endif

	fan::color clear_color = {
		/*0.10f, 0.10f, 0.131f, 1.f */
		0.f, 0.f, 0.f, 1.f
	};

	fan::graphics::lighting_t lighting;

	//gui
#if defined(fan_gui)
  void toggle_console();
  void toggle_console(bool active);

	fan::console_t console;
	bool render_console = false;
	bool show_fps = false;
	bool render_settings_menu = 0;

	bool allow_docking = true;

	bool imgui_initialized = false;
	static inline bool global_imgui_initialized = false;

	fan::graphics::gui::text_logger_t text_logger;

#include <fan/graphics/gui/settings_menu.h>
	settings_menu_t settings_menu;
#endif

	fan::graphics::texture_pack_t texture_pack;

	bool render_shapes_top = false;
	//gui

  fan::graphics::image_load_properties_t default_noise_image_properties();
  fan::graphics::image_t create_noise_image(const fan::vec2& size, int seed = fan::random::value_i64(0, ((uint32_t)-1) / 2));
  fan::graphics::image_t create_noise_image(const fan::vec2& size, const std::vector<uint8_t>& data);
  fan::vec2 convert_mouse_to_ndc(const fan::vec2& mouse_position) const;
  fan::vec2 convert_mouse_to_ndc() const;
  fan::ray3_t convert_mouse_to_ray(const fan::vec3& camera_position, const fan::mat4& projection, const fan::mat4& view);
  fan::ray3_t convert_mouse_to_ray(const fan::mat4& projection, const fan::mat4& view);
#if defined(loco_cuda)
  struct cuda_textures_t {
    void close(loco_t* loco, fan::graphics::shapes::shape_t& cid);
    void resize(loco_t* loco, fan::graphics::shapes::shape_t& id, uint8_t format, fan::vec2ui size);
    cudaArray_t& get_array(uint32_t index_t);
    bool inited = false;
    struct graphics_resource_t {
      void open(int texture_id);
      void close();
      void map();
      void unmap();
      cudaGraphicsResource_t resource = nullptr;
      cudaArray_t cuda_array = nullptr;
    };
    graphics_resource_t wresources[4];
  };
#endif

#if defined(fan_audio)
	fan::system_audio_t system_audio;
	fan::audio_t audio;
#endif
  void camera_move_to(const fan::graphics::shapes::shape_t& shape, const fan::graphics::render_view_t& render_view);
  void camera_move_to(const fan::graphics::shapes::shape_t& shape);
  void camera_move_to_smooth(const fan::graphics::shapes::shape_t& shape, const fan::graphics::render_view_t& render_view);
  void camera_move_to_smooth(const fan::graphics::shapes::shape_t& shape);
  bool shader_update_fragment(uint16_t shape_type, const std::string& fragment);
};

//vk

#if defined(fan_vulkan)
#include <fan/graphics/vulkan/uniform_block.h>
#include <fan/graphics/vulkan/memory.h>
#endif

#if defined(fan_gui)
namespace fan {
	namespace graphics {
		using texture_packe0 = fan::graphics::texture_pack::internal_t;
		using ti_t = fan::graphics::texture_pack::ti_t;
	}
}
#endif

export namespace fan::graphics {
  using engine_t = loco_t;
  void shader_set_camera(fan::graphics::shader_t nr, fan::graphics::camera_t camera_nr);

  template <typename T>
  void shader_set_value(fan::graphics::shader_nr_t nr, const std::string& name, const T& val) {
  #if defined(fan_opengl)
    if (fan::graphics::get_window().renderer == fan::window_t::renderer_t::opengl) {
      get_gl_context().shader_set_value(nr, name, val);
    }
    else 
#endif
    if (fan::graphics::get_window().renderer == fan::window_t::renderer_t::vulkan) {
      fan::throw_error("todo");
    }
  }
}