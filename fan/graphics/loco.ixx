module;

#define loco_framebuffer
#define loco_post_process
#define loco_vfi

#include <fan/graphics/opengl/init.h>
#if defined(fan_vulkan)
  #include <vulkan/vulkan.h>
#endif
#include <fan/types/bll_raii.h>
#include <fan/event/types.h>
#include <uv.h>
#undef min
#undef max
// +cuda
#if __has_include("cuda.h")
	//#include "cuda_runtime.h"
	//#include <cuda.h>
	#include <nvcuvid.h>
	//#define loco_cuda
#endif

#include <source_location>
#include <deque>
#include <cstdlib>
#include <sstream>
#include <set>
#include <iostream>
#include <coroutine>
#include <map>
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
  import fan.graphics.gui.base;
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
// include memory. after, it expands
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
// 
//#define debug_shape_t

export struct loco_t;

// to set new loco use gloco = new_loco;
struct global_loco_t {
	loco_t* loco = nullptr;
	operator loco_t* () { return loco; }
	global_loco_t& operator=(loco_t* l) {
		loco = l;
		return *this;
	}
	loco_t* operator->() {
		return loco;
	}
};

export inline thread_local global_loco_t gloco;

struct next_frame_awaiter {
  bool await_ready() const noexcept { return false; }
  void await_suspend(std::coroutine_handle<> handle) {
    pending.push_back(handle);
  }
  void await_resume() const noexcept {}
  static inline std::vector<std::coroutine_handle<>> pending;
};

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

	std::uint32_t get_draw_mode(std::uint8_t internal_draw_mode);

  next_frame_awaiter co_next_frame() {
    return {};
  }
}

export struct loco_t {
  fan::window_t window; // destruct last
// for shaper_get_* functions
private:
	using shader_t = fan::graphics::shader_nr_t;
	using image_t = fan::graphics::image_nr_t;
	using camera_t = fan::graphics::camera_nr_t;
	using viewport_t = fan::graphics::viewport_nr_t;
public:
	using image_load_properties_t = fan::graphics::image_load_properties_t;

	using image_sampler_address_mode = fan::graphics::image_sampler_address_mode;

  std::uint8_t get_renderer();
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

  std::vector<std::uint8_t> image_get_pixel_data(fan::graphics::image_nr_t nr, int image_format, fan::vec2 uvp = 0, fan::vec2 uvs = 1);
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

		std::uint32_t fb_vao;
		std::uint32_t fb_vbo;

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
    int window_open_mode = fan::window_t::mode::windowed;
		std::uint8_t renderer = fan::window_t::renderer_t::opengl;
		std::uint8_t samples = 0;
	};

	fan::time::timer start_time;
  f32_t time = 0;

  void add_shape_to_immediate_draw(fan::graphics::shapes::shape_t&& s);
  auto add_shape_to_static_draw(fan::graphics::shapes::shape_t&& s);
  void remove_static_shape_draw(const fan::graphics::shapes::shape_t& s);

  static void generate_commands(loco_t* loco);
	// -1 no reload, opengl = 0 etc
	std::uint8_t reload_renderer_to = -1;

#if defined(fan_vulkan)
  // todo move to vulkan context
  static void check_vk_result(VkResult err);
#endif

#if defined(fan_gui)
  void init_gui();
  void destroy_gui();
  bool enable_overlay = true;
#endif
  void init_framebuffer();

  loco_t();
  loco_t(const properties_t& p);
  ~loco_t();

  void destroy();
  void close();
  void setup_input_callbacks();

	// to change renderer, pass: fan::window_t::renderer_t::*
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

  void process_render();
  bool should_close();

  bool process_frame(const std::function<void()>& cb = [] {});
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
  using keys_handle_t = fan::window_t::keys_handle_t;
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
  using text_callback_handle_t = fan::window_t::text_callback_handle_t;

  using buttons_cb_t = fan::window_t::buttons_cb_t;

  FORWARD_CB_TO_WINDOW(mouse_click,  buttons_handle_t, buttons_cb_t);
  FORWARD_CB_TO_WINDOW(mouse_down,   mouse_down_handle_t, buttons_cb_t);
  FORWARD_CB_TO_WINDOW(mouse_up,     buttons_handle_t, buttons_cb_t);
  FORWARD_CB_TO_WINDOW(key_click,    key_handle_t, key_cb_t);
  FORWARD_CB_TO_WINDOW(key_down,     key_handle_t, key_cb_t);
  FORWARD_CB_TO_WINDOW(key_up,       key_handle_t, key_cb_t);

  FORWARD_CB_TO_WINDOW_NOARG(mouse_move, mouse_move_handle_t, mouse_move_cb_t);
  FORWARD_CB_TO_WINDOW_NOARG(resize, resize_handle_t, resize_cb_t);

  buttons_handle_t buttons_handle;
  keys_handle_t keys_handle;
  mouse_move_handle_t mouse_move_handle;
  text_callback_handle_t text_callback_handle;

#if defined(fan_physics)
	fan::physics::context_t physics_context{ {} };
  void update_physics();
  fan::physics::physics_update_cbs_t::nr_t add_physics_update(const fan::physics::physics_update_data_t& cb_data);
  fan::physics::physics_update_cbs_t::nd_t& get_physics_update_data(fan::physics::physics_update_cbs_t::nr_t nr);
  void remove_physics_update(fan::physics::physics_update_cbs_t::nr_t nr);
	
	fan::physics::physics_update_cbs_t shape_physics_update_cbs;
#endif

	// clears shapes after drawing, good for debug draw, not best for performance
	std::vector<fan::graphics::shapes::shape_t> immediate_render_list;
	std::unordered_map<std::uint32_t, fan::graphics::shapes::shape_t> static_render_list;

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

	bool gui_initialized = false;

	fan::graphics::gui::text_logger_t text_logger;

#include <fan/graphics/gui/settings_menu.h>
	settings_menu_t settings_menu;
#endif

	fan::graphics::texture_pack_t texture_pack;

	bool render_shapes_top = false;
	//gui

  fan::graphics::image_load_properties_t default_noise_image_properties();
  fan::graphics::image_t create_noise_image(const fan::vec2& size, int seed = fan::random::value_i64(0, ((std::uint32_t)-1) / 2));
  fan::graphics::image_t create_noise_image(const fan::vec2& size, const std::vector<std::uint8_t>& data);
  fan::vec2 convert_mouse_to_ndc(const fan::vec2& mouse_position) const;
  fan::vec2 convert_mouse_to_ndc() const;
  fan::ray3_t convert_mouse_to_ray(const fan::vec3& camera_position, const fan::mat4& projection, const fan::mat4& view);
  fan::ray3_t convert_mouse_to_ray(const fan::mat4& projection, const fan::mat4& view);
#if defined(loco_cuda)
  struct cuda_textures_t {
    void close(loco_t* loco, fan::graphics::shapes::shape_t& cid);
    void resize(loco_t* loco, fan::graphics::shapes::shape_t& id, std::uint8_t format, fan::vec2ui size);
    cudaArray_t& get_array(std::uint32_t index_t);
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
    gloco->shader_set_value<T>(nr, name, val);
  }
}