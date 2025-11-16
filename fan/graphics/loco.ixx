module;

#include <fan/graphics/opengl/init.h>

#define loco_framebuffer
#define loco_post_process
#define loco_vfi

#define loco_physics

#if defined(fan_gui)
	#include <deque>
#endif

#include <cstring>
#include <array>
#include <source_location>
#include <cmath>
#include <memory>
#include <functional>

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

export module fan.graphics.loco;

#include <fan/memory/memory.h>
#ifndef __generic_malloc
#define __generic_malloc(n) malloc(n)
#endif

#ifndef __generic_realloc
#define __generic_realloc(ptr, n) realloc(ptr, n)
#endif

#ifndef __generic_free
#define __generic_free(ptr) free(ptr)
#endif

import fan.utility;
import fan.graphics.gui.text_logger;
export import fan.fmt;

export import fan.event;
export import fan.file_dialog;

export import fan.window;
export import fan.window.input_action;
export import fan.types.color;
export import fan.random;
export import fan.texture_pack.tp0;

export import fan.io.file;
export import fan.types.fstring;

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

import fan.graphics.webp;

export import fan.graphics.opengl.core;


#if defined(fan_vulkan)
	export import fan.graphics.vulkan.core;
#endif

export import fan.graphics.shapes;

export import fan.physics.collision.rectangle;

export import fan.noise;

#if defined(fan_json)

export import fan.types.json;

export namespace fan {
	struct json_stream_parser_t {
		std::string buf;

		struct parsed_result {
			bool success;
			fan::json value;
			std::string error;
		};

		[[nodiscard]]
		std::pair<size_t, size_t> find_next_json_bounds(std::string_view s, size_t pos = 0) const noexcept {
			pos = s.find('{', pos);
			if (pos == std::string::npos) return { pos, pos };

			int depth = 0;
			bool in_str = false;

			for (size_t i = pos; i < s.length(); i++) {
				char c = s[i];
				if (c == '"' && (i == 0 || s[i - 1] != '\\')) in_str = !in_str;
				else if (!in_str) {
					if (c == '{') depth++;
					else if (c == '}' && --depth == 0) return { pos, i + 1 };
				}
			}
			return { pos, std::string::npos };
		}

		std::vector<parsed_result> process(std::string_view chunk) {
			std::vector<parsed_result> results;
			buf += chunk;
			size_t pos = 0;

			while (pos < buf.length()) {
				auto [start, end] = find_next_json_bounds(buf, pos);
				if (start == std::string::npos) break;
				if (end == std::string::npos) {
					buf = buf.substr(start);
					break;
				}

				try {
					results.push_back({ true, fan::json::parse(buf.data() + start, buf.data() + end - start), "" });
				}
				catch (const fan::json::parse_error& e) {
					results.push_back({ false, fan::json{}, e.what() });
				}

				pos = buf.find('{', end);
				if (pos == std::string::npos) pos = end;
			}

			buf = pos < buf.length() ? buf.substr(pos) : "";
			return results;
		}

		void clear() noexcept { buf.clear(); }
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

#if defined(fan_physics)
namespace fan {
	namespace graphics {
		void open_bcol();
		void close_bcol();
	}
}
#endif

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

export namespace fan {
	namespace graphics {

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
	 inline engine_init_t::init_callback_t engine_init_cbs;

		inline uint32_t get_draw_mode(uint8_t internal_draw_mode);

#if defined(fan_gui)
		namespace gui {
			bool render_blank_window(const std::string& name) {
				ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
				ImGui::SetNextWindowPos(ImVec2(0, 0));
				return ImGui::Begin(name.c_str(), 0,
					ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
					ImGuiWindowFlags_NoResize |
					ImGuiWindowFlags_NoTitleBar
				);
			}
		}
#endif
	}
}

//#include <fan/graphics/vulkan/ssbo.h>
export struct loco_t {

	bool initialize_lists() {
		fan::graphics::get_camera_list = [](uint8_t* context) -> uint8_t* {
			auto ptr = OFFSETLESS(context, loco_t, context);
			return (uint8_t*)&ptr->camera_list;
		};
		fan::graphics::get_shader_list = [](uint8_t* context) -> uint8_t* {
			return (uint8_t*)&OFFSETLESS(context, loco_t, context)->shader_list;
		};
		fan::graphics::get_image_list = [](uint8_t* context) -> uint8_t* {
			return (uint8_t*)&OFFSETLESS(context, loco_t, context)->image_list;
		};
		fan::graphics::get_viewport_list = [](uint8_t* context) -> uint8_t* {
			return (uint8_t*)&OFFSETLESS(context, loco_t, context)->viewport_list;
		};
		return 0;
	}

	bool fan__init_list = initialize_lists();

	uint8_t get_renderer() {
		return window.renderer;
	}

// for shaper_get_* functions
private:
	using shader_t = fan::graphics::shader_nr_t;
	using image_t = fan::graphics::image_nr_t;
	using camera_t = fan::graphics::camera_nr_t;
	using viewport_t = fan::graphics::viewport_nr_t;
public:
	using image_load_properties_t = fan::graphics::image_load_properties_t;

	using image_sampler_address_mode = fan::graphics::image_sampler_address_mode;

	fan::graphics::shader_nr_t shader_create() {
		return context_functions.shader_create(&context);
	}
	// warning does deep copy, addresses can die
	fan::graphics::context_shader_t shader_get(fan::graphics::shader_nr_t nr) {
		fan::graphics::context_shader_t context_shader;
		if (window.renderer == fan::window_t::renderer_t::opengl) {
			context_shader.gl = *(fan::opengl::context_t::shader_t*)context_functions.shader_get(&context, nr);
		}
#if defined(fan_vulkan)
		else if (window.renderer == fan::window_t::renderer_t::vulkan) {
			context_shader.vk = *(fan::vulkan::context_t::shader_t*)context_functions.shader_get(&context, nr);
		}
#endif
		return context_shader;
	}

	void shader_erase(fan::graphics::shader_nr_t nr) {
		context_functions.shader_erase(&context, nr);
	}

	void shader_use(fan::graphics::shader_nr_t nr) {
		context_functions.shader_use(&context, nr);
	}

	void shader_set_vertex(fan::graphics::shader_nr_t nr, const std::string& vertex_code) {
		context_functions.shader_set_vertex(&context, nr, vertex_code);
	}

	void shader_set_fragment(fan::graphics::shader_nr_t nr, const std::string& fragment_code) {
		context_functions.shader_set_fragment(&context, nr, fragment_code);
	}

	bool shader_compile(fan::graphics::shader_nr_t nr) {
		return context_functions.shader_compile(&context, nr);
	}

	template <typename T>
	void shader_set_value(fan::graphics::shader_nr_t nr, const std::string& name, const T& val) {
		if (window.renderer == fan::window_t::renderer_t::opengl) {
			context.gl.shader_set_value(nr, name, val);
		}
		else if (window.renderer == fan::window_t::renderer_t::vulkan) {
			fan::throw_error("todo");
		}
	}
	void shader_set_camera(shader_t nr, camera_t camera_nr) {
		if (window.renderer == fan::window_t::renderer_t::opengl) {
			context.gl.shader_set_camera(nr, camera_nr);
		}
#if defined(fan_vulkan)
		else if (window.renderer == fan::window_t::renderer_t::vulkan) {
			fan::throw_error("todo");
		}
#endif
	}

	fan::graphics::shader_nr_t shader_get_nr(uint16_t shape_type) {
		return fan::graphics::g_shapes->shaper.GetShader(shape_type);
	}
	auto& shader_get_data(uint16_t shape_type) {
		return shader_list[shader_get_nr(shape_type)];
	}

	fan::graphics::camera_list_t camera_list;
	fan::graphics::shader_list_t shader_list;
	fan::graphics::image_list_t image_list;
	fan::graphics::viewport_list_t viewport_list;

	std::vector<uint8_t> image_get_pixel_data(fan::graphics::image_nr_t nr, int image_format, fan::vec2 uvp = 0, fan::vec2 uvs = 1) {
		if (window.renderer == fan::window_t::renderer_t::opengl) {
			return context_functions.image_get_pixel_data(&context, nr, fan::opengl::context_t::global_to_opengl_format(image_format), uvp, uvs);
		}
		else {
			fan::throw_error("");
			return {};
		}
	}

	fan::graphics::image_nr_t image_create() {
		return context_functions.image_create(&context);
	}

	fan::graphics::context_image_t image_get(fan::graphics::image_nr_t nr) {
		fan::graphics::context_image_t img;
		if (window.renderer == fan::window_t::renderer_t::opengl) {
			img.gl = *(fan::opengl::context_t::image_t*)context_functions.image_get(&context, nr);
		}
#if defined(fan_vulkan)
		else if (window.renderer == fan::window_t::renderer_t::vulkan) {
			img.vk = *(fan::vulkan::context_t::image_t*)context_functions.image_get(&context, nr);
		}
#endif
		return img;
	}

	uint64_t image_get_handle(fan::graphics::image_nr_t nr) {
		return context_functions.image_get_handle(&context, nr);
	}

	fan::graphics::image_data_t& image_get_data(fan::graphics::image_nr_t nr) {
		return image_list[nr];
	}

	void image_erase(fan::graphics::image_nr_t nr) {
		context_functions.image_erase(&context, nr);
	}

	void image_bind(fan::graphics::image_nr_t nr) {
		context_functions.image_bind(&context, nr);
	}

	void image_unbind(fan::graphics::image_nr_t nr) {
		context_functions.image_unbind(&context, nr);
	}

	fan::graphics::image_load_properties_t& image_get_settings(fan::graphics::image_nr_t nr) {
		return context_functions.image_get_settings(&context, nr);
	}

	void image_set_settings(fan::graphics::image_nr_t nr, const fan::graphics::image_load_properties_t& settings) {
		context_functions.image_set_settings(&context, nr, settings);
	}

	fan::graphics::image_nr_t image_load(const fan::image::info_t& image_info) {
		return context_functions.image_load_info(&context, image_info);
	}

	fan::graphics::image_nr_t image_load(const fan::image::info_t& image_info, const fan::graphics::image_load_properties_t& p) {
		return context_functions.image_load_info_props(&context, image_info, p);
	}

	fan::graphics::image_nr_t image_load(const std::string& path, const std::source_location& callers_path = std::source_location::current()) {
		return context_functions.image_load_path(&context, path, callers_path);
	}

	fan::graphics::image_nr_t image_load(const std::string& path, const fan::graphics::image_load_properties_t& p, const std::source_location& callers_path = std::source_location::current()) {
		return context_functions.image_load_path_props(&context, path, p, callers_path);
	}

	fan::graphics::image_nr_t image_load(fan::color* colors, const fan::vec2ui& size) {
		return context_functions.image_load_colors(&context, colors, size);
	}

	fan::graphics::image_nr_t image_load(fan::color* colors, const fan::vec2ui& size, const fan::graphics::image_load_properties_t& p) {
		return context_functions.image_load_colors_props(&context, colors, size, p);
	}

	void image_unload(fan::graphics::image_nr_t nr) {
		context_functions.image_unload(&context, nr);
	}

	bool is_image_valid(fan::graphics::image_nr_t nr) {
		return nr != default_texture && nr.iic() == false;
	}

	fan::graphics::image_nr_t create_missing_texture() {
		return context_functions.create_missing_texture(&context);
	}

	fan::graphics::image_nr_t create_transparent_texture() {
		return context_functions.create_transparent_texture(&context);
	}

	void image_reload(fan::graphics::image_nr_t nr, const fan::image::info_t& image_info) {
		context_functions.image_reload_image_info(&context, nr, image_info);
	}
	void image_reload(fan::graphics::image_nr_t nr, const fan::image::info_t& image_info, const fan::graphics::image_load_properties_t& p) {
		context_functions.image_reload_image_info_props(&context, nr, image_info, p);
	}
	void image_reload(fan::graphics::image_nr_t nr, const std::string& path, const std::source_location& callers_path = std::source_location::current()) {
		context_functions.image_reload_path(&context, nr, path, callers_path);
	}
	void image_reload(fan::graphics::image_nr_t nr, const std::string& path, const fan::graphics::image_load_properties_t& p, const std::source_location& callers_path = std::source_location::current()) {
		context_functions.image_reload_path_props(&context, nr, path, p, callers_path);
	}

	fan::graphics::image_nr_t image_create(const fan::color& color) {
		return context_functions.image_create_color(&context, color);
	}

	fan::graphics::image_nr_t image_create(const fan::color& color, const fan::graphics::image_load_properties_t& p) {
		return context_functions.image_create_color_props(&context, color, p);
	}

	fan::graphics::camera_nr_t camera_create() {
		return context_functions.camera_create(&context);
	}

	fan::graphics::context_camera_t& camera_get(fan::graphics::camera_nr_t nr) {
		return context_functions.camera_get(&context, nr);
	}

	void camera_erase(fan::graphics::camera_nr_t nr) {
		context_functions.camera_erase(&context, nr);
	}

	fan::graphics::camera_nr_t camera_create(const fan::vec2& x, const fan::vec2& y) {
		return context_functions.camera_create_params(&context, x, y);
	}

	fan::vec3 camera_get_position(fan::graphics::camera_nr_t nr) {
		return context_functions.camera_get_position(&context, nr);
	}

	void camera_set_position(fan::graphics::camera_nr_t nr, const fan::vec3& cp) {
		context_functions.camera_set_position(&context, nr, cp);
	}

	fan::vec2 camera_get_size(fan::graphics::camera_nr_t nr) {
		return context_functions.camera_get_size(&context, nr);
	}

	// estimate for -s to s coordinate system
	f32_t camera_get_zoom(fan::graphics::camera_nr_t nr, fan::graphics::viewport_nr_t viewport) {
		fan::vec2 s = viewport_get_size(viewport);

		auto& camera = camera_get(nr);

		return (s.x * 2) / (camera.coordinates.right - camera.coordinates.left);
	}

	void camera_set_ortho(fan::graphics::camera_nr_t nr, fan::vec2 x, fan::vec2 y) {
		context_functions.camera_set_ortho(&context, nr, x, y);
	}

	void camera_set_perspective(fan::graphics::camera_nr_t nr, f32_t fov, const fan::vec2& window_size) {
		context_functions.camera_set_perspective(&context, nr, fov, window_size);
	}

	void camera_rotate(fan::graphics::camera_nr_t nr, const fan::vec2& offset) {
		context_functions.camera_rotate(&context, nr, offset);
	}

	void camera_set_target(fan::graphics::camera_nr_t nr, const fan::vec2& target, f32_t move_speed = 10) {
		f32_t screen_height = window.get_size()[1];
		f32_t pixels_from_bottom = 400.0f;

		/* target - (screen_height / 2 - pixels_from_bottom) / (ic.zoom * 1.5))*/;

		fan::vec2 src = camera_get_position(orthographic_render_view.camera);
		camera_set_position(
			orthographic_render_view.camera,
			move_speed == 0 ? target : src + (target - src) * delta_time * move_speed
		);
	}

	fan::graphics::viewport_nr_t viewport_create() {
		return context_functions.viewport_create(&context);
	}
	fan::graphics::viewport_nr_t viewport_create(const fan::vec2& viewport_position, const fan::vec2& viewport_size) {
		return context_functions.viewport_create_params(&context, viewport_position, viewport_size, window.get_size());
	}
	fan::graphics::context_viewport_t& viewport_get(fan::graphics::viewport_nr_t nr) {
		return context_functions.viewport_get(&context, nr);
	}
	void viewport_erase(fan::graphics::viewport_nr_t nr) {
		context_functions.viewport_erase(&context, nr);
	}
	fan::vec2 viewport_get_position(fan::graphics::viewport_nr_t nr) {
		return context_functions.viewport_get_position(&context, nr);
	}
	fan::vec2 viewport_get_size(fan::graphics::viewport_nr_t nr) {
		return context_functions.viewport_get_size(&context, nr);
	}
	void viewport_set(const fan::vec2& viewport_position, const fan::vec2& viewport_size) {
		context_functions.viewport_set(&context, viewport_position, viewport_size, window.get_size());
	}
	void viewport_set(fan::graphics::viewport_nr_t nr, const fan::vec2& viewport_position, const fan::vec2& viewport_size) {
		context_functions.viewport_set_nr(&context, nr, viewport_position, viewport_size, window.get_size());
	}
  void viewport_set_size(fan::graphics::viewport_nr_t nr, const fan::vec2& viewport_size) {
    fan::vec2 position = viewport_get_position(nr);
    context_functions.viewport_set_nr(&context, nr, position, viewport_size, window.get_size());
  }
  void viewport_set_position(fan::graphics::viewport_nr_t nr, const fan::vec2& viewport_position) {
    fan::vec2 size = viewport_get_size(nr);
    context_functions.viewport_set_nr(&context, nr, viewport_position, size, window.get_size());
  }
	void viewport_zero(fan::graphics::viewport_nr_t nr) {
		context_functions.viewport_zero(&context, nr);
	}

	bool inside(fan::graphics::viewport_nr_t nr, const fan::vec2& position) {
		return context_functions.viewport_inside(&context, nr, position);
	}

	bool inside_wir(fan::graphics::viewport_nr_t nr, const fan::vec2& position) {
		return context_functions.viewport_inside_wir(&context, nr, position);
	}

	bool inside(const fan::graphics::render_view_t& render_view, const fan::vec2& position) const {
		fan::vec2 tp = translate_position(position, render_view.viewport, render_view.camera);

		auto c = gloco->camera_get(render_view.camera);
		f32_t l = c.coordinates.left;
		f32_t r = c.coordinates.right;
		f32_t t = c.coordinates.up;
		f32_t b = c.coordinates.down;

		return tp.x >= l && tp.x <= r &&
					 tp.y >= t && tp.y <= b;
	}

	bool is_mouse_inside(const fan::graphics::render_view_t& render_view) const {
		return inside(render_view, get_mouse_position());
	}

	fan::graphics::context_functions_t context_functions;
	fan::graphics::context_t context;

	static std::string read_shader(const std::string& path, const std::source_location& callers_path = std::source_location::current()) {
		std::string code;
		fan::io::file::read(fan::io::file::find_relative_path(path, callers_path), &code);
		return code;
	}

	// unsafe
	//loco_t(const loco_t&) = delete;
	//loco_t& operator=(const loco_t&) = delete;
	//loco_t(loco_t&&) = delete;
	//loco_t& operator=(loco_t&&) = delete;

#if defined (fan_gui)
	using console_t = fan::console_t;
#endif

	void use() {
		gloco = this;
		fan__init_list = initialize_lists();
		window.make_context_current();
	}

	void camera_move(fan::graphics::context_camera_t& camera, f64_t dt, f32_t movement_speed, f32_t friction = 12) {
		camera.velocity /= friction * dt + 1;
		static constexpr auto minimum_velocity = 0.001;
		static constexpr f32_t camera_rotate_speed = 100;
		if (camera.velocity.x < minimum_velocity && camera.velocity.x > -minimum_velocity) {
			camera.velocity.x = 0;
		}
		if (camera.velocity.y < minimum_velocity && camera.velocity.y > -minimum_velocity) {
			camera.velocity.y = 0;
		}
		if (camera.velocity.z < minimum_velocity && camera.velocity.z > -minimum_velocity) {
			camera.velocity.z = 0;
		}

		f64_t msd = (movement_speed * dt);
		if (gloco->window.key_pressed(fan::input::key_w)) {
			camera.velocity += camera.m_front * msd;
		}
		if (gloco->window.key_pressed(fan::input::key_s)) {
			camera.velocity -= camera.m_front * msd;
		}
		if (gloco->window.key_pressed(fan::input::key_a)) {
			camera.velocity -= camera.m_right * msd;
		}
		if (gloco->window.key_pressed(fan::input::key_d)) {
			camera.velocity += camera.m_right * msd;
		}

		if (gloco->window.key_pressed(fan::input::key_space)) {
			 camera.velocity.y += msd;
		}
		if (gloco->window.key_pressed(fan::input::key_left_shift)) {
			camera.velocity.y -= msd;
		}

		f64_t rotate = camera.sensitivity * camera_rotate_speed * gloco->delta_time;
		if (gloco->window.key_pressed(fan::input::key_left)) {
			camera.set_yaw(camera.get_yaw() - rotate);
		}
		if (gloco->window.key_pressed(fan::input::key_right)) {
			camera.set_yaw(camera.get_yaw() + rotate);
		}
		if (gloco->window.key_pressed(fan::input::key_up)) {
			camera.set_pitch(camera.get_pitch() + rotate);
		}
		if (gloco->window.key_pressed(fan::input::key_down)) {
			camera.set_pitch(camera.get_pitch() - rotate);
		}

		camera.position += camera.velocity * gloco->delta_time;
		camera.update_view();

		camera.m_view = camera.get_view_matrix();
	}

#if defined(loco_opengl)
	// opengl namespace
	struct opengl {
#include <fan/graphics/opengl/engine_functions.h>
#include <fan/graphics/opengl/2D/effects/blur.h>

		blur_t blur;

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

	void add_shape_to_immediate_draw(fan::graphics::shapes::shape_t&& s) {
		immediate_render_list.emplace_back(std::move(s));
	}
	auto add_shape_to_static_draw(fan::graphics::shapes::shape_t&& s) {
		auto ret = s.NRI;
		static_render_list[ret] = std::move(s);
		return ret;
	}
	void remove_static_shape_draw(const fan::graphics::shapes::shape_t& s) {
		static_render_list.erase(s.NRI);
	}

	static void generate_commands(loco_t* loco) {
#if defined(fan_gui)
		loco->console.open();

		loco->console.commands.add("echo", [](const fan::commands_t::arg_t& args) {
			fan::commands_t::output_t out;
			out.text = fan::append_args(args) + "\n";
			out.highlight = fan::graphics::highlight_e::info;
			gloco->console.commands.output_cb(out);
			}).description = "prints something - usage echo [args]";

		loco->console.commands.add("help", [](const fan::commands_t::arg_t& args) {
			if (args.empty()) {
				fan::commands_t::output_t out;
				out.highlight = fan::graphics::highlight_e::info;
				std::string out_str;
				out_str += "{\n";
				for (const auto& i : gloco->console.commands.func_table) {
					out_str += "\t" + i.first + ",\n";
				}
				out_str += "}\n";
				out.text = out_str;
				gloco->console.commands.output_cb(out);
				return;
			}
			else if (args.size() == 1) {
				auto found = gloco->console.commands.func_table.find(args[0]);
				if (found == gloco->console.commands.func_table.end()) {
					gloco->console.commands.print_command_not_found(args[0]);
					return;
				}
				fan::commands_t::output_t out;
				out.text = found->second.description + "\n";
				out.highlight = fan::graphics::highlight_e::info;
				gloco->console.commands.output_cb(out);
			}
			else {
				gloco->console.commands.print_invalid_arg_count();
			}
			}).description = "get info about specific command - usage help command";

		loco->console.commands.add("list", [](const fan::commands_t::arg_t& args) {
			std::string out_str;
			for (const auto& i : gloco->console.commands.func_table) {
				out_str += i.first + "\n";
			}

			fan::commands_t::output_t out;
			out.text = out_str;
			out.highlight = fan::graphics::highlight_e::info;

			gloco->console.commands.output_cb(out);
			}).description = "lists all commands - usage list";

		loco->console.commands.add("alias", [](const fan::commands_t::arg_t& args) {
			if (args.size() < 2 || args[1].empty()) {
				gloco->console.commands.print_invalid_arg_count();
				return;
			}
			if (gloco->console.commands.insert_to_command_chain(args)) {
				return;
			}
			gloco->console.commands.func_table[args[0]] = gloco->console.commands.func_table[args[1]];
			}).description = "can create alias commands - usage alias [cmd name] [cmd]";


		loco->console.commands.add("show_fps", [](const fan::commands_t::arg_t& args) {
			if (args.size() != 1) {
				gloco->console.commands.print_invalid_arg_count();
				return;
			}
			gloco->show_fps = std::stoi(args[0]);
			}).description = "toggles fps - usage show_fps [value]";

		loco->console.commands.add("quit", [](const fan::commands_t::arg_t& args) {
			exit(0);
			}).description = "quits program - usage quit";

		loco->console.commands.add("clear", [](const fan::commands_t::arg_t& args) {
			gloco->console.output_buffer.clear();
			gloco->console.editor.SetText("");
			}).description = "clears output buffer - usage clear";

		loco->console.commands.add("set_gamma", [](const fan::commands_t::arg_t& args) {
			if (args.size() != 1) {
				gloco->console.commands.print_invalid_arg_count();
				return;
			}
			gloco->shader_set_value(gloco->gl.m_fbo_final_shader, "gamma", std::stof(args[0]));
			}).description = "sets gamma for postprocessing shader";

		loco->console.commands.add("set_gamma", [](const fan::commands_t::arg_t& args) {
			if (args.size() != 1) {
				gloco->console.commands.print_invalid_arg_count();
				return;
			}
			gloco->shader_set_value(gloco->gl.m_fbo_final_shader, "gamma", std::stof(args[0]));
		}).description = "sets gamma for postprocessing shader";
		loco->console.commands.add("set_contrast", [](const fan::commands_t::arg_t& args) {
			if (args.size() != 1) {
				gloco->console.commands.print_invalid_arg_count();
				return;
			}
			gloco->shader_set_value(gloco->gl.m_fbo_final_shader, "contrast", std::stof(args[0]));
		}).description = "sets contrast for postprocessing shader";

		loco->console.commands.add("set_exposure", [](const fan::commands_t::arg_t& args) {
			if (args.size() != 1) {
				gloco->console.commands.print_invalid_arg_count();
				return;
			}
			gloco->shader_set_value(gloco->gl.m_fbo_final_shader, "exposure", std::stof(args[0]));
		}).description = "sets exposure for postprocessing shader";

		loco->console.commands.add("set_bloom_strength", [](const fan::commands_t::arg_t& args) {
			if (args.size() != 1) {
				gloco->console.commands.print_invalid_arg_count();
				return;
			}
			gloco->shader_set_value(gloco->gl.m_fbo_final_shader, "bloom_strength", std::stof(args[0]));
			}).description = "sets bloom strength for postprocessing shader";

		loco->console.commands.add("set_vsync", [](const fan::commands_t::arg_t& args) {
			if (args.size() != 1) {
				gloco->console.commands.print_invalid_arg_count();
				return;
			}
			gloco->set_vsync(std::stoi(args[0]));
			}).description = "sets vsync";

		loco->console.commands.add("set_target_fps", [](const fan::commands_t::arg_t& args) {
			if (args.size() != 1) {
				gloco->console.commands.print_invalid_arg_count();
				return;
			}
			gloco->set_target_fps(std::stoi(args[0]));
		}).description = "sets target fps";

		loco->console.commands.add("debug_memory", [loco, nr = fan::console_t::frame_cb_t::nr_t()](const fan::commands_t::arg_t& args) mutable {
			if (args.size() != 1) {
				loco->console.commands.print_invalid_arg_count();
				return;
			}
			if (nr.iic() && std::stoi(args[0])) {
				nr = loco->console.push_frame_process([] {
					ImGui::SetNextWindowBgAlpha(0.99f);
					static int init = 0;
					ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoFocusOnAppearing;
					if (init == 0) {
						ImGui::SetNextWindowSize(fan::vec2(600, 300));
						//window_flags |= ImGuiWindowFlags_AlwaysAutoResize;
						init = 1;
					}
					ImGui::Begin("fan_memory_dbg_wnd", 0, window_flags);
					fan::graphics::gui::render_allocations_plot();
					ImGui::End();
					});
			}
			else if (!nr.iic() && !std::stoi(args[0])) {
				loco->console.erase_frame_process(nr);
			}
		}).description = "opens memory debug window";
		loco->console.commands.add("set_clear_color", [](const fan::commands_t::arg_t& args) {
			if (args.size() != 1) {
				gloco->console.commands.print_invalid_arg_count();
				return;
			}
			gloco->clear_color = fan::color::parse(args[0]);
			}).description = "sets clear color of window - input example {1,0,0,1} red";

		// shapes
		loco->console.commands.add("rectangle", [](const fan::commands_t::arg_t& args) {
			if (args.size() < 1 || args.size() > 3) {
				gloco->console.commands.print_invalid_arg_count();
				return;
			}

			try {
				fan::graphics::shapes::rectangle_t::properties_t props;
				props.position = fan::vec3::parse(args[0]);
				// optional
				if (args.size() >= 2) props.size = fan::vec2::parse(args[1]);
				// optional
				props.color = args.size() == 3 ? fan::color::parse(args[2]) : fan::colors::white;

				auto NRI = gloco->add_shape_to_static_draw(props);
				gloco->console.println_colored(
					"Added rectangle",
					fan::colors::green
				);
				gloco->console.println(
					fan::format(
						"  id: {}\n  position {}\n  size {}\n  color {}",
						NRI,
						props.position,
						props.size,
						props.color
					),
					fan::graphics::highlight_e::info
				);
			}
			catch (const std::exception& e) {
				gloco->console.println_colored("Invalid arguments: " + std::string(e.what()), fan::colors::red);
			}
		}).description = "Adds static rectangle {x,y[,z]} {w,h} [{r,g,b,a}]";

		loco->console.commands.add("remove_shape", [](const fan::commands_t::arg_t& args) {
			if (args.size() != 1) {
				gloco->console.commands.print_invalid_arg_count();
				return;
			}

			try {
				uint32_t shape_id = std::stoull(args[0]);
				//shape_id
				fan::graphics::shapes::shape_t* s = reinterpret_cast<fan::graphics::shapes::shape_t*>(&shape_id);
				gloco->remove_static_shape_draw(*s);
				gloco->console.println_colored(
						fan::format("Removed shape with id {}", shape_id),
						fan::colors::green
				);
			}
			catch (const std::exception& e) {
				gloco->console.println_colored(
					"Invalid argument: " + std::string(e.what()),
					fan::colors::red
				);
			}
			}).description = "Removes a shape by its id";


#endif
	}

	// -1 no reload, opengl = 0 etc
	uint8_t reload_renderer_to = -1;
#if defined(fan_gui)
	void load_fonts(ImFont* (&fonts)[std::size(fan::graphics::gui::font_sizes)], const std::string& name, ImFontConfig* cfg = nullptr) {
		ImGuiIO& io = ImGui::GetIO();
		for (std::size_t i = 0; i < std::size(fonts); ++i) {
			fonts[i] = io.Fonts->AddFontFromFileTTF(name.c_str(), fan::graphics::gui::font_sizes[i] * 2, cfg);

			if (fonts[i] == nullptr) {
				fan::throw_error(std::string("failed to load font:") + name);
			}
		}
	}
	void build_fonts() {
		ImGuiIO& io = ImGui::GetIO();
		io.Fonts->Build();
	}

	ImFont* get_font(f32_t font_size, bool bold = false) {
		font_size /= 2;
		int best_index = 0;
		f32_t best_diff = std::abs(fan::graphics::gui::font_sizes[0] - font_size);

		for (std::size_t i = 1; i < std::size(fan::graphics::gui::font_sizes); ++i) {
			f32_t diff = std::abs(fan::graphics::gui::font_sizes[i] - font_size);
			if (diff < best_diff) {
				best_diff = diff;
				best_index = i;
			}
		}

		return !bold ? fan::graphics::gui::fonts[best_index] : fan::graphics::gui::fonts_bold[best_index];
	}

#if defined(fan_vulkan)
	// todo move to vulkan context
	static void check_vk_result(VkResult err) {
		if (err != VK_SUCCESS) {
			fan::print("vkerr", (int)err);
		}
	}
#endif

	void init_imgui() {
		if (global_imgui_initialized) {
			imgui_initialized = true;
			return;
		}

		ImGui::CreateContext();
		ImPlot::CreateContext();
		auto& input_map = ImPlot::GetInputMap();
		input_map.Pan = ImGuiMouseButton_Middle;

		ImGuiIO& io = ImGui::GetIO(); (void)io;
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
		//io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
		io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
		///    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

		ImGuiStyle& style = ImGui::GetStyle();
		if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
		{
			style.WindowRounding = 0.;
		}
		style.FrameRounding = 5.f;
		style.FramePadding = ImVec2(12.f, 5.f);
		style.Colors[ImGuiCol_WindowBg].w = 1.0f;

		imgui_themes::dark();

		if (window.renderer == fan::window_t::renderer_t::opengl) {
			glfwMakeContextCurrent(window);
			ImGui_ImplGlfw_InitForOpenGL(window, true);
			const char* glsl_version = "#version 120";
			ImGui_ImplOpenGL3_Init(glsl_version);
		}
#if defined(fan_vulkan)
		else if (window.renderer == fan::window_t::renderer_t::vulkan) {
			ImGui_ImplGlfw_InitForVulkan(window, true);
			ImGui_ImplVulkan_InitInfo init_info = {};
			init_info.Instance = context.vk.instance;
			init_info.PhysicalDevice = context.vk.physical_device;
			init_info.Device = context.vk.device;
			init_info.QueueFamily = context.vk.queue_family;
			init_info.Queue = context.vk.graphics_queue;
			init_info.DescriptorPool = context.vk.descriptor_pool.m_descriptor_pool;
			init_info.RenderPass = context.vk.MainWindowData.RenderPass;
			init_info.Subpass = 0;
			init_info.MinImageCount = context.vk.MinImageCount;
			init_info.ImageCount = context.vk.MainWindowData.ImageCount;
			init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
			init_info.CheckVkResultFn = check_vk_result;

			ImGui_ImplVulkan_Init(&init_info);
		}
#endif

		load_fonts(fan::graphics::gui::fonts_bold, "fonts/SourceCodePro-Bold.ttf");

		ImFontConfig emoji_cfg;
		emoji_cfg.FontBuilderFlags |= ImGuiFreeTypeBuilderFlags_LoadColor | ImGuiFreeTypeBuilderFlags_Bitmap;

		// TODO
		static const ImWchar emoji_ranges[] = {
			//0x2600, 0x26FF,    // Miscellaneous Symbols
			//0x2700, 0x27BF,    // Dingbats
			//0x2B00, 0x2BFF,  
			//0x1F300, 0x1F5FF,  // Miscellaneous Symbols and Pictographs
			//0x1F600, 0x1F64F,  // Emoticons
			//0x1F680, 0x1F6FF,  // Transport and Map Symbols
			//0x1F900, 0x1F9FF,  // Supplemental Symbols and Pictographs
			//0x1FA70, 0x1FAFF,  // Symbols and Pictographs Extended-A
			//0
			0x2600, 0x26FF,    // Miscellaneous Symbols
			0x2B00, 0x2BFF, // Miscellaneous Symbols and Arrows
			0x1F600, 0x1F64F,  // Emoticons
			0
		};

		io.Fonts->FontBuilderIO = ImGuiFreeType::GetBuilderForFreeType();
		io.Fonts->FontBuilderFlags |= ImGuiFreeTypeBuilderFlags_LoadColor;
		
		for (std::size_t i = 0; i < std::size(fan::graphics::gui::fonts); ++i) {
			f32_t font_size = fan::graphics::gui::font_sizes[i] * 2; // load 2x font size and possibly downscale for better quality

			ImFontConfig main_cfg;
			fan::graphics::gui::fonts[i] = io.Fonts->AddFontFromFileTTF("fonts/SourceCodePro-Regular.ttf", font_size, &main_cfg);

			ImFontConfig emoji_cfg;
			emoji_cfg.MergeMode = true;
			emoji_cfg.FontBuilderFlags |= ImGuiFreeTypeBuilderFlags_LoadColor;
			emoji_cfg.SizePixels = 0;
			emoji_cfg.RasterizerDensity = 1.0f;
			emoji_cfg.GlyphMinAdvanceX = font_size;
			io.Fonts->AddFontFromFileTTF("fonts/seguiemj.ttf", font_size, &emoji_cfg, emoji_ranges);
		}
		build_fonts();
		io.FontDefault = fan::graphics::gui::fonts[9];

		input_action.add(fan::key_escape, "open_settings");
		input_action.add(fan::key_a, "move_left");
		input_action.add(fan::key_d, "move_right");
		input_action.add(fan::key_w, "move_forward");
		input_action.add(fan::key_s, "move_back");
		input_action.add(fan::key_space, "move_up");
		global_imgui_initialized = true;
		imgui_initialized = true;
	}
	void destroy_imgui() {
		if (!imgui_initialized || !global_imgui_initialized) {
			return;
		}

		if (reload_renderer_to != (decltype(reload_renderer_to))-1) {
			if (window.renderer == fan::window_t::renderer_t::opengl) {
				ImGui_ImplOpenGL3_Shutdown();
			}
#if defined(fan_vulkan)
			else if (window.renderer == fan::window_t::renderer_t::vulkan) {
				vkDeviceWaitIdle(context.vk.device);
				ImGui_ImplVulkan_Shutdown();
			}
#endif
			imgui_initialized = false;
			return;
		}

		if (window.renderer == fan::window_t::renderer_t::opengl) {
			ImGui_ImplOpenGL3_Shutdown();
		}
#if defined(fan_vulkan)
		else if (window.renderer == fan::window_t::renderer_t::vulkan) {
			vkDeviceWaitIdle(context.vk.device);
			ImGui_ImplVulkan_Shutdown();
		}
#endif
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();
		ImPlot::DestroyContext();
#if defined(fan_vulkan)
		if (window.renderer == fan::window_t::renderer_t::vulkan) {
			context.vk.imgui_close();
		}
#endif

		global_imgui_initialized = false;
		imgui_initialized = false;
	}
	bool enable_overlay = true;
#endif
	void init_framebuffer() {
		if (window.renderer == fan::window_t::renderer_t::opengl) {
			gl.init_framebuffer();
		}
	}

	loco_t() : loco_t(properties_t()) {

	}
	loco_t(const properties_t& p) {

        // init globals

    fan::graphics::ctx().render_functions = &context_functions;
    fan::graphics::ctx().render_context = &context;
    fan::graphics::ctx().image_list = &image_list;
    fan::graphics::ctx().shader_list = &shader_list;
    fan::graphics::ctx().window = &window;
    fan::graphics::ctx().orthographic_render_view = &orthographic_render_view;
    fan::graphics::ctx().perspective_render_view = &perspective_render_view;
    fan::graphics::ctx().update_callback = &m_update_callback;
    fan::graphics::ctx().input_action = &input_action;
    fan::graphics::ctx().lighting = &lighting;

  #if defined(fan_gui)
    fan::graphics::ctx().console = &console;
    fan::graphics::ctx().text_logger = &text_logger;
  #endif

    shapes.texture_pack = &texture_pack;
    shapes.immediate_render_list = &immediate_render_list;
    shapes.static_render_list = &static_render_list;

    physics_context.physics_updates = &shape_physics_update_cbs;

    input_action.is_active_func = [this] (int key) -> int{
      return window.key_state(key);
    };

    fan::graphics::shaper_t::gl_add_shape_type = [&](
      fan::graphics::shaper_t::ShapeTypes_NodeData_t& nd,
      const fan::graphics::shaper_t::BlockProperties_t& bp) {
        gl.add_shape_type(nd, bp);
    };
    fan::graphics::g_shapes = &shapes;

	#if defined(fan_gui) && defined(fan_std23)
		fan::setup_imgui_with_heap_profiler();
	#endif

	#if defined(fan_platform_windows)
		// use utf8 for console output
		SetConsoleOutputCP(CP_UTF8);
	#endif

		if (fan::init_manager_t::initialized() == false) {
			fan::init_manager_t::initialize();
		}
		render_shapes_top = p.render_shapes_top;
		window.renderer = p.renderer;
		if (window.renderer == fan::window_t::renderer_t::opengl) {
			new (&context.gl) fan::opengl::context_t();
			context_functions = fan::graphics::get_gl_context_functions();
			gl.open();
		}

		window.set_antialiasing(p.samples);
		window.open(p.window_size, fan::window_t::default_window_name, p.window_flags);
		gloco = this;


#if fan_debug >= fan_debug_high && !defined(fan_vulkan)
		if (window.renderer == fan::window_t::renderer_t::vulkan) {
			fan::throw_error("trying to use vulkan renderer, but fan_vulkan build flag is disabled");
		}
#endif

#if defined(fan_vulkan)
		if (window.renderer == fan::window_t::renderer_t::vulkan) {
			context_functions = fan::graphics::get_vk_context_functions();
			new (&context.vk) fan::vulkan::context_t();
			context.vk.enable_clear = !render_shapes_top;
			context.vk.shapes_top = render_shapes_top;
			context.vk.open(window);
		}
#endif

		start_time.start();

		set_vsync(false); // using libuv
		//fan::print("less pain", this, (void*)&lighting, (void*)((uint8_t*)&lighting - (uint8_t*)this), sizeof(*this), lighting.ambient);
		if (window.renderer == fan::window_t::renderer_t::opengl) {
			window.make_context_current();

#if fan_debug >= fan_debug_high
			get_context().gl.set_error_callback();
#endif

			if (window.get_antialiasing() > 0) {
				glEnable(GL_MULTISAMPLE);
			}

			gl.initialize_fb_vaos();
		}


		load_engine_images();

		fan::graphics::g_shapes->shaper.Open();

		{

			// filler
			fan::graphics::g_shapes->shaper.AddKey(fan::graphics::Key_e::light, sizeof(uint8_t), fan::graphics::shaper_t::KeyBitOrderAny);
			fan::graphics::g_shapes->shaper.AddKey(fan::graphics::Key_e::light_end, sizeof(uint8_t), fan::graphics::shaper_t::KeyBitOrderAny);
			fan::graphics::g_shapes->shaper.AddKey(fan::graphics::Key_e::depth, sizeof(fan::graphics::depth_t), fan::graphics::shaper_t::KeyBitOrderLow);
			fan::graphics::g_shapes->shaper.AddKey(fan::graphics::Key_e::blending, sizeof(fan::graphics::blending_t), fan::graphics::shaper_t::KeyBitOrderLow);
			fan::graphics::g_shapes->shaper.AddKey(fan::graphics::Key_e::image, sizeof(fan::graphics::image_t), fan::graphics::shaper_t::KeyBitOrderLow);
			fan::graphics::g_shapes->shaper.AddKey(fan::graphics::Key_e::viewport, sizeof(fan::graphics::viewport_t), fan::graphics::shaper_t::KeyBitOrderAny);
			fan::graphics::g_shapes->shaper.AddKey(fan::graphics::Key_e::camera, sizeof(loco_t::camera_t), fan::graphics::shaper_t::KeyBitOrderAny);
			fan::graphics::g_shapes->shaper.AddKey(fan::graphics::Key_e::ShapeType, sizeof(fan::graphics::shaper_t::ShapeTypeIndex_t), fan::graphics::shaper_t::KeyBitOrderAny);
			fan::graphics::g_shapes->shaper.AddKey(fan::graphics::Key_e::filler, sizeof(uint8_t), fan::graphics::shaper_t::KeyBitOrderAny);
			fan::graphics::g_shapes->shaper.AddKey(fan::graphics::Key_e::draw_mode, sizeof(uint8_t), fan::graphics::shaper_t::KeyBitOrderAny);
			fan::graphics::g_shapes->shaper.AddKey(fan::graphics::Key_e::vertex_count, sizeof(uint32_t), fan::graphics::shaper_t::KeyBitOrderAny);
			fan::graphics::g_shapes->shaper.AddKey(fan::graphics::Key_e::shadow, sizeof(uint8_t), fan::graphics::shaper_t::KeyBitOrderAny);

			//fan::graphics::g_shapes->shaper.AddKey(fan::graphics::Key_e::image4, sizeof(fan::graphics::image_t) * 4, fan::graphics::shaper_t::KeyBitOrderLow);
		}
		// order of open needs to be same with shapes enum

		{
			fan::vec2 window_size = window.get_size();
			{
				orthographic_render_view.camera = open_camera(
					fan::vec2(0, window_size.x),
					fan::vec2(0, window_size.y)
				);
				orthographic_render_view.viewport = open_viewport(
					fan::vec2(0, 0),
					window_size
				);
			}
			{
				perspective_render_view.camera = open_camera_perspective();
				perspective_render_view.viewport = open_viewport(
					fan::vec2(0, 0),
					window_size
				);
			}
		}

		if (window.renderer == fan::window_t::renderer_t::opengl) {
			gl.shapes_open();
		}
#if defined(fan_vulkan)
		else if (window.renderer == fan::window_t::renderer_t::vulkan) {
			vk.shapes_open();
		}
#endif


#if defined(fan_physics)
		fan::graphics::open_bcol();
#endif

#if defined(fan_gui)
		init_imgui();
		generate_commands(this);
#endif

    setup_input_callbacks();

#if defined(fan_gui)
		settings_menu.open();
#endif

		auto it = fan::graphics::engine_init_cbs.GetNodeFirst();
		while (it != fan::graphics::engine_init_cbs.dst) {
			fan::graphics::engine_init_cbs.StartSafeNext(it);
			fan::graphics::engine_init_cbs[it](this);
			it = fan::graphics::engine_init_cbs.EndSafeNext();
		}

#if defined(fan_audio)

		if (system_audio.Open() != 0) {
			fan::throw_error("failed to open fan audio");
		}
		audio.bind(&system_audio);
		fan::audio::piece_hover.open_piece("audio/hover.sac", 0);
		fan::audio::piece_click.open_piece("audio/click.sac", 0);

    fan::audio::g_audio = &audio;

#endif

    fan::graphics::g_render_context_handle.default_texture = default_texture;
	}
	~loco_t() {
		destroy();
	}

	void destroy() {
    // TODO fix destruct order to not do manually, because shaper closes before them?
    static_render_list.clear();
    immediate_render_list.clear();

		if (window == nullptr) {
			return;
		}

    //unload_engine_images();
#if defined(fan_opengl)
    gl.close();
#endif

#if defined(fan_gui)
		console.commands.func_table.clear();
		console.close();
#endif
#if defined(fan_physics)
		fan::graphics::close_bcol();
#endif
#if defined(fan_vulkan)
		if (window.renderer == fan::window_t::renderer_t::vulkan) {
			vkDeviceWaitIdle(context.vk.device);
			vkDestroySampler(context.vk.device, vk.post_process_sampler, nullptr);
			vk.d_attachments.close(context.vk);
			vk.post_process.close(context.vk);
		}
#endif
		fan::graphics::g_shapes->shaper.Close();
#if defined(fan_gui)
		destroy_imgui();
#endif
		window.close();
#if defined(fan_audio)
		audio.unbind();
		system_audio.Close();
#endif
	}
	void close() {
		destroy();
	}

  using mouse_click_callback_t   = std::function<void()>;
  using mouse_down_callback_t    = std::function<void(fan::vec2 mouse_pos)>;
  using mouse_up_callback_t      = std::function<void()>;
  using mouse_move_callback_t    = std::function<void(fan::vec2 mouse_pos, fan::vec2 delta)>;
  using key_click_callback_t     = std::function<void()>;
  using key_up_callback_t        = std::function<void()>;
  using key_down_callback_t      = std::function<void()>;
  using on_resize_callback_t     = std::function<void(fan::vec2)>; // window resize

private:
  #define BLL_set_prefix mouse_click_callbacks
  #define BLL_set_NodeData mouse_click_callback_t data;
  #include <fan/window/cb_list_builder_settings.h>
  #include <BLL/BLL.h>

  #define BLL_set_prefix mouse_down_callbacks
  #define BLL_set_NodeData mouse_down_callback_t data;
  #include <fan/window/cb_list_builder_settings.h>
  #include <BLL/BLL.h>

  #define BLL_set_prefix mouse_up_callbacks
  #define BLL_set_NodeData mouse_up_callback_t data;
  #include <fan/window/cb_list_builder_settings.h>
  #include <BLL/BLL.h>

  #define BLL_set_prefix mouse_move_callbacks
  #define BLL_set_NodeData mouse_move_callback_t data;
  #include <fan/window/cb_list_builder_settings.h>
  #include <BLL/BLL.h>

  #define BLL_set_prefix key_click_callbacks
  #define BLL_set_NodeData key_click_callback_t data;
  #include <fan/window/cb_list_builder_settings.h>
  #include <BLL/BLL.h>

  #define BLL_set_prefix key_up_callbacks
  #define BLL_set_NodeData key_up_callback_t data;
  #include <fan/window/cb_list_builder_settings.h>
  #include <BLL/BLL.h>

  #define BLL_set_prefix key_down_callbacks
  #define BLL_set_NodeData key_down_callback_t data;
  #include <fan/window/cb_list_builder_settings.h>
  #include <BLL/BLL.h>

  #define BLL_set_prefix on_resize_callbacks
  #define BLL_set_NodeData on_resize_callback_t data;
  #include <fan/window/cb_list_builder_settings.h>
  #include <BLL/BLL.h>

public:

  mouse_click_callbacks_t    m_mouse_click_callbacks;
  mouse_down_callbacks_t     m_mouse_down_callbacks;
  mouse_up_callbacks_t       m_mouse_up_callbacks;
  mouse_move_callbacks_t     m_mouse_move_callbacks;
  key_click_callbacks_t      m_key_click_callbacks;
  key_up_callbacks_t         m_key_up_callbacks;
  key_down_callbacks_t       m_key_down_callbacks;
  on_resize_callbacks_t      m_on_resize_callbacks;

  using mouse_click_nr_t    = mouse_click_callbacks_NodeReference_t;
  using mouse_down_nr_t     = mouse_down_callbacks_NodeReference_t;
  using mouse_up_nr_t       = mouse_up_callbacks_NodeReference_t;
  using mouse_move_nr_t     = mouse_move_callbacks_NodeReference_t;
  using key_click_nr_t      = key_click_callbacks_NodeReference_t;
  using key_up_nr_t         = key_up_callbacks_NodeReference_t;
  using key_down_nr_t       = key_down_callbacks_NodeReference_t;
  using on_resize_nr_t      = on_resize_callbacks_NodeReference_t;

  void setup_input_callbacks() {
    // TODO callbacks leaking
    window.add_buttons_callback([this](const fan::window_t::mouse_buttons_cb_data_t& d) {
      fan::vec2 pos = fan::vec2(d.window->get_mouse_position());

      if (d.state == fan::mouse_state::press && d.button < 3) {
        auto it = m_mouse_click_callbacks.GetNodeFirst();
        while (it != m_mouse_click_callbacks.dst) {
          m_mouse_click_callbacks[it].data();
          it = it.Next(&m_mouse_click_callbacks);
        }
      }
      else if (d.state == fan::mouse_state::release && d.button < 3) {
        auto it = m_mouse_up_callbacks.GetNodeFirst();
        while (it != m_mouse_up_callbacks.dst) {
          m_mouse_up_callbacks[it].data();
          it = it.Next(&m_mouse_up_callbacks);
        }
      }

    #if defined(loco_vfi)
      fan::graphics::g_shapes->vfi.feed_mouse_button(d.button, d.state);
    #endif
      });

    window.add_keys_callback([&](const fan::window_t::keyboard_keys_cb_data_t& d) {
      if (d.state == fan::keyboard_state_t::press) {
        auto it = m_key_click_callbacks.GetNodeFirst();
        while (it != m_key_click_callbacks.dst) {
          m_key_click_callbacks[it].data();
          it = it.Next(&m_key_click_callbacks);
        }
      }
      else if (d.state == fan::keyboard_state_t::release) {
        auto it = m_key_up_callbacks.GetNodeFirst();
        while (it != m_key_up_callbacks.dst) {
          m_key_up_callbacks[it].data();
          it = it.Next(&m_key_up_callbacks);
        }
      }
    #if defined(loco_vfi)
      fan::graphics::g_shapes->vfi.feed_keyboard(d.key, d.state);
    #endif
      });

    window.add_mouse_move_callback([&](const fan::window_t::mouse_move_cb_data_t& d) {
      fan::vec2 pos = fan::vec2(d.position);
      fan::vec2 delta = pos - d.window->previous_mouse_position;

      auto it = m_mouse_move_callbacks.GetNodeFirst();
      while (it != m_mouse_move_callbacks.dst) {
        m_mouse_move_callbacks[it].data(pos, delta);
        it = it.Next(&m_mouse_move_callbacks);
      }

    #if defined(loco_vfi)
      fan::graphics::g_shapes->vfi.feed_mouse_move(d.position);
    #endif
      });

    window.add_text_callback([&](const fan::window_t::text_cb_data_t& d) {
    #if defined(loco_vfi)
      fan::graphics::g_shapes->vfi.feed_text(d.character);
    #endif
      });

    bool windowed = true;
    // free this xd
    gloco->window.add_keys_callback(
      [windowed](const fan::window_t::keyboard_keys_cb_data_t& data) mutable {
        if (data.key == fan::key_enter && data.state == fan::keyboard_state::press && gloco->window.key_pressed(fan::key_left_alt)) {
          windowed = !windowed;
          gloco->window.set_display_mode(windowed ? fan::window_t::mode::windowed : fan::window_t::mode::borderless);
        }
      }
    );

    window.add_resize_callback([this](const auto& d) {
      auto it = m_on_resize_callbacks.GetNodeFirst();
      while (it != m_on_resize_callbacks.dst) {
        m_on_resize_callbacks[it].data(d.size);
        it = it.Next(&m_on_resize_callbacks);
      }
    });
  }


	// for renderer switch
	// input fan::window_t::renderer_t::
	void switch_renderer(uint8_t renderer) {
		std::vector<std::string> image_paths;
		fan::vec2 window_size = window.get_size();
		fan::vec2 window_position = window.get_position();
		uint64_t flags = window.flags;

#if defined(fan_gui)
		bool was_imgui_init = imgui_initialized;
#endif

		{// close
#if defined(fan_vulkan)
			if (window.renderer == fan::window_t::renderer_t::vulkan) {
				// todo wrap to vk.
				vkDeviceWaitIdle(context.vk.device);
				vkDestroySampler(context.vk.device, vk.post_process_sampler, nullptr);
				vk.d_attachments.close(context.vk);
				vk.post_process.close(context.vk);
				for (auto& st : fan::graphics::g_shapes->shaper.ShapeTypes) {
					if (st.sti == (decltype(st.sti))-1) {
						continue;
					}
				#if defined(fan_vulkan)
					auto& str = st.renderer.vk;
					str.shape_data.close(context.vk);
					str.pipeline.close(context.vk);
				#endif
					//st.BlockList.Close();
				}
				//CLOOOOSEEE POSTPROCESSS IMAGEEES
			}
			else
#endif
				if (window.renderer == fan::window_t::renderer_t::opengl) {
					glDeleteVertexArrays(1, &gl.fb_vao);
					glDeleteBuffers(1, &gl.fb_vbo);
					context.gl.internal_close();
				}

#if defined(fan_gui)
			if (imgui_initialized) {
				if (window.renderer == fan::window_t::renderer_t::opengl) {
					ImGui_ImplOpenGL3_Shutdown();
				}
#if defined(fan_vulkan)
				else if (window.renderer == fan::window_t::renderer_t::vulkan) {
					vkDeviceWaitIdle(context.vk.device);
					ImGui_ImplVulkan_Shutdown();
				}
#endif
				ImGui_ImplGlfw_Shutdown();
				imgui_initialized = false;
			}
#endif

			window.close();
		}

		{// reopen
			window.renderer = reload_renderer_to; // i dont like this {window.renderer = ...}
			if (window.renderer == fan::window_t::renderer_t::opengl) {
				context_functions = fan::graphics::get_gl_context_functions();
				new (&context.gl) fan::opengl::context_t();
				gl.open();
			}

			window.open(window_size, fan::window_t::default_window_name, flags | fan::window_t::flags::hidden);
			window.set_position(window_position);
			window.set_position(window_position);
			glfwShowWindow(window);
			window.flags = flags;

#if defined(fan_vulkan)
			if (window.renderer == fan::window_t::renderer_t::vulkan) {
				new (&context.vk) fan::vulkan::context_t();
				context_functions = fan::graphics::get_vk_context_functions();
				context.vk.open(window);
			}
#endif
		}

		{// reload
			{
				{
					fan::graphics::camera_list_t::nrtra_t nrtra;
					fan::graphics::camera_nr_t nr;
					nrtra.Open(&__fan_internal_camera_list, &nr);
					while (nrtra.Loop(&__fan_internal_camera_list, &nr)) {
						auto& cam = __fan_internal_camera_list[nr];
						camera_set_ortho(
							nr,
							fan::vec2(cam.coordinates.left, cam.coordinates.right),
							fan::vec2(cam.coordinates.up, cam.coordinates.down)
						);
					}
					nrtra.Close(&__fan_internal_camera_list);
				}
				{
					fan::graphics::viewport_list_t::nrtra_t nrtra;
					fan::graphics::viewport_nr_t nr;
					nrtra.Open(&__fan_internal_viewport_list, &nr);
					while (nrtra.Loop(&__fan_internal_viewport_list, &nr)) {
						auto& viewport = __fan_internal_viewport_list[nr];
						viewport_set(
							nr,
							viewport.viewport_position,
							viewport.viewport_size
						);
					}
					nrtra.Close(&__fan_internal_viewport_list);
				}
			}

			{
				{
					{
						fan::graphics::image_list_t::nrtra_t nrtra;
						fan::graphics::image_nr_t nr;
						nrtra.Open(&image_list, &nr);
						while (nrtra.Loop(&image_list, &nr)) {

							if (window.renderer == fan::window_t::renderer_t::opengl) {
								// illegal
								image_list[nr].internal = new fan::opengl::context_t::image_t;
								fan_opengl_call(glGenTextures(1, &((fan::opengl::context_t::image_t*)context_functions.image_get(&context.gl, nr))->texture_id));
							}
#if defined(fan_vulkan)
							else if (window.renderer == fan::window_t::renderer_t::vulkan) {
								// illegal
								image_list[nr].internal = new fan::vulkan::context_t::image_t;
							}
#endif
							// handle blur?
							auto image_path = image_list[nr].image_path;
							if (image_path.empty()) {
								fan::image::info_t info;
								info.data = (void*)fan::image::missing_texture_pixels;
								info.size = 2;
								info.channels = 4;
								fan::graphics::image_load_properties_t lp;
								lp.min_filter = fan::graphics::image_filter::nearest;
								lp.mag_filter = fan::graphics::image_filter::nearest;
								lp.visual_output = fan::graphics::image_sampler_address_mode::repeat;
								image_reload(nr, info, lp);
							}
							else {
								image_reload(nr, image_list[nr].image_path);
							}
						}
						nrtra.Close(&image_list);
					}
					{
						fan::graphics::shader_list_t::nrtra_t nrtra;
						fan::graphics::shader_nr_t nr;
						nrtra.Open(&__fan_internal_shader_list, &nr);
						while (nrtra.Loop(&__fan_internal_shader_list, &nr)) {
							if (window.renderer == fan::window_t::renderer_t::opengl) {
								__fan_internal_shader_list[nr].internal = new fan::opengl::context_t::shader_t;
							}
#if defined(fan_vulkan)
							else if (window.renderer == fan::window_t::renderer_t::vulkan) {
								__fan_internal_shader_list[nr].internal = new fan::vulkan::context_t::shader_t;
								((fan::vulkan::context_t::shader_t*)__fan_internal_shader_list[nr].internal)->projection_view_block = new std::remove_pointer_t<decltype(fan::vulkan::context_t::shader_t::projection_view_block)>;
							}
#endif
						}
						nrtra.Close(&__fan_internal_shader_list);
					}
				}
				fan::image::info_t info;
				info.data = (void*)fan::image::missing_texture_pixels;
				info.size = 2;
				info.channels = 4;
				fan::graphics::image_load_properties_t lp;
				lp.min_filter = fan::graphics::image_filter::nearest;
				lp.mag_filter = fan::graphics::image_filter::nearest;
				lp.visual_output = fan::graphics::image_sampler_address_mode::repeat;
				image_reload(default_texture, info, lp);
			}

			if (window.renderer == fan::window_t::renderer_t::opengl) {
				gl.shapes_open();
				gl.initialize_fb_vaos();
				if (window.get_antialiasing() > 0) {
					glEnable(GL_MULTISAMPLE);
				}
			}
#if defined(fan_vulkan)
			else if (window.renderer == fan::window_t::renderer_t::vulkan) {
				vk.shapes_open();
			}
#endif

#if defined(fan_gui)
			if (was_imgui_init && global_imgui_initialized) {
				if (window.renderer == fan::window_t::renderer_t::opengl) {
					glfwMakeContextCurrent(window);
					ImGui_ImplGlfw_InitForOpenGL(window, true);
					const char* glsl_version = "#version 120";
					ImGui_ImplOpenGL3_Init(glsl_version);
				}
#if defined(fan_vulkan)
				else if (window.renderer == fan::window_t::renderer_t::vulkan) {
					ImGui_ImplGlfw_InitForVulkan(window, true);
					ImGui_ImplVulkan_InitInfo init_info = {};
					init_info.Instance = context.vk.instance;
					init_info.PhysicalDevice = context.vk.physical_device;
					init_info.Device = context.vk.device;
					init_info.QueueFamily = context.vk.queue_family;
					init_info.Queue = context.vk.graphics_queue;
					init_info.DescriptorPool = context.vk.descriptor_pool.m_descriptor_pool;
					init_info.RenderPass = context.vk.MainWindowData.RenderPass;
					init_info.Subpass = 0;
					init_info.MinImageCount = context.vk.MinImageCount;
					init_info.ImageCount = context.vk.MainWindowData.ImageCount;
					init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
					init_info.CheckVkResultFn = check_vk_result;
					ImGui_ImplVulkan_Init(&init_info);
				}
#endif
				imgui_initialized = true;
				settings_menu.set_settings_theme();
			}
#endif

			fan::graphics::g_shapes->shaper._BlockListCapacityChange(fan::graphics::shapes::shape_type_t::rectangle, 0, 1);
			fan::graphics::g_shapes->shaper._BlockListCapacityChange(fan::graphics::shapes::shape_type_t::sprite, 0, 1);

#if defined(fan_audio)
			if (system_audio.Open() != 0) {
				fan::throw_error("failed to open fan audio");
			}
			audio.bind(&system_audio);
#endif
		}
		reload_renderer_to = -1;
	}

	void draw_shapes() {
    shape_draw_timer.start();
		if (window.renderer == fan::window_t::renderer_t::opengl) {
			gl.draw_shapes();
		}
#if defined(fan_vulkan)
		else
			if (window.renderer == fan::window_t::renderer_t::vulkan) {
				vk.draw_shapes();
			}
#endif
    shape_draw_time_s = shape_draw_timer.seconds();

		immediate_render_list.clear();
	}
	void process_shapes() {

#if defined(fan_vulkan)
		if (window.renderer == fan::window_t::renderer_t::vulkan) {
			if (render_shapes_top == true) {
				vk.begin_render_pass();
			}
		}
#endif
		for (const auto& i : m_pre_draw) {
			i();
		}

		draw_shapes();

		for (const auto& i : m_post_draw) {
			i();
		}

#if defined(fan_vulkan)
		if (window.renderer == fan::window_t::renderer_t::vulkan) {
			auto& cmd_buffer = context.vk.command_buffers[context.vk.current_frame];
			if (vk.image_error != (decltype(vk.image_error))-0xfff) {
				vkCmdNextSubpass(cmd_buffer, VK_SUBPASS_CONTENTS_INLINE);
				vkCmdBindPipeline(cmd_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk.post_process);
				vkCmdBindDescriptorSets(
					cmd_buffer,
					VK_PIPELINE_BIND_POINT_GRAPHICS,
					vk.post_process.m_layout,
					0,
					1,
					vk.d_attachments.m_descriptor_set,
					0,
					nullptr
				);

				// render post process
				vkCmdDraw(cmd_buffer, 6, 1, 0, 0);
			}
			if (render_shapes_top == true) {
				vkCmdEndRenderPass(cmd_buffer);
			}
		}
#endif
	}
	void process_gui() {
    gui_draw_timer.start();
#if defined(fan_gui)
		fan::graphics::gui::process_loop();

		// append
		ImGui::Begin("##global_renderer");
		text_logger.render();
		ImGui::End();

		if (ImGui::IsKeyPressed(ImGuiKey_F3, false)) {
			render_console = !render_console;

			// force focus xd
			console.input.InsertText("a");
			console.input.SetText("");
			console.init_focus = true;
			console.input.IsFocused() = false;
		}
		if (render_console) {
			console.render();
		}
		if (input_action.is_active("open_settings")) {
			render_settings_menu = !render_settings_menu;
		}
		if (render_settings_menu) {
			settings_menu.render();
		}

    if (show_fps) {
      ImGui::SetNextWindowBgAlpha(0.99f);

      ImGuiWindowFlags window_flags =
        ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoFocusOnAppearing;

      ImGui::SetNextWindowSize(fan::vec2(831.0000, 693.0000), ImGuiCond_Once);
      ImGui::Begin("Performance window", nullptr, window_flags);

      frame_monitor.update(delta_time);
      shape_monitor.update(shape_draw_time_s);
      gui_monitor.update(gui_draw_time_s);

      auto frame_stats = frame_monitor.calculate_stats(delta_time);
      auto shape_stats = shape_monitor.calculate_stats(shape_draw_time_s);
      auto gui_stats = gui_monitor.calculate_stats(gui_draw_time_s);

      ImGui::Text("FPS: %d", (int)(1.f / delta_time));
      ImGui::Text("Frame Time Avg: %.4f ms", frame_stats.average * 1e3);
      ImGui::Text("Shape Draw Avg: %.4f ms", shape_stats.average * 1e3);
      ImGui::Text("GUI Draw Avg: %.4f ms", gui_stats.average * 1e3);

      ImGui::Text("Lowest FPS: %.4f", frame_stats.lowest);
      ImGui::Text("Highest FPS: %.4f", frame_stats.highest);

      if (ImGui::Button(frame_monitor.paused ? "Continue" : "Pause")) {
        frame_monitor.paused = !frame_monitor.paused;
        shape_monitor.paused = frame_monitor.paused;
        gui_monitor.paused = frame_monitor.paused;
      }

      if (ImGui::Button("Reset data")) {
        frame_monitor.reset();
        shape_monitor.reset();
        gui_monitor.reset();
      }

      if (ImPlot::BeginPlot("Times", ImVec2(-1, 0),
        ImPlotFlags_NoFrame)) {
        ImPlot::SetupAxes("Frame Index", "Frame Time (ms)",
          ImPlotAxisFlags_AutoFit,
          ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit
        );
        ImPlot::SetupAxisTicks(ImAxis_Y1, 0.0, 10.0, 11);
        frame_monitor.plot("Frame Draw Time");
        shape_monitor.plot("Shape Draw Time");
        gui_monitor.plot("GUI Draw Time");
        ImPlot::EndPlot();
      }

      ImGui::Text("Frame Draw Time: %.4f ms", delta_time * 1e3);
      ImGui::Text("Shape Draw Time: %.4f ms", shape_draw_time_s * 1e3);
      ImGui::Text("GUI Draw Time: %.4f ms", gui_draw_time_s * 1e3);

      ImGui::End();
    }

#if defined(loco_framebuffer)

#endif

		ImGui::Render();

		if (window.renderer == fan::window_t::renderer_t::opengl) {
			//glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
			//glClear(GL_COLOR_BUFFER_BIT);

			ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		}
#if defined(fan_vulkan)
		else if (window.renderer == fan::window_t::renderer_t::vulkan) {
			auto& cmd_buffer = context.vk.command_buffers[context.vk.current_frame];
			// did draw
			if (vk.image_error == (decltype(vk.image_error))-0xfff) {
				vk.image_error = VK_SUCCESS;
			}
			if (render_shapes_top == false) {
				vkCmdEndRenderPass(cmd_buffer);
			}

			ImDrawData* draw_data = ImGui::GetDrawData();
			const bool is_minimized = (draw_data->DisplaySize.x <= 0.0f || draw_data->DisplaySize.y <= 0.0f);
			if (!is_minimized) {
				context.vk.ImGuiFrameRender(vk.image_error, clear_color);
			}
		}
#endif
#endif
    gui_draw_time_s = gui_draw_timer.seconds();
	}

  struct time_monitor_t {
    static constexpr int buffer_size = 128;
    int insert_index = 0;
    int valid_samples = 0;
    f32_t running_sum = 0.0f;
    f32_t running_min = std::numeric_limits<f32_t>::max();
    f32_t running_max = std::numeric_limits<f32_t>::min();
    fan::time::timer refresh_speed{ (uint64_t)0.05e9, true };
    std::array<f32_t, buffer_size> samples{};
    bool paused = false;

    void update(f32_t value) {
      if (paused) return;
      if (!refresh_speed.finished()) return;

      f32_t old_value = (valid_samples >= buffer_size) ? samples[insert_index] : 0.0f;
      samples[insert_index] = value;

      if (valid_samples < buffer_size) {
        running_sum += value;
        valid_samples++;
      }
      else {
        running_sum += value - old_value;
      }

      running_min = std::min(running_min, value);
      running_max = std::max(running_max, value);

      insert_index = (insert_index + 1) % buffer_size;
      refresh_speed.restart();
    }

    void reset() {
      running_min = std::numeric_limits<f32_t>::max();
      running_max = std::numeric_limits<f32_t>::min();
      running_sum = 0.0f;
      insert_index = 0;
      valid_samples = 0;
      samples.fill(0.0f);
    }

    struct stats_t {
      f32_t average;
      f32_t lowest;
      f32_t highest;
    };

    stats_t calculate_stats(f32_t last_value) const {
      int sample_count = std::min(valid_samples, buffer_size);
      f32_t avg = (sample_count > 0) ? running_sum / sample_count : last_value;
      f32_t low = (running_max > 0) ? 1.0f / running_max : 0.0f;
      f32_t high = (running_min < std::numeric_limits<f32_t>::max()) ? 1.0f / running_min : 0.0f;
      return { avg, low, high };
    }

    void plot(const char* label) {
      if (valid_samples == 0) return;
      static std::array<f32_t, buffer_size> plot_data{};
      int plot_count = std::min(valid_samples, buffer_size);

      if (valid_samples >= buffer_size) {
        for (int i = 0; i < buffer_size; ++i) {
          int src_index = (insert_index + i) % buffer_size;
          plot_data[i] = samples[src_index] * 1e3f;
        }
        ImPlot::PlotLine(label, plot_data.data(), buffer_size);
      }
      else {
        for (int i = 0; i < valid_samples; ++i) {
          plot_data[i] = samples[i] * 1e3f;
        }
        ImPlot::PlotLine(label, plot_data.data(), valid_samples);
      }
    }
  };
  time_monitor_t frame_monitor;
  time_monitor_t shape_monitor;
  time_monitor_t gui_monitor;

	std::vector<std::function<void()>> draw_end_cb;
	void process_frame() {

		if (window.renderer == fan::window_t::renderer_t::opengl) {
			gl.begin_process_frame();
		}

		{
			auto it = m_update_callback.GetNodeFirst();
			while (it != m_update_callback.dst) {
				m_update_callback.StartSafeNext(it);
				m_update_callback[it](this);
				it = m_update_callback.EndSafeNext();
			}
		}

#if defined(fan_physics)
		{
			auto it = shape_physics_update_cbs.GetNodeFirst();
			while (it != shape_physics_update_cbs.dst) {
				shape_physics_update_cbs.StartSafeNext(it);
				((fan::physics::shape_physics_update_cb)shape_physics_update_cbs[it].cb)(shape_physics_update_cbs[it]);
				it = shape_physics_update_cbs.EndSafeNext();
			}
		}
#endif

		for (const auto& i : single_queue) {
			i();
		}

		single_queue.clear();

#if defined(fan_gui)
		ImGui::End();
#endif

		fan::graphics::g_shapes->shaper.ProcessBlockEditQueue();

#if defined(fan_vulkan)
		if (window.renderer == fan::window_t::renderer_t::vulkan) {
			vk.begin_draw();
		}
#endif

		viewport_set(0, window.get_size());

		if (render_shapes_top == false) {
			process_shapes();
			process_gui();
		}
		else {
			process_gui();
			process_shapes();
		}
		for (auto& i : draw_end_cb) {
			i();
		}
		if (window.renderer == fan::window_t::renderer_t::opengl) {
			glfwSwapBuffers(window);
		}
#if defined(fan_vulkan)
		else if (window.renderer == fan::window_t::renderer_t::vulkan) {
#if !defined(fan_gui)
			auto& cmd_buffer = context.vk.command_buffers[context.vk.current_frame];
			// did draw
			vkCmdNextSubpass(cmd_buffer, VK_SUBPASS_CONTENTS_INLINE);
			vkCmdEndRenderPass(cmd_buffer);
#endif
			VkResult err = context.vk.end_render();
			context.vk.recreate_swap_chain(&window, err);
		}
#endif
	}

	bool should_close() {
		if (window == nullptr) {
			return true;
		}
		return glfwWindowShouldClose(window);
	}

	bool process_loop(const std::function<void()>& lambda = [] {}) {
		window.handle_events();
    time = start_time.seconds();

    { // these require manual calling per frame, since key down callback gives repeat delay
      {
        auto it = m_mouse_down_callbacks.GetNodeFirst();
        while (it != m_mouse_down_callbacks.dst) {
          m_mouse_down_callbacks.StartSafeNext(it);
          for (int i = 0; i < 3; ++i) {
            if (fan::window::is_key_down(i)) {
              m_mouse_down_callbacks[it].data(fan::window::get_mouse_position());
            }
          }
          it = m_mouse_down_callbacks.EndSafeNext();
        }
      }
      {
        auto it = m_key_down_callbacks.GetNodeFirst();
        while (it != m_key_down_callbacks.dst) {
          m_key_down_callbacks.StartSafeNext(it);
          for (int i = fan::key_first; i <= fan::key_last; ++i) {
            if (fan::window::is_key_down(i)) {
              m_key_down_callbacks[it].data();
            }
          }
          it = m_key_down_callbacks.EndSafeNext();
        }
      }
    }

		if (should_close()) {
			return 1;
		}

		delta_time = window.m_delta_time;

#if defined(fan_physics)
		physics_context.begin_frame(delta_time);
#endif

#if defined(fan_gui)
		if (reload_renderer_to != (decltype(reload_renderer_to))-1) {
			switch_renderer(reload_renderer_to);
		}

		if (window.renderer == fan::window_t::renderer_t::opengl) {
			ImGui_ImplOpenGL3_NewFrame();
		}
#if defined(fan_vulkan)
		else if (window.renderer == fan::window_t::renderer_t::vulkan) {
			ImGui_ImplVulkan_NewFrame();
		}
#endif

		lighting.update(delta_time);

		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		auto& style = ImGui::GetStyle();
		ImVec4* colors = style.Colors;
		ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0));
		ImGui::PushStyleColor(ImGuiCol_DockingEmptyBg, ImVec4(0, 0, 0, 0));
		ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport());

		if (allow_docking || is_key_down(fan::key_left_control)) {
			ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_DockingEnable;
		}
		else {
			ImGui::GetIO().ConfigFlags &= ~ImGuiConfigFlags_DockingEnable;
		}

		ImGui::PopStyleColor(2);
		ImGui::SetNextWindowPos(ImVec2(0, 0));
		ImGui::SetNextWindowSize(fan::vec2(window.get_size()));

		int flags = ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoSavedSettings |
			ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoMove |
			ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBackground |
			ImGuiWindowFlags_NoResize | ImGuiDockNodeFlags_NoDockingSplit |
			ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoBringToFrontOnFocus;

		if (!enable_overlay) {
			flags |= ImGuiWindowFlags_NoNav;
		}

		ImGui::Begin("##global_renderer", 0, flags);
#endif

		lambda();

		// user can terminate from main loop
		if (should_close()) {
			return 1;
		}

		process_frame();

		return 0;
	}
	void loop(const std::function<void()>& lambda = []{}) {
		main_loop = lambda;
	g_loop:
		double delay = std::round(1.0 / target_fps * 1000.0);

		if (!timer_init) {
			uv_timer_init(fan::event::get_loop(), &timer_handle);
			timer_init = true;
		}
		if (!idle_init) {
			uv_idle_init(fan::event::get_loop(), &idle_handle);
			idle_init = true;
		}

		timer_handle.data = this;
		idle_handle.data = this;

		if (target_fps > 0) {
			start_timer();
		}
		else {
			start_idle();
		}

		uv_run(fan::event::get_loop(), UV_RUN_DEFAULT);
		if (should_close() == false) {
			goto g_loop;
		}
	}

	loco_t::camera_t open_camera(const fan::vec2& x, const fan::vec2& y) {
		loco_t::camera_t camera = camera_create();
		camera_set_ortho(camera, fan::vec2(x.x, x.y), fan::vec2(y.x, y.y));
		return camera;
	}
	loco_t::camera_t open_camera_perspective(f32_t fov = 90.0f) {
		loco_t::camera_t camera = camera_create();
		camera_set_perspective(camera, fov, window.get_size());
		return camera;
	}

	fan::graphics::viewport_t open_viewport(const fan::vec2& viewport_position, const fan::vec2& viewport_size) {
		fan::graphics::viewport_t viewport = viewport_create();
		viewport_set(viewport, viewport_position, viewport_size);
		return viewport;
	}

	void set_viewport(fan::graphics::viewport_t viewport, const fan::vec2& viewport_position, const fan::vec2& viewport_size) {
		viewport_set(viewport, viewport_position, viewport_size);
	}

	// for checking whether you set depth or no
	//struct position3_t : public fan::vec3 {
	//  using fan::vec3::vec3;
	//  using fan::vec3::operator=;
	//  position3_t& operator=(const position3_t& p) {
	//    fan::vec3::operator=(p);
	//    return *this;
	//  }
	//};

	fan::vec2 get_input_vector(
		const std::string& forward = "move_forward",
		const std::string& back = "move_back",
		const std::string& left = "move_left",
		const std::string& right = "move_right"
	) {
		fan::vec2 v(
			input_action.is_action_down(right) - input_action.is_action_down(left),
			input_action.is_action_down(back) - input_action.is_action_down(forward)
		);
		return v.length() > 0 ? v.normalized() : v;
	}


	//
	fan::vec2 transform_matrix(const fan::vec2& position) {
		fan::vec2 window_size = window.get_size();
		// not custom ortho friendly - made for -1 1
		return position / window_size * 2 - 1;
	}

	fan::vec2 screen_to_ndc(const fan::vec2& screen_pos) {
		fan::vec2 window_size = window.get_size();
		return screen_pos / window_size * 2 - 1;
	}

	fan::vec2 ndc_to_screen(const fan::vec2& ndc_position) {
		fan::vec2 window_size = window.get_size();
		fan::vec2 normalized_position = (ndc_position + 1) / 2;
		return normalized_position * window_size;
	}
	//

	void set_vsync(bool flag) {
		vsync = flag;
		// vulkan vsync is enabled by presentation mode in swap chain
		if (window.renderer == fan::window_t::renderer_t::opengl) {
			context.gl.set_vsync(&window, flag);
		}
	}
	void start_timer() {
		double delay;
		if (target_fps <= 0) {
			delay = 0;
		}
		else {
			delay = std::round(1.0 / target_fps * 1000.0);
		}
		if (delay > 0) {
			uv_timer_start(&timer_handle, [](uv_timer_t* handle) {
				loco_t* loco = static_cast<loco_t*>(handle->data);
				if (loco->process_loop(loco->main_loop)) {
					uv_timer_stop(handle);
					uv_stop(fan::event::get_loop());
				}
				}, 0, delay);
		}
	}
	static void idle_cb(uv_idle_t* handle) {
		loco_t* loco = static_cast<loco_t*>(handle->data);
		if (loco->process_loop(loco->main_loop)) {
			uv_idle_stop(handle);
			uv_stop(fan::event::get_loop());
		}
	}
	void start_idle(bool start_idle = true) {
		if (!start_idle) {
			return;
		}
		uv_idle_start(&idle_handle, idle_cb);
	}
	void update_timer_interval(bool idle = true) {
		double delay;
		if (target_fps <= 0) {
			delay = 0;
		}
		else {
			delay = std::round(1.0 / target_fps * 1000.0);
		}

		if (delay > 0) {
			if (idle_init) {
				uv_idle_stop(&idle_handle);
			}

			if (timer_enabled == false) {
				start_timer();
				timer_enabled = true;
			}
			uv_timer_set_repeat(&timer_handle, delay);
			uv_timer_again(&timer_handle);
		}
		else {
			if (timer_init) {
				uv_timer_stop(&timer_handle);
				timer_enabled = false;
			}

			if (idle_init && idle) {
				uv_idle_start(&idle_handle, idle_cb);
			}
		}
	}
	void set_target_fps(int32_t new_target_fps, bool idle = true) {
		target_fps = new_target_fps;
		update_timer_interval(idle);
	}

	fan::graphics::context_t& get_context() {
		return context;
	}

	fan::graphics::render_view_t render_view_create() {
		fan::graphics::render_view_t render_view;
		render_view.create();
		return render_view;
	}
	fan::graphics::render_view_t render_view_create(
		const fan::vec2& ortho_x, const fan::vec2& ortho_y,
		const fan::vec2& viewport_position, const fan::vec2& viewport_size
	) {
		fan::graphics::render_view_t render_view;
		render_view.create();
		render_view.set(ortho_x, ortho_y, viewport_position, viewport_size, window.get_size());
		return render_view;
	}

	fan::window::input_action_t input_action;

	fan::graphics::update_callback_t m_update_callback;

	std::vector<std::function<void()>> single_queue;

	#include "engine_images.h"

	fan::graphics::render_view_t orthographic_render_view;
	fan::graphics::render_view_t perspective_render_view;

	fan::window_t window;

  fan::graphics::shapes shapes;

	void set_window_name(const std::string& name) {
		window.set_name(name);
	}
	void set_window_icon(const fan::image::info_t& info) {
		window.set_icon(info);
	}
	void set_window_icon(const fan::graphics::image_t& image) {
		auto& image_data = image_list[image];
		auto image_pixels = image_get_pixel_data(image, image_data.image_settings.format);
		fan::image::info_t info;
		info.size = image_data.size;
		info.data = image_pixels.data();
		window.set_icon(info);
	}

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
  void update_physics() {
    physics_context.step(delta_time);
  }

  fan::physics::physics_update_cbs_t::nr_t add_physics_update(const fan::physics::physics_update_data_t& cb_data) {
    auto it = shape_physics_update_cbs.NewNodeLast();
    shape_physics_update_cbs[it] = (fan::physics::physics_update_data_t)cb_data;
    return it;
  }
  void remove_physics_update(fan::physics::physics_update_cbs_t::nr_t nr) {
    shape_physics_update_cbs.unlrec(nr);
  }
	
	fan::physics::physics_update_cbs_t shape_physics_update_cbs;
#endif

	// clears shapes after drawing, good for debug draw, not best for performance
	std::vector<fan::graphics::shapes::shape_t> immediate_render_list;
	std::unordered_map<uint32_t, fan::graphics::shapes::shape_t> static_render_list;

	fan::vec2 get_mouse_position(const camera_t& camera, const viewport_t& viewport) const {
		return fan::graphics::transform_position(get_mouse_position(), viewport, camera);
	}
	fan::vec2 get_mouse_position(const fan::graphics::render_view_t& render_view) const {
		return get_mouse_position(render_view.camera, render_view.viewport);
	}

	fan::vec2 get_mouse_position() const {
		return window.get_mouse_position();
		//return get_mouse_position(gloco->default_camera->camera, gloco->default_camera->viewport); behaving oddly
	}

	fan::vec2 translate_position(const fan::vec2& p, viewport_t viewport, camera_t camera) const {

		auto v = gloco->viewport_get(viewport);
		fan::vec2 viewport_position = v.viewport_position;
		fan::vec2 viewport_size = v.viewport_size;

		auto c = gloco->camera_get(camera);

		f32_t l = c.coordinates.left;
		f32_t r = c.coordinates.right;
		f32_t t = c.coordinates.up;
		f32_t b = c.coordinates.down;

		fan::vec2 tp = p - viewport_position;
		fan::vec2 d = viewport_size;
		tp /= d;
		tp = fan::vec2(r * tp.x - l * tp.x + l, b * tp.y - t * tp.y + t);
		return tp;
	}

  fan::vec2 translate_position(const fan::vec2& p) const {
    return translate_position(p, orthographic_render_view.viewport, orthographic_render_view.camera);
  }

  mouse_click_nr_t on_mouse_click(int button, mouse_click_callback_t cb) {
    auto nr = m_mouse_click_callbacks.NewNodeLast();
    m_mouse_click_callbacks[nr].data = [this, cb, button]() {
      if (window.key_state(button) == fan::mouse_state::press) {
        cb();
      }
    };
    return nr;
  }
  void remove_on_mouse_click(mouse_click_nr_t nr) {
    m_mouse_click_callbacks.unlrec(nr);
  }
  mouse_down_nr_t on_mouse_down(int button, mouse_down_callback_t cb) {
    auto nr = m_mouse_down_callbacks.NewNodeLast();
    m_mouse_down_callbacks[nr].data = [this, cb, button](fan::vec2 pos) {
      int state = window.key_state(button);
      if (state == fan::mouse_state::press || state == fan::mouse_state::repeat) {
        cb(pos);
      }
    };
    return nr;
  }
  void remove_on_mouse_down(mouse_down_nr_t nr) {
    m_mouse_down_callbacks.unlrec(nr);
  }
  mouse_up_nr_t on_mouse_up(int button, mouse_up_callback_t cb) {
    auto nr = m_mouse_up_callbacks.NewNodeLast();
    m_mouse_up_callbacks[nr].data = [this, cb, button]() {
      if (window.key_state(button) == fan::mouse_state::press) {
        cb();
      }
    };
    return nr;
  }
  void remove_on_mouse_up(mouse_up_nr_t nr) {
    m_mouse_up_callbacks.unlrec(nr);
  }
  mouse_move_nr_t on_mouse_move(mouse_move_callback_t cb) {
    auto nr = m_mouse_move_callbacks.NewNodeLast();
    m_mouse_move_callbacks[nr].data = std::move(cb);
    return nr;
  }
  void remove_on_mouse_move(mouse_move_nr_t nr) {
    m_mouse_move_callbacks.unlrec(nr);
  }
  key_click_nr_t on_key_click(int key, key_click_callback_t cb) {
    auto nr = m_key_click_callbacks.NewNodeLast();
    m_key_click_callbacks[nr].data = [this, cb, key]() {
      int state = window.key_state(key);
      if (state == fan::keyboard_state::press) {
        cb();
      }
    };
    return nr;
  }
  void remove_on_key_click(key_click_nr_t nr) {
    m_key_click_callbacks.unlrec(nr);
  }
  key_up_nr_t on_key_up(int key, key_up_callback_t cb) {
    auto nr = m_key_up_callbacks.NewNodeLast();
    m_key_up_callbacks[nr].data = [this, cb, key]() {
      int state = window.key_state(key);
      if (state == fan::keyboard_state::release) {
        cb();
      }
    };
    return nr;
  }
  void remove_on_key_up(key_up_nr_t nr) {
    m_key_up_callbacks.unlrec(nr);
  }
  key_down_nr_t on_key_down(int key, key_down_callback_t cb) {
    auto nr = m_key_down_callbacks.NewNodeLast();
    m_key_down_callbacks[nr].data = [this, cb, key]() {
      int state = window.key_state(key);
      if (state == fan::keyboard_state::press || state == fan::keyboard_state::repeat) {
        cb();
      }
    };
    return nr;
  }
  void remove_on_key_down(key_down_nr_t nr) {
    m_key_down_callbacks.unlrec(nr);
  }

  on_resize_nr_t on_resize(on_resize_callback_t cb) {
    auto nr = m_on_resize_callbacks.NewNodeLast();
    m_on_resize_callbacks[nr].data = std::move(cb);
    return nr;
  }
  void remove_on_resize(on_resize_nr_t nr) {
    m_on_resize_callbacks.unlrec(nr);
  }

	bool is_mouse_clicked(int button = fan::mouse_left) {
		return window.key_state(button) == (int)fan::mouse_state::press;
	}
	bool is_mouse_down(int button = fan::mouse_left) {
		int state = window.key_state(button);
		return
			state == (int)fan::mouse_state::press ||
			state == (int)fan::mouse_state::repeat;
	}
	bool is_mouse_released(int button = fan::mouse_left) {
		return window.key_state(button) == (int)fan::mouse_state::release;
	}
	fan::vec2 get_mouse_drag(int button = fan::mouse_left) {
		if (is_mouse_down(button)) {
			if (window.drag_delta_start != fan::vec2(-1)) {
				return window.get_mouse_position() - window.drag_delta_start;
			}
		}
		return fan::vec2();
	}

	bool is_key_pressed(int key) {
		return window.key_state(key) == (int)fan::mouse_state::press;
	}
	bool is_key_down(int key) {
		int state = window.key_state(key);
		return
			state == (int)fan::mouse_state::press ||
			state == (int)fan::mouse_state::repeat;
	}
	bool is_key_released(int key) {
		return window.key_state(key) == (int)fan::mouse_state::release;
	}


	// ShapeID_t must be at the beginning of fan::graphics::shapes::shape_t's memory since there are reinterpret_casts,
	// which assume that


	// pointer
	using shape_shader_locations_t = decltype(fan::graphics::shaper_t::BlockProperties_t::gl_t::locations);

	inline void shape_open(
		uint16_t shape_type,
		std::size_t sizeof_vi,
		std::size_t sizeof_ri,
		shape_shader_locations_t shape_shader_locations,
		const std::string& vertex,
		const std::string& fragment,
		fan::graphics::shaper_t::ShapeRenderDataSize_t instance_count = 1,
		bool instanced = true
	) {
		fan::graphics::shader_t shader = shader_create();

		shader_set_vertex(shader,
			read_shader(vertex)
		);

		shader_set_fragment(shader,
			read_shader(fragment)
		);

		shader_compile(shader);

		fan::graphics::shaper_t::BlockProperties_t bp;
		bp.MaxElementPerBlock = (fan::graphics::shaper_t::MaxElementPerBlock_t)fan::graphics::MaxElementPerBlock;
		bp.RenderDataSize = (decltype(fan::graphics::shaper_t::BlockProperties_t::RenderDataSize))(sizeof_vi * instance_count);
		bp.DataSize = sizeof_ri;

		if (window.renderer == fan::window_t::renderer_t::opengl) {
			std::construct_at(&bp.renderer.gl);
			fan::graphics::shaper_t::BlockProperties_t::gl_t d;
			d.locations = shape_shader_locations;
			d.shader = shader;
			d.instanced = instanced;
			bp.renderer.gl = d;
		}
#if defined(fan_vulkan)
		else if (window.renderer == fan::window_t::renderer_t::vulkan) {
			std::construct_at(&bp.renderer.vk);
			fan::graphics::shaper_t::BlockProperties_t::vk_t vk;

			// 2 for rect instance, upv
			static constexpr auto vulkan_buffer_count = 3;
			decltype(vk.shape_data.m_descriptor)::properties_t rectp;
			// image
			//uint32_t ds_offset = 3;
			auto& shaderd = *(fan::vulkan::context_t::shader_t*)gloco->context_functions.shader_get(&gloco->context.vk, shader);
			uint32_t ds_offset = 2;
			vk.shape_data.open(gloco->context.vk, 1);
			vk.shape_data.allocate(gloco->context.vk, 0xffffff);

			std::array<fan::vulkan::write_descriptor_set_t, vulkan_buffer_count> ds_properties{ {{0}} };
			{
				ds_properties[0].binding = 0;
				ds_properties[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				ds_properties[0].flags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
				ds_properties[0].range = VK_WHOLE_SIZE;
				ds_properties[0].buffer = vk.shape_data.common.memory[gloco->get_context().vk.current_frame].buffer;
				ds_properties[0].dst_binding = 0;

				ds_properties[1].binding = 1;
				ds_properties[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
				ds_properties[1].flags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
				ds_properties[1].buffer = shaderd.projection_view_block->common.memory[gloco->get_context().vk.current_frame].buffer;
				ds_properties[1].range = shaderd.projection_view_block->m_size;
				ds_properties[1].dst_binding = 1;

				VkDescriptorImageInfo imageInfo{};
				auto img = gloco->image_get(gloco->default_texture).vk;
				imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				imageInfo.imageView = img.image_view;
				imageInfo.sampler = img.sampler;

				ds_properties[2].use_image = 1;
				ds_properties[2].binding = 2;
				ds_properties[2].dst_binding = 2;
				ds_properties[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				ds_properties[2].flags = VK_SHADER_STAGE_FRAGMENT_BIT;
				for (uint32_t i = 0; i < fan::vulkan::max_textures; ++i) {
					ds_properties[ds_offset].image_infos[i] = imageInfo;
				}

				//imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				//imageInfo.imageView = gloco->get_context().vk.postProcessedColorImageViews[0].image_view;
				//imageInfo.sampler = sampler;

				//imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				//ds_properties[ds_offset + 1].use_image = 1;
				//ds_properties[ds_offset + 1].binding = 4;
				//ds_properties[ds_offset + 1].dst_binding = 4;
				//ds_properties[ds_offset + 1].type = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
				//ds_properties[ds_offset + 1].flags = VK_SHADER_STAGE_FRAGMENT_BIT;
				//for (uint32_t i = 0; i < fan::vulkan::max_textures; ++i) {
				//  ds_properties[ds_offset + 1].image_infos[i] = imageInfo;
				//}
			}

			vk.shape_data.open_descriptors(gloco->context.vk, { ds_properties.begin(), ds_properties.end() });
			vk.shape_data.m_descriptor.update(context.vk, 3, 0);
			fan::vulkan::context_t::pipeline_t p;
			fan::vulkan::context_t::pipeline_t::properties_t pipe_p;
			VkPipelineColorBlendAttachmentState attachment = fan::vulkan::get_default_color_blend();
			pipe_p.color_blend_attachment_count = 1;
			pipe_p.color_blend_attachment = &attachment;
			pipe_p.shader = shader;
			pipe_p.descriptor_layout = &vk.shape_data.m_descriptor.m_layout;
			pipe_p.descriptor_layout_count = /*vulkan_buffer_count*/1;
			pipe_p.push_constants_size = sizeof(fan::vulkan::context_t::push_constants_t);
			p.open(context.vk, pipe_p);
			vk.pipeline = p;
			bp.renderer.vk = vk;
		}
#endif

		fan::graphics::g_shapes->shaper.SetShapeType(shape_type, bp);
	}

#if defined(loco_sprite)
	fan::graphics::shader_t get_sprite_vertex_shader(const std::string& fragment) {
		if (get_renderer() == fan::window_t::renderer_t::opengl) {
			fan::graphics::shader_t shader = shader_create();
			shader_set_vertex(
				shader,
				loco_t::read_shader("shaders/opengl/2D/objects/sprite.vs")
			);
			shader_set_fragment(shader, fragment);
			if (!shader_compile(shader)) {
				shader_erase(shader);
				shader.sic();
			}
			return shader;
		}
		else {
			fan::print("todo");
		}
		return {};
	}
#endif

	//#if defined(loco_texture_pack)
	//#endif

	fan::color clear_color = {
		/*0.10f, 0.10f, 0.131f, 1.f */
		0.f, 0.f, 0.f, 1.f
	};

	fan::graphics::lighting_t lighting;

	void set_current_directory(const std::string& new_directory) {
		current_directory = new_directory;
	}
	std::string current_directory = "./";

	//gui
#if defined(fan_gui)

	void toggle_console() {
		render_console = !render_console;
	}
	void toggle_console(bool active) {
		render_console = active;
	}

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

  fan::graphics::image_load_properties_t default_noise_image_properties() {
    fan::graphics::image_load_properties_t lp;
    lp.format = fan::graphics::image_format::rgb_unorm;
    lp.internal_format = fan::graphics::image_format::rgb_unorm;
    lp.min_filter = fan::graphics::image_filter::linear;
    lp.mag_filter = fan::graphics::image_filter::linear;
    lp.visual_output = fan::graphics::image_sampler_address_mode::mirrored_repeat;
    return lp;
  }
  fan::graphics::image_t create_noise_image(const fan::vec2& size,
    int seed = fan::random::value_i64(0, ((uint32_t)-1) / 2)) {
    fan::noise_t noise(seed);
    auto data = noise.generate_data(size);

    auto lp = default_noise_image_properties();
    fan::image::info_t ii{ (void*)data.data(), size, 3 };
    return image_load(ii, lp);
  }
  fan::graphics::image_t create_noise_image(const fan::vec2& size,
    const std::vector<uint8_t>& data) {
    auto lp = default_noise_image_properties();
    fan::image::info_t ii{ (void*)data.data(), size, 3 };
    return image_load(ii, lp);
  }

	fan::vec2 convert_mouse_to_ndc(const fan::vec2& mouse_position) const {
		return fan::math::convert_position_ndc(mouse_position, gloco->window.get_size());
	}
	fan::vec2 convert_mouse_to_ndc() const {
		return fan::math::convert_position_ndc(gloco->get_mouse_position(), gloco->window.get_size());
	}
	fan::ray3_t convert_mouse_to_ray(const fan::vec3& camera_position, const fan::mat4& projection, const fan::mat4& view) {
		return fan::math::convert_position_to_ray(get_mouse_position(), window.get_size(), camera_position, projection, view);
	}
	fan::ray3_t convert_mouse_to_ray(const fan::mat4& projection, const fan::mat4& view) {
		return fan::math::convert_position_to_ray(get_mouse_position(), window.get_size(), camera_get_position(perspective_render_view.camera), projection, view);
	}

#if defined(loco_cuda)

	struct cuda_textures_t {

		cuda_textures_t() {
			inited = false;
		}
		~cuda_textures_t() {
		}
		void close(loco_t* loco, fan::graphics::shapes::shape_t& cid) {
			loco_t::universal_image_renderer_t::ri_t& ri = *(loco_t::universal_image_renderer_t::ri_t*)cid.GetData(fan::graphics::g_shapes->shaper);
			uint8_t image_amount = fan::graphics::get_channel_amount(ri.format);
			for (uint32_t i = 0; i < image_amount; ++i) {
				wresources[i].close();
				if (ri.images_rest[i] != loco->default_texture) {
					gloco->image_unload(ri.images_rest[i]);
				}
				ri.images_rest[i] = loco->default_texture;
			}
			inited = false;
		}

		void resize(loco_t* loco, fan::graphics::shapes::shape_t& id, uint8_t format, fan::vec2ui size) {
			auto vi_image = id.get_image();
			if (vi_image.iic() || vi_image == loco->default_texture) {
				id.reload(format, size);
			}

			auto& ri = *(universal_image_renderer_t::ri_t*)id.GetData(loco->shaper);

			if (inited == false) {
				id.reload(format, size);
				vi_image = id.get_image();

				uint8_t image_amount = fan::graphics::get_channel_amount(format);

				for (uint32_t i = 0; i < image_amount; ++i) {
					if (i == 0) {
						wresources[i].open(gloco->image_get_handle(vi_image));
					}
					else {
						wresources[i].open(gloco->image_get_handle(ri.images_rest[i - 1]));
					}
				}
				inited = true;
			}
			else {
				if (gloco->image_get_data(vi_image).size == size) {
					return;
				}

				for (uint32_t i = 0; i < fan::graphics::get_channel_amount(ri.format); ++i) {
					wresources[i].close();
				}

				id.reload(format, size);
				vi_image = id.get_image();

				ri = *(universal_image_renderer_t::ri_t*)id.GetData(loco->shaper);

				uint8_t image_amount = fan::graphics::get_channel_amount(format);

				// Re-register with CUDA after successful reload
				for (uint32_t i = 0; i < image_amount; ++i) {
					if (i == 0) {
						wresources[i].open(gloco->image_get_handle(vi_image));
					}
					else {
						wresources[i].open(gloco->image_get_handle(ri.images_rest[i - 1]));
					}
				}
			}
		}

		cudaArray_t& get_array(uint32_t index_t) {
			return wresources[index_t].cuda_array;
		}

		struct graphics_resource_t {
			void open(int texture_id) {
				fan::cuda::check_error(cudaGraphicsGLRegisterImage(&resource, texture_id, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone));
				map();
			}
			void close() {
				if (resource == nullptr) {
					return;
				}
				unmap();
				fan::cuda::check_error(cudaGraphicsUnregisterResource(resource));
				resource = nullptr;
			}
			void map() {
				fan::cuda::check_error(cudaGraphicsMapResources(1, &resource, 0));
				fan::cuda::check_error(cudaGraphicsSubResourceGetMappedArray(&cuda_array, resource, 0, 0));
			}
			void unmap() {
				fan::cuda::check_error(cudaGraphicsUnmapResources(1, &resource));
				//fan::cuda::check_error(cudaGraphicsResourceSetMapFlags(resource, 0));
			}
			//void reload(int texture_id) {
			//  close();
			//  open(texture_id);
			//}
			cudaGraphicsResource_t resource = nullptr;
			cudaArray_t cuda_array = nullptr;
		};

		bool inited = false;
		graphics_resource_t wresources[4];
	};

#endif

#if defined(fan_audio)
	fan::system_audio_t system_audio;
	fan::audio_t audio;
#endif
	void camera_move_to(const fan::graphics::shapes::shape_t& shape, const fan::graphics::render_view_t& render_view) {
		camera_set_position(
			orthographic_render_view.camera,
			shape.get_position()
		);
	}
	void camera_move_to(const fan::graphics::shapes::shape_t& shape) {
		camera_move_to(shape, orthographic_render_view);
	}

	void camera_move_to_smooth(const fan::graphics::shapes::shape_t& shape, const fan::graphics::render_view_t& render_view) {
		fan::vec2 current = camera_get_position(render_view.camera);
		fan::vec2 target = shape.get_position();
		f32_t t = 0.1f;
		camera_set_position(
			orthographic_render_view.camera,
			current.lerp(target, t)
		);
	}

	void camera_move_to_smooth(const fan::graphics::shapes::shape_t& shape) {
		camera_move_to_smooth(shape, orthographic_render_view);
	}

	bool shader_update_fragment(uint16_t shape_type, const std::string& fragment) {
		auto shader_nr = shader_get_nr(shape_type);
		auto shader_data = shader_get_data(shape_type);
		gloco->shader_set_vertex(shader_nr, shader_data.svertex);
		gloco->shader_set_fragment(shader_nr, fragment);
		return gloco->shader_compile(shader_nr);
	}
};

#include <fan/graphics/collider.h>

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

		namespace gui {
			void process_loop() {
				auto it = gloco->gui_draw_cb.GetNodeFirst();
				while (it != gloco->gui_draw_cb.dst) {
					gloco->gui_draw_cb.StartSafeNext(it);
					gloco->gui_draw_cb[it]();
					it = gloco->gui_draw_cb.EndSafeNext();
				}
			}
			// fan_track_allocations() must be called in global scope before calling this function
			void render_allocations_plot() {
#if defined(fan_std23)
				static std::vector<f32_t> allocation_sizes;
				static std::vector<fan::heap_profiler_t::memory_data_t> allocations;

				allocation_sizes.clear();
				allocations.clear();


				f32_t max_y = 0;
				for (const auto& entry : fan::heap_profiler_t::instance().memory_map) {
					f32_t v = (f32_t)entry.second.n / (1024 * 1024);
					/*if (v < 0.001) {
						continue;
					}*/
					allocation_sizes.push_back(v);
					max_y = std::max(max_y, v);
					allocations.push_back(entry.second);
				}
				static std::stacktrace stack;
				if (allocation_sizes.size() && ImPlot::BeginPlot("Memory Allocations", ImGui::GetWindowSize(), ImPlotFlags_NoFrame | ImPlotFlags_NoLegend)) {
					f32_t max_allocation = *std::max_element(allocation_sizes.begin(), allocation_sizes.end());
					ImPlot::SetupAxis(ImAxis_Y1, "Memory (MB)");
					ImPlot::SetupAxisLimits(ImAxis_Y1, 0, max_y);
					ImPlot::SetupAxis(ImAxis_X1, "Allocations");
					ImPlot::SetupAxisLimits(ImAxis_X1, 0, static_cast<double>(allocation_sizes.size()));

					ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.25f);
					ImPlot::PlotBars("Allocations", allocation_sizes.data(), allocation_sizes.size());
					//if (ImPlot::IsPlotHovered()) {
					//  fan::print("A");
					//}
					ImPlot::PopStyleVar();

					bool hovered = false;
					if (ImPlot::IsPlotHovered()) {
						ImPlotPoint mouse = ImPlot::GetPlotMousePos();
						f32_t half_width = 0.25;
						//mouse.x             = ImPlot::RoundTime(ImPlotTime::FromDouble(mouse.x), ImPlotTimeUnit_Day).ToDouble();
						mouse.x = (int)mouse.x;
						f32_t  tool_l = ImPlot::PlotToPixels(mouse.x - half_width * 1.5, mouse.y).x;
						f32_t  tool_r = ImPlot::PlotToPixels(mouse.x + half_width * 1.5, mouse.y).x;
						f32_t  tool_t = ImPlot::GetPlotPos().y;
						f32_t  tool_b = tool_t + ImPlot::GetPlotSize().y;
						ImPlot::PushPlotClipRect();
						auto draw_list = ImGui::GetWindowDrawList();
						draw_list->AddRectFilled(ImVec2(tool_l, tool_t), ImVec2(tool_r, tool_b), IM_COL32(128, 128, 128, 64));
						ImPlot::PopPlotClipRect();

						if (mouse.x >= 0 && mouse.x < allocation_sizes.size()) {
							if (ImGui::IsMouseClicked(0)) {
								ImGui::OpenPopup("view stack");
							}
							stack = allocations[(int)mouse.x].line_data;
							hovered = true;
						}
					}
					if (hovered) {
						ImGui::BeginTooltip();
						std::ostringstream oss;
						oss << stack;
						std::string stack_str = oss.str();
						std::string final_str;
						std::size_t pos = 0;
						while (true) {
							auto end = stack_str.find(')', pos);
							if (end != std::string::npos) {
								end += 1;
								auto begin = stack_str.rfind('\\', end);
								if (begin != std::string::npos) {
									begin += 1;
									final_str += stack_str.substr(begin, end - begin);
									final_str += "\n";
									pos = end + 1;
								}
								else {
									break;
								}
							}
							else {
								break;
							}
						}
						ImGui::TextUnformatted(final_str.c_str());
						ImGui::EndTooltip();
					}
					if (ImGui::BeginPopup("view stack", ImGuiWindowFlags_AlwaysHorizontalScrollbar)) {
						std::ostringstream oss;
						oss << stack;
						ImGui::TextUnformatted(oss.str().c_str());
						ImGui::EndPopup();
					}
					ImPlot::EndPlot();
				}

#else
				ImGui::Text("std::stacktrace not supported");
#endif
			}
		}
	}
}
#endif

inline uint32_t fan::graphics::get_draw_mode(uint8_t internal_draw_mode) {
	if (gloco->get_renderer() == fan::window_t::renderer_t::opengl) {
#if defined(loco_opengl)
		return fan::opengl::core::get_draw_mode(internal_draw_mode);
#endif
	}
	else if (gloco->get_renderer() == fan::window_t::renderer_t::vulkan) {
#if defined(fan_vulkan)
		return fan::vulkan::core::get_draw_mode(internal_draw_mode);
#endif
	}
#if fan_debug >= fan_debug_medium
	fan::throw_error("invalid get");
#endif
	return -1;
}

export namespace fan::graphics {
  using engine_t = loco_t;
}