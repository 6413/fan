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

//#include <fan/graphics/algorithm/FastNoiseLite.h>

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
export import fan.types.color;
export import fan.random;

export import fan.io.file;
export import fan.types.fstring;
#if defined(fan_physics)
	import fan.physics.b2_integration;
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



export import fan.physics.collision.rectangle;

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

export struct loco_t;

#if defined(fan_physics)
namespace fan {
	namespace graphics {
		void open_bcol();
		void close_bcol();
	}
}
#endif

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
		struct context_shader_t {
			context_shader_t() {}
			~context_shader_t() {}
			union {
				fan::opengl::context_t::shader_t gl;
#if defined(fan_vulkan)
				fan::vulkan::context_t::shader_t vk;
#endif
			};
		};
		struct context_image_t {
			context_image_t() {}
			~context_image_t() {}
			union {
				fan::opengl::context_t::image_t gl;
#if defined(fan_vulkan)
				fan::vulkan::context_t::image_t vk; // note vk::image_t uses vector 
#endif
			};
		};
		struct context_t {
			context_t() {}
			~context_t() {}
			union {
				fan::opengl::context_t gl;
#if defined(fan_vulkan)
				fan::vulkan::context_t vk;
#endif
			};
		};
	}
}

namespace fan {
	template <bool cond>
	struct type_or_uint8_t {
		template <typename T>
		using d = std::conditional_t<cond, T, uint8_t>;
	};
}

#if defined(fan_audio)
export namespace fan {
	namespace audio {
		using sound_play_id_t = fan::audio_t::SoundPlayID_t;

		struct piece_t : fan::audio_t::piece_t {
			using fan::audio_t::piece_t::piece_t;

			piece_t();
			piece_t(const fan::audio_t::piece_t& piece);
			piece_t(
				const std::string& path,
				fan::audio_t::PieceFlag::t flags = 0,
				const std::source_location& callers_path = std::source_location::current()
			);

			operator fan::audio_t::piece_t& ();

			piece_t open_piece(
				const std::string& path,
				fan::audio_t::PieceFlag::t flags = 0,
				const std::source_location& callers_path = std::source_location::current()
			);

			bool is_valid();

			sound_play_id_t play(uint32_t group_id = 0, bool loop = false);
			void stop(sound_play_id_t id);
			void resume(uint32_t group_id = 0);
			void pause(uint32_t group_id = 0);

			f32_t get_volume();
			void set_volume(f32_t volume);
		};

		piece_t piece_invalid;

		piece_t open_piece(
			const std::string& path,
			fan::audio_t::PieceFlag::t flags = 0,
			const std::source_location& callers_path = std::source_location::current()
		);
		bool is_piece_valid(piece_t piece);
		sound_play_id_t play(piece_t piece, uint32_t group_id = 0, bool loop = false);
		void stop(sound_play_id_t id);
		void resume(uint32_t group_id = 0);
		void pause(uint32_t group_id = 0);
		f32_t get_volume();
		void set_volume(f32_t volume);
	}
}
#endif

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

	using renderer_t = fan::window_t::renderer_t;
	uint8_t get_renderer() {
		return window.renderer;
	}

	using shader_t = fan::graphics::shader_nr_t;
	using image_t = fan::graphics::image_nr_t;
	using camera_t = fan::graphics::camera_nr_t;
	using viewport_t = fan::graphics::viewport_nr_t;
	using image_load_properties_t = fan::graphics::image_load_properties_t;

	using image_sampler_address_mode = fan::graphics::image_sampler_address_mode;

	struct shape_t;

	fan::graphics::shader_nr_t shader_create() {
		return context_functions.shader_create(&context);
	}
	// warning does deep copy, addresses can die
	fan::graphics::context_shader_t shader_get(fan::graphics::shader_nr_t nr) {
		fan::graphics::context_shader_t context_shader;
		if (window.renderer == renderer_t::opengl) {
			context_shader.gl = *(fan::opengl::context_t::shader_t*)context_functions.shader_get(&context, nr);
		}
#if defined(fan_vulkan)
		else if (window.renderer == renderer_t::vulkan) {
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
		if (window.renderer == renderer_t::opengl) {
			context.gl.shader_set_value(nr, name, val);
		}
		else if (window.renderer == renderer_t::vulkan) {
			fan::throw_error("todo");
		}
	}
	void shader_set_camera(shader_t nr, camera_t camera_nr) {
		if (window.renderer == renderer_t::opengl) {
			context.gl.shader_set_camera(nr, camera_nr);
		}
#if defined(fan_vulkan)
		else if (window.renderer == renderer_t::vulkan) {
			fan::throw_error("todo");
		}
#endif
	}

	fan::graphics::shader_nr_t shader_get_nr(uint16_t shape_type) {
		return gloco->shaper.GetShader(shape_type);
	}
	auto& shader_get_data(uint16_t shape_type) {
		return shader_list[shader_get_nr(shape_type)];
	}

	fan::graphics::camera_list_t camera_list;
	fan::graphics::shader_list_t shader_list;
	fan::graphics::image_list_t image_list;
	fan::graphics::viewport_list_t viewport_list;

	std::vector<uint8_t> image_get_pixel_data(fan::graphics::image_nr_t nr, int image_format, fan::vec2 uvp = 0, fan::vec2 uvs = 1) {
		if (window.renderer == renderer_t::opengl) {
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
		if (window.renderer == renderer_t::opengl) {
			img.gl = *(fan::opengl::context_t::image_t*)context_functions.image_get(&context, nr);
		}
#if defined(fan_vulkan)
		else if (window.renderer == renderer_t::vulkan) {
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

	void viewport_zero(fan::graphics::viewport_nr_t nr) {
		context_functions.viewport_zero(&context, nr);
	}

	bool inside(fan::graphics::viewport_nr_t nr, const fan::vec2& position) {
		return context_functions.viewport_inside(&context, nr, position);
	}

	bool inside_wir(fan::graphics::viewport_nr_t nr, const fan::vec2& position) {
		return context_functions.viewport_inside_wir(&context, nr, position);
	}

	struct render_view_t;

	bool inside(const loco_t::render_view_t& render_view, const fan::vec2& position) const {
		fan::vec2 tp = translate_position(position, render_view.viewport, render_view.camera);

		auto c = gloco->camera_get(render_view.camera);
		f32_t l = c.coordinates.left;
		f32_t r = c.coordinates.right;
		f32_t t = c.coordinates.up;
		f32_t b = c.coordinates.down;

		return tp.x >= l && tp.x <= r &&
					 tp.y >= t && tp.y <= b;
	}

	bool is_mouse_inside(const loco_t::render_view_t& render_view) const {
		return inside(render_view, get_mouse_position());
	}

	fan::graphics::context_functions_t context_functions;
	fan::graphics::context_t context;


#include <fan/texture_pack/tp0.h>

	static std::string read_shader(const std::string& path, const std::source_location& callers_path = std::source_location::current()) {
		std::string code;
		fan::io::file::read(fan::io::file::find_relative_path(path, callers_path), &code);
		return code;
	}

	static uint8_t* A_resize(void* ptr, uintptr_t size) {
		if (ptr) {
			if (size) {
				void* rptr = (void*)__generic_realloc(ptr, size);
				if (rptr == 0) {
					fan::throw_error_impl();
				}
				return (uint8_t*)rptr;
			}
			else {
				__generic_free(ptr);
				return 0;
			}
		}
		else {
			if (size) {
				void* rptr = (void*)__generic_malloc(size);
				if (rptr == 0) {
					fan::throw_error_impl();
				}
				return (uint8_t*)rptr;
			}
			else {
				return 0;
			}
		}
	}

	static constexpr uint32_t MaxElementPerBlock = 0x100;

	struct shape_gl_init_t {
		std::pair<int, const char*> index;
		uint32_t size;
		uint32_t type; // for example GL_FLOAT
		uint32_t stride;
		void* pointer;
	};

#define shaper_set_MaxMaxElementPerBlock 0x100
#define shaper_set_fan 1
	// sizeof(image_t) == 2
	static_assert(sizeof(loco_t::image_t) == 2, "update shaper_set_MaxKeySize");
#define shaper_set_MaxKeySize 2 * 30
	/*
	* void _ShapeTypeChange(
		ShapeTypeIndex_t sti,
		KeyPackSize_t keypack_size,
		uint8_t *keypack,
		MaxElementPerBlock_t element_count,
		const void *old_renderdata,
		const void *old_data,
		void *new_renderdata,
		void *new_data
	){
	*/
	// will die if renderer has different sizes of structs
#define shaper_set_ShapeTypeChange \
		__builtin_memcpy(new_renderdata, old_renderdata, element_count * get_loco()->shaper.GetRenderDataSize(sti)); \
		__builtin_memcpy(new_data, old_data, element_count * get_loco()->shaper.GetDataSize(sti));
#include <fan/graphics/shaper.h>

	static void shaper_deep_copy(loco_t::shape_t* dst, const loco_t::shape_t* const src, loco_t::shaper_t::ShapeTypeIndex_t sti) {
		// alloc can be avoided inside switch
		uint8_t* KeyPack = new uint8_t[gloco->shaper.GetKeysSize(*src)];
		gloco->shaper.WriteKeys(*src, KeyPack);

		auto _vi = src->GetRenderData(gloco->shaper);
		auto vlen = gloco->shaper.GetRenderDataSize(sti);
		uint8_t* vi = new uint8_t[vlen];
		std::memcpy(vi, _vi, vlen);

		auto _ri = src->GetData(gloco->shaper);
		auto rlen = gloco->shaper.GetDataSize(sti);

		uint8_t* ri = new uint8_t[rlen];
		std::memcpy(ri, _ri, rlen);

		if (sti == loco_t::shape_type_t::sprite) {
			if (((loco_t::sprite_t::ri_t*)_ri)->sprite_sheet_data.frame_update_nr) {
				((loco_t::sprite_t::ri_t*)ri)->sprite_sheet_data.frame_update_nr = gloco->m_update_callback.NewNodeLast(); // since hard copy, let it leak
			}
		}

		*dst = gloco->shaper.add(
			sti,
			KeyPack,
			gloco->shaper.GetKeysSize(*src),
			vi,
			ri
		);

#if defined(debug_shape_t)
		fan::print("+", NRI);
#endif

		delete[] KeyPack;
		delete[] vi;
		delete[] ri;
	}

	template<
		typename... Ts,
		uintptr_t s = (sizeof(Ts) + ...)
	>static constexpr shaper_t::ShapeID_t shape_add(
		shaper_t::ShapeTypeIndex_t sti,
		const auto& rd,
		const auto& d,
		Ts... args
	) {
		struct structarr_t {
			uint8_t p[s];
			uint8_t& operator[](uintptr_t i) {
				return p[i];
			}
		};
		structarr_t a;
		uintptr_t i = 0;
		([&](auto arg) {
			__builtin_memcpy(&a[i], &arg, sizeof(arg));
			i += sizeof(arg);
			}(args), ...);

		constexpr uintptr_t count = (!!(sizeof(Ts) + 1) + ...);
		static_assert(count % 2 == 0);
		constexpr uintptr_t last_sizeof = (static_cast<uintptr_t>(0), ..., sizeof(Ts));
		uintptr_t LastKeyOffset = s - last_sizeof - 1;
		gloco->shaper.PrepareKeysForAdd(&a, LastKeyOffset);
		return gloco->shaper.add(sti, &a, s, &rd, &d);
	}

	// unsafe
	//loco_t(const loco_t&) = delete;
	//loco_t& operator=(const loco_t&) = delete;
	//loco_t(loco_t&&) = delete;
	//loco_t& operator=(loco_t&&) = delete;

	#if defined(fan_3D)
		#define IF_FAN_3D(X) X(rectangle3d) X(line3d)
	#else
		#define IF_FAN_3D(X)
	#endif

	#define TO_ENUM(x) x,
	#define TO_STRING(x) #x,

	#define GEN_SHAPES(X, SKIP) \
		X(sprite) X(text) SKIP(X(hitbox)) SKIP(X(mark)) X(line) X(rectangle) \
		X(light) X(unlit_sprite) X(circle) X(capsule) X(polygon) X(grid) \
		X(vfi) X(particles) X(universal_image_renderer) X(gradient) \
		SKIP(X(light_end)) X(shader_shape) IF_FAN_3D(X) X(shadow)

	#define GEN_SHAPES_SKIP_ENUM(x) x
	#define GEN_SHAPES_SKIP_STRING(x) x

	struct shape_type_t {
		enum {
			invalid = -1,
			GEN_SHAPES(TO_ENUM, GEN_SHAPES_SKIP_ENUM)
			last
		};
	};
	static constexpr const char* shape_names[] = {
		GEN_SHAPES(TO_STRING, GEN_SHAPES_SKIP_STRING)
	};

	#undef TO_ENUM
	#undef TO_STRING
	#undef GEN_SHAPES_SKIP_ENUM
	#undef GEN_SHAPES_SKIP_STRING


	struct kp {
		enum {
			light,
			common,
			vfi,
			texture,
		};
	};

#if defined(fan_json)
	fan::json image_to_json(const auto& image) {
		fan::json image_json;
		if (image.iic()) {
			return image_json;
		}

		auto shape_data = image_list[image];
		if (shape_data.image_path.size()) {
			image_json["image_path"] = shape_data.image_path;
		}
		else {
			return image_json;
		}

		auto lp = image_get_settings(image);
		fan::graphics::image_load_properties_t defaults;
		if (lp.visual_output != defaults.visual_output) {
			image_json["image_visual_output"] = lp.visual_output;
		}
		if (lp.format != defaults.format) {
			image_json["image_format"] = lp.format;
		}
		if (lp.type != defaults.type) {
			image_json["image_type"] = lp.type;
		}
		if (lp.min_filter != defaults.min_filter) {
			image_json["image_min_filter"] = lp.min_filter;
		}
		if (lp.mag_filter != defaults.mag_filter) {
			image_json["image_mag_filter"] = lp.mag_filter;
		}

		return image_json;
	}
	loco_t::image_t json_to_image(const fan::json& image_json) {
		if (!image_json.contains("image_path")) {
			return default_texture;
		}

		std::string path = image_json["image_path"];

		if (!fan::io::file::exists(path)) {
			return default_texture;
		}

		fan::graphics::image_load_properties_t lp;

		if (image_json.contains("image_visual_output")) {
			lp.visual_output = image_json["image_visual_output"];
		}
		if (image_json.contains("image_format")) {
			lp.format = image_json["image_format"];
		}
		if (image_json.contains("image_type")) {
			lp.type = image_json["image_type"];
		}
		if (image_json.contains("image_min_filter")) {
			lp.min_filter = image_json["image_min_filter"];
		}
		if (image_json.contains("image_mag_filter")) {
			lp.mag_filter = image_json["image_mag_filter"];
		}

		fan::graphics::image_nr_t image = image_load(path, lp);
		image_list[image].image_path = path;
		return image;
	}
#endif
	
	//-----------------------sprite sheet animations-----------------------

	struct sprite_sheet_animation_t {
		struct image_t {
			loco_t::image_t image = gloco->default_texture;
			int hframes = 1, vframes = 1;
		#if defined(fan_json)
			operator fan::json() const {
				fan::json j;
				image_t defaults;
				if (hframes != defaults.hframes) {
					j["hframes"] = hframes;
				}
				if (hframes != defaults.vframes) {
					j["vframes"] = vframes;
				}
				j.update(gloco->image_to_json(image), true);
				return j;
			}

			sprite_sheet_animation_t::image_t& operator=(const fan::json& j) {
				image = gloco->json_to_image(j);
				if (j.contains("hframes")) {
					hframes = j.at("hframes");
				}
				if (j.contains("vframes")) {
					vframes = j.at("vframes");
				}
				return *this;
			}
		#endif
		};

		std::vector<int> selected_frames;
		std::vector<sprite_sheet_animation_t::image_t> images;
		std::string name;
		int fps = 15;
	};
	struct animation_nr_t {
		animation_nr_t() = default;
		animation_nr_t(uint32_t id) {
			this->id = id;
		}
		operator uint32_t() const {
			return id;
		}
		operator bool() const {
			return id != (decltype(id))-1;
		}
		animation_nr_t operator++(int) {
			animation_nr_t temp(*this);
			++id;
			return temp;
		}
		bool operator==(const animation_nr_t& other) const {
			return id == other.id;
		}

		bool operator!=(const animation_nr_t& other) const {
			return id != other.id;
		}
		uint32_t id = -1;
	};
	using animation_shape_nr_t = animation_nr_t;
	struct animation_nr_hash_t {
		size_t operator()(const animation_nr_t& anim_nr) const noexcept {
			return std::hash<uint32_t>()(anim_nr.id);
		}
	};
	struct animation_pair_hash_t {
		std::size_t operator()(const std::pair<animation_nr_t, std::string>& p) const noexcept {
			std::size_t h1 = animation_nr_hash_t{}(p.first);
			std::size_t h2 = std::hash<std::string>{}(p.second);
			return h1 ^ (h2 << 1);
		}
	};

	sprite_sheet_animation_t& get_sprite_sheet_animation(animation_nr_t nr) {
		auto found_anim = all_animations.find(nr);
		if (found_anim == all_animations.end()) {
			fan::throw_error("animation not found");
		}
		return found_anim->second;
	}
	sprite_sheet_animation_t& get_sprite_sheet_animation(animation_nr_t shape_animation_id, const std::string& anim_name) {
		auto found = shape_animation_lookup_table.find(std::make_pair(shape_animation_id, anim_name));
		if (found == shape_animation_lookup_table.end()) {
			fan::throw_error("Failed to find sprite sheet animation:" + anim_name);
		}
		return get_sprite_sheet_animation(found->second);
	}
	auto& get_sprite_sheet_shape_animation(animation_nr_t shape_animation_id) {
		auto found = shape_animations.find(shape_animation_id);
		if (found == shape_animations.end()) {
			fan::throw_error("Failed to find sprite sheet animation:" + std::to_string((uint32_t)shape_animation_id));
		}
		return found->second;
	}
	void rename_sprite_sheet_shape_animation(animation_nr_t shape_animation_id, const std::string& old_name, const std::string& new_name) {
		auto& previous_anims = shape_animations[shape_animation_id];
		auto found = std::find_if(previous_anims.begin(), previous_anims.end(), [old_name] (const animation_nr_t nr) {
			auto found = gloco->all_animations.find(nr);
			if (found == gloco->all_animations.end()) {
				fan::throw_error("animation nr expired (bug)");
			}
			return found->second.name == old_name;
		});
		if (found == previous_anims.end()) {
			fan::throw_error("animation:" + old_name, ", not found");
		}
		animation_nr_t previous_anim_nr = *found;
		auto prev_found = all_animations.find(previous_anim_nr);
		if (prev_found == all_animations.end()) {
			fan::throw_error("animation nr expried (bug)");
		}
		auto& previous_anim = prev_found->second;
		{
			auto found = gloco->shape_animation_lookup_table.find(std::make_pair(shape_animation_id, previous_anim.name));
			if (found != gloco->shape_animation_lookup_table.end()) {
				gloco->shape_animation_lookup_table.erase(found);
			}
		}
		previous_anim.name = new_name;
		shape_animation_lookup_table[std::make_pair(shape_animation_id, new_name)] = previous_anim_nr;
	}

	// adds animation to shape collection
	animation_nr_t add_sprite_sheet_shape_animation(animation_nr_t new_anim) {
		shape_animations[shape_animation_counter].emplace_back(new_anim);
		return shape_animation_counter++;
	}

	animation_nr_t add_existing_sprite_sheet_shape_animation(animation_nr_t existing_anim, animation_nr_t shape_animation_id, const sprite_sheet_animation_t& new_anim) {
		animation_nr_t new_anim_nr = existing_anim;
		all_animations[existing_anim] = new_anim;
		// if shape_animation_id is invalid
		if (!shape_animation_id) {
			shape_animation_id = add_sprite_sheet_shape_animation(new_anim_nr);
			shape_animation_lookup_table[std::make_pair(shape_animation_id, new_anim.name)] = new_anim_nr;
			return shape_animation_id;
		}
		shape_animation_lookup_table[std::make_pair(shape_animation_id, new_anim.name)] = new_anim_nr;
		auto found = shape_animations.find(shape_animation_id);
		if (found == shape_animations.end()) {
			fan::throw_error("add_sprite_sheet_shape_animation:given shape animation id not found");
		}
		found->second.emplace_back(new_anim_nr);
		return shape_animation_id;
	}

	// returns unique key to access list of animation keys
	animation_nr_t add_sprite_sheet_shape_animation(animation_nr_t shape_animation_id, const sprite_sheet_animation_t& new_anim) {
		animation_nr_t new_anim_nr = all_animations_counter++;
		return add_existing_sprite_sheet_shape_animation(new_anim_nr, shape_animation_id, new_anim);
	}

#if defined(fan_json)
	fan::json sprite_sheet_serialize() {
		fan::json result = fan::json::object();
		fan::json animations_arr = fan::json::array();

		for (auto& anim : all_animations) {
			fan::json ss;
			ss["name"] = anim.second.name;
			ss["selected_frames"] = anim.second.selected_frames;
			ss["fps"] = anim.second.fps;
			ss["id"] = anim.first.id;

			if (!anim.second.images.empty()) {
				fan::json images_arr = fan::json::array();
				for (const auto& img : anim.second.images) {
					images_arr.push_back(img);
				}
				ss["images"] = images_arr;
			}

			animations_arr.push_back(ss);
		}

		if (!animations_arr.empty()) {
			result["animations"] = animations_arr;
		}

		return result;
	}

	void sprite_sheet_deserialize(fan::json& json) {
		animation_nr_t counter_offset = all_animations_counter;

		if (json.contains("animations")) {
			for (const auto& item : json["animations"]) {
				sprite_sheet_animation_t anim;
				anim.name = item.value("name", std::string{});
				if (item.contains("selected_frames") && item["selected_frames"].is_array()) {
					anim.selected_frames.clear();
					for (const auto& frame_json : item["selected_frames"]) {
						anim.selected_frames.push_back(frame_json.get<int>());
					}
					if (item.contains("images")) {
						for (const auto& frame_json : item["images"]) {
							loco_t::sprite_sheet_animation_t::image_t img;
							img = frame_json;
							anim.images.push_back(img);
						}
					}
				}
				else {
					anim.selected_frames.clear();
				}
				anim.fps = item.value("fps", 0.0f);

				animation_nr_t original_id = item.value("id", uint32_t());
				animation_nr_t new_id = original_id.id + counter_offset.id;

				all_animations[new_id] = anim;
				all_animations_counter = std::max(all_animations_counter.id, static_cast<uint32_t>(new_id.id + 1));
			}
		}

		// update animation id table
		if (json.contains("shapes")) {
			for (auto& shape : json["shapes"]) {
				// Update animation references in each shape
				if (shape.contains("animations")) {
					for (auto& anim_id : shape["animations"]) {
						animation_nr_t original_id = anim_id.get<uint32_t>();
						anim_id = original_id.id + counter_offset.id;
					}
				}
			}
		}
	}
#endif
	std::unordered_map<animation_nr_t, sprite_sheet_animation_t, animation_nr_hash_t> all_animations;
	animation_nr_t all_animations_counter = 0;
	std::unordered_map<std::pair<animation_shape_nr_t, std::string>, animation_nr_t, animation_pair_hash_t> shape_animation_lookup_table;
	std::unordered_map<animation_shape_nr_t, std::vector<animation_nr_t>, animation_nr_hash_t> shape_animations;
	animation_nr_t shape_animation_counter = 0;

	#if defined(fan_json)
	void parse_animations(fan::json& json_in) {
		gloco->sprite_sheet_deserialize(json_in);

		for (auto& item : json_in["animations"]) {
			loco_t::sprite_sheet_animation_t anim;
			anim.name = item.value("name", std::string{});
			anim.selected_frames = item.value("selected_frames", std::vector<int>{});
			for (const auto& frame_json : item["images"]) {
				loco_t::sprite_sheet_animation_t::image_t img;
				img = frame_json;
				anim.images.push_back(img);
			}
			anim.fps = item.value("fps", 0.0f);

			loco_t::animation_nr_t id = item.value("id", uint32_t());
			gloco->all_animations[id] = anim;
		}
	}
#endif

	//-----------------------sprite sheet animations-----------------------


#if defined (fan_gui)
	using console_t = fan::console_t;
#endif

	using blending_t = uint8_t;
	using depth_t = uint16_t;

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

#include <fan/graphics/opengl/texture_pack.h>

#pragma pack(push, 1)

	struct vertex_t {
		fan::vec3 position;
		fan::color color;
	};

	struct polygon_vertex_t {
		fan::vec3 position;
		fan::color color;
		fan::vec3 offset;
		fan::vec3 angle;
		fan::vec2 rotation_point;
	};

#pragma pack(pop)

#if defined(loco_opengl)
	// opengl namespace
	struct opengl {
#include <fan/graphics/opengl/engine_functions.h>
#include <fan/graphics/opengl/2D/effects/blur.h>

		blur_t blur;

		fan::opengl::core::framebuffer_t m_framebuffer;
		fan::opengl::core::renderbuffer_t m_rbo;
		loco_t::image_t color_buffers[4];
		loco_t::shader_t m_fbo_final_shader;

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
	static T2& get_render_data(shape_t* shape, T2 T::* attribute) {
		shaper_t::ShapeRenderData_t* data = shape->GetRenderData(gloco->shaper);
		return ((T*)data)->*attribute;
	}

	template <typename T, typename T2, typename T3, typename T4>
	static void modify_render_data_element_arr(shape_t* shape, T2 T::* attribute, std::size_t i, auto T4::* arr_member, const T3& value) {
		shaper_t::ShapeRenderData_t* data = shape->GetRenderData(gloco->shaper);

		// remove gloco
		if (gloco->window.renderer == renderer_t::opengl) {
			gloco->gl.modify_render_data_element_arr(shape, data, attribute, i, arr_member, value);
		}
#if defined(fan_vulkan)
		else if (gloco->window.renderer == renderer_t::vulkan) {
			(((T*)data)->*attribute)[i].*arr_member = value;
			auto& data = gloco->shaper.ShapeList[*shape];
			gloco->shaper.ElementIsPartiallyEdited(
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
	static void modify_render_data_element(shape_t* shape, T2 T::* attribute, const T3& value) {
		shaper_t::ShapeRenderData_t* data = shape->GetRenderData(gloco->shaper);

		// remove gloco
		if (gloco->window.renderer == renderer_t::opengl) {
			gloco->gl.modify_render_data_element(shape, data, attribute, value);
		}
#if defined(fan_vulkan)
		else if (gloco->window.renderer == renderer_t::vulkan) {
			((T*)data)->*attribute = value;
			auto& data = gloco->shaper.ShapeList[*shape];
			gloco->shaper.ElementIsPartiallyEdited(
				data.sti,
				data.blid,
				data.ElementIndex,
				fan::member_offset(attribute),
				sizeof(T3)
			);
		}
#endif
	}


#pragma pack(push, 1)

#define st(name, viewport_inside) \
	template <bool cond> \
	struct CONCAT(name, _cond) { \
		template <typename T> \
		using d = typename fan::type_or_uint8_t<cond>::template d<T>; \
		viewport_inside \
	}; \
	using name = CONCAT(name, _cond)<1>; \
	struct CONCAT(_, name) : CONCAT(name, _cond<0>) {};

	using multitexture_image_t = std::array<loco_t::image_t, 30>;

	struct kps_t {
		st(light_t,
			d<uint8_t> genre;
			d<loco_t::viewport_t> viewport;
			d<loco_t::camera_t> camera;
			d<shaper_t::ShapeTypeIndex_t> ShapeType;
			d<uint8_t> draw_mode;
			d<uint32_t> vertex_count;
		);
		st(common_t,
			d<depth_t> depth;
			d<blending_t> blending;
			d<loco_t::viewport_t> viewport;
			d<loco_t::camera_t> camera;
			d<shaper_t::ShapeTypeIndex_t> ShapeType;
			d<uint8_t> draw_mode;
			d<uint32_t> vertex_count;
		);
		st(vfi_t,
			d<uint8_t> filler = 0;
		);
		st(texture_t,
			d<depth_t> depth;
			d<blending_t> blending;
			d<loco_t::image_t> image;
			d<loco_t::viewport_t> viewport;
			d<loco_t::camera_t> camera;
			d<shaper_t::ShapeTypeIndex_t> ShapeType;
			d<uint8_t> draw_mode;
			d<uint32_t> vertex_count;
		);
		// for universal_image_renderer
		// struct texture4_t {
		//   blending_t blending;
		//   depth_t depth;
		//   loco_t::image_t image; // 4 - 1
		//   loco_t::viewport_t viewport;
		//   loco_t::camera_t camera;
		//   shaper_t::ShapeTypeIndex_t ShapeType;
		// };
	};

#undef st
#pragma pack(pop)

public:

	std::vector<std::function<void()>> m_pre_draw;
	std::vector<std::function<void()>> m_post_draw;


	struct properties_t {
		bool render_shapes_top = false;
		bool vsync = true;
		fan::vec2 window_size = -1;
		uint64_t window_flags = 0;
		uint8_t renderer = renderer_t::opengl;
		uint8_t samples = 0;
	};

	fan::time::timer start_time;

	void add_shape_to_immediate_draw(loco_t::shape_t&& s) {
		immediate_render_list.emplace_back(std::move(s));
	}
	auto add_shape_to_static_draw(loco_t::shape_t&& s) {
		auto ret = s.NRI;
		static_render_list[ret] = std::move(s);
		return ret;
	}
	void remove_static_shape_draw(const loco_t::shape_t& s) {
		static_render_list.erase(s.NRI);
	}

#define shaper_get_key_safe(return_type, kps_type, variable) \
	[key_pack] ()-> auto& { \
		auto o = gloco->shaper.GetKeyOffset( \
			offsetof(loco_t::kps_t::CONCAT(_, kps_type), variable), \
			offsetof(loco_t::kps_t::kps_type, variable) \
		);\
		static_assert(std::is_same_v<decltype(loco_t::kps_t::kps_type::variable), loco_t::return_type>, "possibly unwanted behaviour"); \
		return *(loco_t::return_type*)&key_pack[o];\
	}()

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
				loco_t::rectangle_t::properties_t props;
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
				loco_t::shape_t* s = reinterpret_cast<loco_t::shape_t*>(&shape_id);
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

	inline static constexpr f32_t font_sizes[] = {
		4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 
		16, 18, 20, 22, 24, 28, 
		32, 36, 48, 60, 72
	};

	// -1 no reload, opengl = 0 etc
	uint8_t reload_renderer_to = -1;
#if defined(fan_gui)
	void load_fonts(ImFont* (&fonts)[std::size(loco_t::font_sizes)], const std::string& name, ImFontConfig* cfg = nullptr) {
		ImGuiIO& io = ImGui::GetIO();
		for (std::size_t i = 0; i < std::size(fonts); ++i) {
			fonts[i] = io.Fonts->AddFontFromFileTTF(name.c_str(), font_sizes[i] * 2, cfg);

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
		f32_t best_diff = std::abs(font_sizes[0] - font_size);

		for (std::size_t i = 1; i < std::size(font_sizes); ++i) {
			f32_t diff = std::abs(font_sizes[i] - font_size);
			if (diff < best_diff) {
				best_diff = diff;
				best_index = i;
			}
		}

		return !bold ? fonts[best_index] : fonts_bold[best_index];
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

		if (window.renderer == renderer_t::opengl) {
			glfwMakeContextCurrent(window);
			ImGui_ImplGlfw_InitForOpenGL(window, true);
			const char* glsl_version = "#version 120";
			ImGui_ImplOpenGL3_Init(glsl_version);
		}
#if defined(fan_vulkan)
		else if (window.renderer == renderer_t::vulkan) {
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

		load_fonts(fonts_bold, "fonts/SourceCodePro-Bold.ttf");

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
		
		for (std::size_t i = 0; i < std::size(fonts); ++i) {
			f32_t font_size = font_sizes[i] * 2; // load 2x font size and possibly downscale for better quality

			ImFontConfig main_cfg;
			fonts[i] = io.Fonts->AddFontFromFileTTF("fonts/Roboto-Regular.ttf", font_size, &main_cfg);

			ImFontConfig emoji_cfg;
			emoji_cfg.MergeMode = true;
			emoji_cfg.FontBuilderFlags |= ImGuiFreeTypeBuilderFlags_LoadColor;
			emoji_cfg.SizePixels = 0;
			emoji_cfg.RasterizerDensity = 1.0f;
			emoji_cfg.GlyphMinAdvanceX = font_size;
			io.Fonts->AddFontFromFileTTF("fonts/seguiemj.ttf", font_size, &emoji_cfg, emoji_ranges);
		}
		build_fonts();
		io.FontDefault = fonts[9];

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
			if (window.renderer == renderer_t::opengl) {
				ImGui_ImplOpenGL3_Shutdown();
			}
#if defined(fan_vulkan)
			else if (window.renderer == renderer_t::vulkan) {
				vkDeviceWaitIdle(context.vk.device);
				ImGui_ImplVulkan_Shutdown();
			}
#endif
			imgui_initialized = false;
			return;
		}

		if (window.renderer == renderer_t::opengl) {
			ImGui_ImplOpenGL3_Shutdown();
		}
#if defined(fan_vulkan)
		else if (window.renderer == renderer_t::vulkan) {
			vkDeviceWaitIdle(context.vk.device);
			ImGui_ImplVulkan_Shutdown();
		}
#endif
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();
		ImPlot::DestroyContext();
#if defined(fan_vulkan)
		if (window.renderer == renderer_t::vulkan) {
			context.vk.imgui_close();
		}
#endif

		global_imgui_initialized = false;
		imgui_initialized = false;
	}
	bool enable_overlay = true;
#endif
	void init_framebuffer() {
		if (window.renderer == renderer_t::opengl) {
			gl.init_framebuffer();
		}
	}

	loco_t() : loco_t(properties_t()) {

	}
	loco_t(const properties_t& p) {

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
		if (window.renderer == renderer_t::opengl) {
			new (&context.gl) fan::opengl::context_t();
			context_functions = fan::graphics::get_gl_context_functions();
			gl.open();
		}

		window.set_antialiasing(p.samples);
		window.open(p.window_size, fan::window_t::default_window_name, p.window_flags);
		gloco = this;


#if fan_debug >= fan_debug_high && !defined(fan_vulkan)
		if (window.renderer == renderer_t::vulkan) {
			fan::throw_error("trying to use vulkan renderer, but fan_vulkan build flag is disabled");
		}
#endif

#if defined(fan_vulkan)
		if (window.renderer == renderer_t::vulkan) {
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
		if (window.renderer == renderer_t::opengl) {
			window.make_context_current();

#if fan_debug >= fan_debug_high
			get_context().gl.set_error_callback();
#endif

			if (window.get_antialiasing() > 0) {
				glEnable(GL_MULTISAMPLE);
			}

			gl.initialize_fb_vaos();
		}

#if defined(loco_vfi)
		window.add_buttons_callback([this](const fan::window_t::mouse_buttons_cb_data_t& d) {
			fan::vec2 window_size = window.get_size();
			vfi.feed_mouse_button(d.button, d.state);
			});

		window.add_keys_callback([&](const fan::window_t::keyboard_keys_cb_data_t& d) {
			vfi.feed_keyboard(d.key, d.state);
			});

		window.add_mouse_move_callback([&](const fan::window_t::mouse_move_cb_data_t& d) {
			vfi.feed_mouse_move(d.position);
			});

		window.add_text_callback([&](const fan::window_t::text_cb_data_t& d) {
			vfi.feed_text(d.character);
			});
#endif

		load_engine_images();

		shaper.Open();

		{

			// filler
			shaper.AddKey(Key_e::light, sizeof(uint8_t), shaper_t::KeyBitOrderAny);
			shaper.AddKey(Key_e::light_end, sizeof(uint8_t), shaper_t::KeyBitOrderAny);
			shaper.AddKey(Key_e::depth, sizeof(loco_t::depth_t), shaper_t::KeyBitOrderLow);
			shaper.AddKey(Key_e::blending, sizeof(loco_t::blending_t), shaper_t::KeyBitOrderLow);
			shaper.AddKey(Key_e::image, sizeof(loco_t::image_t), shaper_t::KeyBitOrderLow);
			shaper.AddKey(Key_e::viewport, sizeof(loco_t::viewport_t), shaper_t::KeyBitOrderAny);
			shaper.AddKey(Key_e::camera, sizeof(loco_t::camera_t), shaper_t::KeyBitOrderAny);
			shaper.AddKey(Key_e::ShapeType, sizeof(shaper_t::ShapeTypeIndex_t), shaper_t::KeyBitOrderAny);
			shaper.AddKey(Key_e::filler, sizeof(uint8_t), shaper_t::KeyBitOrderAny);
			shaper.AddKey(Key_e::draw_mode, sizeof(uint8_t), shaper_t::KeyBitOrderAny);
			shaper.AddKey(Key_e::vertex_count, sizeof(uint32_t), shaper_t::KeyBitOrderAny);
			shaper.AddKey(Key_e::shadow, sizeof(uint8_t), shaper_t::KeyBitOrderAny);

			//gloco->shaper.AddKey(Key_e::image4, sizeof(loco_t::image_t) * 4, shaper_t::KeyBitOrderLow);
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

		if (window.renderer == renderer_t::opengl) {
			gl.shapes_open();
		}
#if defined(fan_vulkan)
		else if (window.renderer == renderer_t::vulkan) {
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
		piece_hover.open_piece("audio/hover.sac", 0);
		piece_click.open_piece("audio/click.sac", 0);

#endif
	}
	~loco_t() {
		destroy();
	}

	void destroy() {
		if (window == nullptr) {
			return;
		}
#if defined(fan_gui)
		console.commands.func_table.clear();
		console.close();
#endif
#if defined(fan_physics)
		fan::graphics::close_bcol();
#endif
#if defined(fan_vulkan)
		if (window.renderer == loco_t::renderer_t::vulkan) {
			vkDeviceWaitIdle(context.vk.device);
			vkDestroySampler(context.vk.device, vk.post_process_sampler, nullptr);
			vk.d_attachments.close(context.vk);
			vk.post_process.close(context.vk);
		}
#endif
		shaper.Close();
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

	// for renderer switch
	// input loco_t::renderer_t::
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
			if (window.renderer == loco_t::renderer_t::vulkan) {
				// todo wrap to vk.
				vkDeviceWaitIdle(context.vk.device);
				vkDestroySampler(context.vk.device, vk.post_process_sampler, nullptr);
				vk.d_attachments.close(context.vk);
				vk.post_process.close(context.vk);
				for (auto& st : shaper.ShapeTypes) {
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
				if (window.renderer == loco_t::renderer_t::opengl) {
					glDeleteVertexArrays(1, &gl.fb_vao);
					glDeleteBuffers(1, &gl.fb_vbo);
					context.gl.internal_close();
				}

#if defined(fan_gui)
			if (imgui_initialized) {
				if (window.renderer == renderer_t::opengl) {
					ImGui_ImplOpenGL3_Shutdown();
				}
#if defined(fan_vulkan)
				else if (window.renderer == renderer_t::vulkan) {
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
			if (window.renderer == renderer_t::opengl) {
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
			if (window.renderer == renderer_t::vulkan) {
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

							if (window.renderer == renderer_t::opengl) {
								// illegal
								image_list[nr].internal = new fan::opengl::context_t::image_t;
								fan_opengl_call(glGenTextures(1, &((fan::opengl::context_t::image_t*)context_functions.image_get(&context.gl, nr))->texture_id));
							}
#if defined(fan_vulkan)
							else if (window.renderer == renderer_t::vulkan) {
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
							if (window.renderer == renderer_t::opengl) {
								__fan_internal_shader_list[nr].internal = new fan::opengl::context_t::shader_t;
							}
#if defined(fan_vulkan)
							else if (window.renderer == renderer_t::vulkan) {
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

			if (window.renderer == renderer_t::opengl) {
				gl.shapes_open();
				gl.initialize_fb_vaos();
				if (window.get_antialiasing() > 0) {
					glEnable(GL_MULTISAMPLE);
				}
			}
#if defined(fan_vulkan)
			else if (window.renderer == renderer_t::vulkan) {
				vk.shapes_open();
			}
#endif

#if defined(fan_gui)
			if (was_imgui_init && global_imgui_initialized) {
				if (window.renderer == renderer_t::opengl) {
					glfwMakeContextCurrent(window);
					ImGui_ImplGlfw_InitForOpenGL(window, true);
					const char* glsl_version = "#version 120";
					ImGui_ImplOpenGL3_Init(glsl_version);
				}
#if defined(fan_vulkan)
				else if (window.renderer == renderer_t::vulkan) {
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

			shaper._BlockListCapacityChange(shape_type_t::rectangle, 0, 1);
			shaper._BlockListCapacityChange(shape_type_t::sprite, 0, 1);

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
		if (window.renderer == renderer_t::opengl) {
			gl.draw_shapes();
		}
#if defined(fan_vulkan)
		else
			if (window.renderer == renderer_t::vulkan) {
				vk.draw_shapes();
			}
#endif
    shape_draw_time_s = shape_draw_timer.seconds();

		immediate_render_list.clear();
	}
	void process_shapes() {

#if defined(fan_vulkan)
		if (window.renderer == renderer_t::vulkan) {
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
		if (window.renderer == renderer_t::vulkan) {
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
      fan::print(fan::vec2(ImGui::GetWindowSize()));

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

		if (window.renderer == renderer_t::opengl) {
			//glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
			//glClear(GL_COLOR_BUFFER_BIT);

			ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		}
#if defined(fan_vulkan)
		else if (window.renderer == renderer_t::vulkan) {
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

		if (window.renderer == renderer_t::opengl) {
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
				((shape_physics_update_cb)shape_physics_update_cbs[it].cb)(shape_physics_update_cbs[it]);
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

		shaper.ProcessBlockEditQueue();

#if defined(fan_vulkan)
		if (window.renderer == renderer_t::vulkan) {
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
		if (window.renderer == renderer_t::opengl) {
			glfwSwapBuffers(window);
		}
#if defined(fan_vulkan)
		else if (window.renderer == renderer_t::vulkan) {
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

		if (window.renderer == renderer_t::opengl) {
			ImGui_ImplOpenGL3_NewFrame();
		}
#if defined(fan_vulkan)
		else if (window.renderer == renderer_t::vulkan) {
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

	loco_t::viewport_t open_viewport(const fan::vec2& viewport_position, const fan::vec2& viewport_size) {
		loco_t::viewport_t viewport = viewport_create();
		viewport_set(viewport, viewport_position, viewport_size);
		return viewport;
	}

	void set_viewport(loco_t::viewport_t viewport, const fan::vec2& viewport_position, const fan::vec2& viewport_size) {
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
		if (window.renderer == renderer_t::opengl) {
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

	struct render_view_t {
		loco_t::camera_t camera;
		loco_t::viewport_t viewport;

		void create() {
			camera = gloco->camera_create();
			viewport = gloco->viewport_create();
		}
		void remove() {
			gloco->camera_erase(camera);
			gloco->viewport_erase(viewport);
		}
		void set(
			const fan::vec2& ortho_x, const fan::vec2& ortho_y,
			const fan::vec2& viewport_position, const fan::vec2& viewport_size
		) {
			gloco->camera_set_ortho(camera, ortho_x, ortho_y);
			gloco->viewport_set(viewport, viewport_position, viewport_size);
		}
	};

	loco_t::render_view_t render_view_create() {
		loco_t::render_view_t render_view;
		render_view.create();
		return render_view;
	}
	loco_t::render_view_t render_view_create(
		const fan::vec2& ortho_x, const fan::vec2& ortho_y,
		const fan::vec2& viewport_position, const fan::vec2& viewport_size
	) {
		loco_t::render_view_t render_view;
		render_view.create();
		render_view.set(ortho_x, ortho_y, viewport_position, viewport_size);
		return render_view;
	}

	struct input_action_t {
		enum {
			none = -1,
			release = (int)fan::keyboard_state::release,
			press = (int)fan::keyboard_state::press,
			repeat = (int)fan::keyboard_state::repeat,
			press_or_repeat
		};

		struct action_data_t {
			static constexpr int max_keys_per_action = 5;
			int keys[max_keys_per_action]{};
			uint8_t count = 0;
			static constexpr int max_keys_combos = 5;
			int key_combos[max_keys_combos]{};
			uint8_t combo_count = 0;
		};

		void add(const int* keys, std::size_t count, const std::string& action_name) {
			action_data_t action_data;
			action_data.count = (uint8_t)count;
			std::memcpy(action_data.keys, keys, sizeof(int) * count);
			input_actions[action_name] = action_data;
		}
		void add(int key, const std::string& action_name) {
			add(&key, 1, action_name);
		}
		void add(std::initializer_list<int> keys, const std::string& action_name) {
			add(keys.begin(), keys.size(), action_name);
		}

		void edit(int key, const std::string& action_name) {
			auto found = input_actions.find(action_name);
			if (found == input_actions.end()) {
				fan::throw_error("trying to modify non existing action");
			}
			std::memset(found->second.keys, 0, sizeof(found->second.keys));
			found->second.keys[0] = key;
			found->second.count = 1;
			found->second.combo_count = 0;
		}

		void add_keycombo(std::initializer_list<int> keys, const std::string& action_name) {
			action_data_t action_data;
			action_data.combo_count = (uint8_t)keys.size();
			std::memcpy(action_data.key_combos, keys.begin(), sizeof(int) * action_data.combo_count);
			input_actions[action_name] = action_data;
		}

		bool is_active(const std::string& action_name, int pstate = loco_t::input_action_t::press) {
			auto found = input_actions.find(action_name);
			if (found != input_actions.end()) {
				action_data_t& action_data = found->second;

				if (action_data.combo_count) {
					int state = none;
					for (int i = 0; i < action_data.combo_count; ++i) {
						int s = gloco->window.key_state(action_data.key_combos[i]);
						if (s == none) {
							return none == loco_t::input_action_t::press;
						}
						if (state == input_action_t::press && s == input_action_t::repeat) {
							state = 1;
						}
						if (state == input_action_t::press_or_repeat) {
							if (state == input_action_t::press && s == input_action_t::repeat) {
							}
						}
						else {
							state = s;
						}
					}
					if (pstate == input_action_t::press_or_repeat) {
						return state == input_action_t::press ||
							state == input_action_t::repeat;
					}
					return state == pstate;
				}
				else if (action_data.count) {
					int state = none;
					for (int i = 0; i < action_data.count; ++i) {
						int s = gloco->window.key_state(action_data.keys[i]);
						if (s != none) {
							state = s;
						}
					}
					if (pstate == input_action_t::press_or_repeat) {
						return state == input_action_t::press ||
							state == input_action_t::repeat;
					}
					//fan::print(state, pstate, state == pstate);
					return state == pstate;
				}
			}
			return none == pstate;
		}
		bool is_action_clicked(const std::string& action_name) {
			return is_active(action_name);
		}
		bool is_action_down(const std::string& action_name) {
			return is_active(action_name, press_or_repeat);
		}
		bool exists(const std::string& action_name) {
			return input_actions.find(action_name) != input_actions.end();
		}
		void insert_or_assign(int key, const std::string& action_name) {
			action_data_t action_data;
			action_data.count = (uint8_t)1;
			std::memcpy(action_data.keys, &key, sizeof(int) * 1);
			input_actions.insert_or_assign(action_name, action_data);
		}
		void remove(const std::string& action_name) {
			input_actions.erase(action_name);
		}

		std::unordered_map<std::string, action_data_t> input_actions;
	}input_action;

	static fan::vec2 transform_position(const fan::vec2& p, loco_t::viewport_t viewport, loco_t::camera_t camera) {

		auto v = gloco->viewport_get(viewport);
		auto c = gloco->camera_get(camera);

		fan::vec2 viewport_position = v.viewport_position;
		fan::vec2 viewport_size = v.viewport_size;

		f32_t l = c.coordinates.left;
		f32_t r = c.coordinates.right;
		f32_t t = c.coordinates.up;
		f32_t b = c.coordinates.down;

		fan::vec2 tp = p - viewport_position;
		fan::vec2 d = viewport_size;
		tp /= d;
		tp = fan::vec2(r * tp.x - l * tp.x + l, b * tp.y - t * tp.y + t);
		tp += c.position;
		return tp;
	}

protected:
#define BLL_set_SafeNext 1
#define BLL_set_AreWeInsideStruct 1
#define BLL_set_prefix update_callback
#include <fan/fan_bll_preset.h>
#define BLL_set_Link 1
#define BLL_set_type_node uint16_t
#define BLL_set_NodeDataType std::function<void(loco_t*)>
#define BLL_set_CPP_CopyAtPointerChange 1
#include <BLL/BLL.h>
public:

	using update_callback_nr_t = update_callback_NodeReference_t;

	update_callback_t m_update_callback;

	std::vector<std::function<void()>> single_queue;

	#include "engine_images.h"

	render_view_t orthographic_render_view;
	render_view_t perspective_render_view;

	fan::window_t window;
	void set_window_name(const std::string& name) {
		window.set_name(name);
	}
	void set_window_icon(const fan::image::info_t& info) {
		window.set_icon(info);
	}
	void set_window_icon(const loco_t::image_t& image) {
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
#define BLL_set_SafeNext 1
#define BLL_set_AreWeInsideStruct 1
#define BLL_set_prefix gui_draw_cb
#include <fan/fan_bll_preset.h>
#define BLL_set_Link 1
#define BLL_set_type_node uint16_t
#define BLL_set_NodeDataType std::function<void()>
#include <BLL/BLL.h>

	gui_draw_cb_t gui_draw_cb;
#endif

	using push_back_cb = loco_t::shape_t(*)(void*);
	using set_position2_cb = void (*)(loco_t::shape_t*, const fan::vec2&);
	// depth
	using set_position3_cb = void (*)(loco_t::shape_t*, const fan::vec3&);
	using set_size_cb = void (*)(loco_t::shape_t*, const fan::vec2&);
	using set_size3_cb = void (*)(loco_t::shape_t*, const fan::vec3&);

	using get_position_cb = fan::vec3(*)(const loco_t::shape_t*);
	using get_size_cb = fan::vec2(*)(const loco_t::shape_t*);
	using get_size3_cb = fan::vec3(*)(const loco_t::shape_t*);

	using set_rotation_point_cb = void (*)(loco_t::shape_t*, const fan::vec2&);
	using get_rotation_point_cb = fan::vec2(*)(const loco_t::shape_t*);

	using set_color_cb = void (*)(loco_t::shape_t*, const fan::color&);
	using get_color_cb = fan::color(*)(const loco_t::shape_t*);

	using set_angle_cb = void (*)(loco_t::shape_t*, const fan::vec3&);
	using get_angle_cb = fan::vec3(*)(const loco_t::shape_t*);

	using get_tc_position_cb = fan::vec2(*)(loco_t::shape_t*);
	using set_tc_position_cb = void (*)(loco_t::shape_t*, const fan::vec2&);

	using get_tc_size_cb = fan::vec2(*)(loco_t::shape_t*);
	using set_tc_size_cb = void (*)(loco_t::shape_t*, const fan::vec2&);

	using load_tp_cb = bool(*)(loco_t::shape_t*, loco_t::texturepack_t::ti_t*);

	using get_grid_size_cb = fan::vec2(*)(loco_t::shape_t*);
	using set_grid_size_cb = void (*)(loco_t::shape_t*, const fan::vec2&);

	using get_camera_cb = loco_t::camera_t(*)(const loco_t::shape_t*);
	using set_camera_cb = void (*)(loco_t::shape_t*, loco_t::camera_t);

	using get_viewport_cb = loco_t::viewport_t(*)(const loco_t::shape_t*);
	using set_viewport_cb = void (*)(loco_t::shape_t*, loco_t::viewport_t);


	using get_image_cb = loco_t::image_t(*)(loco_t::shape_t*);
	using set_image_cb = void (*)(loco_t::shape_t*, loco_t::image_t);

	using get_image_data_cb = fan::graphics::image_data_t& (*)(loco_t::shape_t*);

	using get_parallax_factor_cb = f32_t(*)(loco_t::shape_t*);
	using set_parallax_factor_cb = void (*)(loco_t::shape_t*, f32_t);
	using get_flags_cb = uint32_t(*)(loco_t::shape_t*);
	using set_flags_cb = void(*)(loco_t::shape_t*, uint32_t);
	//
	using get_radius_cb = f32_t(*)(loco_t::shape_t*);
	using get_src_cb = fan::vec3(*)(loco_t::shape_t*);
	using get_dst_cb = fan::vec3(*)(loco_t::shape_t*);
	using get_outline_size_cb = f32_t(*)(loco_t::shape_t*);
	using get_outline_color_cb = fan::color(*)(const loco_t::shape_t*);
	using set_outline_color_cb = void(*)(loco_t::shape_t*, const fan::color&);

	using reload_cb = void (*)(loco_t::shape_t*, uint8_t format, void** image_data, const fan::vec2& image_size, uint32_t filter);

	using draw_cb = void (*)(uint8_t draw_range);

	using set_line_cb = void (*)(loco_t::shape_t*, const fan::vec3&, const fan::vec3&);
	using set_line3_cb = void (*)(loco_t::shape_t*, const fan::vec3&, const fan::vec3&);

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

	// needs continous buffer
	std::vector<shaper_t::BlockProperties_t> BlockProperties;

	shaper_t shaper;

#if defined(fan_physics)
	fan::physics::context_t physics_context{ {} };
	struct physics_update_data_t {
		shaper_t::ShapeID_t shape_id;
		fan::vec2 draw_offset = 0;
		uint64_t body_id;
		void* cb;
	};
	using shape_physics_update_cb = void(*)(const physics_update_data_t& data);
#define BLL_set_SafeNext 1
#define BLL_set_AreWeInsideStruct 1
#define BLL_set_prefix physics_update_cbs
#include <fan/fan_bll_preset.h>
#define BLL_set_Link 1
#define BLL_set_type_node uint16_t
#define BLL_set_NodeDataType physics_update_data_t
#include <BLL/BLL.h>
	physics_update_cbs_t::nr_t add_physics_update(const physics_update_data_t& cb_data) {
		auto it = shape_physics_update_cbs.NewNodeLast();
		shape_physics_update_cbs[it] = (physics_update_data_t)cb_data;
		return it;
	}
	void remove_physics_update(physics_update_cbs_t::nr_t nr) {
		shape_physics_update_cbs.unlrec(nr);
	}
	physics_update_cbs_t shape_physics_update_cbs;
#endif

	// clears shapes after drawing, good for debug draw, not best for performance
	std::vector<loco_t::shape_t> immediate_render_list;

	std::unordered_map<uint32_t, loco_t::shape_t> static_render_list;

#pragma pack(push, 1)

	struct Key_e {
		enum : shaper_t::KeyTypeIndex_t {
			light,
			light_end,
			blending,
			depth,
			image,
			viewport,
			camera,
			ShapeType,
			filler,
			draw_mode,
			vertex_count,
			shadow
		};
	};

#pragma pack(pop)

	fan::vec2 get_mouse_position(const camera_t& camera, const viewport_t& viewport) const {
		return transform_position(get_mouse_position(), viewport, camera);
	}
	fan::vec2 get_mouse_position(const loco_t::render_view_t& render_view) const {
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

	#define shape_get_vi(shape) (*(loco_t::shape##_t::vi_t*)GetRenderData(gloco->shaper))
	#define shape_get_ri(shape) (*(loco_t::shape##_t::ri_t*)GetData(gloco->shaper))

	
	#include <fan/graphics/shape_functions.h>

	shape_functions_t shape_functions;


	// ShapeID_t must be at the beginning of shape_t's memory since there are reinterpret_casts 
	// which assume that
	struct shape_t : shaper_t::ShapeID_t {
		using shaper_t::ShapeID_t::ShapeID_t;
		shape_t() {
			sic();
		}
		template <typename T>
		requires requires(T t) { typename T::type_t; }
		shape_t(const T& properties) : shape_t() {
			auto shape_type = T::type_t::shape_type;
			*this = gloco->shape_functions[shape_type].push_back((void*)&properties);
		}

		shape_t(shaper_t::ShapeID_t&& s) {
			//if (s.iic() == false) {
			//  if (((shape_t*)&s)->get_shape_type() == shape_type_t::polygon) {
			//    loco_t::polygon_t::ri_t* src_data = (loco_t::polygon_t::ri_t*)s.GetData(gloco->shaper);
			//    loco_t::polygon_t::ri_t* dst_data = (loco_t::polygon_t::ri_t*)GetData(gloco->shaper);
			//    *dst_data = *src_data;
			//  }
			//}

			NRI = s.NRI;

			if (get_shape_type() == loco_t::shape_type_t::sprite) {
				auto& ri = *(loco_t::sprite_t::ri_t*)s.GetData(gloco->shaper);
				if (ri.sprite_sheet_data.frame_update_nr) {
					gloco->m_update_callback[ri.sprite_sheet_data.frame_update_nr] = [nr = NRI] (loco_t* loco) {
						loco_t::shape_t::sprite_sheet_frame_update_cb(loco, (loco_t::shape_t*)&nr);
					};
				}
			}

			s.sic();
		}

		shape_t(const shaper_t::ShapeID_t& s) : shape_t() {

			if (s.iic()) {
				return;
			}

			auto sti = gloco->shaper.ShapeList[s].sti;
			{
				shaper_deep_copy(this, (const loco_t::shape_t*)&s, sti);
			}
			if (sti == shape_type_t::polygon) {
				loco_t::polygon_t::ri_t* src_data = (loco_t::polygon_t::ri_t*)s.GetData(gloco->shaper);
				loco_t::polygon_t::ri_t* dst_data = (loco_t::polygon_t::ri_t*)GetData(gloco->shaper);
				if (gloco->get_renderer() == renderer_t::opengl) {
					dst_data->vao.open(gloco->context.gl);
					dst_data->vbo.open(gloco->context.gl, src_data->vbo.m_target);

					auto& shape_data = gloco->shaper.GetShapeTypes(shape_type_t::polygon).renderer.gl;
					fan::graphics::context_shader_t shader;
					if (!shape_data.shader.iic()) {
						shader = gloco->shader_get(shape_data.shader);
					}
					dst_data->vao.bind(gloco->context.gl);
					dst_data->vbo.bind(gloco->context.gl);
					uint64_t ptr_offset = 0;
					for (shape_gl_init_t& location : gloco->polygon.locations) {
						if ((gloco->context.gl.opengl.major == 2 && gloco->context.gl.opengl.minor == 1) && !shape_data.shader.iic()) {
							location.index.first = fan_opengl_call(glGetAttribLocation(shader.gl.id, location.index.second));
						}
						fan_opengl_call(glEnableVertexAttribArray(location.index.first));
						switch (location.type) {
						case GL_UNSIGNED_INT:
						case GL_INT: {
							fan_opengl_call(glVertexAttribIPointer(location.index.first, location.size, location.type, location.stride, (void*)ptr_offset));
							break;
						}
						default: {
							fan_opengl_call(glVertexAttribPointer(location.index.first, location.size, location.type, GL_FALSE, location.stride, (void*)ptr_offset));
						}
						}
						// instancing
						if ((gloco->context.gl.opengl.major > 3) || (gloco->context.gl.opengl.major == 3 && gloco->context.gl.opengl.minor >= 3)) {
							if (shape_data.instanced) {
								fan_opengl_call(glVertexAttribDivisor(location.index.first, 1));
							}
						}
						switch (location.type) {
						case GL_FLOAT: {
							ptr_offset += location.size * sizeof(GLfloat);
							break;
						}
						case GL_UNSIGNED_INT: {
							ptr_offset += location.size * sizeof(GLuint);
							break;
						}
						default: {
							fan::throw_error_impl();
						}
						}
					}
					fan::opengl::core::write_glbuffer(gloco->context.gl, dst_data->vbo.m_buffer, 0, dst_data->buffer_size, dst_data->vbo.m_usage, dst_data->vbo.m_target);
					glBindBuffer(GL_COPY_READ_BUFFER, src_data->vbo.m_buffer);
					glBindBuffer(GL_COPY_WRITE_BUFFER, dst_data->vbo.m_buffer);
					glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, dst_data->buffer_size);
					loco_t::polygon_vertex_t* ri = new loco_t::polygon_vertex_t[dst_data->buffer_size / sizeof(loco_t::polygon_vertex_t)];
					loco_t::polygon_vertex_t* ri2 = new loco_t::polygon_vertex_t[dst_data->buffer_size / sizeof(loco_t::polygon_vertex_t)];
					fan::opengl::core::get_glbuffer(gloco->context.gl, ri, dst_data->vbo.m_buffer, dst_data->buffer_size, 0, dst_data->vbo.m_target);
					fan::opengl::core::get_glbuffer(gloco->context.gl, ri2, src_data->vbo.m_buffer, src_data->buffer_size, 0, src_data->vbo.m_target);
					delete[] ri;
				}
				else {
					fan::throw_error_impl();
				}
			}
		}

		shape_t(shape_t&& s) : shape_t(std::move(*dynamic_cast<shaper_t::ShapeID_t*>(&s))) {

		}
		shape_t(const loco_t::shape_t& s) : shape_t(*dynamic_cast<const shaper_t::ShapeID_t*>(&s)) {
			//NRI = s.NRI;
		}
		loco_t::shape_t& operator=(const loco_t::shape_t& s) {
			if (iic() == false) {
				remove();
			}
			if (s.iic()) {
				return *this;
			}
			if (this != &s) {
				auto sti = gloco->shaper.ShapeList[s].sti;
				{

					shaper_deep_copy(this, (const loco_t::shape_t*)&s, sti);
				}
				if (sti == shape_type_t::polygon) {
					loco_t::polygon_t::ri_t* src_data = (loco_t::polygon_t::ri_t*)s.GetData(gloco->shaper);
					loco_t::polygon_t::ri_t* dst_data = (loco_t::polygon_t::ri_t*)GetData(gloco->shaper);
					if (gloco->get_renderer() == renderer_t::opengl) {
						dst_data->vao.open(gloco->context.gl);
						dst_data->vbo.open(gloco->context.gl, src_data->vbo.m_target);

						auto& shape_data = gloco->shaper.GetShapeTypes(shape_type_t::polygon).renderer.gl;
						fan::graphics::context_shader_t shader;
						if (!shape_data.shader.iic()) {
							shader = gloco->shader_get(shape_data.shader);
						}
						dst_data->vao.bind(gloco->context.gl);
						dst_data->vbo.bind(gloco->context.gl);
						uint64_t ptr_offset = 0;
						for (shape_gl_init_t& location : gloco->polygon.locations) {
							if ((gloco->context.gl.opengl.major == 2 && gloco->context.gl.opengl.minor == 1) && !shape_data.shader.iic()) {
								location.index.first = fan_opengl_call(glGetAttribLocation(shader.gl.id, location.index.second));
							}
							fan_opengl_call(glEnableVertexAttribArray(location.index.first));
							switch (location.type) {
							case GL_UNSIGNED_INT:
							case GL_INT: {
								fan_opengl_call(glVertexAttribIPointer(location.index.first, location.size, location.type, location.stride, (void*)ptr_offset));
								break;
							}
							default: {
								fan_opengl_call(glVertexAttribPointer(location.index.first, location.size, location.type, GL_FALSE, location.stride, (void*)ptr_offset));
							}
							}
							// instancing
							if ((gloco->context.gl.opengl.major > 3) || (gloco->context.gl.opengl.major == 3 && gloco->context.gl.opengl.minor >= 3)) {
								if (shape_data.instanced) {
									fan_opengl_call(glVertexAttribDivisor(location.index.first, 1));
								}
							}
							switch (location.type) {
							case GL_FLOAT: {
								ptr_offset += location.size * sizeof(GLfloat);
								break;
							}
							case GL_UNSIGNED_INT: {
								ptr_offset += location.size * sizeof(GLuint);
								break;
							}
							default: {
								fan::throw_error_impl();
							}
							}
							fan::opengl::core::write_glbuffer(gloco->context.gl, dst_data->vbo.m_buffer, 0, dst_data->buffer_size, dst_data->vbo.m_usage, dst_data->vbo.m_target);
							glBindBuffer(GL_COPY_READ_BUFFER, src_data->vbo.m_buffer);
							glBindBuffer(GL_COPY_WRITE_BUFFER, dst_data->vbo.m_buffer);
							glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, dst_data->buffer_size);
						}
					}
					else {
						fan::throw_error_impl();
					}
				}
				else if (sti == loco_t::shape_type_t::sprite) {
					// handle sprite sheet specific updates
					loco_t::sprite_t::ri_t* ri = (loco_t::sprite_t::ri_t*)GetData(gloco->shaper);
					loco_t::sprite_t::ri_t* _ri = (loco_t::sprite_t::ri_t*)s.GetData(gloco->shaper);
					//if (((loco_t::sprite_t::ri_t*)_ri)->sprite_sheet_data.frame_update_nr) {
						ri->sprite_sheet_data.frame_update_nr = gloco->m_update_callback.NewNodeLast(); // since hard copy, let it leak
						gloco->m_update_callback[ri->sprite_sheet_data.frame_update_nr] = [nr = NRI](loco_t* loco) {
							loco_t::shape_t::sprite_sheet_frame_update_cb(loco, (loco_t::shape_t*)&nr);
						};
				 // }
				}
				//fan::print("i dont know what to do");
				//NRI = s.NRI;
			}
			return *this;
		}
		loco_t::shape_t& operator=(loco_t::shape_t&& s) {
			if (NRI == s.NRI) {
				s.sic();
				return *this;
			}
			if (iic() == false) {
				remove();
			}
			if (s.iic()) {
				return *this;
			}

			if (this != &s) {
				if (s.iic() == false) {

				}
				NRI = s.NRI;

				if (get_shape_type() == loco_t::shape_type_t::sprite) {
					auto& ri = *(loco_t::sprite_t::ri_t*)s.GetData(gloco->shaper);
					if (ri.sprite_sheet_data.frame_update_nr) {
						gloco->m_update_callback[ri.sprite_sheet_data.frame_update_nr] = [nr = NRI](loco_t* loco) {
							loco_t::shape_t::sprite_sheet_frame_update_cb(loco, (loco_t::shape_t*)&nr);
						};
					}
				}

				s.sic();
			}
			return *this;
		}

#if defined(fan_json)
		operator fan::json();
		operator std::string();
		shape_t(const fan::json& json);
		shape_t(const std::string&); // assume json string
		shape_t& operator=(const fan::json& json);
		shape_t& operator=(const std::string&); // assume json string
#endif
		~shape_t() {
			remove();
		}

		operator bool() const {
			return !iic();
		}
		bool operator==(const shape_t& shape) const {
			return NRI == shape.NRI;
		}

		void remove() {
			if (iic()) {
				return;
			}
#if defined(debug_shape_t)
			fan::print("-", NRI);
#endif
			if (gloco->shaper.ShapeList.Usage() == 0) {
				return;
			}
			auto shape_type = get_shape_type();
#if defined(loco_vfi)
			if (shape_type == loco_t::shape_type_t::vfi) {
				gloco->vfi.erase(*this);
				sic();
				return;
			}
#endif
			if (shape_type == loco_t::shape_type_t::polygon) {
				auto ri = (polygon_t::ri_t*)GetData(gloco->shaper);
				ri->vbo.close(gloco->context.gl);
				ri->vao.close(gloco->context.gl);
			}
			else if (shape_type == loco_t::shape_type_t::sprite) {
				auto& ri = *(loco_t::sprite_t::ri_t*)GetData(gloco->shaper);
				if (ri.sprite_sheet_data.frame_update_nr) {
					gloco->m_update_callback.unlrec(ri.sprite_sheet_data.frame_update_nr);
					ri.sprite_sheet_data.frame_update_nr.sic();
				}
			}
			gloco->shaper.remove(*this);
			sic();
		}
		void erase() {
			remove();
		}


		// many things assume uint16_t so thats why not shaper_t::ShapeTypeIndex_t

		uint16_t get_shape_type() const {
			return gloco->shaper.ShapeList[*this].sti;
		}

		void set_position(const fan::vec2& position) {
			gloco->shape_functions[get_shape_type()].set_position2(this, position);
		}
		void set_position(const fan::vec3& position) {
			gloco->shape_functions[get_shape_type()].set_position3(this, position);
		}
		void set_x(f32_t x) { set_position(fan::vec2(x, get_position().y)); }
		void set_y(f32_t y) { set_position(fan::vec2(get_position().x, y)); }
		void set_z(f32_t z) { set_position(fan::vec3(get_position().x, get_position().y, z)); }

		fan::vec3 get_position() const {
			auto shape_type = get_shape_type();
			return gloco->shape_functions[shape_type].get_position(this);
		}
		f32_t get_x() const { return get_position().x; }
		f32_t get_y() const { return get_position().y; }
		f32_t get_z() const { return get_position().z; }

		void set_size(const fan::vec2& size) {
			gloco->shape_functions[get_shape_type()].set_size(this, size);
		}

		void set_size3(const fan::vec3& size) {
			gloco->shape_functions[get_shape_type()].set_size3(this, size);
		}

		// returns half extents of draw
		fan::vec2 get_size() const {
			return gloco->shape_functions[get_shape_type()].get_size(this);
		}

		fan::vec3 get_size3() {
			return gloco->shape_functions[get_shape_type()].get_size3(this);
		}

		void set_rotation_point(const fan::vec2& rotation_point) {
			gloco->shape_functions[get_shape_type()].set_rotation_point(this, rotation_point);
		}

		fan::vec2 get_rotation_point() const {
			return gloco->shape_functions[get_shape_type()].get_rotation_point(this);
		}

		void set_color(const fan::color& color) {
			gloco->shape_functions[get_shape_type()].set_color(this, color);
		}

		fan::color get_color() {
			return gloco->shape_functions[get_shape_type()].get_color(this);
		}

		void set_angle(const fan::vec3& angle) {
			gloco->shape_functions[get_shape_type()].set_angle(this, angle);
		}

		fan::vec3 get_angle() const {
			return gloco->shape_functions[get_shape_type()].get_angle(this);
		}

		fan::basis get_basis() const {
			auto zangle = get_angle().z;
			auto c = std::cos(zangle);
			auto s = std::sin(zangle);

			return fan::basis{
				.right = { c, s, 0 },
				.forward = { s, -c, 0 },
				.up = { 0, 0, 1 }
			};
		}

		fan::vec3 get_forward() const { return get_basis().forward; }
		fan::vec3 get_right() const { return get_basis().right; }
		fan::vec3 get_up() const { return get_basis().up; }

		fan::mat3 get_rotation_matrix() const {
			return get_basis();
		}

		fan::vec3 transform(const fan::vec3& local) const {
			// sign conflict, when forward y is -1, then moving y by -1 would move it down when we want it up
			// so flip the y sign since coordinate system is +y down
			fan::vec3 flipped_y{ local.x, -local.y, local.z };
			return get_position() + get_basis() * flipped_y;
		}

		fan::mat4 get_transform() const {
			fan::mat4 m = fan::mat4::identity();
			m = m.translate(get_position());
			m = m * get_rotation_matrix();  
			m = m.scale(get_size());        
			return m;
		}

		fan::physics::aabb_t get_aabb() const {
			fan::vec2 pos = get_position();
			fan::vec2 he = get_size(); // half extents
			f32_t cs = std::cos(get_angle().z);
			f32_t sn = std::sin(get_angle().z);
			fan::vec2 pivot = get_rotation_point();

			fan::vec2 minp(FLT_MAX, FLT_MAX), maxp(-FLT_MAX, -FLT_MAX);

			for (int i = -1; i <= 1; i += 2) {
				for (int j = -1; j <= 1; j += 2) {
					fan::vec2 r = { i * he.x - pivot.x, j * he.y - pivot.y };
					r = { r.x * cs - r.y * sn, r.x * sn + r.y * cs };
					r += pos + pivot;
					minp.x = std::min(minp.x, r.x);
					minp.y = std::min(minp.y, r.y);
					maxp.x = std::max(maxp.x, r.x);
					maxp.y = std::max(maxp.y, r.y);
				}
			}

			return { minp, maxp };
		}

		fan::vec2 get_tc_position() {
			return gloco->shape_functions[get_shape_type()].get_tc_position(this);
		}

		void set_tc_position(const fan::vec2& tc_position) {
			auto st = get_shape_type();
			gloco->shape_functions[st].set_tc_position(this, tc_position);
		}

		fan::vec2 get_tc_size() {
			return gloco->shape_functions[get_shape_type()].get_tc_size(this);
		}

		void set_tc_size(const fan::vec2& tc_size) {
			gloco->shape_functions[get_shape_type()].set_tc_size(this, tc_size);
		}

		bool load_tp(loco_t::texturepack_t::ti_t* ti) {
			auto st = get_shape_type();
			if (st == loco_t::shape_type_t::sprite || 
					st == loco_t::shape_type_t::unlit_sprite) {
				auto image = ti->image;
				set_image(image);
				set_tc_position(ti->position / gloco->image_get_data(image).size);
				set_tc_size(ti->size / gloco->image_get_data(image).size);
				if (st == loco_t::shape_type_t::sprite) {
					loco_t::sprite_t::ri_t* ram_data = (loco_t::sprite_t::ri_t*)GetData(gloco->shaper);
					ram_data->texture_pack_unique_id = ti->unique_id;
				}
				else if (st == loco_t::shape_type_t::unlit_sprite) {
					loco_t::unlit_sprite_t::ri_t* ram_data = (loco_t::unlit_sprite_t::ri_t*)GetData(gloco->shaper);
					ram_data->texture_pack_unique_id = ti->unique_id;
				}
				return false;
			}
			fan::throw_error("invalid function call for current shape:"_str + shape_names[st]);
			return true;
		}

		loco_t::texturepack_t::ti_t get_tp() {
			loco_t::texturepack_t::ti_t ti;
			ti.image = gloco->default_texture;
			auto& image_data = gloco->image_get_data(ti.image);
			ti.position = get_tc_position() * image_data.size;
			ti.size = get_tc_size() * image_data.size;
			return ti;
			//return gloco->shape_functions[gloco->shaper.GetSTI(*this)].get_tp(this);
		}

		bool set_tp(loco_t::texturepack_t::ti_t* ti) {
			return load_tp(ti);
		}

		loco_t::camera_t get_camera() const {
			return gloco->shape_functions[get_shape_type()].get_camera(this);
		}

		void set_camera(loco_t::camera_t camera) {
			gloco->shape_functions[get_shape_type()].set_camera(this, camera);
		}

		loco_t::viewport_t get_viewport() const {
			return gloco->shape_functions[get_shape_type()].get_viewport(this);
		}

		void set_viewport(loco_t::viewport_t viewport) {
			gloco->shape_functions[get_shape_type()].set_viewport(this, viewport);
		}

		void set_render_view(const loco_t::render_view_t& render_view) {
			set_camera(render_view.camera);
			set_viewport(render_view.viewport);
		}

		fan::vec2 get_grid_size() {
			return gloco->shape_functions[get_shape_type()].get_grid_size(this);
		}

		void set_grid_size(const fan::vec2& grid_size) {
			gloco->shape_functions[get_shape_type()].set_grid_size(this, grid_size);
		}

		loco_t::image_t get_image() const {
			if (gloco->shape_functions[get_shape_type()].get_image) {
				return gloco->shape_functions[get_shape_type()].get_image(this);
			}
			return gloco->default_texture;
		}

		void set_image(loco_t::image_t image) {
			gloco->shape_functions[get_shape_type()].set_image(this, image);
		}

		fan::graphics::image_data_t& get_image_data() {
			return gloco->shape_functions[get_shape_type()].get_image_data(this);
		}

		std::array<loco_t::image_t, 30> get_images() {
			auto shape_type = get_shape_type();
			if (shape_type == shape_type_t::sprite) {
				return ((sprite_t::ri_t*)ShapeID_t::GetData(gloco->shaper))->images;
			}
			else if (shape_type == shape_type_t::unlit_sprite) {
				return ((unlit_sprite_t::ri_t*)ShapeID_t::GetData(gloco->shaper))->images;
			}
			else if (shape_type == shape_type_t::universal_image_renderer) {
				std::array<loco_t::image_t, 30> ret;
				auto uni_images = ((universal_image_renderer_t::ri_t*)ShapeID_t::GetData(gloco->shaper))->images_rest;
				std::copy(uni_images.begin(), uni_images.end(), ret.begin());

				return ret;
			}
#if fan_debug >= fan_debug_medium
			fan::throw_error("only for sprite and unlit_sprite");
#endif
			return {};
		}

		void set_images(const std::array<loco_t::image_t, 30>& images) {
			auto shape_type = get_shape_type();
			if (shape_type == shape_type_t::sprite) {
				((sprite_t::ri_t*)ShapeID_t::GetData(gloco->shaper))->images = images;
			}
			else if (shape_type == shape_type_t::unlit_sprite) {
				((unlit_sprite_t::ri_t*)ShapeID_t::GetData(gloco->shaper))->images = images;
			}
#if fan_debug >= fan_debug_medium
			else {
				fan::throw_error("only for sprite and unlit_sprite");
			}
#endif
		}

		f32_t get_parallax_factor() {
			return gloco->shape_functions[get_shape_type()].get_parallax_factor(this);
		}

		void set_parallax_factor(f32_t parallax_factor) {
			gloco->shape_functions[get_shape_type()].set_parallax_factor(this, parallax_factor);
		}

		uint32_t get_flags() {
			auto f = gloco->shape_functions[get_shape_type()].get_flags;
			if (f) {
				return f(this);
			}
			return 0;
		}

		void set_flags(uint32_t flag) {
			auto st = get_shape_type();
			return gloco->shape_functions[st].set_flags(this, flag);
		}

		f32_t get_radius() {
			return gloco->shape_functions[get_shape_type()].get_radius(this);
		}

		fan::vec3 get_src() {
			return gloco->shape_functions[get_shape_type()].get_src(this);
		}

		fan::vec3 get_dst() {
			return gloco->shape_functions[get_shape_type()].get_dst(this);
		}

		f32_t get_outline_size() {
			return gloco->shape_functions[get_shape_type()].get_outline_size(this);
		}

		fan::color get_outline_color() {
			return gloco->shape_functions[get_shape_type()].get_outline_color(this);
		}

		void set_outline_color(const fan::color& color) {
			return gloco->shape_functions[get_shape_type()].set_outline_color(this, color);
		}

		void reload(uint8_t format, void** image_data, const fan::vec2& image_size) {
			auto& settings = gloco->image_get_settings(get_image());
			uint32_t filter = settings.min_filter;
			loco_t::universal_image_renderer_t::ri_t& ri = *(loco_t::universal_image_renderer_t::ri_t*)GetData(gloco->shaper);
			uint8_t image_count_new = fan::graphics::get_channel_amount(format);
			if (format != ri.format) {
				auto sti = get_shape_type();
				uint8_t* key_pack = gloco->shaper.GetKeys(*this);
				loco_t::image_t vi_image = shaper_get_key_safe(loco_t::image_t, texture_t, image);


				auto shader = gloco->shaper.GetShader(sti);
				gloco->shader_set_vertex(
					shader,
					loco_t::read_shader("shaders/opengl/2D/objects/pixel_format_renderer.vs")
				);
				{
					std::string fs;
					switch (format) {
					case fan::graphics::image_format::yuv420p: {
						fs = loco_t::read_shader("shaders/opengl/2D/objects/yuv420p.fs");
						break;
					}
					case fan::graphics::image_format::nv12: {
						fs = loco_t::read_shader("shaders/opengl/2D/objects/nv12.fs");
						break;
					}
					default: {
						fan::throw_error("unimplemented format");
					}
					}
					gloco->shader_set_fragment(shader, fs);
					gloco->shader_compile(shader);
				}

				uint8_t image_count_old = fan::graphics::get_channel_amount(ri.format);
				if (image_count_new < image_count_old) {
					uint8_t textures_to_remove = image_count_old - image_count_new;
					if (vi_image.iic() || vi_image == gloco->default_texture) { // uninitialized
						textures_to_remove = 0;
					}
					for (int i = 0; i < textures_to_remove; ++i) {
						int index = image_count_old - i - 1; // not tested
						if (index == 0) {
							gloco->image_erase(vi_image);
							set_image(gloco->default_texture);
						}
						else {
							gloco->image_erase(ri.images_rest[index - 1]);
							ri.images_rest[index - 1] = gloco->default_texture;
						}
					}
				}
				else if (image_count_new > image_count_old) {
					loco_t::image_t images[4];
					for (uint32_t i = image_count_old; i < image_count_new; ++i) {
						images[i] = gloco->image_create();
					}
					set_image(images[0]);
					std::copy(&images[1], &images[0] + ri.images_rest.size(), ri.images_rest.data());
				}
			}

			auto vi_image = get_image();

			for (uint32_t i = 0; i < image_count_new; ++i) {
				if (i == 0) {
					if (vi_image.iic() || vi_image == gloco->default_texture) {
						vi_image = gloco->image_create();
						set_image(vi_image);
					}
				}
				else {
					if (ri.images_rest[i - 1].iic() || ri.images_rest[i - 1] == gloco->default_texture) {
						ri.images_rest[i - 1] = gloco->image_create();
					}
				}
			}

			for (uint32_t i = 0; i < image_count_new; i++) {
				fan::image::info_t image_info;
				image_info.data = image_data[i];
				image_info.size = fan::graphics::get_image_sizes(format, image_size)[i];
				auto lp = fan::graphics::get_image_properties<loco_t::image_load_properties_t>(format)[i];
				lp.min_filter = filter;
				if (filter == fan::graphics::image_filter::linear ||
					filter == fan::graphics::image_filter::nearest) {
					lp.mag_filter = filter;
				}
				else {
					lp.mag_filter = fan::graphics::image_filter::linear;
				}
				if (i == 0) {
					gloco->image_reload(
						vi_image,
						image_info,
						lp
					);
				}
				else {
					gloco->image_reload(
						ri.images_rest[i - 1],
						image_info,
						lp
					);
				}
			}
			ri.format = format;
		}

		void reload(uint8_t format, const fan::vec2& image_size) {
			auto& settings = gloco->image_get_settings(get_image());
			void* data[4]{};
			reload(format, data, image_size);
		}

		// universal image specific
		void reload(uint8_t format, loco_t::image_t images[4]) {
			loco_t::universal_image_renderer_t::ri_t& ri = *(loco_t::universal_image_renderer_t::ri_t*)GetData(gloco->shaper);
			uint8_t image_count_new = fan::graphics::get_channel_amount(format);
			if (format != ri.format) {
				auto sti = gloco->shaper.ShapeList[*this].sti;
				uint8_t* key_pack = gloco->shaper.GetKeys(*this);
				loco_t::image_t vi_image = shaper_get_key_safe(loco_t::image_t, texture_t, image);


				auto shader = gloco->shaper.GetShader(sti);
				gloco->shader_set_vertex(
					shader,
					loco_t::read_shader("shaders/opengl/2D/objects/pixel_format_renderer.vs")
				);
				{
					std::string fs;
					switch (format) {
					case fan::graphics::image_format::yuv420p: {
						fs = loco_t::read_shader("shaders/opengl/2D/objects/yuv420p.fs");
						break;
					}
					case fan::graphics::image_format::nv12: {
						fs = loco_t::read_shader("shaders/opengl/2D/objects/nv12.fs");
						break;
					}
					default: {
						fan::throw_error("unimplemented format");
					}
					}
					gloco->shader_set_fragment(shader, fs);
					gloco->shader_compile(shader);
				}
				set_image(images[0]);
				std::copy(&images[1], &images[0] + ri.images_rest.size(), ri.images_rest.data());
				ri.format = format;
			}
		}

		void set_line(const fan::vec2& src, const fan::vec2& dst) {
			auto st = get_shape_type();
			if (st == loco_t::shape_type_t::line) {
				auto data = reinterpret_cast<loco_t::line_t::vi_t*>(GetRenderData(gloco->shaper));
				data->src = fan::vec3(src.x, src.y, 0);
				data->dst = fan::vec3(dst.x, dst.y, 0);
				if (gloco->window.renderer == loco_t::renderer_t::opengl) {
					auto& data = gloco->shaper.ShapeList[*this];
					gloco->shaper.ElementIsPartiallyEdited(
						data.sti,
						data.blid,
						data.ElementIndex,
						fan::member_offset(&loco_t::line_t::vi_t::src),
						sizeof(loco_t::line_t::vi_t::src)
					);
					gloco->shaper.ElementIsPartiallyEdited(
						data.sti,
						data.blid,
						data.ElementIndex,
						fan::member_offset(&loco_t::line_t::vi_t::dst),
						sizeof(loco_t::line_t::vi_t::dst)
					);
				}
			}
		#if defined(fan_3D)
			if (st == loco_t::shape_type_t::line3d) {
				auto data = reinterpret_cast<loco_t::line3d_t::vi_t*>(GetRenderData(gloco->shaper));
				data->src = fan::vec3(src.x, src.y, 0);
				data->dst = fan::vec3(dst.x, dst.y, 0);
				if (gloco->window.renderer == loco_t::renderer_t::opengl) {
					auto& data = gloco->shaper.ShapeList[*this];
					gloco->shaper.ElementIsPartiallyEdited(
						data.sti,
						data.blid,
						data.ElementIndex,
						fan::member_offset(&loco_t::line3d_t::vi_t::src),
						sizeof(loco_t::line3d_t::vi_t::src)
					);
					gloco->shaper.ElementIsPartiallyEdited(
						data.sti,
						data.blid,
						data.ElementIndex,
						fan::member_offset(&loco_t::line3d_t::vi_t::dst),
						sizeof(loco_t::line3d_t::vi_t::dst)
					);
				}
			}
			#endif
		}

		bool is_mouse_inside() {
			switch (get_shape_type()) {
			case shape_type_t::rectangle: {
				return fan_2d::collision::rectangle::point_inside_no_rotation(
					gloco->get_mouse_position(get_camera(), get_viewport()),
					get_position(),
					get_size()
				);
			}
			default: {
				break;
			}
			}
		}

		bool intersects(const loco_t::shape_t& shape) {
			switch (get_shape_type()) {
			case shape_type_t::capsule: // inaccurate
			case shape_type_t::shader_shape:
			case shape_type_t::unlit_sprite:
			case shape_type_t::sprite:
			case shape_type_t::rectangle: {
				fan::physics::aabb_t aabb = get_aabb();
				fan::physics::aabb_t aabb2 = shape.get_aabb();
				return fan_2d::collision::rectangle::check_collision(
					 aabb.min + (aabb.max - aabb.min) / 2.f,
					(aabb.max - aabb.min) / 2.f,
					 aabb2.min + (aabb2.max - aabb2.min) / 2.f,
					(aabb2.max - aabb2.min) / 2.f
				);
			}
			}
			fan::throw_error("todo");
			return true;
		}
		bool collides(const loco_t::shape_t& shape) {
			return intersects(shape);
		}

		void add_existing_animation(animation_nr_t nr) {
			if (get_shape_type() == loco_t::shape_type_t::sprite) {
				auto& ri = shape_get_ri(sprite);
				auto& animation = gloco->get_sprite_sheet_animation(nr);
				ri.shape_animations = gloco->add_existing_sprite_sheet_shape_animation(nr, ri.shape_animations, animation);
				ri.current_animation = gloco->shape_animations[ri.shape_animations].back();
			}
			else {
				fan::throw_error("Unimplemented for this shape");
			}
		}

		// sprite sheet - sprite specific
		void set_sprite_sheet_next_frame(int advance = 1) {
			if (get_shape_type() == loco_t::shape_type_t::sprite) {
				auto& ri = *(loco_t::sprite_t::ri_t*)GetData(gloco->shaper);
				auto found = gloco->all_animations.find(ri.current_animation);
				if (found == gloco->all_animations.end()) {
					fan::throw_error("current_animation not found");
				}
				auto& animation = found->second;
				loco_t::sprite_sheet_data_t& sheet_data = ri.sprite_sheet_data;
				int actual_frame = animation.selected_frames[sheet_data.current_frame];

				// Find which image this frame belongs to and the local frame within that image
				int image_index = 0;
				int local_frame = actual_frame;
				int frame_count = 0;

				for (int i = 0; i < animation.images.size(); ++i) {
					int frames_in_this_image = animation.images[i].hframes * animation.images[i].vframes;
					if (actual_frame < frame_count + frames_in_this_image) {
						image_index = i;
						local_frame = actual_frame - frame_count;
						break;
					}
					frame_count += frames_in_this_image;
				}

				auto& current_image = animation.images[image_index];
				set_image(current_image.image);
				sheet_data.current_frame += advance;
				sheet_data.current_frame %= animation.selected_frames.size();
				sheet_data.update_timer.restart();

				fan::vec2 tc_size = fan::vec2(1.0 / current_image.hframes, 1.0 / current_image.vframes);
				int frame_x = local_frame % current_image.hframes;
				int frame_y = local_frame / current_image.hframes;
				set_tc_position(fan::vec2(
					frame_x * tc_size.x,
					frame_y * tc_size.y
				));
				fan::vec2 sign = get_tc_size().sign();
				set_tc_size(tc_size * sign);
			}
			else {
				fan::throw_error("Unimplemented for this shape");
			}
		}
		// Takes in seconds
		void set_sprite_sheet_fps(f32_t fps) {
			if (get_shape_type() == loco_t::shape_type_t::sprite) {
				auto& ri = *(loco_t::sprite_t::ri_t*)GetData(gloco->shaper);
				loco_t::sprite_sheet_data_t& sheet_data = ri.sprite_sheet_data;
				for (auto& animation_nrs : gloco->shape_animations[ri.shape_animations]) {
					gloco->get_sprite_sheet_animation(animation_nrs).fps = fps;
				}
				if (sheet_data.update_timer.m_time == (uint64_t)-1) {
					sheet_data.update_timer.start(1.0 / fps * 1e+9);
				}
				else {
					sheet_data.update_timer.set_time(1.0 / fps * 1e+9);
				}
			}
			else {
				fan::throw_error("Unimplemented for this shape");
			}
		}
		bool has_animation() {
			if (get_shape_type() != loco_t::shape_type_t::sprite) {
				return false;
			}

			auto& ri = *(loco_t::sprite_t::ri_t*)GetData(gloco->shaper);
			loco_t::sprite_sheet_data_t& sheet_data = ri.sprite_sheet_data;
			return sheet_data.update_timer.started();
		}
		static void sprite_sheet_frame_update_cb(loco_t* loco, shape_t* shape) {
			auto& ri = *(loco_t::sprite_t::ri_t*)shape->GetData(gloco->shaper);
			loco_t::sprite_sheet_data_t& sheet_data = ri.sprite_sheet_data;
			if (sheet_data.update_timer) { // is it possible to just remove frame_udpate_cb if its not valid
				if (ri.current_animation) {
					auto& selected_frames = loco->all_animations[ri.current_animation].selected_frames;
					if (selected_frames.empty()) {
						return;
					}
					shape->set_sprite_sheet_next_frame();
				}
				else {
					shape->set_sprite_sheet_next_frame();
				}
				sheet_data.update_timer.restart();
			}
		}
		
		// returns currently active sprite sheet animation
		sprite_sheet_animation_t& get_sprite_sheet_animation() {
			return gloco->get_sprite_sheet_animation(shape_get_ri(sprite).current_animation);
		}

		void start_sprite_sheet_animation() {
			auto& ri = shape_get_ri(sprite);
			auto& current_anim = get_sprite_sheet_animation();

			loco_t::sprite_sheet_data_t& sheet_data = ri.sprite_sheet_data;
			sheet_data.current_frame = 0;
			sheet_data.update_timer.start(1.0 / current_anim.fps * 1e+9);
			int actual_frame = current_anim.selected_frames[sheet_data.current_frame];
			if (current_anim.images.empty()) {
				return;
			}
			auto& current_image = current_anim.images[actual_frame];

			set_tc_position(fan::vec2(0, 0));
			set_tc_size(fan::vec2(1.0 / current_image.hframes, 1.0 / current_image.vframes));
			// No frames to process, remove frame update function
			//if (current_image.vframes * current_image.hframes == 1) {
			//  if (sheet_data.frame_update_nr) {
			//    gloco->m_update_callback.unlrec(sheet_data.frame_update_nr);
			//    sheet_data.frame_update_nr.sic();
			//  }
			//}
			//else {
				if (sheet_data.frame_update_nr == false) {
					sheet_data.frame_update_nr = gloco->m_update_callback.NewNodeLast();
				}
				gloco->m_update_callback[sheet_data.frame_update_nr] = [nr = NRI](loco_t* loco) {
					sprite_sheet_frame_update_cb(loco, (loco_t::shape_t*)&nr);
				};
			//}
		}

		// overwrites 'ri.current_animation' animation
		void set_sprite_sheet_animation(const sprite_sheet_animation_t& animation) {
			if (get_shape_type() == loco_t::shape_type_t::sprite) {
				auto& ri = shape_get_ri(sprite);
				auto& previous_anim = gloco->get_sprite_sheet_animation(ri.current_animation);
				{
					auto found = gloco->shape_animation_lookup_table.find(std::make_pair(ri.shape_animations, previous_anim.name));
					if (found != gloco->shape_animation_lookup_table.end()) {
						gloco->shape_animation_lookup_table.erase(found);
					}
				}
				previous_anim = animation;
				gloco->shape_animation_lookup_table[std::make_pair(ri.shape_animations, animation.name)] = ri.current_animation;

				start_sprite_sheet_animation();
			}
			else {
				fan::throw_error("Unimplemented for this shape");
			}
		}

		void add_sprite_sheet_animation(const sprite_sheet_animation_t& animation) {
			if (get_shape_type() == loco_t::shape_type_t::sprite) {
				auto& ri = shape_get_ri(sprite);
				// adds animation to 
				ri.shape_animations = gloco->add_sprite_sheet_shape_animation(ri.shape_animations, animation);
				ri.current_animation = gloco->shape_animations[ri.shape_animations].back();
				start_sprite_sheet_animation();
			}
			else {
				fan::throw_error("Unimplemented for this shape");
			}
		}

		void set_sprite_sheet_frames(uint32_t image_index, int horizontal_frames, int vertical_frames) {
			if (get_shape_type() == loco_t::shape_type_t::sprite) {
				auto& current_anim = get_sprite_sheet_animation();
				current_anim.images[image_index].hframes = horizontal_frames;
				current_anim.images[image_index].vframes = vertical_frames;
				start_sprite_sheet_animation();
			}
			else {
				fan::throw_error("Unimplemented for this shape");
			}
		}

		void set_light_position(const fan::vec3& new_pos) {
			if (get_shape_type() != loco_t::shape_type_t::shadow) {
				fan::throw_error("invalid function call for current shape");
			}
			reinterpret_cast<loco_t::shadow_t::vi_t*>(GetRenderData(gloco->shaper))->light_position = new_pos;
			if (gloco->window.renderer == loco_t::renderer_t::opengl) {
				auto& data = gloco->shaper.ShapeList[*this];
				gloco->shaper.ElementIsPartiallyEdited(
					data.sti,
					data.blid,
					data.ElementIndex,
					fan::member_offset(&loco_t::shadow_t::vi_t::light_position),
					sizeof(loco_t::shadow_t::vi_t::light_position)
				);
			}
		}
		void set_light_radius(f32_t radius) {
			if (get_shape_type() != loco_t::shape_type_t::shadow) {
				fan::throw_error("invalid function call for current shape");
			}

			reinterpret_cast<loco_t::shadow_t::vi_t*>(GetRenderData(gloco->shaper))->light_radius = radius;
			if (gloco->window.renderer == loco_t::renderer_t::opengl) {
				auto& data = gloco->shaper.ShapeList[*this];
				gloco->shaper.ElementIsPartiallyEdited(
					data.sti,
					data.blid,
					data.ElementIndex,
					fan::member_offset(&loco_t::shadow_t::vi_t::light_radius),
					sizeof(loco_t::shadow_t::vi_t::light_radius)
				);
			}
		}

		// for line
		void set_thickness(f32_t new_thickness) {
		#if fan_debug >= 3
			if (get_shape_type() != loco_t::shape_type_t::line) {
				fan::throw_error("Invalid function call 'set_thickness', shape was not line");
			}
		#endif
			((loco_t::line_t::vi_t*)GetRenderData(gloco->shaper))->thickness = new_thickness;
			auto& data = gloco->shaper.ShapeList[*this];
			gloco->shaper.ElementIsPartiallyEdited(
				data.sti,
				data.blid,
				data.ElementIndex,
				fan::member_offset(&loco_t::line_t::vi_t::thickness),
				sizeof(loco_t::line_t::vi_t::thickness)
			);
		}

		void apply_floating_motion(
			f32_t time = gloco->start_time.seconds(),
			f32_t amplitude = 5.f,
			f32_t speed = 2.f,
			f32_t phase = 0.f
		) {
			f32_t y = std::sin(time * speed + phase) * amplitude;
			set_y(y);
		}

	private:
	};

	#undef shape_get_vi
	#undef shape_get_ri

	struct light_flags_e {
		enum {
			circle = 0,
			square = 1 << 0,
			lava = 1 << 1, // does this belong here
			additive = 1 << 2,
			multiplicative = 1 << 3,
		};
	};

	struct light_t {

		static inline shaper_t::KeyTypeIndex_t shape_type = shape_type_t::light;
		static constexpr int kpi = kp::light;

#pragma pack(push, 1)

		struct vi_t {
			fan::vec3 position;
			f32_t parallax_factor;
			fan::vec2 size;
			fan::vec2 rotation_point;
			fan::color color;
			uint32_t flags = 0;
			fan::vec3 angle;
		};;
		struct ri_t {

		};

#pragma pack(pop)

		inline static std::vector<shape_gl_init_t> locations = {
			shape_gl_init_t{{0, "in_position"}, 3, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position)},
			shape_gl_init_t{{1, "in_parallax_factor"}, 1, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, parallax_factor))},
			shape_gl_init_t{{2, "in_size"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, size))},
			shape_gl_init_t{{3, "in_rotation_point"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, rotation_point))},
			shape_gl_init_t{{4, "in_color"}, 4, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color))},
			shape_gl_init_t{{5, "in_flags"}, 1, GL_UNSIGNED_INT , sizeof(vi_t), (void*)(offsetof(vi_t, flags))},
			shape_gl_init_t{{6, "in_angle"}, 3, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, angle))}
		};

		struct properties_t {
			using type_t = light_t;


			fan::vec3 position = 0;
			f32_t parallax_factor = 0;
			fan::vec2 size = 0;
			fan::vec2 rotation_point = 0;
			fan::color color = fan::colors::white;
			uint32_t flags = 0;
			fan::vec3 angle = 0;

			loco_t::camera_t camera = gloco->orthographic_render_view.camera;
			loco_t::viewport_t viewport = gloco->orthographic_render_view.viewport;

			uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
			uint32_t vertex_count = 6;
		};

		shape_t push_back(const properties_t& properties) {
			vi_t vi;
			vi.position = properties.position;
			vi.parallax_factor = properties.parallax_factor;
			vi.size = properties.size;
			vi.rotation_point = properties.rotation_point;
			vi.color = properties.color;
			vi.flags = properties.flags;
			vi.angle = properties.angle;
			ri_t ri;

			return shape_add(shape_type, vi, ri,
				Key_e::light, (uint8_t)0,
				Key_e::viewport, properties.viewport,
				Key_e::camera, properties.camera,
				Key_e::ShapeType, shape_type,
				Key_e::draw_mode, properties.draw_mode,
				Key_e::vertex_count, properties.vertex_count
			);
		}
	}light;

	struct line_t {

		static constexpr shaper_t::KeyTypeIndex_t shape_type = shape_type_t::line;
		static constexpr int kpi = kp::common;

#pragma pack(push, 1)

		struct vi_t {
			fan::color color;
			fan::vec3 src;
			fan::vec2 dst;
			f32_t thickness;
			f32_t pad;
		};
		struct ri_t {

		};

#pragma pack(pop)

		inline static std::vector<shape_gl_init_t> locations = {
			shape_gl_init_t{{0, "in_color"}, decltype(vi_t::color)::size(), GL_FLOAT, sizeof(line_t::vi_t), (void*)offsetof(line_t::vi_t, color)},
			shape_gl_init_t{{1, "in_src"}, decltype(vi_t::src)::size(), GL_FLOAT, sizeof(line_t::vi_t), (void*)offsetof(line_t::vi_t, src)},
			shape_gl_init_t{{2, "in_dst"}, decltype(vi_t::dst)::size(), GL_FLOAT, sizeof(line_t::vi_t), (void*)offsetof(line_t::vi_t, dst)},
			shape_gl_init_t{{3, "line_thickness"}, 1, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, thickness)}
		};

		struct properties_t {
			using type_t = line_t;

			fan::color color = fan::colors::white;
			fan::vec3 src;
			fan::vec3 dst;
			f32_t thickness = 4.0f;

			bool blending = true;

			loco_t::camera_t camera = gloco->orthographic_render_view.camera;
			loco_t::viewport_t viewport = gloco->orthographic_render_view.viewport;

			uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
			uint32_t vertex_count = 6;
		};


		shape_t push_back(const properties_t& properties) {
			vi_t vi;
			vi.src = properties.src;
			vi.dst = properties.dst;
			vi.color = properties.color;
			vi.thickness = properties.thickness;
			ri_t ri;

			return shape_add(shape_type, vi, ri,
				Key_e::depth, (uint16_t)properties.src.z,
				Key_e::blending, (uint8_t)properties.blending,
				Key_e::viewport, properties.viewport,
				Key_e::camera, properties.camera,
				Key_e::ShapeType, shape_type,
				Key_e::draw_mode, properties.draw_mode,
				Key_e::vertex_count, properties.vertex_count
			);
		}

	}line;

	struct rectangle_t {

		static constexpr shaper_t::KeyTypeIndex_t shape_type = shape_type_t::rectangle;
		static constexpr int kpi = kp::common;

#pragma pack(push, 1)

		struct vi_t {
			fan::vec3 position;
			f32_t pad;
			fan::vec2 size;
			fan::vec2 rotation_point;
			fan::color color;
			fan::color outline_color;
			fan::vec3 angle;
			f32_t pad2;
		};
		struct ri_t {

		};

#pragma pack(pop)

		// accounts padding
		inline static std::vector<shape_gl_init_t> locations = {
			shape_gl_init_t{{0, "in_position"}, 4, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position)},
			shape_gl_init_t{{1, "in_size"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, size))},
			shape_gl_init_t{{2, "in_rotation_point"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, rotation_point))},
			shape_gl_init_t{{3, "in_color"}, 4, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color))},
			shape_gl_init_t{{4, "in_outline_color"}, 4, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, outline_color))},
			shape_gl_init_t{{5, "in_angle"}, 4, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, angle))}
		};

		struct properties_t {
			using type_t = rectangle_t;

			fan::vec3 position = fan::vec3(fan::vec2(gloco->window.get_size() / 2), 0);
			fan::vec2 size = fan::vec2(32, 32);
			fan::color color = fan::colors::white;
			fan::color outline_color = color;
			bool blending = false;
			fan::vec3 angle = 0;
			fan::vec2 rotation_point = 0;

			loco_t::camera_t camera = gloco->orthographic_render_view.camera;
			loco_t::viewport_t viewport = gloco->orthographic_render_view.viewport;
			uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
			uint32_t vertex_count = 6;
		};


		shape_t push_back(const properties_t& properties) {
			vi_t vi;
			vi.position = properties.position;
			vi.size = properties.size;
			vi.color = properties.color;
			vi.outline_color = properties.outline_color;
			vi.angle = properties.angle;
			vi.rotation_point = properties.rotation_point;
			ri_t ri;

			return shape_add(shape_type, vi, ri,
				Key_e::depth, (uint16_t)properties.position.z,
				Key_e::blending, (uint8_t)properties.blending,
				Key_e::viewport, properties.viewport,
				Key_e::camera, properties.camera,
				Key_e::ShapeType, shape_type,
				Key_e::draw_mode, properties.draw_mode,
				Key_e::vertex_count, properties.vertex_count
			);
		}

	}rectangle;

	//----------------------------------------------------------

#pragma pack(push, 1)
	struct sprite_sheet_data_t {
		// current_frame in 'selected_frames'
		int current_frame;
		fan::time::timer update_timer;
		// sprite sheet update function nr
		loco_t::update_callback_nr_t frame_update_nr;
	};
#pragma pack(pop)

	struct sprite_t {

		static constexpr shaper_t::KeyTypeIndex_t shape_type = shape_type_t::sprite;
		static constexpr int kpi = kp::texture;

#pragma pack(push, 1)

		struct vi_t {
			fan::vec3 position;
			f32_t parallax_factor;
			fan::vec2 size;
			fan::vec2 rotation_point;
			fan::color color;
			fan::vec3 angle;
			uint32_t flags;
			fan::vec2 tc_position;
			fan::vec2 tc_size;
			f32_t seed;
			fan::vec3 pad;
		};
		struct ri_t {
			// main image + light buffer + 30
			std::array<loco_t::image_t, 30> images; // what about tc_pos and tc_size
			texture_pack_unique_t texture_pack_unique_id;

			sprite_sheet_data_t sprite_sheet_data;

			animation_shape_nr_t shape_animations; 
			animation_nr_t current_animation;
		};

#pragma pack(pop)

		inline static std::vector<shape_gl_init_t> locations = {
			shape_gl_init_t{{0, "in_position"}, 3, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position)},
			shape_gl_init_t{{1, "in_parallax_factor"}, 1, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, parallax_factor))},
			shape_gl_init_t{{2, "in_size"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, size))},
			shape_gl_init_t{{3, "in_rotation_point"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, rotation_point))},
			shape_gl_init_t{{4, "in_color"}, 4, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color))},
			shape_gl_init_t{{5, "in_angle"}, 3, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, angle))},
			shape_gl_init_t{{6, "in_flags"}, 1, GL_UNSIGNED_INT , sizeof(vi_t), (void*)(offsetof(vi_t, flags))},
			shape_gl_init_t{{7, "in_tc_position"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, tc_position))},
			shape_gl_init_t{{8, "in_tc_size"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, tc_size))},
			shape_gl_init_t{{9, "in_seed"}, 1, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, seed)},
		};

		struct properties_t {
			using type_t = sprite_t;

			fan::vec3 position = fan::vec3(fan::vec2(gloco->window.get_size() / 2), 0);
			f32_t parallax_factor = 0;
			fan::vec2 size = fan::vec2(32, 32);
			fan::vec2 rotation_point = 0;
			fan::color color = fan::colors::white;
			fan::vec3 angle = fan::vec3(0);
			uint32_t flags = light_flags_e::circle | light_flags_e::multiplicative;
			fan::vec2 tc_position = 0;
			fan::vec2 tc_size = 1;
			f32_t seed = 0;
			texture_pack_unique_t texture_pack_unique_id;
			animation_shape_nr_t shape_animations;
			animation_nr_t current_animation;

			bool load_tp(loco_t::texturepack_t::ti_t* ti) {
				auto& im = ti->image;
				image = im;
				auto& img = gloco->image_get_data(im);
				tc_position = ti->position / img.size;
				tc_size = ti->size / img.size;
				texture_pack_unique_id = ti->unique_id;
				return 0;
			}

			bool blending = false;

			loco_t::image_t image = gloco->default_texture;
			std::array<loco_t::image_t, 30> images;

			loco_t::camera_t camera = gloco->orthographic_render_view.camera;
			loco_t::viewport_t viewport = gloco->orthographic_render_view.viewport;
			uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
			uint32_t vertex_count = 6;
		};

		shape_t push_back(const properties_t& properties) {

			bool uses_texture_pack = properties.texture_pack_unique_id.iic() == false && gloco->texture_pack;
			loco_t::texturepack_t::ti_t ti;
			if (uses_texture_pack) {
				uses_texture_pack = !gloco->texture_pack.qti(gloco->texture_pack[properties.texture_pack_unique_id].name, &ti);
				if (uses_texture_pack) {
					auto& img = gloco->image_get_data(gloco->texture_pack.get_pixel_data(properties.texture_pack_unique_id).image);
					ti.position /= img.size;
					ti.size /= img.size;
				}
			}

			vi_t vi;
			vi.position = properties.position;
			vi.size = properties.size;
			vi.rotation_point = properties.rotation_point;
			vi.color = properties.color;
			vi.angle = properties.angle;
			vi.flags = properties.flags;
			vi.tc_position = uses_texture_pack ? ti.position : properties.tc_position;
			vi.tc_size = uses_texture_pack ? ti.size : properties.tc_size;
			vi.parallax_factor = properties.parallax_factor;
			vi.seed = properties.seed;

			ri_t ri;
			ri.images = properties.images;
			ri.sprite_sheet_data.current_frame = 0;
			ri.shape_animations = properties.shape_animations;
			ri.current_animation = properties.current_animation;

			if (uses_texture_pack) {
				ri.texture_pack_unique_id = properties.texture_pack_unique_id;
			}

			loco_t& loco = *OFFSETLESS(this, loco_t, sprite);
			if (loco.window.renderer == loco_t::renderer_t::opengl) {

				if ((loco.context.gl.opengl.major > 3) || (loco.context.gl.opengl.major == 3 && loco.context.gl.opengl.minor >= 3)) {
					return shape_add(
						shape_type, vi, ri,
						Key_e::depth,
						static_cast<uint16_t>(properties.position.z),
						Key_e::blending, static_cast<uint8_t>(properties.blending),
						Key_e::image, uses_texture_pack ? ti.image : properties.image,
						Key_e::viewport, properties.viewport,
						Key_e::camera, properties.camera,
						Key_e::ShapeType, shape_type,
						Key_e::draw_mode, properties.draw_mode,
						Key_e::vertex_count, properties.vertex_count
					);
				}
				else {
					// Legacy version requires array of 6 identical vertices
					vi_t vertices[6];
					for (int i = 0; i < 6; i++) {
						vertices[i] = vi;
					}

					return shape_add(
						shape_type, vertices[0], ri, Key_e::depth,
						static_cast<uint16_t>(properties.position.z),
						Key_e::blending, static_cast<uint8_t>(properties.blending),
						Key_e::image, uses_texture_pack ? ti.image : properties.image, Key_e::viewport,
						properties.viewport, Key_e::camera, properties.camera,
						Key_e::ShapeType, shape_type,
						Key_e::draw_mode, properties.draw_mode,
						Key_e::vertex_count, properties.vertex_count
					);
				}
			}
			else if (loco.window.renderer == renderer_t::vulkan) {
				return shape_add(
					shape_type, vi, ri, Key_e::depth,
					static_cast<uint16_t>(properties.position.z),
					Key_e::blending, static_cast<uint8_t>(properties.blending),
					Key_e::image, uses_texture_pack ? ti.image : properties.image, Key_e::viewport,
					properties.viewport, Key_e::camera, properties.camera,
					Key_e::ShapeType, shape_type,
					Key_e::draw_mode, properties.draw_mode,
					Key_e::vertex_count, properties.vertex_count
				);
			}

			return {};
		}

	}sprite;

	struct unlit_sprite_t {

		static constexpr shaper_t::KeyTypeIndex_t shape_type = shape_type_t::unlit_sprite;
		static constexpr int kpi = kp::texture;

#pragma pack(push, 1)

		struct vi_t {
			fan::vec3 position;
			f32_t parallax_factor;
			fan::vec2 size;
			fan::vec2 rotation_point;
			fan::color color;
			fan::vec3 angle;
			uint32_t flags;
			fan::vec2 tc_position;
			fan::vec2 tc_size;
			f32_t seed = 0;
		};
		struct ri_t {
			// main image + light buffer + 30
			std::array<loco_t::image_t, 30> images;
			texture_pack_unique_t texture_pack_unique_id;
		};

#pragma pack(pop)

		inline static std::vector<shape_gl_init_t> locations = {
			shape_gl_init_t{{0, "in_position"}, 3, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position)},
			shape_gl_init_t{{1, "in_parallax_factor"}, 1, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, parallax_factor))},
			shape_gl_init_t{{2, "in_size"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, size))},
			shape_gl_init_t{{3, "in_rotation_point"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, rotation_point))},
			shape_gl_init_t{{4, "in_color"}, 4, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color))},
			shape_gl_init_t{{5, "in_angle"}, 3, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, angle))},
			shape_gl_init_t{{6, "in_flags"}, 1, GL_UNSIGNED_INT , sizeof(vi_t), (void*)(offsetof(vi_t, flags))},
			shape_gl_init_t{{7, "in_tc_position"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, tc_position))},
			shape_gl_init_t{{8, "in_tc_size"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, tc_size))},
			shape_gl_init_t{{9, "in_seed"}, 1, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, seed)},
		};

		struct properties_t {
			using type_t = unlit_sprite_t;

			fan::vec3 position = fan::vec3(fan::vec2(gloco->window.get_size() / 2), 0);
			f32_t parallax_factor = 0;
			fan::vec2 size = 32;
			fan::vec2 rotation_point = 0;
			fan::color color = fan::colors::white;
			fan::vec3 angle = fan::vec3(0);
			int flags = 0;
			fan::vec2 tc_position = 0;
			fan::vec2 tc_size = 1;
			f32_t seed = 0;

			bool blending = false;

			loco_t::image_t image = gloco->default_texture;
			std::array<loco_t::image_t, 30> images;
			loco_t::camera_t camera = gloco->orthographic_render_view.camera;
			loco_t::viewport_t viewport = gloco->orthographic_render_view.viewport;

			uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
			uint32_t vertex_count = 6;
			texture_pack_unique_t texture_pack_unique_id;

			bool load_tp(loco_t::texturepack_t::ti_t* ti) {
				auto& im = ti->image;
				image = im;
				auto& img = gloco->image_get_data(im);
				tc_position = ti->position / img.size;
				tc_size = ti->size / img.size;
				texture_pack_unique_id = ti->unique_id;
				return 0;
			}
		};

		shape_t push_back(const properties_t& properties) {
			
			bool uses_texture_pack = properties.texture_pack_unique_id.iic() == false && gloco->texture_pack;
			loco_t::texturepack_t::ti_t ti;
			if (uses_texture_pack) {
				uses_texture_pack = !gloco->texture_pack.qti(gloco->texture_pack[properties.texture_pack_unique_id].name, &ti);
				if (uses_texture_pack) {
					auto& img = gloco->image_get_data(gloco->texture_pack.get_pixel_data(properties.texture_pack_unique_id).image);
					ti.position /= img.size;
					ti.size /= img.size;
				}
			}

			vi_t vi;
			vi.position = properties.position;
			vi.size = properties.size;
			vi.rotation_point = properties.rotation_point;
			vi.color = properties.color;
			vi.angle = properties.angle;
			vi.flags = properties.flags;
			vi.tc_position = uses_texture_pack ? ti.position : properties.tc_position;
			vi.tc_size = uses_texture_pack ? ti.size : properties.tc_size;
			vi.parallax_factor = properties.parallax_factor;
			vi.seed = properties.seed;
			ri_t ri;
			ri.images = properties.images;
			if (uses_texture_pack) {
				ri.texture_pack_unique_id = properties.texture_pack_unique_id;
			}

			return shape_add(shape_type, vi, ri,
				Key_e::depth, (uint16_t)properties.position.z,
				Key_e::blending, (uint8_t)properties.blending,
				Key_e::image, uses_texture_pack ? ti.image : properties.image,
				Key_e::viewport, properties.viewport,
				Key_e::camera, properties.camera,
				Key_e::ShapeType, shape_type,
				Key_e::draw_mode, properties.draw_mode,
				Key_e::vertex_count, properties.vertex_count
			);
		}

	}unlit_sprite;

	struct text_t {

		struct vi_t {

		};

		struct ri_t {

		};

		struct properties_t {
			using type_t = text_t;

			loco_t::camera_t camera = gloco->orthographic_render_view.camera;
			loco_t::viewport_t viewport = gloco->orthographic_render_view.viewport;

			fan::vec3 position;
			f32_t outline_size = 1;
			fan::vec2 size;
			fan::vec2 tc_position;
			fan::color color = fan::colors::white;
			fan::color outline_color;
			fan::vec2 tc_size;
			fan::vec3 angle = 0;

			std::string text;

			uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
			uint32_t vertex_count = 6;
		};

		shape_t push_back(const properties_t& properties) {
			return gloco->shaper.add(shape_type_t::text, nullptr, 0, nullptr, nullptr);
		}
	}text;

	struct circle_t {

		static constexpr shaper_t::KeyTypeIndex_t shape_type = shape_type_t::circle;
		static constexpr int kpi = kp::common;

#pragma pack(push, 1)

		struct vi_t {
			fan::vec3 position;
			f32_t radius;
			fan::vec2 rotation_point;
			fan::color color;
			fan::vec3 angle;
			uint32_t flags;
		};
		struct ri_t {

		};

#pragma pack(pop)

		inline static std::vector<shape_gl_init_t> locations = {
			shape_gl_init_t{{0, "in_position"}, 3, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position) },
			shape_gl_init_t{{1, "in_radius"}, 1, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, radius)) },
			shape_gl_init_t{{2, "in_rotation_point"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, rotation_point)) },
			shape_gl_init_t{{3, "in_color"}, 4, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color)) },
			shape_gl_init_t{{5, "in_angle"}, 3, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, angle)) },
			shape_gl_init_t{{6, "in_flags"}, 1, GL_UNSIGNED_INT , sizeof(vi_t), (void*)(offsetof(vi_t, flags))}
		};

		struct properties_t {
			using type_t = circle_t;

			fan::vec3 position = 0;
			f32_t radius = 0;
			fan::vec2 rotation_point = 0;
			fan::color color = fan::colors::white;
			fan::vec3 angle = 0;
			uint32_t flags = 0;

			bool blending = false;

			loco_t::camera_t camera = gloco->orthographic_render_view.camera;
			loco_t::viewport_t viewport = gloco->orthographic_render_view.viewport;

			uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
			uint32_t vertex_count = 6;
		};


		loco_t::shape_t push_back(const circle_t::properties_t& properties) {
			circle_t::vi_t vi;
			vi.position = properties.position;
			vi.radius = properties.radius;
			vi.rotation_point = properties.rotation_point;
			vi.color = properties.color;
			vi.angle = properties.angle;
			vi.flags = properties.flags;
			circle_t::ri_t ri;
			return shape_add(shape_type, vi, ri,
				Key_e::depth, (uint16_t)properties.position.z,
				Key_e::blending, (uint8_t)properties.blending,
				Key_e::viewport, properties.viewport,
				Key_e::camera, properties.camera,
				Key_e::ShapeType, shape_type,
				Key_e::draw_mode, properties.draw_mode,
				Key_e::vertex_count, properties.vertex_count
			);
		}

	}circle;

	struct capsule_t {

		static constexpr shaper_t::KeyTypeIndex_t shape_type = shape_type_t::capsule;
		static constexpr int kpi = kp::common;

#pragma pack(push, 1)

		struct vi_t {
			fan::vec3 position;
			fan::vec2 center0;
			fan::vec2 center1;
			f32_t radius;
			fan::vec2 rotation_point;
			fan::color color;
			fan::vec3 angle;
			uint32_t flags;
			fan::color outline_color;
		};
		struct ri_t {

		};

#pragma pack(pop)

		inline static std::vector<shape_gl_init_t> locations = {
			shape_gl_init_t{{0, "in_position"}, 3, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position) },
			shape_gl_init_t{{1, "in_center0"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, center0)) },
			shape_gl_init_t{{2, "in_center1"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, center1)) },
			shape_gl_init_t{{3, "in_radius"}, 1, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, radius)) },
			shape_gl_init_t{{4, "in_rotation_point"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, rotation_point)) },
			shape_gl_init_t{{5, "in_color"}, 4, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color)) },
			shape_gl_init_t{{6, "in_angle"}, 3, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, angle)) },
			shape_gl_init_t{{7, "in_flags"}, 1, GL_UNSIGNED_INT , sizeof(vi_t), (void*)(offsetof(vi_t, flags))},
			shape_gl_init_t{{8, "in_outline_color"}, 4, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, outline_color)) },
		};

		struct properties_t {
			using type_t = capsule_t;

			fan::vec3 position = 0;
			fan::vec2 center0 = 0;
			fan::vec2 center1 = { 0, 1.f };
			f32_t radius = 0;
			fan::vec2 rotation_point = 0;
			fan::color color = fan::colors::white;
			fan::color outline_color = color;
			fan::vec3 angle = 0;
			uint32_t flags = 0;

			bool blending = true;

			loco_t::camera_t camera = gloco->orthographic_render_view.camera;
			loco_t::viewport_t viewport = gloco->orthographic_render_view.viewport;

			uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
			uint32_t vertex_count = 6;
		};
		loco_t::shape_t push_back(const capsule_t::properties_t& properties) {
			capsule_t::vi_t vi;
			vi.position = properties.position;
			vi.center0 = properties.center0;
			vi.center1 = properties.center1;
			vi.radius = properties.radius;
			vi.rotation_point = properties.rotation_point;
			vi.color = properties.color;
			vi.outline_color = properties.outline_color;
			vi.angle = properties.angle;
			vi.flags = properties.flags;
			capsule_t::ri_t ri;
			return shape_add(shape_type, vi, ri,
				Key_e::depth, (uint16_t)properties.position.z,
				Key_e::blending, (uint8_t)properties.blending,
				Key_e::viewport, properties.viewport,
				Key_e::camera, properties.camera,
				Key_e::ShapeType, shape_type,
				Key_e::draw_mode, properties.draw_mode,
				Key_e::vertex_count, properties.vertex_count
			);
		}
	}capsule;


#pragma pack(push, 1)

	struct polygon_t {
		static constexpr shaper_t::KeyTypeIndex_t shape_type = shape_type_t::polygon;
		static constexpr int kpi = kp::common;


		struct vi_t {

		};
		struct ri_t {
			uint32_t buffer_size = 0;
			fan::opengl::core::vao_t vao;
			fan::opengl::core::vbo_t vbo;
		};

#pragma pack(pop)

		inline static std::vector<shape_gl_init_t> locations = {
			shape_gl_init_t{{0, "in_position"}, 3, GL_FLOAT, sizeof(polygon_vertex_t), (void*)(offsetof(polygon_vertex_t, position)) },
			shape_gl_init_t{{1, "in_color"}, 4, GL_FLOAT, sizeof(polygon_vertex_t), (void*)(offsetof(polygon_vertex_t, color)) },
			shape_gl_init_t{{2, "in_offset"}, 3, GL_FLOAT, sizeof(polygon_vertex_t), (void*)(offsetof(polygon_vertex_t, offset)) },
			shape_gl_init_t{{3, "in_angle"}, 3, GL_FLOAT, sizeof(polygon_vertex_t), (void*)(offsetof(polygon_vertex_t, angle)) },
			shape_gl_init_t{{4, "in_rotation_point"}, 2, GL_FLOAT, sizeof(polygon_vertex_t), (void*)(offsetof(polygon_vertex_t, rotation_point)) },
		};

		struct properties_t {
			using type_t = polygon_t;
			fan::vec3 position = 0;
			fan::vec3 angle = 0;
			fan::vec2 rotation_point = 0;
			std::vector<vertex_t> vertices;
			bool blending = true;
			loco_t::camera_t camera = gloco->orthographic_render_view.camera;
			loco_t::viewport_t viewport = gloco->orthographic_render_view.viewport;

			uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
			uint32_t vertex_count = 3;
		};
		loco_t::shape_t push_back(const properties_t& properties) {
			if (properties.vertices.empty()) {
				fan::throw_error("invalid vertices");
			}

			std::vector<loco_t::polygon_vertex_t> polygon_vertices(properties.vertices.size());
			for (std::size_t i = 0; i < properties.vertices.size(); ++i) {
				polygon_vertices[i].position = properties.vertices[i].position;
				polygon_vertices[i].color = properties.vertices[i].color;
				polygon_vertices[i].offset = properties.position;
				polygon_vertices[i].angle = properties.angle;
				polygon_vertices[i].rotation_point = properties.rotation_point;
			}

			vi_t vis;
			ri_t ri;
			ri.buffer_size = sizeof(decltype(polygon_vertices)::value_type) * polygon_vertices.size();
			ri.vao.open(gloco->context.gl);
			ri.vao.bind(gloco->context.gl);
			ri.vbo.open(gloco->context.gl, GL_ARRAY_BUFFER);
			fan::opengl::core::write_glbuffer(
				gloco->context.gl,
				ri.vbo.m_buffer,
				polygon_vertices.data(),
				ri.buffer_size,
				GL_STATIC_DRAW,
				ri.vbo.m_target
			);

			auto& shape_data = gloco->shaper.GetShapeTypes(shape_type).renderer.gl;

			fan::graphics::context_shader_t shader;
			if (!shape_data.shader.iic()) {
				shader = gloco->shader_get(shape_data.shader);
			}
			uint64_t ptr_offset = 0;
			for (shape_gl_init_t& location : locations) {
				if ((gloco->context.gl.opengl.major == 2 && gloco->context.gl.opengl.minor == 1) && !shape_data.shader.iic()) {
					location.index.first = fan_opengl_call(glGetAttribLocation(shader.gl.id, location.index.second));
				}
				fan_opengl_call(glEnableVertexAttribArray(location.index.first));
				switch (location.type) {
				case GL_UNSIGNED_INT:
				case GL_INT: {
					fan_opengl_call(glVertexAttribIPointer(location.index.first, location.size, location.type, location.stride, (void*)ptr_offset));
					break;
				}
				default: {
					fan_opengl_call(glVertexAttribPointer(location.index.first, location.size, location.type, GL_FALSE, location.stride, (void*)ptr_offset));
				}
				}
				// instancing
				if ((gloco->context.gl.opengl.major > 3) || (gloco->context.gl.opengl.major == 3 && gloco->context.gl.opengl.minor >= 3)) {
					if (shape_data.instanced) {
						fan_opengl_call(glVertexAttribDivisor(location.index.first, 1));
					}
				}
				switch (location.type) {
				case GL_FLOAT: {
					ptr_offset += location.size * sizeof(GLfloat);
					break;
				}
				case GL_UNSIGNED_INT: {
					ptr_offset += location.size * sizeof(GLuint);
					break;
				}
				default: {
					fan::throw_error_impl();
				}
				}
			}

			return shape_add(shape_type, vis, ri,
				Key_e::depth, (uint16_t)properties.position.z,
				Key_e::blending, (uint8_t)properties.blending,
				Key_e::viewport, properties.viewport,
				Key_e::camera, properties.camera,
				Key_e::ShapeType, shape_type,
				Key_e::draw_mode, properties.draw_mode,
				Key_e::vertex_count, (uint32_t)properties.vertices.size()
			);
		}
	}polygon;

	struct grid_t {

		static constexpr shaper_t::KeyTypeIndex_t shape_type = shape_type_t::grid;
		static constexpr int kpi = kp::common;

#pragma pack(push, 1)

		struct vi_t {
			fan::vec3 position;
			fan::vec2 size;
			fan::vec2 grid_size;
			fan::vec2 rotation_point;
			fan::color color;
			fan::vec3 angle;
		};
		struct ri_t {

		};

#pragma pack(pop)

		inline static std::vector<shape_gl_init_t> locations = {
			shape_gl_init_t{{0, "in_position"}, 3, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position)},
			shape_gl_init_t{{1, "in_size"}, 2, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, size)},
			shape_gl_init_t{{2, "in_grid_size"}, 2, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, grid_size)},
			shape_gl_init_t{{3, "in_rotation_point"}, 2, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, rotation_point)},
			shape_gl_init_t{{4, "in_color"}, 4, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, color)},
			shape_gl_init_t{{5, "in_angle"}, 3, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, angle)},
		};

		struct properties_t {
			using type_t = grid_t;

			fan::vec3 position = 0;
			fan::vec2 size = 0;
			fan::vec2 grid_size;
			fan::vec2 rotation_point = 0;
			fan::color color = fan::colors::white;
			fan::vec3 angle = 0;

			bool blending = false;

			loco_t::camera_t camera = gloco->orthographic_render_view.camera;
			loco_t::viewport_t viewport = gloco->orthographic_render_view.viewport;

			uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
			uint32_t vertex_count = 6;
		};

		shape_t push_back(const properties_t& properties) {
			vi_t vi;
			vi.position = properties.position;
			vi.size = properties.size;
			vi.grid_size = properties.grid_size;
			vi.rotation_point = properties.rotation_point;
			vi.color = properties.color;
			vi.angle = properties.angle;
			ri_t ri;
			return shape_add(shape_type, vi, ri,
				Key_e::depth, (uint16_t)properties.position.z,
				Key_e::blending, (uint8_t)properties.blending,
				Key_e::viewport, properties.viewport,
				Key_e::camera, properties.camera,
				Key_e::ShapeType, shape_type,
				Key_e::draw_mode, properties.draw_mode,
				Key_e::vertex_count, properties.vertex_count
			);
		}
	}grid;


	struct particles_t {

		static constexpr shaper_t::KeyTypeIndex_t shape_type = shape_type_t::particles;
		static constexpr int kpi = kp::texture;

		inline static std::vector<shape_gl_init_t> locations = {};

#pragma pack(push, 1)

		struct vi_t {

		};

		struct shapes_e {
			enum {
				circle,
				rectangle
			};
		};

		struct ri_t {

			fan::vec3 position;
			fan::vec2 size;
			fan::color color;

			uint64_t begin_time;
			uint64_t alive_time;
			uint64_t respawn_time;
			uint32_t count;
			fan::vec2 position_velocity;
			fan::vec3 angle_velocity;
			f32_t begin_angle;
			f32_t end_angle;

			fan::vec3 angle;

			fan::vec2 gap_size;
			fan::vec2 max_spread_size;
			fan::vec2 size_velocity;

			uint32_t shape;

			bool blending;
		};
#pragma pack(pop)

		struct properties_t {
			using type_t = particles_t;

			fan::vec3 position = 0;
			fan::vec2 size = 100;
			fan::color color = fan::colors::red;

			uint64_t begin_time;
			uint64_t alive_time = (uint64_t)1e+9;
			uint64_t respawn_time = 0;
			uint32_t count = 10;
			fan::vec2 position_velocity = 130;
			fan::vec3 angle_velocity = fan::vec3(0, 0, 0);
			f32_t begin_angle = 0;
			f32_t end_angle = fan::math::pi * 2;

			fan::vec3 angle = 0;

			fan::vec2 gap_size = 1;
			fan::vec2 max_spread_size = 100;
			fan::vec2 size_velocity = 1;

			uint32_t shape = shapes_e::circle;

			bool blending = true;

			loco_t::image_t image = gloco->default_texture;
			loco_t::camera_t camera = gloco->orthographic_render_view.camera;
			loco_t::viewport_t viewport = gloco->orthographic_render_view.viewport;

			uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
			uint32_t vertex_count = 6;
		};

		shape_t push_back(const properties_t& properties) {
			//KeyPack.ShapeType = shape_type;
			vi_t vi;
			ri_t ri;
			ri.position = properties.position;
			ri.size = properties.size;
			ri.color = properties.color;

			ri.begin_time = fan::time::clock::now();
			ri.alive_time = properties.alive_time;
			ri.respawn_time = properties.respawn_time;
			ri.count = properties.count;
			ri.position_velocity = properties.position_velocity;
			ri.angle_velocity = properties.angle_velocity;
			ri.begin_angle = properties.begin_angle;
			ri.end_angle = properties.end_angle;
			ri.angle = properties.angle;
			ri.gap_size = properties.gap_size;
			ri.max_spread_size = properties.max_spread_size;
			ri.size_velocity = properties.size_velocity;
			ri.shape = properties.shape;

			return shape_add(shape_type, vi, ri,
				Key_e::depth, (uint16_t)properties.position.z,
				Key_e::blending, (uint8_t)properties.blending,
				Key_e::image, properties.image,
				Key_e::viewport, properties.viewport,
				Key_e::camera, properties.camera,
				Key_e::ShapeType, shape_type,
				Key_e::draw_mode, properties.draw_mode,
				Key_e::vertex_count, properties.vertex_count
			);
		}

	}particles;

	struct universal_image_renderer_t {

		static constexpr shaper_t::KeyTypeIndex_t shape_type = shape_type_t::universal_image_renderer;
		static constexpr int kpi = kp::texture;

#pragma pack(push, 1)

		struct vi_t {
			fan::vec3 position = 0;
			fan::vec2 size = 0;
			fan::vec2 tc_position = 0;
			fan::vec2 tc_size = 1;
		};
		struct ri_t {
			std::array<loco_t::image_t, 3> images_rest; // 3 + 1 (pk)
			uint8_t format = fan::graphics::image_format::undefined;
		};

#pragma pack(pop)

		inline static std::vector<shape_gl_init_t> locations = {
			shape_gl_init_t{{0, "in_position"}, 3, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position)},
			shape_gl_init_t{{1, "in_size"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, size))},
			shape_gl_init_t{{2, "in_tc_position"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, tc_position))},
			shape_gl_init_t{{3, "in_tc_size"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, tc_size))}
			};

		struct properties_t {
			using type_t = universal_image_renderer_t;

			fan::vec3 position = 0;
			fan::vec2 size = 0;
			fan::vec2 tc_position = 0;
			fan::vec2 tc_size = 1;

			bool blending = false;

			std::array<loco_t::image_t, 4> images = {
				gloco->default_texture,
				gloco->default_texture,
				gloco->default_texture,
				gloco->default_texture
			};
			loco_t::camera_t camera = gloco->orthographic_render_view.camera;
			loco_t::viewport_t viewport = gloco->orthographic_render_view.viewport;

			uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
			uint32_t vertex_count = 6;
		};

		shape_t push_back(const properties_t& properties) {
			vi_t vi;
			vi.position = properties.position;
			vi.size = properties.size;
			vi.tc_position = properties.tc_position;
			vi.tc_size = properties.tc_size;
			ri_t ri;
			// + 1
			std::copy(&properties.images[1], &properties.images[0] + properties.images.size(), ri.images_rest.data());
			shape_t shape = shape_add(shape_type, vi, ri,
				Key_e::depth, (uint16_t)properties.position.z,
				Key_e::blending, (uint8_t)properties.blending,
				Key_e::image, properties.images[0],
				Key_e::viewport, properties.viewport,
				Key_e::camera, properties.camera,
				Key_e::ShapeType, shape_type,
				Key_e::draw_mode, properties.draw_mode,
				Key_e::vertex_count, properties.vertex_count
			);
			((ri_t*)shape.GetData(gloco->shaper))->format = shape.get_image_data().image_settings.format;

			return shape;
		}

	}universal_image_renderer;

	struct gradient_t {

		static constexpr shaper_t::KeyTypeIndex_t shape_type = shape_type_t::gradient;
		static constexpr int kpi = kp::common;

#pragma pack(push, 1)

		struct vi_t {
			fan::vec3 position;
			fan::vec2 size;
			fan::vec2 rotation_point;
			// top left, top right
			// bottom left, bottom right
			std::array<fan::color, 4> color;
			fan::vec3 angle;
		};
		struct ri_t {

		};

#pragma pack(pop)

		inline static std::vector<shape_gl_init_t> locations = {
			shape_gl_init_t{{0, "in_position"}, 3, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position)},
			shape_gl_init_t{{1, "in_size"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, size))},
			shape_gl_init_t{{2, "in_rotation_point"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, rotation_point))},
			shape_gl_init_t{{3, "in_color"}, 4, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color) + sizeof(fan::color) * 0)},
			shape_gl_init_t{{4, "in_color"}, 4, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color) + sizeof(fan::color) * 1)},
			shape_gl_init_t{{5, "in_color"}, 4, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color) + sizeof(fan::color) * 2)},
			shape_gl_init_t{{6, "in_color"}, 4, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color) + sizeof(fan::color) * 3)},
			shape_gl_init_t{{7, "in_angle"}, 3, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, angle))}
		};

		struct properties_t {
			using type_t = gradient_t;

			fan::vec3 position = 0;
			fan::vec2 size = 0;
			std::array<fan::color, 4> color = {
				fan::random::color(),
				fan::random::color(),
				fan::random::color(),
				fan::random::color()
			};
			bool blending = false;
			fan::vec3 angle = 0;
			fan::vec2 rotation_point = 0;

			loco_t::camera_t camera = gloco->orthographic_render_view.camera;
			loco_t::viewport_t viewport = gloco->orthographic_render_view.viewport;

			uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
			uint32_t vertex_count = 6;
		};

		shape_t push_back(const properties_t& properties) {
			kps_t::common_t KeyPack;
			KeyPack.ShapeType = shape_type;
			KeyPack.depth = properties.position.z;
			KeyPack.blending = properties.blending;
			KeyPack.camera = properties.camera;
			KeyPack.viewport = properties.viewport;
			vi_t vi;
			vi.position = properties.position;
			vi.size = properties.size;
			vi.color = properties.color;
			vi.angle = properties.angle;
			vi.rotation_point = properties.rotation_point;
			ri_t ri;

			return shape_add(shape_type, vi, ri,
				Key_e::depth, (uint16_t)properties.position.z,
				Key_e::blending, (uint8_t)properties.blending,
				Key_e::viewport, properties.viewport,
				Key_e::camera, properties.camera,
				Key_e::ShapeType, shape_type,
				Key_e::draw_mode, properties.draw_mode,
				Key_e::vertex_count, properties.vertex_count
			);
		}


	}gradient;

	struct shadow_t {

		static inline shaper_t::KeyTypeIndex_t shape_type = shape_type_t::shadow;
		static constexpr int kpi = kp::light;

#pragma pack(push, 1)

		enum shape_e{
			rectangle,
			circle
		};

		struct vi_t {
			fan::vec3 position;
			int shape;
			fan::vec2 size;
			fan::vec2 rotation_point;
			fan::color color;
			uint32_t flags = 0;
			fan::vec3 angle;
			fan::vec2 light_position;
			f32_t light_radius;
			f32_t pad;
		};
		struct ri_t {

		};

#pragma pack(pop)

		inline static std::vector<shape_gl_init_t> locations = {
			shape_gl_init_t{{0, "in_position"}, 3, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position)},
			shape_gl_init_t{{1, "in_shape"}, 1, GL_INT, sizeof(vi_t), (void*)(offsetof(vi_t, shape))},
			shape_gl_init_t{{2, "in_size"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, size))},
			shape_gl_init_t{{3, "in_rotation_point"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, rotation_point))},
			shape_gl_init_t{{4, "in_color"}, 4, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color))},
			shape_gl_init_t{{5, "in_flags"}, 1, GL_UNSIGNED_INT , sizeof(vi_t), (void*)(offsetof(vi_t, flags))},
			shape_gl_init_t{{6, "in_angle"}, 3, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, angle))},
			shape_gl_init_t{{7, "in_light_position"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, light_position))},
			shape_gl_init_t{{8, "in_light_radius"}, 1, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, light_radius))},
			shape_gl_init_t{{9, "in_pad"}, 1, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, pad))},
		};

		struct properties_t {
			using type_t = shadow_t;


			fan::vec3 position = 0;
			int shape = shadow_t::rectangle;
			fan::vec2 size = 0;
			fan::vec2 rotation_point = 0;
			fan::color color = fan::colors::white;
			uint32_t flags = 0;
			fan::vec3 angle = 0;
			fan::vec2 light_position = 0;
			f32_t light_radius = 100.f;

			loco_t::camera_t camera = gloco->orthographic_render_view.camera;
			loco_t::viewport_t viewport = gloco->orthographic_render_view.viewport;

			uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
			uint32_t vertex_count = 6;
		};

		shape_t push_back(const properties_t& properties) {
			vi_t vi;
			vi.position = properties.position;
			vi.shape = properties.shape;
			vi.size = properties.size;
			vi.rotation_point = properties.rotation_point;
			vi.color = properties.color;
			vi.flags = properties.flags;
			vi.angle = properties.angle;
			vi.light_position = properties.light_position;
			vi.light_radius = properties.light_radius;
			ri_t ri;

			return shape_add(shape_type, vi, ri,
				Key_e::shadow, (uint8_t)0,
				Key_e::viewport, properties.viewport,
				Key_e::camera, properties.camera,
				Key_e::ShapeType, shape_type,
				Key_e::draw_mode, properties.draw_mode,
				Key_e::vertex_count, properties.vertex_count
			);
		}
	}shadow;

	struct shader_shape_t {

		static constexpr shaper_t::KeyTypeIndex_t shape_type = shape_type_t::shader_shape;
		static constexpr int kpi = kp::texture;

#pragma pack(push, 1)

		struct vi_t {
			fan::vec3 position;
			f32_t parallax_factor;
			fan::vec2 size;
			fan::vec2 rotation_point;
			fan::color color;
			fan::vec3 angle;
			uint32_t flags;
			fan::vec2 tc_position;
			fan::vec2 tc_size;
			f32_t seed;
		};
		struct ri_t {
			// main image + light buffer + 30
			std::array<loco_t::image_t, 30> images;
		};

#pragma pack(pop)

		inline static std::vector<shape_gl_init_t> locations = {
			shape_gl_init_t{{0, "in_position"}, 3, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position)},
			shape_gl_init_t{{1, "in_parallax_factor"}, 1, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, parallax_factor))},
			shape_gl_init_t{{2, "in_size"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, size))},
			shape_gl_init_t{{3, "in_rotation_point"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, rotation_point))},
			shape_gl_init_t{{4, "in_color"}, 4, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color))},
			shape_gl_init_t{{5, "in_angle"}, 3, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, angle))},
			shape_gl_init_t{{6, "in_flags"}, 1, GL_UNSIGNED_INT , sizeof(vi_t), (void*)(offsetof(vi_t, flags))},
			shape_gl_init_t{{7, "in_tc_position"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, tc_position))},
			shape_gl_init_t{{8, "in_tc_size"}, 2, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, tc_size))},
			shape_gl_init_t{{9, "in_seed"}, 1, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, seed)},
			};

		struct properties_t {
			using type_t = shader_shape_t;

			fan::vec3 position = 0;
			f32_t parallax_factor = 0;
			fan::vec2 size = 0;
			fan::vec2 rotation_point = 0;
			fan::color color = fan::colors::white;
			fan::vec3 angle = fan::vec3(0);
			uint32_t flags = 0;
			fan::vec2 tc_position = 0;
			fan::vec2 tc_size = 1;
			f32_t seed = 0;
			loco_t::shader_t shader;
			bool blending = true;

			loco_t::image_t image = gloco->default_texture;
			std::array<loco_t::image_t, 30> images;

			loco_t::camera_t camera = gloco->orthographic_render_view.camera;
			loco_t::viewport_t viewport = gloco->orthographic_render_view.viewport;

			uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
			uint32_t vertex_count = 6;
		};

		shape_t push_back(const properties_t& properties) {
			//KeyPack.ShapeType = shape_type;
			vi_t vi;
			vi.position = properties.position;
			vi.size = properties.size;
			vi.rotation_point = properties.rotation_point;
			vi.color = properties.color;
			vi.angle = properties.angle;
			vi.flags = properties.flags;
			vi.tc_position = properties.tc_position;
			vi.tc_size = properties.tc_size;
			vi.parallax_factor = properties.parallax_factor;
			vi.seed = properties.seed;
			ri_t ri;
			ri.images = properties.images;
			loco_t::shape_t ret = shape_add(shape_type, vi, ri,
				Key_e::depth, (uint16_t)properties.position.z,
				Key_e::blending, (uint8_t)properties.blending,
				Key_e::image, properties.image,
				Key_e::viewport, properties.viewport,
				Key_e::camera, properties.camera,
				Key_e::ShapeType, shape_type,
				Key_e::draw_mode, properties.draw_mode,
				Key_e::vertex_count, properties.vertex_count
			);
			gloco->shaper.GetShader(shape_type) = properties.shader;
			return ret;
		}

	}shader_shape;

	#if defined(fan_3D)
	struct rectangle3d_t {

		static constexpr shaper_t::KeyTypeIndex_t shape_type = shape_type_t::rectangle3d;
		static constexpr int kpi = kp::common;

#pragma pack(push, 1)

		struct vi_t {
			fan::vec3 position;
			fan::vec3 size;
			fan::color color;
			fan::vec3 angle;
		};
		struct ri_t {

		};

#pragma pack(pop)

		inline static std::vector<shape_gl_init_t> locations = {
			shape_gl_init_t{{0, "in_position"}, 3, GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t,  position)},
			shape_gl_init_t{{1, "in_size"}, 3, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, size))},
			shape_gl_init_t{{2, "in_color"}, 4, GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color))}
		};

		struct properties_t {
			using type_t = rectangle_t;

			fan::vec3 position = 0;
			fan::vec3 size = 0;
			fan::color color = fan::colors::white;
			bool blending = false;
			fan::vec3 angle = 0;

			loco_t::camera_t camera = gloco->perspective_render_view.camera;
			loco_t::viewport_t viewport = gloco->perspective_render_view.viewport;

			uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
			uint32_t vertex_count = 36;
		};


		shape_t push_back(const properties_t& properties) {
			vi_t vi;
			vi.position = properties.position;
			vi.size = properties.size;
			vi.color = properties.color;
			//vi.angle = properties.angle;
			ri_t ri;

			loco_t& loco = *OFFSETLESS(this, loco_t, rectangle3d);

			if (loco.window.renderer == loco_t::renderer_t::opengl) {
				if ((loco.context.gl.opengl.major > 3) || (loco.context.gl.opengl.major == 3 && loco.context.gl.opengl.minor >= 3)) {
					// might not need depth
					return shape_add(shape_type, vi, ri,
						Key_e::depth, (uint16_t)properties.position.z,
						Key_e::blending, (uint8_t)properties.blending,
						Key_e::viewport, properties.viewport,
						Key_e::camera, properties.camera,
						Key_e::ShapeType, shape_type,
						Key_e::draw_mode, properties.draw_mode,
						Key_e::vertex_count, properties.vertex_count
					);
				}
				else {
					vi_t vertices[36];
					for (int i = 0; i < 36; i++) {
						vertices[i] = vi;
					}

					return shape_add(shape_type, vertices[0], ri,
						Key_e::depth, (uint16_t)properties.position.z,
						Key_e::blending, (uint8_t)properties.blending,
						Key_e::viewport, properties.viewport,
						Key_e::camera, properties.camera,
						Key_e::ShapeType, shape_type,
						Key_e::draw_mode, properties.draw_mode,
						Key_e::vertex_count, properties.vertex_count
					);
				}
			}
			else if (loco.window.renderer == loco_t::renderer_t::vulkan) {

			}
			fan::throw_error();
			return{};
		}

	}rectangle3d;

	struct line3d_t {

		static constexpr shaper_t::KeyTypeIndex_t shape_type = shape_type_t::line3d;
		static constexpr int kpi = kp::common;

#pragma pack(push, 1)

		struct vi_t {
			fan::color color;
			fan::vec3 src;
			fan::vec3 dst;
		};
		struct ri_t {

		};

#pragma pack(pop)

		inline static std::vector<shape_gl_init_t> locations = {
			shape_gl_init_t{{0, "in_color"}, 4, GL_FLOAT, sizeof(line_t::vi_t), (void*)offsetof(line_t::vi_t, color)},
			shape_gl_init_t{{1, "in_src"}, 3, GL_FLOAT, sizeof(line_t::vi_t), (void*)offsetof(line_t::vi_t, src)},
			shape_gl_init_t{{2, "in_dst"}, 3, GL_FLOAT, sizeof(line_t::vi_t), (void*)offsetof(line_t::vi_t, dst)}
		};

		struct properties_t {
			using type_t = line_t;

			fan::color color = fan::colors::white;
			fan::vec3 src;
			fan::vec3 dst;

			bool blending = false;

			loco_t::camera_t camera = gloco->perspective_render_view.camera;
			loco_t::viewport_t viewport = gloco->perspective_render_view.viewport;

			uint8_t draw_mode = fan::graphics::primitive_topology_t::lines;
			uint32_t vertex_count = 2;
		};

		shape_t push_back(const properties_t& properties) {
			vi_t vi;
			vi.src = properties.src;
			vi.dst = properties.dst;
			vi.color = properties.color;
			ri_t ri;

			return shape_add(shape_type, vi, ri,
				Key_e::depth, (uint16_t)properties.src.z,
				Key_e::blending, (uint8_t)properties.blending,
				Key_e::viewport, properties.viewport,
				Key_e::camera, properties.camera,
				Key_e::ShapeType, shape_type,
				Key_e::draw_mode, properties.draw_mode,
				Key_e::vertex_count, properties.vertex_count
			);
		}

	}line3d;

	#endif

	//-------------------------------------shapes-------------------------------------
	
	// pointer
	using shape_shader_locations_t = decltype(loco_t::shaper_t::BlockProperties_t::gl_t::locations);

	inline void shape_open(
		uint16_t shape_type,
		std::size_t sizeof_vi,
		std::size_t sizeof_ri,
		shape_shader_locations_t shape_shader_locations,
		const std::string& vertex,
		const std::string& fragment,
		loco_t::shaper_t::ShapeRenderDataSize_t instance_count = 1,
		bool instanced = true
	) {
		loco_t::shader_t shader = shader_create();

		shader_set_vertex(shader,
			read_shader(vertex)
		);

		shader_set_fragment(shader,
			read_shader(fragment)
		);

		shader_compile(shader);

		shaper_t::BlockProperties_t bp;
		bp.MaxElementPerBlock = (loco_t::shaper_t::MaxElementPerBlock_t)MaxElementPerBlock;
		bp.RenderDataSize = (decltype(loco_t::shaper_t::BlockProperties_t::RenderDataSize))(sizeof_vi * instance_count);
		bp.DataSize = sizeof_ri;

		if (window.renderer == renderer_t::opengl) {
			std::construct_at(&bp.renderer.gl);
			loco_t::shaper_t::BlockProperties_t::gl_t d;
			d.locations = shape_shader_locations;
			d.shader = shader;
			d.instanced = instanced;
			bp.renderer.gl = d;
		}
#if defined(fan_vulkan)
		else if (window.renderer == renderer_t::vulkan) {
			std::construct_at(&bp.renderer.vk);
			loco_t::shaper_t::BlockProperties_t::vk_t vk;

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

		gloco->shaper.SetShapeType(shape_type, bp);
	}


#if defined(loco_sprite)
	loco_t::shader_t get_sprite_vertex_shader(const std::string& fragment) {
		if (get_renderer() == renderer_t::opengl) {
			loco_t::shader_t shader = shader_create();
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


#if defined(loco_vfi)
#include <fan/graphics/gui/vfi.h>
	vfi_t vfi;
#endif

	//#if defined(loco_texture_pack)
	//#endif

	fan::color clear_color = {
		/*0.10f, 0.10f, 0.131f, 1.f */
		0.f, 0.f, 0.f, 1.f
	};

	struct lighting_t {
		static constexpr const char* ambient_name = "lighting_ambient";
		fan::vec3 ambient = fan::vec3(1, 1, 1);

		fan::vec3 start = ambient;
		fan::vec3 target = fan::vec3(1, 1, 1);
		f32_t duration = 0.5f; // seconds to reach target
		f32_t elapsed = 0.f;

		void set_target(const fan::vec3& t, f32_t d = 0.5f) {
			start = ambient;
			target = t;
			duration = d;
			elapsed = 0.0f;
		}

		void update(f32_t delta_time) {
			if (elapsed < duration) {
				elapsed += delta_time;
				f32_t t = std::min(elapsed / duration, 1.0f);
				ambient = fan::math::lerp(start, target, t);
			}
		}

		bool is_near(const fan::vec3& t, f32_t eps = 0.01f) const {
			return ambient.distance(t) < eps;
		}
		bool is_near_target(f32_t eps = 0.01f) const {
			return is_near(target, eps);
		}

	}lighting;

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

	ImFont* fonts[std::size(font_sizes)]{};
	ImFont* fonts_bold[std::size(font_sizes)]{};

	bool imgui_initialized = false;
	static inline bool global_imgui_initialized = false;

	fan::graphics::gui::text_logger_t text_logger;

#include <fan/graphics/gui/settings_menu.h>
	settings_menu_t settings_menu;
#endif

	texturepack_t texture_pack;

	bool render_shapes_top = false;
	//gui

	std::vector<uint8_t> create_noise_image_data(const fan::vec2& image_size, int seed = fan::random::value_i64(0, ((uint32_t)-1) / 2)) {
		fan::print("TODOO");
		//FastNoiseLite noise;
		//noise.SetFractalType(FastNoiseLite::FractalType_FBm);
		//noise.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
		//noise.SetFrequency(0.010);
		//noise.SetFractalGain(0.5);
		//noise.SetFractalLacunarity(2.0);
		//noise.SetFractalOctaves(5);
		//noise.SetSeed(seed);
		//noise.SetFractalPingPongStrength(2.0);
		f32_t noise_tex_min = -1;
		f32_t noise_tex_max = 0.1;

		std::vector<uint8_t> noise_data_rgb(image_size.multiply() * 3);

		//int index = 0;

		//f32_t scale = 255.f / (noise_tex_max - noise_tex_min);

		//for (int y = 0; y < image_size.y; y++)
		//{
		//  for (int x = 0; x < image_size.x; x++)
		//  {
		//    f32_t noiseValue = noise.GetNoise((f32_t)x, (f32_t)y);
		//    unsigned char cNoise = (unsigned char)std::max(0.0f, std::min(255.0f, (noiseValue - noise_tex_min) * scale));
		//    noise_data_rgb[index * 3 + 0] = cNoise;
		//    noise_data_rgb[index * 3 + 1] = cNoise;
		//    noise_data_rgb[index * 3 + 2] = cNoise;
		//    index++;
		//  }
		//}

		return noise_data_rgb;
	}

	loco_t::image_t create_noise_image(const fan::vec2& image_size) {

		loco_t::image_load_properties_t lp;
		lp.format = fan::graphics::image_format::rgb_unorm;
		lp.internal_format = fan::graphics::image_format::rgb_unorm;
		lp.min_filter = fan::graphics::image_filter::linear;
		lp.mag_filter = fan::graphics::image_filter::linear;
		lp.visual_output = fan::graphics::image_sampler_address_mode::mirrored_repeat;

		loco_t::image_t image;

		auto noise_data = create_noise_image_data(image_size);

		fan::image::info_t ii;
		ii.data = noise_data.data();
		ii.size = image_size;
		ii.channels = 3;

		image = image_load(ii, lp);
		return image;
	}
	loco_t::image_t create_noise_image(const fan::vec2& image_size, const std::vector<uint8_t>& noise_data) {

		loco_t::image_load_properties_t lp;
		lp.format = fan::graphics::image_format::rgb_unorm;
		lp.internal_format = fan::graphics::image_format::rgb_unorm;
		lp.min_filter = fan::graphics::image_filter::linear;
		lp.mag_filter = fan::graphics::image_filter::linear;
		lp.visual_output = fan::graphics::image_sampler_address_mode::mirrored_repeat;

		loco_t::image_t image;

		fan::image::info_t ii;
		ii.data = (void*)noise_data.data();
		ii.size = image_size;
		ii.channels = 3;

		image = image_load(ii, lp);
		return image;
	}
	static fan::vec2 convert_mouse_to_ndc(const fan::vec2& mouse_position, const fan::vec2i& window_size) {
		return fan::vec2((2.0f * mouse_position.x) / window_size[0] - 1.0f, 1.0f - (2.0f * mouse_position.y) / window_size[1]);
	}
	fan::vec2 convert_mouse_to_ndc(const fan::vec2& mouse_position) const {
		return convert_mouse_to_ndc(mouse_position, gloco->window.get_size());
	}
	fan::vec2 convert_mouse_to_ndc() const {
		return convert_mouse_to_ndc(gloco->get_mouse_position(), gloco->window.get_size());
	}
	static fan::ray3_t convert_mouse_to_ray(const fan::vec2i& mouse_position, const fan::vec2& screen_size, const fan::vec3& camera_position, const fan::mat4& projection, const fan::mat4& view) {

		fan::vec4 ray_ndc((2.0f * mouse_position[0]) / screen_size.x - 1.0f, 1.0f - (2.0f * mouse_position[1]) / screen_size.y, 1.0f, 1.0f);

		fan::mat4 inverted_projection = projection.inverse();

		fan::vec4 ray_clip = inverted_projection * ray_ndc;

		ray_clip.z = -1.0f;
		ray_clip.w = 0.0f;

		fan::mat4 inverted_view = view.inverse();

		fan::vec4 ray_world = inverted_view * ray_clip;

		fan::vec3 ray_dir = fan::vec3(ray_world.x, ray_world.y, ray_world.z).normalized();

		fan::vec3 ray_origin = camera_position;
		return fan::ray3_t(ray_origin, ray_dir);
	}
	fan::ray3_t convert_mouse_to_ray(const fan::vec3& camera_position, const fan::mat4& projection, const fan::mat4& view) {
		return convert_mouse_to_ray(get_mouse_position(), window.get_size(), camera_position, projection, view);
	}
	fan::ray3_t convert_mouse_to_ray(const fan::mat4& projection, const fan::mat4& view) {
		return convert_mouse_to_ray(get_mouse_position(), window.get_size(), camera_get_position(perspective_render_view.camera), projection, view);
	}
	static bool is_ray_intersecting_cube(const fan::ray3_t& ray, const fan::vec3& position, const fan::vec3& size) {
		fan::vec3 min_bounds = position - size;
		fan::vec3 max_bounds = position + size;

		fan::vec3 t_min = (min_bounds - ray.origin) / (ray.direction + fan::vec3(1e-6f));
		fan::vec3 t_max = (max_bounds - ray.origin) / (ray.direction + fan::vec3(1e-6f));

		fan::vec3 t1 = t_min.min(t_max);
		fan::vec3 t2 = t_min.max(t_max);

		f32_t t_near = std::max(t1.x, std::max(t1.y, t1.z));
		f32_t t_far = std::min(t2.x, std::min(t2.y, t2.z));

		return t_near <= t_far && t_far >= 0.0f;
	}


	void printclnn(auto&&... values) {
#if defined (fan_gui)
		([&](const auto& value) {
			std::ostringstream oss;
			oss << value;
			console.print(oss.str() + " ", 0);
			}(values), ...);
#endif
	}
	void printcl(auto&&... values) {
#if defined(fan_gui)
		printclnn(values...);
		console.print("\n", 0);
#endif
	}

	void printclnnh(int highlight, auto&&... values) {
#if defined(fan_gui)
		([&](const auto& value) {
			std::ostringstream oss;
			oss << value;
			console.print(oss.str() + " ", highlight);
			}(values), ...);
#endif
	}

	void printclh(int highlight, auto&&... values) {
#if defined(fan_gui)
		printclnnh(highlight, values...);
		console.print("\n", highlight);
#endif
	}


#if defined(loco_cuda)

	struct cuda_textures_t {

		cuda_textures_t() {
			inited = false;
		}
		~cuda_textures_t() {
		}
		void close(loco_t* loco, loco_t::shape_t& cid) {
			loco_t::universal_image_renderer_t::ri_t& ri = *(loco_t::universal_image_renderer_t::ri_t*)cid.GetData(gloco->shaper);
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

		void resize(loco_t* loco, loco_t::shape_t& id, uint8_t format, fan::vec2ui size) {
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

	fan::audio::piece_t piece_hover, piece_click;
#endif
	void camera_move_to(const loco_t::shape_t& shape, const loco_t::render_view_t& render_view) {
		camera_set_position(
			orthographic_render_view.camera,
			shape.get_position()
		);
	}
	void camera_move_to(const loco_t::shape_t& shape) {
		camera_move_to(shape, orthographic_render_view);
	}

	void camera_move_to_smooth(const loco_t::shape_t& shape, const loco_t::render_view_t& render_view) {
		fan::vec2 current = camera_get_position(render_view.camera);
		fan::vec2 target = shape.get_position();
		f32_t t = 0.1f;
		camera_set_position(
			orthographic_render_view.camera,
			current.lerp(target, t)
		);
	}

	void camera_move_to_smooth(const loco_t::shape_t& shape) {
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

#if defined(fan_json)
export namespace fan {
	namespace graphics {

		bool shape_to_json(loco_t::shape_t& shape, fan::json* json) {
			fan::json& out = *json;
			switch (shape.get_shape_type()) {
			case loco_t::shape_type_t::light: {
				loco_t::light_t::properties_t defaults;
				out["shape"] = "light";
				if (shape.get_position() != defaults.position) {
					out["position"] = shape.get_position();
				}
				if (shape.get_parallax_factor() != defaults.parallax_factor) {
					out["parallax_factor"] = shape.get_parallax_factor();
				}
				if (shape.get_size() != defaults.size) {
					out["size"] = shape.get_size();
				}
				if (shape.get_rotation_point() != defaults.rotation_point) {
					out["rotation_point"] = shape.get_rotation_point();
				}
				if (shape.get_color() != defaults.color) {
					out["color"] = shape.get_color();
				}
				if (shape.get_flags() != defaults.flags) {
					out["flags"] = shape.get_flags();
				}
				if (shape.get_angle() != defaults.angle) {
					out["angle"] = shape.get_angle();
				}
				break;
			}
			case loco_t::shape_type_t::line: {
				loco_t::line_t::properties_t defaults;
				out["shape"] = "line";
				if (shape.get_color() != defaults.color) {
					out["color"] = shape.get_color();
				}
				if (shape.get_src() != defaults.src) {
					out["src"] = shape.get_src();
				}
				if (shape.get_dst() != defaults.dst) {
					out["dst"] = shape.get_dst();
				}
				break;
			}
			case loco_t::shape_type_t::rectangle: {
				loco_t::rectangle_t::properties_t defaults;
				out["shape"] = "rectangle";
				if (shape.get_position() != defaults.position) {
					out["position"] = shape.get_position();
				}
				if (shape.get_size() != defaults.size) {
					out["size"] = shape.get_size();
				}
				if (shape.get_rotation_point() != defaults.rotation_point) {
					out["rotation_point"] = shape.get_rotation_point();
				}
				if (shape.get_color() != defaults.color) {
					out["color"] = shape.get_color();
				}
				if (shape.get_outline_color() != defaults.outline_color) {
					out["outline_color"] = shape.get_outline_color();
				}
				if (shape.get_angle() != defaults.angle) {
					out["angle"] = shape.get_angle();
				}
				break;
			}
			case loco_t::shape_type_t::sprite: {
				loco_t::sprite_t::properties_t defaults;
				out["shape"] = "sprite";
				if (shape.get_position() != defaults.position) {
					out["position"] = shape.get_position();
				}
				if (shape.get_parallax_factor() != defaults.parallax_factor) {
					out["parallax_factor"] = shape.get_parallax_factor();
				}
				if (shape.get_size() != defaults.size) {
					out["size"] = shape.get_size();
				}
				if (shape.get_rotation_point() != defaults.rotation_point) {
					out["rotation_point"] = shape.get_rotation_point();
				}
				if (shape.get_color() != defaults.color) {
					out["color"] = shape.get_color();
				}
				if (shape.get_angle() != defaults.angle) {
					out["angle"] = shape.get_angle();
				}
				if (shape.get_flags() != defaults.flags) {
					out["flags"] = shape.get_flags();
				}
				if (shape.get_tc_position() != defaults.tc_position) {
					out["tc_position"] = shape.get_tc_position();
				}
				if (shape.get_tc_size() != defaults.tc_size) {
					out["tc_size"] = shape.get_tc_size();
				}
				auto* ri = ((loco_t::sprite_t::ri_t*)shape.GetData(gloco->shaper));
				if (gloco->texture_pack) {
					out["texture_pack_name"] = gloco->texture_pack[ri->texture_pack_unique_id].name;
				}
				if (ri->shape_animations) {
					fan::json animation_array = fan::json::array();
					for (auto& animation_nrs : gloco->shape_animations[ri->shape_animations]) {
						animation_array.push_back(animation_nrs.id);
					}
					if (animation_array.empty() == false) {
						out["animations"] = animation_array;
					}
				}
				fan::json images_array = fan::json::array();

				auto main_image = shape.get_image();
				auto img_json = gloco->image_to_json(main_image);
				if (!img_json.empty()) {
					images_array.push_back(img_json);
				}

				auto images = shape.get_images();
				for (auto& image : images) {
					img_json = gloco->image_to_json(image);
					if (!img_json.empty()) {
						images_array.push_back(img_json);
					}
				}

				if (!images_array.empty()) {
					out["images"] = images_array;
				}
				break;
			}
			case loco_t::shape_type_t::unlit_sprite: {
				loco_t::unlit_sprite_t::properties_t defaults;
				out["shape"] = "unlit_sprite";
				if (shape.get_position() != defaults.position) {
					out["position"] = shape.get_position();
				}
				if (shape.get_parallax_factor() != defaults.parallax_factor) {
					out["parallax_factor"] = shape.get_parallax_factor();
				}
				if (shape.get_size() != defaults.size) {
					out["size"] = shape.get_size();
				}
				if (shape.get_rotation_point() != defaults.rotation_point) {
					out["rotation_point"] = shape.get_rotation_point();
				}
				if (shape.get_color() != defaults.color) {
					out["color"] = shape.get_color();
				}
				if (shape.get_angle() != defaults.angle) {
					out["angle"] = shape.get_angle();
				}
				if (shape.get_flags() != defaults.flags) {
					out["flags"] = shape.get_flags();
				}
				if (shape.get_tc_position() != defaults.tc_position) {
					out["tc_position"] = shape.get_tc_position();
				}
				if (shape.get_tc_size() != defaults.tc_size) {
					out["tc_size"] = shape.get_tc_size();
				}
				if (gloco->texture_pack) {
					out["texture_pack_name"] = gloco->texture_pack[((loco_t::unlit_sprite_t::ri_t*)shape.GetData(gloco->shaper))->texture_pack_unique_id].name;
				}

				fan::json images_array = fan::json::array();

				auto main_image = shape.get_image();
				auto img_json = gloco->image_to_json(main_image);
				if (!img_json.empty()) {
					images_array.push_back(img_json);
				}

				auto images = shape.get_images();
				for (auto& image : images) {
					img_json = gloco->image_to_json(image);
					if (!img_json.empty()) {
						images_array.push_back(img_json);
					}
				}

				if (!images_array.empty()) {
					out["images"] = images_array;
				}

				break;
			}
			case loco_t::shape_type_t::text: {
				out["shape"] = "text";
				break;
			}
			case loco_t::shape_type_t::circle: {
				loco_t::circle_t::properties_t defaults;
				out["shape"] = "circle";
				if (shape.get_position() != defaults.position) {
					out["position"] = shape.get_position();
				}
				if (shape.get_radius() != defaults.radius) {
					out["radius"] = shape.get_radius();
				}
				if (shape.get_rotation_point() != defaults.rotation_point) {
					out["rotation_point"] = shape.get_rotation_point();
				}
				if (shape.get_color() != defaults.color) {
					out["color"] = shape.get_color();
				}
				if (shape.get_angle() != defaults.angle) {
					out["angle"] = shape.get_angle();
				}
				break;
			}
			case loco_t::shape_type_t::grid: {
				loco_t::grid_t::properties_t defaults;
				out["shape"] = "grid";
				if (shape.get_position() != defaults.position) {
					out["position"] = shape.get_position();
				}
				if (shape.get_size() != defaults.size) {
					out["size"] = shape.get_size();
				}
				if (shape.get_grid_size() != defaults.grid_size) {
					out["grid_size"] = shape.get_grid_size();
				}
				if (shape.get_rotation_point() != defaults.rotation_point) {
					out["rotation_point"] = shape.get_rotation_point();
				}
				if (shape.get_color() != defaults.color) {
					out["color"] = shape.get_color();
				}
				if (shape.get_angle() != defaults.angle) {
					out["angle"] = shape.get_angle();
				}
				break;
			}
			case loco_t::shape_type_t::particles: {
				loco_t::particles_t::properties_t defaults;
				auto& ri = *(loco_t::particles_t::ri_t*)shape.GetData(gloco->shaper);
				out["shape"] = "particles";
				if (ri.position != defaults.position) {
					out["position"] = ri.position;
				}
				if (ri.size != defaults.size) {
					out["size"] = ri.size;
				}
				if (ri.color != defaults.color) {
					out["color"] = ri.color;
				}
				if (ri.begin_time != defaults.begin_time) {
					out["begin_time"] = ri.begin_time;
				}
				if (ri.alive_time != defaults.alive_time) {
					out["alive_time"] = ri.alive_time;
				}
				if (ri.respawn_time != defaults.respawn_time) {
					out["respawn_time"] = ri.respawn_time;
				}
				if (ri.count != defaults.count) {
					out["count"] = ri.count;
				}
				if (ri.position_velocity != defaults.position_velocity) {
					out["position_velocity"] = ri.position_velocity;
				}
				if (ri.angle_velocity != defaults.angle_velocity) {
					out["angle_velocity"] = ri.angle_velocity;
				}
				if (ri.begin_angle != defaults.begin_angle) {
					out["begin_angle"] = ri.begin_angle;
				}
				if (ri.end_angle != defaults.end_angle) {
					out["end_angle"] = ri.end_angle;
				}
				if (ri.angle != defaults.angle) {
					out["angle"] = ri.angle;
				}
				if (ri.gap_size != defaults.gap_size) {
					out["gap_size"] = ri.gap_size;
				}
				if (ri.max_spread_size != defaults.max_spread_size) {
					out["max_spread_size"] = ri.max_spread_size;
				}
				if (ri.size_velocity != defaults.size_velocity) {
					out["size_velocity"] = ri.size_velocity;
				}
				if (ri.shape != defaults.shape) {
					out["particle_shape"] = ri.shape;
				}
				if (ri.blending != defaults.blending) {
					out["blending"] = ri.blending;
				}
				loco_t::image_t image = shape.get_image();
				if (image) {
					out.update(gloco->image_to_json(image), true);
				}
				break;
			}
			default: {
				fan::throw_error("unimplemented shape");
			}
			}
			return false;
		}

		bool json_to_shape(const fan::json& in, loco_t::shape_t* shape) {
			std::string shape_type = in["shape"];
			switch (fan::get_hash(shape_type.c_str())) {
			case fan::get_hash("rectangle"): {
				loco_t::rectangle_t::properties_t p;
				if (in.contains("position")) {
					p.position = in["position"];
				}
				if (in.contains("size")) {
					p.size = in["size"];
				}
				if (in.contains("rotation_point")) {
					p.rotation_point = in["rotation_point"];
				}
				if (in.contains("color")) {
					p.color = in["color"];
				}
				if (in.contains("outline_color")) {
					p.outline_color = in["outline_color"];
				}
				if (in.contains("angle")) {
					p.angle = in["angle"];
				}
				*shape = p;
				break;
			}
			case fan::get_hash("light"): {
				loco_t::light_t::properties_t p;
				if (in.contains("position")) {
					p.position = in["position"];
				}
				if (in.contains("parallax_factor")) {
					p.parallax_factor = in["parallax_factor"];
				}
				if (in.contains("size")) {
					p.size = in["size"];
				}
				if (in.contains("rotation_point")) {
					p.rotation_point = in["rotation_point"];
				}
				if (in.contains("color")) {
					p.color = in["color"];
				}
				if (in.contains("flags")) {
					p.flags = in["flags"];
				}
				if (in.contains("angle")) {
					p.angle = in["angle"];
				}
				*shape = p;
				break;
			}
			case fan::get_hash("line"): {
				loco_t::line_t::properties_t p;
				if (in.contains("color")) {
					p.color = in["color"];
				}
				if (in.contains("src")) {
					p.src = in["src"];
				}
				if (in.contains("dst")) {
					p.dst = in["dst"];
				}
				*shape = p;
				break;
			}
			case fan::get_hash("sprite"): {
				loco_t::sprite_t::properties_t p;
				p.blending = true;
				if (in.contains("position")) {
					p.position = in["position"];
				}
				if (in.contains("parallax_factor")) {
					p.parallax_factor = in["parallax_factor"];
				}
				if (in.contains("size")) {
					p.size = in["size"];
				}
				if (in.contains("rotation_point")) {
					p.rotation_point = in["rotation_point"];
				}
				if (in.contains("color")) {
					p.color = in["color"];
				}
				if (in.contains("angle")) {
					p.angle = in["angle"];
				}
				if (in.contains("flags")) {
					p.flags = in["flags"];
				}
				if (in.contains("tc_position")) {
					p.tc_position = in["tc_position"];
				}
				if (in.contains("tc_size")) {
					p.tc_size = in["tc_size"];
				}
				bool contains_animations = in.contains("animations");
				// load texture pack only if no sprite sheet animations
				// because sprite sheet animations use image for it
				if (contains_animations == false && in.contains("texture_pack_name") && gloco->texture_pack) {
					p.texture_pack_unique_id = gloco->texture_pack[in["texture_pack_name"]];
				}
				*shape = p;

				fan::graphics::image_load_properties_t lp;
				if (in.contains("image_visual_output")) {
					lp.visual_output = in["image_visual_output"];
				}
				if (in.contains("image_format")) {
					lp.format = in["image_format"];
				}
				if (in.contains("image_type")) {
					lp.type = in["image_type"];
				}
				if (in.contains("image_min_filter")) {
					lp.min_filter = in["image_min_filter"];
				}
				if (in.contains("image_mag_filter")) {
					lp.mag_filter = in["image_mag_filter"];
				}
				if (in.contains("images") && in["images"].is_array()) {
					for (const auto [i, image_json] : fan::enumerate(in["images"])) {
						loco_t::image_t image = gloco->json_to_image(image_json);
						if (i == 0) {
							shape->set_image(image);
						}
						else {
							auto images = shape->get_images();
							images[i - 1] = image;
							shape->set_images(images);
						}
					}
				}

				if (contains_animations) {
					for (auto& item : in["animations"]) {
						uint32_t anim_id = item.get<uint32_t>();
						auto existing_animation = gloco->get_sprite_sheet_animation(anim_id);
						shape->add_existing_animation(anim_id);
					}
				}

				break;
			}
			case fan::get_hash("unlit_sprite"): {
				loco_t::unlit_sprite_t::properties_t p;
				p.blending = true;
				if (in.contains("position")) {
					p.position = in["position"];
				}
				if (in.contains("parallax_factor")) {
					p.parallax_factor = in["parallax_factor"];
				}
				if (in.contains("size")) {
					p.size = in["size"];
				}
				if (in.contains("rotation_point")) {
					p.rotation_point = in["rotation_point"];
				}
				if (in.contains("color")) {
					p.color = in["color"];
				}
				if (in.contains("angle")) {
					p.angle = in["angle"];
				}
				if (in.contains("flags")) {
					p.flags = in["flags"];
				}
				if (in.contains("tc_position")) {
					p.tc_position = in["tc_position"];
				}
				if (in.contains("tc_size")) {
					p.tc_size = in["tc_size"];
				}
				if (in.contains("texture_pack_name") && gloco->texture_pack) {
					p.texture_pack_unique_id = gloco->texture_pack[in["texture_pack_name"]];
				}
				*shape = p;
				fan::graphics::image_load_properties_t lp;
				if (in.contains("image_visual_output")) {
					lp.visual_output = in["image_visual_output"];
				}
				if (in.contains("image_format")) {
					lp.format = in["image_format"];
				}
				if (in.contains("image_type")) {
					lp.type = in["image_type"];
				}
				if (in.contains("image_min_filter")) {
					lp.min_filter = in["image_min_filter"];
				}
				if (in.contains("image_mag_filter")) {
					lp.mag_filter = in["image_mag_filter"];
				}

				if (in.contains("images") && in["images"].is_array()) {
					for (const auto [i, image_json] : fan::enumerate(in["images"])) {
						if (!image_json.contains("image_path")) continue;

						auto path = image_json["image_path"];
						if (fan::io::file::exists(path)) {
							fan::graphics::image_load_properties_t lp;

							if (image_json.contains("image_visual_output")) lp.visual_output = image_json["image_visual_output"];
							if (image_json.contains("image_format")) lp.format = image_json["image_format"];
							if (image_json.contains("image_type")) lp.type = image_json["image_type"];
							if (image_json.contains("image_min_filter")) lp.min_filter = image_json["image_min_filter"];
							if (image_json.contains("image_mag_filter")) lp.mag_filter = image_json["image_mag_filter"];

							auto image = gloco->image_load(path, lp);

							if (i == 0) {
								shape->set_image(image);
							}
							else {
								auto images = shape->get_images();
								images[i - 1] = image;
								shape->set_images(images);
							}
							gloco->image_list[image].image_path = path;
						}
					}
				}
				break;
			}
			case fan::get_hash("circle"): {
				loco_t::circle_t::properties_t p;
				if (in.contains("position")) {
					p.position = in["position"];
				}
				if (in.contains("radius")) {
					p.radius = in["radius"];
				}
				if (in.contains("rotation_point")) {
					p.rotation_point = in["rotation_point"];
				}
				if (in.contains("color")) {
					p.color = in["color"];
				}
				if (in.contains("angle")) {
					p.angle = in["angle"];
				}
				*shape = p;
				break;
			}
			case fan::get_hash("grid"): {
				loco_t::grid_t::properties_t p;
				if (in.contains("position")) {
					p.position = in["position"];
				}
				if (in.contains("size")) {
					p.size = in["size"];
				}
				if (in.contains("grid_size")) {
					p.grid_size = in["grid_size"];
				}
				if (in.contains("rotation_point")) {
					p.rotation_point = in["rotation_point"];
				}
				if (in.contains("color")) {
					p.color = in["color"];
				}
				if (in.contains("angle")) {
					p.angle = in["angle"];
				}
				*shape = p;
				break;
			}
			case fan::get_hash("particles"): {
				loco_t::particles_t::properties_t p;
				if (in.contains("position")) {
					p.position = in["position"];
				}
				if (in.contains("size")) {
					p.size = in["size"];
				}
				if (in.contains("color")) {
					p.color = in["color"];
				}
				if (in.contains("begin_time")) {
					p.begin_time = in["begin_time"];
				}
				if (in.contains("alive_time")) {
					p.alive_time = in["alive_time"];
				}
				if (in.contains("respawn_time")) {
					p.respawn_time = in["respawn_time"];
				}
				if (in.contains("count")) {
					p.count = in["count"];
				}
				if (in.contains("position_velocity")) {
					p.position_velocity = in["position_velocity"];
				}
				if (in.contains("angle_velocity")) {
					p.angle_velocity = in["angle_velocity"];
				}
				if (in.contains("begin_angle")) {
					p.begin_angle = in["begin_angle"];
				}
				if (in.contains("end_angle")) {
					p.end_angle = in["end_angle"];
				}
				if (in.contains("angle")) {
					p.angle = in["angle"];
				}
				if (in.contains("gap_size")) {
					p.gap_size = in["gap_size"];
				}
				if (in.contains("max_spread_size")) {
					p.max_spread_size = in["max_spread_size"];
				}
				if (in.contains("size_velocity")) {
					p.size_velocity = in["size_velocity"];
				}
				if (in.contains("particle_shape")) {
					p.shape = in["particle_shape"];
				}
				if (in.contains("blending")) {
					p.blending = in["blending"];
				}
				p.image = gloco->json_to_image(in);
				*shape = p;
				break;
			}
			default: {
				fan::throw_error("unimplemented shape");
			}
			}
			return false;
		}

		bool shape_serialize(loco_t::shape_t& shape, fan::json* out) {
			return shape_to_json(shape, out);
		}
	}
}

export namespace fan {

	namespace graphics {
		bool shape_to_bin(loco_t::shape_t& shape, std::vector<uint8_t>* data) {
			std::vector<uint8_t>& out = *data;
			fan::write_to_vector(out, shape.get_shape_type());
			fan::write_to_vector(out, shape.gint());
			switch (shape.get_shape_type()) {
			case loco_t::shape_type_t::light: {
				fan::write_to_vector(out, shape.get_position());
				fan::write_to_vector(out, shape.get_parallax_factor());
				fan::write_to_vector(out, shape.get_size());
				fan::write_to_vector(out, shape.get_rotation_point());
				fan::write_to_vector(out, shape.get_color());
				fan::write_to_vector(out, shape.get_flags());
				fan::write_to_vector(out, shape.get_angle());
				break;
			}
			case loco_t::shape_type_t::line: {
				fan::write_to_vector(out, shape.get_color());
				fan::write_to_vector(out, shape.get_src());
				fan::write_to_vector(out, shape.get_dst());
				break;
			case loco_t::shape_type_t::rectangle: {
				fan::write_to_vector(out, shape.get_position());
				fan::write_to_vector(out, shape.get_size());
				fan::write_to_vector(out, shape.get_rotation_point());
				fan::write_to_vector(out, shape.get_color());
				fan::write_to_vector(out, shape.get_angle());
				break;
			}
			case loco_t::shape_type_t::sprite: {
				fan::write_to_vector(out, shape.get_position());
				fan::write_to_vector(out, shape.get_parallax_factor());
				fan::write_to_vector(out, shape.get_size());
				fan::write_to_vector(out, shape.get_rotation_point());
				fan::write_to_vector(out, shape.get_color());
				fan::write_to_vector(out, shape.get_angle());
				fan::write_to_vector(out, shape.get_flags());
				fan::write_to_vector(out, shape.get_image_data().image_path);
				fan::graphics::image_load_properties_t ilp = gloco->image_get_settings(shape.get_image());
				fan::write_to_vector(out, ilp.visual_output);
				fan::write_to_vector(out, ilp.format);
				fan::write_to_vector(out, ilp.type);
				fan::write_to_vector(out, ilp.min_filter);
				fan::write_to_vector(out, ilp.mag_filter);
				fan::write_to_vector(out, shape.get_tc_position());
				fan::write_to_vector(out, shape.get_tc_size());
				break;
			}
			case loco_t::shape_type_t::unlit_sprite: {
				fan::write_to_vector(out, shape.get_position());
				fan::write_to_vector(out, shape.get_parallax_factor());
				fan::write_to_vector(out, shape.get_size());
				fan::write_to_vector(out, shape.get_rotation_point());
				fan::write_to_vector(out, shape.get_color());
				fan::write_to_vector(out, shape.get_angle());
				fan::write_to_vector(out, shape.get_flags());
				fan::write_to_vector(out, shape.get_image_data().image_path);
				fan::graphics::image_load_properties_t ilp = gloco->image_get_settings(shape.get_image());
				fan::write_to_vector(out, ilp.visual_output);
				fan::write_to_vector(out, ilp.format);
				fan::write_to_vector(out, ilp.type);
				fan::write_to_vector(out, ilp.min_filter);
				fan::write_to_vector(out, ilp.mag_filter);
				fan::write_to_vector(out, shape.get_tc_position());
				fan::write_to_vector(out, shape.get_tc_size());
				break;
			}
			case loco_t::shape_type_t::circle: {
				fan::write_to_vector(out, shape.get_position());
				fan::write_to_vector(out, shape.get_radius());
				fan::write_to_vector(out, shape.get_rotation_point());
				fan::write_to_vector(out, shape.get_color());
				fan::write_to_vector(out, shape.get_angle());
				break;
			}
			case loco_t::shape_type_t::grid: {
				fan::write_to_vector(out, shape.get_position());
				fan::write_to_vector(out, shape.get_size());
				fan::write_to_vector(out, shape.get_grid_size());
				fan::write_to_vector(out, shape.get_rotation_point());
				fan::write_to_vector(out, shape.get_color());
				fan::write_to_vector(out, shape.get_angle());
				break;
			}
			case loco_t::shape_type_t::particles: {
				auto& ri = *(loco_t::particles_t::ri_t*)shape.GetData(gloco->shaper);
				fan::write_to_vector(out, ri.position);
				fan::write_to_vector(out, ri.size);
				fan::write_to_vector(out, ri.color);
				fan::write_to_vector(out, ri.begin_time);
				fan::write_to_vector(out, ri.alive_time);
				fan::write_to_vector(out, ri.respawn_time);
				fan::write_to_vector(out, ri.count);
				fan::write_to_vector(out, ri.position_velocity);
				fan::write_to_vector(out, ri.angle_velocity);
				fan::write_to_vector(out, ri.begin_angle);
				fan::write_to_vector(out, ri.end_angle);
				fan::write_to_vector(out, ri.angle);
				fan::write_to_vector(out, ri.gap_size);
				fan::write_to_vector(out, ri.max_spread_size);
				fan::write_to_vector(out, ri.size_velocity);
				fan::write_to_vector(out, ri.shape);
				fan::write_to_vector(out, ri.blending);
				break;
			}
			}
			case loco_t::shape_type_t::light_end: {
				break;
			}
			default: {
				fan::throw_error("unimplemented shape");
			}
			}
			return false;
		}

		bool bin_to_shape(const std::vector<uint8_t>& in, loco_t::shape_t* shape, uint64_t& offset) {
			using sti_t = std::remove_reference_t<decltype(loco_t::shape_t().get_shape_type())>;
			using nr_t = std::remove_reference_t<decltype(loco_t::shape_t().gint())>;
			sti_t shape_type = fan::vector_read_data<sti_t>(in, offset);
			nr_t nri = fan::vector_read_data<nr_t>(in, offset);
			switch (shape_type) {
			case loco_t::shape_type_t::rectangle: {
				loco_t::rectangle_t::properties_t p;
				p.position = fan::vector_read_data<decltype(p.position)>(in, offset);
				p.size = fan::vector_read_data<decltype(p.size)>(in, offset);
				p.rotation_point = fan::vector_read_data<decltype(p.rotation_point)>(in, offset);
				p.color = fan::vector_read_data<decltype(p.color)>(in, offset);
				p.angle = fan::vector_read_data<decltype(p.angle)>(in, offset);
				p.outline_color = p.color;
				*shape = p;
				return false;
			}
			case loco_t::shape_type_t::light: {
				loco_t::light_t::properties_t p;
				p.position = fan::vector_read_data<decltype(p.position)>(in, offset);
				p.parallax_factor = fan::vector_read_data<decltype(p.parallax_factor)>(in, offset);
				p.size = fan::vector_read_data<decltype(p.size)>(in, offset);
				p.rotation_point = fan::vector_read_data<decltype(p.rotation_point)>(in, offset);
				p.color = fan::vector_read_data<decltype(p.color)>(in, offset);
				p.flags = fan::vector_read_data<decltype(p.flags)>(in, offset);
				p.angle = fan::vector_read_data<decltype(p.angle)>(in, offset);
				*shape = p;
				break;
			}
			case loco_t::shape_type_t::line: {
				loco_t::line_t::properties_t p;
				p.color = fan::vector_read_data<decltype(p.color)>(in, offset);
				p.src = fan::vector_read_data<decltype(p.src)>(in, offset);
				p.dst = fan::vector_read_data<decltype(p.dst)>(in, offset);
				*shape = p;
				break;
			}
			case loco_t::shape_type_t::sprite: {
				loco_t::sprite_t::properties_t p;
				p.position = fan::vector_read_data<decltype(p.position)>(in, offset);
				p.parallax_factor = fan::vector_read_data<decltype(p.parallax_factor)>(in, offset);
				p.size = fan::vector_read_data<decltype(p.size)>(in, offset);
				p.rotation_point = fan::vector_read_data<decltype(p.rotation_point)>(in, offset);
				p.color = fan::vector_read_data<decltype(p.color)>(in, offset);
				p.angle = fan::vector_read_data<decltype(p.angle)>(in, offset);
				p.flags = fan::vector_read_data<decltype(p.flags)>(in, offset);

				std::string image_path = fan::vector_read_data<std::string>(in, offset);
				fan::graphics::image_load_properties_t ilp;
				ilp.visual_output = fan::vector_read_data<decltype(ilp.visual_output)>(in, offset);
				ilp.format = fan::vector_read_data<decltype(ilp.format)>(in, offset);
				ilp.type = fan::vector_read_data<decltype(ilp.type)>(in, offset);
				ilp.min_filter = fan::vector_read_data<decltype(ilp.min_filter)>(in, offset);
				ilp.mag_filter = fan::vector_read_data<decltype(ilp.mag_filter)>(in, offset);
				p.tc_position = fan::vector_read_data<decltype(p.tc_position)>(in, offset);
				p.tc_size = fan::vector_read_data<decltype(p.tc_size)>(in, offset);
				*shape = p;
				if (image_path.size()) {
					shape->get_image_data().image_path = image_path;
					shape->set_image(gloco->image_load(image_path, ilp));
				}
				break;
			}
			case loco_t::shape_type_t::unlit_sprite: {
				loco_t::unlit_sprite_t::properties_t p;
				p.position = fan::vector_read_data<decltype(p.position)>(in, offset);
				p.parallax_factor = fan::vector_read_data<decltype(p.parallax_factor)>(in, offset);
				p.size = fan::vector_read_data<decltype(p.size)>(in, offset);
				p.rotation_point = fan::vector_read_data<decltype(p.rotation_point)>(in, offset);
				p.color = fan::vector_read_data<decltype(p.color)>(in, offset);
				p.angle = fan::vector_read_data<decltype(p.angle)>(in, offset);
				p.flags = fan::vector_read_data<decltype(p.flags)>(in, offset);
				std::string image_path = fan::vector_read_data<std::string>(in, offset);
				fan::graphics::image_load_properties_t ilp;
				ilp.visual_output = fan::vector_read_data<decltype(ilp.visual_output)>(in, offset);
				ilp.format = fan::vector_read_data<decltype(ilp.format)>(in, offset);
				ilp.type = fan::vector_read_data<decltype(ilp.type)>(in, offset);
				ilp.min_filter = fan::vector_read_data<decltype(ilp.min_filter)>(in, offset);
				ilp.mag_filter = fan::vector_read_data<decltype(ilp.mag_filter)>(in, offset);
				p.tc_position = fan::vector_read_data<decltype(p.tc_position)>(in, offset);
				p.tc_size = fan::vector_read_data<decltype(p.tc_size)>(in, offset);
				*shape = p;
				if (image_path.size()) {
					shape->get_image_data().image_path = image_path;
					shape->set_image(gloco->image_load(image_path, ilp));
				}
				break;
			}
			case loco_t::shape_type_t::circle: {
				loco_t::circle_t::properties_t p;
				p.position = fan::vector_read_data<decltype(p.position)>(in, offset);
				p.radius = fan::vector_read_data<decltype(p.radius)>(in, offset);
				p.rotation_point = fan::vector_read_data<decltype(p.rotation_point)>(in, offset);
				p.color = fan::vector_read_data<decltype(p.color)>(in, offset);
				p.angle = fan::vector_read_data<decltype(p.angle)>(in, offset);
				*shape = p;
				break;
			}
			case loco_t::shape_type_t::grid: {
				loco_t::grid_t::properties_t p;
				p.position = fan::vector_read_data<decltype(p.position)>(in, offset);
				p.size = fan::vector_read_data<decltype(p.size)>(in, offset);
				p.grid_size = fan::vector_read_data<decltype(p.grid_size)>(in, offset);
				p.rotation_point = fan::vector_read_data<decltype(p.rotation_point)>(in, offset);
				p.color = fan::vector_read_data<decltype(p.color)>(in, offset);
				p.angle = fan::vector_read_data<decltype(p.angle)>(in, offset);
				*shape = p;
				break;
			}
			case loco_t::shape_type_t::particles: {
				loco_t::particles_t::properties_t p;
				p.position = fan::vector_read_data<decltype(p.position)>(in, offset);
				p.size = fan::vector_read_data<decltype(p.size)>(in, offset);
				p.color = fan::vector_read_data<decltype(p.color)>(in, offset);
				p.begin_time = fan::vector_read_data<decltype(p.begin_time)>(in, offset);
				p.alive_time = fan::vector_read_data<decltype(p.alive_time)>(in, offset);
				p.respawn_time = fan::vector_read_data<decltype(p.respawn_time)>(in, offset);
				p.count = fan::vector_read_data<decltype(p.count)>(in, offset);
				p.position_velocity = fan::vector_read_data<decltype(p.position_velocity)>(in, offset);
				p.angle_velocity = fan::vector_read_data<decltype(p.angle_velocity)>(in, offset);
				p.begin_angle = fan::vector_read_data<decltype(p.begin_angle)>(in, offset);
				p.end_angle = fan::vector_read_data<decltype(p.end_angle)>(in, offset);
				p.angle = fan::vector_read_data<decltype(p.angle)>(in, offset);
				p.gap_size = fan::vector_read_data<decltype(p.gap_size)>(in, offset);
				p.max_spread_size = fan::vector_read_data<decltype(p.max_spread_size)>(in, offset);
				p.size_velocity = fan::vector_read_data<decltype(p.size_velocity)>(in, offset);
				p.shape = fan::vector_read_data<decltype(p.shape)>(in, offset);
				p.blending = fan::vector_read_data<decltype(p.blending)>(in, offset);
				*shape = p;
				break;
			}
			case loco_t::shape_type_t::light_end: {
				return false;
			}
			default: {
				fan::throw_error("unimplemented");
			}
			}
			if (shape->gint() != nri) {
				fan::throw_error("");
			}
			return false;
		}

		bool shape_serialize(loco_t::shape_t& shape, std::vector<uint8_t>* out) {
			return shape_to_bin(shape, out);
		}

		struct shape_deserialize_t {
			struct {
				// json::iterator doesnt support union
				// i dont want to use variant either so i accept few extra bytes
				json::const_iterator it;
				uint64_t offset = 0;
			}data;
			bool init = false;
			bool was_object = false;

			bool iterate(const fan::json& json, loco_t::shape_t* shape) {
				if (init == false) {
					data.it = json.cbegin();
					init = true;
				}
				if (data.it == json.cend() || was_object) {
					return 0;
				}
				if (json.type() == fan::json::value_t::object) {
					json_to_shape(json, shape);
					was_object = true;
					return 1;
				}
				else {
					json_to_shape(*data.it, shape);
					++data.it;
				}
				return 1;
			}
			bool iterate(const std::vector<uint8_t>& bin_data, loco_t::shape_t* shape) {
				if (bin_data.empty()) {
					return 0;
				}
				else if (data.offset >= bin_data.size()) {
					return 0;
				}
				bin_to_shape(bin_data, shape, data.offset);
				return 1;
			}
		};


		loco_t::shape_t extract_single_shape(const fan::json& json_data) {
			fan::graphics::shape_deserialize_t iterator;
			loco_t::shape_t shape;
			iterator.iterate(json_data["shapes"], &shape);
			return shape;
		}
		fan::json read_json(const std::string& path, const std::source_location& callers_path = std::source_location::current()) {
			std::string json_bytes;
			fan::io::file::read(fan::io::file::find_relative_path(path, callers_path), &json_bytes);
			return fan::json::parse(json_bytes);
		}
		struct animation_t {
			loco_t::animation_nr_t nr;
		};
		// for dme type
		void map_animations(auto& anims) {
			for (auto [i, animation] : fan::enumerate(gloco->all_animations)) {
				for (int j = 0; j < anims.size(); ++j) {
					auto& anim = *anims.NA(j);
					if (animation.second.name == (const char*)anim) {
						anim = animation_t{ .nr = animation.first };
						break;
					}
				}
			}
		}
	}
}

#endif

#if defined(fan_json)
loco_t::shape_t::operator fan::json() {
	fan::json out;
	fan::graphics::shape_to_json(*this, &out);
	return out;
}
loco_t::shape_t::operator std::string() {
	fan::json out;
	fan::graphics::shape_to_json(*this, &out);
	return out.dump(2);
}
loco_t::shape_t::shape_t(const fan::json& json) {
	fan::graphics::json_to_shape(json, this);
}
loco_t::shape_t::shape_t(const std::string& json_string) : shape_t() {
	*this = fan::json::parse(json_string);
}
loco_t::shape_t& loco_t::shape_t::operator=(const fan::json& json) {
	fan::graphics::json_to_shape(json, this);
	return *this;
}
loco_t::shape_t& loco_t::shape_t::operator=(const std::string& json_string) {
	return loco_t::shape_t::operator=(fan::json::parse(json_string));
}
#endif

#include <fan/graphics/collider.h>

//vk

#if defined(fan_vulkan)
#include <fan/graphics/vulkan/uniform_block.h>
#include <fan/graphics/vulkan/memory.h>
#endif

#if defined(fan_gui)
namespace fan {
	namespace graphics {
		using texture_packe0 = loco_t::texture_packe0;
		using ti_t = loco_t::ti_t;

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
	if (gloco->get_renderer() == loco_t::renderer_t::opengl) {
#if defined(loco_opengl)
		return fan::opengl::core::get_draw_mode(internal_draw_mode);
#endif
	}
	else if (gloco->get_renderer() == loco_t::renderer_t::vulkan) {
#if defined(fan_vulkan)
		return fan::vulkan::core::get_draw_mode(internal_draw_mode);
#endif
	}
#if fan_debug >= fan_debug_medium
	fan::throw_error("invalid get");
#endif
	return -1;
}

#if defined(fan_audio)
namespace fan::audio {

	piece_t::piece_t() : fan::audio_t::piece_t{ nullptr } {}

	piece_t::piece_t(const fan::audio_t::piece_t& piece)
		: fan::audio_t::piece_t(piece) {}

	piece_t::piece_t(const std::string& path,
		fan::audio_t::PieceFlag::t flags,
		const std::source_location& callers_path)
		: fan::audio_t::piece_t(open_piece(path, flags, callers_path)) {}

	piece_t::operator fan::audio_t::piece_t& () {
		return *dynamic_cast<fan::audio_t::piece_t*>(this);
	}

	piece_t piece_t::open_piece(const std::string& path,
		fan::audio_t::PieceFlag::t flags,
		const std::source_location& callers_path) {
		fan::audio_t::piece_t* piece = &(fan::audio_t::piece_t&)*this;
		sint32_t err = gloco->audio.Open(piece, fan::io::file::find_relative_path(path, callers_path).string(), flags);
		if (err != 0) {
			fan::throw_error("failed to open piece:" + path, "with error:", err);
		}
		return *this;
	}

	bool piece_t::is_valid() {
		char test_block[sizeof(fan::audio_t::piece_t)];
		memset(test_block, 0, sizeof(fan::audio_t::piece_t));
		return memcmp(&(fan::audio_t::piece_t&)*this, test_block, sizeof(fan::audio_t::piece_t));
	}

	sound_play_id_t piece_t::play(uint32_t group_id, bool loop) {
		fan::audio_t::PropertiesSoundPlay_t p{};
		p.Flags.Loop = loop;
		p.GroupID = 0;
		return gloco->audio.SoundPlay(&*this, &p);
	}

	void piece_t::stop(sound_play_id_t id) {
		fan::audio_t::PropertiesSoundStop_t p{};
		p.FadeOutTo = 0;
		gloco->audio.SoundStop(id, &p);
	}

	void piece_t::resume(uint32_t group_id) {
		gloco->audio.Resume();
	}

	void piece_t::pause(uint32_t group_id) {
		gloco->audio.Pause();
	}

	f32_t piece_t::get_volume() {
		return gloco->audio.GetVolume();
	}

	void piece_t::set_volume(f32_t volume) {
		gloco->audio.SetVolume(volume);
	}

	piece_t open_piece(const std::string& path,
		fan::audio_t::PieceFlag::t flags,
		const std::source_location& callers_path) {
		return piece_t(path, flags, callers_path);
	}

	bool is_piece_valid(piece_t piece) {
		return piece.is_valid();
	}

	sound_play_id_t play(piece_t piece, uint32_t group_id, bool loop) {
		return piece.play(group_id, loop);
	}

	void stop(sound_play_id_t id) {
		fan::audio_t::PropertiesSoundStop_t p{};
		p.FadeOutTo = 0;
		gloco->audio.SoundStop(id, &p);
	}

	void resume(uint32_t group_id) {
		gloco->audio.Resume();
	}

	void pause(uint32_t group_id) {
		gloco->audio.Pause();
	}

	f32_t get_volume() {
		return gloco->audio.GetVolume();
	}

	void set_volume(f32_t volume) {
		gloco->audio.SetVolume(volume);
	}

} // namespace fan::audio

#endif