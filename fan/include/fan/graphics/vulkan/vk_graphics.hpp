#pragma once

#include <fan/graphics/renderer.hpp>

#if fan_renderer == fan_renderer_vulkan

#include <fan/graphics/camera.hpp>

#include <fan/graphics/vulkan/vk_core.hpp>

#include <fan/graphics/shared_graphics.hpp>

namespace fan_2d {

	namespace graphics {

		namespace shader_paths {

			constexpr auto rectangle_vs("glsl/2D/vulkan/rectangle.vert");
			constexpr auto rectangle_fs("glsl/2D/vulkan/rectangle.frag");

			constexpr auto sprite_vs("glsl/2D/vulkan/sprite.vert");
			constexpr auto sprite_fs("glsl/2D/vulkan/sprite.frag");

			constexpr auto text_renderer_vs("glsl/2D/vulkan/text.vert");
			constexpr auto text_renderer_fs("glsl/2D/vulkan/text.frag");

		}

		struct view_projection_t {
			alignas(16) fan::mat4 view;
			alignas(16) fan::mat4 projection;
		};

		struct rectangle {

			rectangle(fan::camera* camera);
			~rectangle();

			void push_back(const fan::vec2& position, const fan::vec2& size, const fan::color& color, f32_t angle = 0);

			void draw();
			// begin must not be bigger than end, otherwise not drawn
			void draw(uint32_t begin, uint32_t end);

			// edits data in buffer. used when editing data, not push_back. requires begin_queue and end_queue
			void release_queue(uint16_t avoid_flags = 0);

			uint32_t size() const;

			void insert(uint32_t i, const fan::vec2& position, const fan::vec2& size, const fan::color& color, f32_t angle = 0);

			void reserve(uint32_t size);
			void resize(uint32_t size, const fan::color& color);

			void draw(uint32_t begin = fan::uninitialized, uint32_t end = fan::uninitialized) const;

			void erase(uint32_t i);
			void erase(uint32_t begin, uint32_t end);

			// erases everything
			void clear();

			rectangle_corners_t get_corners(uint32_t i = 0) const;

			f32_t get_angle(uint32_t i) const;
			void set_angle(uint32_t i, f32_t angle);

			const fan::color get_color(uint32_t i = 0) const;
			void set_color(uint32_t i, const fan::color& color);

			fan::vec2 get_position(uint32_t i = 0) const;
			void set_position(uint32_t i, const fan::vec2& position);

			fan::vec2 get_size(uint32_t i = 0) const;
			void set_size(uint32_t i, const fan::vec2& size);

			bool inside(uint_t i, const fan::vec2& position = fan::math::inf) const;

			fan::camera* m_camera = nullptr;

			view_projection_t view_projection{};


		protected:

			struct instance_t {

				alignas(8) fan::vec2 position;
				alignas(8) fan::vec2 size;
				alignas(16) fan::color color;
				alignas(4) f32_t angle;

			};

			using instance_buffer_t = fan::gpu_memory::buffer_object<instance_t, fan::gpu_memory::buffer_type::buffer>;

			instance_buffer_t* instance_buffer = nullptr;

			uint32_t m_begin = 0;
			uint32_t m_end = 0;

			bool realloc_buffer = false;

			fan::gpu_memory::uniform_handler* uniform_handler = nullptr;

			VkDeviceSize descriptor_offset = 0;
		};

		fan_2d::graphics::image_info load_image(fan::window* window, const std::string& path);

		class sprite {
		public:

			sprite(fan::camera* camera);
			~sprite();

			void push_back(std::unique_ptr<fan_2d::graphics::texture_id_handler>& handler, const fan::vec2& position, const fan::vec2& size, const sprite_properties& properties = sprite_properties());

			void insert(uint32_t i, uint32_t texture_coordinates_i, std::unique_ptr<fan_2d::graphics::texture_id_handler>& handler, const fan::vec2& position, const fan::vec2& size, const sprite_properties& properties = sprite_properties());

			void draw();

			// edits data in buffer. used when editing data, not push_back. requires begin_queue and end_queue
			void release_queue(uint16_t avoid_flags = 0);

			std::size_t size() const;

			void erase(uint32_t i);
			void erase(uint32_t begin, uint32_t end);

			// removes everything
			void clear();

			rectangle_corners_t get_corners(uint32_t i = 0) const;

			f32_t get_angle(uint32_t i) const;
			void set_angle(uint32_t i, f32_t angle);

			fan::vec2 get_position(uint32_t i = 0) const;
			void set_position(uint32_t i, const fan::vec2& position);

			fan::vec2 get_size(uint32_t i = 0) const;
			void set_size(uint32_t i, const fan::vec2& size);

			bool inside(uint_t i, const fan::vec2& position = fan::math::inf) const;

			fan::camera* m_camera = nullptr;

			view_projection_t view_projection{};

		protected:

			struct instance_t {

				fan::vec2 position;
				fan::vec2 size;
				f32_t angle;
				std::array<fan::vec2, 6> texture_coordinate;

			};

			void regenerate_texture_switch();

			using instance_buffer_t = fan::gpu_memory::buffer_object<instance_t, fan::gpu_memory::buffer_type::buffer>;

			instance_buffer_t* instance_buffer = nullptr;

			uint32_t m_begin = 0;
			uint32_t m_end = 0;

			bool realloc_buffer = false;

			std::vector<VkImage> m_textures;

			std::vector<uint32_t> m_switch_texture;

			fan::gpu_memory::uniform_handler* uniform_handler = nullptr;

			VkDeviceSize descriptor_offset = 0;
		};

	}

}

#endif