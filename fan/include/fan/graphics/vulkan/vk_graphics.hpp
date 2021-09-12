#pragma once

#include <fan/graphics/renderer.hpp>

#if fan_renderer == fan_renderer_vulkan

#include <fan/graphics/camera.hpp>

#include <fan/graphics/vulkan/vk_core.hpp>

#include <fan/graphics/shared_graphics.hpp>

#include <fan/physics/collision/rectangle.hpp>
#include <fan/physics/collision/circle.hpp>

namespace fan_2d {

	namespace graphics {

		namespace shader_paths {

			constexpr auto vertice_vector_vs("glsl/2D/vulkan/vertice_vector.vert");
			constexpr auto vertice_vector_fs("glsl/2D/vulkan/vertice_vector.frag");

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

		template <typename instance_t>
		class basic_vertice_vector {
		public:

			//virtual basic_vertice_vector(fan::camera* camera) = 0;

			void reserve(uint32_t size) {
				instance_buffer->m_instance.reserve(size);
			}
			virtual void resize(uint32_t size, const fan::color& color) = 0;

			virtual void draw(fan_2d::graphics::shape shape, uint32_t single_draw_amount, uint32_t begin, uint32_t end) = 0;

			void erase(uint32_t i) {
				instance_buffer->m_instance.erase(instance_buffer->m_instance.begin() + i);
			}
			void erase(uint32_t begin, uint32_t end) {
				instance_buffer->m_instance.erase(instance_buffer->m_instance.begin() + begin, instance_buffer->m_instance.begin() + end);
			}

			virtual uint32_t size() const {
				return instance_buffer->m_instance.size();
			}

			void write_data() {
				instance_buffer->write_data();
			}

			void edit_data(uint32_t i) {
				instance_buffer->edit_data(i);
			}

			void edit_data(uint32_t begin, uint32_t end) {
				instance_buffer->edit_data(begin, end);
			}

			fan::camera* m_camera = nullptr;

			view_projection_t view_projection{};

		protected:

			using instance_buffer_t = fan::gpu_memory::buffer_object<instance_t, fan::gpu_memory::buffer_type::buffer>;

			instance_buffer_t* instance_buffer = nullptr;

			uint32_t m_begin = 0;
			uint32_t m_end = 0;

			uint32_t m_single_draw = 0;

			fan::gpu_memory::uniform_handler* uniform_handler = nullptr;

			VkDeviceSize descriptor_offset = 0;

		};

		struct vertex_instance_t {
			alignas(8) fan::vec2 position;
			alignas(4) f32_t angle;
			alignas(8) fan::vec2 rotation_point;
			alignas(16) fan::vec3 rotation_vector = fan::vec3(0, 0, 1);
			alignas(16) fan::color color;
		};

		struct vertice_vector : public basic_vertice_vector<vertex_instance_t> {

			using properties_t = vertex_instance_t;

			using instance_t = vertex_instance_t;

			vertice_vector(fan::camera* camera) {

				m_camera = camera;

				VkVertexInputBindingDescription binding_description;

				binding_description.binding = 0;
				binding_description.inputRate = VkVertexInputRate::VK_VERTEX_INPUT_RATE_VERTEX;
				binding_description.stride = sizeof(instance_t);

				std::vector<VkVertexInputAttributeDescription> attribute_descriptions;

				attribute_descriptions.resize(5);

				attribute_descriptions[0].binding = 0;
				attribute_descriptions[0].location = 0;
				attribute_descriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
				attribute_descriptions[0].offset = offsetof(properties_t, position);
				attribute_descriptions[1].binding = 0;
				attribute_descriptions[1].location = 1;
				attribute_descriptions[1].format = VK_FORMAT_R32_SFLOAT;
				attribute_descriptions[1].offset = offsetof(properties_t, angle);
				attribute_descriptions[2].binding = 0;
				attribute_descriptions[2].location = 2;
				attribute_descriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
				attribute_descriptions[2].offset = offsetof(properties_t, rotation_point);
				attribute_descriptions[3].binding = 0;
				attribute_descriptions[3].location = 3;
				attribute_descriptions[3].format = VK_FORMAT_R32G32B32_SFLOAT;
				attribute_descriptions[3].offset = offsetof(properties_t, rotation_vector);
				attribute_descriptions[4].binding = 0;
				attribute_descriptions[4].location = 4;
				attribute_descriptions[4].format = VK_FORMAT_R32G32B32A32_SFLOAT;
				attribute_descriptions[4].offset = offsetof(properties_t, color);

				fan::vulkan* vk_instance = camera->m_window->m_vulkan;

				fan::vk::shader::recompile_shaders = true;

				vk_instance->pipelines[(int)fan_2d::graphics::shape::line]->push_back(
					binding_description,
					attribute_descriptions,
					camera->m_window->get_size(),
					shader_paths::vertice_vector_vs,
					shader_paths::vertice_vector_fs,
					camera->m_window->m_vulkan->swapChainExtent
				);

				vk_instance->pipelines[(int)fan_2d::graphics::shape::line_strip]->push_back(
					binding_description,
					attribute_descriptions,
					camera->m_window->get_size(),
					shader_paths::vertice_vector_vs,
					shader_paths::vertice_vector_fs,
					camera->m_window->m_vulkan->swapChainExtent
				);

				vk_instance->pipelines[(int)fan_2d::graphics::shape::triangle]->push_back(
					binding_description,
					attribute_descriptions,
					camera->m_window->get_size(),
					shader_paths::vertice_vector_vs,
					shader_paths::vertice_vector_fs,
					camera->m_window->m_vulkan->swapChainExtent
				);

				vk_instance->pipelines[(int)fan_2d::graphics::shape::triangle_strip]->push_back(
					binding_description,
					attribute_descriptions,
					camera->m_window->get_size(),
					shader_paths::vertice_vector_vs,
					shader_paths::vertice_vector_fs,
					camera->m_window->m_vulkan->swapChainExtent
				);

				vk_instance->pipelines[(int)fan_2d::graphics::shape::triangle_fan]->push_back(
					binding_description,
					attribute_descriptions,
					camera->m_window->get_size(),
					shader_paths::vertice_vector_vs,
					shader_paths::vertice_vector_fs,
					camera->m_window->m_vulkan->swapChainExtent
				);

				fan::vk::shader::recompile_shaders = false;

				instance_buffer = new instance_buffer_t(
					&vk_instance->device, 
					&vk_instance->physicalDevice,
					&vk_instance->commandPool,
					&vk_instance->graphicsQueue,
					[&]{
						vkDeviceWaitIdle(m_camera->m_window->m_vulkan->device);

						m_camera->m_window->m_vulkan->erase_command_buffers();
						m_camera->m_window->m_vulkan->create_command_buffers();
					}
				);

				uniform_handler = new fan::gpu_memory::uniform_handler(
					&vk_instance->device,
					&vk_instance->physicalDevice,
					vk_instance->swapChainImages.size(),
					&view_projection,
					sizeof(view_projection_t)
				);

				vk_instance->uniform_buffers.emplace_back(
					uniform_handler
				);

				vk_instance->texture_handler->image_views.push_back(nullptr);

				descriptor_offset = vk_instance->texture_handler->descriptor_handler->descriptor_sets.size();

				vk_instance->texture_handler->descriptor_handler->push_back(
					vk_instance->device,
					uniform_handler,
					vk_instance->texture_handler->descriptor_handler->descriptor_set_layout,
					vk_instance->texture_handler->descriptor_handler->descriptor_pool,
					nullptr,
					nullptr,
					vk_instance->swapChainImages.size()
				);

				fan_2d::graphics::shape* arr = new fan_2d::graphics::shape[(int)fan_2d::graphics::shape::last];

				for (int i = 0; i < (int)fan_2d::graphics::shape::last; i++) {
					arr[i] = (fan_2d::graphics::shape)i;
				}

				vk_instance->push_back_draw_call(vk_instance->draw_order_id++, arr, (int)fan_2d::graphics::shape::last, (void*)this, [&](uint32_t i, uint32_t j, void* base, fan_2d::graphics::shape shape) {

					if (!instance_buffer->buffer->m_buffer_object || !m_single_draw || shape != fan::vulkan::draw_topology) {
						return;
					}

					vkCmdBindPipeline(
						m_camera->m_window->m_vulkan->commandBuffers[0][i],
						VK_PIPELINE_BIND_POINT_GRAPHICS,
						m_camera->m_window->m_vulkan->pipelines[(int)fan::vulkan::draw_topology]->pipeline_info[j].pipeline
					);

					VkDeviceSize offsets[] = { 0 };

					vkCmdBindVertexBuffers(m_camera->m_window->m_vulkan->commandBuffers[0][i], 0, 1, &instance_buffer->buffer->m_buffer_object, offsets);

					vkCmdBindDescriptorSets(
						m_camera->m_window->m_vulkan->commandBuffers[0][i],
						VK_PIPELINE_BIND_POINT_GRAPHICS,
						m_camera->m_window->m_vulkan->pipelines[(int)fan::vulkan::draw_topology]->pipeline_layout,
						0,
						1,
						&m_camera->m_window->m_vulkan->texture_handler->descriptor_handler->descriptor_sets[descriptor_offset + i],
						0,
						nullptr
					);

					for (int k = m_begin; k < instance_buffer->size() / m_single_draw && k < m_end; k++) {
						fan::print(k, k * m_single_draw);
						vkCmdDraw(m_camera->m_window->m_vulkan->commandBuffers[0][i], m_single_draw, 1, k * m_single_draw, 0);
					}

				});

				delete[] arr;
			}

			void push_back(const vertice_vector::properties_t& properties) {
				instance_buffer->push_back(properties);
			}

			void reserve(uint32_t size) {
				instance_buffer->m_instance.reserve(size);
			}
			void resize(uint32_t size, const fan::color& color) {
				instance_buffer->m_instance.resize(size, { 0, 0, 0, 0, color });
			}

			void draw(fan_2d::graphics::shape shape, uint32_t single_draw_amount, uint32_t begin, uint32_t end) {
				fan::vulkan::draw_topology = (decltype(fan::vulkan::draw_topology))shape;

				if (begin > end) {
					return;
				}

				m_single_draw = single_draw_amount;

				bool reload = false;

				if (begin != m_begin) {
					reload = true;
					m_begin = begin;
				}

				if (end != m_end) {
					reload = true;
					m_end = end;
				}

				if (reload) {

					vkDeviceWaitIdle(m_camera->m_window->m_vulkan->device);

					m_camera->m_window->m_vulkan->erase_command_buffers();
					m_camera->m_window->m_vulkan->create_command_buffers();

				}

				fan::vec2 window_size = m_camera->m_window->get_size();

				view_projection.view = fan::math::look_at_left<fan::mat4>(m_camera->get_position() + fan::vec3(0, 0, 0.1), m_camera->get_position() + fan::vec3(0, 0, 0.1) + m_camera->get_front(), m_camera->world_up);

				view_projection.projection = fan::math::ortho<fan::mat4>((f32_t)0, (f32_t)window_size.x, (f32_t)0, (f32_t)window_size.y, 0.1, 100);

				fan_2d::graphics::shape* arr = new fan_2d::graphics::shape[(int)fan_2d::graphics::shape::last];

				for (int i = 0; i < (int)fan_2d::graphics::shape::last; i++) {
					arr[i] = (fan_2d::graphics::shape)i;
				}

				if (m_camera->m_window->m_vulkan->set_draw_call_order(m_camera->m_window->m_vulkan->draw_order_id++, arr, (int)fan_2d::graphics::shape::last, (void*)this)) {
					m_camera->m_window->m_vulkan->reload_swapchain = true;
				}

				delete[] arr;
			}

			void erase(uint32_t i) {
				instance_buffer->m_instance.erase(instance_buffer->m_instance.begin() + i);
			}
			void erase(uint32_t begin, uint32_t end) {
				instance_buffer->m_instance.erase(instance_buffer->m_instance.begin() + begin, instance_buffer->m_instance.begin() + end);
			}

			fan::vec2 get_rotation_point(uint32_t i = 0) const  {
				return instance_buffer->get_value(i).rotation_point;
			}

			fan::vec2 get_position(uint32_t i) const {
				return instance_buffer->m_instance[i].position;
			}
			void set_position(uint32_t i, const fan::vec2& position) {
				instance_buffer->get_value(i).position = position;
			}

			fan::color get_color(uint32_t i) const {
				return instance_buffer->m_instance[i].color;
			}
			void set_color(uint32_t i, const fan::color& color) {
				instance_buffer->get_value(i).color = color;
			}

			properties_t get_property(uint32_t i) const {
				return instance_buffer->m_instance[i];
			}

			void set_rotation_point(uint32_t i, const fan::vec2& rotation_point)  {
				instance_buffer->get_value(i).rotation_point = rotation_point;
			}

			fan::vec2 get_rotation_vector(uint32_t i = 0) const  {
				return instance_buffer->get_value(i).rotation_vector;
			}
			void set_rotation_vector(uint32_t i, const fan::vec2& rotation_vector)  {
				instance_buffer->get_value(i).rotation_vector = rotation_vector;
			}

			uint32_t size() const {
				return instance_buffer->m_instance.size();
			}
		};


		struct rectangle {

			// angle in radians
			// rotation point from world position
			// must init rotation_point
			struct properties_t {
				alignas(8) fan::vec2 position;
				alignas(8) fan::vec2 size;
				alignas(4) f32_t angle = 0;
				alignas(8) fan::vec2 rotation_point = fan::math::inf;
				alignas(16) fan::vec3 rotation_vector = fan::vec3(0, 0, 1);
				alignas(16) fan::color color;
			};

			rectangle(fan::camera* camera) : m_camera(camera), m_begin(0), m_end(0)
			{

				VkVertexInputBindingDescription binding_description;

				binding_description.binding = 0;
				binding_description.inputRate = VkVertexInputRate::VK_VERTEX_INPUT_RATE_INSTANCE;
				binding_description.stride = sizeof(properties_t);

				std::vector<VkVertexInputAttributeDescription> attribute_descriptions;

				attribute_descriptions.resize(6);

				attribute_descriptions[0].binding = 0;
				attribute_descriptions[0].location = 0;
				attribute_descriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
				attribute_descriptions[0].offset = offsetof(properties_t, position);

				attribute_descriptions[1].binding = 0;
				attribute_descriptions[1].location = 1;
				attribute_descriptions[1].format = VK_FORMAT_R32G32_SFLOAT;
				attribute_descriptions[1].offset = offsetof(properties_t, size);

				attribute_descriptions[2].binding = 0;
				attribute_descriptions[2].location = 2;
				attribute_descriptions[2].format = VK_FORMAT_R32_SFLOAT;
				attribute_descriptions[2].offset = offsetof(properties_t, angle);

				attribute_descriptions[3].binding = 0;
				attribute_descriptions[3].location = 3;
				attribute_descriptions[3].format = VK_FORMAT_R32G32_SFLOAT;
				attribute_descriptions[3].offset = offsetof(properties_t, rotation_point);

				attribute_descriptions[4].binding = 0;
				attribute_descriptions[4].location = 4;
				attribute_descriptions[4].format = VK_FORMAT_R32G32B32_SFLOAT;
				attribute_descriptions[4].offset = offsetof(properties_t, rotation_vector);

				attribute_descriptions[5].binding = 0;
				attribute_descriptions[5].location = 5;
				attribute_descriptions[5].format = VK_FORMAT_R32G32B32A32_SFLOAT;
				attribute_descriptions[5].offset = offsetof(properties_t, color);

				fan::vulkan* vk_instance = camera->m_window->m_vulkan;

				fan::vk::shader::recompile_shaders = true;

				vk_instance->pipelines[(int)fan_2d::graphics::shape::triangle]->push_back(
					binding_description,
					attribute_descriptions,
					camera->m_window->get_size(),
					shader_paths::rectangle_vs,
					shader_paths::rectangle_fs,
					camera->m_window->m_vulkan->swapChainExtent
				);

				fan::vk::shader::recompile_shaders = false;

				instance_buffer = new instance_buffer_t(
					&vk_instance->device, 
					&vk_instance->physicalDevice,
					&vk_instance->commandPool,
					&vk_instance->graphicsQueue,
					[&]{
						vkDeviceWaitIdle(m_camera->m_window->m_vulkan->device);

						m_camera->m_window->m_vulkan->erase_command_buffers();
						m_camera->m_window->m_vulkan->create_command_buffers();
					}
				);

				uniform_handler = new fan::gpu_memory::uniform_handler(
					&vk_instance->device,
					&vk_instance->physicalDevice,
					vk_instance->swapChainImages.size(),
					&view_projection,
					sizeof(view_projection_t)
				);

				vk_instance->uniform_buffers.emplace_back(
					uniform_handler
				);

				vk_instance->texture_handler->image_views.push_back(nullptr);

				descriptor_offset = vk_instance->texture_handler->descriptor_handler->descriptor_sets.size();

				vk_instance->texture_handler->descriptor_handler->push_back(
					vk_instance->device,
					uniform_handler,
					vk_instance->texture_handler->descriptor_handler->descriptor_set_layout,
					vk_instance->texture_handler->descriptor_handler->descriptor_pool,
					nullptr,
					nullptr,
					vk_instance->swapChainImages.size()
				);

				fan_2d::graphics::shape shape = fan_2d::graphics::shape::triangle;

				vk_instance->push_back_draw_call(vk_instance->draw_order_id++, &shape, 1, (void*)this, [&](uint32_t i, uint32_t j, void* base, fan_2d::graphics::shape shape) {

					if (!instance_buffer->buffer->m_buffer_object || shape != fan_2d::graphics::shape::triangle || base != this) {
						return;
					}

					vkCmdBindPipeline(
						m_camera->m_window->m_vulkan->commandBuffers[0][i], 
						VK_PIPELINE_BIND_POINT_GRAPHICS, 
						m_camera->m_window->m_vulkan->pipelines[(int)fan_2d::graphics::shape::triangle]->pipeline_info[j].pipeline
					);

					VkDeviceSize offsets[] = { 0 };

					vkCmdBindVertexBuffers(
						m_camera->m_window->m_vulkan->commandBuffers[0][i], 
						0, 
						1, 
						&instance_buffer->buffer->m_buffer_object, 
						offsets
					);

					vkCmdBindDescriptorSets(
						m_camera->m_window->m_vulkan->commandBuffers[0][i], 
						VK_PIPELINE_BIND_POINT_GRAPHICS, 
						m_camera->m_window->m_vulkan->pipelines[(int)fan_2d::graphics::shape::triangle]->pipeline_layout, 
						0, 
						1, 
						&m_camera->m_window->m_vulkan->texture_handler->descriptor_handler->descriptor_sets[descriptor_offset + i], 
						0, 
						nullptr
					);

					vkCmdDraw(m_camera->m_window->m_vulkan->commandBuffers[0][i], 6, m_end, 0, m_begin);

					});

			}
			~rectangle() {
				if (instance_buffer) {
					delete instance_buffer;
					instance_buffer = nullptr;
				}

				if (uniform_handler) {
					delete uniform_handler;
					uniform_handler = nullptr;
				}

			}

			// must init rotation_point
			void push_back(const properties_t& properties) {
				instance_buffer->push_back(properties);
			}

			void draw() {
				this->draw(0, this->size());
			}

			// begin must not be bigger than end, otherwise not drawn
			void draw(uint32_t begin, uint32_t end) {

				if (begin > end) {
					return;
				}

				bool reload = false;

				if (begin != m_begin) {
					reload = true;
					m_begin = begin;
				}

				if (end != m_end) {
					reload = true;
					m_end = end;
				}

				if (reload) {

					vkDeviceWaitIdle(m_camera->m_window->m_vulkan->device);

					m_camera->m_window->m_vulkan->erase_command_buffers();
					m_camera->m_window->m_vulkan->create_command_buffers();

				}

				fan::vec2 window_size = m_camera->m_window->get_size();

				view_projection.view = fan::math::look_at_left<fan::mat4>(m_camera->get_position() + fan::vec3(0.1, 0, 0.1), m_camera->get_position() + fan::vec3(0.1, 0, 0.1) + m_camera->get_front(), m_camera->world_up);

				view_projection.projection = fan::math::ortho<fan::mat4>((f32_t)0, (f32_t)window_size.x, (f32_t)0, (f32_t)window_size.y, 0.01, 1000);

				fan_2d::graphics::shape shape = fan_2d::graphics::shape::triangle;
				if (m_camera->m_window->m_vulkan->set_draw_call_order(m_camera->m_window->m_vulkan->draw_order_id++, &shape, 1, (void*)this)) {
					m_camera->m_window->m_vulkan->reload_swapchain = true;
				}
			}

			uint32_t size() const {
				return instance_buffer->size();
			}

			// must init rotation_point
			void insert(uint32_t i, const properties_t& properties) {
				instance_buffer->m_instance.insert(instance_buffer->m_instance.begin() + i, properties);
			}

			void reserve(uint32_t size) {
				instance_buffer->m_instance.reserve(size);
			}
			void resize(uint32_t size, const fan::color& color) {
				instance_buffer->m_instance.resize(size, properties_t{ 0, 0, 0, 0, 0, color });
			}

			void erase(uint32_t i) {
				instance_buffer->m_instance.erase(instance_buffer->m_instance.begin() + i);
			}
			void erase(uint32_t begin, uint32_t end) {
				instance_buffer->m_instance.erase(instance_buffer->m_instance.begin() + begin, instance_buffer->m_instance.begin() + end);
			}

			// erases everything
			void clear() {
				instance_buffer->m_instance.clear();
			}

			rectangle_corners_t get_corners(uint32_t i = 0) const {
				auto position = this->get_position(i);
				auto size = this->get_size(i);

				fan::vec2 mid = position + size / 2;

				auto corners = get_rectangle_corners_no_rotation(position, size);

				f32_t angle = -instance_buffer->get_value(i).angle;

				fan::vec2 top_left = get_transformed_point(corners[0] - mid, angle) + mid;
				fan::vec2 top_right = get_transformed_point(corners[1] - mid, angle) + mid;
				fan::vec2 bottom_left = get_transformed_point(corners[2] - mid, angle) + mid;
				fan::vec2 bottom_right = get_transformed_point(corners[3] - mid, angle) + mid;

				return { top_left, top_right, bottom_left, bottom_right };
			}

			properties_t get_property(uint32_t i) const {
				return instance_buffer->m_instance[i];
			}

			f32_t get_angle(uint32_t i) const {
				return instance_buffer->get_value(i).angle;
			}
			void set_angle(uint32_t i, f32_t angle) {
				instance_buffer->get_value(i).angle = angle;
			}

			const fan::color get_color(uint32_t i = 0) const  {
				return instance_buffer->get_value(i).color;
			}
			void set_color(uint32_t i, const fan::color& color)  {
				instance_buffer->get_value(i).color = color;
			}

			fan::vec2 get_position(uint32_t i = 0) const {
				return instance_buffer->get_value(i).position;
			}
			void set_position(uint32_t i, const fan::vec2& position) {
				instance_buffer->m_instance[i].position = position;
			}

			fan::vec2 get_size(uint32_t i = 0) const  {
				return instance_buffer->get_value(i).size;
			}
			void set_size(uint32_t i, const fan::vec2& size)  {
				instance_buffer->get_value(i).size = size;
			}

			fan::vec2 get_rotation_point(uint32_t i = 0) const  {
				return instance_buffer->get_value(i).rotation_point;
			}
			void set_rotation_point(uint32_t i, const fan::vec2& rotation_point)  {
				instance_buffer->get_value(i).rotation_point = rotation_point;
			}

			fan::vec2 get_rotation_vector(uint32_t i = 0) const  {
				return instance_buffer->get_value(i).rotation_vector;
			}
			void set_rotation_vector(uint32_t i, const fan::vec2& rotation_vector)  {
				instance_buffer->get_value(i).rotation_vector = rotation_vector;
			}

			bool inside(uintptr_t i, const fan::vec2& position = fan::math::inf) const  {
				auto corners = get_corners(i);

				return fan_2d::collision::rectangle::point_inside(corners[0], corners[1], corners[2], corners[3], position == fan::math::inf ? fan::cast<fan::vec2::value_type>(this->m_camera->m_window->get_mouse_position()) : position);
			}

			void write_data() {
				instance_buffer->write_data();
			}

			void edit_data(uint32_t i) {
				instance_buffer->edit_data(i);
			}

			void edit_data(uint32_t begin, uint32_t end) {
				instance_buffer->edit_data(begin, end);
			}

			fan::camera* m_camera = nullptr;

			view_projection_t view_projection{};


		protected:

			using instance_buffer_t = fan::gpu_memory::buffer_object<properties_t, fan::gpu_memory::buffer_type::buffer>;

			instance_buffer_t* instance_buffer = nullptr;

			uint32_t m_begin = 0;
			uint32_t m_end = 0;

			fan::gpu_memory::uniform_handler* uniform_handler = nullptr;

			VkDeviceSize descriptor_offset = 0;
		};

		// makes line from src (line start top left) to dst (line end top left)
		struct line : protected fan_2d::graphics::rectangle {

		public:

			using fan_2d::graphics::rectangle::rectangle;

			void push_back(const fan::vec2& src, const fan::vec2& dst, const fan::color& color, f32_t thickness = 1) {

				line_instance.emplace_back(line_instance_t{
					src,
					dst,
					thickness
				});

				rectangle::push_back({
					src,
					fan::vec2((dst - src).length(), thickness),
					-fan::math::aim_angle(src, dst),
					src,
					fan::vec3(0, 0, 1),
					color
				});

			}

			fan::vec2 get_src(uint32_t i) const {
				return line_instance[i].src;
			}
			fan::vec2 get_dst(uint32_t i) const {
				return line_instance[i].dst;
			}

			void set_line(uint32_t i, const fan::vec2& src, const fan::vec2& dst) {

				const auto thickness = this->get_thickness(i);

				rectangle::instance_buffer->m_instance[i] = {
					src - (this->get_rotation_point(i) - src),
					fan::vec2((dst - src).length(), thickness),
					-fan::math::aim_angle(src, dst),
					this->get_rotation_point(i),
					fan::vec3(0, 0, 1),
					this->get_color(i)
				};
			}

			f32_t get_thickness(uint32_t i) const {
				return line_instance[i].thickness;
			}
			void set_thickness(uint32_t i, const f32_t thickness) {

				const auto src = line_instance[i].src;
				const auto dst = line_instance[i].dst;

				const auto new_src = src;
				const auto new_dst = fan::vec2((dst - src).length(), thickness);

				line_instance[i].thickness = thickness;

				rectangle::set_position(i, new_src);
				rectangle::set_size(i, new_dst);
			}

			using fan_2d::graphics::rectangle::draw;
			using fan_2d::graphics::rectangle::edit_data;
			using fan_2d::graphics::rectangle::write_data;
			using fan_2d::graphics::rectangle::get_color;
			using fan_2d::graphics::rectangle::set_color;
			using fan_2d::graphics::rectangle::get_rotation_point;
			using fan_2d::graphics::rectangle::set_rotation_point;
			using fan_2d::graphics::rectangle::size;

		protected:

			struct line_instance_t {
				fan::vec2 src;
				fan::vec2 dst;
				f32_t thickness;
			};

			std::vector<line_instance_t> line_instance;

		};

		static void generate_mipmaps(fan::vulkan* vulkan, VkImage image, VkFormat image_format, fan::vec2i image_size, uint32_t mipLevels) {
			// Check if image format supports linear blitting
			VkFormatProperties formatProperties;
			vkGetPhysicalDeviceFormatProperties(vulkan->physicalDevice, image_format, &formatProperties);

			if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) {
				throw std::runtime_error("texture image format does not support linear blitting");
			}

			VkCommandBuffer commandBuffer = fan::gpu_memory::begin_command_buffer(vulkan->device, vulkan->commandPool);

			VkImageMemoryBarrier barrier{};
			barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			barrier.image = image;
			barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			barrier.subresourceRange.baseArrayLayer = 0;
			barrier.subresourceRange.layerCount = 1;
			barrier.subresourceRange.levelCount = 1;

			int32_t mipWidth = image_size.x;
			int32_t mipHeight = image_size.y;

			for (uint32_t i = 1; i < mipLevels; i++) {
				barrier.subresourceRange.baseMipLevel = i - 1;
				barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
				barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
				barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
				barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

				vkCmdPipelineBarrier(commandBuffer,
					VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
					0, nullptr,
					0, nullptr,
					1, &barrier);

				VkImageBlit blit{};
				blit.srcOffsets[0] = {0, 0, 0};
				blit.srcOffsets[1] = {mipWidth, mipHeight, 1};
				blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
				blit.srcSubresource.mipLevel = i - 1;
				blit.srcSubresource.baseArrayLayer = 0;
				blit.srcSubresource.layerCount = 1;
				blit.dstOffsets[0] = {0, 0, 0};
				blit.dstOffsets[1] = { mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1 };
				blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
				blit.dstSubresource.mipLevel = i;
				blit.dstSubresource.baseArrayLayer = 0;
				blit.dstSubresource.layerCount = 1;

				vkCmdBlitImage(commandBuffer,
					image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
					image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
					1, &blit,
					VK_FILTER_LINEAR);

				barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
				barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
				barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

				vkCmdPipelineBarrier(commandBuffer,
					VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
					0, nullptr,
					0, nullptr,
					1, &barrier);

				if (mipWidth > 1) mipWidth /= 2;
				if (mipHeight > 1) mipHeight /= 2;
			}

			barrier.subresourceRange.baseMipLevel = mipLevels - 1;
			barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

			vkCmdPipelineBarrier(commandBuffer,
				VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
				0, nullptr,
				0, nullptr,
				1, &barrier);

			fan::gpu_memory::end_command_buffer(commandBuffer, vulkan->device, vulkan->commandPool, vulkan->graphicsQueue);
		}

		static fan_2d::graphics::image_t load_image(fan::window* window, const pixel_data_t& pixel_data) {

			fan::image_loader::image_data id;

			for (int i = 0; i < std::size(pixel_data.pixels); i++) {
				id.data[i] = (uint8_t*)pixel_data.pixels[i];
				id.linesize[i] = pixel_data.linesize[i];
			}

			id.format = pixel_data.format;
			id.size = pixel_data.size;

			auto image_data = fan::image_loader::convert_format(id, AVPixelFormat::AV_PIX_FMT_RGB24);

			window->m_vulkan->image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
			window->m_vulkan->image_info.imageType = VK_IMAGE_TYPE_2D;
			window->m_vulkan->image_info.extent.width = image_data.size.x;
			window->m_vulkan->image_info.extent.height = image_data.size.y;
			window->m_vulkan->image_info.extent.depth = 1;
			window->m_vulkan->image_info.mipLevels = std::floor(std::log2(std::max(image_data.size.x, image_data.size.y))) + 1;
			window->m_vulkan->image_info.arrayLayers = 1;

			window->m_vulkan->image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
			window->m_vulkan->image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			window->m_vulkan->image_info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
			window->m_vulkan->image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
			window->m_vulkan->image_info.samples = window->m_vulkan->msaa_samples;
			window->m_vulkan->image_info.flags = 0;

			window->m_vulkan->image_info.format = VK_FORMAT_R8G8B8A8_UNORM;

			fan_2d::graphics::image_t image = new std::remove_pointer<fan_2d::graphics::image_t>::type(window);

			image->size = image_data.size;

			if (vkCreateImage(window->m_vulkan->device, &window->m_vulkan->image_info, nullptr, &image->texture) != VK_SUCCESS) {
				throw std::runtime_error("failed to create image->");
			}

			VkDeviceSize image_size = image_data.linesize[0] * image_data.size.y;

			if (window->m_vulkan->staging_buffer->buffer_size < image_size) {
				window->m_vulkan->staging_buffer->allocate(image_size);
			}

			void* data;
			vkMapMemory(window->m_vulkan->device, window->m_vulkan->staging_buffer->m_device_memory, 0, image_size, 0, &data);
			memcpy(data, image_data.data[0], static_cast<size_t>(image_size));
			vkUnmapMemory(window->m_vulkan->device, window->m_vulkan->staging_buffer->m_device_memory);

			delete[] image_data.data[0];

			window->m_vulkan->texture_handler->allocate(image->texture);

			window->m_vulkan->texture_handler->transition_image_layout(image->texture, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, window->m_vulkan->image_info.mipLevels);
			window->m_vulkan->texture_handler->copy_buffer_to_image(window->m_vulkan->staging_buffer->m_buffer_object, image->texture, image->size.x, image->size.y);

			generate_mipmaps(window->m_vulkan, image->texture, VK_FORMAT_R8G8B8A8_UNORM, image_data.size, window->m_vulkan->image_info.mipLevels);

			return image;
		}

		static fan_2d::graphics::image_t load_image(fan::window* window, const std::string& path) {
			auto image_data = fan::image_loader::load_image(path);

			window->m_vulkan->image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
			window->m_vulkan->image_info.imageType = VK_IMAGE_TYPE_2D;
			window->m_vulkan->image_info.extent.width = image_data.size.x;
			window->m_vulkan->image_info.extent.height = image_data.size.y;
			window->m_vulkan->image_info.extent.depth = 1;
			window->m_vulkan->image_info.mipLevels = std::floor(std::log2(std::max(image_data.size.x, image_data.size.y))) + 1;
			window->m_vulkan->image_info.arrayLayers = 1;

			window->m_vulkan->image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
			window->m_vulkan->image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			window->m_vulkan->image_info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
			window->m_vulkan->image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
			window->m_vulkan->image_info.samples = window->m_vulkan->msaa_samples;
			window->m_vulkan->image_info.flags = 0;

			window->m_vulkan->image_info.format = VK_FORMAT_R8G8B8A8_UNORM;

			fan_2d::graphics::image_t image = new std::remove_pointer<fan_2d::graphics::image_t>::type(window);

			image->size = image_data.size;

			if (vkCreateImage(window->m_vulkan->device, &window->m_vulkan->image_info, nullptr, &image->texture) != VK_SUCCESS) {
				throw std::runtime_error("failed to create image->");
			}

			VkDeviceSize image_size = image_data.linesize[0] * image_data.size.y;

			if (window->m_vulkan->staging_buffer->buffer_size < image_size) {
				window->m_vulkan->staging_buffer->allocate(image_size);
			}

			void* data;
			vkMapMemory(window->m_vulkan->device, window->m_vulkan->staging_buffer->m_device_memory, 0, image_size, 0, &data);
			memcpy(data, image_data.data[0], static_cast<size_t>(image_size));
			vkUnmapMemory(window->m_vulkan->device, window->m_vulkan->staging_buffer->m_device_memory);

			delete[] image_data.data[0];

			window->m_vulkan->texture_handler->allocate(image->texture);

			window->m_vulkan->texture_handler->transition_image_layout(image->texture, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, window->m_vulkan->image_info.mipLevels);
			window->m_vulkan->texture_handler->copy_buffer_to_image(window->m_vulkan->staging_buffer->m_buffer_object, image->texture, image->size.x, image->size.y);

			generate_mipmaps(window->m_vulkan, image->texture, VK_FORMAT_R8G8B8A8_UNORM, image_data.size, window->m_vulkan->image_info.mipLevels);

			return image;
		}

		class sprite {
		public:

			struct properties_t {

				fan_2d::graphics::image_t image; 
				fan::vec2 position;
				fan::vec2 size;

				std::array<fan::vec2, 6> texture_coordinates = {
					fan::vec2(0, 1),
					fan::vec2(1, 1),
					fan::vec2(1, 0),

					fan::vec2(0, 1),
					fan::vec2(0, 0),
					fan::vec2(1, 0)
				};

				f32_t transparency = 1;

			};

			sprite(fan::camera* camera) : 
				m_camera(camera)
			{
				VkVertexInputBindingDescription binding_description;

				binding_description.binding = 0;
				binding_description.inputRate = VkVertexInputRate::VK_VERTEX_INPUT_RATE_INSTANCE;
				binding_description.stride = sizeof(instance_t);

				std::vector<VkVertexInputAttributeDescription> attribute_descriptions;

				attribute_descriptions.resize(9);

				attribute_descriptions[0].binding = 0;
				attribute_descriptions[0].location = 0;
				attribute_descriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
				attribute_descriptions[0].offset = offsetof(instance_t, position);

				attribute_descriptions[1].binding = 0;
				attribute_descriptions[1].location = 1;
				attribute_descriptions[1].format = VK_FORMAT_R32G32_SFLOAT;
				attribute_descriptions[1].offset = offsetof(instance_t, size);

				attribute_descriptions[2].binding = 0;
				attribute_descriptions[2].location = 2;
				attribute_descriptions[2].format = VK_FORMAT_R32_SFLOAT;
				attribute_descriptions[2].offset = offsetof(instance_t, angle);

				for (int i = 0; i < 6; i++) {
					attribute_descriptions[i + 3].binding = 0;
					attribute_descriptions[i + 3].location = i + 3;
					attribute_descriptions[i + 3].format = VK_FORMAT_R32G32_SFLOAT;
					attribute_descriptions[i + 3].offset = offsetof(instance_t, texture_coordinate) + i * sizeof(fan::vec2);
				}

				fan::vk::shader::recompile_shaders = true;

				camera->m_window->m_vulkan->pipelines[(int)fan_2d::graphics::shape::triangle]->push_back(
					binding_description,
					attribute_descriptions,
					camera->m_window->get_size(),
					shader_paths::sprite_vs,
					shader_paths::sprite_fs,
					camera->m_window->m_vulkan->swapChainExtent
				);

				fan::vk::shader::recompile_shaders = false;

				auto vk_instance = camera->m_window->m_vulkan;

				instance_buffer = new instance_buffer_t(
					&camera->m_window->m_vulkan->device,
					&camera->m_window->m_vulkan->physicalDevice,
					&camera->m_window->m_vulkan->commandPool,
					&camera->m_window->m_vulkan->graphicsQueue,
					[&]{
						vkDeviceWaitIdle(m_camera->m_window->m_vulkan->device);

						m_camera->m_window->m_vulkan->erase_command_buffers();
						m_camera->m_window->m_vulkan->create_command_buffers();
					}
				);

				uniform_handler = new fan::gpu_memory::uniform_handler(
					&vk_instance->device,
					&vk_instance->physicalDevice,
					vk_instance->swapChainImages.size(),
					&view_projection,
					sizeof(view_projection_t)
				);

				vk_instance->uniform_buffers.emplace_back(
					uniform_handler
				);

				fan_2d::graphics::shape shape = fan_2d::graphics::shape::triangle;

				camera->m_window->m_vulkan->push_back_draw_call(vk_instance->draw_order_id++, &shape, 1, (void*)this, [&](uint32_t i, uint32_t j, void* base, fan_2d::graphics::shape shape) {

					if (!instance_buffer->buffer->m_buffer_object) {
						return;
					}

					vkCmdBindPipeline(
						m_camera->m_window->m_vulkan->commandBuffers[0][i], 
						VK_PIPELINE_BIND_POINT_GRAPHICS, 
						m_camera->m_window->m_vulkan->pipelines[(int)fan_2d::graphics::shape::triangle]->pipeline_info[j].pipeline
					);

					VkDeviceSize offsets[] = { 0 };
					vkCmdBindVertexBuffers(m_camera->m_window->m_vulkan->commandBuffers[0][i], 0, 1, &instance_buffer->buffer->m_buffer_object, offsets);

					for (int k = 0; k < m_switch_texture.size(); k++) {

						for (int h = 0; h < descriptor_offsets.size(); h++) {

							vkCmdBindDescriptorSets(
								m_camera->m_window->m_vulkan->commandBuffers[0][i], 
								VK_PIPELINE_BIND_POINT_GRAPHICS, 
								m_camera->m_window->m_vulkan->pipelines[(int)fan_2d::graphics::shape::triangle]->pipeline_layout, 
								0, 
								1, 
								&m_camera->m_window->m_vulkan->texture_handler->descriptor_handler->get(i + descriptor_offsets[h] * m_camera->m_window->m_vulkan->swapChainImages.size() + k * m_camera->m_window->m_vulkan->swapChainImages.size()), 
								0, 
								nullptr
							);
						}

						if (k == m_switch_texture.size() - 1) {
							vkCmdDraw(m_camera->m_window->m_vulkan->commandBuffers[0][i], 6, this->size(), 0, m_switch_texture[k]);
						}
						else {
							vkCmdDraw(m_camera->m_window->m_vulkan->commandBuffers[0][i], 6, m_switch_texture[k + 1], 0, m_switch_texture[k]);
						}
					}

					});

			}
			~sprite() {
				if (instance_buffer) {
					delete instance_buffer;
					instance_buffer = nullptr;
				}
				if (uniform_handler) {
					delete uniform_handler;
					uniform_handler = nullptr;
				}
			}

			// fan_2d::graphics::load_image::texture
			void push_back(const sprite::properties_t& properties) {

				if (m_switch_texture.size() >= fan::vk::graphics::maximum_textures_per_instance) {
					throw std::runtime_error("maximum textures achieved, create new sprite");
				}

				instance_buffer->m_instance.emplace_back(
					instance_t{ 
						properties.position, 
						properties.size, 
						0,
						properties.texture_coordinates
					}
				);

				if (m_textures.empty() || properties.image->texture != m_textures[m_textures.size() - 1]) {

					descriptor_offsets.emplace_back(m_camera->m_window->m_vulkan->texture_handler->push_back(
						properties.image->texture, uniform_handler, m_camera->m_window->m_vulkan->swapChainImages.size(),
						std::floor(std::log2(std::max(properties.size.x, properties.size.y))) + 1
					));
				}

				if (m_switch_texture.empty()) {
					m_switch_texture.emplace_back(0);
				}
				else if (m_textures.size() && m_textures[m_textures.size() - 1] != properties.image->texture) {
					m_switch_texture.emplace_back(this->size() - 1);
				}

				m_textures.emplace_back(properties.image->texture);
			}

			/*void insert(uint32_t i, uint32_t texture_coordinates_i, std::unique_ptr<fan_2d::graphics::texture_id_handler>& handler, const fan::vec2& position, const fan::vec2& size, const sprite_properties& properties = sprite_properties()) {
				instance_buffer->m_instance.insert(instance_buffer->m_instance.begin() + i, 
					instance_t{ 
						position, 
						size, 
						0,
						properties.texture_coordinates
					}
				);

				if (m_textures.empty() || handler->texture_id != m_textures[m_textures.size() - 1]) {
					assert(0);
					m_camera->m_window->m_vulkan->texture_handler->push_back(
						handler->texture_id, uniform_handler, m_camera->m_window->m_vulkan->swapChainImages.size()
					);
				}

				m_textures.insert(m_textures.begin() + texture_coordinates_i / 6, handler->texture_id);

				regenerate_texture_switch();
			}*/

			void draw(uint32_t begin = fan::uninitialized, uint32_t end = fan::uninitialized) {
				begin = 0;
				end = this->size();

				bool reload = false;

				if (begin != m_begin) {
					reload = true;
					m_begin = begin;
				}

				if (end != m_end) {
					reload = true;
					m_end = end;
				}

				if (reload) {

					vkDeviceWaitIdle(m_camera->m_window->m_vulkan->device);

					m_camera->m_window->m_vulkan->erase_command_buffers();
					m_camera->m_window->m_vulkan->create_command_buffers();

				}

				fan::vec2 window_size = m_camera->m_window->get_size();

				view_projection.view = fan::math::look_at_left<fan::mat4>(m_camera->get_position() + fan::vec3(0, 0, 0.1), m_camera->get_position() + fan::vec3(0, 0, 0.1) + m_camera->get_front(), m_camera->world_up);

				view_projection.projection = fan::math::ortho<fan::mat4>((f32_t)0, (f32_t)window_size.x, (f32_t)0, (f32_t)window_size.y, 0.1, 100);

				fan_2d::graphics::shape shape = fan_2d::graphics::shape::triangle;
				if (m_camera->m_window->m_vulkan->set_draw_call_order(m_camera->m_window->m_vulkan->draw_order_id++, &shape, 1, (void*)this)) {
					m_camera->m_window->m_vulkan->reload_swapchain = true;
				}
			}

			std::size_t size() const {
				return instance_buffer->m_instance.size();
			}

			void erase(uint32_t i) {
				instance_buffer->m_instance.erase(instance_buffer->m_instance.begin() + i);

				m_textures.erase(m_textures.begin() + i);

				regenerate_texture_switch();
			}
			
			void erase(uint32_t begin, uint32_t end) {
				instance_buffer->m_instance.erase(instance_buffer->m_instance.begin() + begin, instance_buffer->m_instance.begin() + end);

				m_textures.erase(m_textures.begin() + begin, m_textures.begin() + end);

				regenerate_texture_switch();
			}

			// removes everything
			void clear() {
				instance_buffer->m_instance.clear();

				m_textures.clear();

				m_switch_texture.clear();
			}

			rectangle_corners_t get_corners(uint32_t i = 0) const {
				auto position = this->get_position(i);
				auto size = this->get_size(i);

				fan::vec2 mid = position + size / 2;

				auto corners = get_rectangle_corners_no_rotation(position, size);

				f32_t angle = -instance_buffer->get_value(i).angle;

				fan::vec2 top_left = get_transformed_point(corners[0] - mid, angle) + mid;
				fan::vec2 top_right = get_transformed_point(corners[1] - mid, angle) + mid;
				fan::vec2 bottom_left = get_transformed_point(corners[2] - mid, angle) + mid;
				fan::vec2 bottom_right = get_transformed_point(corners[3] - mid, angle) + mid;

				return { top_left, top_right, bottom_left, bottom_right };
			}

			f32_t get_angle(uint32_t i) const {
				return instance_buffer->get_value(i).angle;
			}
			void set_angle(uint32_t i, f32_t angle) {
				instance_buffer->get_value(i).angle = angle;
			}

			fan::vec2 get_position(uint32_t i = 0) const  {
				return instance_buffer->get_value(i).position;
			}
			void set_position(uint32_t i, const fan::vec2& position)  {
				instance_buffer->get_value(i).position = position;
			}

			fan::vec2 get_size(uint32_t i = 0) const  {
				return instance_buffer->get_value(i).size;
			}
			void set_size(uint32_t i, const fan::vec2& size)  {
				instance_buffer->get_value(i).size = size;
			}

			bool inside(uintptr_t i, const fan::vec2& position = fan::math::inf) const {
				auto corners = get_corners(i);

				return fan_2d::collision::rectangle::point_inside(corners[0], corners[1], corners[2], corners[3], position == fan::math::inf ? fan::cast<fan::vec2::value_type>(this->m_camera->m_window->get_mouse_position()) : position);
			}

			void write_data() {
				instance_buffer->write_data();
			}

			void edit_data(uint32_t i) {
				instance_buffer->edit_data(i);
			}

			void edit_data(uint32_t begin, uint32_t end) {
				instance_buffer->edit_data(begin, end);
			}

			fan::camera* m_camera = nullptr;

			view_projection_t view_projection{};

		protected:

			struct instance_t {

				alignas(8) fan::vec2 position;
				alignas(8) fan::vec2 size;
				alignas(4) f32_t angle;
				alignas(64) /* ? */ std::array<fan::vec2, 6> texture_coordinate;

			};

			void regenerate_texture_switch() {
				m_switch_texture.clear();

				for (int i = 0; i < m_textures.size(); i++) {
					if (m_switch_texture.empty()) {
						m_switch_texture.emplace_back(0);
					}
					else if (m_textures.size() && m_textures[i] != m_textures[i - 1]) {
						m_switch_texture.emplace_back(i);
					}
				}
			}

			using instance_buffer_t = fan::gpu_memory::buffer_object<instance_t, fan::gpu_memory::buffer_type::buffer>;

			instance_buffer_t* instance_buffer = nullptr;

			uint32_t m_begin = 0;
			uint32_t m_end = 0;

			std::vector<VkImage> m_textures;

			std::vector<uint32_t> m_switch_texture;

			fan::gpu_memory::uniform_handler* uniform_handler = nullptr;

			std::vector<VkDeviceSize> descriptor_offsets;
		};

		#include <fan/graphics/shared_inline_graphics.hpp>

	}

}

#endif