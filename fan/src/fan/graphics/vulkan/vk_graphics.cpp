#include <fan/graphics/renderer.hpp>

#if fan_renderer == fan_renderer_vulkan

#include <fan/graphics/vulkan/vk_graphics.hpp>

#include <fan/physics/collision/rectangle.hpp>

#include <fan/graphics/image.hpp>

fan_2d::graphics::rectangle::rectangle(fan::camera* camera)
	: m_camera(camera), m_begin(0), m_end(0)
{

	VkVertexInputBindingDescription binding_description;

	binding_description.binding = 0;
	binding_description.inputRate = VkVertexInputRate::VK_VERTEX_INPUT_RATE_INSTANCE;
	binding_description.stride = sizeof(instance_t);

	std::vector<VkVertexInputAttributeDescription> attribute_descriptions;

	attribute_descriptions.resize(4);

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
	attribute_descriptions[2].format = VK_FORMAT_R32G32B32A32_SFLOAT;
	attribute_descriptions[2].offset = offsetof(instance_t, color);

	attribute_descriptions[3].binding = 0;
	attribute_descriptions[3].location = 3;
	attribute_descriptions[3].format = VK_FORMAT_R32_SFLOAT;
	attribute_descriptions[3].offset = offsetof(instance_t, angle);

	fan::vulkan* vk_instance = camera->m_window->m_vulkan;

	fan::vk::shader::recompile_shaders = true;

	vk_instance->pipelines->push_back(
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
		vk_instance->staging_buffer,
		&vk_instance->graphicsQueue
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

	vk_instance->push_back_draw_call([&](uint32_t i, uint32_t j) {

		if (!instance_buffer->buffer->m_buffer_object) {
			return;
		}

		vkCmdBindPipeline(m_camera->m_window->m_vulkan->commandBuffers[0][i], VK_PIPELINE_BIND_POINT_GRAPHICS, m_camera->m_window->m_vulkan->pipelines->pipeline_info[j].pipeline);

		VkDeviceSize offsets[] = { 0 };

		vkCmdBindVertexBuffers(m_camera->m_window->m_vulkan->commandBuffers[0][i], 0, 1, &instance_buffer->buffer->m_buffer_object, offsets);

		vkCmdBindDescriptorSets(m_camera->m_window->m_vulkan->commandBuffers[0][i], VK_PIPELINE_BIND_POINT_GRAPHICS, m_camera->m_window->m_vulkan->pipelines->pipeline_layout, 0, 1, &m_camera->m_window->m_vulkan->texture_handler->descriptor_handler->descriptor_sets[descriptor_offset + i], 0, nullptr);

		vkCmdDraw(m_camera->m_window->m_vulkan->commandBuffers[0][i], 6, m_end, 0, m_begin);

	});

}

fan_2d::graphics::rectangle::~rectangle()
{
	if (instance_buffer) {
		delete instance_buffer;
		instance_buffer = nullptr;
	}

	if (uniform_handler) {
		delete uniform_handler;
		uniform_handler = nullptr;
	}

}

void fan_2d::graphics::rectangle::push_back(const fan::vec2& position, const fan::vec2& size, const fan::color& color, f32_t angle) {

	instance_buffer->push_back(instance_t{ position, size, color, angle });

	realloc_buffer = true;

	if (!fan::gpu_queue) {
		this->release_queue();
	}

}

void fan_2d::graphics::rectangle::draw()
{

	uint32_t begin = 0;
	uint32_t end = this->size();

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
}

void fan_2d::graphics::rectangle::draw(uint32_t begin, uint32_t end)
{
	if (begin >= end) {
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

	view_projection.view = fan::math::look_at_left<fan::mat4>(m_camera->get_position() + fan::vec3(0, 0, 0.1), m_camera->get_position() + fan::vec3(0, 0, 0.1) + m_camera->get_front(), m_camera->world_up);

	view_projection.projection = fan::math::ortho<fan::mat4>((f32_t)0, (f32_t)window_size.x, (f32_t)0, (f32_t)window_size.y, 0.1, 100);
}

void fan_2d::graphics::rectangle::release_queue(uint16_t avoid_flags)
{
	vkDeviceWaitIdle(m_camera->m_window->m_vulkan->device);

	if (realloc_buffer) {
		auto previous_size = instance_buffer->buffer->buffer_size;

		instance_buffer->buffer->free();

		auto new_size = sizeof(instance_t) * instance_buffer->size();

		if (new_size) {
			instance_buffer->buffer->allocate(new_size);

			if (!previous_size) {
				instance_buffer->recreate_command_buffer(new_size, 0, 0);
			}

		}

		m_camera->m_window->m_vulkan->erase_command_buffers();
		m_camera->m_window->m_vulkan->create_command_buffers();

	}

	instance_buffer->write_data();

}

uint32_t fan_2d::graphics::rectangle::size() const {

	return instance_buffer->size();
}

void fan_2d::graphics::rectangle::insert(uint32_t i, const fan::vec2& position, const fan::vec2& size, const fan::color& color, f32_t angle) {
	instance_buffer->m_instance.insert(instance_buffer->m_instance.begin() + i, instance_t{ position, size, color, angle });

	realloc_buffer = true;

	if (!fan::gpu_queue) {
		this->release_queue();
	}
}

void fan_2d::graphics::rectangle::reserve(uint32_t size) {
	instance_buffer->m_instance.reserve(size);
}
void fan_2d::graphics::rectangle::resize(uint32_t size, const fan::color& color) {
	instance_buffer->m_instance.resize(size, instance_t{ 0, 0, color, 0 });

	realloc_buffer = true;

	if (!fan::gpu_queue) {
		this->release_queue();
	}
}

void fan_2d::graphics::rectangle::erase(uint32_t i) {
	instance_buffer->m_instance.erase(instance_buffer->m_instance.begin() + i);

	realloc_buffer = true;

	if (!fan::gpu_queue) {
		this->release_queue();
	}
}
void fan_2d::graphics::rectangle::erase(uint32_t begin, uint32_t end) {
	instance_buffer->m_instance.erase(instance_buffer->m_instance.begin() + begin, instance_buffer->m_instance.begin() + end);

	realloc_buffer = true;

	if (!fan::gpu_queue) {
		this->release_queue();
	}
}

// erases everything
void fan_2d::graphics::rectangle::clear() {
	instance_buffer->m_instance.clear();

	realloc_buffer = true;

	if (!fan::gpu_queue) {
		this->release_queue();
	}
}

fan_2d::graphics::rectangle_corners_t fan_2d::graphics::rectangle::get_corners(uint32_t i) const
{
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

f32_t fan_2d::graphics::rectangle::get_angle(uint32_t i) const
{
	return instance_buffer->get_value(i).angle;
}

void fan_2d::graphics::rectangle::set_angle(uint32_t i, f32_t angle)
{

	instance_buffer->get_value(i).angle = angle;

	if (!fan::gpu_queue) {

		instance_buffer->edit_data(i);

		m_camera->m_window->m_vulkan->erase_command_buffers();

		m_camera->m_window->m_vulkan->create_command_buffers();

	}

}

const fan::color fan_2d::graphics::rectangle::get_color(uint32_t i) const {
	return instance_buffer->get_value(i).color;
}
void fan_2d::graphics::rectangle::set_color(uint32_t i, const fan::color& color) {

	instance_buffer->get_value(i).color = color;

	if (!fan::gpu_queue) {

		instance_buffer->edit_data(i);

		m_camera->m_window->m_vulkan->erase_command_buffers();

		m_camera->m_window->m_vulkan->create_command_buffers();

	}
}

fan::vec2 fan_2d::graphics::rectangle::get_position(uint32_t i) const {
	return instance_buffer->get_value(i).position;
}
void fan_2d::graphics::rectangle::set_position(uint32_t i, const fan::vec2& position) {
	instance_buffer->get_value(i).position = position;

	if (!fan::gpu_queue) {

		instance_buffer->edit_data(i);

		m_camera->m_window->m_vulkan->erase_command_buffers();

		m_camera->m_window->m_vulkan->create_command_buffers();

	}
}

fan::vec2 fan_2d::graphics::rectangle::get_size(uint32_t i) const {
	return instance_buffer->get_value(i).size;
}
void fan_2d::graphics::rectangle::set_size(uint32_t i, const fan::vec2& size) {
	instance_buffer->get_value(i).size = size;

	if (!fan::gpu_queue) {

		instance_buffer->edit_data(i);

		m_camera->m_window->m_vulkan->erase_command_buffers();

		m_camera->m_window->m_vulkan->create_command_buffers();

	}
}

//	void release_queue(bool position, bool size, bool angle, bool color, bool indices);

bool fan_2d::graphics::rectangle::inside(uint_t i, const fan::vec2& position) const {
	auto corners = get_corners(i);

	return fan_2d::collision::rectangle::point_inside(corners[0], corners[1], corners[2], corners[3], position == fan::math::inf ? fan::cast<fan::vec2::value_type>(this->m_camera->m_window->get_mouse_position()) : position);
}

fan_2d::graphics::image_info fan_2d::graphics::load_image(fan::window* window, const std::string& path)
{
	auto image_data = fan::image_loader::load_image(path);

	window->m_vulkan->image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	window->m_vulkan->image_info.imageType = VK_IMAGE_TYPE_2D;
	window->m_vulkan->image_info.extent.width = image_data.size.x;
	window->m_vulkan->image_info.extent.height = image_data.size.y;
	window->m_vulkan->image_info.extent.depth = 1;
	window->m_vulkan->image_info.mipLevels = 1;
	window->m_vulkan->image_info.arrayLayers = 1;
	
	window->m_vulkan->image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
	window->m_vulkan->image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	window->m_vulkan->image_info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
	window->m_vulkan->image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	window->m_vulkan->image_info.samples = VK_SAMPLE_COUNT_1_BIT;
	window->m_vulkan->image_info.flags = 0;
	
	window->m_vulkan->image_info.format = VK_FORMAT_R8G8B8A8_UNORM;

	fan_2d::graphics::image_info info(window);

	info.size = image_data.size;

	if (vkCreateImage(window->m_vulkan->device, &window->m_vulkan->image_info, nullptr, &info.texture->texture_id) != VK_SUCCESS) {
		throw std::runtime_error("failed to create image.");
	}

	VkDeviceSize image_size = image_data.linesize[0] * image_data.size.y;

	if (window->m_vulkan->staging_buffer->buffer_size < image_size) {
		window->m_vulkan->staging_buffer->allocate(image_size);
	}

	void* data;
	vkMapMemory(window->m_vulkan->device, window->m_vulkan->staging_buffer->m_device_memory, 0, image_size, 0, &data);
	memcpy(data, image_data.data[0], static_cast<size_t>(image_size));
	vkUnmapMemory(window->m_vulkan->device, window->m_vulkan->staging_buffer->m_device_memory);

	window->m_vulkan->texture_handler->allocate(info.texture->texture_id);

	window->m_vulkan->texture_handler->transition_image_layout(info.texture->texture_id, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
	window->m_vulkan->texture_handler->copy_buffer_to_image(window->m_vulkan->staging_buffer->m_buffer_object, info.texture->texture_id, info.size.x, info.size.y);
	window->m_vulkan->texture_handler->transition_image_layout(info.texture->texture_id, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);


	delete[] image_data.data[0];

	return info;
}

fan_2d::graphics::sprite::sprite(fan::camera* camera) 
	: 
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

	camera->m_window->m_vulkan->pipelines->push_back(
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
		 camera->m_window->m_vulkan->staging_buffer,
		&camera->m_window->m_vulkan->graphicsQueue
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

	descriptor_offset = vk_instance->texture_handler->descriptor_handler->descriptor_sets.size();

	camera->m_window->m_vulkan->push_back_draw_call([&](uint32_t i, uint32_t j) {

		if (!instance_buffer->buffer->m_buffer_object) {
			return;
		}

		vkCmdBindPipeline(m_camera->m_window->m_vulkan->commandBuffers[0][i], VK_PIPELINE_BIND_POINT_GRAPHICS, m_camera->m_window->m_vulkan->pipelines->pipeline_info[j].pipeline);

		VkDeviceSize offsets[] = { 0 };
		vkCmdBindVertexBuffers(m_camera->m_window->m_vulkan->commandBuffers[0][i], 0, 1, &instance_buffer->buffer->m_buffer_object, offsets);

		for (int k = 0; k < m_switch_texture.size(); k++) {

			vkCmdBindDescriptorSets(m_camera->m_window->m_vulkan->commandBuffers[0][i], VK_PIPELINE_BIND_POINT_GRAPHICS, m_camera->m_window->m_vulkan->pipelines->pipeline_layout, 0, 1, &m_camera->m_window->m_vulkan->texture_handler->descriptor_handler->get(i + descriptor_offset + k * m_camera->m_window->m_vulkan->swapChainImages.size()), 0, nullptr);

			if (k == m_switch_texture.size() - 1) {
				vkCmdDraw(m_camera->m_window->m_vulkan->commandBuffers[0][i], 6, this->size(), 0, m_switch_texture[k]);
			}
			else {
				vkCmdDraw(m_camera->m_window->m_vulkan->commandBuffers[0][i], 6, m_switch_texture[k + 1], 0, m_switch_texture[k]);
			}
		}

	});

}

fan_2d::graphics::sprite::~sprite()
{
	if (instance_buffer) {
		delete instance_buffer;
		instance_buffer = nullptr;
	}
	if (uniform_handler) {
		delete uniform_handler;
		uniform_handler = nullptr;
	}
}

void fan_2d::graphics::sprite::push_back(std::unique_ptr<fan_2d::graphics::texture_id_handler>& handler, const fan::vec2& position, const fan::vec2& size, const sprite_properties& properties)
{
	if (m_switch_texture.size() >= fan::vk::graphics::maximum_textures_per_instance) {
		throw std::runtime_error("maximum textures achieved, create new sprite");
	}

	instance_buffer->m_instance.emplace_back(
		instance_t{ 
			position, 
			size, 
			0,
			properties.texture_coordinates
		}
	);

	realloc_buffer = true;

	if (m_textures.empty() || handler->texture_id != m_textures[m_textures.size() - 1]) {

		m_camera->m_window->m_vulkan->texture_handler->push_back(
			handler->texture_id, uniform_handler, m_camera->m_window->m_vulkan->swapChainImages.size()
		);
	}

	if (m_switch_texture.empty()) {
		m_switch_texture.emplace_back(0);
	}
	else if (m_textures.size() && m_textures[m_textures.size() - 1] != handler->texture_id) {
		m_switch_texture.emplace_back(this->size() - 1);
	}

	m_textures.emplace_back(handler->texture_id);

	if (!fan::gpu_queue) {

		this->release_queue();
	}

}

void fan_2d::graphics::sprite::insert(uint32_t i, uint32_t texture_coordinates_i, std::unique_ptr<fan_2d::graphics::texture_id_handler>& handler, const fan::vec2& position, const fan::vec2& size, const sprite_properties& properties)
{
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

	realloc_buffer = true;

	if (!fan::gpu_queue) {

		this->release_queue();
	}

	m_textures.insert(m_textures.begin() + texture_coordinates_i / 6, handler->texture_id);

	regenerate_texture_switch();
}

void fan_2d::graphics::sprite::draw()
{
	uint32_t begin = 0;
	uint32_t end = this->size();

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
}

void fan_2d::graphics::sprite::release_queue(uint16_t avoid_flags)
{
	if (realloc_buffer) {
		auto previous_size = instance_buffer->buffer->buffer_size;

		instance_buffer->buffer->free();

		auto new_size = sizeof(instance_t) * instance_buffer->size();

		if (new_size) {
			instance_buffer->buffer->allocate(new_size);

			if (!previous_size) {
				instance_buffer->recreate_command_buffer(new_size, 0, 0);
			}
		}

		m_camera->m_window->m_vulkan->erase_command_buffers();
		m_camera->m_window->m_vulkan->create_command_buffers();

		realloc_buffer = false;
	}

	instance_buffer->write_data();
}

std::size_t fan_2d::graphics::sprite::size() const
{
	return instance_buffer->m_instance.size();
}

void fan_2d::graphics::sprite::regenerate_texture_switch()
{
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

void fan_2d::graphics::sprite::erase(uint32_t i)
{
	instance_buffer->m_instance.erase(instance_buffer->m_instance.begin() + i);

	realloc_buffer = true;

	if (!fan::gpu_queue) {
		this->release_queue();
	}

	m_textures.erase(m_textures.begin() + i);

	regenerate_texture_switch();
}

void fan_2d::graphics::sprite::erase(uint32_t begin, uint32_t end)
{
	instance_buffer->m_instance.erase(instance_buffer->m_instance.begin() + begin, instance_buffer->m_instance.begin() + end);

	realloc_buffer = true;

	if (!fan::gpu_queue) {
		this->release_queue();
	}

	m_textures.erase(m_textures.begin() + begin, m_textures.begin() + end);

	regenerate_texture_switch();
}

void fan_2d::graphics::sprite::clear()
{
	instance_buffer->m_instance.clear();

	m_textures.clear();

	m_switch_texture.clear();
}

fan_2d::graphics::rectangle_corners_t fan_2d::graphics::sprite::get_corners(uint32_t i) const
{
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

f32_t fan_2d::graphics::sprite::get_angle(uint32_t i) const
{
	return instance_buffer->get_value(i).angle;
}

void fan_2d::graphics::sprite::set_angle(uint32_t i, f32_t angle)
{

	instance_buffer->get_value(i).angle = angle;

	if (!fan::gpu_queue) {

		instance_buffer->edit_data(i);

		m_camera->m_window->m_vulkan->erase_command_buffers();

		m_camera->m_window->m_vulkan->create_command_buffers();

	}

}

fan::vec2 fan_2d::graphics::sprite::get_position(uint32_t i) const {
	return instance_buffer->get_value(i).position;
}
void fan_2d::graphics::sprite::set_position(uint32_t i, const fan::vec2& position) {
	instance_buffer->get_value(i).position = position;

	if (!fan::gpu_queue) {

		instance_buffer->edit_data(i);

		m_camera->m_window->m_vulkan->erase_command_buffers();

		m_camera->m_window->m_vulkan->create_command_buffers();

	}
}

fan::vec2 fan_2d::graphics::sprite::get_size(uint32_t i) const {
	return instance_buffer->get_value(i).size;
}
void fan_2d::graphics::sprite::set_size(uint32_t i, const fan::vec2& size) {
	instance_buffer->get_value(i).size = size;

	if (!fan::gpu_queue) {

		instance_buffer->edit_data(i);

		m_camera->m_window->m_vulkan->erase_command_buffers();

		m_camera->m_window->m_vulkan->create_command_buffers();

	}
}

//	void release_queue(bool position, bool size, bool angle, bool color, bool indices);

bool fan_2d::graphics::sprite::inside(uint_t i, const fan::vec2& position) const {
	auto corners = get_corners(i);

	return fan_2d::collision::rectangle::point_inside(corners[0], corners[1], corners[2], corners[3], position == fan::math::inf ? fan::cast<fan::vec2::value_type>(this->m_camera->m_window->get_mouse_position()) : position);
}

#endif