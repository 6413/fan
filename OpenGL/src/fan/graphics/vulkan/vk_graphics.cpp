#include <fan/graphics/renderer.hpp>

#if fan_renderer == fan_renderer_vulkan

#include <fan/graphics/vulkan/vk_graphics.hpp>

#include <fan/physics/collision/rectangle.hpp>


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

	vk_instance->pipelines.push_back(
		binding_description,
		attribute_descriptions,
		camera->m_window->get_size(),
		"glsl/2D/test_vertex.vert",
		"glsl/2D/test_fragment.frag"
	);

	instance_buffer = new instance_buffer_t(
		&vk_instance->device, 
		&vk_instance->physicalDevice,
		&vk_instance->commandPool,
		vk_instance->staging_buffer,
		&vk_instance->graphicsQueue,
		1
	);

	vk_instance->push_back_draw_call([&](uint32_t i, uint32_t j) {

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

		if (vkBeginCommandBuffer(m_camera->m_window->m_vulkan->commandBuffers[0][i], &beginInfo) != VK_SUCCESS) {
			throw std::runtime_error("failed to begin recording command buffer!");
		}

		VkRenderPassBeginInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		renderPassInfo.renderPass = m_camera->m_window->m_vulkan->renderPass;
		renderPassInfo.framebuffer = m_camera->m_window->m_vulkan->swapChainFramebuffers[i];
		renderPassInfo.renderArea.offset = { 0, 0 };
		renderPassInfo.renderArea.extent = m_camera->m_window->m_vulkan->swapChainExtent;

		VkClearValue clearColor = { 0.0f, 0.0f, 0.0f, 1.0f };
		renderPassInfo.clearValueCount = 1;
		renderPassInfo.pClearValues = &clearColor;

		vkCmdBeginRenderPass(m_camera->m_window->m_vulkan->commandBuffers[0][i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

		vkCmdBindPipeline(m_camera->m_window->m_vulkan->commandBuffers[0][i], VK_PIPELINE_BIND_POINT_GRAPHICS, m_camera->m_window->m_vulkan->pipelines.pipeline_info[j].pipeline);

		VkBuffer vertexBuffers[] = { instance_buffer->buffer->m_buffer_object };
		VkDeviceSize offsets[] = { 0 };
		vkCmdBindVertexBuffers(m_camera->m_window->m_vulkan->commandBuffers[0][i], 0, 1, vertexBuffers, offsets);

		vkCmdBindDescriptorSets(m_camera->m_window->m_vulkan->commandBuffers[0][i], VK_PIPELINE_BIND_POINT_GRAPHICS, m_camera->m_window->m_vulkan->pipelines.pipeline_layout, 0, 1, &m_camera->m_window->m_vulkan->descriptorSets[i], 0, nullptr);

		vkCmdDraw(m_camera->m_window->m_vulkan->commandBuffers[0][i], 6, m_end, 0, m_begin);

		vkCmdEndRenderPass(m_camera->m_window->m_vulkan->commandBuffers[0][i]);
		
		if (vkEndCommandBuffer(m_camera->m_window->m_vulkan->commandBuffers[0][i]) != VK_SUCCESS) {
			throw std::runtime_error("failed to record command buffer!");
		}

	});

}

fan_2d::graphics::rectangle::~rectangle()
{
	delete instance_buffer;
}

void fan_2d::graphics::rectangle::push_back(const fan::vec2& position, const fan::vec2& size, const fan::color& color, f32_t angle) {

	instance_buffer->push_back(instance_t{ position, size, color, angle });

	realloc_buffer = true;

	if (!fan::gpu_queue) {
		instance_buffer->write_data();

		m_camera->m_window->m_vulkan->erase_command_buffers();
		m_camera->m_window->m_vulkan->create_command_buffers();
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

}

void fan_2d::graphics::rectangle::release_queue(uint16_t avoid_flags)
{
	if (realloc_buffer) {
		instance_buffer->buffer->free_buffer();
		instance_buffer->buffer->allocate_buffer(sizeof(instance_t) * instance_buffer->size());
	}

	instance_buffer->write_data();

	if (realloc_buffer) {
		m_camera->m_window->m_vulkan->erase_command_buffers();
		m_camera->m_window->m_vulkan->create_command_buffers();

		realloc_buffer = false;
	}

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

#endif