#include <fan/graphics/vulkan/vk_gui.hpp>

#if fan_renderer == fan_renderer_vulkan

fan_2d::graphics::gui::text_renderer::text_renderer(fan::camera* camera)
	: m_camera(camera)
{
	font_info = fan::font::parse_font("fonts/arial.fnt");

	font_info.font[' '] = fan::font::font_t({ 0, fan::vec2(fan_2d::graphics::gui::font_properties::space_width, font_info.line_height), 0, (fan::vec2::value_type)fan_2d::graphics::gui::font_properties::space_width });
	font_info.font['\n'] = fan::font::font_t({ 0, fan::vec2(0, font_info.line_height), 0, 0 });

	VkVertexInputBindingDescription binding_description;

	binding_description.binding = 0;
	binding_description.inputRate = VkVertexInputRate::VK_VERTEX_INPUT_RATE_INSTANCE;
	binding_description.stride = sizeof(instance_t);

	std::vector<VkVertexInputAttributeDescription> attribute_descriptions;

	attribute_descriptions.resize(12);

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

	attribute_descriptions[3].binding = 0;
	attribute_descriptions[3].location = 3;
	attribute_descriptions[3].format = VK_FORMAT_R32G32B32A32_SFLOAT;
	attribute_descriptions[3].offset = offsetof(instance_t, color);

	attribute_descriptions[4].binding = 0;
	attribute_descriptions[4].location = 4;
	attribute_descriptions[4].format = VK_FORMAT_R32_SFLOAT;
	attribute_descriptions[4].offset = offsetof(instance_t, font_size);

	attribute_descriptions[5].binding = 0;
	attribute_descriptions[5].location = 5;
	attribute_descriptions[5].format = VK_FORMAT_R32G32_SFLOAT;
	attribute_descriptions[5].offset = offsetof(instance_t, text_rotation_point);

	for (int i = 0; i < 6; i++) {
		attribute_descriptions[i + 6].binding = 0;
		attribute_descriptions[i + 6].location = i + 6;
		attribute_descriptions[i + 6].format = VK_FORMAT_R32G32_SFLOAT;
		attribute_descriptions[i + 6].offset = offsetof(instance_t, texture_coordinate) + i * sizeof(fan::vec2);
	}

	fan::vk::shader::recompile_shaders = true;

	camera->m_window->m_vulkan->pipelines[(int)fan_2d::graphics::shape::triangle]->push_back(
		binding_description,
		attribute_descriptions,
		camera->m_window->get_size(),
		shader_paths::text_renderer_vs,
		shader_paths::text_renderer_fs,
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
			m_camera->m_window->m_vulkan->erase_command_buffers();
			camera->m_window->m_vulkan->create_command_buffers();
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

	if (!image) {
		fan::vulkan* vk_instance = camera->m_window->m_vulkan;

		image = std::make_unique<fan_2d::graphics::image_info>(fan_2d::graphics::load_image(camera->m_window, "fonts/arial.png"));

		descriptor_offsets.emplace_back(vk_instance->texture_handler->push_back(
			image.get()->texture.get()->texture_id,
			uniform_handler,
			vk_instance->swapChainImages.size(), 1)
		);
	}

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

		for (int k = 0; k < descriptor_offsets.size(); k++) {
			vkCmdBindDescriptorSets(
				m_camera->m_window->m_vulkan->commandBuffers[0][i], 
				VK_PIPELINE_BIND_POINT_GRAPHICS, 
				m_camera->m_window->m_vulkan->pipelines[(int)fan_2d::graphics::shape::triangle]->pipeline_layout, 
				0, 
				1, 
				&m_camera->m_window->m_vulkan->texture_handler->descriptor_handler->get(i + descriptor_offsets[k] * m_camera->m_window->m_vulkan->swapChainImages.size()), 
				0, 
				nullptr
			);
		}

		vkCmdDraw(m_camera->m_window->m_vulkan->commandBuffers[0][i], 6, instance_buffer->size(), 0, 0);
	});

}

fan_2d::graphics::gui::text_renderer::~text_renderer() {
	
	if (instance_buffer) {
		delete instance_buffer;
		instance_buffer = nullptr;
	}

	if (uniform_handler) {
		delete uniform_handler;
		uniform_handler = nullptr;
	}
}

void fan_2d::graphics::gui::text_renderer::draw()
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

	fan_2d::graphics::shape shape = fan_2d::graphics::shape::triangle;
	if (m_camera->m_window->m_vulkan->set_draw_call_order(m_camera->m_window->m_vulkan->draw_order_id++, &shape, 1, (void*)this)) {
		m_camera->m_window->m_vulkan->reload_swapchain = true;
	}
}

void fan_2d::graphics::gui::text_renderer::push_back(const fan::utf16_string& text, f32_t font_size, fan::vec2 position, const fan::color& text_color) {

	m_text.emplace_back(text);

	f32_t advance = 0;

	const auto original_position = position;

	m_position.emplace_back(position);

	auto previous_offset = instance_buffer->size();

	auto convert = convert_font_size(font_size);

	for (int i = 0; i < text.size(); i++) {

		auto letter_offset = font_info.font[text[i]].offset;
		auto letter_size = font_info.font[text[i]].size;

		push_back_letter(text[i], font_size, position, text_color, advance);

		if (text[i] != ' ' && text[i] != '\n') {
			advance += font_info.font[text[i]].advance;
		}

	}

	advance = 0;

	auto text_size = get_text_size(text, font_size) / 2;

	for (int i = 0; i < text.size(); i++) {
		auto letter_offset = font_info.font[text[i]].offset;
		auto letter_size = font_info.font[text[i]].size;

		instance_buffer->get_value(previous_offset + i).text_rotation_point = original_position + fan::vec2(text_size.x, text_size.y);

		if (text[i] != ' ' && text[i] != '\n') {
			advance += font_info.font[text[i]].advance;
		}
	}

	if (m_indices.empty()) {
		m_indices.emplace_back(text.size());
	}
	else {
		m_indices.emplace_back(m_indices[m_indices.size() - 1] + text.size());
	}
}

void fan_2d::graphics::gui::text_renderer::insert(uint32_t i, const fan::utf16_string& text, f32_t font_size, fan::vec2 position, const fan::color& text_color) {
	
	m_text.insert(m_text.begin() + i, text);

	f32_t advance = 0;

	const auto original_position = position;

	m_position.insert(m_position.begin() + i, position);

	auto previous_offset = instance_buffer->size();

	for (int j = 0; j < text.size(); j++) {

		insert_letter(i, j, text[j], font_size, position, text_color, advance);

		if (text[j] != ' ' && text[j] != '\n') {
			advance += font_info.font[text[j]].advance;
		}
	}

	advance = 0;

	auto text_size = get_text_size(text, font_size) / 2;

	for (int i = 0; i < text.size(); i++) {
		auto letter_offset = font_info.font[text[i]].offset;
		auto letter_size = font_info.font[text[i]].size;

		instance_buffer->get_value(previous_offset + i).text_rotation_point = original_position + fan::vec2(text_size.x, text_size.y);

		if (text[i] != ' ' && text[i] != '\n') {
			advance += font_info.font[text[i]].advance;
		}
	}

	regenerate_indices();
}

void fan_2d::graphics::gui::text_renderer::set_position(uint32_t i, const fan::vec2& position) {

	if (m_position[i] == position) {
		return;
	}

	m_position[i] = position;

	f32_t advance = 0;

	auto index = i == 0 ? 0 : m_indices[i - 1];

	auto font_size = get_font_size(i);

	auto convert_size = convert_font_size(font_size);

	fan::vec2 original_position = position;

	for (int j = 0; j < m_text[i].size(); j++) {

		auto letter = get_letter_info(m_text[i][j], font_size);

		fan::vec2 new_position = original_position + (letter.offset + fan::vec2(advance, 0) + letter.size / 2);

		if (m_text[i][j] == '\n') {
			advance = 0;																		
			original_position.y += font_info.line_height;
		}																						
		else if (m_text[i][j] == ' ') {
			advance += get_letter_info(' ', font_size).advance;
		}
		else {
			advance += letter.advance;
		}

		instance_buffer->get_value(index + j).position = new_position;
	}
}

uint32_t fan_2d::graphics::gui::text_renderer::size() const {
	return m_text.size();
}

f32_t fan_2d::graphics::gui::text_renderer::get_font_size(uintptr_t i) const {
	return instance_buffer->get_value(i == 0 ? 0 : m_indices[i - 1]).font_size;
}

void fan_2d::graphics::gui::text_renderer::set_font_size(uint32_t i, f32_t font_size) {
	const auto text = get_text(i);

	const auto position = get_position(i);

	const auto color = get_text_color(i);

	this->erase(i);
	this->insert(i, text, font_size, position, color);
}

void fan_2d::graphics::gui::text_renderer::set_angle(uint32_t i, f32_t angle)
{
	for (int j = 0; j < m_text[i].size(); j++) {
		instance_buffer->m_instance[(i == 0 ? 0 : m_indices[i - 1]) + j].angle = angle;	
	}
}

void fan_2d::graphics::gui::text_renderer::erase(uintptr_t i) {
	
	uint64_t begin = i == 0 ? 0 : m_indices[i - 1];
	uint64_t end = m_indices[i];

	instance_buffer->m_instance.erase(instance_buffer->m_instance.begin() + begin, instance_buffer->m_instance.begin() + end);

	m_position.erase(m_position.begin() + i);
	m_text.erase(m_text.begin() + i);

	this->regenerate_indices();
}

void fan_2d::graphics::gui::text_renderer::erase(uintptr_t begin, uintptr_t end) {

	uint64_t begin_ = begin == 0 ? 0 : m_indices[begin - 1];
	uint64_t end_ = end == 0 ? 0 : m_indices[end - 1];

	instance_buffer->m_instance.erase(instance_buffer->m_instance.begin() + begin_, instance_buffer->m_instance.begin() + end_);

	m_position.erase(m_position.begin() + begin, m_position.begin() + end);
	m_text.erase(m_text.begin() + begin, m_text.begin() + end);

	this->regenerate_indices();
}

void fan_2d::graphics::gui::text_renderer::clear() {
	
	instance_buffer->m_instance.clear();

	m_position.clear();
	m_text.clear();

	m_indices.clear();
}

void fan_2d::graphics::gui::text_renderer::set_text(uint32_t i, const fan::utf16_string& text) {

	auto font_size = this->get_font_size(i);
	auto position = this->get_position(i);
	auto color = this->get_text_color(i);

	this->erase(i);

	this->insert(i, text, font_size, position, color);
}

fan::color fan_2d::graphics::gui::text_renderer::get_text_color(uint32_t i, uint32_t j) {
	return instance_buffer->get_value(i == 0 ? 0 : m_indices[i - 1] + j).color;
}
// hint edit_data - i, i + text.size()
void fan_2d::graphics::gui::text_renderer::set_text_color(uint32_t i, const fan::color& color) {
	
	auto index = i == 0 ? 0 : m_indices[i - 1];

	for (int j = 0; j < m_text[i].size(); j++) {
		instance_buffer->m_instance[index + j].color = color;
	}
}
void fan_2d::graphics::gui::text_renderer::set_text_color(uint32_t i, uint32_t j, const fan::color& color) {
	auto index = i == 0 ? 0 : m_indices[i - 1];

	instance_buffer->m_instance[index + j].color = color;
}

void fan_2d::graphics::gui::text_renderer::write_data() {
	instance_buffer->write_data();
}

void fan_2d::graphics::gui::text_renderer::edit_data(uint32_t i) {

	uint32_t begin = 0;

	for (int j = 0; j < i; j++) {
		begin += m_text[j].size();
	}

	instance_buffer->edit_data(begin, begin + m_text[i].size() - 1);
}

void fan_2d::graphics::gui::text_renderer::edit_data(uint32_t begin, uint32_t end) {

	uint32_t size = 0;
	uint32_t begin_ = 0;

	for (int i = 0; i < begin; i++) {
		begin_ += m_text[i].size();
	}

	for (int i = begin; i <= end; i++) {
		size += m_text[i].size();
	}

	instance_buffer->edit_data(begin_, (size - begin_) - 1);
}

#define get_letter_infos 																\
const fan::vec2 letter_position = font_info.font[letter].position;						\
const fan::vec2 letter_size = font_info.font[letter].size;								\
const fan::vec2 letter_offset = font_info.font[letter].offset;							\
																						\
fan::vec2 texture_position = fan::vec2(letter_position + 1) / image->size;				\
fan::vec2 texture_size = fan::vec2(letter_position + letter_size - 1) / image->size;		\
																						\
const auto converted_font_size = convert_font_size(font_size);							\
																						\
if (letter == '\n') {																	\
	advance = 0;																		\
	position.y += font_info.line_height * converted_font_size;							\
	texture_position = 0;																\
	texture_size = 0;																	\
}																						\
else if (letter == ' ') {																\
	advance += font_info.font[' '].advance;												\
	texture_position = 0;																\
	texture_size = 0;																	\
}																						\
																						\
std::array<fan::vec2, 6> texture_coordiantes;												\
																						\
texture_coordiantes = {																    \
	fan::vec2(texture_position.x, texture_size.y),										\
	fan::vec2(texture_size.x, texture_size.y),											\
	fan::vec2(texture_size.x, texture_position.y),										\
																						\
	fan::vec2(texture_position.x, texture_size.y),										\
	fan::vec2(texture_position.x, texture_position.y),									\
	fan::vec2(texture_size.x, texture_position.y)										\
};


void fan_2d::graphics::gui::text_renderer::insert_letter(uint32_t i, uint32_t j, wchar_t letter, f32_t font_size, fan::vec2& position, const fan::color& color, f32_t& advance) {
	
	get_letter_infos;

	auto index = i == 0 ? 0 : m_indices[i - 1 >= m_indices.size() ? m_indices.size() - 1 : i - 1];

	// ?
	instance_buffer->m_instance.insert(instance_buffer->m_instance.begin() + index + j, 
		instance_t{
			position + (letter_offset + fan::vec2(advance, 0) + letter_size / 2) * converted_font_size,
			letter_size * converted_font_size,
			0,
			color,
			font_size,
			0,
			texture_coordiantes
		}
	);
}

void fan_2d::graphics::gui::text_renderer::push_back_letter(wchar_t letter, f32_t font_size, fan::vec2& position, const fan::color& color, f32_t& advance) {

	get_letter_infos;

	instance_buffer->m_instance.emplace_back(
		instance_t{
			position + (letter_offset + fan::vec2(advance, 0) + letter_size / 2) * converted_font_size,
			letter_size * converted_font_size,
			0,
			color,
			font_size,
			0,
			texture_coordiantes
		}
	);

}

#endif