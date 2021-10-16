#include <fan/graphics/renderer.hpp>

#if fan_renderer == fan_renderer_opengl

#include <fan/graphics/opengl/gl_gui.hpp>

fan_2d::graphics::gui::circle::circle(fan::camera* camera)
	: fan_2d::graphics::circle(camera)
{
	this->m_camera->m_window->add_resize_callback([&] (const fan::window* window, const fan::vec2i&)  {

		auto offset = this->m_camera->m_window->get_size() - this->m_camera->m_window->get_previous_size();

		for (int i = 0; i < this->size(); i++) {
			this->set_position(i, this->get_position(i) + fan::vec2(offset.x, 0));
		}
	});
}

fan_2d::graphics::gui::text_renderer::text_renderer(fan::camera* camera) : m_camera(camera), sprite(camera, fan::shader(fan_2d::graphics::shader_paths::text_renderer_vs, fan_2d::graphics::shader_paths::text_renderer_fs)) {

	if (!image) {
		image = fan_2d::graphics::load_image(camera->m_window, "fonts/monoscape.png");
	}

	font_info = fan::font::parse_font("fonts/monoscape.fnt");

	font_info.font[' '] = fan::font::font_t({ 0, fan::vec2(fan_2d::graphics::gui::font_properties::space_width, font_info.line_height), 0, (fan::vec2::value_type)fan_2d::graphics::gui::font_properties::space_width });
	font_info.font['\n'] = fan::font::font_t({ 0, fan::vec2(0, font_info.line_height), 0, 0 });

	fan::bind_vao(sprite::vao_handler::m_buffer_object, [&] {
		font_size_t::initialize_buffers(fan::shader::id, location_font_size, true, 1);
		rotation_point_t::initialize_buffers(fan::shader::id, location_rotation_point, true, 2);
	});
}

fan_2d::graphics::gui::text_renderer::~text_renderer() {

}

void fan_2d::graphics::gui::text_renderer::draw() {
	fan_2d::graphics::sprite::draw();
}

void fan_2d::graphics::gui::text_renderer::push_back(const fan::utf16_string& text, f32_t font_size, fan::vec2 position, const fan::color& text_color) {

	if (text.empty()) {
		throw std::runtime_error("text cannot be empty");
	}

	m_text.emplace_back(text);
	m_new_lines.emplace_back(0);

	f32_t advance = 0;

	const auto original_position = position;

	m_position.emplace_back(position);

	auto convert = convert_font_size(font_size);

	advance -= font_info.font[text[0]].offset.x;

	for (int i = 0; i < text.size(); i++) {

		push_back_letter(text[i], font_size, position, text_color, advance);

		if (text[i] != ' ' && text[i] != '\n') {
			advance += font_info.font[text[i]].advance;
		}
		if (text[i] == '\n') {
			m_new_lines[this->size() - 1]++;
		}

	}

	auto text_size = get_text_size(text, font_size) / 2;

	for (int i = 0; i < text.size(); i++) {
		auto letter_offset = font_info.font[text[i]].offset;
		auto letter_size = font_info.font[text[i]].size;

		rotation_point_t::push_back(original_position + fan::vec2(advance / 2 * convert, fan_2d::graphics::gui::text_renderer::font_info.font['\n'].size.y / 2 * convert));

	}

	if (m_indices.empty()) {
		m_indices.emplace_back(text.size());
	}
	else {
		m_indices.emplace_back(m_indices[m_indices.size() - 1] + text.size());
	}
}

void fan_2d::graphics::gui::text_renderer::insert(uint32_t i, const fan::utf16_string& text, f32_t font_size, fan::vec2 position, const fan::color& text_color)
{
	m_text.insert(m_text.begin() + i, text);
	m_new_lines.insert(m_new_lines.begin() + i, 0);

	f32_t advance = 0;

	const auto original_position = position;

	m_position.insert(m_position.begin() + i, position);
	 
	advance -= font_info.font[text[0]].offset.x;

	for (int j = 0; j < text.size(); j++) {

		insert_letter(i, j, text[j], font_size, position, text_color, advance);

		if (text[j] != ' ' && text[j] != '\n') {
			advance += font_info.font[text[j]].advance;
		}
		if (text[j] == '\n') {
			m_new_lines[i]++;
		}
	}

	advance = 0;

	auto text_size = get_text_size(text, font_size) / 2;

	for (int i = 0; i < text.size(); i++) {
		auto letter_offset = font_info.font[text[i]].offset;
		auto letter_size = font_info.font[text[i]].size;

		rotation_point_t::push_back(original_position + fan::vec2(text_size.x, text_size.y));

		if (text[i] != ' ' && text[i] != '\n') {
			advance += font_info.font[text[i]].advance;
		}
	}

	regenerate_indices();
}

void fan_2d::graphics::gui::text_renderer::set_position(uint32_t i, const fan::vec2& position)
{
	const uint32_t index = i == 0 ? 0 : m_indices[i - 1];

	const fan::vec2 offset = position - get_position(i);

	for (int j = 0; j < m_text[i].size(); j++) {

		sprite::set_position(index + j, sprite::get_position(index + j) + offset);

	}
}

uint32_t fan_2d::graphics::gui::text_renderer::size() const {
	return m_text.size();
}

f32_t fan_2d::graphics::gui::text_renderer::get_font_size(uintptr_t i) const
{
	if (font_size_t::m_buffer_object.size() <= i) { // for empty strings
		return 0;
	}
	return font_size_t::get_value(i == 0 ? 0 : m_indices[i - 1]);
}

void fan_2d::graphics::gui::text_renderer::set_font_size(uint32_t i, f32_t font_size)
{
	const auto text = get_text(i);

	const auto position = get_position(i);

	const auto color = get_text_color(i);

	this->erase(i);
	this->insert(i, text, font_size, position, color);
}

void fan_2d::graphics::gui::text_renderer::set_angle(uint32_t i, f32_t angle)
{
	std::fill(angle_t::m_buffer_object.begin() + (i == 0 ? 0 : m_indices[i - 1]), angle_t::m_buffer_object.begin() + m_indices[i], angle);

	angle_t::write_data();
}

void fan_2d::graphics::gui::text_renderer::erase(uintptr_t i) {

	uint64_t begin = i == 0 ? 0 : m_indices[i - 1];
	uint64_t end = m_indices[i];
	
	sprite::erase(begin, end);
	font_size_t::erase(begin, end);
	rotation_point_t::erase(begin, end);

	m_text.erase(m_text.begin() + i);
	m_position.erase(m_position.begin() + i);
	m_new_lines.erase(m_new_lines.begin() + i);

	this->regenerate_indices();
}

void fan_2d::graphics::gui::text_renderer::erase(uintptr_t begin, uintptr_t end) {

	uint64_t begin_ = begin == 0 ? 0 : m_indices[begin - 1];
	uint64_t end_ = end == 0 ? 0 : m_indices[end - 1];

	sprite::erase(begin_, end_);
	font_size_t::erase(begin_, end_);
	rotation_point_t::erase(begin_, end_);

	m_text.erase(m_text.begin() + begin, m_text.begin() + end);
	m_position.erase(m_position.begin() + begin, m_position.begin() + end);
	m_new_lines.erase(m_new_lines.begin() + begin, m_new_lines.begin() + end);

	this->regenerate_indices();
}

void fan_2d::graphics::gui::text_renderer::clear() {
	sprite::clear();
	font_size_t::clear();

	rotation_point_t::clear();

	m_text.clear();
	m_position.clear();

	m_indices.clear();
	m_new_lines.clear();
}

void fan_2d::graphics::gui::text_renderer::set_text(uint32_t i, const fan::utf16_string& text)
{
	fan::utf16_string str = text;

	if (str.empty()) {
		str.resize(1);
	}

	auto font_size = this->get_font_size(i);
	auto position = this->get_position(i);
	auto color = this->get_text_color(i);

	this->erase(i);

	this->insert(i, str, font_size, position, color);
}

fan::color fan_2d::graphics::gui::text_renderer::get_text_color(uint32_t i, uint32_t j) const
{
	return color_t::get_value(i == 0 ? 0 : m_indices[i - 1] + j);
}

void fan_2d::graphics::gui::text_renderer::set_text_color(uint32_t i, const fan::color& color)
{
	std::fill(color_t::m_buffer_object.begin() + (i == 0 ? 0 : m_indices[i - 1]), color_t::m_buffer_object.begin() + (i == 0 ? 0 : m_indices[i - 1]) + m_text[i].size(), color);
}

void fan_2d::graphics::gui::text_renderer::set_text_color(uint32_t i, uint32_t j, const fan::color& color)
{
	color_t::set_value(m_indices[i] + j, color);
}

void fan_2d::graphics::gui::text_renderer::write_data() {
	sprite::write_data();
	font_size_t::write_data();
	rotation_point_t::write_data();
}

void fan_2d::graphics::gui::text_renderer::edit_data(uint32_t i) {

	uint32_t begin = 0;

	for (int j = 0; j < i; j++) {
		begin += m_text[j].size();
	}

	sprite::edit_data(begin, begin + m_text[i].size());
	font_size_t::edit_data(begin, begin + m_text[i].size());
	rotation_point_t::edit_data(begin, begin + m_text[i].size());
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

	sprite::edit_data(begin_, size - begin_);
	font_size_t::edit_data(begin_, size - begin_);
	rotation_point_t::edit_data(begin_, size - begin_);
}

#define get_letter_infos 				\
												\
 \
fan::vec2 letter_position; \
fan::vec2 letter_size; \
fan::vec2 letter_offset; \
 \
fan::vec2 texture_position; \
fan::vec2 texture_size; \
 \
auto found = font_info.font.find(letter); \
if (found != font_info.font.end()) { \
	letter_position = font_info.font[letter].position;						\
	letter_size = font_info.font[letter].size;								\
	letter_offset = font_info.font[letter].offset;							\
	\
	texture_position = fan::vec2(letter_position) / image->size;				\
	texture_size = fan::vec2(letter_position + letter_size) / image->size;		\
} \
\
const auto converted_font_size = convert_font_size(font_size);							\
 \
switch (letter) { \
	case '\n': { \
		\
		advance = 0;																		\
		position.y += font_info.line_height * converted_font_size;							\
		texture_position = 0;																\
		texture_size = 0; \
		 \
		break; \
	} \
	case ' ': { \
		advance += font_info.font[' '].advance;												\
		texture_position = 0;																\
		texture_size = 0;			\
			\
		break; \
	} \
	case '\0': { \
		advance += 0;												\
		texture_position = 0;																\
		texture_size = 0;			\
		letter_position = 0;						\
		letter_size = 0;								\
		letter_offset = 0;	\
		break; \
	} \
}																				\
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
	
void fan_2d::graphics::gui::text_renderer::insert_letter(uint32_t i, uint32_t j, wchar_t letter, f32_t font_size, fan::vec2& position, const fan::color& color, f32_t& advance)
{
	get_letter_infos;

	sprite::properties_t properties;

	properties.image = image;
	properties.position = position + (fan::vec2(letter_offset.x, letter_offset.y + letter_size.y) + fan::vec2(advance, 0)) * converted_font_size;
	properties.size = letter_size * converted_font_size;
	properties.rotation_point = properties.position;
	properties.texture_coordinates = texture_coordiantes;

	auto index = i == 0 ? 0 : m_indices[i - 1 >= m_indices.size() ? m_indices.size() - 1 : i - 1];

	sprite::insert(j + index, (index + j) * 6, properties);
	sprite::rectangle::color_t::m_buffer_object[index + j] = color;
	font_size_t::m_buffer_object.insert(font_size_t::m_buffer_object.begin() + index + j, font_size);
}

void fan_2d::graphics::gui::text_renderer::push_back_letter(wchar_t letter, f32_t font_size, fan::vec2& position, const fan::color& color, f32_t& advance) {

	get_letter_infos;

	sprite::properties_t properties;
	properties.image = image;

	properties.position = position + (fan::vec2(letter_offset.x, letter_offset.y + letter_size.y) + fan::vec2(advance, 0)) * converted_font_size;
	properties.size = letter_size * converted_font_size;
	properties.rotation_point = properties.position;
	properties.texture_coordinates = texture_coordiantes;

	sprite::push_back(properties);
	sprite::rectangle::set_color(sprite::rectangle::size() - 1, color);
	font_size_t::push_back(font_size);
}

#endif