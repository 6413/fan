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

fan_2d::graphics::gui::text_renderer::text_renderer(fan::camera* camera) : 
	m_camera(camera), sprite(camera, true) {

	m_shader->set_vertex(
    #include <fan/graphics/glsl/opengl/2D/text.vs>
  );

	m_shader->set_fragment(
    #include <fan/graphics/glsl/opengl/2D/text.fs>
  );

	m_shader->compile();

	sprite::initialize();

	if (!font_image) {
		font_image = fan_2d::graphics::load_image(camera->m_window, "fonts/comic.webp");
	}

	font = fan::font::parse_font("fonts/comic_metrics.txt");

	/*font_info.font[' '] = fan::font::font_t({ 0, fan::vec2(fan_2d::graphics::gui::font_properties::space_width, font_info.line_height), 0, (fan::vec2::value_type)fan_2d::graphics::gui::font_properties::space_width });
	font_info.font['\n'] = fan::font::font_t({ 0, fan::vec2(0, font_info.line_height), 0, 0 });*/

	fan::bind_vao(sprite::vao_handler::m_buffer_object, [&] {
		font_size_t::initialize_buffers(m_shader->id, location_font_size, false, 1);
		rotation_point_t::initialize_buffers(m_shader->id, location_rotation_point, false, 2);
		outline_color_t::initialize_buffers(m_shader->id, location_outline_color, false, 4);
		outline_size_t::initialize_buffers(m_shader->id, location_outline_size, false, 1);
	});
}

fan_2d::graphics::gui::text_renderer::~text_renderer() {
}

void fan_2d::graphics::gui::text_renderer::enable_draw()
{
	if (m_draw_index == -1 || m_camera->m_window->m_draw_queue[m_draw_index].first != this) {
		m_draw_index = m_camera->m_window->push_draw_call(this, [&] {
			this->draw();
		});
	}
	else {
		m_camera->m_window->edit_draw_call(m_draw_index, this, [&] {
			this->draw();
		});
	}
}

void fan_2d::graphics::gui::text_renderer::disable_draw()
{
	if (m_draw_index == -1) {
		return;
	}

	m_camera->m_window->erase_draw_call(m_draw_index);
	m_draw_index = -1;
}

void fan_2d::graphics::gui::text_renderer::draw(uint32_t begin, uint32_t end) {
	
	if (!fan_2d::graphics::sprite::size()) {
		return;
	}

	auto begin_ = begin == 0 || begin == fan::uninitialized ? 0 : m_indices[begin - 1];
	auto end_ = end == 0 ? 0 : end == fan::uninitialized ? m_indices[m_indices.size() - 1] : m_indices[end - 1];

	fan_2d::graphics::sprite::draw(begin_, end_);
}

void fan_2d::graphics::gui::text_renderer::push_back(properties_t properties) {

	if (properties.text.empty()) {
		throw std::runtime_error("text cannot be empty");
	}

	m_position.emplace_back(properties.position);
	m_text.emplace_back(properties.text);

	m_indices.emplace_back(m_indices.empty() ? properties.text.size() : m_indices[m_indices.size() - 1] + properties.text.size());

	fan::vec2 text_size = get_text_size(properties.text, properties.font_size);

	f32_t left = properties.position.x - text_size.x / 2;

	properties.position.y += font.size * convert_font_size(properties.font_size) / 2;

	f32_t average_height = 0;

	m_new_lines.resize(m_new_lines.size() + 1);

	for (int i = 0; i < properties.text.size(); i++) {

		if (properties.text[i] == '\n') {
			m_new_lines[m_new_lines.size() - 1]++;
			left = properties.position.x - text_size.x / 2;
			properties.position.y += get_line_height(properties.font_size);
		}

		auto letter = get_letter_info(properties.text[i], properties.font_size);

		properties_t letter_properties;
		letter_properties.position = fan::vec2(left - letter.metrics.offset.x, properties.position.y);
		letter_properties.font_size = properties.font_size;
		letter_properties.text_color = properties.text_color;
		letter_properties.outline_color = properties.outline_color;
		letter_properties.outline_size = properties.outline_size;

		push_letter(properties.text[i], letter_properties);

		left += letter.metrics.advance;
		average_height += letter.metrics.size.y;

		rotation_point_t::m_buffer_object.insert(rotation_point_t::m_buffer_object.end(), 6, (properties.position - fan::vec2(0, text_size.y / 2 - (average_height / properties.text.size()) / 2)));
	}

}

void fan_2d::graphics::gui::text_renderer::insert(uint32_t i, properties_t properties)
{
	if (properties.text.empty()) {
		throw std::runtime_error("text cannot be empty");
	}

	m_position.insert(m_position.begin() + i, properties.position);
	m_text.insert(m_text.begin() + i, properties.text);

	fan::vec2 text_size = get_text_size(properties.text, properties.font_size);

	f32_t left = properties.position.x - text_size.x / 2;

	properties.position.y += font.size * convert_font_size(properties.font_size) / 2;

	f32_t average_height = 0;

	m_new_lines.insert(m_new_lines.begin() + i, 0);

	for (int j = 0; j < properties.text.size(); j++) {

		if (properties.text[j] == '\n') {
			m_new_lines[i]++;
			left = properties.position.x - text_size.x / 2;
			properties.position.y += font.line_height;
		}

		auto letter = get_letter_info(properties.text[j], properties.font_size);

		properties_t letter_properties;
		letter_properties.position = fan::vec2(left - letter.metrics.offset.x, properties.position.y);
		letter_properties.font_size = properties.font_size;
		letter_properties.text_color = properties.text_color;
		letter_properties.outline_color = properties.outline_color;
		letter_properties.outline_size = properties.outline_size;

		insert_letter(get_index(i) + j, properties.text[j], letter_properties);

		left += letter.metrics.advance;
		average_height += letter.metrics.size.y;
		rotation_point_t::m_buffer_object.insert(rotation_point_t::m_buffer_object.begin() + i * 6 + j, 6, properties.position - fan::vec2(0, text_size.y / 2 - (average_height / properties.text.size()) / 2));
	}

	regenerate_indices();
}

void fan_2d::graphics::gui::text_renderer::set_position(uint32_t i, const fan::vec2& position)
{
	const uint32_t index = i == 0 ? 0 : m_indices[i - 1];

	const fan::vec2 offset = position - get_position(i);

	m_position[i] = position;

	for (int j = 0; j < m_text[i].size(); j++) {

		sprite::set_position(index + j, sprite::get_position(index + j) + offset);

	}

	rotation_point_t::set_value(i, position);
}

uint32_t fan_2d::graphics::gui::text_renderer::size() const {
	return m_text.size();
}

f32_t fan_2d::graphics::gui::text_renderer::get_font_size(uintptr_t i) const
{
	if (font_size_t::m_buffer_object.size() <= i) { // for empty strings
		return 0;
	}
	return font_size_t::get_value(i == 0 ? 0 : m_indices[i - 1] * 6);
}

void fan_2d::graphics::gui::text_renderer::set_font_size(uint32_t i, f32_t font_size)
{
	const auto text = get_text(i);

	const auto position = get_position(i);

	const auto color = get_text_color(i);

	const auto outline_color = get_outline_color(i);
	const auto outline_size = get_outline_size(i);

	this->erase(i);

	text_renderer::properties_t properties;
	properties.text = text;
	properties.font_size = font_size;
	properties.position = position;
	properties.text_color = color;
	properties.outline_color = outline_color;
	properties.outline_size = outline_size;

	this->insert(i, properties);
}

void fan_2d::graphics::gui::text_renderer::set_angle(uint32_t i, f32_t angle)
{
	std::fill(angle_t::m_buffer_object.begin() + (i == 0 ? 0 : m_indices[i - 1] * 6), angle_t::m_buffer_object.begin() + m_indices[i] * 6, angle);

	m_queue_helper.edit(i, i + 1, [&] {
		this->edit_data(m_queue_helper.m_min_edit, m_queue_helper.m_max_edit);
	});
}

void fan_2d::graphics::gui::text_renderer::set_rotation_point(uint32_t i, const fan::vec2 & rotation_point)
{
	std::fill(rotation_point_t::m_buffer_object.begin() + (i == 0 ? 0 : m_indices[i - 1] * 6), rotation_point_t::m_buffer_object.begin() + m_indices[i] * 6, rotation_point);

	m_queue_helper.edit(i, i + 1, [&] {
		this->edit_data(m_queue_helper.m_min_edit, m_queue_helper.m_max_edit);
	});
}

fan::color fan_2d::graphics::gui::text_renderer::get_outline_color(uint32_t i) const
{
	return outline_color_t::get_value(i == 0 ? 0 : m_indices[i - 1] * 6);
}

void fan_2d::graphics::gui::text_renderer::set_outline_color(uint32_t i, const fan::color& outline_color)
{
	std::fill(outline_color_t::m_buffer_object.begin() + (i == 0 ? 0 : m_indices[i - 1] * 6), outline_color_t::m_buffer_object.begin() + m_indices[i] * 6, outline_color);

	m_queue_helper.edit(i, i + 1, [&] {
		this->edit_data(m_queue_helper.m_min_edit, m_queue_helper.m_max_edit);
	});
}

f32_t fan_2d::graphics::gui::text_renderer::get_outline_size(uint32_t i) const
{
	return outline_size_t::get_value(i == 0 ? 0 : m_indices[i - 1] * 6);
}

void fan_2d::graphics::gui::text_renderer::set_outline_size(uint32_t i, f32_t outline_size)
{
	std::fill(outline_size_t::m_buffer_object.begin() + (i == 0 ? 0 : m_indices[i - 1] * 6), outline_size_t::m_buffer_object.begin() + m_indices[i] * 6, outline_size);

	m_queue_helper.edit(i, i + 1, [&] {
		this->edit_data(m_queue_helper.m_min_edit, m_queue_helper.m_max_edit);
	});
}

void fan_2d::graphics::gui::text_renderer::erase(uintptr_t i) {

	uint64_t begin = get_index(i);
	uint64_t end = get_index(i + 1);
	
	font_size_t::erase(begin * 6, end * 6);
	rotation_point_t::erase(begin * 6, end * 6);
	outline_color_t::erase(begin * 6, end * 6);
	outline_size_t::erase(begin * 6, end * 6);

	bool write_ = m_queue_helper.m_write;

	sprite::erase(begin, end);

	m_text.erase(m_text.begin() + i);
	m_position.erase(m_position.begin() + i);
	m_new_lines.erase(m_new_lines.begin() + i);

	this->regenerate_indices();

	if (!write_) {
		m_camera->m_window->edit_write_call(m_queue_helper.m_write_index, this, [&] {
			this->write_data();
		});
	}
}

void fan_2d::graphics::gui::text_renderer::erase(uintptr_t begin, uintptr_t end) {

	uint64_t begin_ = begin == 0 ? 0 : m_indices[begin - 1];
	uint64_t end_ = end == 0 ? 0 : m_indices[end - 1];

	bool write_ = m_queue_helper.m_write;

	sprite::erase(begin_, end_);
	font_size_t::erase(begin_, end_);
	rotation_point_t::erase(begin_, end_);
	outline_color_t::erase(begin_, end_);
	outline_size_t::erase(begin_, end_);

	m_text.erase(m_text.begin() + begin, m_text.begin() + end);
	m_position.erase(m_position.begin() + begin, m_position.begin() + end);
	m_new_lines.erase(m_new_lines.begin() + begin, m_new_lines.begin() + end);

	this->regenerate_indices();

	if (!write_) {
		m_camera->m_window->edit_write_call(m_queue_helper.m_write_index, this, [&] {
			this->write_data();
		});
	}
}

void fan_2d::graphics::gui::text_renderer::clear() {

	font_size_t::clear();
	rotation_point_t::clear();
	outline_color_t::clear();
	outline_size_t::clear();

	bool write_ = m_queue_helper.m_write;

	sprite::clear();

	m_text.clear();
	m_position.clear();

	m_indices.clear();
	m_new_lines.clear();

	if (!write_) {
		m_camera->m_window->edit_write_call(m_queue_helper.m_write_index, this, [&] {
			this->write_data();
		});
	}
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

	const auto outline_color = get_outline_color(i);
	const auto outline_size = get_outline_size(i);

	this->erase(i);

	text_renderer::properties_t properties;

	properties.text = text;
	properties.font_size = font_size;
	properties.position = position;
	properties.text_color = color;
	properties.outline_color = outline_color;
	properties.outline_size = outline_size;

	this->insert(i, properties);
}

fan::color fan_2d::graphics::gui::text_renderer::get_text_color(uint32_t i, uint32_t j) const
{
	return color_t::get_value(get_index(i) * 6 + j);
}

void fan_2d::graphics::gui::text_renderer::set_text_color(uint32_t i, const fan::color& color)
{
	std::fill(color_t::m_buffer_object.begin() + (i == 0 ? 0 : m_indices[i - 1]) * 6, color_t::m_buffer_object.begin() + m_indices[i] * 6, color);

	m_queue_helper.edit(i, i + 1, [&] {
		this->edit_data(m_queue_helper.m_min_edit, m_queue_helper.m_max_edit);
	});
}

void fan_2d::graphics::gui::text_renderer::set_text_color(uint32_t i, uint32_t j, const fan::color& color)
{
	sprite::set_color(get_index(i) + j, color);
}

void fan_2d::graphics::gui::text_renderer::write_data() {
	sprite::write_data();
	font_size_t::write_data();
	rotation_point_t::write_data();
	outline_color_t::write_data();
	outline_size_t::write_data();
}

void fan_2d::graphics::gui::text_renderer::edit_data(uint32_t i) {

	uint32_t begin = 0;

	for (int j = 0; j < i; j++) {
		begin += m_text[j].size();
	}

	sprite::edit_data(begin, begin + m_text[i].size());
	font_size_t::edit_data(begin, begin + m_text[i].size());
	rotation_point_t::edit_data(begin, begin + m_text[i].size());
	outline_color_t::edit_data(begin, begin + m_text[i].size());
	outline_size_t::edit_data(begin, begin + m_text[i].size());
}

void fan_2d::graphics::gui::text_renderer::edit_data(uint32_t begin, uint32_t end) {

	uint32_t begin_ = 0;
	uint32_t end_ = 0;

	for (int i = 0; i < begin; i++) {
		begin_ += m_text[i].size();
	}

	end_ = begin_;

	for (int i = begin_; i < end; i++) {
		end_ += m_text[i].size();
	}

	sprite::edit_data(begin_, end_);
	font_size_t::edit_data(begin_ * 6, end_ *6);
	rotation_point_t::edit_data(begin, end);
	outline_color_t::edit_data(begin_ * 6, end_ * 6);
	outline_size_t::edit_data(begin_ * 6, end_ * 6);
}

#define get_letter_infos 				\
												\
 \
fan::vec2 letter_position; \
fan::vec2 letter_size; \
fan::vec2 letter_offset; \
 \
fan::vec2 src; \
fan::vec2 dst; \
 \
auto found = font.font.find(letter); \
if (found != font.font.end()) { \
	letter_position = found->second.glyph.position;						\
	letter_size = found->second.glyph.size;								\
	letter_offset = found->second.metrics.offset;							\
	\
	src = fan::vec2(letter_position) / image->size;				\
	dst = fan::vec2(letter_position + letter_size) / image->size;		\
} \
\
\
const auto converted_font_size = convert_font_size(font_size);							\
 \
switch (letter) { \
	case '\n': { \
		\
		advance = 0;																		\
		position.y += font.line_height * converted_font_size;							\
		src = 0;																\
		dst = 0; \
		 \
		break; \
	} \
	case ' ': { \
		advance += font.font[' '].metrics.advance;												\
		src = 0;																\
		dst = 0;			\
			\
		break; \
	} \
	case '\0': { \
		advance += 0;												\
		src = 0;																\
		dst = 0;			\
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
	fan::vec2(src.x, src.y),										\
	fan::vec2(dst.x, src.y),											\
	fan::vec2(dst.x, dst.y),										\
																						\
	fan::vec2(dst.x, dst.y),										\
	fan::vec2(src.x, dst.y),									\
	fan::vec2(src.x, src.y)										\
};		

/*

vec2 rectangle_vertices[] = vec2[](
vec2(-0.5, -0.5),
vec2(0.5, -0.5),
vec2(0.5, 0.5),

vec2(0.5, 0.5),
vec2(-0.5, 0.5),
vec2(-0.5, -0.5)
);

vec2 rectangle_vertices[] = vec2[](
vec2(0, 0),
vec2(1, 0),
vec2(1, -1),

vec2(0, 0),
vec2(0, -1),
vec2(1, -1)
);
*/
	
void fan_2d::graphics::gui::text_renderer::insert_letter(uint32_t i, uint32_t j, wchar_t letter, f32_t font_size, fan::vec2& position, const fan::color& color, f32_t& advance)
{
	assert(0);

	//get_letter_infos;

	//sprite::properties_t properties;

	//properties.image = font_image;
	//properties.position = position + (fan::vec2(letter_offset.x, -letter_offset.y) + fan::vec2(advance, 0)) * converted_font_size;
	//properties.size = letter_size * converted_font_size;
	//properties.texture_coordinates = texture_coordiantes;

	//auto index = i == 0 ? 0 : m_indices[i - 1 >= m_indices.size() ? m_indices.size() - 1 : i - 1];

	//sprite::insert(j + index, (index + j) * 6, properties);
	//sprite::rectangle::color_t::m_buffer_object.insert(
	//	sprite::rectangle::color_t::m_buffer_object.begin() + index + j,
	//	6,
	//	color
	//);
	//font_size_t::m_buffer_object.insert(font_size_t::m_buffer_object.begin() + index + j, font_size);
}

#endif