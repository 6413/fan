#include <fan/graphics/renderer.hpp>

#if fan_renderer == fan_renderer_opengl

#include <fan/graphics/opengl/gl_gui.hpp>

fan::vec2 fan_2d::graphics::gui::get_resize_movement_offset(fan::window* window)
{
	return fan::cast<f32_t>(window->get_size() - window->get_previous_size());
}

void fan_2d::graphics::gui::add_resize_callback(fan::window* window, fan::vec2& position) {
	window->add_resize_callback([&] {
		position += fan_2d::graphics::gui::get_resize_movement_offset(window);
	});
}

fan_2d::graphics::gui::rectangle::rectangle(fan::camera* camera)
	: fan_2d::graphics::rectangle(camera)
{
	this->m_camera->m_window->add_resize_callback([&] {

		auto offset = this->m_camera->m_window->get_size() - this->m_camera->m_window->get_previous_size();

		bool write_after = !fan::gpu_queue;


		fan::begin_queue();

		for (int i = 0; i < this->size(); i++) {
			this->set_position(i, this->get_position(i) + fan::vec2(offset.x, 0));
		}

		if (write_after) {
			fan::end_queue();
		}

		this->release_queue();

	});
}

fan_2d::graphics::gui::circle::circle(fan::camera* camera)
	: fan_2d::graphics::circle(camera)
{
	this->m_camera->m_window->add_resize_callback([&] {

		auto offset = this->m_camera->m_window->get_size() - this->m_camera->m_window->get_previous_size();

		bool write_after = !fan::gpu_queue;


		fan::begin_queue();

		for (int i = 0; i < this->size(); i++) {
			this->set_position(i, this->get_position(i) + fan::vec2(offset.x, 0));
		}

		if (write_after) {
			fan::end_queue();
		}

		this->release_queue(true, false, false);
	});
}

fan_2d::graphics::gui::text_renderer::text_renderer()
	: sprite(fan::shader(fan_2d::graphics::shader_paths::text_renderer_vs, fan_2d::graphics::shader_paths::text_renderer_fs)) {
	image = fan_2d::graphics::load_image("fonts/arial.png");

	font_info = fan::io::file::parse_font("fonts/arial.fnt");

	font_info.font[' '] = fan::io::file::font_t({ 0, fan::vec2(fan_2d::graphics::gui::font_properties::space_width, font_info.line_height), 0, (fan::vec2::value_type)fan_2d::graphics::gui::font_properties::space_width });
	font_info.font['\n'] = fan::io::file::font_t({ 0, fan::vec2(0, font_info.line_height), 0, 0 });

	fan::bind_vao(sprite::vao_handler::m_buffer_object, [&] {
		font_size_t::initialize_buffers(fan::shader::id, location_font_size, true, 1);
		rotation_point_t::initialize_buffers(fan::shader::id, location_rotation_point, true, 2);
		});
}

fan_2d::graphics::gui::text_renderer::text_renderer(fan::camera* camera) : sprite(camera, fan::shader(fan_2d::graphics::shader_paths::text_renderer_vs, fan_2d::graphics::shader_paths::text_renderer_fs)) {

	image = fan_2d::graphics::load_image("fonts/arial.png");

	font_info = fan::io::file::parse_font("fonts/arial.fnt");

	font_info.font[' '] = fan::io::file::font_t({ 0, fan::vec2(fan_2d::graphics::gui::font_properties::space_width, font_info.line_height), 0, (fan::vec2::value_type)fan_2d::graphics::gui::font_properties::space_width });
	font_info.font['\n'] = fan::io::file::font_t({ 0, fan::vec2(0, font_info.line_height), 0, 0 });

	fan::bind_vao(sprite::vao_handler::m_buffer_object, [&] {
		font_size_t::initialize_buffers(fan::shader::id, location_font_size, true, 1);
		rotation_point_t::initialize_buffers(fan::shader::id, location_rotation_point, true, 2);
	});
}

fan::vec2 fan_2d::graphics::gui::text_renderer::push_back(const fan::fstring& text, f32_t font_size, fan::vec2 position, const fan::color& text_color) {

	m_text.emplace_back(text);

	f32_t advance = 0;

	const auto original_position = position;

	m_position.emplace_back(position);

	bool write_after = !fan::gpu_queue;

	fan::begin_queue();

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

		rotation_point_t::push_back(original_position + fan::vec2(text_size.x, text_size.y));

		if (text[i] != ' ' && text[i] != '\n') {
			advance += font_info.font[text[i]].advance;
		}
	}

	fan::end_queue();

	if (write_after) {
		this->write_data();
	}

	if (m_indices.empty()) {
		m_indices.emplace_back(text.size());
	}
	else {
		m_indices.emplace_back(m_indices[m_indices.size() - 1] + text.size());
	}

	return original_position + text_size;
}

void fan_2d::graphics::gui::text_renderer::insert(uint32_t i, const fan::fstring& text, f32_t font_size, fan::vec2 position, const fan::color& text_color)
{
	m_text.insert(m_text.begin() + i, text);

	f32_t advance = 0;

	const auto original_position = position;

	m_position.insert(m_position.begin() + i, position);

	bool write_after = !fan::gpu_queue;

	fan::begin_queue();
	 
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

		rotation_point_t::push_back(original_position + fan::vec2(text_size.x, text_size.y));

		if (text[i] != ' ' && text[i] != '\n') {
			advance += font_info.font[text[i]].advance;
		}
	}

	fan::end_queue();

	if (write_after) {
		this->write_data();
	}

	regenerate_indices();
}

fan::vec2 fan_2d::graphics::gui::text_renderer::get_position(uint32_t i) const {
	return m_position[i];
}

void fan_2d::graphics::gui::text_renderer::set_position(uint32_t i, const fan::vec2& position)
{
	bool write_after = !fan::gpu_queue;

	const uint32_t index = i == 0 ? 0 : m_indices[i - 1];

	f32_t advance = 0;

	fan::begin_queue();

	for (int j = 0; j < m_text[i].size(); j++) {

		auto letter = get_letter_info(m_text[i][j], get_font_size(i));

		const fan::vec2 new_position = position + (fan::vec2(advance, 0) + letter.size / 2 + letter.offset);
		
		fan_2d::graphics::gui::text_renderer::sprite::set_position(index + j, new_position);

		advance += letter.advance;
	}

	fan::end_queue();

	if (write_after) {
		this->release_queue(true, true, true, true, true, true);
	}
}

uint32_t fan_2d::graphics::gui::text_renderer::size() const {
	return m_text.size();
}

fan::io::file::font_t fan_2d::graphics::gui::text_renderer::get_letter_info(fan::fstring::value_type c, f32_t font_size)
{
	auto found = font_info.font.find(c);

	if (found == font_info.font.end()) {
		throw std::runtime_error("failed to find character: " + std::to_string(c));
	}

	f32_t converted_size = fan_2d::graphics::gui::text_renderer::convert_font_size(font_size);

	return fan::io::file::font_t{
		found->second.position * converted_size,	
		found->second.size * converted_size,
		found->second.offset * converted_size,
		(found->second.advance * converted_size)
	};
}

f32_t fan_2d::graphics::gui::text_renderer::get_lowest(f32_t font_size) const
{
	auto found = font_info.font.find(font_info.lowest);

	return (found->second.offset.y + found->second.size.y) * this->convert_font_size(font_size);
}

f32_t fan_2d::graphics::gui::text_renderer::get_highest(f32_t font_size) const
{
	return std::abs(font_info.font.find(font_info.highest)->second.offset.y * this->convert_font_size(font_size));
}

f32_t fan_2d::graphics::gui::text_renderer::get_highest_size(f32_t font_size) const
{
	return font_info.font.find(font_info.highest)->second.size.y * this->convert_font_size(font_size);
}

f32_t fan_2d::graphics::gui::text_renderer::get_lowest_size(f32_t font_size) const
{
	return font_info.font.find(font_info.lowest)->second.size.y * this->convert_font_size(font_size);
}

fan::vec2 fan_2d::graphics::gui::text_renderer::get_character_position(uint32_t i, uint32_t j, f32_t font_size) const
{
	fan::vec2 position = text_renderer::get_position(i);

	auto converted_size = convert_font_size(font_size);

	for (int k = 0; k < j; k++) {
		position.x += font_info.font[m_text[i][k]].advance * converted_size;
	}

	position.y = i * (font_info.line_height * converted_size);

	return position;
}

f32_t fan_2d::graphics::gui::text_renderer::get_font_size(uint_t i) const
{
	return font_size_t::get_value(i == 0 ? 0 : m_indices[i - 1]);
}

void fan_2d::graphics::gui::text_renderer::set_font_size(uint32_t i, f32_t font_size)
{
	const auto text = L"Hello";

	const auto position = 0;

	const auto color = fan::colors::white;

	this->erase(i);
	this->insert(i, text, font_size, position, color);

}

void fan_2d::graphics::gui::text_renderer::set_angle(uint32_t i, f32_t angle)
{
	std::fill(angle_t::m_buffer_object.begin() + (i == 0 ? 0 : m_indices[i - 1]), angle_t::m_buffer_object.begin() + m_indices[i], angle);

	angle_t::write_data();
}

f32_t fan_2d::graphics::gui::text_renderer::convert_font_size(f32_t font_size) {
	return font_size / font_info.size;
}

void fan_2d::graphics::gui::text_renderer::erase(uint_t i) {

	uint64_t begin = i == 0 ? 0 : m_indices[i - 1];
	uint32_t end = m_indices[i];

	bool write_after = !fan::gpu_queue;

	fan::begin_queue();

	sprite::erase(begin, end);
	font_size_t::erase(begin, end);
	rotation_point_t::erase(begin, end);

	m_text.erase(m_text.begin() + i);
	m_position.erase(m_position.begin() + i);

	fan::end_queue();

	if (write_after) {
		this->write_data();
	}

	this->regenerate_indices();
}

void fan_2d::graphics::gui::text_renderer::erase(uint_t begin, uint_t end) {

	uint64_t begin_ = begin == 0 ? 0 : m_indices[begin - 1];
	uint32_t end_ = end == 0 ? 0 : m_indices[end - 1];

	bool write_after = !fan::gpu_queue;

	fan::begin_queue();

	sprite::erase(begin_, end_);
	font_size_t::erase(begin_, end_);
	rotation_point_t::erase(begin_, end_);

	m_text.erase(m_text.begin() + begin, m_text.begin() + end);
	m_position.erase(m_position.begin() + begin, m_position.begin() + end);

	fan::end_queue();

	if (write_after) {
		this->write_data();
	}


	this->regenerate_indices();
}

void fan_2d::graphics::gui::text_renderer::clear() {
	sprite::clear();
	font_size_t::clear();

	m_text.clear();
	m_position.clear();

	m_indices.clear();
}

f32_t fan_2d::graphics::gui::text_renderer::get_line_height(f32_t font_size) const
{
	return font_info.line_height * convert_font_size(font_size);
}

fan::fstring fan_2d::graphics::gui::text_renderer::get_text(uint32_t i) const
{
	return m_text[i];
}

void fan_2d::graphics::gui::text_renderer::set_text(uint32_t i, const fan::fstring& text)
{
	auto font_size = this->get_font_size(i);
	auto position = this->get_position(i);
	auto color = this->get_text_color(i);

	bool write_after = !fan::gpu_queue;

	fan::begin_queue();

	this->erase(i);

	this->insert(i, text, font_size, position, color);

	fan::end_queue();

	if (write_after) {
		this->write_data();
	}
}

fan::color fan_2d::graphics::gui::text_renderer::get_text_color(uint32_t i, uint32_t j)
{
	return color_t::get_value(i == 0 ? 0 : m_indices[i - 1] + j);
}

void fan_2d::graphics::gui::text_renderer::set_text_color(uint32_t i, const fan::color& color)
{
	std::fill(color_t::m_buffer_object.begin() + m_indices[i], color_t::m_buffer_object.begin() + m_indices[i] + m_text[i].size(), color);

	if (!fan::gpu_queue) {
		color_t::write_data();
	}
}

void fan_2d::graphics::gui::text_renderer::set_text_color(uint32_t i, uint32_t j, const fan::color& color)
{
	color_t::set_value(m_indices[i] + j, color);
}

fan::vec2 fan_2d::graphics::gui::text_renderer::get_text_size(const fan::fstring& text, f32_t font_size)
{
	fan::vec2 length;

	f32_t current = 0;

	int new_lines = 0;

	uint32_t last_n = 0;

	for (int i = 0; i < text.size(); i++) {

		if (text[i] == '\n') {
			length.x = std::max((f32_t)length.x, current);
			length.y += font_info.line_height;
			new_lines++;
			current = 0;
			last_n = i;
			continue;
		}

		auto found = font_info.font.find(text[i]);
		if (found == font_info.font.end()) {
			throw std::runtime_error("failed to find character: " + std::to_string(text[i]));
		}

		current += found->second.advance;
		length.y = std::max((f32_t)length.y, font_info.line_height * new_lines + (f32_t)found->second.size.y);
	}

	length.x = std::max((f32_t)length.x, current);

	if (text.size()) {
		auto found = font_info.font.find(text[text.size() - 1]);
		if (found != font_info.font.end()) {
			length.x -= found->second.offset.x;
		}
	}

	length.y -= font_info.line_height * convert_font_size(font_size);

	f32_t average = 0;

	for (int i = last_n; i < text.size(); i++) {

		auto found = font_info.font.find(text[i]);
		if (found == font_info.font.end()) {
			throw std::runtime_error("failed to find character: " + std::to_string(text[i]));
		}

		average += found->second.size.y + found->second.offset.y;
	}

	average /= text.size() - last_n;

	length.y += average;

	return length * convert_font_size(font_size);
}

void fan_2d::graphics::gui::text_renderer::write_data()
{
	sprite::release_queue(true, true, true, true, true, true);
	font_size_t::write_data();
	rotation_point_t::write_data();
}

f32_t fan_2d::graphics::gui::text_renderer::get_original_font_size()
{
	return font_info.size;
}

void fan_2d::graphics::gui::text_renderer::regenerate_indices()
{
	m_indices.clear();

	for (int i = 0; i < this->size(); i++) {
		if (m_indices.empty()) {
			m_indices.emplace_back(m_text[i].size());
		}
		else {
			m_indices.emplace_back(m_indices[i - 1] + m_text[i].size());
		}
	}
}

#define get_letter_infos 																\
const fan::vec2 letter_position = font_info.font[letter].position;						\
const fan::vec2 letter_size = font_info.font[letter].size;								\
const fan::vec2 letter_offset = font_info.font[letter].offset;							\
																						\
fan::vec2 texture_position = fan::vec2(letter_position + 1) / image.size;				\
fan::vec2 texture_size = fan::vec2(letter_position + letter_size - 1) / image.size;		\
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
fan_2d::graphics::sprite_properties sp;													\
																						\
sp.texture_coordinates = {																\
	fan::vec2(texture_position.x, texture_size.y),										\
	fan::vec2(texture_size.x, texture_size.y),											\
	fan::vec2(texture_size.x, texture_position.y),										\
																						\
	fan::vec2(texture_position.x, texture_size.y),										\
	fan::vec2(texture_position.x, texture_position.y),									\
	fan::vec2(texture_size.x, texture_position.y)										\
};

void fan_2d::graphics::gui::text_renderer::insert_letter(uint32_t i, uint32_t j, fan::fstring::value_type letter, f32_t font_size, fan::vec2& position, const fan::color& color, f32_t& advance)
{
	get_letter_infos;

	auto index = i == 0 ? 0 : m_indices[i - 1 >= m_indices.size() ? m_indices.size() - 1 : i - 1];

	sprite::insert(i + j, (index + j) * 6, image.texture_id, position + (letter_offset + fan::vec2(advance, 0)) * converted_font_size + (letter_size * converted_font_size) / 2, letter_size * converted_font_size, sp);
	sprite::rectangle::color_t::m_buffer_object.insert(sprite::rectangle::color_t::m_buffer_object.begin() + index + j, color);
	font_size_t::m_buffer_object.insert(font_size_t::m_buffer_object.begin() + index + j, font_size);
}

void fan_2d::graphics::gui::text_renderer::push_back_letter(fan::fstring::value_type letter, f32_t font_size, fan::vec2& position, const fan::color& color, f32_t& advance) {

	get_letter_infos;

	sprite::push_back(image.texture_id, position + (letter_offset + fan::vec2(advance, 0) + letter_size / 2) * converted_font_size, letter_size * converted_font_size, sp);
	sprite::rectangle::set_color(rectangle::size() - 1, color);
	font_size_t::push_back(font_size);

}

//fan_2d::graphics::gui::rectangle_text_button::rectangle_text_button(fan::camera* camera, fan_2d::graphics::gui::theme theme)
//	: 
//	  class_duplicator<fan_2d::graphics::rectangle, 0>(camera),
//	  class_duplicator<fan_2d::graphics::rectangle, 1>(camera),
//	  base::mouse(*this),
//	  fan_2d::graphics::gui::text_renderer(camera), theme(theme) {
//
//	rectangle_text_button::mouse::on_hover<0>([&] (uint32_t i) {
//
//		if (m_held_button_id[i] != (uint32_t)fan::uninitialized) {
//			return;
//		}
//
//		class_duplicator<fan_2d::graphics::rectangle, 0>::set_color(i, theme.text_button.hover_color);
//
//		class_duplicator<fan_2d::graphics::rectangle, 1>::set_color(i * 4 + 0, theme.text_button.hover_outline_color);
//		class_duplicator<fan_2d::graphics::rectangle, 1>::set_color(i * 4 + 1, theme.text_button.hover_outline_color);
//		class_duplicator<fan_2d::graphics::rectangle, 1>::set_color(i * 4 + 2, theme.text_button.hover_outline_color);
//		class_duplicator<fan_2d::graphics::rectangle, 1>::set_color(i * 4 + 3, theme.text_button.hover_outline_color);
//	});
//
//	rectangle_text_button::mouse::on_exit<0>([&] (uint32_t i) {
//		if (m_held_button_id[i] != (uint32_t)fan::uninitialized) {
//			return;
//		}
//
//		class_duplicator<fan_2d::graphics::rectangle, 0>::set_color(i, theme.text_button.color);
//
//		class_duplicator<fan_2d::graphics::rectangle, 1>::set_color(i * 4 + 0, theme.text_button.outline_color);
//		class_duplicator<fan_2d::graphics::rectangle, 1>::set_color(i * 4 + 1, theme.text_button.outline_color);
//		class_duplicator<fan_2d::graphics::rectangle, 1>::set_color(i * 4 + 2, theme.text_button.outline_color);
//		class_duplicator<fan_2d::graphics::rectangle, 1>::set_color(i * 4 + 3, theme.text_button.outline_color);
//	});
//
//	rectangle_text_button::mouse::on_click<0>([&] (uint32_t i) {
//
//		class_duplicator<fan_2d::graphics::rectangle, 0>::set_color(i, theme.text_button.click_color);
//
//		class_duplicator<fan_2d::graphics::rectangle, 1>::set_color(i * 4 + 0, theme.text_button.click_outline_color);
//		class_duplicator<fan_2d::graphics::rectangle, 1>::set_color(i * 4 + 1, theme.text_button.click_outline_color);
//		class_duplicator<fan_2d::graphics::rectangle, 1>::set_color(i * 4 + 2, theme.text_button.click_outline_color);
//		class_duplicator<fan_2d::graphics::rectangle, 1>::set_color(i * 4 + 3, theme.text_button.click_outline_color);
//	});
//
//	rectangle_text_button::mouse::on_release<0>([&] (uint32_t i) {
//
//		if (mouse::m_hover_button_id[0] != (uint32_t)fan::uninitialized) {
//
//			class_duplicator<fan_2d::graphics::rectangle, 0>::set_color(i, theme.text_button.hover_color);
//
//			class_duplicator<fan_2d::graphics::rectangle, 1>::set_color(i * 4 + 0, theme.text_button.hover_outline_color);
//			class_duplicator<fan_2d::graphics::rectangle, 1>::set_color(i * 4 + 1, theme.text_button.hover_outline_color);
//			class_duplicator<fan_2d::graphics::rectangle, 1>::set_color(i * 4 + 2, theme.text_button.hover_outline_color);
//			class_duplicator<fan_2d::graphics::rectangle, 1>::set_color(i * 4 + 3, theme.text_button.hover_outline_color);
//
//			return;
//		}
//
//		class_duplicator<fan_2d::graphics::rectangle, 0>::set_color(i, theme.text_button.color);
//
//		class_duplicator<fan_2d::graphics::rectangle, 1>::set_color(i * 4 + 0, theme.text_button.outline_color);
//		class_duplicator<fan_2d::graphics::rectangle, 1>::set_color(i * 4 + 1, theme.text_button.outline_color);
//		class_duplicator<fan_2d::graphics::rectangle, 1>::set_color(i * 4 + 2, theme.text_button.outline_color);
//		class_duplicator<fan_2d::graphics::rectangle, 1>::set_color(i * 4 + 3, theme.text_button.outline_color);
//	});
//
//	rectangle_text_button::mouse::on_outside_release([&] (uint32_t i) {
//		class_duplicator<fan_2d::graphics::rectangle, 0>::set_color(i, theme.text_button.color);
//
//		class_duplicator<fan_2d::graphics::rectangle, 1>::set_color(i * 4 + 0, theme.text_button.outline_color);
//		class_duplicator<fan_2d::graphics::rectangle, 1>::set_color(i * 4 + 1, theme.text_button.outline_color);
//		class_duplicator<fan_2d::graphics::rectangle, 1>::set_color(i * 4 + 2, theme.text_button.outline_color);
//		class_duplicator<fan_2d::graphics::rectangle, 1>::set_color(i * 4 + 3, theme.text_button.outline_color);
//	});
//
////void on_release(std::function<void(uint32_t i)> function, uint16_t key = fan::mouse_left) { \
////base::mouse::on_release<1>(object, function, key); \
////	} \
////		void on_hover(std::function<void(uint32_t i)> function) { \
////		base::mouse::on_hover<1>(object, function); \
////	} \
////		\
////		void on_exit(std::function<void(uint32_t i)> function) { \
////		base::mouse::on_exit<1>(object, function); \
////
////	this->on_hover([&](uint32_t i) {
////		b.set_color(i, b.get_color(i) + fan_2d::graphics::gui::themes::deep_blue().button_hover_color_offset);
////		fan::print("enterted button id:", i);
////	});
////
////	b.on_exit([&](uint32_t i) {
////		b.set_color(i, b.get_color(i) - fan_2d::graphics::gui::themes::deep_blue().button_hover_color_offset);
////		fan::print("exited button id:", i);
////	});
//}
//
//void fan_2d::graphics::gui::rectangle_text_button::push_back(const rectangle_button_properties& properties)
//{
//	m_properties.emplace_back(properties);
//
//	switch (properties.text_position) {
//		case text_position_e::left:
//		{
//			fan_2d::graphics::gui::text_renderer::push_back(
//				properties.text,
//				fan::vec2(properties.position.x + theme.text_button.outline_thickness, properties.position.y + properties.border_size.y * 0.5),
//				theme.text_button.text_color,
//				properties.font_size
//			);
//
//			break;
//		}
//		case text_position_e::middle:
//		{
//			fan_2d::graphics::gui::text_renderer::push_back(
//				properties.text,
//				fan::vec2(properties.position.x + properties.border_size.x * 0.5, properties.position.y + properties.border_size.y * 0.5),
//				theme.text_button.text_color,
//				properties.font_size
//			);
//
//			break;
//		}
//	}
//
//	class_duplicator<fan_2d::graphics::rectangle, 0>::push_back(properties.position, get_button_size(m_properties.size() - 1), theme.text_button.color, properties.fan::gpu_queue);
//
//	auto corners = class_duplicator<fan_2d::graphics::rectangle, 0>::get_corners(m_properties.size() - 1);
//
//	const f32_t t = theme.text_button.outline_thickness;
//
//	corners[1].x -= t;
//	corners[3].x -= t;
//
//	corners[2].y -= t;
//	corners[3].y -= t;
//
//	class_duplicator<fan_2d::graphics::rectangle, 1>::push_back(corners[0], fan::vec2(corners[1].x - corners[0].x + t, t), theme.text_button.outline_color, true);
//	class_duplicator<fan_2d::graphics::rectangle, 1>::push_back(corners[1], fan::vec2(t, corners[3].y - corners[1].y + t), theme.text_button.outline_color, true);
//	class_duplicator<fan_2d::graphics::rectangle, 1>::push_back(corners[2], fan::vec2(corners[3].x - corners[2].x + t, t), theme.text_button.outline_color, true);
//	class_duplicator<fan_2d::graphics::rectangle, 1>::push_back(corners[0], fan::vec2(t, corners[2].y - corners[0].y + t), theme.text_button.outline_color, true);
//	
//	class_duplicator<fan_2d::graphics::rectangle, 1>::release_queue(!properties.fan::gpu_queue, !properties.fan::gpu_queue);
//}
//
//void fan_2d::graphics::gui::rectangle_text_button::draw(uint32_t begin, uint32_t end) const
//{
//	fan_2d::graphics::draw([&] {
//		class_duplicator<fan_2d::graphics::rectangle, 0>::draw(begin, end);
//		class_duplicator<fan_2d::graphics::rectangle, 1>::draw(begin, end);
//		fan_2d::graphics::gui::text_renderer::draw();
//	});
//}
//
//bool fan_2d::graphics::gui::rectangle_text_button::inside(uint_t i, const fan::vec2& position) const
//{
//	return fan::class_duplicator<fan_2d::graphics::rectangle, 0>::inside(i, position);
//}
//
//fan::camera* fan_2d::graphics::gui::rectangle_text_button::get_camera()
//{
//	return fan::class_duplicator<fan_2d::graphics::rectangle, 0>::m_camera;
//}
//
//uint_t fan_2d::graphics::gui::rectangle_text_button::size() const
//{
//	return fan::class_duplicator<fan_2d::graphics::rectangle, 0>::size();
//}
//
//fan_2d::graphics::gui::sprite_text_button::sprite_text_button(fan::camera* camera, const std::string& path)
//	:  fan_2d::graphics::sprite(camera, path), base::mouse(*this), fan_2d::graphics::gui::text_renderer(camera) { }
//
//fan::camera* fan_2d::graphics::gui::sprite_text_button::get_camera()
//{
//	return fan_2d::graphics::sprite::m_camera;
//}
//
//void fan_2d::graphics::gui::sprite_text_button::push_back(const sprite_button_properties& properties)
//{
//	m_properties.emplace_back(properties);
//
//	switch (properties.text_position) {
//		case text_position_e::left:
//		{
//			fan_2d::graphics::gui::text_renderer::push_back(
//				properties.text,
//				fan::vec2(properties.position.x, properties.position.y + properties.border_size.y * 0.5),
//				fan_2d::graphics::gui::defaults::text_color,
//				properties.font_size
//			);
//
//			break;
//		}
//		case text_position_e::middle:
//		{
//			fan_2d::graphics::gui::text_renderer::push_back(
//				properties.text,
//				fan::vec2(properties.position.x + properties.border_size.x * 0.5, properties.position.y + properties.border_size.y * 0.5),
//				fan_2d::graphics::gui::defaults::text_color,
//				properties.font_size
//			);
//
//			break;
//		}
//	}
//
//	if (properties.texture_id != (uint32_t)fan::uninitialized) {
//		fan_2d::graphics::sprite::push_back(properties.texture_id, properties.position, get_button_size(m_properties.size() - 1), properties.fan::gpu_queue);
//	}
//	else {
//		fan_2d::graphics::sprite::push_back(properties.position, get_button_size(m_properties.size() - 1), properties.fan::gpu_queue);
//	}
//}
//
//void fan_2d::graphics::gui::sprite_text_button::draw(uint32_t begin, uint32_t end) const
//{
//	fan_2d::graphics::draw([&] {
//		fan_2d::graphics::sprite::draw(begin, end);
//		fan_2d::graphics::gui::text_renderer::draw();
//	});
//}
//
//fan_2d::graphics::gui::checkbox::checkbox(fan::camera* camera, fan_2d::graphics::gui::theme theme)
//	: fan_2d::graphics::rectangle(camera), 
//	 fan_2d::graphics::line(camera), 
//	 fan_2d::graphics::gui::text_renderer(camera),
//	 fan_2d::graphics::gui::base::mouse(*this),
//	 m_theme(theme) {
//
//	fan_2d::graphics::gui::base::mouse::on_hover<0>([&] (uint32_t i) {
//		fan_2d::graphics::rectangle::set_color(i, m_theme.checkbox.hover_color);
//	});
//
//	fan_2d::graphics::gui::base::mouse::on_exit<0>([&] (uint32_t i) {
//		fan_2d::graphics::rectangle::set_color(i, m_theme.checkbox.color);
//	});
//
//	fan_2d::graphics::gui::base::mouse::on_click<0>([&](uint32_t i) {
//		fan_2d::graphics::rectangle::set_color(i, m_theme.checkbox.click_color);
//	});
//
//	fan_2d::graphics::gui::base::mouse::on_release<0>([&](uint32_t i) {
//		fan_2d::graphics::rectangle::set_color(i, m_theme.checkbox.hover_color);
//
//		m_visible[i] = !m_visible[i];
//
//		if (m_visible[i]) {
//			m_on_check(i);
//		}
//		else {
//			if (m_on_check) {
//				m_on_uncheck(i);
//			}
//		}
//	});
//
//	fan_2d::graphics::gui::base::mouse::on_outside_release<0>([&](uint32_t i) {
//		fan_2d::graphics::rectangle::set_color(i, m_theme.checkbox.color);
//	});
//}
//
//void fan_2d::graphics::gui::checkbox::push_back(const checkbox_property& property)
//{
//	m_properties.emplace_back(property);
//
//	m_visible.emplace_back(property.checked);
//
//	f32_t text_middle_height = get_average_text_height(property.text, property.font_size);
//	fan_2d::graphics::rectangle::push_back(property.position, text_middle_height * property.box_size_multiplier, m_theme.checkbox.color, true);
//
//	fan_2d::graphics::line::push_back(property.position, property.position + text_middle_height * property.box_size_multiplier, m_theme.checkbox.check_color, true);
//	fan_2d::graphics::line::push_back(property.position + fan::vec2(text_middle_height * property.box_size_multiplier, 0), property.position + fan::vec2(0, text_middle_height * property.box_size_multiplier), m_theme.checkbox.check_color, true);
//
//	fan_2d::graphics::gui::text_renderer::push_back(property.text, property.position + fan::vec2(property.font_size * property.box_size_multiplier, (text_middle_height * property.box_size_multiplier) / 2 - text_middle_height / 2  - (get_magic_size() * convert_font_size(property.font_size) - text_middle_height)), m_theme.checkbox.text_color, property.font_size);
//
//	fan_2d::graphics::rectangle::release_queue(!property.fan::gpu_queue, !property.fan::gpu_queue);
//	fan_2d::graphics::line::release_queue(!property.fan::gpu_queue, !property.fan::gpu_queue);
//}
//
//void fan_2d::graphics::gui::checkbox::draw() const
//{
//	fan_2d::graphics::draw([&] {
//		
//		uint8_t previous_thickness = fan_2d::graphics::global_vars::line_thickness;
//		
//		for (int i = 0; i < fan_2d::graphics::line::size() / 2; i++) {
//			fan_2d::graphics::rectangle::draw(i);
//
//			if (m_visible[i]) {
//				fan_2d::graphics::global_vars::line_thickness = m_properties[i].line_thickness;
//				fan_2d::graphics::line::draw(i * 2);
//				fan_2d::graphics::line::draw(i * 2 + 1);
//			}
//		}
//
//		fan_2d::graphics::global_vars::line_thickness = previous_thickness;
//
//		fan_2d::graphics::gui::text_renderer::draw();
//	});
//}
//
//void fan_2d::graphics::gui::checkbox::on_check(std::function<void(uint32_t i)> function)
//{
//	m_on_check = function;
//}
//
//void fan_2d::graphics::gui::checkbox::on_uncheck(std::function<void(uint32_t i)> function)
//{
//	m_on_uncheck = function;
//}
//
//fan::camera* fan_2d::graphics::gui::checkbox::get_camera()
//{
//	return fan_2d::graphics::rectangle::m_camera;
//}

#endif