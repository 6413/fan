#include <fan/graphics/gui.hpp>

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

		for (int i = 0; i < this->size(); i++) {
			this->set_position(i, this->get_position(i) + fan::vec2(offset.x, 0), true);
		}

		this->release_queue(true, false);
	});
}

fan_2d::graphics::gui::circle::circle(fan::camera* camera)
	: fan_2d::graphics::circle(camera)
{
	this->m_camera->m_window->add_resize_callback([&] {

		auto offset = this->m_camera->m_window->get_size() - this->m_camera->m_window->get_previous_size();

		for (int i = 0; i < this->size(); i++) {
			this->set_position(i, this->get_position(i) + fan::vec2(offset.x, 0), true);
		}

		this->release_queue(true, false, false);
	});
}

fan_2d::graphics::gui::text_renderer::text_renderer(fan::camera* camera) 
	: text_color_t(), outline_color_t(), m_camera(camera), m_shader(fan_2d::graphics::shader_paths::text_renderer_vs, fan_2d::graphics::shader_paths::text_renderer_fs)
{

	auto info = fan_2d::graphics::load_image("fonts/arial.png");

	this->m_original_image_size = info.size;
	texture_handler::m_buffer_object = info.texture_id;

	initialize_buffers();

	font_info = fan::io::file::parse_font("fonts/arial.fnt");

	font_info.font[' '] = fan::io::file::font_t({ 0, fan::vec2(font_properties::space_width, font_info.line_height), 0, (fan::vec2ui::value_type)font_properties::space_width });
	font_info.font['\n'] = fan::io::file::font_t({ 0, fan::vec2(0, font_info.line_height), 0, 0 });

}

fan_2d::graphics::gui::text_renderer::text_renderer(fan::camera* camera, const fan::fstring& text, const fan::vec2& position, const fan::color& text_color, f32_t font_size, const fan::color& outline_color, bool queue)
	: fan_2d::graphics::gui::text_renderer::text_renderer(camera)
{
	this->push_back(text, position, text_color, font_size, outline_color, queue);
}

fan_2d::graphics::gui::text_renderer::text_renderer(const text_renderer& tr)
	: text_renderer::text_color_t(tr), 
	text_renderer::outline_color_t(tr),
	fan::texture_handler<>(tr),
	fan::vao_handler<>(tr),
	text_renderer::font_sizes_ssbo_t(tr),
	text_renderer::vertex_vbo_t(tr),
	text_renderer::texture_vbo_t(tr),
	m_camera(tr.m_camera)
{

	font_info = tr.font_info;
	m_shader = tr.m_shader;
	m_original_image_size = tr.m_original_image_size;
	m_text = tr.m_text;
	m_position = tr.m_position;
	m_font_size = tr.m_font_size;
	m_vertices = tr.m_vertices;
	m_texture_coordinates = tr.m_texture_coordinates;

	initialize_buffers();
}

fan_2d::graphics::gui::text_renderer::text_renderer(text_renderer&& tr)
	: text_renderer::text_color_t(std::move(tr)), 
	text_renderer::outline_color_t(std::move(tr)),
	fan::texture_handler<>(std::move(tr)),
	fan::vao_handler<>(std::move(tr)),
	text_renderer::font_sizes_ssbo_t(std::move(tr)),
	vertex_vbo_t(std::move(tr)),
	texture_vbo_t(std::move(tr)),
	m_camera(tr.m_camera)
{

	font_info = std::move(tr.font_info);
	m_shader = std::move(tr.m_shader);
	m_original_image_size = std::move(tr.m_original_image_size);
	m_text = std::move(tr.m_text);
	m_position = std::move(tr.m_position);
	m_font_size = std::move(tr.m_font_size);
	m_vertices = std::move(tr.m_vertices);
	m_texture_coordinates = std::move(tr.m_texture_coordinates);
}

fan_2d::graphics::gui::text_renderer& fan_2d::graphics::gui::text_renderer::operator=(const text_renderer& tr)
{
	m_camera = tr.m_camera;
	m_shader = tr.m_shader;

	text_renderer::text_color_t::operator=(tr);
	text_renderer::outline_color_t::operator=(tr);
	fan::texture_handler<>::operator=(tr);
	fan::vao_handler<>::operator=(tr);
	text_renderer::font_sizes_ssbo_t::operator=(tr);
	vertex_vbo_t::operator=(tr);
	texture_vbo_t::operator=(tr);

	font_info = tr.font_info;
	m_original_image_size = tr.m_original_image_size;
	m_text = tr.m_text;
	m_position = tr.m_position;
	m_font_size = tr.m_font_size;
	m_vertices = tr.m_vertices;
	m_texture_coordinates = tr.m_texture_coordinates;

	initialize_buffers();

	return *this;
}

fan_2d::graphics::gui::text_renderer& fan_2d::graphics::gui::text_renderer::operator=(text_renderer&& tr)
{
	m_camera = tr.m_camera;
	text_renderer::text_color_t::operator=(std::move(tr));
	text_renderer::outline_color_t::operator=(std::move(tr));
	fan::texture_handler<>::operator=(std::move(tr));
	fan::vao_handler<>::operator=(std::move(tr));
	text_renderer::font_sizes_ssbo_t::operator=(std::move(tr));
	vertex_vbo_t::operator=(std::move(tr));
	texture_vbo_t::operator=(std::move(tr));

	font_info = std::move(tr.font_info);
	m_shader = std::move(tr.m_shader);
	m_original_image_size = std::move(tr.m_original_image_size);
	m_text = std::move(tr.m_text);
	m_position = std::move(tr.m_position);
	m_font_size = std::move(tr.m_font_size);
	m_vertices = std::move(tr.m_vertices);
	m_texture_coordinates = std::move(tr.m_texture_coordinates);

	return *this;
}

fan::vec2 fan_2d::graphics::gui::text_renderer::get_position(uint_t i) const {
	return this->m_position[i];
}

void fan_2d::graphics::gui::text_renderer::set_position(uint_t i, const fan::vec2& position, bool queue)
{

	if (this->get_position(i) == position) {
		return;
	}
	auto current_position = this->get_position(i);

	for (uint_t j = 0; j < this->m_vertices[i].size(); j++) {
		this->m_vertices[i][j] -= current_position;
		this->m_vertices[i][j] += position;
	}

	this->m_position[i] = position;

	if (!queue) {
		this->write_vertices();
	}
}

f32_t fan_2d::graphics::gui::text_renderer::get_font_size(uint_t i) const
{
	return m_font_size[i][0];
}

void fan_2d::graphics::gui::text_renderer::set_font_size(uint_t i, f32_t font_size, bool queue) {
	if (this->get_font_size(i) == font_size) {
		return;
	}

	std::fill(m_font_size[i].begin(), m_font_size[i].end(), font_size);

	this->load_characters(i, this->get_position(i), m_text[i], true, false);

	if (!queue) {
		this->write_data();
	}
}

void fan_2d::graphics::gui::text_renderer::set_text(uint_t i, const fan::fstring& text, bool queue) {

	if (m_text.size() && m_text[i] == text) {
		return;
	}

	const auto position = this->get_position(i);

	const auto text_color = text_color_t::get_color(i)[0];
	const auto outline_color = outline_color_t::get_color(i)[0];
	const auto font_size = this->m_font_size[i][0];

	this->erase(i, true);

	this->insert(i, text, position, text_color, font_size, outline_color, true);

	if (!queue) {
		this->write_data();
	}
}

void fan_2d::graphics::gui::text_renderer::set_text_color(uint_t i, const fan::color& color, bool queue)
{
	if (text_color_t::m_color[i][0] == color) {
		return;
	}
	std::fill(text_color_t::m_color[i].begin(), text_color_t::m_color[i].end(), color);
	if (!queue) {
		text_color_t::edit_data(i);
	}
}

void fan_2d::graphics::gui::text_renderer::set_text_color(uint_t i, uint_t j, const fan::color& color, bool queue)
{
	//std::fill(&, &text_color_t::m_color[i][j] + 6, color);

	text_color_t::m_color[i][j] = color;

	if (!queue) {
		text_color_t::edit_data(i);
	}
}

void fan_2d::graphics::gui::text_renderer::set_outline_color(uint_t i, const fan::color& color, bool queue)
{
	if (outline_color_t::m_color[i][0] == color) {
		return;
	}
	std::fill(outline_color_t::m_color[i].begin(), outline_color_t::m_color[i].end(), color);
	if (!queue) {
		outline_color_t::edit_data(i);
	}
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
		(fan::vec2ui::value_type)(found->second.advance * converted_size)
	};
}

fan::vec2 fan_2d::graphics::gui::text_renderer::get_text_size(uint_t i) const
{
	return get_text_size(get_text(i), get_font_size(i));
}

fan::vec2 fan_2d::graphics::gui::text_renderer::get_text_size(const fan::fstring& text, f32_t font_size)
{
	fan::vec2 length;

	f32_t current = 0;

	int new_lines = 0;

	for (const auto& i : text) {

		if (i == '\n') {
			length.x = std::max((f32_t)length.x, current);
			length.y += font_info.line_height;
			new_lines++;
			current = 0;
			continue;
		}

		auto found = font_info.font.find(i);
		if (found == font_info.font.end()) {
			throw std::runtime_error("failed to find character: " + std::to_string(i));
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

	return length * convert_font_size(font_size);
}

f32_t fan_2d::graphics::gui::text_renderer::get_average_text_height(const fan::fstring& text, f32_t font_size) const
{
	f32_t height = 0;

	for (const auto& i : text) {

		auto found = font_info.font.find(i);
		if (found == font_info.font.end()) {
			throw std::runtime_error("failed to find character: " + std::to_string(i));
		}

		height += (f32_t)found->second.size.y;
	}

	return (height / text.size()) * convert_font_size(font_size);
}

f32_t fan_2d::graphics::gui::text_renderer::get_longest_text() const
{

	f32_t longest = -fan::math::inf;

	for (uint_t i = 0; i < this->size(); i++) {
		longest = std::max(longest, (f32_t)get_text_size(i).x);
	}

	return longest;
}

f32_t fan_2d::graphics::gui::text_renderer::get_highest_text() const
{

	f32_t highest = -fan::math::inf;

	for (uint_t i = 0; i < this->size(); i++) {
		highest = std::max(highest, (f32_t)get_text_size(i).y);
	}

	return highest;
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

fan::color fan_2d::graphics::gui::text_renderer::get_color(uint_t i, uint_t j) const
{
	return text_color_t::m_color[i][j];
}

fan::fstring fan_2d::graphics::gui::text_renderer::get_text(uint_t i) const
{
	return this->m_text[i];
}

f32_t fan_2d::graphics::gui::text_renderer::convert_font_size(f32_t font_size) 
{
	return font_size / font_info.size;
}

void fan_2d::graphics::gui::text_renderer::push_back(const fan::fstring& text, const fan::vec2& position, const fan::color& text_color, f32_t font_size, const fan::color& outline_color, bool queue) {

	m_text.emplace_back(text);
	this->m_position.emplace_back(position);

	m_font_size.resize(m_font_size.size() + 1);
	text_color_t::m_color.resize(text_color_t::m_color.size() + 1);
	outline_color_t::m_color.resize(outline_color_t::m_color.size() + 1);

	m_vertices.resize(m_vertices.size() + 1);
	m_texture_coordinates.resize(m_texture_coordinates.size() + 1);

	m_font_size[m_font_size.size() - 1].insert(m_font_size[m_font_size.size() - 1].end(), text.empty() ? 1 : text.size(), font_size);

	text_color_t::m_color[text_color_t::m_color.size() - 1].insert(text_color_t::m_color[text_color_t::m_color.size() - 1].end(), text.empty() ? 1 : text.size(), text_color);
	outline_color_t::m_color[outline_color_t::m_color.size() - 1].insert(outline_color_t::m_color[outline_color_t::m_color.size() - 1].end(), text.empty() ? 1 : text.size(), outline_color);

	if (text.empty()) {
		return;
	}

	this->load_characters(m_font_size.size() - 1, position, text, false, false);

	if (!queue) {
		this->write_data();
	}
}

void fan_2d::graphics::gui::text_renderer::insert(uint_t i, const fan::fstring& text, const fan::vec2& position, const fan::color& text_color, f32_t font_size, const fan::color& outline_color, bool queue)
{
	m_font_size.insert(m_font_size.begin() + i, std::vector<f32_t>(text.empty() ? 1 : text.size(), font_size));

	m_text.insert(m_text.begin() + i, text);
	this->m_position.insert(m_position.begin() + i, position);

	text_color_t::m_color.insert(text_color_t::m_color.begin() + i, std::vector<fan::color>(text.empty() ? 1 : text.size(), text_color));

	outline_color_t::m_color.insert(outline_color_t::m_color.begin() + i, std::vector<fan::color>(text.empty() ? 1 : text.size(), outline_color));

	m_vertices.reserve(m_vertices.size() + i);
	m_vertices.insert(m_vertices.begin() + i, std::vector<fan::vec2>(text.size() * 6));

	m_texture_coordinates.reserve(m_texture_coordinates.size() + i);
	m_texture_coordinates.insert(m_texture_coordinates.begin() + i, std::vector<fan::vec2>(text.size() * 6));

	if (text.empty()) {
		if (!queue) {
			this->write_data();
		}
		return;
	}

	this->load_characters(i, position, text, true, false);

	if (!queue) {
		this->write_data();
	}
}

void fan_2d::graphics::gui::text_renderer::draw() const {
	const fan::vec2i window_size = m_camera->m_window->get_size();
	fan::mat4 projection = fan::math::ortho<fan::mat4>(0, window_size.x, window_size.y, 0);

	this->m_shader.use();
	this->m_shader.set_mat4("projection", projection);

	this->m_shader.set_int("texture_sampler", 0);
	this->m_shader.set_float("original_font_size", font_info.size);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture_handler::m_buffer_object);

	fan::bind_vao(vao_handler::m_buffer_object, [&] {
		glDisable(GL_DEPTH_TEST);
		glDrawArrays(GL_TRIANGLES, 0, fan::vector_size(m_vertices));
		glEnable(GL_DEPTH_TEST);
	});
}

void fan_2d::graphics::gui::text_renderer::erase(uint_t i, bool queue) {
	if (i >= m_vertices.size()) {
		return;
	}

	m_vertices.erase(m_vertices.begin() + i);
	m_texture_coordinates.erase(m_texture_coordinates.begin() + i);
	m_font_size.erase(m_font_size.begin() + i);
	m_position.erase(m_position.begin() + i);
	text_color_t::m_color.erase(text_color_t::m_color.begin() + i);
	outline_color_t::m_color.erase(outline_color_t::m_color.begin() + i);
	m_text.erase(m_text.begin() + i);

	if (!queue) {
		this->write_data();
	}
}

void fan_2d::graphics::gui::text_renderer::erase(uint_t begin, uint_t end, bool queue) {
	m_vertices.erase(m_vertices.begin() + begin, m_vertices.begin() + end);
	m_texture_coordinates.erase(m_texture_coordinates.begin() + begin, m_texture_coordinates.begin() + end);
	m_font_size.erase(m_font_size.begin() + begin, m_font_size.begin() + end);
	m_position.erase(m_position.begin() + begin, m_position.begin() + end);
	text_color_t::m_color.erase(text_color_t::m_color.begin() + begin, text_color_t::m_color.begin() + end);
	outline_color_t::m_color.erase(outline_color_t::m_color.begin() + begin, outline_color_t::m_color.begin() + end);
	m_text.erase(m_text.begin() + begin, m_text.begin() + end);

	if (!queue) {
		this->write_data();
	}
}

uint_t fan_2d::graphics::gui::text_renderer::size() const
{
	return this->m_text.size();
}

void fan_2d::graphics::gui::text_renderer::write_data() {

	this->write_vertices();
	this->write_texture_coordinates();
	this->write_font_sizes();

	text_color_t::write_data();
	outline_color_t::write_data();
}

void fan_2d::graphics::gui::text_renderer::release_queue(bool vertices, bool texture_coordinates, bool font_sizes)
{
	if (vertices) {
		write_vertices();
	}
	if (texture_coordinates) {
		write_texture_coordinates();
	}
	if (font_sizes) {
		write_font_sizes();
	}
}

int64_t fan_2d::graphics::gui::text_renderer::get_new_lines(uint32_t i)
{
	int64_t new_lines = 0;
	const wchar_t * p = m_text[i].data();
	for ( int i = 0; i < m_text[i].size(); i++ ) {
		if (p[i] == '\n') {
			new_lines++;
		}
	}

	return new_lines;
}

int64_t fan_2d::graphics::gui::text_renderer::get_new_lines(const fan::fstring& str)
{
	int64_t new_lines = 0;
	const wchar_t * p = str.data();
	for ( int i = 0; i < str.size(); i++ ) {
		if (p[i] == '\n') {
			new_lines++;
		}
	}

	return new_lines;
}

fan::vec2 fan_2d::graphics::gui::text_renderer::get_character_position(uint32_t i, uint32_t j)
{
	fan::vec2 position = text_renderer::get_position(i);

	for (int k = 0; k < j; k++) {
		position.x += font_info.font[m_text[i][k]].advance;
	}

	position.y = i * font_info.line_height;

	return position * (get_font_size(i) / (font_info.size));
}

f32_t fan_2d::graphics::gui::text_renderer::get_line_height(f32_t font_size) const
{
	return font_info.line_height * convert_font_size(font_size);
}

f32_t fan_2d::graphics::gui::text_renderer::get_original_font_size()
{
	return font_info.size;
}

void fan_2d::graphics::gui::text_renderer::initialize_buffers()
{
	glBindVertexArray(vao_handler::m_buffer_object);

	text_color_t::initialize_buffers(m_shader.m_id, text_color_location_name, false);
	//outline_color_t::initialize_buffers(false);

	vertex_vbo_t::initialize_buffers(nullptr, 0, false, fan::vec2::size(), m_shader.m_id, vertex_location_name);
	texture_vbo_t::initialize_buffers(nullptr, 0, false, fan::vec2::size(), m_shader.m_id, texture_coordinates_location_name);

	font_sizes_ssbo_t::initialize_buffers(nullptr, 0, false, 1, m_shader.m_id, font_sizes_location_name);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

//
//std::vector<fan::vec2> fan_2d::graphics::gui::text_renderer::get_vertices(uint_t i) {
//	uint_t offset = this->get_character_offset(i, true) * 6;
//	return std::vector<fan::vec2>(m_vertices.begin() + offset, m_vertices.begin() + offset + m_text[i].size() * 6);
//}
//
//std::vector<fan::vec2> fan_2d::graphics::gui::text_renderer::get_texture_coordinates(uint_t i) {
//	uint_t offset = this->get_character_offset(i, true) * 6;
//	return std::vector<fan::vec2>(m_texture_coordinates.begin() + offset, m_texture_coordinates.begin() + offset + m_text[i].size() * 6);
//}
//
//std::vector<f32_t> fan_2d::graphics::gui::text_renderer::get_font_size(uint_t i) {
//	uint_t offset = this->get_character_offset(i, true);
//	return std::vector<f32_t>(m_font_size.begin() + offset, m_font_size.begin() + offset + m_text[i].size());
//}

void fan_2d::graphics::gui::text_renderer::load_characters(uint_t i, fan::vec2 position, const fan::fstring& text, bool edit, bool insert) {
	const f32_t converted_font_size(convert_font_size(m_font_size[i][0]));

	int advance = 0;

	uint_t iletter = 0;

	for (const auto& letter : text) {
		if (letter == '\n') {
			advance = 0;
			position.y += font_info.line_height * converted_font_size;
			continue;
		}
		else if (letter == ' ') {
			advance += font_info.font[' '].advance;
			continue;
		}
		if (insert) {
			this->insert_letter_data(i, letter, position, advance, converted_font_size);
		}
		else if (edit) {
			this->edit_letter_data(i, iletter, letter, position, advance, converted_font_size);
			iletter += 6;
		}
		else {
			this->write_letter_data(i, letter, position, advance, converted_font_size);
		}
	}
}

void fan_2d::graphics::gui::text_renderer::edit_letter_data(uint_t i, uint_t j, const fan::fstring::value_type letter, const fan::vec2& position, int& advance, f32_t converted_font_size) {

	const fan::vec2 letter_position = font_info.font[letter].position;
	const fan::vec2 letter_size = font_info.font[letter].size;
	const fan::vec2 letter_offset = font_info.font[letter].offset;

	const fan::vec2 texture_offset = letter_position / this->m_original_image_size;
	const fan::vec2 texture_width = (letter_position + letter_size) / this->m_original_image_size;

	m_vertices[i][j + 0] = position + fan::vec2(1, 0) * letter_size * converted_font_size + (fan::vec2(advance, 0) + letter_offset) * converted_font_size;
	m_vertices[i][j + 1] = position + fan::vec2(0, 0) * letter_size * converted_font_size + (fan::vec2(advance, 0) + letter_offset) * converted_font_size;
	m_vertices[i][j + 2] = position + fan::vec2(0, 1) * letter_size * converted_font_size + (fan::vec2(advance, 0) + letter_offset) * converted_font_size;												   
	m_vertices[i][j + 3] = position + fan::vec2(0, 1) * letter_size * converted_font_size + (fan::vec2(advance, 0) + letter_offset) * converted_font_size;
	m_vertices[i][j + 4] = position + fan::vec2(1, 1) * letter_size * converted_font_size + (fan::vec2(advance, 0) + letter_offset) * converted_font_size;
	m_vertices[i][j + 5] = position + fan::vec2(1, 0) * letter_size * converted_font_size + (fan::vec2(advance, 0) + letter_offset) * converted_font_size;

	m_texture_coordinates[i][j + 0] = fan::vec2(texture_width.x , texture_offset.y);
	m_texture_coordinates[i][j + 1] = fan::vec2(texture_offset.x, texture_offset.y);
	m_texture_coordinates[i][j + 2] = fan::vec2(texture_offset.x, texture_width.y);
	m_texture_coordinates[i][j + 3] = fan::vec2(texture_offset.x, texture_width.y);
	m_texture_coordinates[i][j + 4] = fan::vec2(texture_width.x , texture_width.y);
	m_texture_coordinates[i][j + 5] = fan::vec2(texture_width.x , texture_offset.y);

	advance += font_info.font[letter].advance;
}

void fan_2d::graphics::gui::text_renderer::insert_letter_data(uint_t i, const fan::fstring::value_type letter, const fan::vec2& position, int& advance, f32_t converted_font_size)
{
	const fan::vec2 letter_position = font_info.font[letter].position;
	const fan::vec2 letter_size = font_info.font[letter].size;
	const fan::vec2 letter_offset = font_info.font[letter].offset;

	const fan::vec2 texture_offset = letter_position / this->m_original_image_size;
	const fan::vec2 texture_width = (letter_position + letter_size) / this->m_original_image_size;

	m_vertices[i].insert(m_vertices[i].begin() + i    , position + fan::vec2(1, 0) * letter_size * converted_font_size + (fan::vec2(advance, 0) + letter_offset) * converted_font_size);
	m_vertices[i].insert(m_vertices[i].begin() + i + 1, position + fan::vec2(0, 0) * letter_size * converted_font_size + (fan::vec2(advance, 0) + letter_offset) * converted_font_size);
	m_vertices[i].insert(m_vertices[i].begin() + i + 2, position + fan::vec2(0, 1) * letter_size * converted_font_size + (fan::vec2(advance, 0) + letter_offset) * converted_font_size);																						   
	m_vertices[i].insert(m_vertices[i].begin() + i + 3, position + fan::vec2(0, 1) * letter_size * converted_font_size + (fan::vec2(advance, 0) + letter_offset) * converted_font_size);
	m_vertices[i].insert(m_vertices[i].begin() + i + 4, position + fan::vec2(1, 1) * letter_size * converted_font_size + (fan::vec2(advance, 0) + letter_offset) * converted_font_size);
	m_vertices[i].insert(m_vertices[i].begin() + i + 5, position + fan::vec2(1, 0) * letter_size * converted_font_size + (fan::vec2(advance, 0) + letter_offset) * converted_font_size);

	m_texture_coordinates[i].insert(m_texture_coordinates[i].begin() + i    , fan::vec2(texture_width.x , texture_offset.y));
	m_texture_coordinates[i].insert(m_texture_coordinates[i].begin() + i + 1, fan::vec2(texture_offset.x, texture_offset.y));
	m_texture_coordinates[i].insert(m_texture_coordinates[i].begin() + i + 2, fan::vec2(texture_offset.x, texture_width.y));
	m_texture_coordinates[i].insert(m_texture_coordinates[i].begin() + i + 3, fan::vec2(texture_offset.x, texture_width.y));
	m_texture_coordinates[i].insert(m_texture_coordinates[i].begin() + i + 4, fan::vec2(texture_width.x , texture_width.y));
	m_texture_coordinates[i].insert(m_texture_coordinates[i].begin() + i + 5, fan::vec2(texture_width.x , texture_offset.y));

	advance += font_info.font[letter].advance;
}

void fan_2d::graphics::gui::text_renderer::write_letter_data(uint_t i, const fan::fstring::value_type letter, const fan::vec2& position, int& advance, f32_t converted_font_size) {
	const fan::vec2 letter_position = font_info.font[letter].position;
	const fan::vec2 letter_size = font_info.font[letter].size;
	const fan::vec2 letter_offset = font_info.font[letter].offset;

	const fan::vec2 texture_offset = fan::vec2(letter_position + 1) / this->m_original_image_size;
	const fan::vec2 texture_width = fan::vec2(letter_position + letter_size - 1) / this->m_original_image_size;

	m_vertices[i].emplace_back((position + fan::vec2(1, 0) * letter_size * converted_font_size + (fan::vec2(advance, 0) + letter_offset) * converted_font_size));
	m_vertices[i].emplace_back((position + fan::vec2(0, 0) * letter_size * converted_font_size + (fan::vec2(advance, 0) + letter_offset) * converted_font_size));
	m_vertices[i].emplace_back((position + fan::vec2(0, 1) * letter_size * converted_font_size + (fan::vec2(advance, 0) + letter_offset) * converted_font_size));																						   
	m_vertices[i].emplace_back((position + fan::vec2(0, 1) * letter_size * converted_font_size + (fan::vec2(advance, 0) + letter_offset) * converted_font_size));
	m_vertices[i].emplace_back((position + fan::vec2(1, 1) * letter_size * converted_font_size + (fan::vec2(advance, 0) + letter_offset) * converted_font_size));
	m_vertices[i].emplace_back((position + fan::vec2(1, 0) * letter_size * converted_font_size + (fan::vec2(advance, 0) + letter_offset) * converted_font_size));

	m_texture_coordinates[i].emplace_back(fan::vec2(texture_width.x , texture_offset.y));
	m_texture_coordinates[i].emplace_back(fan::vec2(texture_offset.x, texture_offset.y));
	m_texture_coordinates[i].emplace_back(fan::vec2(texture_offset.x, texture_width.y));
	m_texture_coordinates[i].emplace_back(fan::vec2(texture_offset.x, texture_width.y));
	m_texture_coordinates[i].emplace_back(fan::vec2(texture_width.x , texture_width.y));
	m_texture_coordinates[i].emplace_back(fan::vec2(texture_width.x , texture_offset.y));

	advance += font_info.font[letter].advance;
}

void fan_2d::graphics::gui::text_renderer::write_vertices()
{
	std::vector<fan::vec2> vertices;

	for (uint_t i = 0; i < m_vertices.size(); i++) {
		vertices.insert(vertices.end(), m_vertices[i].begin(), m_vertices[i].end());
	}

	vertex_vbo_t::write_data(vertices.data(), sizeof(fan::vec2) * fan::vector_size(vertices));
}

void fan_2d::graphics::gui::text_renderer::write_texture_coordinates()
{
	std::vector<fan::vec2> texture_coordinates;

	for (uint_t i = 0; i < m_texture_coordinates.size(); i++) {
		texture_coordinates.insert(texture_coordinates.end(), m_texture_coordinates[i].begin(), m_texture_coordinates[i].end());
	}

	texture_vbo_t::write_data(texture_coordinates.data(), sizeof(fan::vec2) * fan::vector_size(texture_coordinates));
}

void fan_2d::graphics::gui::text_renderer::write_font_sizes()
{
	std::vector<f32_t> font_sizes;

	for (uint_t i = 0; i < m_font_size.size(); i++) {
		for (int j = 0; j < 6; j++) {
			font_sizes.insert(font_sizes.end(), m_font_size[i].begin(), m_font_size[i].end());
		}
	}

	font_sizes_ssbo_t::bind_gl_storage_buffer([&] {
		glBufferData(font_sizes_ssbo_t::gl_buffer, sizeof(font_sizes[0]) * fan::vector_size(font_sizes), font_sizes.data(), GL_STATIC_DRAW);
	});
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
//	class_duplicator<fan_2d::graphics::rectangle, 0>::push_back(properties.position, get_button_size(m_properties.size() - 1), theme.text_button.color, properties.queue);
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
//	class_duplicator<fan_2d::graphics::rectangle, 1>::release_queue(!properties.queue, !properties.queue);
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
//		fan_2d::graphics::sprite::push_back(properties.texture_id, properties.position, get_button_size(m_properties.size() - 1), properties.queue);
//	}
//	else {
//		fan_2d::graphics::sprite::push_back(properties.position, get_button_size(m_properties.size() - 1), properties.queue);
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
//	fan_2d::graphics::rectangle::release_queue(!property.queue, !property.queue);
//	fan_2d::graphics::line::release_queue(!property.queue, !property.queue);
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