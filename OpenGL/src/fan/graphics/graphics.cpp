#include <fan/graphics/graphics.hpp>

#include <functional>
#include <numeric>

#include <fan/fast_noise/fast_noise.hpp>
#include <fan/physics/collision/rectangle.hpp>
#include <fan/physics/collision/circle.hpp>

#include <fan/graphics/image.hpp>

void fan::depth_test(bool value)
{
	if (value) {
		glEnable(GL_DEPTH_TEST);
	}
	else {
		glDisable(GL_DEPTH_TEST);
	}
}

void fan::print_opengl_version()
{
	fan::print("OpenGL version:", glGetString(GL_VERSION));
}

uint32_t load_texture(const std::string_view path, const std::string& directory, bool flip_image) {

	std::string file_name = std::string(directory + (directory.empty() ? "" : "/") + path.data());

	auto info = fan_2d::graphics::load_image(file_name, flip_image);

	glBindTexture(GL_TEXTURE_2D, info.texture_id);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, fan_2d::graphics::image_load_properties::visual_output);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, fan_2d::graphics::image_load_properties::visual_output);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glGenerateMipmap(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, 0);

	return info.texture_id;
}

fan::mat4 fan_2d::graphics::get_projection(const fan::vec2i& window_size) {

	return fan::math::ortho<fan::mat4>((f32_t)window_size.x * 0.5, (f32_t)window_size.x + (f32_t)window_size.x * 0.5, (f32_t)window_size.y + (f32_t)window_size.y * 0.5, (f32_t)window_size.y * 0.5, 0.1f, 1000.0f);
}

fan::mat4 fan_2d::graphics::get_view_translation(const fan::vec2i& window_size, const fan::mat4& view)
{
	return fan::math::translate(view, fan::vec3((f_t)window_size.x * 0.5, (f_t)window_size.y * 0.5, -700.0f));
}

fan_2d::graphics::image_info fan_2d::graphics::load_image(const std::string& path, bool flip_image)
{
	std::ifstream file(path);

	if (!file.good()) {
		fan::print("sprite loading error: File path does not exist for", path.c_str());
		exit(1);
	}

	fan_2d::graphics::image_info info;

	glGenTextures(1, &info.texture_id);

	return fan_2d::graphics::load_image(info.texture_id, path, flip_image);
}

fan_2d::graphics::image_info fan_2d::graphics::load_image(uint32_t texture_id, const std::string& path, bool flip_image)
{
	fan_2d::graphics::image_info info;

	info.texture_id = texture_id;

	glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, info.size.begin());
	glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, info.size.begin() + 1);

	glBindTexture(GL_TEXTURE_2D, info.texture_id);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, fan_2d::graphics::image_load_properties::visual_output);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, fan_2d::graphics::image_load_properties::visual_output);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, fan_2d::graphics::image_load_properties::filter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, fan_2d::graphics::image_load_properties::filter);
	glGenerateMipmap(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, 0);

	glBindTexture(GL_TEXTURE_2D, info.texture_id);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, fan_2d::graphics::image_load_properties::visual_output);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, fan_2d::graphics::image_load_properties::visual_output);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, fan_2d::graphics::image_load_properties::filter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, fan_2d::graphics::image_load_properties::filter);

	auto image_data = fan::image_loader::load_image(path);

	info.size = image_data.size;

	uint_t internal_format = 0, format = 0, type = 0;

	switch (image_data.format) {
		case AVPixelFormat::AV_PIX_FMT_BGR24:
		{
			glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
			internal_format = GL_RGB;
			format = GL_BGR_EXT;
			type = GL_UNSIGNED_BYTE;
			break;
		}
		case AVPixelFormat::AV_PIX_FMT_RGB24:
		{
			//	glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
			internal_format = GL_RGB;
			format = GL_RGB;
			type = GL_UNSIGNED_BYTE;
			break;
		}
		case AVPixelFormat::AV_PIX_FMT_RGBA:
		{
			//glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
			internal_format = GL_RGBA;
			format = GL_RGBA;
			type = GL_UNSIGNED_BYTE;
			break;
		}
		//case AVPixelFormat::AV_PIX_FMT_BGRA:
	}

	glTexImage2D(GL_TEXTURE_2D, 0, internal_format, info.size.x, info.size.y, 0, format, type, image_data.data[0]);

	glGenerateMipmap(GL_TEXTURE_2D);

//	av_freep((void*)&image_data.data);

	glBindTexture(GL_TEXTURE_2D, 0);

	return info;
}

fan_2d::graphics::image_info fan_2d::graphics::load_image(unsigned char* pixels, const fan::vec2i& size)
{
	unsigned int texture_id = 0;

	glGenTextures(1, &texture_id);
	glBindTexture(GL_TEXTURE_2D, texture_id);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, fan_2d::graphics::image_load_properties::visual_output);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, fan_2d::graphics::image_load_properties::visual_output);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, image_load_properties::filter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, image_load_properties::filter);

	if (pixels != nullptr) {
		glTexImage2D(GL_TEXTURE_2D, 0, image_load_properties::internal_format, size.x, size.y, 0, image_load_properties::format, image_load_properties::type, pixels);
		glGenerateMipmap(GL_TEXTURE_2D);
	}

	glBindTexture(GL_TEXTURE_2D, 0);
	return { size, texture_id };
}

fan_2d::graphics::vertice_vector::vertice_vector(fan::camera* camera, uint_t index_restart)
	: basic_vertice_vector(camera, fan::shader(fan_2d::graphics::shader_paths::shape_vector_vs, fan_2d::graphics::shader_paths::shape_vector_fs)), m_index_restart(index_restart), m_offset(0)
{
	this->initialize_buffers();
}

fan_2d::graphics::vertice_vector::vertice_vector(fan::camera* camera, const fan::vec2& position, const fan::color& color, uint_t index_restart)
	: basic_vertice_vector(camera, fan::shader(fan_2d::graphics::shader_paths::shape_vector_vs, fan_2d::graphics::shader_paths::shape_vector_fs), position, color), m_index_restart(index_restart), m_offset(0)
{
	this->initialize_buffers();
}

fan_2d::graphics::vertice_vector::vertice_vector(const vertice_vector& vector)
	: fan::basic_vertice_vector<fan::vec2>(vector), ebo_handler(vector) {

	this->m_indices = vector.m_indices;
	this->m_offset = vector.m_offset;
	this->m_index_restart = vector.m_index_restart;

	this->initialize_buffers();
}

fan_2d::graphics::vertice_vector::vertice_vector(vertice_vector&& vector) noexcept
	: fan::basic_vertice_vector<fan::vec2>(std::move(vector)), ebo_handler(std::move(vector)) { 

	this->m_indices = std::move(vector.m_indices);
	this->m_offset = vector.m_offset;
	this->m_index_restart = vector.m_index_restart;
}

fan_2d::graphics::vertice_vector& fan_2d::graphics::vertice_vector::operator=(const vertice_vector& vector)
{
	this->m_indices = vector.m_indices;
	this->m_offset = vector.m_offset;
	this->m_index_restart = vector.m_index_restart;

	ebo_handler::operator=(vector);
	basic_vertice_vector::operator=(vector);

	this->initialize_buffers();

	return *this;
}

fan_2d::graphics::vertice_vector& fan_2d::graphics::vertice_vector::operator=(vertice_vector&& vector) noexcept
{
	this->m_indices = std::move(vector.m_indices);
	this->m_offset = vector.m_offset;
	this->m_index_restart = vector.m_index_restart;

	ebo_handler::operator=(std::move(vector));
	basic_vertice_vector::operator=(std::move(vector));

	return *this;
}


void fan_2d::graphics::vertice_vector::release_queue(bool position, bool color, bool indices)
{
	if (position) {
		fan::basic_vertice_vector<fan::vec2>::basic_shape_position::write_data();
	}
	if (color) {
		fan::basic_vertice_vector<fan::vec2>::basic_shape_color_vector::write_data();
	}
	if (indices) {
		this->write_data();
	}
}

void fan_2d::graphics::vertice_vector::push_back(const fan::vec2& position, const fan::color& color, bool queue)
{
	fan::basic_vertice_vector<fan::vec2>::basic_shape_position::basic_push_back(position, true);
	fan::basic_vertice_vector<fan::vec2>::basic_shape_color_vector::basic_push_back(color, true);

	m_indices.emplace_back(m_offset);
	m_offset++;

	//if (!(m_offset % this->m_index_restart) && !m_indices.empty()) {
	//	m_indices.emplace_back(UINT32_MAX);
	//}
	if (!queue) {
		this->write_data();
	}
}

void fan_2d::graphics::vertice_vector::reserve(uint_t size)
{
	vertice_vector::basic_vertice_vector::reserve(size);
}

void fan_2d::graphics::vertice_vector::resize(uint_t size, const fan::color& color)
{
	vertice_vector::basic_vertice_vector::resize(size, color);
}

void fan_2d::graphics::vertice_vector::draw(uint32_t mode, uint32_t single_draw_amount, uint32_t begin, uint32_t end, bool texture) const
{
	fan::mat4 projection(1);
	projection = fan_2d::graphics::get_projection(m_camera->m_window->get_size());

	fan::mat4 view(1);
	view = m_camera->get_view_matrix(fan_2d::graphics::get_view_translation(m_camera->m_window->get_size(), view));

	this->m_shader.use();

	this->m_shader.set_mat4("projection", projection);
	this->m_shader.set_mat4("view", view);

	this->m_shader.set_int("enable_texture", texture);

	//glEnable(GL_PRIMITIVE_RESTART);
	//	glPrimitiveRestartIndex(this->m_index_restart);

	
	
	//	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	fan_2d::graphics::draw([&] {
		vertice_vector::basic_vertice_vector::basic_draw(begin, end, m_indices, mode, size(), m_index_restart, single_draw_amount);
	});


	//glDisable(GL_PRIMITIVE_RESTART);
}

void fan_2d::graphics::vertice_vector::erase(uint_t i, bool queue)
{
	m_indices.erase(m_indices.begin() + i);

	fan::basic_vertice_vector<fan::vec2>::erase(i, queue);

	for (uint_t j = i; j < this->size(); j++) {
		m_indices[j]--;
	}

	m_offset--; // ? FIX

	if (!queue) {
		this->write_data();
	}
}

void fan_2d::graphics::vertice_vector::erase(uint_t begin, uint_t end, bool queue)
{
	if (!begin && end == m_indices.size()) {
		this->m_indices.clear();
		m_offset = 0;
	}
	else {
		m_indices.erase(m_indices.begin() + begin, m_indices.begin() + end);
		for (uint_t i = begin; i < m_indices.size(); i++) { // bad performance
			m_indices[i] -= end - begin;
		}
		m_offset -= end - begin;
	}

	fan::basic_vertice_vector<fan::vec2>::erase(begin, end, true);

	if (!queue) {
		this->write_data();
	}
}

void fan_2d::graphics::vertice_vector::initialize_buffers()
{
	fan::bind_vao(vao_handler::m_buffer_object, [&] {
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_handler::m_buffer_object);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(m_indices[0]) * m_indices.size(), m_indices.data(), GL_STATIC_DRAW);
		fan::basic_vertice_vector<fan::vec2>::basic_shape_position::initialize_buffers(m_shader.m_id, position_location_name, false);
		fan::basic_vertice_vector<fan::vec2>::basic_shape_color_vector::initialize_buffers(m_shader.m_id, color_location_name, false);
	});
}

void fan_2d::graphics::vertice_vector::write_data()
{
	fan::basic_vertice_vector<fan::vec2>::basic_shape_position::write_data();
	fan::basic_vertice_vector<fan::vec2>::basic_shape_color_vector::write_data();

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_handler::m_buffer_object);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(m_indices[0]) * m_indices.size(), m_indices.data(), GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

fan_2d::graphics::line::line(fan::camera* camera) : fan_2d::graphics::vertice_vector(camera) {}

fan_2d::graphics::line::line(const line& line_)
	: fan_2d::graphics::vertice_vector(line_) {}

fan_2d::graphics::line::line(line&& line_) noexcept : fan_2d::graphics::vertice_vector(std::move(line_)) {}

fan_2d::graphics::line& fan_2d::graphics::line::operator=(const line& line_)
{
	fan_2d::graphics::vertice_vector::operator=(line_);

	return *this;
}

fan_2d::graphics::line& fan_2d::graphics::line::operator=(line&& line_) noexcept
{
	fan_2d::graphics::vertice_vector::operator=(std::move(line_));

	return *this;
}

fan::mat2 fan_2d::graphics::line::get_line(uint_t i) const
{
	return fan::mat2(
		fan_2d::graphics::vertice_vector::get_position(i * 2), 
		fan_2d::graphics::vertice_vector::get_position(i * 2 + 1)
	);
}

void fan_2d::graphics::line::set_line(uint_t i, const fan::vec2& start, const fan::vec2& end, bool queue)
{
	fan_2d::graphics::vertice_vector::set_position(i * 2, start, true);
	fan_2d::graphics::vertice_vector::set_position(i * 2 + 1, end, true);

	if (!queue) {
		fan_2d::graphics::vertice_vector::write_data();
	}

}

void fan_2d::graphics::line::push_back(const fan::vec2& start, const fan::vec2& end, const fan::color& color, bool queue)
{
	fan_2d::graphics::vertice_vector::push_back(start, color, true);
	fan_2d::graphics::vertice_vector::push_back(end, color, true);

	if (!queue) {
		fan_2d::graphics::vertice_vector::write_data();
	}
}

void fan_2d::graphics::line::reserve(uint_t size)
{
	line::vertice_vector::reserve(size);
}

void fan_2d::graphics::line::resize(uint_t size, const fan::color& color)
{
	line::vertice_vector::resize(size * 2, color);
	for (uint_t i = 0; i < size * 2; i++) {
		m_indices.emplace_back(m_offset);
		m_offset++;
	}
	write_data();
}

void fan_2d::graphics::line::draw(uint_t i) const
{
	fan_2d::graphics::line::set_thickness(fan_2d::graphics::global_vars::line_thickness);

	fan_2d::graphics::vertice_vector::draw(GL_LINES, 2, i);
}

void fan_2d::graphics::line::erase(uint_t i, bool queue)
{
	fan_2d::graphics::vertice_vector::erase(i * 2, queue);
	fan_2d::graphics::vertice_vector::erase(i * 2, queue);
}

// ?
void fan_2d::graphics::line::erase(uint_t begin, uint_t end, bool queue)
{
	fan_2d::graphics::vertice_vector::erase(begin * 2, end * 2, queue);
}

const fan::color fan_2d::graphics::line::get_color(uint_t i) const
{
	return fan_2d::graphics::vertice_vector::get_color(i * 2);
}

void fan_2d::graphics::line::set_color(uint_t i, const fan::color& color)
{
	fan_2d::graphics::vertice_vector::set_color(i * 2, color);
	fan_2d::graphics::vertice_vector::set_color(i * 2 + 1, color);
}

void fan_2d::graphics::line::release_queue(bool line, bool color)
{
	if (line) {
		fan_2d::graphics::vertice_vector::release_queue(true, false, true);
	}
	if (color) {
		fan_2d::graphics::vertice_vector::release_queue(false, true, false);
	}
}

uint_t fan_2d::graphics::line::size() const
{
	return fan_2d::graphics::vertice_vector::size() / 2;
}

void fan_2d::graphics::line::set_thickness(f32_t thickness) {

	fan_2d::graphics::global_vars::line_thickness = thickness;

	glLineWidth(thickness);
}

fan_2d::graphics::line fan_2d::graphics::create_grid(fan::camera* camera, const fan::vec2i& block_size, const fan::vec2i& grid_size, const fan::color& color)
{
	fan_2d::graphics::line lv(camera);

	const fan::vec2 view = (fan::cast<f_t>(grid_size) / block_size).ceiled();

	for (int i = 0; i < view.x; i++) {
		lv.push_back(fan::vec2(i * block_size.x, 0), fan::vec2(i * block_size.x, grid_size.y), color, true);
	}

	for (int i = 0; i < view.y; i++) {
		lv.push_back(fan::vec2(0, i * block_size.y), fan::vec2(grid_size.x, i * block_size.y), color, true);
	}

	lv.release_queue(true, true);

	return lv;
}

fan_2d::graphics::rectangle::rectangle(fan::camera* camera) 
	: m_camera(camera), m_shader(fan::shader(fan_2d::graphics::shader_paths::shape_vector_rectangle_vs, fan_2d::graphics::shader_paths::shape_vector_rectangle_fs)) {

	fan::bind_vao(fan::vao_handler<>::m_buffer_object, [&] {
		color_t::initialize_buffers(m_shader.m_id, location_color, true, color_t::value_type::size());
		position_t::initialize_buffers(m_shader.m_id, location_position, true, position_t::value_type::size());
		size_t::initialize_buffers(m_shader.m_id, location_size, true, size_t::value_type::size());
		angle_t::initialize_buffers(m_shader.m_id, location_angle, true, 1);
		//ebo_t::bind();
	});
}

// protected
fan_2d::graphics::rectangle::rectangle(fan::camera* camera, const fan::shader& shader) : m_camera(camera), m_shader(shader) { }
//

void fan_2d::graphics::rectangle::push_back(const fan::vec2& position, const fan::vec2& size, const fan::color& color, f32_t angle, bool queue)
{
	color_t::basic_push_back(color, queue);
	position_t::basic_push_back(position, queue);
	size_t::basic_push_back(size, queue);
	angle_t::basic_push_back(angle, queue);
}

void fan_2d::graphics::rectangle::reserve(uint32_t size)
{
	color_t::reserve(size);
	position_t::reserve(size);
	size_t::resize(size);
	angle_t::reserve(size);
}

void fan_2d::graphics::rectangle::resize(uint32_t size, const fan::color& color)
{
	color_t::resize(size, color);
	position_t::resize(size);
	size_t::resize(size);
	angle_t::resize(size);
}

void fan_2d::graphics::rectangle::draw(uint32_t i) const
{
	fan::bind_vao(fan::vao_handler<>::m_buffer_object, [&] {

		fan::mat4 projection(1);
		projection = fan_2d::graphics::get_projection(m_camera->m_window->get_size());

		fan::mat4 view(1);
		view = m_camera->get_view_matrix(fan_2d::graphics::get_view_translation(m_camera->m_window->get_size(), view));

		this->m_shader.use();

		this->m_shader.set_mat4("projection", projection);
		this->m_shader.set_mat4("view", view);

		fan_2d::graphics::draw([&] {
			if (i != (uint32_t)fan::uninitialized) {
				glDrawArraysInstancedBaseInstance(GL_TRIANGLES, 0, 6, 1, i);
			}
			else {

				glDrawArraysInstanced(GL_TRIANGLES, 0, 6, this->size());
				//glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0, this->size());

			}
		});

	});

}

void fan_2d::graphics::rectangle::erase(uint32_t i)
{
	color_t::erase(i);
	position_t::erase(i);
	size_t::erase(i);
	angle_t::erase(i);
}

void fan_2d::graphics::rectangle::erase(uint32_t begin, uint32_t end)
{
	color_t::erase(begin, end, false);
	position_t::erase(begin, end, false);
	size_t::erase(begin, end, false);
	angle_t::erase(begin, end, false);
}

fan::vec2 get_transformed_point(fan::vec2 input, float a) {
	float x = input.x * cos(a) - input.y * sin(a);
	float y = input.x * sin(a) + input.y * cos(a);
	return fan::vec2(x, y);
}

// 0 top left, 1 top right, 2 bottom left, 3 bottom right
fan_2d::graphics::rectangle_corners_t fan_2d::graphics::rectangle::get_corners(uint32_t i) const
{
	auto position = this->get_position(i);
	auto size = this->get_size(i);

	fan::vec2 mid = position + size / 2;

	auto corners = get_rectangle_corners_no_rotation(position, size);

	f32_t angle = -angle_t::get_value(i);

	fan::vec2 top_left = get_transformed_point(corners[0] - mid, angle) + mid;
	fan::vec2 top_right = get_transformed_point(corners[1] - mid, angle) + mid;
	fan::vec2 bottom_left = get_transformed_point(corners[2] - mid, angle) + mid;
	fan::vec2 bottom_right = get_transformed_point(corners[3] - mid, angle) + mid;

	return { top_left, top_right, bottom_left, bottom_right };
}
//
//fan::vec2 fan_2d::graphics::rectangle::get_center(uint_t i) const
//{
//	auto corners = this->get_corners(i);
//	return corners[0] + (corners[3] - corners[0]) / 2;
//}
//
f32_t fan_2d::graphics::rectangle::get_rotation(uint32_t i) const
{	
	return angle_t::m_buffer_object[i];
}

// radians
void fan_2d::graphics::rectangle::set_rotation(uint32_t i, f32_t angle, bool queue)
{
	angle_t::set_value(i, fmod(angle, fan::math::pi * 2), queue);
}

const fan::color fan_2d::graphics::rectangle::get_color(uint32_t i) const
{
	return color_t::m_buffer_object[i];
}

void fan_2d::graphics::rectangle::set_color(uint32_t i, const fan::color& color, bool queue)
{
	color_t::set_value(i, color, queue);
}

fan::vec2 fan_2d::graphics::rectangle::get_position(uint32_t i) const
{
	return position_t::get_value(i);
}

void fan_2d::graphics::rectangle::set_position(uint32_t i, const fan::vec2& position, bool queue)
{
	position_t::set_value(i, position);
}

fan::vec2 fan_2d::graphics::rectangle::get_size(uint32_t i) const
{
	return size_t::get_value(i);
}

void fan_2d::graphics::rectangle::set_size(uint32_t i, const fan::vec2& size, bool queue)
{
	size_t::set_value(i, size, queue);
}

uint_t fan_2d::graphics::rectangle::size() const
{
	return position_t::size();
}

void fan_2d::graphics::rectangle::release_queue(bool position, bool size, bool angle, bool color)
{
	if (position) {
		position_t::write_data();
	}
	if (size) {
		size_t::write_data();
	}
	if (angle) {
		angle_t::write_data();
	}
	if (color) {
		color_t::write_data();
	}
}

bool fan_2d::graphics::rectangle::inside(uint_t i, const fan::vec2& position) const {

	auto corners = get_corners(i);
	
	return fan_2d::collision::rectangle::point_inside(corners[0], corners[1], corners[2], corners[3], position == fan::math::inf ? fan::cast<fan::vec2::value_type>(this->m_camera->m_window->get_mouse_position()) : position);
}

//void fan_2d::graphics::rectangle::set_draw_order(draw_mode mode) {
//	for (int i = 0; i < ebo_t::size(); i++) {
//		ebo_t::m_buffer_object[i] = mode == draw_mode::no_draw ? primitive_restart : i;
//	}
//	ebo_t::write_data();
//}

//void fan_2d::graphics::rectangle::set_draw_order(uint32_t i, draw_mode mode) {
//	for (int j = 0; j < 6; j++) {
//		ebo_t::m_buffer_object[i * 6 + j] = mode == draw_mode::no_draw ? primitive_restart : i * 6 + j;
//	}
//	ebo_t::write_data();
//}

fan_2d::graphics::rounded_rectangle::rounded_rectangle(fan::camera* camera) : fan_2d::graphics::vertice_vector(camera) { }

fan_2d::graphics::rounded_rectangle::rounded_rectangle(fan::camera* camera, const fan::vec2& position, const fan::vec2& size, f_t radius, const fan::color& color)
	: fan_2d::graphics::vertice_vector(camera)
{
	this->push_back(position, size, radius, color);
}

void fan_2d::graphics::rounded_rectangle::push_back(const fan::vec2& position, const fan::vec2& size, f_t radius, const fan::color& color, bool queue)
{

	if ((position.y + radius) - (position.y + size.y - radius) > 0) {
		radius = size.y * 0.5;
	}
	if ((position.x + radius) - (position.x + size.x - radius) > 0) {
		radius = size.x * 0.5;
	}

	fan_2d::graphics::vertice_vector::push_back(fan::vec2(position.x + radius, position.y), color, true);
	fan_2d::graphics::vertice_vector::push_back(fan::vec2(position.x + radius, position.y + size.y), color, true);
	fan_2d::graphics::vertice_vector::push_back(fan::vec2(position.x + size.x - radius, position.y + size.y), color, true);

	fan_2d::graphics::vertice_vector::push_back(fan::vec2(position.x + radius, position.y), color, true);
	fan_2d::graphics::vertice_vector::push_back(fan::vec2(position.x + size.x - radius, position.y), color, true);
	fan_2d::graphics::vertice_vector::push_back(fan::vec2(position.x + size.x - radius, position.y + size.y), color, true);

	fan_2d::graphics::vertice_vector::push_back(fan::vec2(position.x, position.y + radius), color, true);
	fan_2d::graphics::vertice_vector::push_back(fan::vec2(position.x + size.x, position.y + radius), color, true);
	fan_2d::graphics::vertice_vector::push_back(fan::vec2(position.x + size.x, position.y + size.y - radius), color, true);

	fan_2d::graphics::vertice_vector::push_back(fan::vec2(position.x, position.y + radius), color, true);
	fan_2d::graphics::vertice_vector::push_back(fan::vec2(position.x, position.y + size.y - radius), color, true);
	fan_2d::graphics::vertice_vector::push_back(fan::vec2(position.x + size.x, position.y + size.y - radius), color, true);

	std::array<std::array<fan::vec2, 3>, 4> positions{
		std::array<fan::vec2, 3>{
		fan::vec2(position.x + radius, position.y),
			fan::vec2(position.x, position.y + radius),
			position + radius
	},
		std::array<fan::vec2, 3>{
			fan::vec2(position.x + size.x, position.y + radius),
				fan::vec2(position.x + size.x - radius, position.y),
				fan::vec2(position.x + size.x - radius, position.y + radius)
		},
			std::array<fan::vec2, 3>{
				fan::vec2(position.x, position.y + size.y - radius),
					fan::vec2(position.x + radius, position.y + size.y),
					fan::vec2(position.x + radius, position.y + size.y - radius)
			},
				std::array<fan::vec2, 3>{
					fan::vec2(position.x + size.x - radius, position.y + size.y),
						fan::vec2(position.x + size.x, position.y + size.y - radius),
						position + size - radius
				}
	};

	std::array<fan::vec2, 4> old_positions{
		positions[0][2],
		positions[1][2],
		positions[2][2],
		positions[3][2]
	};

	const auto get_offsets = [&](uint_t i, auto t) {
		switch (i) {
			case 0: { return fan::vec2(std::cos(fan::math::half_pi + t), std::sin(fan::math::half_pi + t)) * radius; }
			case 1: { return fan::vec2(std::cos(t), std::sin(fan::math::pi + t)) * radius; }
			case 2: { return fan::vec2(std::cos(fan::math::pi + t), std::sin(fan::math::pi + t)) * radius; }			
			default: { return fan::vec2(std::cos(fan::math::pi + fan::math::half_pi + t), std::sin(fan::math::pi + fan::math::half_pi + fan::math::pi + t)) * radius; }
		}
	};

	for (uint_t i = 0; i < m_segments; i++) {

		f_t t = fan::math::half_pi * f_t(i) / (m_segments - 1);

		for (int j = 0; j < 4; j++) {
			const fan::vec2 offset = get_offsets(j, t);

			fan_2d::graphics::vertice_vector::push_back(old_positions[j], color, true);
			fan_2d::graphics::vertice_vector::push_back(fan::vec2(positions[j][((j + 1) >> 1) & 1].x + offset.x, positions[j][~((j + 1) >> 1) & 1].y + ((j & 1) ? offset.y : -offset.y)), color, true);
			fan_2d::graphics::vertice_vector::push_back(positions[j][2], color, true);

			old_positions[j] = fan_2d::graphics::vertice_vector::m_position[fan_2d::graphics::vertice_vector::m_position.size() - 2];

		}
	}

	if (!queue) {
		this->release_queue(true, true, true);
	}

	fan_2d::graphics::rounded_rectangle::m_position.emplace_back(position);
	fan_2d::graphics::rounded_rectangle::m_size.emplace_back(fan::vec2(size.x, size.y));
	m_radius.emplace_back(radius);

	m_data_offset.emplace_back(rounded_rectangle::basic_vertice_vector::size());
}

fan::vec2 fan_2d::graphics::rounded_rectangle::get_position(uint_t i) const
{
	return fan_2d::graphics::rounded_rectangle::m_position[i];
}

void fan_2d::graphics::rounded_rectangle::set_position(uint_t i, const fan::vec2& position, bool queue)
{
	const auto offset = fan_2d::graphics::rounded_rectangle::m_data_offset[i];
	const auto previous_offset = !i ? 0 : fan_2d::graphics::rounded_rectangle::m_data_offset[i - 1];
	const auto distance = position - fan_2d::graphics::rounded_rectangle::get_position(i);
	for (uint_t j = 0; j < offset - previous_offset; j++) {
		fan_2d::graphics::vertice_vector::m_position[previous_offset + j] += distance;
	}

	if (!queue) {
		basic_shape_position::glsl_location_handler::edit_data(fan_2d::graphics::vertice_vector::m_position.data() + previous_offset, sizeof(fan::vec2) * previous_offset, sizeof(fan::vec2) * (offset - previous_offset));
	}

	fan_2d::graphics::rounded_rectangle::m_position[i] = position;
}

fan::vec2 fan_2d::graphics::rounded_rectangle::get_size(uint_t i) const
{
	return fan_2d::graphics::rounded_rectangle::m_size[i];
}

void fan_2d::graphics::rounded_rectangle::set_size(uint_t i, const fan::vec2& size, bool queue)
{
	fan_2d::graphics::rounded_rectangle::m_size[i] = fan::vec2(size.x, size.y);
	this->edit_rectangle(i, queue);
}

f_t fan_2d::graphics::rounded_rectangle::get_radius(uint_t i) const
{
	return m_radius[i];
}

void fan_2d::graphics::rounded_rectangle::set_radius(uint_t i, f_t radius, bool queue)
{
	const fan::vec2 position = get_position(i);
	const fan::vec2 size = get_size(i);
	f_t previous_radius = get_radius(i);

	if ((position.y + previous_radius) - (position.y + size.y - previous_radius) > 0 ||
		(position.x + previous_radius) - (position.x + size.x - previous_radius) > 0) {
		return;
	}

	m_radius[i] = radius;
	this->edit_rectangle(i, queue);
}

void fan_2d::graphics::rounded_rectangle::draw() const
{
	fan_2d::graphics::vertice_vector::draw(GL_TRIANGLES, 10); // 10 ?
}

bool fan_2d::graphics::rounded_rectangle::inside(uint_t i) const
{
	const auto position = m_camera->m_window->get_mouse_position();

	if (position.x > m_position[i].x + m_radius[i] &&
		position.x < m_position[i].x + m_size[i].x - m_radius[i] &&
		position.y > m_position[i].y && 
		position.y < m_position[i].y + m_size[i].y)
	{
		return true;
	}

	if (position.x > m_position[i].x &&
		position.x < m_position[i].x + m_size[i].x &&
		position.y > m_position[i].y + m_radius[i] && 
		position.y < m_position[i].y + m_size[i].y - m_radius[i])
	{
		return true;
	}

	if (fan_2d::collision::circle::point_inside(position, m_position[i] + m_radius[i], m_radius[i])) {
		return true;
	}

	if (fan_2d::collision::circle::point_inside(position, m_position[i] + fan::vec2(m_radius[i], m_size[i].y - m_radius[i]), m_radius[i])) {
		return true;
	}

	if (fan_2d::collision::circle::point_inside(position, m_position[i] + fan::vec2(m_size[i].x - m_radius[i], m_radius[i]), m_radius[i])) {
		return true;
	}

	if (fan_2d::collision::circle::point_inside(position, m_position[i] + m_size[i] - m_radius[i], m_radius[i])) {
		return true;
	}

	return false;
}

fan::color fan_2d::graphics::rounded_rectangle::get_color(uint_t i) const {
	return m_color[!i ? 0 : fan_2d::graphics::rounded_rectangle::m_data_offset[i - 1]];
}

void fan_2d::graphics::rounded_rectangle::set_color(uint_t i, const fan::color& color, bool queue)
{
	const auto offset = fan_2d::graphics::rounded_rectangle::m_data_offset[i];
	const auto previous_offset = !i ? 0 : fan_2d::graphics::rounded_rectangle::m_data_offset[i - 1];

	for (uint_t i = previous_offset; i != offset; i++) {
		m_color[i] = color;
	}

	if (!queue) {
		rounded_rectangle::basic_shape_color_vector::edit_data(fan_2d::graphics::vertice_vector::m_color.data() + previous_offset, sizeof(fan::color) * previous_offset, sizeof(fan::color) * (offset - previous_offset));
	}
}

uint_t fan_2d::graphics::rounded_rectangle::size() const
{
	return m_size.size();
}

void fan_2d::graphics::rounded_rectangle::edit_rectangle(uint_t i, bool queue)
{
	const auto offset = fan_2d::graphics::rounded_rectangle::m_data_offset[i];
	const auto previous_offset = !i ? 0 : fan_2d::graphics::rounded_rectangle::m_data_offset[i - 1];

	uint_t current = previous_offset;

	const fan::vec2 position = this->get_position(i);
	f_t radius = m_radius[i];
	const fan::vec2 size = m_size[i];

	if ((position.y + radius) - (position.y + size.y - radius) > 0) {
		radius = size.y * 0.5;
	}
	if ((position.x + radius) - (position.x + size.x - radius) > 0) {
		radius = size.x * 0.5;
	}

	fan_2d::graphics::vertice_vector::m_position[current++] = fan::vec2(position.x + radius, position.y);
	fan_2d::graphics::vertice_vector::m_position[current++] = fan::vec2(position.x + radius, position.y + size.y);
	fan_2d::graphics::vertice_vector::m_position[current++] = fan::vec2(position.x + size.x - radius, position.y + size.y);

	fan_2d::graphics::vertice_vector::m_position[current++] = fan::vec2(position.x + radius, position.y);
	fan_2d::graphics::vertice_vector::m_position[current++] = fan::vec2(position.x + size.x - radius, position.y);
	fan_2d::graphics::vertice_vector::m_position[current++] = fan::vec2(position.x + size.x - radius, position.y + size.y);

	fan_2d::graphics::vertice_vector::m_position[current++] = fan::vec2(position.x, position.y + radius);
	fan_2d::graphics::vertice_vector::m_position[current++] = fan::vec2(position.x + size.x, position.y + radius);
	fan_2d::graphics::vertice_vector::m_position[current++] = fan::vec2(position.x + size.x, position.y + size.y - radius);

	fan_2d::graphics::vertice_vector::m_position[current++] = fan::vec2(position.x, position.y + radius);
	fan_2d::graphics::vertice_vector::m_position[current++] = fan::vec2(position.x, position.y + size.y - radius);
	fan_2d::graphics::vertice_vector::m_position[current++] = fan::vec2(position.x + size.x, position.y + size.y - radius);

	std::array<std::array<fan::vec2, 3>, 4> positions{
		std::array<fan::vec2, 3>{
		fan::vec2(position.x + radius, position.y),
			fan::vec2(position.x, position.y + radius),
			position + radius
	},
		std::array<fan::vec2, 3>{
			fan::vec2(position.x + size.x, position.y + radius),
				fan::vec2(position.x + size.x - radius, position.y),
				fan::vec2(position.x + size.x - radius, position.y + radius)
		},
			std::array<fan::vec2, 3>{
				fan::vec2(position.x, position.y + size.y - radius),
					fan::vec2(position.x + radius, position.y + size.y),
					fan::vec2(position.x + radius, position.y + size.y - radius)
			},
				std::array<fan::vec2, 3>{
					fan::vec2(position.x + size.x - radius, position.y + size.y),
						fan::vec2(position.x + size.x, position.y + size.y - radius),
						position + size - radius
				}
	};

	std::array<fan::vec2, 4> old_positions{
		positions[0][2],
		positions[1][2],
		positions[2][2],
		positions[3][2]
	};

	const auto get_offsets = [&](uint_t i, auto t) {
		switch (i) {
			case 0: { return fan::vec2(std::cos(fan::math::half_pi + t), std::sin(fan::math::half_pi + t)) * radius; }
			case 1: { return fan::vec2(std::cos(t), std::sin(fan::math::pi + t)) * radius; }
			case 2: { return fan::vec2(std::cos(fan::math::pi + t), std::sin(fan::math::pi + t)) * radius; }			
			default: { return fan::vec2(std::cos(fan::math::pi + fan::math::half_pi + t), std::sin(fan::math::pi + fan::math::half_pi + fan::math::pi + t)) * radius; }
		}
	};

	for (uint_t i = 0; i < m_segments; i++) {

		f_t t = fan::math::half_pi * f_t(i) / (m_segments - 1);

		for (int j = 0; j < 4; j++) {
			const fan::vec2 offset = get_offsets(j, t);

			fan_2d::graphics::vertice_vector::m_position[current++] = old_positions[j];
			fan_2d::graphics::vertice_vector::m_position[current++] = fan::vec2(positions[j][((j + 1) >> 1) & 1].x + offset.x, positions[j][~((j + 1) >> 1) & 1].y + ((j & 1) ? offset.y : -offset.y));
			fan_2d::graphics::vertice_vector::m_position[current++] = positions[j][2];

			old_positions[j] = fan_2d::graphics::vertice_vector::m_position[current - 2];

		}
	}

	if (!queue) {
		basic_shape_position::glsl_location_handler::edit_data(fan_2d::graphics::vertice_vector::m_position.data() + previous_offset, sizeof(fan::vec2) * previous_offset, sizeof(fan::vec2) * (offset - previous_offset));
	}
}

fan_2d::graphics::circle::circle(fan::camera* camera) : fan_2d::graphics::vertice_vector(camera) {
	m_index_restart = m_segments;
}

void fan_2d::graphics::circle::push_back(const fan::vec2& position, f32_t radius, const fan::color& color, bool queue)
{
	this->m_position.emplace_back(position);
	this->m_radius.emplace_back(radius);

	for (int i = 0; i < m_segments; i++) {

		f32_t theta = fan::math::two_pi * f32_t(i) / m_segments;

		vertice_vector::push_back(position + fan::vec2(radius * std::cos(theta), radius * std::sin(theta)), color, true);
	}

	vertice_vector::release_queue(!queue, !queue, !queue);
}

fan::vec2 fan_2d::graphics::circle::get_position(uint_t i) const
{
	return this->m_position[i];
}

void fan_2d::graphics::circle::set_position(uint_t i, const fan::vec2& position, bool queue)
{
	this->m_position[i] = position;

	for (int j = 0; j < m_segments; j++) {

		f32_t theta = fan::math::two_pi * f32_t(j) / m_segments;

		vertice_vector::set_position(i * m_segments + j, position + fan::vec2(m_radius[i] * std::cos(theta), m_radius[i] * std::sin(theta)), true);

	}

	vertice_vector::release_queue(!queue, false, false);
}

f32_t fan_2d::graphics::circle::get_radius(uint_t i) const
{
	return this->m_radius[i];
}

void fan_2d::graphics::circle::set_radius(uint_t i, f32_t radius, bool queue)
{
	this->m_radius[i] = radius;

	const fan::vec2 position = this->get_position(i);

	for (int j = 0; j < m_segments; j++) {

		f32_t theta = fan::math::two_pi * f32_t(j) / m_segments;

		vertice_vector::set_position(i * m_segments + j, position + fan::vec2(m_radius[i] * std::cos(theta), m_radius[i] * std::sin(theta)), true);

	}

	vertice_vector::release_queue(!queue, !queue, !queue);
}

void fan_2d::graphics::circle::draw() const
{
	fan_2d::graphics::vertice_vector::draw(GL_TRIANGLE_FAN, m_segments);
}

bool fan_2d::graphics::circle::inside(uint_t i) const
{
	const fan::vec2 position = this->get_position(i);
	const fan::vec2 mouse_position = this->m_camera->m_window->get_mouse_position();

	if ((mouse_position.x - position.x) * (mouse_position.x - position.x) +
		(mouse_position.y - position.y) * (mouse_position.y - position.y) <= m_radius[i] * m_radius[i]) {
		return true;
	}

	return false;
}

fan::color fan_2d::graphics::circle::get_color(uint_t i) const
{
	return this->m_color[i * m_segments];
}

void fan_2d::graphics::circle::set_color(uint_t i, const fan::color& color, bool queue)
{
	for (int j = 0; j < m_segments; j++) {
		vertice_vector::set_color(i * m_segments + j, color, true);
	}

	vertice_vector::release_queue(false, !queue, false);
}

uint_t fan_2d::graphics::circle::size() const
{
	return this->m_position.size();
}

void fan_2d::graphics::circle::erase(uint_t i, bool queue) {
	fan_2d::graphics::vertice_vector::erase(i * m_segments, i * m_segments + m_segments, queue);

	this->m_position.erase(this->m_position.begin() + i);
	this->m_radius.erase(this->m_radius.begin() + i);
}

void fan_2d::graphics::circle::erase(uint_t begin, uint_t end, bool queue) {
	fan_2d::graphics::vertice_vector::erase(begin * m_segments, end * m_segments);

	this->m_position.erase(this->m_position.begin() + begin, this->m_position.begin() + end);
	this->m_radius.erase(this->m_radius.begin() + begin, this->m_radius.begin() + end);
}

fan_2d::graphics::sprite::sprite(fan::camera* camera, const std::string& path) 
	: fan_2d::graphics::rectangle(camera, fan::shader(shader_paths::sprite_vs, shader_paths::sprite_fs)), 
	m_screen_shader(shader_paths::shape_vector_vs, shader_paths::shape_vector_fs), m_amount_of_textures(0) {

	//rectangle
	fan::bind_vao(fan::vao_handler<>::m_buffer_object, [&] {
		position_t::initialize_buffers(m_shader.m_id, location_position, true, position_t::value_type::size());
		size_t::initialize_buffers(m_shader.m_id, location_size, true, size_t::value_type::size());
		angle_t::initialize_buffers(m_shader.m_id, location_angle, true, 1);
	});

	auto info = fan_2d::graphics::load_image(path);

	m_amount_of_textures++;

	m_textures.emplace_back(info.texture_id);

	m_texture_offsets.emplace_back(0);

	m_original_image_size = info.size;

}

fan_2d::graphics::sprite::sprite(fan::camera* camera) 
	: fan_2d::graphics::rectangle(camera), m_screen_shader(shader_paths::shape_vector_vs, shader_paths::shape_vector_fs), m_amount_of_textures(0)
{
	//this->initialize_buffers({ 800, 600 });
}

fan_2d::graphics::sprite::sprite(fan::camera* camera, const std::string& path, const fan::vec2& position, const fan::vec2& size)
	: fan_2d::graphics::rectangle(camera), m_screen_shader(shader_paths::shape_vector_vs, shader_paths::shape_vector_fs), m_amount_of_textures(0)
{
	auto info = fan_2d::graphics::load_image(path);

	m_textures.emplace_back(info.texture_id);

	fan_2d::graphics::rectangle::push_back(position, !size ? fan::cast<fan::vec2::value_type>(info.size) : size, 0);

	//initialize_buffers(!size ? image_size : size);
	m_transparency.emplace_back(1);
	m_amount_of_textures++;

	m_texture_offsets.emplace_back(0);

	m_original_image_size = info.size;
}

fan_2d::graphics::sprite::sprite(fan::camera* camera, unsigned char* pixels, const fan::vec2& position, const fan::vec2i& size)
	: fan_2d::graphics::rectangle(camera), m_screen_shader(shader_paths::shape_vector_vs, shader_paths::shape_vector_fs), m_amount_of_textures(0)
{
	auto info = fan_2d::graphics::load_image(pixels, size);

	m_textures.emplace_back(info.texture_id);

	fan_2d::graphics::rectangle::push_back(position, !size ? info.size : size, 0);

	m_transparency.emplace_back(1);
	m_amount_of_textures++;

	m_texture_offsets.emplace_back(0);

	m_original_image_size = info.size;
}

fan_2d::graphics::sprite::sprite(fan::camera* camera, uint32_t texture_id, const fan::vec2& position, const fan::vec2& size)
	: fan_2d::graphics::rectangle(camera), m_screen_shader(shader_paths::shape_vector_vs, shader_paths::shape_vector_fs), m_amount_of_textures(0)
{
	this->push_back(texture_id, position, size);
}

fan_2d::graphics::sprite::sprite(const fan_2d::graphics::sprite& sprite)
	: 
	fan_2d::graphics::rectangle(sprite),
	fan::texture_handler<1>(sprite),
	fan::render_buffer_handler<>(sprite),
	fan::frame_buffer_handler<>(sprite)
{
	m_screen_shader = sprite.m_screen_shader;
	m_transparency = sprite.m_transparency;
	m_textures = sprite.m_textures;
	m_texture_offsets = sprite.m_texture_offsets;
	m_amount_of_textures = sprite.m_amount_of_textures;
	m_original_image_size = sprite.m_original_image_size;
}

fan_2d::graphics::sprite::sprite(fan_2d::graphics::sprite&& sprite) noexcept
	: 
	fan_2d::graphics::rectangle(std::move(sprite)),

	fan::texture_handler<1>(std::move(sprite)),
	fan::render_buffer_handler<>(std::move(sprite)),
	fan::frame_buffer_handler<>(std::move(sprite))
{
	m_screen_shader = std::move(sprite.m_screen_shader);
	m_transparency = std::move(sprite.m_transparency);
	m_textures = std::move(sprite.m_textures);
	m_texture_offsets = std::move(sprite.m_texture_offsets);
	m_amount_of_textures = std::move(sprite.m_amount_of_textures);
	m_original_image_size = std::move(sprite.m_original_image_size);
}

fan_2d::graphics::sprite& fan_2d::graphics::sprite::operator=(const fan_2d::graphics::sprite& sprite)
{
	fan_2d::graphics::rectangle::operator=(sprite);

	fan::texture_handler<1>::operator=(sprite);
	fan::render_buffer_handler<>::operator=(sprite);
	fan::frame_buffer_handler<>::operator=(sprite);

	m_screen_shader = sprite.m_screen_shader;
	m_transparency = sprite.m_transparency;
	m_textures = sprite.m_textures;
	m_texture_offsets = sprite.m_texture_offsets;
	m_amount_of_textures = sprite.m_amount_of_textures;
	m_original_image_size = sprite.m_original_image_size;

	return *this;
}

fan_2d::graphics::sprite& fan_2d::graphics::sprite::operator=(fan_2d::graphics::sprite&& sprite)
{
	fan_2d::graphics::rectangle::operator=(std::move(sprite));

	fan::texture_handler<1>::operator=(std::move(sprite));
	fan::render_buffer_handler<>::operator=(std::move(sprite));
	fan::frame_buffer_handler<>::operator=(std::move(sprite));

	m_screen_shader = std::move(sprite.m_screen_shader);
	m_transparency = std::move(sprite.m_transparency);
	m_textures = std::move(sprite.m_textures);
	m_texture_offsets = std::move(sprite.m_texture_offsets);
	m_amount_of_textures = std::move(sprite.m_amount_of_textures);
	m_original_image_size = std::move(sprite.m_original_image_size);

	return *this;
}

fan_2d::graphics::sprite::~sprite()
{
	glDeleteTextures(m_textures.size(), m_textures.data());
}

void fan_2d::graphics::sprite::reload_sprite(uint32_t i, const std::string& path, const fan::vec2& size)
{
	uint32_t new_texture = 0;

	glGenTextures(1, &new_texture);

	m_textures.emplace_back(new_texture);

	m_amount_of_textures++;

	auto image_info = load_image(m_textures[i], path);
	this->set_size(i, !size ? fan::cast<fan::vec2::value_type>(image_info.size) : size);
}

void fan_2d::graphics::sprite::reload_sprite(uint32_t i, unsigned char* pixels, const fan::vec2i& size) // fix
{
	if (i >= m_textures.size()) {
		auto info = fan_2d::graphics::load_image(pixels, size);

		m_textures.emplace_back(info.texture_id);

		fan_2d::graphics::rectangle::push_back(0, !size ? info.size : size, 0);

		m_transparency.emplace_back(1);
		m_amount_of_textures++;
		if (m_texture_offsets.empty()) {
			m_texture_offsets.emplace_back(0);
		}
		else {
			m_texture_offsets.emplace_back(m_texture_offsets[m_texture_offsets.size() - 1]);
		}
	}

	glBindTexture(GL_TEXTURE_2D, m_textures[i]);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, image_load_properties::filter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, image_load_properties::filter);

	glTexImage2D(GL_TEXTURE_2D, 0, image_load_properties::internal_format, size.x, size.y, 0, image_load_properties::format, image_load_properties::type, pixels);
	glGenerateMipmap(GL_TEXTURE_2D);

	this->set_size(i, size);

	glBindTexture(GL_TEXTURE_2D, 0);
}

void fan_2d::graphics::sprite::push_back(const fan::vec2& position, const fan::vec2& size, bool queue)
{
	m_transparency.emplace_back(1);
	
	fan_2d::graphics::rectangle::push_back(position, size, 0, queue);
}

void fan_2d::graphics::sprite::push_back(uint32_t texture_id, const fan::vec2& position, const fan::vec2& size, bool queue)
{
	m_transparency.emplace_back(1);
	fan_2d::graphics::rectangle::push_back(position, size, 0, queue);
	auto found = std::lower_bound(m_textures.begin(), m_textures.end(), texture_id);
	if (found == m_textures.end()) {
		m_texture_offsets.emplace_back(m_textures.size());
		m_amount_of_textures++;
	}
	m_textures.emplace_back(texture_id);
	//std::sort(m_textures.begin(), m_textures.end());

}

void fan_2d::graphics::sprite::draw(uint32_t begin, uint32_t end) const
{
//	fan::mat4 projection(1);
//	projection = fan_2d::graphics::get_projection(m_camera->m_window->get_size());
//
//	fan::mat4 view(1);
//	view = m_camera->get_view_matrix(fan_2d::graphics::get_view_translation(m_camera->m_window->get_size(), view));
//
//	fan::mat4 model(1);
//	model = fan::math::translate(model, get_position() + get_size() / 2);
//	model = fan::math::rotate(model, fan::math::radians(get_rotation()), fan::vec3(0, 0, 1));
//	model = fan::math::translate(model, -get_size() / 2);
//	model = fan::math::scale(model, get_size());
//
//
//	//fbo_t::bind_gl_storage_buffer([&] {
//	glEnable(GL_DEPTH_TEST); 
//	m_camera->m_window->set_background_color(0);
//	//fan::window::set_background_color(0);
//
//	m_shader.use();
//	/*m_shader.set_mat4("projection", projection);
//	m_shader.set_mat4("view", view);
//	m_shader.set_mat4("model", model);*/
//	m_shader.set_int("texture_sampler", 0);
////	m_shader.set_float("transparency", m_transparency);
//
//	glActiveTexture(GL_TEXTURE0);
//
//	glBindTexture(GL_TEXTURE_2D, m_textures[0]);
//	fan_2d::graphics::vertice_vector::draw(GL_TRIANGLES, 6, 0, 1, true);
////	basic_shape::basic_draw(GL_TRIANGLES, 6);
//
//	//});
//
//	glDisable(GL_DEPTH_TEST);
//
//	m_camera->m_window->set_background_color(0);
//
//	m_screen_shader.use();
//	m_screen_shader.set_mat4("projection", projection);
//	m_screen_shader.set_mat4("view", view);
//	m_screen_shader.set_mat4("model", model);
//	m_screen_shader.set_int("texture_sampler", 0);
//	//m_screen_shader.set_float("transparency", m_transparency);
//	glBindTexture(GL_TEXTURE_2D, fan::texture_handler<1>::m_buffer_object);
//	fan_2d::graphics::vertice_vector::draw(GL_TRIANGLES, 6, 0, 0, true);

	this->m_shader.use();

	uint32_t real_begin = begin == (uint32_t)fan::uninitialized ? 0 : begin;
	uint32_t real_end = end == (uint32_t)fan::uninitialized ? this->size() : end;

	for (uint32_t j = real_begin; j < real_end - real_begin + 1 && j < m_textures.size(); j++) {
		this->m_shader.set_int("texture_sampler", j);
		glActiveTexture(GL_TEXTURE0 + j);
		glBindTexture(GL_TEXTURE_2D, m_textures[(begin == (uint32_t)fan::uninitialized ? 0 : begin) + j]);

		fan_2d::graphics::rectangle::draw(j);
	}

}

void fan_2d::graphics::sprite::initialize_buffers(const fan::vec2& size) {
	fan::frame_buffer_handler<>::bind_gl_storage_buffer([&] {
		fan::texture_handler<1>::bind_gl_storage_buffer([&] {
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, size.x, size.y, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, image_load_properties::filter);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, image_load_properties::filter);
			glGenerateMipmap(GL_TEXTURE_2D);

			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fan::texture_handler<1>::m_buffer_object, 0);

			fan::render_buffer_handler<>::bind_gl_storage_buffer([&] {
				glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, size.x, size.y);
				glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, fan::render_buffer_handler<>::m_buffer_object); 

				if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
					fan::print("frame buffer is not complete");
				}
			});

		});
	});
}

//
//fan_2d::graphics::sprite_sheet::sprite_sheet(fan::camera* camera, uint32_t time)
//	: fan_2d::graphics::sprite(camera), current_sheet(0) {
//	sheet_timer = fan::timer<>(fan::timer<>::start(), time);
//}
//
//void fan_2d::graphics::sprite_sheet::draw() {
//
//	fan_2d::graphics::sprite::draw(current_sheet, current_sheet);
//
//	if (sheet_timer.finished()) {
//
//		current_sheet = (current_sheet + 1) % size();
//
//		sheet_timer.restart();
//	}
//
//}
//
////fan_2d::graphics::rope::rope(fan::camera* camera)
////	: fan_2d::graphics::line(camera) {}
//
////void fan_2d::graphics::rope::push_back(const std::vector<std::pair<fan::vec2, fan::vec2>>& joints, const fan::color& color, bool queue)
////{
////	for (int i = 0; i < joints.size(); i++) {
////		fan_2d::graphics::line::push_back(joints[i].first, joints[i].second, color, true);
////	}
////
////	fan_2d::graphics::line::release_queue(!queue, !queue);
////}
//
fan_2d::graphics::particles::particles(fan::camera* camera)
	: rectangle(camera) {
	m_shader = fan::shader(fan_2d::graphics::shader_paths::shape_vector_vs, fan_2d::graphics::shader_paths::shape_vector_fs);
}

void fan_2d::graphics::particles::push_back(const fan::vec2& position, const fan::vec2& size, f32_t angle, f32_t angle_velocity, const fan::vec2& velocity, const fan::color& color, uint_t time, bool queue)
{
	fan_2d::graphics::rectangle::push_back(position, size, color, true);
	this->set_rotation(this->size() - 1, angle, true);

	this->release_queue(!queue, !queue, !queue, !queue);

	this->m_particles.push_back({ angle_velocity, velocity, fan::timer(fan::timer<>::start(), time) });
}

void fan_2d::graphics::particles::update()
{
	for (uint_t i = 0; i < this->size(); i++) {
		if (this->m_particles[i].m_timer.finished()) {
			this->erase(i);
			this->m_particles.erase(this->m_particles.begin() + i);
			continue;
		}
		this->set_position(i, this->get_position(i) + this->m_particles[i].m_velocity * m_camera->m_window->get_delta_time(), true);
		this->set_rotation(i, this->get_rotation(i) + this->m_particles[i].m_angle_velocity * m_camera->m_window->get_delta_time(), true);
	}
	this->release_queue(true, false, true, false);
}

void fan_3d::graphics::add_camera_rotation_callback(fan::camera* camera) {
	camera->m_window->add_mouse_move_callback(std::function<void(fan::window*)>(std::bind(&fan::camera::rotate_camera, camera, 0)));
}

fan_3d::graphics::terrain_generator::terrain_generator(fan::camera* camera, const std::string& path, const f32_t texture_scale, const fan::vec3& position, const fan::vec2ui& map_size, f_t triangle_size, const fan::vec2& mesh_size)
	: m_camera(camera), m_shader(fan_3d::graphics::shader_paths::triangle_vector_vs, fan_3d::graphics::shader_paths::triangle_vector_fs), m_triangle_size(triangle_size)
{
	std::vector<fan::vec2> texture_coordinates;
	this->m_triangle_vertices.reserve((uint64_t)map_size.x * (uint64_t)map_size.y);

	FastNoiseLite noise;
	noise.SetNoiseType(FastNoiseLite::NoiseType_Perlin);

	std::vector<fan::vec3> normals;
	uint_t index = 0;
	fan::vec2 mesh_height;
	for (uint_t y = 0; y < map_size.y; y++) {
		mesh_height.x = 0;
		for (uint_t x = 0; x < map_size.x; x++) {
			this->m_triangle_vertices.emplace_back(fan::vec3(x * triangle_size, y * triangle_size, noise.GetNoise(mesh_height.x, mesh_height.y) * 100) + position);
			index++;
			texture_coordinates.emplace_back(fan::vec2(x * triangle_size, y * triangle_size) / texture_scale);
			mesh_height.x += mesh_size.x;
		}
		mesh_height.y += mesh_size.y;
	}

	for (uint_t y = 0; y < map_size.y - 1; y++) {
		for (uint_t x = 0; x < map_size.x; x++) {
			for (uint_t i = 0; i < 2; i++) {
				this->m_indices.emplace_back(((y + i) * map_size.x) + x);
			}
		}
		for (uint_t x = map_size.x - 1; x != (uint_t)-1; x--) {
			this->m_indices.emplace_back(((y + 1) * map_size.x) + x);
			if (x == map_size.x - 1 || !x) {
				continue;
			}
			this->m_indices.emplace_back(((y + 1) * map_size.x) + x);
		}
	}

	const uint_t first_corners[] = { 1, 0, map_size.x };

	for (uint_t i = 0; i + first_corners[2] < map_size.x * map_size.y; i++) {
		auto v = fan_3d::math::normalize(fan::math::cross(
			m_triangle_vertices[first_corners[2] + i] - m_triangle_vertices[first_corners[0] + i],
			m_triangle_vertices[first_corners[1] + i] - m_triangle_vertices[first_corners[0] + i]
		));
		if (v.z < 0) {
			v = -v;
		}
		normals.emplace_back(
			v
		);
	}

	glGenVertexArrays(1, &m_vao);
	glGenBuffers(1, &m_normals_vbo);
	glGenBuffers(1, &m_vertices_vbo);
	glGenBuffers(1, &m_texture_vbo);
	glGenBuffers(1, &m_ebo);
	glBindVertexArray(m_vao);

	basic_shape_color_vector::initialize_buffers(false);
	m_texture = load_texture(path, std::string(), false);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(m_indices[0]) * m_indices.size(), m_indices.data(), GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, m_vertices_vbo);
	glBufferData(GL_ARRAY_BUFFER, m_vertice_size * m_triangle_vertices.size(), m_triangle_vertices.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(1, fan::vec3::size(), fan::GL_FLOAT_T, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(1);

	glBindBuffer(GL_ARRAY_BUFFER, m_texture_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(texture_coordinates[0]) * texture_coordinates.size(), texture_coordinates.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(2, fan::vec2::size(), fan::GL_FLOAT_T, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(2);

	glBindBuffer(GL_ARRAY_BUFFER, m_normals_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(normals[0]) * normals.size(), normals.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(3, fan::vec3::size(), fan::GL_FLOAT_T, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(3);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	fan::print("amount of veritices:", m_triangle_vertices.size());

}

fan_3d::graphics::terrain_generator::~terrain_generator()
{
	fan_validate_buffer(m_texture, glDeleteTextures(1, &m_texture));
	fan_validate_buffer(m_vao, glDeleteVertexArrays(1, &m_vao));
	fan_validate_buffer(m_texture_vbo, glDeleteBuffers(1, &m_texture_vbo));
	fan_validate_buffer(m_vertices_vbo, glDeleteBuffers(1, &m_vertices_vbo));
	fan_validate_buffer(m_ebo, glDeleteBuffers(1, &m_ebo));
	fan_validate_buffer(m_normals_vbo, glDeleteBuffers(1, &m_normals_vbo));
	fan_validate_buffer(m_shader.m_id, glDeleteProgram(m_shader.m_id));
}

void fan_3d::graphics::terrain_generator::insert(const std::vector<triangle_vertices_t>& vertices, const std::vector<fan::color>& color, bool queue)
{
	fan_3d::graphics::terrain_generator::m_triangle_vertices.insert(fan_3d::graphics::terrain_generator::m_triangle_vertices.end(), vertices.begin(), vertices.end());

	basic_shape_color_vector::m_color.insert(basic_shape_color_vector::m_color.end(), color.begin(), color.end());

	if (!queue) {
		basic_shape_color_vector::write_data();
		fan::write_glbuffer(m_vertices_vbo, m_triangle_vertices.data(), m_vertice_size * m_triangle_vertices.size());
	}
}

void fan_3d::graphics::terrain_generator::push_back(const triangle_vertices_t& vertices, const fan::color& color, bool queue) {
	fan_3d::graphics::terrain_generator::m_triangle_vertices.emplace_back(vertices);

	std::fill_n(std::back_inserter(basic_shape_color_vector::m_color), 4, color);

	if (!queue) {
		basic_shape_color_vector::write_data();
		fan::write_glbuffer(m_vertices_vbo, m_triangle_vertices.data(), m_vertice_size * m_triangle_vertices.size());
	}
}

void fan_3d::graphics::terrain_generator::edit_data(uint_t i, const triangle_vertices_t& vertices, const fan::color& color)
{
	basic_shape_color_vector::m_color[i] = color;
	fan_3d::graphics::terrain_generator::m_triangle_vertices[i] = vertices;

	glBindBuffer(GL_ARRAY_BUFFER, fan_3d::graphics::terrain_generator::m_vertices_vbo);
	glBufferSubData(GL_ARRAY_BUFFER, fan_3d::graphics::terrain_generator::m_vertice_size * i, fan_3d::graphics::terrain_generator::m_vertice_size, vertices.data());

	basic_shape_color_vector::edit_data(i); // could be broken
}

void fan_3d::graphics::terrain_generator::release_queue()
{
	basic_shape_color_vector::write_data();
	fan::write_glbuffer(m_vertices_vbo, m_triangle_vertices.data(), m_vertice_size * m_triangle_vertices.size());
}

void fan_3d::graphics::terrain_generator::draw() {
	fan::mat4 projection(1);
	projection = fan::math::perspective<fan::mat4>(fan::math::radians(90.f), (f32_t)m_camera->m_window->get_size().x / (f32_t)m_camera->m_window->get_size().y, 0.1f, 10000000.0f);

	fan::mat4 view(1);

	view = m_camera->get_view_matrix();

	fan_3d::graphics::terrain_generator::m_shader.use();
	fan_3d::graphics::terrain_generator::m_shader.set_mat4("projection", projection);
	fan_3d::graphics::terrain_generator::m_shader.set_mat4("view", view);
	fan_3d::graphics::terrain_generator::m_shader.set_int("triangle_size", m_triangle_size);
	fan_3d::graphics::terrain_generator::m_shader.set_vec3("light_position", m_camera->get_position());
	fan_3d::graphics::terrain_generator::m_shader.set_vec3("view_position",  m_camera->get_position());

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, m_texture);

	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glBindVertexArray(m_vao);
	glDrawElements(GL_TRIANGLE_STRIP, fan_3d::graphics::terrain_generator::size(), GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);
	//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	//lv.draw();

}

void fan_3d::graphics::terrain_generator::erase_all()
{
	fan_3d::graphics::terrain_generator::m_triangle_vertices.clear();
	basic_shape_color_vector::m_color.clear();
	basic_shape_color_vector::write_data();
	fan::write_glbuffer(m_vertices_vbo, m_triangle_vertices.data(), m_vertice_size * m_triangle_vertices.size());
}

uint_t fan_3d::graphics::terrain_generator::size() {
	return fan_3d::graphics::terrain_generator::m_indices.size();
}

fan_3d::graphics::rectangle_vector::rectangle_vector(fan::camera* camera, const std::string& path, uint_t block_size)
	: basic_shape(camera, fan::shader(fan_3d::graphics::shader_paths::shape_vector_vs, fan_3d::graphics::shader_paths::shape_vector_fs)),
	block_size(block_size)
{
	glBindVertexArray(vao_handler::m_buffer_object);

	rectangle_vector::basic_shape_position::initialize_buffers(true);
	rectangle_vector::basic_shape_size::initialize_buffers(true);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	generate_textures(path, block_size);
}

//fan_3d::graphics::rectangle_vector::rectangle_vector(fan::camera* camera, const fan::color& color, uint_t block_size)
//	: basic_shape(camera, fan::shader(fan_3d::graphics::shader_paths::shape_vector_vs, fan_3d::graphics::shader_paths::shape_vector_fs)),
//	block_size(block_size)
//{
//	glBindVertexArray(m_vao);
//
//	rectangle_vector::basic_shape_color_vector::initialize_buffers(true);
//	rectangle_vector::basic_shape_position::initialize_buffers(true);
//	rectangle_vector::basic_shape_size::initialize_buffers(true);
//
//	glBindBuffer(GL_ARRAY_BUFFER, 0);
//	glBindVertexArray(0);
//	//TODO
//	//generate_textures(path, block_size);
//}

fan_3d::graphics::rectangle_vector::~rectangle_vector()
{
	fan_validate_buffer(m_texture_ssbo, glDeleteBuffers(1, &m_texture_ssbo));
	fan_validate_buffer(m_texture_id_ssbo, glDeleteBuffers(1, &m_texture_id_ssbo));
}

void fan_3d::graphics::rectangle_vector::push_back(const fan::vec3& src, const fan::vec3& dst, const fan::vec2& texture_id, bool queue)
{
	basic_shape::basic_push_back(src, dst, queue);

	this->m_textures.emplace_back(((m_amount_of_textures.y - 1) - texture_id.y) * m_amount_of_textures.x + texture_id.x);

	if (!queue) {
		this->write_textures();
	}
}

fan::vec3 fan_3d::graphics::rectangle_vector::get_src(uint_t i) const
{
	return this->m_position[i];
}

fan::vec3 fan_3d::graphics::rectangle_vector::get_dst(uint_t i) const
{
	return this->m_size[i];
}

fan::vec3 fan_3d::graphics::rectangle_vector::get_size(uint_t i) const
{
	return this->get_dst(i) - this->get_src(i);
}

void fan_3d::graphics::rectangle_vector::set_position(uint_t i, const fan::vec3& src, const fan::vec3& dst, bool queue)
{
	this->m_position[i] = src;
	this->m_size[i] = dst;

	if (!queue) {
		rectangle_vector::basic_shape::write_data(true, true);
	}
}

void fan_3d::graphics::rectangle_vector::set_size(uint_t i, const fan::vec3& size, bool queue)
{
	rectangle_vector::basic_shape::set_size(i, this->get_src(i) + size, queue);
}

// make sure glEnable(GL_DEPTH_TEST) and glDepthFunc(GL_ALWAYS) is set
void fan_3d::graphics::rectangle_vector::draw() {

	fan::mat4 projection(1);
	projection = fan::math::perspective<fan::mat4>(fan::math::radians(90.f), (f32_t)m_camera->m_window->get_size().x / (f32_t)m_camera->m_window->get_size().y, 0.1f, 10000.0f);

	fan::mat4 view(m_camera->get_view_matrix());

	this->m_shader.use();
	this->m_shader.set_mat4("projection", projection);
	this->m_shader.set_mat4("view", view);
	
	this->m_shader.set_bool("texture_sampler", 0);

	this->m_shader.set_vec3("player_position", m_camera->get_position());

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture_handler::m_buffer_object);

	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_texture_ssbo);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_texture_id_ssbo);
	//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	basic_shape::basic_draw(GL_TRIANGLES, 36, size());
	//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void fan_3d::graphics::rectangle_vector::set_texture(uint_t i, const fan::vec2& texture_id, bool queue)
{
	this->m_textures[i] = (f_t)block_size.x / 6 * texture_id.y + texture_id.x;

	if (!queue) {
		write_textures();
	}
}

void fan_3d::graphics::rectangle_vector::generate_textures(const std::string& path, const fan::vec2& block_size)
{
	glGenBuffers(1, &m_texture_ssbo);
	glGenBuffers(1, &m_texture_id_ssbo);

	auto info = fan_2d::graphics::load_image(path, true);

	glBindTexture(GL_TEXTURE_2D, texture_handler::m_buffer_object = info.texture_id);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glGenerateMipmap(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, 0);

	const fan::vec2 texturepack_size = fan::vec2(info.size.x / block_size.x, info.size.y / block_size.y);
	m_amount_of_textures = fan::vec2(texturepack_size.x / 6, texturepack_size.y);
	// 0 = up, 1 = down, 2 = front, 3 = right, 4 = back, 5 = left

	constexpr int side_order[] = { 0, 1, 2, 3, 4, 5 };
	std::vector<fan::vec2> textures;
	for (fan::vec2i m_texture; m_texture.y < m_amount_of_textures.y; m_texture.y++) {
		const fan::vec2 begin(1.f / texturepack_size.x, 1.f / texturepack_size.y);
		const float up = 1 - begin.y * m_texture.y;
		const float down = 1 - begin.y * (m_texture.y + 1);
		for (m_texture.x = 0; m_texture.x < m_amount_of_textures.x; m_texture.x++) {
			for (uint_t side = 0; side < std::size(side_order); side++) {
				const float left = begin.x * side_order[side] + ((begin.x * (m_texture.x)) * 6);
				const float right = begin.x * (side_order[side] + 1) + ((begin.x * (m_texture.x)) * 6);
				const fan::vec2 texture_coordinates[] = {
					fan::vec2(left,  up),
					fan::vec2(left,  down),
					fan::vec2(right, down),
					fan::vec2(right, down),
					fan::vec2(right, up),
					fan::vec2(left,  up)
				};
				for (auto coordinate : texture_coordinates) {
					textures.emplace_back(coordinate);
				}
			}
		}
	}

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_texture_ssbo);
	glBufferData(
		GL_SHADER_STORAGE_BUFFER,
		sizeof(textures[0]) * textures.size(),
		textures.data(),
		GL_STATIC_DRAW
	);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_texture_ssbo);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void fan_3d::graphics::rectangle_vector::write_textures()
{
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_texture_id_ssbo);
	glBufferData(
		GL_SHADER_STORAGE_BUFFER,
		sizeof(m_textures[0]) * m_textures.size(),
		m_textures.data(),
		GL_STATIC_DRAW
	);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_texture_id_ssbo);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void fan_3d::graphics::rectangle_vector::release_queue(bool position, bool size, bool textures)
{
	if (position) {
		basic_shape::write_data(true, false);
	}
	if (size) {
		basic_shape::write_data(false, true);
	}
	if (textures) {
		this->write_textures();
	}
}

fan_3d::graphics::square_corners fan_3d::graphics::rectangle_vector::get_corners(uint_t i) const
{

	const fan::vec3 position = fan::da_t<f32_t, 2, 3>{ this->get_position(i), this->get_size(i) }.avg();
	const fan::vec3 size = this->get_size(i);
	const fan::vec3 half_size = size * 0.5;

	return square_corners{ 
		{ 
			{ position[0] + half_size[0], position[1] + half_size[1], position[2] + half_size[2] }, 
		{ position[0] - half_size[0], position[1] + half_size[1], position[2] + half_size[2] }, 
		{ position[0] + half_size[0], position[1] + half_size[1], position[2] - half_size[2] }, 
		{ position[0] - half_size[0], position[1] + half_size[1], position[2] - half_size[2] } 
		}, // left
		{ 
			{ position[0] - half_size[0], position[1] - half_size[1], position[2] + half_size[2] }, 
		{ position[0] + half_size[0], position[1] - half_size[1], position[2] + half_size[2] }, 
		{ position[0] - half_size[0], position[1] - half_size[1], position[2] - half_size[2] }, 
		{ position[0] + half_size[0], position[1] - half_size[1], position[2] - half_size[2] } 
		}, // right
		{ 
			{ position[0] - half_size[0], position[1] + half_size[1], position[2] + half_size[2] }, 
		{ position[0] - half_size[0], position[1] - half_size[1], position[2] + half_size[2] }, 
		{ position[0] - half_size[0], position[1] + half_size[1], position[2] - half_size[2] }, 
		{ position[0] - half_size[0], position[1] - half_size[1], position[2] - half_size[2] } 
		}, // front
		{ 
			{ position[0] + half_size[0], position[1] - half_size[1], position[2] + half_size[2] }, 
		{ position[0] + half_size[0], position[1] + half_size[1], position[2] + half_size[2] }, 
		{ position[0] + half_size[0], position[1] - half_size[1], position[2] - half_size[2] }, 
		{ position[0] + half_size[0], position[1] + half_size[1], position[2] - half_size[2] } 
		}, // back
		{ 
			{ position[0] + half_size[0], position[1] + half_size[1], position[2] + half_size[2] }, 
		{ position[0] + half_size[0], position[1] - half_size[1], position[2] + half_size[2] }, 
		{ position[0] - half_size[0], position[1] + half_size[1], position[2] + half_size[2] }, 
		{ position[0] - half_size[0], position[1] - half_size[1], position[2] + half_size[2] } 
		}, // top
		{ 
			{ position[0] + half_size[0], position[1] - half_size[1], position[2] + half_size[2] }, 
		{ position[0] + half_size[0], position[1] + half_size[1], position[2] + half_size[2] }, 
		{ position[0] - half_size[0], position[1] - half_size[1], position[2] + half_size[2] }, 
		{ position[0] - half_size[0], position[1] + half_size[1], position[2] + half_size[2] } 
		}  // bottom
	};
}

uint_t fan_3d::graphics::rectangle_vector::size() const
{
	return this->m_position.size();
}

fan_3d::graphics::skybox::skybox(
	fan::camera* camera,
	const std::string& left,
	const std::string& right,
	const std::string& front,
	const std::string back,
	const std::string bottom,
	const std::string& top
) : m_shader(fan_3d::graphics::shader_paths::skybox_vs, fan_3d::graphics::shader_paths::skybox_fs), m_camera(camera) {

//	throw std::runtime_error("nope");

	std::array<std::string, 6> images{ right, left, top, bottom, back, front };

	for (uint_t i = 0; i < images.size(); i++) {
		if (!fan::io::file::exists(images[i])) {
			fan::print("path does not exist:", images[i]);
			exit(1);
		}
	}

	glGenTextures(1, &m_texture_id);

	glBindTexture(GL_TEXTURE_CUBE_MAP, m_texture_id);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_LINEAR);

	for (uint_t i = 0; i < images.size(); i++) {
		fan::vec2i size;
		//unsigned char* image = SOIL_load_image(images[i].c_str(), image_size.data(), image_size.data() + 1, 0, SOIL_LOAD_AUTO);
		//glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB, image_size.x, image_size.y, 0, GL_RGB, GL_UNSIGNED_BYTE, image);
		//SOIL_free_image_data(image);
	}

	glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

	glGenVertexArrays(1, &m_skybox_vao);
	glBindVertexArray(0);
}

fan_3d::graphics::skybox::~skybox() {
	fan_validate_buffer(m_skybox_vao, glDeleteVertexArrays(1, &m_skybox_vao));
	fan_validate_buffer(m_texture_id, glDeleteTextures(1, &m_texture_id));
}

void fan_3d::graphics::skybox::draw() {

	fan::mat4 view(1);
	view = m_camera->get_view_matrix();

	fan::mat4 projection = fan::math::perspective<fan::mat4>(fan::math::radians(90.f), (f32_t)m_camera->m_window->get_size().x / (f32_t)m_camera->m_window->get_size().y, 0.1, 100.0);


	m_shader.use();

	view[3][0] = 0;
	view[3][1] = 0;
	view[3][2] = 0;

	m_shader.set_mat4("view", view);
	m_shader.set_mat4("projection", projection);
	//	m_shader.set_vec3("fog_color", fan::vec3(220.f / 255.f, 219.f / 255.f, 223.f / 255.f));

	glDepthFunc(GL_LEQUAL);
	glBindVertexArray(m_skybox_vao);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_CUBE_MAP, m_texture_id);
	glDrawArrays(GL_TRIANGLES, 0, 36);
	glBindVertexArray(0);
	glDepthFunc(GL_LESS);
}

fan_3d::graphics::model_mesh::model_mesh(
	const std::vector<mesh_vertex>& vertices,
	const std::vector<unsigned int>& indices,
	const std::vector<mesh_texture>& textures
) : vertices(vertices), indices(indices), textures(textures) {
	initialize_mesh();
}

void fan_3d::graphics::model_mesh::initialize_mesh() {
	glGenVertexArrays(1, &vao);
	glGenBuffers(1, &vbo);
	glGenBuffers(1, &ebo);

	glBindVertexArray(vao);

	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(mesh_vertex), &vertices[0], GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), &indices[0], GL_STATIC_DRAW);

	glEnableVertexAttribArray(3);
	glVertexAttribPointer(3, 3, fan::GL_FLOAT_T, GL_FALSE, sizeof(mesh_vertex), 0);

	glEnableVertexAttribArray(4);
	glVertexAttribPointer(4, 3, fan::GL_FLOAT_T, GL_FALSE, sizeof(mesh_vertex), reinterpret_cast<void*>(offsetof(mesh_vertex, normal)));

	glEnableVertexAttribArray(5);
	glVertexAttribPointer(5, 2, fan::GL_FLOAT_T, GL_FALSE, sizeof(mesh_vertex), reinterpret_cast<void*>(offsetof(mesh_vertex, texture_coordinates)));
	glBindVertexArray(0);
}

//fan_3d::graphics::model_loader::model_loader(const std::string& path, const fan::vec3& size) {
//	load_model(path, size);
//}

//void fan_3d::graphics::model_loader::load_model(const std::string& path, const fan::vec3& size) {
//	Assimp::Importer importer;
//	const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs);
//
//	if (scene == nullptr || scene->mFlags == AI_SCENE_FLAGS_INCOMPLETE || scene->mRootNode == nullptr) {
//		std::cout << "assimp error: " << importer.GetErrorString() << '\n';
//		return;
//	}
//
//	directory = path.substr(0, path.find_last_of('/'));
//
//	process_node(scene->mRootNode, scene, size);
//}

//void fan_3d::graphics::model_loader::process_node(aiNode* node, const aiScene* scene, const fan::vec3& size) {
//	for (GLuint i = 0; i < node->mNumMeshes; i++) {
//		aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
//
//		meshes.emplace_back(process_mesh(mesh, scene, size));
//	}
//
//	for (GLuint i = 0; i < node->mNumChildren; i++) {
//		process_node(node->mChildren[i], scene, size);
//	}
//}

//fan_3d::graphics::model_mesh fan_3d::graphics::model_loader::process_mesh(aiMesh* mesh, const aiScene* scene, const fan::vec3& size) {
//	std::vector<mesh_vertex> vertices;
//	std::vector<GLuint> indices;
//	std::vector<mesh_texture> textures;
//
//	for (GLuint i = 0; i < mesh->mNumVertices; i++)
//	{
//		mesh_vertex vertex;
//		fan::vec3 vector;
//
//		vector.x = mesh->mVertices[i].x / 2 * size.x;
//		vector.y = mesh->mVertices[i].y / 2 * size.y;
//		vector.z = mesh->mVertices[i].z / 2 * size.z;
//		vertex.position = vector;
//		if (mesh->mNormals != nullptr) {
//			vector.x = mesh->mNormals[i].x;
//			vector.y = mesh->mNormals[i].y;
//			vector.z = mesh->mNormals[i].z;
//			vertex.normal = vector;
//		}
//		else {
//			vertex.normal = fan::vec3();
//		}
//
//		if (mesh->mTextureCoords[0]) {
//			fan::vec2 vec;
//			vec.x = mesh->mTextureCoords[0][i].x;
//			vec.y = mesh->mTextureCoords[0][i].y;
//			vertex.texture_coordinates = vec;
//		}
//
//		vertices.emplace_back(vertex);
//	}
//
//	for (GLuint i = 0; i < mesh->mNumFaces; i++) {
//		aiFace face = mesh->mFaces[i];
//		for (uint_t j = 0; j < face.mNumIndices; j++) {
//			indices.emplace_back(face.mIndices[j]);
//		}
//	}
//
//	aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
//
//	std::vector<mesh_texture> diffuseMaps = this->load_material_textures(material, aiTextureType_DIFFUSE, "texture_diffuse");
//	textures.insert(textures.end(), diffuseMaps.begin(), diffuseMaps.end());
//
//	std::vector<mesh_texture> specularMaps = this->load_material_textures(material, aiTextureType_SPECULAR, "texture_specular");
//	textures.insert(textures.end(), specularMaps.begin(), specularMaps.end());
//
//	if (textures.empty()) {
//		mesh_texture m_texture;
//		unsigned int texture_id;
//		glGenTextures(1, &texture_id);
//
//		aiColor4D color(0.f, 0.f, 0.f, 0.f);
//		aiGetMaterialColor(material, AI_MATKEY_COLOR_DIFFUSE, &color);
//		std::vector<unsigned char> pixels;
//		pixels.emplace_back(color.r * 255.f);
//		pixels.emplace_back(color.g * 255.f);
//		pixels.emplace_back(color.b * 255.f);
//		pixels.emplace_back(color.a * 255.f);
//
//		glBindTexture(GL_TEXTURE_2D, texture_id);
//		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
//		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
//		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
//		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
//
//		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
//		glGenerateMipmap(GL_TEXTURE_2D);
//
//		glBindTexture(GL_TEXTURE_2D, 0);
//
//		m_texture.id = texture_id;
//		textures.emplace_back(m_texture);
//		textures_loaded.emplace_back(m_texture);
//	}
//	return model_mesh(vertices, indices, textures);
//}

//std::vector<fan_3d::graphics::mesh_texture> fan_3d::graphics::model_loader::load_material_textures(aiMaterial* mat, aiTextureType type, const std::string& type_name) {
//	std::vector<mesh_texture> textures;
//
//	for (uint_t i = 0; i < mat->GetTextureCount(type); i++) {
//		aiString a_str;
//		mat->GetTexture(type, i, &a_str);
//		bool skip = false;
//		for (const auto& j : textures_loaded) {
//			if (j.path == a_str) {
//				textures.emplace_back(j);
//				skip = true;
//				break;
//			}
//		}
//
//		if (!skip) {
//			mesh_texture m_texture;
//			m_texture.id = load_texture(a_str.C_Str(), directory, false);
//			m_texture.type = type_name;
//			m_texture.path = a_str;
//			textures.emplace_back(m_texture);
//			textures_loaded.emplace_back(m_texture);
//		}
//	}
//	return textures;
//}

//fan_3d::graphics::model::model(fan::camera* camera) : model_loader("", fan::vec3()), m_camera(camera), m_shader(fan_3d::graphics::shader_paths::model_vs, fan_3d::graphics::shader_paths::model_fs) {}
//
//fan_3d::graphics::model::model(fan::camera* camera, const std::string& path, const fan::vec3& position, const fan::vec3& size)
//	: model_loader(path, size / 2.f), m_camera(camera), m_shader(fan_3d::graphics::shader_paths::model_vs, fan_3d::graphics::shader_paths::model_fs),
//	m_position(position), m_size(size)
//{
//	for (uint_t i = 0; i < this->meshes.size(); i++) {
//		glBindVertexArray(this->meshes[i].vao);
//	}
//	glBindVertexArray(0);
//}
//
//void fan_3d::graphics::model::draw() {
//
//	fan::mat4 model(1);
//	model = translate(model, get_position());
//	model = scale(model, get_size());
//
//	this->m_shader.use();
//
//	fan::mat4 projection(1);
//	projection = fan::perspective<fan::mat4>(fan::radians(90.f), (f32_t)m_camera->m_window->get_size().x / (f32_t)m_camera->m_window->get_size().y, 0.1f, 1000.0f);
//
//	fan::mat4 view(m_camera->get_view_matrix());
//
//	this->m_shader.set_int("texture_sampler", 0);
//	this->m_shader.set_mat4("projection", projection);
//	this->m_shader.set_mat4("view", view);
//	this->m_shader.set_vec3("light_position", m_camera->get_position());
//	this->m_shader.set_vec3("view_position",m_camera->get_position());
//	this->m_shader.set_vec3("light_color", fan::vec3(1, 1, 1));
//	this->m_shader.set_int("texture_diffuse", 0);
//	this->m_shader.set_mat4("model", model);
//
//	//_Shader.set_vec3("sky_color", fan::vec3(220.f / 255.f, 219.f / 255.f, 223.f / 255.f));
//	glActiveTexture(GL_TEXTURE0);
//	glBindTexture(GL_TEXTURE_2D, this->textures_loaded[0].id);
//
//	glDepthFunc(GL_LEQUAL);
//	for (uint_t i = 0; i < this->meshes.size(); i++) {
//		glBindVertexArray(this->meshes[i].vao);
//		glDrawElementsInstanced(GL_TRIANGLES, (GLsizei)this->meshes[i].indices.size(), GL_UNSIGNED_INT, 0, 1);
//	}
//	glDepthFunc(GL_LESS);
//
//	glBindVertexArray(0);
//}
//
//fan::vec3 fan_3d::graphics::model::get_position()
//{
//	return this->m_position;
//}
//
//void fan_3d::graphics::model::set_position(const fan::vec3& position)
//{
//	this->m_position = position;
//}
//
//fan::vec3 fan_3d::graphics::model::get_size()
//{
//	return this->m_size;
//}
//
//void fan_3d::graphics::model::set_size(const fan::vec3& size)
//{
//	this->m_size = size;
//}

fan::vec3 line_triangle_intersection(const fan::vec3& ray_begin, const fan::vec3& ray_end, const fan::vec3& p0, const fan::vec3& p1, const fan::vec3& p2) {

	const auto lab = (ray_begin + ray_end) - ray_begin;

	const auto p01 = p1 - p0;
	const auto p02 = p2 - p0;

	const auto normal = fan::math::cross(p01, p02);

	const auto t = fan_3d::math::dot(normal, ray_begin - p0) / fan_3d::math::dot(-lab, normal);
	const auto u = fan_3d::math::dot(fan::math::cross(p02, -lab), ray_begin - p0) / fan_3d::math::dot(-lab, normal);
	const auto v = fan_3d::math::dot(fan::math::cross(-lab, p01), ray_begin - p0) / fan_3d::math::dot(-lab, normal);

	if (t >= 0 && t <= 1 && u >= 0 && u <= 1 && v >= 0 && v <= 1 && (u + v) <= 1) {
		return ray_begin + lab * t;
	}

	return INFINITY;

}

fan::vec3 fan_3d::graphics::line_triangle_intersection(const fan::da_t<f32_t, 2, 3>& line, const fan::da_t<f32_t, 3, 3>& triangle) {

	const auto lab = (line[0] + line[1]) - line[0];

	const auto p01 = triangle[1] - triangle[0];
	const auto p02 = triangle[2] - triangle[0];

	const auto normal = fan::math::cross(p01, p02);

	const auto t = fan_3d::math::dot(normal, line[0] - triangle[0]) / fan_3d::math::dot(-lab, normal);
	const auto u = fan_3d::math::dot(fan::math::cross(p02, -lab), line[0] - triangle[0]) / fan_3d::math::dot(-lab, normal);
	const auto v = fan_3d::math::dot(fan::math::cross(-lab, p01), line[0] - triangle[0]) / fan_3d::math::dot(-lab, normal);

	if (t >= 0 && t <= 1 && u >= 0 && u <= 1 && v >= 0 && v <= 1 && (u + v) <= 1) {
		return line[0] + lab * t;
	}

	return INFINITY;
}

fan::vec3 fan_3d::graphics::line_plane_intersection(const fan::da_t<f32_t, 2, 3>& line, const fan::da_t<f32_t, 4, 3>& square) {
	const fan::da_t<f32_t, 3> plane_normal = fan::math::normalize_no_sqrt(fan::math::cross(square[3] - square[2], square[0] - square[2]));
	const f32_t nl_dot(math::dot(plane_normal, line[1]));

	if (!nl_dot) {
		return fan::vec3(INFINITY);
	}

	const f32_t ray_length = fan_3d::math::dot(square[2] - line[0], plane_normal) / nl_dot;
	if (ray_length <= 0) {
		return fan::vec3(INFINITY);
	}
	if (fan::math::custom_pythagorean_no_sqrt(fan::vec3(line[0]), fan::vec3(line[0] + line[1])) < ray_length) {
		return fan::vec3(INFINITY);
	}
	const fan::vec3 intersection(line[0] + line[1] * ray_length);

	auto result = fan_3d::math::dot((square[2] - line[0]), plane_normal);
	fan::print(result);
	if (!result) {
		fan::print("on plane");
	}

	if (intersection[1] >= square[3][1] && intersection[1] <= square[0][1] &&
		intersection[2] >= square[3][2] && intersection[2] <= square[0][2])
	{
		return intersection;
	}
	return fan::vec3(INFINITY);
}