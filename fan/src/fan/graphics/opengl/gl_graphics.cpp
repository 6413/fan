#include <fan/graphics/renderer.hpp>

#define fan_assert_if_same_path_loaded_multiple_times 0

#if fan_renderer == fan_renderer_opengl

#include <fan/graphics/graphics.hpp>

#include <functional>
#include <numeric>

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

static constexpr auto zoom = 1; 

fan::mat4 fan_2d::graphics::get_projection(const fan::vec2i& window_size) {

	return fan::math::ortho<fan::mat4>((f32_t)window_size.x * 0.5 * zoom, ((f32_t)window_size.x + (f32_t)window_size.x * 0.5) * zoom, ((f32_t)window_size.y + (f32_t)window_size.y * 0.5) * zoom, ((f32_t)window_size.y * 0.5) * zoom, -1, 1000.0f);
}

fan::mat4 fan_2d::graphics::get_view_translation(const fan::vec2i& window_size, const fan::mat4& view)
{
	return fan::math::translate(view, fan::vec3((f_t)window_size.x * 0.5, (f_t)window_size.y * 0.5, -700.0f));
}

//fan_2d::graphics::image_t fan_2d::graphics::load_image(fan::window* window, const std::string& path)
//{
//	fan_2d::graphics::image_t info = new std::remove_pointer<fan_2d::graphics::image_t>::type(window);
//
//	glGenTextures(1, &info->texture);
//	glBindTexture(GL_TEXTURE_2D, info->texture);
//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, fan_2d::graphics::image_load_properties::visual_output);
//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, fan_2d::graphics::image_load_properties::visual_output);
//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR );
//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, fan_2d::graphics::image_load_properties::filter);
//
//	auto image_data = fan::image_loader::load_image(path);
//
//	info->size = image_data.size;
//
//	uintptr_t internal_format = 0, format = 0, type = 0;
//
//	switch (image_data.format) {
//		case AVPixelFormat::AV_PIX_FMT_BGR24:
//		{
//			glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
//			internal_format = GL_RGB;
//			format = GL_BGR_EXT;
//			type = GL_UNSIGNED_BYTE;
//			break;
//		}
//		case AVPixelFormat::AV_PIX_FMT_RGB24:
//		{
//			//	glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
//			internal_format = GL_RGB;
//			format = GL_RGB;
//			type = GL_UNSIGNED_BYTE;
//			break;
//		}
//		case AVPixelFormat::AV_PIX_FMT_RGBA:
//		{
//		//	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
//			internal_format = GL_RGBA;
//			format = GL_RGBA;
//			type = GL_UNSIGNED_BYTE;
//			break;
//		}
//		//case AVPixelFormat::AV_PIX_FMT_BGRA:
//	}
//
//	info->properties.filter = fan_2d::graphics::image_load_properties::filter;
//	info->properties.format = format;
//	info->properties.internal_format = internal_format;
//	info->properties.type = type;
//	info->properties.visual_output = fan_2d::graphics::image_load_properties::visual_output;
//
//	//glTextureParameterf(GL_TEXTURE_2D, GL_TEXTURE_LOD_BIAS, 100);
//	glTexImage2D(GL_TEXTURE_2D, 0, internal_format, info->size.x, info->size.y, 0, format, type, image_data.data[0]);
//
//	glGenerateMipmap(GL_TEXTURE_2D);
//
//	delete[] image_data.data[0];
//
////	av_freep((void*)&image_data.data);
//
//	glBindTexture(GL_TEXTURE_2D, 0);
//
//	return info;
//}

//fan_2d::graphics::image_t fan_2d::graphics::load_image(fan::window* window, const pixel_data_t& pixel_data)
//{
//	fan_2d::graphics::image_t info = new std::remove_pointer<fan_2d::graphics::image_t>::type(window);
//
//	glGenTextures(1, &info->texture);
//
//	glBindTexture(GL_TEXTURE_2D, info->texture);
//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, fan_2d::graphics::image_load_properties::visual_output);
//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, fan_2d::graphics::image_load_properties::visual_output);
//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, fan_2d::graphics::image_load_properties::filter);
//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, fan_2d::graphics::image_load_properties::filter);
//
//	fan::image_loader::image_data image_d;
//
//	for (int i = 0; i < std::size(image_d.data); i++) {
//
//		image_d.data[i] = pixel_data.pixels[i];
//		image_d.linesize[i] = pixel_data.linesize[i];
//	}
//
//	image_d.size = pixel_data.size;
//	image_d.format = pixel_data.format;
//
//	auto image_data = fan::image_loader::convert_format(image_d, AV_PIX_FMT_RGBA);
//
//	info->size = image_data.size;
//
//	uintptr_t internal_format = 0, format = 0, type = 0;
//
//	switch (image_data.format) {
//		case AVPixelFormat::AV_PIX_FMT_BGR24:
//		{
//			glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
//			internal_format = GL_RGB;
//			format = GL_BGR_EXT;
//			type = GL_UNSIGNED_BYTE;
//			break;
//		}
//		case AVPixelFormat::AV_PIX_FMT_RGB24:
//		{
//			//glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
//			internal_format = GL_RGB;
//			format = GL_RGB;
//			type = GL_UNSIGNED_BYTE;
//			break;
//		}
//		case AVPixelFormat::AV_PIX_FMT_RGBA:
//		{
//			//glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
//			internal_format = GL_RGBA;
//			format = GL_RGBA;
//			type = GL_UNSIGNED_BYTE;
//			break;
//		}
//		//case AVPixelFormat::AV_PIX_FMT_BGRA:
//	}
//
//	glTexImage2D(GL_TEXTURE_2D, 0, internal_format, info->size.x, info->size.y, 0, format, type, image_data.data[0]);
//
//	delete[] image_data.data[0];
//
//	glGenerateMipmap(GL_TEXTURE_2D);
//
//	glBindTexture(GL_TEXTURE_2D, 0);
//
//	return info;
//}

// webp
fan_2d::graphics::image_t fan_2d::graphics::load_image(fan::window* window, const std::string& path)
{
#if fan_assert_if_same_path_loaded_multiple_times

	static std::unordered_map<std::string, bool> existing_images;

	if (existing_images.find(path) != existing_images.end()) {
		fan::throw_error("image already existing " + path);
	}

	existing_images[path] = 0;

#endif

	fan_2d::graphics::image_t info = new std::remove_pointer<fan_2d::graphics::image_t>::type(window);

	auto image = fan::webp::load_image(path);

	glGenTextures(1, &info->texture);

	glBindTexture(GL_TEXTURE_2D, info->texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, fan_2d::graphics::image_load_properties::visual_output);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, fan_2d::graphics::image_load_properties::visual_output);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, fan_2d::graphics::image_load_properties::filter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, fan_2d::graphics::image_load_properties::filter);

	uintptr_t internal_format = 0, format = 0, type = 0;

	internal_format = GL_RGBA;
	format = GL_RGBA;
	type = GL_UNSIGNED_BYTE;

	info->size = image.size;

	glTexImage2D(GL_TEXTURE_2D, 0, internal_format, info->size.x, info->size.y, 0, format, type, image.data);

	fan::webp::free_image(image.data);

	glGenerateMipmap(GL_TEXTURE_2D);

	glBindTexture(GL_TEXTURE_2D, 0);

	return info;
}

// webp
fan_2d::graphics::image_t fan_2d::graphics::load_image(fan::window* window, const fan::webp::image_info_t& image_info)
{
	fan_2d::graphics::image_t info = new std::remove_pointer<fan_2d::graphics::image_t>::type(window);

	glGenTextures(1, &info->texture);

	glBindTexture(GL_TEXTURE_2D, info->texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, fan_2d::graphics::image_load_properties::visual_output);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, fan_2d::graphics::image_load_properties::visual_output);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, fan_2d::graphics::image_load_properties::filter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, fan_2d::graphics::image_load_properties::filter);

	uintptr_t internal_format = 0, format = 0, type = 0;

	internal_format = GL_RGBA;
	format = GL_RGBA;
	type = GL_UNSIGNED_BYTE;

	info->size = image_info.size;

	glTexImage2D(GL_TEXTURE_2D, 0, internal_format, info->size.x, info->size.y, 0, format, type, image_info.data);

	fan::webp::free_image(image_info.data);

	glGenerateMipmap(GL_TEXTURE_2D);

	glBindTexture(GL_TEXTURE_2D, 0);

	return info;
}

fan_2d::graphics::lighting_properties::lighting_properties(fan::shader_t* shader)
	: m_lighting_shader(shader), m_lighting_on(false), m_world_light_strength(0.4)
{
	set_lighting(m_lighting_on);
	set_world_light_strength(m_world_light_strength);
}

bool fan_2d::graphics::lighting_properties::get_lighting() const
{
	return m_lighting_on;
}

void fan_2d::graphics::lighting_properties::set_lighting(bool value)
{
	m_lighting_on = value;
	(*m_lighting_shader)->use();
	(*m_lighting_shader)->set_bool("enable_lighting", m_lighting_on);
}

f32_t fan_2d::graphics::lighting_properties::get_world_light_strength() const
{
	return m_world_light_strength;
}

void fan_2d::graphics::lighting_properties::set_world_light_strength(f32_t value)
{
	m_world_light_strength = value;
	(*m_lighting_shader)->use();
	(*m_lighting_shader)->set_float("world_light_strength", m_world_light_strength);
}

fan_2d::graphics::vertice_vector::vertice_vector(fan::camera* camera)
	: basic_vertice_vector(camera), m_offset(0), m_queue_helper(camera->m_window)
{
	this->initialize_buffers();
}

fan_2d::graphics::vertice_vector::vertice_vector(const vertice_vector& vector)
	: fan::basic_vertice_vector<fan::vec2>(vector), m_queue_helper(vector.m_camera->m_window) {

	this->m_offset = vector.m_offset;

	this->initialize_buffers();
}

fan_2d::graphics::vertice_vector::vertice_vector(vertice_vector&& vector) noexcept
	: fan::basic_vertice_vector<fan::vec2>(std::move(vector)), m_queue_helper(vector.m_camera->m_window) { 

	this->m_offset = vector.m_offset;
}

fan_2d::graphics::vertice_vector& fan_2d::graphics::vertice_vector::operator=(const vertice_vector& vector)
{
	this->m_offset = vector.m_offset;

	basic_vertice_vector::operator=(vector);

	this->initialize_buffers();

	return *this;
}

fan_2d::graphics::vertice_vector& fan_2d::graphics::vertice_vector::operator=(vertice_vector&& vector) noexcept
{
	this->m_offset = vector.m_offset;

	basic_vertice_vector::operator=(std::move(vector));

	return *this;
}

void fan_2d::graphics::vertice_vector::push_back(const vertice_vector::properties_t& properties)
{
	vertice_vector::basic_vertice_vector::basic_shape_position::push_back(properties.position);
	vertice_vector::basic_vertice_vector::basic_shape_color_vector::push_back(properties.color);
	vertice_vector::basic_vertice_vector::basic_shape_angle::push_back(properties.angle);
	vertice_vector::basic_vertice_vector::basic_shape_rotation_point::push_back(properties.rotation_point);
	vertice_vector::basic_vertice_vector::basic_shape_rotation_vector::push_back(properties.rotation_vector);

	m_queue_helper.write([&] {
		this->write_data();
	});
	
}

void fan_2d::graphics::vertice_vector::reserve(uintptr_t size)
{
	vertice_vector::basic_vertice_vector::reserve(size);

	m_queue_helper.write([&] {
		this->write_data();
	});
	
}

void fan_2d::graphics::vertice_vector::resize(uintptr_t size, const fan::color& color)
{
	vertice_vector::basic_vertice_vector::resize(size, color);

	m_queue_helper.write([&] {
		this->write_data();
	});
	
}

void fan_2d::graphics::vertice_vector::draw(fan_2d::graphics::shape shape, uint32_t single_draw_amount, uint32_t begin, uint32_t end, bool texture) const
{
	fan::mat4 projection(1);
	projection = fan_2d::graphics::get_projection(m_camera->m_window->get_size());

	fan::mat4 view(1);
	view = m_camera->get_view_matrix(fan_2d::graphics::get_view_translation(m_camera->m_window->get_size(), view));

	this->m_shader->use();

	this->m_shader->set_mat4("projection", projection);
	this->m_shader->set_mat4("view", view);

	this->m_shader->set_int("enable_texture", texture);

	uint32_t mode = 0;

	switch(shape) {
		case fan_2d::graphics::shape::line: {
			mode = GL_LINES;
			break;
		}
		case fan_2d::graphics::shape::line_strip: {
			mode = GL_LINE_STRIP;
			break;
		}
		case fan_2d::graphics::shape::triangle: {
			mode = GL_TRIANGLES;
			break;
		}
		case fan_2d::graphics::shape::triangle_strip: {
			mode = GL_TRIANGLE_STRIP;
			break;
		}
		case fan_2d::graphics::shape::triangle_fan: {
			mode = GL_TRIANGLE_FAN;
			break;
		}
		default: {
			mode = GL_TRIANGLES;
			fan::print("fan warning - unset input assembly topology in graphics pipeline");
			break;
		}
	}

	if (m_fill_mode == fan_2d::graphics::get_fill_mode(m_camera->m_window)) {
		fan_2d::graphics::draw([&] {
			vertice_vector::basic_vertice_vector::basic_draw(begin, end, mode, size(), single_draw_amount);
		});

		return;
	}

	auto fill_mode = fan_2d::graphics::get_fill_mode(m_camera->m_window);

	fan_2d::graphics::draw_mode(m_camera->m_window, m_fill_mode, fan_2d::graphics::get_face(m_camera->m_window));

	fan_2d::graphics::draw([&] {
		vertice_vector::basic_vertice_vector::basic_draw(begin, end, mode, size(), single_draw_amount);
	});

	fan_2d::graphics::draw_mode(m_camera->m_window, fill_mode, fan_2d::graphics::get_face(m_camera->m_window));
}

fan_2d::graphics::vertice_vector::vertice_vector(fan::camera* camera, bool init)
	: basic_vertice_vector(camera), m_offset(0), m_queue_helper(camera->m_window)
{
}

void fan_2d::graphics::vertice_vector::erase(uintptr_t i)
{
	fan::basic_vertice_vector<fan::vec2>::erase(i);

	m_queue_helper.write([&] {
		this->write_data();
	});
	
}

void fan_2d::graphics::vertice_vector::erase(uintptr_t begin, uintptr_t end)
{
	fan::basic_vertice_vector<fan::vec2>::erase(begin, end);

	m_queue_helper.write([&] {
		this->write_data();
	});
	
}

void fan_2d::graphics::vertice_vector::clear()
{
	vertice_vector::basic_vertice_vector::clear();

	m_queue_helper.write([&] {
		this->write_data();
	});
}

fan::vec2 fan_2d::graphics::vertice_vector::get_position(uint32_t i) const {
	return fan::basic_vertice_vector<fan::vec2>::basic_shape_position::get_value(i);
}
void fan_2d::graphics::vertice_vector::set_position(uint32_t i, const fan::vec2& position) {
	fan::basic_vertice_vector<fan::vec2>::basic_shape_position::set_value(i, position);
	m_queue_helper.edit(i, i + 1, [&] {
		if (m_queue_helper.m_min_edit == -1) {
			return;
		}
		if (m_queue_helper.m_max_edit == -1) {
			return;
		}
		this->edit_data(m_queue_helper.m_min_edit, m_queue_helper.m_max_edit);
	});
}

fan::color fan_2d::graphics::vertice_vector::get_color(uint32_t i) const {
	return fan::basic_vertice_vector<fan::vec2>::basic_shape_color_vector::get_value(i);
}
void fan_2d::graphics::vertice_vector::set_color(uint32_t i, const fan::color& color) {
	fan::basic_vertice_vector<fan::vec2>::basic_shape_color_vector::set_value(i, color);
	m_queue_helper.edit(i, i + 1, [&] {
		this->edit_data(m_queue_helper.m_min_edit, m_queue_helper.m_max_edit);
	});
}

void fan_2d::graphics::vertice_vector::set_angle(uint32_t i, f32_t angle)
{
	vertice_vector::basic_vertice_vector::basic_shape_angle::set_value(i, angle);
	m_queue_helper.edit(i, i + 1, [&] {
		this->edit_data(m_queue_helper.m_min_edit, m_queue_helper.m_max_edit);
	});
}

void fan_2d::graphics::vertice_vector::initialize_buffers()
{
	set_draw_mode(fan_2d::graphics::get_fill_mode(m_camera->m_window));

	fan::bind_vao(vao_handler::m_buffer_object, [&] {
		basic_shape_position::initialize_buffers(m_shader->id, position_location_name, false, fan::vec2::size());
		basic_shape_color_vector::initialize_buffers(m_shader->id, color_location_name, false, fan::color::size());
		basic_vertice_vector::basic_shape_angle::initialize_buffers(m_shader->id, angle_location_name, false, 1);
		basic_vertice_vector::basic_shape_rotation_point::initialize_buffers(m_shader->id, rotation_point_location_name, false, fan::vec2::size());
		basic_vertice_vector::basic_shape_rotation_vector::initialize_buffers(m_shader->id, rotation_vector_location_name, false, fan::vec3::size());
	});
}

void fan_2d::graphics::vertice_vector::enable_draw(fan_2d::graphics::shape shape, uint32_t single_draw_amount)
{
	if (m_draw_index == -1 || m_camera->m_window->m_draw_queue[m_draw_index].first != this) {
		m_draw_index = m_camera->m_window->push_draw_call(this, [&, s = shape, sd = single_draw_amount] {
			this->draw(s, sd);
		});
	}
	else {
		m_camera->m_window->edit_draw_call(m_draw_index, this, [&, s = shape, sd = single_draw_amount] {
			this->draw(s, sd);
		});
	}
}

void fan_2d::graphics::vertice_vector::disable_draw()
{
	if (m_draw_index == -1) {
		return;
	}

	m_camera->m_window->erase_draw_call(m_draw_index);
	m_draw_index = -1;
}

fan_2d::graphics::fill_mode_e fan_2d::graphics::vertice_vector::get_draw_mode() const
{
	return m_fill_mode;
}

void fan_2d::graphics::vertice_vector::set_draw_mode(fan_2d::graphics::fill_mode_e fill_mode)
{
	m_fill_mode = fill_mode;
}

void fan_2d::graphics::vertice_vector::write_data()
{
	fan::basic_vertice_vector<fan::vec2>::basic_shape_position::write_data();
	fan::basic_vertice_vector<fan::vec2>::basic_shape_color_vector::write_data();
	basic_vertice_vector::basic_shape_angle::write_data();
	basic_vertice_vector::basic_shape_rotation_point::write_data();
	basic_vertice_vector::basic_shape_rotation_vector::write_data();

	m_queue_helper.on_write(m_camera->m_window);
}

void fan_2d::graphics::vertice_vector::edit_data(uint32_t i) {
	fan::basic_vertice_vector<fan::vec2>::basic_shape_position::edit_data(i);
	fan::basic_vertice_vector<fan::vec2>::basic_shape_color_vector::edit_data(i);
	basic_vertice_vector::basic_shape_angle::edit_data(i);
	basic_vertice_vector::basic_shape_rotation_point::edit_data(i);
	basic_vertice_vector::basic_shape_rotation_vector::edit_data(i);

	m_queue_helper.on_edit();
}

void fan_2d::graphics::vertice_vector::edit_data(uint32_t begin, uint32_t end) {
	fan::basic_vertice_vector<fan::vec2>::basic_shape_position::edit_data(begin, end);
	fan::basic_vertice_vector<fan::vec2>::basic_shape_color_vector::edit_data(begin, end);
	basic_vertice_vector::basic_shape_angle::edit_data(begin, end);
	basic_vertice_vector::basic_shape_rotation_point::edit_data(begin, end);
	basic_vertice_vector::basic_shape_rotation_vector::edit_data(begin, end);

	m_queue_helper.on_edit();
}

fan_2d::graphics::vertices_sprite::vertices_sprite(fan::camera* camera)
	: fan_2d::graphics::vertice_vector(camera, false)
{
	
  m_shader->set_vertex(
    #include <fan/graphics/glsl/opengl/2D/vertices_sprite.vs>
  );

  m_shader->set_fragment(
    #include <fan/graphics/glsl/opengl/2D/vertices_sprite.fs>
  );

  m_shader->compile();

	this->initialize();
}

void fan_2d::graphics::vertices_sprite::push_back(const vertices_sprite::properties_t& properties) {

	bool write_ = vertice_vector::m_queue_helper.m_write;

	vertice_vector::properties_t p;
	p.position = properties.position;
	p.color = properties.color;
	p.angle = properties.angle;
	p.rotation_point = properties.rotation_point;
	p.rotation_vector = properties.rotation_vector;
	vertice_vector::push_back(p);

	texture_coordinates_t::insert(
		texture_coordinates_t::m_buffer_object.end(),
		properties.texture_coordinates.begin(),
		properties.texture_coordinates.end()
	);

	m_textures.emplace_back(properties.image->texture);

	if (!write_) {
		m_camera->m_window->edit_write_call(m_queue_helper.m_write_index, this, [&] {
			this->write_data();
		});
	}
}

void fan_2d::graphics::vertices_sprite::enable_draw(fan_2d::graphics::shape shape, const std::vector<uint32_t>& single_draw_amount)
{
	if (m_draw_index == -1 || m_camera->m_window->m_draw_queue[m_draw_index].first != this) {
		m_draw_index = m_camera->m_window->push_draw_call(this, [&, s = shape, sd = single_draw_amount] {
			this->draw(s, sd);
		});
	}
	else {
		m_camera->m_window->edit_draw_call(m_draw_index, this, [&, s = shape, sd = single_draw_amount] {
			this->draw(s, sd);
		});
	}
}

void fan_2d::graphics::vertices_sprite::disable_draw()
{
	if (m_draw_index == -1) {
		return;
	}

	m_camera->m_window->erase_draw_call(m_draw_index);
	m_draw_index = -1;
}

void fan_2d::graphics::vertices_sprite::write_data() {
	fan::basic_vertice_vector<fan::vec2>::basic_shape_position::write_data();
	fan::basic_vertice_vector<fan::vec2>::basic_shape_color_vector::write_data();
	basic_vertice_vector::basic_shape_angle::write_data();
	basic_vertice_vector::basic_shape_rotation_point::write_data();
	basic_vertice_vector::basic_shape_rotation_vector::write_data();
	texture_coordinates_t::write_data();

	m_queue_helper.on_write(m_camera->m_window);
}

void fan_2d::graphics::vertices_sprite::draw(fan_2d::graphics::shape shape, const std::vector<uint32_t>& single_draw_amount) {
	fan::mat4 projection(1);
	projection = fan_2d::graphics::get_projection(m_camera->m_window->get_size());

	fan::mat4 view(1);
	view = m_camera->get_view_matrix(fan_2d::graphics::get_view_translation(m_camera->m_window->get_size(), view));

	this->m_shader->use();

	this->m_shader->set_mat4("projection", projection);
	this->m_shader->set_mat4("view", view);

	uint32_t mode = 0;

	switch(shape) {
		case fan_2d::graphics::shape::line: {
			mode = GL_LINES;
			break;
		}
		case fan_2d::graphics::shape::line_strip: {
			mode = GL_LINE_STRIP;
			break;
		}
		case fan_2d::graphics::shape::triangle: {
			mode = GL_TRIANGLES;
			break;
		}
		case fan_2d::graphics::shape::triangle_strip: {
			mode = GL_TRIANGLE_STRIP;
			break;
		}
		case fan_2d::graphics::shape::triangle_fan: {
			mode = GL_TRIANGLE_FAN;
			break;
		}
		default: {
			mode = GL_TRIANGLES;
			fan::print("fan warning - unset input assembly topology in graphics pipeline");
			break;
		}
	}

	if (m_fill_mode == fan_2d::graphics::get_fill_mode(m_camera->m_window)) {
		fan_2d::graphics::draw([&] {
			fan::bind_vao(vao_handler::m_buffer_object, [&] {
				uint32_t offset = 0;
				for (uint32_t j = 0; j < single_draw_amount.size(); j++) {
					m_shader->set_int("texture_sampler", j);
					glActiveTexture(GL_TEXTURE0 + j);
					glBindTexture(GL_TEXTURE_2D, m_textures[offset]);

					glDrawArrays(mode, offset, single_draw_amount[j]);
					offset += single_draw_amount[j];
				}
			});
		});

		return;
	}

	auto fill_mode = fan_2d::graphics::get_fill_mode(m_camera->m_window);

	fan_2d::graphics::draw_mode(m_camera->m_window, m_fill_mode, fan_2d::graphics::get_face(m_camera->m_window));

	fan_2d::graphics::draw([&] {
		fan_2d::graphics::draw([&] {
			fan::bind_vao(vao_handler::m_buffer_object, [&] {
				uint32_t offset = 0;
				for (uint32_t j = 0; j < single_draw_amount.size(); j++) {
					m_shader->set_int("texture_sampler", j);
					glActiveTexture(GL_TEXTURE0 + j);
					glBindTexture(GL_TEXTURE_2D, m_textures[offset]);

					glDrawArrays(mode, offset, single_draw_amount[j]);
					offset += single_draw_amount[j];
				}
			});
		});
	});

	fan_2d::graphics::draw_mode(m_camera->m_window, fill_mode, fan_2d::graphics::get_face(m_camera->m_window));
}

void fan_2d::graphics::vertices_sprite::initialize() {
	set_draw_mode(fan_2d::graphics::get_fill_mode(m_camera->m_window));

	fan::bind_vao(vao_handler::m_buffer_object, [&] {
		basic_shape_position::initialize_buffers(m_shader->id, position_location_name, false, fan::vec2::size());
		basic_shape_color_vector::initialize_buffers(m_shader->id, color_location_name, false, fan::color::size());
		basic_vertice_vector::basic_shape_angle::initialize_buffers(m_shader->id, angle_location_name, false, 1);
		basic_vertice_vector::basic_shape_rotation_point::initialize_buffers(m_shader->id, rotation_point_location_name, false, fan::vec2::size());
		basic_vertice_vector::basic_shape_rotation_vector::initialize_buffers(m_shader->id, rotation_vector_location_name, false, fan::vec3::size());
		texture_coordinates_t::initialize_buffers(m_shader->id, location_texture_coordinate, false, 2);
	});
}

fan_2d::graphics::convex::convex(fan::camera* camera) : 
	vertice_vector(camera) { 

	assert(0);

	/*m_shader->set_vertex(
		#include <fan/graphics/glsl/opengl/2D/convex.vs>
	);

	m_shader->set_fragment(
		#include <fan/graphics/glsl/opengl/2D/vertice_vector.fs>
	);*/

	m_shader->compile();

}


std::size_t fan_2d::graphics::convex::size() const
{
	return convex_amount.size();
}

void fan_2d::graphics::convex::set_angle(uint32_t i, f32_t angle)
{
	uint32_t offset = 0;

	for (int j = 0; j < i; j++) {
		offset += convex_amount[j];
	}

	for (int j = 0; j < convex_amount[i]; j++) {
		vertice_vector::basic_shape_angle::set_value(offset + j, angle);
	}
}

void fan_2d::graphics::convex::set_position(uint32_t i, const fan::vec2& position)
{
	uint32_t offset = 0;

	for (int j = 0; j < i; j++) {
		offset += convex_amount[j];
	}

	for (int j = 0; j < convex_amount[i]; j++) {
		vertice_vector::basic_shape_rotation_point::set_value(offset + j, position);
	}
}

void fan_2d::graphics::convex::draw(fan_2d::graphics::shape shape, uint32_t single_draw_amount, uint32_t begin, uint32_t end)
{
	vertice_vector::draw(shape, single_draw_amount, begin, end);
}

void fan_2d::graphics::convex::push_back(convex::properties_t property)
{

	convex_amount.emplace_back(property.points.size());

	for (int i = 0; i < property.points.size(); i++) {

		vertice_vector::properties_t vv_property = (vertice_vector::properties_t)property;
		vv_property.rotation_point = property.position;
		vv_property.position = property.points[i];

		vertice_vector::push_back(vv_property);
	}
}

//
//fan_2d::graphics::line fan_2d::graphics::create_grid(fan::camera* camera, const fan::vec2i& block_size, const fan::vec2i& grid_size, const fan::color& color)
//{
//	fan_2d::graphics::line lv(camera);
//
//	const fan::vec2 view = (fan::cast<f_t>(grid_size) / block_size).ceiled();
//
//	for (int i = 0; i < view.x; i++) {
//		lv.push_back(fan::vec2(i * block_size.x, 0), fan::vec2(i * block_size.x, grid_size.y), color);
//	}
//
//	for (int i = 0; i < view.y; i++) {
//		lv.push_back(fan::vec2(0, i * block_size.y), fan::vec2(grid_size.x, i * block_size.y), color);
//	}
//
//	lv.release_queue(true, true);
//
//	return lv;
//}

fan_2d::graphics::rectangle::rectangle(fan::camera* camera) 
	: m_camera(camera),
	// init after shader
	m_lighting_properties(&m_shader), 
	m_queue_helper(camera->m_window) {

  m_shader->set_vertex(
    #include <fan/graphics/glsl/opengl/2D/rectangle.vs>
  );

  m_shader->set_fragment(
    #include <fan/graphics/glsl/opengl/2D/rectangle.fs>
  );

  m_shader->compile();

	this->initialize();
}

fan_2d::graphics::rectangle::~rectangle() {

	m_queue_helper.reset();

	if (m_draw_index != -1) {
		m_camera->m_window->erase_draw_call(m_draw_index);
		m_draw_index = -1;
	}
}

fan::shader_t fan_2d::graphics::rectangle::get_shader()
{
	return m_shader;
}

fan_2d::graphics::fill_mode_e fan_2d::graphics::rectangle::get_draw_mode() const
{
	return m_fill_mode;
}

void fan_2d::graphics::rectangle::set_draw_mode(fan_2d::graphics::fill_mode_e fill_mode)
{
	m_fill_mode = fill_mode;
}

bool fan_2d::graphics::rectangle::read(read_t* read, void* ptr, uintptr_t* size)
{
	thread_local static uintptr_t size_;

	switch (read->stage) {
		case 0: {
			*(void**)ptr = &size_;
			*size = sizeof(uintptr_t);
			break;
		}
		case 1: {
			color_t::m_buffer_object.resize(size_ * 6);
			*(void**)ptr = color_t::m_buffer_object.data();
			*size = size_ * sizeof(decltype(color_t::m_buffer_object)::value_type) * 6;

			break;
		}
		case 2: {

			position_t::m_buffer_object.resize(size_ * 6);
			*(void**)ptr = position_t::m_buffer_object.data();
			*size = size_ * sizeof(decltype(position_t::m_buffer_object)::value_type) * 6;

			break;
		}
		case 3: {
			size_t::m_buffer_object.resize(size_ * 6);
			*(void**)ptr = size_t::m_buffer_object.data();
			*size = size_ * sizeof(decltype(size_t::m_buffer_object)::value_type) * 6;

			break;
		}
		case 4: {
			angle_t::m_buffer_object.resize(size_ * 6);
			*(void**)ptr = angle_t::m_buffer_object.data();
			*size = size_ * sizeof(decltype(angle_t::m_buffer_object)::value_type) * 6;

			break;
		}
		case 5: {
			rotation_point_t::m_buffer_object.resize(size_ * 6);
			*(void**)ptr = rotation_point_t::m_buffer_object.data();
			*size = size_ * sizeof(decltype(rotation_point_t::m_buffer_object)::value_type) * 6;

			break;
		}
		case 6: {
			rotation_vector_t::m_buffer_object.resize(size_ * 6);
			*(void**)ptr = rotation_vector_t::m_buffer_object.data();
			*size = size_ * sizeof(decltype(rotation_vector_t::m_buffer_object)::value_type) * 6;

			break;
		}
		case 7: {
			this->write_data();

			return 0;
		}
	}

	read->stage++;

	return 1;
}

bool fan_2d::graphics::rectangle::write(write_t* write, void* ptr, uintptr_t* size)
{
	switch (write->stage) {
		case 0: {
			thread_local static uintptr_t size_;
			size_ = this->size();
			*(void**)ptr = &size_;
			*size = sizeof(uintptr_t);
			break;
		}
		case 1: {
			*(void**)ptr = (void*)color_t::m_buffer_object.data();
			*size = color_t::m_buffer_object.size() * sizeof(decltype(color_t::m_buffer_object)::value_type);
			break;
		}
		case 2: {
			*(void**)ptr = (void*)position_t::m_buffer_object.data();
			*size = position_t::m_buffer_object.size() * sizeof(decltype(position_t::m_buffer_object)::value_type);
			break;
		}
		case 3: {
			*(void**)ptr = (void*)size_t::m_buffer_object.data();
			*size = size_t::m_buffer_object.size() * sizeof(decltype(size_t::m_buffer_object)::value_type);
			break;
		}
		case 4: {
			*(void**)ptr = (void*)angle_t::m_buffer_object.data();
			*size = angle_t::m_buffer_object.size() * sizeof(decltype(angle_t::m_buffer_object)::value_type);
			break;
		}
		case 5: {
			*(void**)ptr = (void*)rotation_point_t::m_buffer_object.data();
			*size = rotation_point_t::m_buffer_object.size() * sizeof(decltype(rotation_point_t::m_buffer_object)::value_type);
			break;
		}
		case 6: {
			*(void**)ptr = (void*)rotation_vector_t::m_buffer_object.data();
			*size = rotation_vector_t::m_buffer_object.size() * sizeof(decltype(rotation_vector_t::m_buffer_object)::value_type);
			break;
		}
		case 7: {
			return 0;
		}
	}

	write->stage++;

	return 1;
}

void fan_2d::graphics::rectangle::enable_draw()
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

void fan_2d::graphics::rectangle::disable_draw()
{
	if (m_draw_index == -1) {
		return;
	}

	m_camera->m_window->erase_draw_call(m_draw_index);
	m_draw_index = -1;
}

//void fan_2d::graphics::rectangle::set_draw_order(uint32_t i)
//{
//	m_camera->m_window->switch_draw_call(m_draw_index, i);
//}

// protected
fan_2d::graphics::rectangle::rectangle(fan::camera* camera, bool init)
: m_camera(camera), 
	// init after shader init
	m_lighting_properties(&m_shader), 
	m_queue_helper(camera->m_window)
{
}

void fan_2d::graphics::rectangle::initialize()
{
	set_draw_mode(fan_2d::graphics::get_fill_mode(m_camera->m_window));

	fan::bind_vao(fan::vao_handler<>::m_buffer_object, [&] {
		color_t::initialize_buffers(m_shader->id, location_color, false, color_t::value_type::size());
		position_t::initialize_buffers(m_shader->id, location_position, false, position_t::value_type::size());
		size_t::initialize_buffers(m_shader->id, location_size, false, size_t::value_type::size());
		angle_t::initialize_buffers(m_shader->id, location_angle, false, 1);
		rotation_point_t::initialize_buffers(m_shader->id, location_rotation_point, false, rotation_point_t::value_type::size());
		rotation_vector_t::initialize_buffers(m_shader->id, location_rotation_vector, false, rotation_vector_t::value_type::size());
	});
}
//

void fan_2d::graphics::rectangle::push_back(const rectangle::properties_t& properties)
{
	for (int i = 0; i < 6; i++) {

		position_t::push_back(properties.position);
		size_t::push_back(properties.size);
		color_t::push_back(properties.color);
		angle_t::push_back(properties.angle);
		rotation_point_t::push_back(properties.rotation_point);
		rotation_vector_t::push_back(properties.rotation_vector);
	}

	m_queue_helper.write([&] {
		this->write_data();
	});
}

void fan_2d::graphics::rectangle::insert(uint32_t i, const rectangle::properties_t& properties)
{
	i *= 6;

	for (int j = 0; j < 6; j++) {

		position_t::insert(position_t::m_buffer_object.begin() + i + j, properties.position);
		size_t::insert(size_t::m_buffer_object.begin() + i + j, properties.size);
		color_t::insert(color_t::m_buffer_object.begin() + i + j, properties.color);
		angle_t::insert(angle_t::m_buffer_object.begin() + i + j, properties.angle);
		rotation_point_t::insert(rotation_point_t::m_buffer_object.begin() + i + j, properties.rotation_point);
		rotation_vector_t::insert(rotation_vector_t::m_buffer_object.begin() + i + j, properties.rotation_vector);
	}

	m_queue_helper.write([&] {
		this->write_data();
	});
}

void fan_2d::graphics::rectangle::reserve(uint32_t size)
{
	size = size * 6;

	color_t::reserve(size);
	position_t::reserve(size);
	size_t::resize(size);
	angle_t::reserve(size);
	rotation_point_t::reserve(size);
	rotation_vector_t::reserve(size);

	m_queue_helper.write([&] {
		this->write_data();
	});
}

void fan_2d::graphics::rectangle::resize(uint32_t size, const fan::color& color)
{
	size = size * 6;

	color_t::resize(size, color);
	position_t::resize(size);
	size_t::resize(size);
	angle_t::resize(size);
	rotation_point_t::resize(size);
	rotation_vector_t::resize(size);

	m_queue_helper.write([&] {
		this->write_data();
	});
}

void fan_2d::graphics::rectangle::draw(uint32_t begin, uint32_t end)
{
	fan::bind_vao(fan::vao_handler<>::m_buffer_object, [&] {

		fan::mat4 projection(1);
		projection = fan_2d::graphics::get_projection(m_camera->m_window->get_size());

		fan::mat4 view(1);
		view = m_camera->get_view_matrix(fan_2d::graphics::get_view_translation(m_camera->m_window->get_size(), view));

		m_shader->use();

		m_shader->set_mat4("projection", projection);
		m_shader->set_mat4("view", view);

		//fan::shader_t::set_bool("enable_lighting", false);
		//set_lighting(false);

		//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		fan_2d::graphics::draw([&] {

			if (m_fill_mode == fan_2d::graphics::get_fill_mode(m_camera->m_window)) {
				if (begin != (uint32_t)fan::uninitialized && end == (uint32_t)fan::uninitialized) {
					glDrawArrays(GL_TRIANGLES, begin * 6, this->size() * 6);
				}
				else {
					glDrawArrays(GL_TRIANGLES, begin == (uint32_t)fan::uninitialized ? 0 : begin * 6, end == (uint32_t)fan::uninitialized ? this->size() * 6 : (end - begin) * 6);
				}

				return;
			}

			auto fill_mode = fan_2d::graphics::get_fill_mode(m_camera->m_window);

			m_fill_mode = fill_mode;

			fan_2d::graphics::draw_mode(m_camera->m_window, m_fill_mode, fan_2d::graphics::get_face(m_camera->m_window));

			if (begin != (uint32_t)fan::uninitialized && end == (uint32_t)fan::uninitialized) {
				glDrawArrays(GL_TRIANGLES, begin * 6, this->size() * 6);
			}
			else {
				glDrawArrays(GL_TRIANGLES, begin == (uint32_t)fan::uninitialized ? 0 : begin * 6, end == (uint32_t)fan::uninitialized ? this->size() * 6 : (end - begin) * 6);
			}

			fan_2d::graphics::draw_mode(m_camera->m_window, fill_mode, fan_2d::graphics::get_face(m_camera->m_window));
		});
	});
}

void fan_2d::graphics::rectangle::erase(uint32_t i)
{
	color_t::erase(i * 6, i * 6 + 6);
	position_t::erase(i * 6, i * 6 + 6);
	size_t::erase(i * 6, i * 6 + 6);
	angle_t::erase(i * 6, i * 6 + 6);
	rotation_point_t::erase(i * 6, i * 6 + 6);
	rotation_vector_t::erase(i * 6, i * 6 + 6);

	m_queue_helper.write([&] {
		this->write_data();
	});
}

void fan_2d::graphics::rectangle::erase(uint32_t begin, uint32_t end)
{
	color_t::erase(begin * 6, end * 6);
	position_t::erase(begin * 6, end * 6);
	size_t::erase(begin * 6, end * 6);
	angle_t::erase(begin * 6, end * 6);
	rotation_point_t::erase(begin * 6, end * 6);
	rotation_vector_t::erase(begin * 6, end * 6);

	m_queue_helper.write([&] {
		this->write_data();
	});
}

void fan_2d::graphics::rectangle::clear()
{
	color_t::clear();
	position_t::clear();
	size_t::clear();
	angle_t::clear();
	rotation_point_t::clear();
	rotation_vector_t::clear();

	m_queue_helper.write([&] {
		this->write_data();
	});
}

// 0 top left, 1 top right, 2 bottom left, 3 bottom right
fan_2d::graphics::rectangle_corners_t fan_2d::graphics::rectangle::get_corners(uint32_t i) const
{
	auto position = this->get_position(i);
	auto size = this->get_size(i);

	fan::vec2 mid = position;

	auto corners = get_rectangle_corners_no_rotation(position, size);

	f32_t angle = -angle_t::get_value(i);

	fan::vec2 top_left = get_transformed_point(corners[0] - mid, angle) + mid;
	fan::vec2 top_right = get_transformed_point(corners[1] - mid, angle) + mid;
	fan::vec2 bottom_left = get_transformed_point(corners[2] - mid, angle) + mid;
	fan::vec2 bottom_right = get_transformed_point(corners[3] - mid, angle) + mid;

	return { top_left, top_right, bottom_left, bottom_right };
}
//
//fan::vec2 fan_2d::graphics::rectangle::get_center(uintptr_t i) const
//{
//	auto corners = this->get_corners(i);
//	return corners[0] + (corners[3] - corners[0]) / 2;
//}
//
f32_t fan_2d::graphics::rectangle::get_angle(uint32_t i) const
{	
	return angle_t::m_buffer_object[i * 6];
}

// radians
void fan_2d::graphics::rectangle::set_angle(uint32_t i, f32_t angle)
{
	for (int j = 0; j < 6; j++) {
		angle_t::set_value(i * 6 + j, fmod(angle, fan::math::pi * 2));
	}

	m_queue_helper.edit(i, i + 1, [&] {
		this->edit_data(m_queue_helper.m_min_edit, m_queue_helper.m_max_edit);
	});
}

const fan::color fan_2d::graphics::rectangle::get_color(uint32_t i) const
{
	return color_t::m_buffer_object[i * 6];
}

void fan_2d::graphics::rectangle::set_color(uint32_t i, const fan::color& color)
{
	for (int j = 0; j < 6; j++) {
		color_t::set_value(i * 6 + j, color);
	}
	m_queue_helper.edit(i, i + 1, [&] {
		this->edit_data(m_queue_helper.m_min_edit, m_queue_helper.m_max_edit);
	});
}

fan::vec2 fan_2d::graphics::rectangle::get_position(uint32_t i) const
{
	return position_t::get_value(i * 6);
}

void fan_2d::graphics::rectangle::set_position(uint32_t i, const fan::vec2& position)
{
	for (int j = 0; j < 6; j++) {
		position_t::set_value(i * 6 + j, position);
	}

	m_queue_helper.edit(i, i + 1, [&] {
		this->edit_data(m_queue_helper.m_min_edit, m_queue_helper.m_max_edit);
	});
}

fan::vec2 fan_2d::graphics::rectangle::get_size(uint32_t i) const
{
	return size_t::get_value(i * 6);
}

void fan_2d::graphics::rectangle::set_size(uint32_t i, const fan::vec2& size)
{
	for (int j = 0; j < 6; j++) {
		size_t::set_value(i * 6 + j, size);
	}
	m_queue_helper.edit(i, i + 1, [&] {
		this->edit_data(m_queue_helper.m_min_edit, m_queue_helper.m_max_edit);
	});
}

fan::vec2 fan_2d::graphics::rectangle::get_rotation_point(uint32_t i) const {
	return rotation_point_t::m_buffer_object[i * 6];
}
void fan_2d::graphics::rectangle::set_rotation_point(uint32_t i, const fan::vec2& rotation_point) {
	
	for (int j = 0; j < 6; j++) {
		rotation_point_t::m_buffer_object[i * 6 + j] = rotation_point;
	}

	m_queue_helper.edit(i, i + 1, [&] {
		this->edit_data(m_queue_helper.m_min_edit, m_queue_helper.m_max_edit);
	});
}

fan::vec2 fan_2d::graphics::rectangle::get_rotation_vector(uint32_t i) const {
	return rotation_vector_t::m_buffer_object[i * 6];
}
void fan_2d::graphics::rectangle::set_rotation_vector(uint32_t i, const fan::vec2& rotation_vector) {

	for (int j = 0; j < 6; j++) {
		rotation_vector_t::m_buffer_object[i * 6 + j] = rotation_vector;
	}

	m_queue_helper.edit(i, i + 1, [&] {
		this->edit_data(m_queue_helper.m_min_edit, m_queue_helper.m_max_edit);
	});
}

uintptr_t fan_2d::graphics::rectangle::size() const
{
	return position_t::size() / 6;
}

bool fan_2d::graphics::rectangle::inside(uintptr_t i, const fan::vec2& position) const {

	auto corners = get_corners(i);
	
	return fan_2d::collision::rectangle::point_inside(corners[0], corners[1], corners[2], corners[3], position == fan::math::inf ? fan::cast<fan::vec2::value_type>(this->m_camera->m_window->get_mouse_position() + fan::vec2(this->m_camera->get_position())) : position);
}

constexpr uint64_t fan_2d::graphics::rectangle::element_size() const
{
	return sizeof(decltype(color_t::m_buffer_object)::value_type) +
		   sizeof(decltype(position_t::m_buffer_object)::value_type) +
		   sizeof(decltype(size_t::m_buffer_object)::value_type) +
		   sizeof(decltype(angle_t::m_buffer_object)::value_type) +
		   sizeof(decltype(rotation_point_t::m_buffer_object)::value_type) +
		   sizeof(decltype(rotation_vector_t::m_buffer_object)::value_type);
}

uint32_t* fan_2d::graphics::rectangle::get_vao()
{
	return fan::vao_handler<>::get_buffer_object();
}

void fan_2d::graphics::rectangle::write_data() {
	color_t::write_data();
	position_t::write_data();
	size_t::write_data();
	angle_t::write_data();
	rotation_point_t::write_data();
	rotation_vector_t::write_data();

	m_queue_helper.on_write(m_camera->m_window);
}

void fan_2d::graphics::rectangle::edit_data(uint32_t i) {
	color_t::edit_data(i * 6, i * 6 + 6);
	position_t::edit_data(i * 6, i * 6 + 6);
	size_t::edit_data(i * 6, i * 6 + 6);
	angle_t::edit_data(i * 6, i * 6 + 6);
	rotation_point_t::edit_data(i * 6, i * 6 + 6);
	rotation_vector_t::edit_data(i * 6, i * 6 + 6);

	m_queue_helper.on_edit();
}

void fan_2d::graphics::rectangle::edit_data(uint32_t begin, uint32_t end) {
	color_t::edit_data(begin * 6, end * 6);
	position_t::edit_data(begin * 6, end * 6);
	size_t::edit_data(begin * 6, end * 6);
	angle_t::edit_data(begin * 6, end * 6);
	rotation_point_t::edit_data(begin * 6, end * 6);
	rotation_vector_t::edit_data(begin * 6, end * 6);

	m_queue_helper.on_edit();
}

uint32_t fan_2d::graphics::rectangle_dynamic::push_back(const rectangle::properties_t& properties)
{
	if (!m_free_slots.empty()) {

		static int i = 0;

		position_t::set_value(i, properties.position);
		size_t::set_value(i, properties.size);
		color_t::set_value(i, properties.color);
		angle_t::set_value(i, properties.angle);
		rotation_point_t::set_value(i, properties.rotation_point);	
		rotation_vector_t::set_value(i, properties.rotation_vector);

		i = (i + 1) % 40000;

		m_free_slots.erase(m_free_slots.end() - 1);

		return i;
	}
	else {
		fan_2d::graphics::rectangle::push_back(properties);
	}

	return fan::uninitialized;
}

void fan_2d::graphics::rectangle_dynamic::erase(uint32_t i)
{
	size_t::set_value(i, 0);
	m_free_slots.emplace_back(i);
}

void fan_2d::graphics::rectangle_dynamic::erase(uint32_t begin, uint32_t end)
{
	auto j = m_free_slots.size() == 0 ? 0 : (m_free_slots.size() - 1);

	/*if (m_free_slots.empty()) {
		m_free_slots.push_back(0);
	}*/
	m_free_slots.resize(m_free_slots.size() + (end - begin));

	for (int i = begin; i < end; i++) {
		size_t::set_value(i, 0);
		m_free_slots[j + i] = i;
	}
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

fan_2d::graphics::sprite::sprite(fan::camera* camera)
	: fan_2d::graphics::rectangle(camera, true)
{

	m_shader->set_vertex(
    #include <fan/graphics/glsl/opengl/2D/sprite.vs>
  );

  m_shader->set_fragment(
    #include <fan/graphics/glsl/opengl/2D/sprite.fs>
  );

  m_shader->compile();

	rectangle::initialize();

	fan::bind_vao(*rectangle::get_vao(), [&] {
		texture_coordinates_t::initialize_buffers(m_shader->id, location_texture_coordinate, false, 2);
		RenderOPCode_t::initialize_buffers(m_shader->id, location_RenderOPCode, false, 1);
	});
}

// protected
fan_2d::graphics::sprite::sprite(fan::camera* camera, bool init) 
	: fan_2d::graphics::rectangle(camera, true)
{
}

fan_2d::graphics::sprite::sprite(const fan_2d::graphics::sprite& sprite)
	: 
	fan_2d::graphics::rectangle(sprite),
	fan::texture_handler<1>(sprite),
	fan::render_buffer_handler<>(sprite),
	fan::frame_buffer_handler<>(sprite)
{
	m_textures = sprite.m_textures;
}

fan_2d::graphics::sprite::sprite(fan_2d::graphics::sprite&& sprite) noexcept
	: 
	fan_2d::graphics::rectangle(std::move(sprite)),

	fan::texture_handler<1>(std::move(sprite)),
	fan::render_buffer_handler<>(std::move(sprite)),
	fan::frame_buffer_handler<>(std::move(sprite))
{
	m_textures = std::move(sprite.m_textures);
}

fan_2d::graphics::sprite& fan_2d::graphics::sprite::operator=(const fan_2d::graphics::sprite& sprite)
{
	fan_2d::graphics::rectangle::operator=(sprite);

	fan::texture_handler<1>::operator=(sprite);
	fan::render_buffer_handler<>::operator=(sprite);
	fan::frame_buffer_handler<>::operator=(sprite);

	m_textures = sprite.m_textures;

	return *this;
}

fan_2d::graphics::sprite& fan_2d::graphics::sprite::operator=(fan_2d::graphics::sprite&& sprite)
{
	fan_2d::graphics::rectangle::operator=(std::move(sprite));

	fan::texture_handler<1>::operator=(std::move(sprite));
	fan::render_buffer_handler<>::operator=(std::move(sprite));
	fan::frame_buffer_handler<>::operator=(std::move(sprite));
	m_textures = std::move(sprite.m_textures);

	return *this;
}

void fan_2d::graphics::sprite::push_back(const sprite::properties_t& properties)
{
	sprite::rectangle::properties_t property;
	property.position = properties.position;
	property.size = properties.size;
	property.angle = properties.angle;
	property.rotation_point = properties.rotation_point;
	property.rotation_vector = properties.rotation_vector;
	property.color = properties.color;

	bool write_ = m_queue_helper.m_write;

	rectangle::push_back(property);

	std::array<fan::vec2, 6> texture_coordinates = {
		properties.texture_coordinates[0],
		properties.texture_coordinates[1],
		properties.texture_coordinates[2],

		properties.texture_coordinates[2],
		properties.texture_coordinates[3],
		properties.texture_coordinates[0]
	};

	texture_coordinates_t::insert(texture_coordinates_t::m_buffer_object.end(), texture_coordinates.begin(), texture_coordinates.end());

	RenderOPCode_t::m_buffer_object.insert(RenderOPCode_t::m_buffer_object.end(), 6, properties.RenderOPCode);

	if (m_switch_texture.empty()) {
		m_switch_texture.emplace_back(0);
	}
	else if (m_textures.size() && m_textures[m_textures.size() - 1] != properties.image->texture) {
		m_switch_texture.emplace_back(this->size() - 1);
	}

	m_textures.emplace_back(properties.image->texture);

	if (!write_) {
		m_camera->m_window->edit_write_call(m_queue_helper.m_write_index, this, [&] {
			this->write_data();
		});
	}
}

void fan_2d::graphics::sprite::insert(uint32_t i, uint32_t texture_coordinates_i, const sprite::properties_t& properties)
{
	sprite::rectangle::properties_t property;
	property.position = properties.position;
	property.size = properties.size;
	property.rotation_point = property.position;
	property.rotation_vector = properties.rotation_vector;

	bool write_ = m_queue_helper.m_write;

	fan_2d::graphics::rectangle::insert(texture_coordinates_i / 6, property);
	 
	std::array<fan::vec2, 6> texture_coordinates = {
		properties.texture_coordinates[0],
		properties.texture_coordinates[1],
		properties.texture_coordinates[2],

		properties.texture_coordinates[2],
		properties.texture_coordinates[3],
		properties.texture_coordinates[0]
	};

	texture_coordinates_t::insert(texture_coordinates_t::m_buffer_object.begin() + texture_coordinates_i, texture_coordinates.begin(), texture_coordinates.end());

	m_textures.insert(m_textures.begin() + texture_coordinates_i / 6, properties.image->texture);

	RenderOPCode_t::m_buffer_object.insert(RenderOPCode_t::m_buffer_object.begin() + texture_coordinates_i / 6, 6, properties.RenderOPCode);

	regenerate_texture_switch();

	if (!write_) {
		m_camera->m_window->edit_write_call(m_queue_helper.m_write_index, this, [&] {
			this->write_data();
		});
	}
}

void fan_2d::graphics::sprite::reload_sprite(uint32_t i, fan_2d::graphics::image_t image)
{
	m_textures[i] = image->texture;
}

void fan_2d::graphics::sprite::enable_draw()
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

void fan_2d::graphics::sprite::disable_draw()
{
	if (m_draw_index == -1) {
		return;
	}

	m_camera->m_window->erase_draw_call(m_draw_index);
	m_draw_index = -1;
}

uint32_t fan_2d::graphics::sprite::get_RenderOPCode(uint32_t i) const
{
	return RenderOPCode_t::m_buffer_object[i * 6];
}

void fan_2d::graphics::sprite::set_RenderOPCode(uint32_t i, uint32_t OPCode)
{
	for (int j = 0; j < 6; j++) {
		RenderOPCode_t::m_buffer_object[i * 6 + j] = OPCode;
	}

	m_queue_helper.edit(i, i + 1, [&] {
		this->edit_data(m_queue_helper.m_min_edit, m_queue_helper.m_max_edit);
	});
	
}

void fan_2d::graphics::sprite::draw(uint32_t begin, uint32_t end)
{
	m_shader->use();

	if (m_switch_texture.empty()) {
		return;
	}

	for (int i = m_switch_texture[begin == fan::uninitialized ? 0 : begin]; i < m_switch_texture.size(); i++) {

		m_shader->set_int("texture_sampler", i);
		glActiveTexture(GL_TEXTURE0 + i);
		glBindTexture(GL_TEXTURE_2D, m_textures[m_switch_texture[i]]);

		if (i == m_switch_texture.size() - 1) {
			fan_2d::graphics::rectangle::draw(m_switch_texture[i], end == fan::uninitialized ? this->size() : end);
		}
		else {

			if (end != fan::uninitialized && m_switch_texture[i + 1] > end) {
				break;
			}

			fan_2d::graphics::rectangle::draw(m_switch_texture[i], m_switch_texture[i + 1]);
		}

	}

}

fan::camera* fan_2d::graphics::sprite::get_camera()
{
	return m_camera;
}

void fan_2d::graphics::sprite::initialize()
{
	fan::bind_vao(*rectangle::get_vao(), [&] {
		color_t::initialize_buffers(m_shader->id, location_color, false, color_t::value_type::size());
		position_t::initialize_buffers(m_shader->id, location_position, false, position_t::value_type::size());
		size_t::initialize_buffers(m_shader->id, location_size, false, size_t::value_type::size());
		angle_t::initialize_buffers(m_shader->id, location_angle, false, 1);
		rotation_point_t::initialize_buffers(m_shader->id, location_rotation_point, false, rotation_point_t::value_type::size());
		rotation_vector_t::initialize_buffers(m_shader->id, location_rotation_vector, false, rotation_vector_t::value_type::size());
		texture_coordinates_t::initialize_buffers(m_shader->id, location_texture_coordinate, false, 2);
	});
}

f32_t fan_2d::graphics::sprite::get_transparency(uint32_t i) const
{
	return color_t::get_value(i * 6).a;
}

void fan_2d::graphics::sprite::set_transparency(uint32_t i, f32_t transparency)
{
	for (int j = 0; j < 6; j++) {
		color_t::set_value(i * 6 + j, fan::color(0, 0, 0, transparency));
	}

	m_queue_helper.edit(i, i + 1, [&] {
		this->edit_data(m_queue_helper.m_min_edit, m_queue_helper.m_max_edit);
	});
}

void fan_2d::graphics::sprite::erase(uint32_t i)
{
	bool write_ = m_queue_helper.m_write;
	rectangle::erase(i);
	
	texture_coordinates_t::erase(i * 6, i * 6 + 6);

	m_textures.erase(m_textures.begin() + i);
	
	regenerate_texture_switch();

	RenderOPCode_t::erase(i * 6, i * 6 + 6);

	if (!write_) {
		m_camera->m_window->edit_write_call(m_queue_helper.m_write_index, this, [&] {
			this->write_data();
		});
	}
}

void fan_2d::graphics::sprite::erase(uint32_t begin, uint32_t end)
{
	bool write_ = m_queue_helper.m_write;

	rectangle::erase(begin, end);

	texture_coordinates_t::erase(begin * 6, end * 6);

	m_textures.erase(m_textures.begin() + begin, m_textures.begin() + end);

	RenderOPCode_t::erase(begin * 6, end * 6);

	regenerate_texture_switch();

	if (!write_) {
		m_camera->m_window->edit_write_call(m_queue_helper.m_write_index, this, [&] {
			this->write_data();
		});
	}
}

void fan_2d::graphics::sprite::clear()
{
	bool write_ = m_queue_helper.m_write;
	rectangle::clear();

	texture_coordinates_t::clear();

	m_textures.clear();

	m_switch_texture.clear();

	RenderOPCode_t::clear();

	if (!write_) {
		m_camera->m_window->edit_write_call(m_queue_helper.m_write_index, this, [&] {
			this->write_data();
		});
	}
}

void fan_2d::graphics::sprite::write_data() {
	RenderOPCode_t::write_data();
	texture_coordinates_t::write_data();
	rectangle::write_data();
}

void fan_2d::graphics::sprite::edit_data(uint32_t i) {
	RenderOPCode_t::write_data();
	texture_coordinates_t::write_data();
	rectangle::edit_data(i);
}

void fan_2d::graphics::sprite::edit_data(uint32_t begin, uint32_t end) {
	RenderOPCode_t::write_data();
	texture_coordinates_t::write_data();
	rectangle::edit_data(begin, end);
}

// todo remove
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

// update same with sprite

fan_2d::graphics::yuv420p_renderer::yuv420p_renderer(fan::camera* camera)
	: fan_2d::graphics::sprite(camera) {

  m_shader->set_vertex(
    #include <fan/graphics/glsl/opengl/2D/rectangle.vs>
  );

  m_shader->set_fragment(
    #include <fan/graphics/glsl/opengl/2D/rectangle.fs>
  );

	m_shader->compile();

}

void fan_2d::graphics::yuv420p_renderer::push_back(const yuv420p_renderer::properties_t& properties) {

	bool write_ = m_queue_helper.m_write;

	m_textures.resize(m_textures.size() + 3);

	glGenTextures(1, &m_textures[m_textures.size() - 3]);
	glGenTextures(1, &m_textures[m_textures.size() - 2]);
	glGenTextures(1, &m_textures[m_textures.size() - 1]);

	glBindTexture(GL_TEXTURE_2D, m_textures[m_textures.size() - 3]);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, fan_2d::graphics::image_load_properties::visual_output);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, fan_2d::graphics::image_load_properties::visual_output);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, fan_2d::graphics::image_load_properties::filter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, fan_2d::graphics::image_load_properties::filter);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, properties.pixel_data.size.x, properties.pixel_data.size.y, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, properties.pixel_data.pixels[0]);

	glBindTexture(GL_TEXTURE_2D, m_textures[m_textures.size() - 2]);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, fan_2d::graphics::image_load_properties::visual_output);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, fan_2d::graphics::image_load_properties::visual_output);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, fan_2d::graphics::image_load_properties::filter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, fan_2d::graphics::image_load_properties::filter);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, properties.pixel_data.size.x / 2, properties.pixel_data.size.y / 2, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, properties.pixel_data.pixels[1]);

	glBindTexture(GL_TEXTURE_2D, m_textures[m_textures.size() - 1]);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, fan_2d::graphics::image_load_properties::visual_output);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, fan_2d::graphics::image_load_properties::visual_output);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, fan_2d::graphics::image_load_properties::filter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, fan_2d::graphics::image_load_properties::filter);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, properties.pixel_data.size.x / 2, properties.pixel_data.size.y / 2, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, properties.pixel_data.pixels[2]);

	glBindTexture(GL_TEXTURE_2D, 0);

	fan::vec2 rotation_point = properties.rotation_point == fan::math::inf ? properties.position : properties.rotation_point;

	sprite::rectangle::properties_t property;
	property.position = properties.position;
	property.size = properties.size;
	property.angle = properties.angle;
	property.rotation_point = properties.rotation_point;
	property.rotation_vector = properties.rotation_vector;

	fan_2d::graphics::rectangle::push_back(property);

	texture_coordinates_t::insert(texture_coordinates_t::m_buffer_object.end(), properties.texture_coordinates.begin(), properties.texture_coordinates.end());

	image_size.emplace_back(properties.pixel_data.size);

	if (!write_) {
		m_camera->m_window->edit_write_call(m_queue_helper.m_write_index, this, [&] {
			this->write_data();
		});
	}

}

void fan_2d::graphics::yuv420p_renderer::reload_pixels(uint32_t i, const fan_2d::graphics::pixel_data_t& pixel_data) {
	glBindTexture(GL_TEXTURE_2D, m_textures[i * 3]);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, fan_2d::graphics::image_load_properties::visual_output);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, fan_2d::graphics::image_load_properties::visual_output);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, fan_2d::graphics::image_load_properties::filter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, fan_2d::graphics::image_load_properties::filter);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, pixel_data.size.x, pixel_data.size.y, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, pixel_data.pixels[0]);

	glBindTexture(GL_TEXTURE_2D, m_textures[i * 3 + 1]);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, fan_2d::graphics::image_load_properties::visual_output);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, fan_2d::graphics::image_load_properties::visual_output);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, fan_2d::graphics::image_load_properties::filter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, fan_2d::graphics::image_load_properties::filter);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, pixel_data.size.x / 2, pixel_data.size.y / 2, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, pixel_data.pixels[1]);

	glBindTexture(GL_TEXTURE_2D, m_textures[i * 3 + 2]);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, fan_2d::graphics::image_load_properties::visual_output);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, fan_2d::graphics::image_load_properties::visual_output);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, fan_2d::graphics::image_load_properties::filter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, fan_2d::graphics::image_load_properties::filter);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, pixel_data.size.x / 2, pixel_data.size.y / 2, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, pixel_data.pixels[2]);

	glBindTexture(GL_TEXTURE_2D, 0);

	image_size[i] = pixel_data.size;
}

void fan_2d::graphics::yuv420p_renderer::write_data() {
	fan_2d::graphics::sprite::write_data();
}

fan::vec2ui fan_2d::graphics::yuv420p_renderer::get_image_size(uint32_t i) const
{
	return this->image_size[i];
}

void fan_2d::graphics::yuv420p_renderer::enable_draw()
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

void fan_2d::graphics::yuv420p_renderer::disable_draw()
{
	if (m_draw_index == -1) {
		return;
	}

	m_camera->m_window->erase_draw_call(m_draw_index);
	m_draw_index = -1;
}

void fan_2d::graphics::yuv420p_renderer::draw()
{
	m_shader->use();

	texture_coordinates_t::bind_gl_storage_buffer([&] {});

	m_shader->set_int("sampler_y", 0);
	m_shader->set_int("sampler_u", 1);
	m_shader->set_int("sampler_v", 2);

	for (int i = 0; i < rectangle::size(); i++) {
		glActiveTexture(GL_TEXTURE0 + 0);
		glBindTexture(GL_TEXTURE_2D, m_textures[i * 3]);

		glActiveTexture(GL_TEXTURE0 + 1);
		glBindTexture(GL_TEXTURE_2D, m_textures[i * 3 + 1]);

		glActiveTexture(GL_TEXTURE0 + 2);
		glBindTexture(GL_TEXTURE_2D, m_textures[i * 3 + 2]);

		fan_2d::graphics::rectangle::draw(i);
	}
}

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
////void fan_2d::graphics::rope::push_back(const std::vector<std::pair<fan::vec2, fan::vec2>>& joints, const fan::color& color)
////{
////	for (int i = 0; i < joints.size(); i++) {
////		fan_2d::graphics::line::push_back(joints[i].first, joints[i].second, color, true);
////	}
////
////	fan_2d::graphics::line::release_queue(!fan::gpu_queue, !fan::gpu_queue);
////}
//
//fan_2d::graphics::particles::particles(fan::camera* camera)
//	: rectangle(camera, fan::shader_t(fan_2d::graphics::shader_paths::shape_vector_vs, fan_2d::graphics::shader_paths::shape_vector_fs)) {
//}
//
//void fan_2d::graphics::particles::push_back(const fan::vec2& position, const fan::vec2& size, f32_t angle, f32_t angle_velocity, const fan::vec2& velocity, const fan::color& color, uintptr_t time)
//{
//	bool queue = fan::gpu_queue;
//
//	if (!queue) {
//		fan::begin_queue();
//	}
//
//	fan_2d::graphics::rectangle::push_back(position, size, color);
//	this->set_angle(this->size() - 1, angle);
//
//	this->m_particles.push_back({ angle_velocity, velocity, fan::timer(fan::timer<>::start(), time) });
//
//	if (!queue) {
//		fan::end_queue();
//	}
//}
//
//void fan_2d::graphics::particles::update()
//{
//	bool queue = fan::gpu_queue;
//
//	if (!queue) {
//		fan::begin_queue();
//	}
//
//	for (uintptr_t i = 0; i < this->size(); i++) {
//		if (this->m_particles[i].m_timer.finished()) {
//			this->erase(i);
//			this->m_particles.erase(this->m_particles.begin() + i);
//			continue;
//		}
//
//		this->set_position(i, this->get_position(i) + this->m_particles[i].m_velocity * m_camera->m_window->get_delta_time());
//		this->set_angle(i, this->get_angle(i) + this->m_particles[i].m_angle_velocity * m_camera->m_window->get_delta_time());
//	}
//
//	if (!queue) {
//		fan::end_queue();
//	}
//}

fan_2d::graphics::base_lighting::base_lighting(fan::shader_t* shader, uint32_t* vao)
	: m_base_lighting_shader(shader), m_vao(vao)
{
	fan::bind_vao(*m_vao, [&] {

		light_position_t::initialize_buffers((*m_base_lighting_shader)->id, location_light_position, true, fan::vec2::size());
		light_color_t::initialize_buffers((*m_base_lighting_shader)->id, location_light_color, true, fan::color::size());
		light_brightness_t::initialize_buffers((*m_base_lighting_shader)->id, location_light_brightness, true, 1);
		light_angle_t::initialize_buffers((*m_base_lighting_shader)->id, location_light_angle, true, 1);

	});

}

void fan_2d::graphics::base_lighting::push_back(const fan::vec2& position, f32_t strength, const fan::color& color, f32_t angle)
{
	light_position_t::push_back(position);
	light_color_t::push_back(color);
	light_brightness_t::push_back(strength);
	light_angle_t::push_back(angle);
	light_position_t::write_data();
	light_color_t::write_data();
	light_brightness_t::write_data();
	light_angle_t::write_data();

}

void fan_2d::graphics::base_lighting::set_position(uint32_t i, const fan::vec2& position)
{
	light_position_t::set_value(i, position);
	light_position_t::write_data();
}

fan::color fan_2d::graphics::base_lighting::get_color(uint32_t i) const
{
	return light_color_t::get_value(i);
}

f32_t fan_2d::graphics::base_lighting::get_brightness(uint32_t i) const
{
	return light_brightness_t::get_value(i);
}

void fan_2d::graphics::base_lighting::set_brightness(uint32_t i, f32_t brightness)
{
	light_brightness_t::set_value(i, brightness);
}

f32_t fan_2d::graphics::base_lighting::get_angle(uint32_t i) const
{
	return light_angle_t::get_value(i);
}

void fan_2d::graphics::base_lighting::set_angle(uint32_t i, f32_t angle)
{
	light_angle_t::set_value(i, angle);
}

fan_2d::graphics::light::light(fan::camera* camera, fan::shader_t* shader, uint32_t* vao)
	:
	rectangle(camera),
	base_lighting(shader, vao)
{
  m_shader->set_vertex(
    #include <fan/graphics/glsl/opengl/2D/rectangle.vs>
  );

  m_shader->set_fragment(
    #include <fan/graphics/glsl/opengl/2D/rectangle.fs>
  );

	m_shader->compile();

	m_lighting_properties.set_lighting(true);
}

void fan_2d::graphics::light::push_back(const fan::vec2& position, const fan::vec2& size, f32_t strength, const fan::color& color, f32_t angle)
{
	rectangle::push_back({ position, size, 0, 0, 0, color });
	base_lighting::push_back(position, strength, color, angle);
}

void fan_2d::graphics::light::set_position(uint32_t i, const fan::vec2& position)
{
	rectangle::set_position(i, position);
	base_lighting::set_position(i, position);
	m_shader->use();
	m_shader->set_vec2("light_position", position);
}

void fan_2d::graphics::light::set_color(uint32_t i, const fan::color& color)
{
	light_color_t::set_value(i, color);
	rectangle::set_color(i, color);
}

void fan_3d::graphics::add_camera_rotation_callback(fan::camera* camera) {
	camera->m_window->add_mouse_move_callback(std::function<void(fan::window* window, const fan::vec2i& position)>(std::bind(&fan::camera::rotate_camera, camera, 0)));
}

//fan_3d::graphics::rectangle_vector::rectangle_vector(fan::camera* camera, const fan::color& color, uintptr_t block_size)
//	: basic_shape(camera, fan::shader_t(fan_3d::graphics::shader_paths::shape_vector_vs, fan_3d::graphics::shader_paths::shape_vector_fs)),
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
//		for (uintptr_t j = 0; j < face.mNumIndices; j++) {
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
//	for (uintptr_t i = 0; i < mat->GetTextureCount(type); i++) {
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
//	for (uintptr_t i = 0; i < this->meshes.size(); i++) {
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
//	this->m_shader->use();
//
//	fan::mat4 projection(1);
//	projection = fan::perspective<fan::mat4>(fan::radians(90.f), (f32_t)m_camera->m_window->get_size().x / (f32_t)m_camera->m_window->get_size().y, 0.1f, 1000.0f);
//
//	fan::mat4 view(m_camera->get_view_matrix());
//
//	this->m_shader->set_int("texture_sampler", 0);
//	this->m_shader->set_mat4("projection", projection);
//	this->m_shader->set_mat4("view", view);
//	this->m_shader->set_vec3("light_position", m_camera->get_position());
//	this->m_shader->set_vec3("view_position",m_camera->get_position());
//	this->m_shader->set_vec3("light_color", fan::vec3(1, 1, 1));
//	this->m_shader->set_int("texture_diffuse", 0);
//	this->m_shader->set_mat4("model", model);
//
//	//_Shader.set_vec3("sky_color", fan::vec3(220.f / 255.f, 219.f / 255.f, 223.f / 255.f));
//	glActiveTexture(GL_TEXTURE0);
//	glBindTexture(GL_TEXTURE_2D, this->textures_loaded[0].id);
//
//	glDepthFunc(GL_LEQUAL);
//	for (uintptr_t i = 0; i < this->meshes.size(); i++) {
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

//fan::vec3 line_triangle_intersection(const fan::vec3& ray_begin, const fan::vec3& ray_end, const fan::vec3& p0, const fan::vec3& p1, const fan::vec3& p2) {
//
//	const auto lab = (ray_begin + ray_end) - ray_begin;
//
//	const auto p01 = p1 - p0;
//	const auto p02 = p2 - p0;
//
//	const auto normal = fan::math::cross(p01, p02);
//
//	const auto t = fan_3d::math::dot(normal, ray_begin - p0) / fan_3d::math::dot(-lab, normal);
//	const auto u = fan_3d::math::dot(fan::math::cross(p02, -lab), ray_begin - p0) / fan_3d::math::dot(-lab, normal);
//	const auto v = fan_3d::math::dot(fan::math::cross(-lab, p01), ray_begin - p0) / fan_3d::math::dot(-lab, normal);
//
//	if (t >= 0 && t <= 1 && u >= 0 && u <= 1 && v >= 0 && v <= 1 && (u + v) <= 1) {
//		return ray_begin + lab * t;
//	}
//
//	return INFINITY;
//
//}

//fan::vec3 fan_3d::graphics::line_triangle_intersection(const fan::da_t<f32_t, 2, 3>& line, const fan::da_t<f32_t, 3, 3>& triangle) {
//
//	const auto lab = (line[0] + line[1]) - line[0];
//
//	const auto p01 = triangle[1] - triangle[0];
//	const auto p02 = triangle[2] - triangle[0];
//
//	const auto normal = fan::math::cross(p01, p02);
//
//	const auto t = fan_3d::math::dot(normal, line[0] - triangle[0]) / fan_3d::math::dot(-lab, normal);
//	const auto u = fan_3d::math::dot(fan::math::cross(p02, -lab), line[0] - triangle[0]) / fan_3d::math::dot(-lab, normal);
//	const auto v = fan_3d::math::dot(fan::math::cross(-lab, p01), line[0] - triangle[0]) / fan_3d::math::dot(-lab, normal);
//
//	if (t >= 0 && t <= 1 && u >= 0 && u <= 1 && v >= 0 && v <= 1 && (u + v) <= 1) {
//		return line[0] + lab * t;
//	}
//
//	return INFINITY;
//}
//
//fan::vec3 fan_3d::graphics::line_plane_intersection(const fan::da_t<f32_t, 2, 3>& line, const fan::da_t<f32_t, 4, 3>& square) {
//	const fan::da_t<f32_t, 3> plane_normal = fan::math::normalize_no_sqrt(fan::math::cross(square[3] - square[2], square[0] - square[2]));
//	const f32_t nl_dot(math::dot(plane_normal, line[1]));
//
//	if (!nl_dot) {
//		return fan::vec3(INFINITY);
//	}
//
//	const f32_t ray_length = fan_3d::math::dot(square[2] - line[0], plane_normal) / nl_dot;
//	if (ray_length <= 0) {
//		return fan::vec3(INFINITY);
//	}
//	if (fan::math::custom_pythagorean_no_sqrt(fan::vec3(line[0]), fan::vec3(line[0] + line[1])) < ray_length) {
//		return fan::vec3(INFINITY);
//	}
//	const fan::vec3 intersection(line[0] + line[1] * ray_length);
//
//	auto result = fan_3d::math::dot((square[2] - line[0]), plane_normal);
//
//	if (!result) {
//		fan::print("on plane");
//	}
//
//	if (intersection[1] >= square[3][1] && intersection[1] <= square[0][1] &&
//		intersection[2] >= square[3][2] && intersection[2] <= square[0][2])
//	{
//		return intersection;
//	}
//	return fan::vec3(INFINITY);
//}

#endif