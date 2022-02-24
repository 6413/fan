#include <fan/graphics/renderer.hpp>
//
#if fan_renderer == fan_renderer_opengl

#include <fan/graphics/opengl/gl_gui.hpp>
////
////fan_2d::graphics::gui::circle::circle(fan::camera* camera)
////	: fan_2d::graphics::circle(camera)
////{
////	this->m_camera->m_window->add_resize_callback([&] (const fan::window* window, const fan::vec2i&)  {
////
////		auto offset = this->m_camera->m_window->get_size() - this->m_camera->m_window->get_previous_size();
////
////		for (int i = 0; i < this->size(); i++) {
////			this->set_position(i, this->get_position(i) + fan::vec2(offset.x, 0));
////		}
////	});
////}
////
//
void fan_2d::graphics::gui::text_renderer::open(fan::opengl::context_t* context) {

	m_shader.open();

	m_store.open();

	m_shader.set_vertex(
    #include <fan/graphics/glsl/opengl/2D/text.vs>
  );

	m_shader.set_fragment(
    #include <fan/graphics/glsl/opengl/2D/text.fs>
  );

	m_shader.compile();

	m_glsl_buffer.open();
  m_glsl_buffer.init(m_shader.id, 
    sizeof(fan_2d::graphics::rectangle::properties_t) +
    sizeof(fan::vec2) +// texture_coordinates
		sizeof(f32_t) +  // font size
		sizeof(fan::color) +  // outline color
		sizeof(f32_t)  // outline size
  );
  m_queue_helper.open();
	m_store_sprite.open();

	static constexpr const char* font_name = "bitter";

	if (!font_image) {
		font_image = fan_2d::graphics::load_image(std::string("fonts/") + font_name + ".webp");
	}

	font = fan::font::parse_font(std::string("fonts/") + font_name + "_metrics.txt");
}

void fan_2d::graphics::gui::text_renderer::close(fan::opengl::context_t* context) {
	for (int i = 0; i < m_store.size(); i++) {
		m_store[i].m_text.close();
	}
	m_store.close();

  sprite::close(context);
}

void fan_2d::graphics::gui::text_renderer::enable_draw(fan::opengl::context_t* context)
{
	m_draw_node_reference = context->enable_draw(this, [] (fan::opengl::context_t* c, void* d) { ((decltype(this))d)->draw(c); });
}

void fan_2d::graphics::gui::text_renderer::disable_draw(fan::opengl::context_t* context)
{
	context->disable_draw(m_draw_node_reference);
}

void fan_2d::graphics::gui::text_renderer::draw(fan::opengl::context_t* context, uint32_t begin, uint32_t end) {
	
	if (!fan_2d::graphics::sprite::size(context)) {
		return;
	}

	auto begin_ = begin == 0 || begin == fan::uninitialized ? 0 : m_store[begin - 1].m_indices;
	auto end_ = end == 0 ? 0 : end == fan::uninitialized ? m_store[m_store.size() - 1].m_indices : m_store[end - 1].m_indices;

	fan_2d::graphics::sprite::draw(context, begin_, end_);
}

void fan_2d::graphics::gui::text_renderer::push_back(fan::opengl::context_t* context, properties_t properties) {

	if (properties.text.empty()) {
		throw std::runtime_error("text cannot be empty");
	}

	store_t store;
	store.m_text.open();

	store.m_position = properties.position;
	*store.m_text = properties.text;
	store.m_indices = m_store.empty() ? properties.text.size() : m_store[m_store.size() - 1].m_indices + properties.text.size();

	fan::vec2 text_size = get_text_size(context, properties.text, properties.font_size);

	f32_t left = properties.position.x - text_size.x / 2;

	uint64_t new_lines = get_new_lines(context, properties.text);

	properties.position.y += font.size * convert_font_size(context, properties.font_size) / 2;
	if (new_lines) {
		properties.position.y -= (get_line_height(context, properties.font_size) * (new_lines - 1)) / 2;
	}

	f32_t average_height = 0;

	store.m_new_lines = new_lines;

	m_store.push_back(store);

	for (int i = 0; i < properties.text.size(); i++) {

		if (properties.text[i] == '\n') {
			left = properties.position.x - text_size.x / 2;
			properties.position.y += get_line_height(context, properties.font_size);
		}

		auto letter = get_letter_info(context, properties.text[i], properties.font_size);

		properties_t letter_properties;
		letter_properties.position = fan::vec2(left - letter.metrics.offset.x, properties.position.y);
		letter_properties.font_size = properties.font_size;
		letter_properties.text_color = properties.text_color;
		letter_properties.outline_color = properties.outline_color;
		letter_properties.outline_size = properties.outline_size;
		letter_properties.angle = properties.angle;

		push_letter(context, properties.text[i], letter_properties);

		left += letter.metrics.advance;
		average_height += letter.metrics.size.y;

		fan::vec2 rotation_point = (properties.position - fan::vec2(0, text_size.y / 2 - (average_height / properties.text.size()) / 2)) + properties.rotation_point;

		for (uint32_t j = 0; j < sprite::vertex_count; j++) {
			m_glsl_buffer.edit_ram_instance(
				(rectangle::size(context) - 1) * sprite::vertex_count + j,
				&rotation_point,
				offset_rotation_point,
				sizeof(fan::vec2)
			);
		}
	}
}

void fan_2d::graphics::gui::text_renderer::insert(fan::opengl::context_t* context, uint32_t i, properties_t properties)
{
	if (properties.text.empty()) {
		throw std::runtime_error("text cannot be empty");
	}

	store_t store;
	store.m_text.open();

	store.m_position = properties.position;
	*store.m_text = properties.text;

	fan::vec2 text_size = get_text_size(context, properties.text, properties.font_size);

	f32_t left = properties.position.x - text_size.x / 2;

	uint64_t new_lines = get_new_lines(context, properties.text);

	properties.position.y += font.size * convert_font_size(context, properties.font_size) / 2;
	if (new_lines) {
		properties.position.y -= (get_line_height(context, properties.font_size) * (new_lines - 1)) / 2;
	}

	f32_t average_height = 0;

	store.m_new_lines = new_lines;
	m_store.insert(m_store.begin() + i, store);

	for (int j = 0; j < properties.text.size(); j++) {

		if (properties.text[j] == '\n') {
			left = properties.position.x - text_size.x / 2;
			properties.position.y += fan_2d::graphics::gui::text_renderer::get_line_height(context, properties.font_size);
		}

		auto letter = get_letter_info(context, properties.text[j], properties.font_size);

		properties_t letter_properties;
		letter_properties.position = fan::vec2(left - letter.metrics.offset.x, properties.position.y);
		letter_properties.font_size = properties.font_size;
		letter_properties.text_color = properties.text_color;
		letter_properties.outline_color = properties.outline_color;
		letter_properties.outline_size = properties.outline_size;
		letter_properties.angle = properties.angle;

		insert_letter(context, get_index(i) + j, properties.text[j], letter_properties);

		left += letter.metrics.advance;
		average_height += letter.metrics.size.y;

		fan::vec2 rotation_point = properties.position - fan::vec2(0, text_size.y / 2 - (average_height / properties.text.size()) / 2) + properties.rotation_point;

		for (uint32_t j = 0; j < sprite::vertex_count; j++) {
			m_glsl_buffer.edit_ram_instance(
				i * sprite::vertex_count + j,
				&rotation_point,
				offset_rotation_point,
				sizeof(fan::vec2)
			);
		}
	}

	regenerate_indices();
}

void fan_2d::graphics::gui::text_renderer::set_position(fan::opengl::context_t* context, uint32_t i, const fan::vec2& position)
{
	const uint32_t index = i == 0 ? 0 : m_store[i - 1].m_indices;

	const fan::vec2 offset = position - get_position(context, i);

	m_store[i].m_position = position;

	for (int j = 0; j < m_store[i].m_text->size(); j++) {

		sprite::set_position(context, index + j, sprite::get_position(context, index + j) + offset);

		for (uint32_t j = 0; j < sprite::vertex_count; j++) {
			m_glsl_buffer.edit_ram_instance(
				i * sprite::vertex_count + j,
				&position,
				offset_rotation_point,
				sizeof(fan::vec2)
			);
		}

	}
}

uint32_t fan_2d::graphics::gui::text_renderer::size(fan::opengl::context_t* context) const {
	return m_store.size();
}

f32_t fan_2d::graphics::gui::text_renderer::get_font_size(fan::opengl::context_t* context, uintptr_t i) const
{
	return *(f32_t*)m_glsl_buffer.get_instance(get_index(i) * sprite::vertex_count, offset_font_size);
}

void fan_2d::graphics::gui::text_renderer::set_font_size(fan::opengl::context_t* context, uint32_t i, f32_t font_size)
{
	const auto text = get_text(context, i);

	const auto position = get_position(context, i);

	const auto color = get_text_color(context, i);

	const auto outline_color = get_outline_color(context, i);
	const auto outline_size = get_outline_size(context, i);

	this->erase(context, i);

	text_renderer::properties_t properties;
	properties.text = text;
	properties.font_size = font_size;
	properties.position = position;
	properties.text_color = color;
	properties.outline_color = outline_color;
	properties.outline_size = outline_size;

	this->insert(context, i, properties);
}

f32_t fan_2d::graphics::gui::text_renderer::get_angle(fan::opengl::context_t* context, uint32_t i) const
{
	return *(f32_t*)m_glsl_buffer.get_instance(get_index(i) * sprite::vertex_count, offset_angle);
}

void fan_2d::graphics::gui::text_renderer::set_angle(fan::opengl::context_t* context, uint32_t i, f32_t angle)
{
	for (int j = get_index(i) * sprite::vertex_count; j < get_index(i + 1) * sprite::vertex_count; j += sprite::vertex_count) {
    m_glsl_buffer.edit_ram_instance(
      j,
      &angle,
      offset_angle,
      sizeof(text_renderer::properties_t::angle)
    );
  }
  m_queue_helper.edit(
    context, 
    get_index(i) * sprite::vertex_count * m_glsl_buffer.m_element_size + offset_angle,
    get_index(i + 1) * (sprite::vertex_count) * m_glsl_buffer.m_element_size - offset_angle, 
    &m_glsl_buffer
  );
}

f32_t fan_2d::graphics::gui::text_renderer::get_rotation_point(fan::opengl::context_t* context, uint32_t i) const
{
	return *(f32_t*)m_glsl_buffer.get_instance(get_index(i) * sprite::vertex_count, offset_rotation_point);
}

void fan_2d::graphics::gui::text_renderer::set_rotation_point(fan::opengl::context_t* context, uint32_t i, const fan::vec2& rotation_point)
{
	for (int j = get_index(i) * sprite::vertex_count; j < get_index(i + 1) * sprite::vertex_count; j += sprite::vertex_count) {
    m_glsl_buffer.edit_ram_instance(
      j,
      &rotation_point,
      offset_rotation_point,
      sizeof(text_renderer::properties_t::rotation_point)
    );
  }
  m_queue_helper.edit(
    context, 
    get_index(i) * sprite::vertex_count * m_glsl_buffer.m_element_size + offset_rotation_point,
    get_index(i + 1) * (sprite::vertex_count) * m_glsl_buffer.m_element_size - offset_rotation_point, 
    &m_glsl_buffer
  );
}

fan::color fan_2d::graphics::gui::text_renderer::get_outline_color(fan::opengl::context_t* context, uint32_t i) const
{
	return *(fan::color*)m_glsl_buffer.get_instance(get_index(i) * sprite::vertex_count, offset_outline_color);
}

void fan_2d::graphics::gui::text_renderer::set_outline_color(fan::opengl::context_t* context, uint32_t i, const fan::color& outline_color)
{
	for (int j = get_index(i) * sprite::vertex_count; j < get_index(i + 1) * sprite::vertex_count; j += sprite::vertex_count) {
    m_glsl_buffer.edit_ram_instance(
      j,
      &outline_color,
      offset_outline_color,
      sizeof(text_renderer::properties_t::outline_color)
    );
  }
  m_queue_helper.edit(
    context, 
    get_index(i) * sprite::vertex_count * m_glsl_buffer.m_element_size + offset_outline_color,
    get_index(i + 1) * (sprite::vertex_count) * m_glsl_buffer.m_element_size - offset_outline_color, 
    &m_glsl_buffer
  );
}

f32_t fan_2d::graphics::gui::text_renderer::get_outline_size(fan::opengl::context_t* context, uint32_t i) const
{
	return *(f32_t*)m_glsl_buffer.get_instance(get_index(i) * sprite::vertex_count, offset_outline_size);
}

void fan_2d::graphics::gui::text_renderer::set_outline_size(fan::opengl::context_t* context, uint32_t i, f32_t outline_size)
{
	for (int j = get_index(i) * sprite::vertex_count; j < get_index(i + 1) * sprite::vertex_count; j += sprite::vertex_count) {
    m_glsl_buffer.edit_ram_instance(
      j,
      &outline_size,
      offset_outline_size,
      sizeof(text_renderer::properties_t::outline_color)
    );
  }
  m_queue_helper.edit(
    context, 
    get_index(i) * sprite::vertex_count * m_glsl_buffer.m_element_size + offset_outline_size,
    get_index(i + 1) * (sprite::vertex_count) * m_glsl_buffer.m_element_size - offset_outline_size, 
    &m_glsl_buffer
  );
}

void fan_2d::graphics::gui::text_renderer::erase(fan::opengl::context_t* context, uintptr_t i) {

	sprite::erase(context, get_index(i), get_index(i + 1));

	m_store.erase(i);

	this->regenerate_indices();
}

void fan_2d::graphics::gui::text_renderer::erase(fan::opengl::context_t* context, uintptr_t begin, uintptr_t end) {

	sprite::erase(context, get_index(begin), get_index(end));

	m_store.erase(begin, end);

	this->regenerate_indices();
}

void fan_2d::graphics::gui::text_renderer::clear(fan::opengl::context_t* context) {

	sprite::clear(context);

	m_store.clear();

	fan::throw_error("erase from glsl_buffer");
}

void fan_2d::graphics::gui::text_renderer::set_text(fan::opengl::context_t* context, uint32_t i, const fan::utf16_string& text)
{
	fan::utf16_string str = text;

	if (str.empty()) {
		str.resize(1);
	}

	auto font_size = this->get_font_size(context, i);
	auto position = this->get_position(context, i);
	auto color = this->get_text_color(context, i);

	const auto outline_color = get_outline_color(context, i);
	const auto outline_size = get_outline_size(context, i);

	this->erase(context, i);

	text_renderer::properties_t properties;

	properties.text = text;
	properties.font_size = font_size;
	properties.position = position;
	properties.text_color = color;
	properties.outline_color = outline_color;
	properties.outline_size = outline_size;

	this->insert(context, i, properties);
}

fan::color fan_2d::graphics::gui::text_renderer::get_text_color(fan::opengl::context_t* context, uint32_t i, uint32_t j) const
{
	return *(fan::color*)m_glsl_buffer.get_instance(get_index(i) * rectangle::vertex_count + j, offset_color);
}

void fan_2d::graphics::gui::text_renderer::set_text_color(fan::opengl::context_t* context, uint32_t i, const fan::color& color)
{
	for (int j = get_index(i) * sprite::vertex_count; j < get_index(i + 1) * sprite::vertex_count; j += sprite::vertex_count) {
    m_glsl_buffer.edit_ram_instance(
      j,
      &color,
      offset_color,
      sizeof(fan::color)
    );
  }
  m_queue_helper.edit(
    context, 
    get_index(i) * sprite::vertex_count * m_glsl_buffer.m_element_size + offset_color,
    get_index(i + 1) * (sprite::vertex_count) * m_glsl_buffer.m_element_size - (m_glsl_buffer.m_element_size - offset_color), 
    &m_glsl_buffer
  );
}
////
////void fan_2d::graphics::gui::text_renderer::set_text_color(uint32_t i, uint32_t j, const fan::color& color)
////{
////	sprite::set_color(get_index(i) + j, color);
////}
#endif