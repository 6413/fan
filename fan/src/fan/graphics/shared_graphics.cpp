#include <fan/graphics/graphics.hpp>

fan_2d::graphics::rounded_rectangle::rounded_rectangle(fan::camera* camera) : fan_2d::graphics::vertice_vector(camera) { }

void fan_2d::graphics::rounded_rectangle::push_back(const rounded_rectangle::properties_t& property)
{
	fan::vec2 box_size = property.size * 2;
	f32_t box_radius = property.radius;

	if (property.radius * 2 > box_size.x) {
		box_radius = box_size.x / 2;
	}
	if (property.radius * 2 > box_size.y) {
		box_radius = box_size.y / 2;
	}

	fan_2d::graphics::rounded_rectangle::properties_t properties = property;

	properties.position = property.position + fan::vec2(box_radius, 0) - properties.size;
	properties.radius = property.radius;
	properties.size = property.size * 2;
	properties.rotation_point = properties.position;

	fan_2d::graphics::vertice_vector::push_back(properties);

	properties.position = property.position + fan::vec2(box_size.x - box_radius, 0) - properties.size / 2;

	fan_2d::graphics::vertice_vector::push_back(properties);

	f32_t increment = fan::math::pi / 2 / m_segments;

	for (f32_t theta = fan::math::pi / 2 * 3; theta <= fan::math::pi / 2 * 4; theta += increment) {
		auto coord = fan::vec2(cos(theta), sin(theta));

		properties.position = property.position + fan::vec2(box_size.x - box_radius, box_radius) + coord * box_radius - properties.size / 2;
		
		fan_2d::graphics::vertice_vector::push_back(properties);
	}

	properties.position = property.position + fan::vec2(box_size.x, box_size.y - box_radius) - properties.size / 2;

	fan_2d::graphics::vertice_vector::push_back(properties);

	for (f32_t theta = fan::math::pi / 2 * 4; theta <= fan::math::pi / 2 * 5; theta += increment) {

		auto coord = fan::vec2(cos(theta), sin(theta));

		properties.position = property.position + fan::vec2(box_size.x - box_radius, box_size.y - box_radius) + coord * box_radius - properties.size / 2;

		fan_2d::graphics::vertice_vector::push_back(properties);
	}

	properties.position = property.position + fan::vec2(box_radius, box_size.y) - properties.size / 2;

	fan_2d::graphics::vertice_vector::push_back(properties);

	for (f32_t theta = fan::math::pi / 2 * 5; theta <= fan::math::pi / 2 * 6; theta += increment) {

		auto coord = fan::vec2(cos(theta), sin(theta));

		properties.position = property.position + fan::vec2(box_radius, box_size.y - box_radius) + coord * box_radius - properties.size / 2;

		fan_2d::graphics::vertice_vector::push_back(properties);
	}

	properties.position = property.position + fan::vec2(0, box_radius) - properties.size / 2;

	fan_2d::graphics::vertice_vector::push_back(properties);

	for (f32_t theta = fan::math::pi / 2 * 6; theta <= fan::math::pi / 2 * 7; theta += increment) {

		auto coord = fan::vec2(cos(theta), sin(theta));

		properties.position = property.position + fan::vec2(box_radius, box_radius) + coord * box_radius - properties.size / 2;

		fan_2d::graphics::vertice_vector::push_back(properties);
	}

	if (!total_points) {
		total_points = fan_2d::graphics::vertice_vector::size();
	}

	m_position.emplace_back(property.position);
	m_size.emplace_back(box_size / 2);
	m_radius.emplace_back(box_radius);
}

fan::vec2 fan_2d::graphics::rounded_rectangle::get_position(uintptr_t i) const
{
	return fan_2d::graphics::rounded_rectangle::m_position[i];
}

void fan_2d::graphics::rounded_rectangle::set_position(uintptr_t i, const fan::vec2& position)
{
	const auto distance = position - fan_2d::graphics::rounded_rectangle::get_position(i);
	for (uintptr_t j = 0; j < total_points; j++) {
		fan_2d::graphics::vertice_vector::set_position(total_points * i + j, fan_2d::graphics::vertice_vector::get_position(total_points * i + j) + distance);
	}

	fan_2d::graphics::rounded_rectangle::m_position[i] = position;
}

fan::vec2 fan_2d::graphics::rounded_rectangle::get_size(uintptr_t i) const
{
	return fan_2d::graphics::rounded_rectangle::m_size[i];
}

void fan_2d::graphics::rounded_rectangle::set_size(uintptr_t i, const fan::vec2& size)
{
	if (m_size[i] == size) {
		return;
	}

	uint32_t offset = i * total_points;

	const auto position = get_position(i);
	const auto radius = get_radius(i);

	m_size[i] = size;

	if (size.x < radius * 2) {
		m_size[i].x = radius * 2;
	}
	if (size.y < radius * 2) {
		m_size[i].y = radius * 2;
	}

	fan_2d::graphics::vertice_vector::set_position(offset++, position + fan::vec2(radius, 0));

	fan_2d::graphics::vertice_vector::set_position(offset++,position + fan::vec2(m_size[i].x - radius, 0));

	f32_t increment = fan::math::pi / 2 / m_segments;

	for (f32_t theta = fan::math::pi / 2 * 3; theta <= fan::math::pi / 2 * 4; theta += increment) {
		auto coord = fan::vec2(cos(theta), sin(theta));

		fan_2d::graphics::vertice_vector::set_position(offset++, position + fan::vec2(m_size[i].x - radius, radius) + coord * radius);
	}

	fan_2d::graphics::vertice_vector::set_position(offset++, position + fan::vec2(m_size[i].x, m_size[i].y - radius));

	for (f32_t theta = fan::math::pi / 2 * 4; theta <= fan::math::pi / 2 * 5; theta += increment) {
		auto coord = fan::vec2(cos(theta), sin(theta));

		fan_2d::graphics::vertice_vector::set_position(offset++, position + fan::vec2(m_size[i].x - radius, m_size[i].y - radius) + coord * radius);
	}

	fan_2d::graphics::vertice_vector::set_position(offset++, position + fan::vec2(radius, m_size[i].y));

	for (f32_t theta = fan::math::pi / 2 * 5; theta <= fan::math::pi / 2 * 6; theta += increment) {
		auto coord = fan::vec2(cos(theta), sin(theta));

		fan_2d::graphics::vertice_vector::set_position(offset++, position + fan::vec2(radius, m_size[i].y - radius) + coord * radius);
	}

	fan_2d::graphics::vertice_vector::set_position(offset++, position + fan::vec2(0, radius));

	for (f32_t theta = fan::math::pi / 2 * 6; theta <= fan::math::pi / 2 * 7; theta += increment) {
		auto coord = fan::vec2(cos(theta), sin(theta));

		fan_2d::graphics::vertice_vector::set_position(offset++, position + fan::vec2(radius, radius) + coord * radius);
	}
}

f32_t fan_2d::graphics::rounded_rectangle::get_radius(uintptr_t i) const
{
	return m_radius[i];
}

void fan_2d::graphics::rounded_rectangle::set_radius(uintptr_t i, f32_t radius)
{
	if (m_radius[i] == radius || radius * 2 > this->get_size(i).x || radius * 2 > this->get_size(i).y) {
		return;
	}

	m_radius[i] = radius;

	uint32_t offset = i * total_points;

	const auto position = get_position(i);
	const auto size = get_size(i);

	fan_2d::graphics::vertice_vector::set_position(offset++, position + fan::vec2(radius, 0));

	fan_2d::graphics::vertice_vector::set_position(offset++,position + fan::vec2(size.x - radius, 0));

	f32_t increment = fan::math::pi / 2 / m_segments;

	for (f32_t theta = fan::math::pi / 2 * 3; theta <= fan::math::pi / 2 * 4; theta += increment) {
		auto coord = fan::vec2(cos(theta), sin(theta));

		fan_2d::graphics::vertice_vector::set_position(offset++, position + fan::vec2(size.x - radius, radius) + coord * radius);
	}

	fan_2d::graphics::vertice_vector::set_position(offset++, position + fan::vec2(size.x, size.y - radius));

	for (f32_t theta = fan::math::pi / 2 * 4; theta <= fan::math::pi / 2 * 5; theta += increment) {
		auto coord = fan::vec2(cos(theta), sin(theta));

		fan_2d::graphics::vertice_vector::set_position(offset++, position + fan::vec2(size.x - radius, size.y - radius) + coord * radius);
	}

	fan_2d::graphics::vertice_vector::set_position(offset++, position + fan::vec2(radius, size.y));

	for (f32_t theta = fan::math::pi / 2 * 5; theta <= fan::math::pi / 2 * 6; theta += increment) {
		auto coord = fan::vec2(cos(theta), sin(theta));

		fan_2d::graphics::vertice_vector::set_position(offset++, position + fan::vec2(radius, size.y - radius) + coord * radius);
	}

	fan_2d::graphics::vertice_vector::set_position(offset++, position + fan::vec2(0, radius));

	for (f32_t theta = fan::math::pi / 2 * 6; theta <= fan::math::pi / 2 * 7; theta += increment) {
		auto coord = fan::vec2(cos(theta), sin(theta));

		fan_2d::graphics::vertice_vector::set_position(offset++, position + fan::vec2(radius, radius) + coord * radius);
	}
}

bool fan_2d::graphics::rounded_rectangle::inside(uintptr_t i) const
{
	const auto position = m_camera->m_window->get_mouse_position();

	if (position.x > m_position[i].x - m_radius[i] &&
		position.x < m_position[i].x + m_size[i].x + m_radius[i] &&
		position.y > m_position[i].y && 
		position.y < m_position[i].y + m_size[i].y)
	{
		return true;
	}

	if (position.x > m_position[i].x &&
		position.x < m_position[i].x + m_size[i].x &&
		position.y > m_position[i].y - m_radius[i] && 
		position.y < m_position[i].y + m_size[i].y + m_radius[i])
	{
		return true;
	}

	return false;
}

fan::color fan_2d::graphics::rounded_rectangle::get_color(uintptr_t i) const {
	return vertice_vector::get_color(total_points * i);
}

void fan_2d::graphics::rounded_rectangle::set_color(uintptr_t i, const fan::color& color)
{
	for (int i = 0; i < total_points; i++) {
		vertice_vector::set_color(i * total_points, color);
	}
}

uint32_t fan_2d::graphics::rounded_rectangle::size() const
{
	return m_size.size();
}

void fan_2d::graphics::rounded_rectangle::write_data() {
	vertice_vector::write_data();
}

void fan_2d::graphics::rounded_rectangle::edit_data(uint32_t i) {
	vertice_vector::edit_data(i * total_points, i * total_points + total_points);
}

void fan_2d::graphics::rounded_rectangle::edit_data(uint32_t begin, uint32_t end) {
	vertice_vector::edit_data(begin * total_points, (end - begin + 1) * total_points);
}

void fan_2d::graphics::rounded_rectangle::enable_draw()
{
	vertice_vector::enable_draw(fan_2d::graphics::shape::triangle_fan, total_points);
}

void fan_2d::graphics::rounded_rectangle::disable_draw()
{
	vertice_vector::disable_draw();
}
