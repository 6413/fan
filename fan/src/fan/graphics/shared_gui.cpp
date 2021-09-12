#include <fan/graphics/graphics.hpp>

#include <fan/graphics/themes.hpp>

#include <fan/graphics/shared_gui.hpp>

fan_2d::graphics::gui::rectangle_text_box::rectangle_text_box(fan::camera* camera, fan_2d::graphics::gui::theme theme)
	: 
	inner_rect_t(camera),
	outer_rect_t(camera),
	fan_2d::graphics::gui::text_renderer(camera), theme(theme) {}

void fan_2d::graphics::gui::rectangle_text_box::push_back(const rectangle_button_properties& properties)
{
	m_properties.emplace_back(properties);

	switch (properties.text_position) {
		case text_position_e::left:
		{
			fan_2d::graphics::gui::text_renderer::push_back(
				properties.text,
				properties.font_size,
				fan::vec2(properties.position.x + theme.text_button.outline_thickness, properties.position.y + properties.border_size.y * 0.5),
				theme.text_button.text_color
			);

			break;
		}
		case text_position_e::middle:
		{
			fan_2d::graphics::gui::text_renderer::push_back(
				properties.text,
				properties.font_size,
				fan::vec2(properties.position.x + properties.border_size.x * 0.5, properties.position.y + properties.border_size.y * 0.5),
				theme.text_button.text_color
			);

			break;
		}
	}

	inner_rect_t::push_back({ properties.position, get_button_size(m_properties.size() - 1), 0, 0, fan::vec3(0, 0, 1), theme.text_button.color });

	auto corners = inner_rect_t::get_corners(m_properties.size() - 1);

	const f32_t t = theme.text_button.outline_thickness;

	corners[1].x -= t;
	corners[3].x -= t;

	corners[2].y -= t;
	corners[3].y -= t;

	outer_rect_t::push_back({ corners[0], fan::vec2(corners[1].x - corners[0].x + t, t), 0, 0, fan::vec3(0, 0, 1), theme.text_button.outline_color });
	outer_rect_t::push_back({ corners[1], fan::vec2(t, corners[3].y - corners[1].y + t), 0, 0, fan::vec3(0, 0, 1), theme.text_button.outline_color });
	outer_rect_t::push_back({ corners[2], fan::vec2(corners[3].x - corners[2].x + t, t), 0, 0, fan::vec3(0, 0, 1), theme.text_button.outline_color });
	outer_rect_t::push_back({ corners[0], fan::vec2(t, corners[2].y - corners[0].y + t), 0, 0, fan::vec3(0, 0, 1), theme.text_button.outline_color });
}

void fan_2d::graphics::gui::rectangle_text_box::draw(uint32_t begin, uint32_t end)
{
	// depth test
	inner_rect_t::draw(begin == (uint32_t)-1 ? 0 : begin, end == (uint32_t)-1 ? outer_rect_t::size() : end);		
	outer_rect_t::draw(begin == (uint32_t)-1 ? 0 : begin, end == (uint32_t)-1 ? outer_rect_t::size() : end);

	fan_2d::graphics::gui::text_renderer::draw();
}

bool fan_2d::graphics::gui::rectangle_text_box::inside(uintptr_t i, const fan::vec2& position) const
{
	return inner_rect_t::inside(i, position);
}

fan::camera* fan_2d::graphics::gui::rectangle_text_box::get_camera()
{
	return inner_rect_t::m_camera;
}

uintptr_t fan_2d::graphics::gui::rectangle_text_box::size() const
{
	return inner_rect_t::size();
}

void fan_2d::graphics::gui::rectangle_text_box::write_data() {
	inner_rect_t::write_data();
	outer_rect_t::write_data();
	graphics::gui::text_renderer::write_data();
}

void fan_2d::graphics::gui::rectangle_text_box::edit_data(uint32_t i) {
	inner_rect_t::edit_data(i);
	outer_rect_t::edit_data(i);
	text_renderer::edit_data(i);
}

void fan_2d::graphics::gui::rectangle_text_box::edit_data(uint32_t begin, uint32_t end) {
	inner_rect_t::edit_data(begin, end);
	outer_rect_t::edit_data(begin, end);
	graphics::gui::text_renderer::edit_data(begin, end);
}

fan_2d::graphics::gui::rectangle_text_button::rectangle_text_button(fan::camera* camera, fan_2d::graphics::gui::theme theme)
	: 
	fan_2d::graphics::gui::rectangle_text_box(camera, theme),
	base::mouse(*this) {

	rectangle_text_button::mouse::on_hover<0>([&] (uint32_t i) {

		if (m_held_button_id[i] != (uint32_t)fan::uninitialized) {
			return;
		}

		inner_rect_t::set_color(i, theme.text_button.hover_color);

		outer_rect_t::set_color(i * 4 + 0, theme.text_button.hover_outline_color);
		outer_rect_t::set_color(i * 4 + 1, theme.text_button.hover_outline_color);
		outer_rect_t::set_color(i * 4 + 2, theme.text_button.hover_outline_color);
		outer_rect_t::set_color(i * 4 + 3, theme.text_button.hover_outline_color);

		inner_rect_t::edit_data(i);

		outer_rect_t::edit_data(i * 4, i * 4 + 3);
		});

	rectangle_text_button::mouse::on_exit<0>([&] (uint32_t i) {
		if (m_held_button_id[i] != (uint32_t)fan::uninitialized) {
			return;
		}

		inner_rect_t::set_color(i, theme.text_button.color);

		outer_rect_t::set_color(i * 4 + 0, theme.text_button.outline_color);
		outer_rect_t::set_color(i * 4 + 1, theme.text_button.outline_color);
		outer_rect_t::set_color(i * 4 + 2, theme.text_button.outline_color);
		outer_rect_t::set_color(i * 4 + 3, theme.text_button.outline_color);

		inner_rect_t::edit_data(i);

		outer_rect_t::edit_data(i * 4, i * 4 + 3);
		});

	rectangle_text_button::mouse::on_click<0>([&] (uint32_t i) {

		inner_rect_t::set_color(i, theme.text_button.click_color);

		outer_rect_t::set_color(i * 4 + 0, theme.text_button.click_outline_color);
		outer_rect_t::set_color(i * 4 + 1, theme.text_button.click_outline_color);
		outer_rect_t::set_color(i * 4 + 2, theme.text_button.click_outline_color);
		outer_rect_t::set_color(i * 4 + 3, theme.text_button.click_outline_color);

		inner_rect_t::edit_data(i);

		outer_rect_t::edit_data(i * 4, i * 4 + 3);
		});

	rectangle_text_button::mouse::on_release<0>([&] (uint32_t i) {

		if (mouse::m_hover_button_id[0] != (uint32_t)fan::uninitialized) {

			inner_rect_t::set_color(i, theme.text_button.hover_color);

			outer_rect_t::set_color(i * 4 + 0, theme.text_button.hover_outline_color);
			outer_rect_t::set_color(i * 4 + 1, theme.text_button.hover_outline_color);
			outer_rect_t::set_color(i * 4 + 2, theme.text_button.hover_outline_color);
			outer_rect_t::set_color(i * 4 + 3, theme.text_button.hover_outline_color);

			inner_rect_t::edit_data(i);

			outer_rect_t::edit_data(i * 4, i * 4 + 3);

			return;
		}

		inner_rect_t::set_color(i, theme.text_button.color);

		outer_rect_t::set_color(i * 4 + 0, theme.text_button.outline_color);
		outer_rect_t::set_color(i * 4 + 1, theme.text_button.outline_color);
		outer_rect_t::set_color(i * 4 + 2, theme.text_button.outline_color);
		outer_rect_t::set_color(i * 4 + 3, theme.text_button.outline_color);

		inner_rect_t::edit_data(i);

		outer_rect_t::edit_data(i * 4, i * 4 + 3);
		});

	rectangle_text_button::mouse::on_outside_release([&] (uint32_t i) {
		inner_rect_t::set_color(i, theme.text_button.color);

		outer_rect_t::set_color(i * 4 + 0, theme.text_button.outline_color);
		outer_rect_t::set_color(i * 4 + 1, theme.text_button.outline_color);
		outer_rect_t::set_color(i * 4 + 2, theme.text_button.outline_color);
		outer_rect_t::set_color(i * 4 + 3, theme.text_button.outline_color);

		inner_rect_t::edit_data(i);

		outer_rect_t::edit_data(i * 4, i * 4 + 3);
		});
}
fan_2d::graphics::gui::sprite_text_box::sprite_text_box(fan::camera* camera, const std::string& path)
	:  sprite_t(camera), fan_2d::graphics::gui::text_renderer(camera), image(fan_2d::graphics::load_image(camera->m_window, path)) { }

fan::camera* fan_2d::graphics::gui::sprite_text_box::get_camera()
{
	return sprite_t::m_camera;
}

uint64_t fan_2d::graphics::gui::sprite_text_box::size() const {
	return sprite_t::size();
}

bool fan_2d::graphics::gui::sprite_text_box::inside(uint32_t i, const fan::vec2& position) const {
	return sprite_t::inside(i, position);
}

void fan_2d::graphics::gui::sprite_text_box::push_back(const sprite_button_properties& properties)
{
	m_properties.emplace_back(properties);

	switch (properties.text_position) {
		case text_position_e::left:
		{
			fan_2d::graphics::gui::text_renderer::push_back(
				properties.text,
				properties.font_size,
				fan::vec2(properties.position.x, properties.position.y + properties.border_size.y * 0.5),
				fan_2d::graphics::gui::defaults::text_color
			);

			break;
		}
		case text_position_e::middle:
		{
			fan_2d::graphics::gui::text_renderer::push_back(
				properties.text,
				properties.font_size,
				fan::vec2(properties.position.x + properties.border_size.x * 0.5, properties.position.y + properties.border_size.y * 0.5),
				fan_2d::graphics::gui::defaults::text_color
			);

			break;
		}
	}

	sprite_t::properties_t s_properties;
	s_properties.image = image;
	s_properties.position = properties.position;
	s_properties.size = get_button_size(m_properties.size() - 1);

	sprite_t::push_back(s_properties);
}

void fan_2d::graphics::gui::sprite_text_box::draw(uint32_t begin, uint32_t end)
{
	sprite_t::draw(begin, end);
	fan_2d::graphics::gui::text_renderer::draw();
}

fan_2d::graphics::gui::sprite_text_button::sprite_text_button(fan::camera* camera, const std::string& path)
	: fan_2d::graphics::gui::sprite_text_box(camera, path),
	base::mouse(*this) {}

fan_2d::graphics::gui::checkbox::checkbox(fan::camera* camera, fan_2d::graphics::gui::theme theme)
	: checkbox::rectangle_t(camera), 
	checkbox::line_t(camera), 
	checkbox::text_renderer_t(camera),
	 fan_2d::graphics::gui::base::mouse(*this),
	 m_theme(theme) {

	fan_2d::graphics::gui::base::mouse::on_hover<0>([&] (uint32_t i) {
		checkbox::rectangle_t::set_color(i, m_theme.checkbox.hover_color);
		this->edit_data(i);
	});

	fan_2d::graphics::gui::base::mouse::on_exit<0>([&] (uint32_t i) {
		checkbox::rectangle_t::set_color(i, m_theme.checkbox.color);
		this->edit_data(i);
	});

	fan_2d::graphics::gui::base::mouse::on_click<0>([&](uint32_t i) {
		checkbox::rectangle_t::set_color(i, m_theme.checkbox.click_color);
		this->edit_data(i);
	});

	fan_2d::graphics::gui::base::mouse::on_release<0>([&](uint32_t i) {
		checkbox::rectangle_t::set_color(i, m_theme.checkbox.hover_color);

		m_visible[i] = !m_visible[i];

		if (m_visible[i]) {
			if (m_on_check) {
				m_on_check(i);
			}
		}
		else {
			if (m_on_uncheck) {
				m_on_uncheck(i);
			}
		}
		this->edit_data(i);
	});

	fan_2d::graphics::gui::base::mouse::on_outside_release<0>([&](uint32_t i) {
		checkbox::rectangle_t::set_color(i, m_theme.checkbox.color);
		this->edit_data(i);
	});
}

void fan_2d::graphics::gui::checkbox::push_back(const checkbox::properties_t& property)
{
	m_properties.emplace_back(property);

	m_visible.emplace_back(property.checked);
	
	f32_t text_middle_height = fan_2d::graphics::gui::text_renderer::get_average_text_height(property.text, property.font_size);

	fan_2d::graphics::rectangle::properties_t properties;
	properties.position = property.position;
	properties.size = text_middle_height * property.box_size_multiplier;
	properties.color = m_theme.checkbox.color;
	properties.rotation_point = properties.position;

	checkbox::rectangle_t::push_back(properties); 

	checkbox::line_t::push_back(property.position, property.position + text_middle_height * property.box_size_multiplier, m_theme.checkbox.check_color, property.line_thickness); // might be varying position
	checkbox::line_t::push_back(property.position + fan::vec2(text_middle_height * property.box_size_multiplier, 0), property.position + fan::vec2(0, text_middle_height * property.box_size_multiplier), m_theme.checkbox.check_color, property.line_thickness); // might be varying position

	auto diff = (convert_font_size(property.font_size * property.box_size_multiplier) - text_middle_height) / 2 + properties.size.y / 2 - text_middle_height / 2;

	fan_2d::graphics::gui::text_renderer::push_back(property.text, property.font_size, property.position + fan::vec2(property.font_size * property.box_size_multiplier, diff), m_theme.checkbox.text_color);
}

void fan_2d::graphics::gui::checkbox::draw()
{
	// depth test
	//fan_2d::graphics::draw([&] {
		
		for (int i = 0; i < fan_2d::graphics::line::size() / 2; i++) {
			checkbox::rectangle_t::draw();

			if (m_visible[i]) {
				checkbox::line_t::draw();
				//fan_2d::graphics::line::draw(i * 2 + 1, i * 2 + 2);
			}
		}

		checkbox::text_renderer_t::draw();
	//});
}

void fan_2d::graphics::gui::checkbox::on_check(std::function<void(uint32_t i)> function)
{
	m_on_check = function;
}

void fan_2d::graphics::gui::checkbox::on_uncheck(std::function<void(uint32_t i)> function)
{
	m_on_uncheck = function;
}

uint32_t fan_2d::graphics::gui::checkbox::size() const
{
	return rectangle_t::size();
}

bool fan_2d::graphics::gui::checkbox::inside(uint32_t i, const fan::vec2& position) const
{
	return rectangle_t::inside(i, position);
}

fan::camera* fan_2d::graphics::gui::checkbox::get_camera()
{
	return checkbox::rectangle_t::m_camera;
}

void fan_2d::graphics::gui::checkbox::write_data() {
	checkbox::line_t::write_data();
	checkbox::rectangle_t::write_data();
	checkbox::text_renderer_t::write_data();
}

void fan_2d::graphics::gui::checkbox::edit_data(uint32_t i) {
	checkbox::line_t::edit_data(i);
	checkbox::rectangle_t::edit_data(i);
	checkbox::text_renderer_t::edit_data(i);
}
void fan_2d::graphics::gui::checkbox::edit_data(uint32_t begin, uint32_t end) {
	checkbox::line_t::edit_data(begin, end);
	checkbox::rectangle_t::edit_data(begin, end);
	checkbox::text_renderer_t::edit_data(begin, end);
}