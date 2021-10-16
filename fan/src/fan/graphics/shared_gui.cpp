#include <fan/graphics/graphics.hpp>

#include <fan/graphics/themes.hpp>

#include <fan/graphics/shared_gui.hpp>

fan_2d::graphics::gui::rectangle_text_box_sized::rectangle_text_box_sized(fan::camera* camera, fan_2d::graphics::gui::theme theme)
	:
	inner_rect_t(camera),
	outer_rect_t(camera),
	fan_2d::graphics::gui::text_renderer(camera), theme(theme) {}

void fan_2d::graphics::gui::rectangle_text_box_sized::push_back(const properties_t& property)
{
	m_properties.emplace_back(property);

	const auto str = property.place_holder.empty() ? property.text : property.place_holder;

	switch (property.text_position) {
		case text_position_e::left:
		{
			f32_t line_height = font_info.font['\n'].size.y * convert_font_size(property.font_size);

			fan_2d::graphics::gui::text_renderer::push_back(
				str,
				property.font_size,
				fan::vec2(property.position.x + theme.text_button.outline_thickness + property.advance, property.position.y + property.size.y * 0.5 - line_height * 0.5 + theme.text_button.outline_thickness),
				str[0] == '\0' ? theme.text_button.text_color : defaults::text_color_place_holder
			);

			break;
		}
		case text_position_e::middle:
		{
			auto text_size = get_text_size(str, property.font_size);

			f32_t line_height = font_info.font['\n'].size.y * convert_font_size(property.font_size);

			fan_2d::graphics::gui::text_renderer::push_back(
				str,
				property.font_size,
				fan::vec2(property.position.x + property.size.x * 0.5 - text_size.x * 0.5  + theme.text_button.outline_thickness, property.position.y + property.size.y * 0.5 - line_height * 0.5  + theme.text_button.outline_thickness),
				property.text.size() && property.text[0] != '\0' ? theme.text_button.text_color : defaults::text_color_place_holder
			);

			break;
		}
	}

	inner_rect_t::properties_t rect_properties;
	rect_properties.position = property.position;
	rect_properties.size = property.size;
	rect_properties.rotation_point = rect_properties.position;
	rect_properties.color = theme.text_button.color;

	inner_rect_t::push_back(rect_properties);

	auto corners = inner_rect_t::get_corners(m_properties.size() - 1);

	const f32_t t = theme.text_button.outline_thickness;

	corners[1].x -= t;
	corners[3].x -= t;

	corners[2].y -= t;
	corners[3].y -= t;

	outer_rect_t::properties_t properties;
	properties.position = corners[0];
	properties.rotation_point = properties.position;
	properties.size = fan::vec2(corners[1].x - corners[0].x + t, t);
	properties.color = theme.text_button.outline_color;

	outer_rect_t::push_back(properties);
	properties.position = corners[1];
	properties.size = fan::vec2(t, corners[3].y - corners[1].y + t);

	outer_rect_t::push_back(properties);

	properties.position = corners[2];
	properties.size = fan::vec2(corners[3].x - corners[2].x + t, t);

	outer_rect_t::push_back(properties);

	properties.position = corners[0];
	properties.size = fan::vec2(t, corners[2].y - corners[0].y + t);

	outer_rect_t::push_back(properties);
}

void fan_2d::graphics::gui::rectangle_text_box_sized::draw(uint32_t begin, uint32_t end)
{
	// depth test
	inner_rect_t::draw(begin == (uint32_t)-1 ? 0 : begin, end == (uint32_t)-1 ? inner_rect_t::size() : end);		
	outer_rect_t::draw(begin == (uint32_t)-1 ? 0 : begin, end == (uint32_t)-1 ? outer_rect_t::size() : end);

	fan_2d::graphics::gui::text_renderer::draw();
}

void fan_2d::graphics::gui::rectangle_text_box_sized::set_position(uint32_t i, const fan::vec2& position)
{
	const auto offset = position - rectangle_text_box_sized::get_position(i);

	for (int j = 0; j < 4; j++) {
		rectangle_text_box_sized::outer_rect_t::set_position(i + j, rectangle_text_box_sized::outer_rect_t::get_position(i + j) + offset);
	}

	rectangle_text_box_sized::inner_rect_t::set_position(i, inner_rect_t::get_position(i) + offset);
	rectangle_text_box_sized::text_renderer::set_position(i, rectangle_text_box_sized::text_renderer::get_position(i) + offset);
}

bool fan_2d::graphics::gui::rectangle_text_box_sized::inside(uintptr_t i, const fan::vec2& position) const
{
	return inner_rect_t::inside(i, position);
}

void fan_2d::graphics::gui::rectangle_text_box_sized::set_text(uint32_t i, const fan::utf16_string& text) {

	switch (m_properties[i].text_position) {
		case text_position_e::left:
		{
			fan_2d::graphics::gui::text_renderer::set_text(i, text);
			break;
		}
		case text_position_e::middle:
		{
			auto position = inner_rect_t::get_position(i);
			auto size = inner_rect_t::get_size(i);
			auto text_size = text_renderer::get_text_size(text, text_renderer::get_font_size(i));

			f32_t line_height = font_info.font['\n'].size.y * convert_font_size(get_font_size(i));

			fan_2d::graphics::gui::text_renderer::set_text(i, text);

			fan_2d::graphics::gui::text_renderer::set_position(i, fan::vec2(position.x + size.x * 0.5 - text_size.x * 0.5, position.y + size.y * 0.5 - line_height * 0.5));

			break;
		}
	}
	
	auto corners = inner_rect_t::get_corners(i);

	const f32_t t = theme.text_button.outline_thickness;

	corners[1].x -= t;
	corners[3].x -= t;

	corners[2].y -= t;
	corners[3].y -= t;

	outer_rect_t::set_position(i * 4, corners[0]);
	outer_rect_t::set_size(i * 4, fan::vec2(corners[1].x - corners[0].x + t, t));

	outer_rect_t::set_position(i * 4 + 1, corners[1]);
	outer_rect_t::set_size(i * 4 + 1, fan::vec2(t, corners[3].y - corners[1].y + t));

	outer_rect_t::set_position(i * 4 + 2, corners[2]);
	outer_rect_t::set_size(i * 4 + 2, fan::vec2(corners[3].x - corners[2].x + t, t));

	outer_rect_t::set_position(i * 4 + 3, corners[0]);
	outer_rect_t::set_size(i * 4 + 3, fan::vec2(t, corners[2].y - corners[0].y + t));
}

fan::color fan_2d::graphics::gui::rectangle_text_box_sized::get_text_color(uint32_t i) const
{
	return text_renderer::get_text_color(i);
}

void fan_2d::graphics::gui::rectangle_text_box_sized::set_text_color(uint32_t i, const fan::color& color)
{
	text_renderer::set_text_color(i, color);
}

fan::vec2 fan_2d::graphics::gui::rectangle_text_box_sized::get_position(uint32_t i) const
{
	return inner_rect_t::get_position(i);
}

fan::vec2 fan_2d::graphics::gui::rectangle_text_box_sized::get_size(uint32_t i) const
{
	return inner_rect_t::get_size(i);
}

f32_t fan_2d::graphics::gui::rectangle_text_box_sized::get_font_size(uint32_t i) const
{
	return text_renderer::get_font_size(i);
}

fan::color fan_2d::graphics::gui::rectangle_text_box_sized::get_color(uint32_t i) const
{
	return inner_rect_t::get_color(i);
}

fan_2d::graphics::gui::cursor_src_dst_t fan_2d::graphics::gui::rectangle_text_box_sized::get_cursor(uint32_t i, uint32_t x, uint32_t y)
{
	f32_t converted = fan_2d::graphics::gui::text_renderer::convert_font_size(this->get_font_size(i));
	auto line_height = fan_2d::graphics::gui::text_renderer::font_info.font['\n'].size.y * converted;

	fan::vec2 src, dst;

	if (m_properties[i].text_position == fan_2d::graphics::gui::text_position_e::left) {

		src += (this->get_position(i) + fan::vec2(0, this->get_size(i).y / 2) + theme.text_button.outline_thickness);

		src.y -= line_height / 2;

		src.x += m_properties[i].advance;
	}
	else if (m_properties[i].text_position == fan_2d::graphics::gui::text_position_e::middle) {
		fan::vec2 text_size;

		if (text_renderer::get_text(i).empty()) {
			text_size = fan::vec2(0, line_height / 2);
			text_size.y = font_info.font['\n'].size.y * convert_font_size(get_font_size(i));
		}
		else {
			text_size.y = font_info.font['\n'].size.y * convert_font_size(get_font_size(i));
		}

		src += this->get_position(i) + this->get_size(i) * 0.5 - text_size * 0.5 + fan::vec2(theme.text_button.outline_thickness, 0);
	}

	uint32_t offset = 0;

	auto str = this->get_text(i);

	for (int j = 0; j < y; j++) {
		while (str[offset++] != '\n') {
			if (offset >= str.size()) {
				throw std::runtime_error("string didnt have endline");
			}
		}
	}

	for (int j = 0; j < x; j++) {
		wchar_t letter = str[j + offset];
		if (letter == '\n') {
			continue;
		}

		std::wstring wstr;

		wstr.push_back(letter);

		auto letter_info = fan_2d::graphics::gui::text_renderer::get_letter_info(fan::utf16_string(wstr).to_utf8().data(), this->get_font_size(i));

		if (j == x - 1) {
			src.x += letter_info.size.x + (letter_info.advance - letter_info.size.x) / 2 - 1;
		}
		else {
			src.x += letter_info.advance;
		}

	}

	src.y += line_height * y;


	dst = src + fan::vec2(0, line_height);

	dst = dst - src + fan::vec2(cursor_properties::line_thickness, 0);


	return { src, dst };
}

fan::vec2 fan_2d::graphics::gui::rectangle_text_box_sized::get_text_starting_point(uint32_t i) const
{
	fan::vec2 src;

	if (m_properties[i].text_position == fan_2d::graphics::gui::text_position_e::left) {
		src = this->get_position(i);
		src.y += font_info.font['\n'].size.y * convert_font_size(get_font_size(i));
		src.x += m_properties[i].advance;
	}
	else if (m_properties[i].text_position == fan_2d::graphics::gui::text_position_e::middle){
		auto text_size = text_renderer::get_text_size(get_text(i), text_renderer::get_font_size(i));
		text_size.y = fan_2d::graphics::gui::text_renderer::font_info.font['\n'].size.y * convert_font_size(get_font_size(i));
		src = this->get_position(i) + this->get_size(i) * 0.5 - text_size * 0.5;
	}

	return src;
}

fan_2d::graphics::gui::rectangle_text_box_sized::properties_t fan_2d::graphics::gui::rectangle_text_box_sized::get_property(uint32_t i) const
{
	return m_properties[i];
}

fan::camera* fan_2d::graphics::gui::rectangle_text_box_sized::get_camera()
{
	return inner_rect_t::m_camera;
}

uintptr_t fan_2d::graphics::gui::rectangle_text_box_sized::size() const
{
	return inner_rect_t::size();
}

void fan_2d::graphics::gui::rectangle_text_box_sized::write_data() {
	inner_rect_t::write_data();
	outer_rect_t::write_data();
	graphics::gui::text_renderer::write_data();
}

void fan_2d::graphics::gui::rectangle_text_box_sized::edit_data(uint32_t i) {
	inner_rect_t::edit_data(i);
	outer_rect_t::edit_data(i, i + 4);
	text_renderer::edit_data(i);
}

void fan_2d::graphics::gui::rectangle_text_box_sized::edit_data(uint32_t begin, uint32_t end) {
	inner_rect_t::edit_data(begin, end);
	outer_rect_t::edit_data(begin * 4, end * 4 + 4);
	graphics::gui::text_renderer::edit_data(begin, end);
}

fan_2d::graphics::gui::rectangle_text_box::rectangle_text_box(fan::camera* camera, fan_2d::graphics::gui::theme theme)
	: 
	inner_rect_t(camera),
	outer_rect_t(camera),
	fan_2d::graphics::gui::text_renderer(camera), theme(theme) {}

void fan_2d::graphics::gui::rectangle_text_box::push_back(const properties_t& property)
{
	m_properties.emplace_back(property);

	fan_2d::graphics::gui::text_renderer::push_back(
		property.text,
		property.font_size,
		fan::vec2(property.position.x + theme.text_button.outline_thickness + property.padding.x * 0.5, property.position.y + property.padding.y * 0.5),
		theme.text_button.text_color
	);

	inner_rect_t::properties_t rect_properties;
	rect_properties.position = property.position;
	rect_properties.size = get_button_size(m_properties.size() - 1);
	rect_properties.rotation_point = rect_properties.position;
	rect_properties.color = theme.text_button.color;

	inner_rect_t::push_back(rect_properties);

	auto corners = inner_rect_t::get_corners(m_properties.size() - 1);

	const f32_t t = theme.text_button.outline_thickness;

	corners[1].x -= t;
	corners[3].x -= t;

	corners[2].y -= t;
	corners[3].y -= t;

	outer_rect_t::properties_t properties;
	properties.position = corners[0];
	properties.rotation_point = properties.position;
	properties.size = fan::vec2(corners[1].x - corners[0].x + t, t);
	properties.color = theme.text_button.outline_color;

	outer_rect_t::push_back(properties);
	properties.position = corners[1];
	properties.size = fan::vec2(t, corners[3].y - corners[1].y + t);

	outer_rect_t::push_back(properties);

	properties.position = corners[2];
	properties.size = fan::vec2(corners[3].x - corners[2].x + t, t);

	outer_rect_t::push_back(properties);

	properties.position = corners[0];
	properties.size = fan::vec2(t, corners[2].y - corners[0].y + t);

	outer_rect_t::push_back(properties);
}

void fan_2d::graphics::gui::rectangle_text_box::draw(uint32_t begin, uint32_t end)
{
	// depth test
	inner_rect_t::draw(begin == (uint32_t)-1 ? 0 : begin, end == (uint32_t)-1 ? outer_rect_t::size() : end);		
	outer_rect_t::draw(begin == (uint32_t)-1 ? 0 : begin, end == (uint32_t)-1 ? outer_rect_t::size() : end);

	fan_2d::graphics::gui::text_renderer::draw();
}

void fan_2d::graphics::gui::rectangle_text_box::set_position(uint32_t i, const fan::vec2& position)
{
	const auto offset = position - rectangle_text_box::get_position(i);

	for (int j = 0; j < 4; j++) {
		rectangle_text_box::outer_rect_t::set_position(i + j, outer_rect_t::get_position(i + j) + offset);
	}

	rectangle_text_box::inner_rect_t::set_position(i, rectangle_text_box::inner_rect_t::get_position(i) + offset);
	rectangle_text_box::text_renderer::set_position(i, rectangle_text_box::text_renderer::get_position(i) + offset);
}

bool fan_2d::graphics::gui::rectangle_text_box::inside(uintptr_t i, const fan::vec2& position) const
{
	return inner_rect_t::inside(i, position);
}

fan::utf16_string fan_2d::graphics::gui::rectangle_text_box::get_text(uint32_t i) const {
	return fan_2d::graphics::gui::rectangle_text_box::text_renderer::get_text(i);
}

void fan_2d::graphics::gui::rectangle_text_box::set_text(uint32_t i, const fan::utf16_string& text) {
	
	auto position = text_renderer::get_position(i);
	auto offset = text_renderer::get_text_size(text_renderer::get_text(i), text_renderer::get_font_size(i)) - text_renderer::get_text_size(text_renderer::get_text(i), text_renderer::get_font_size(i));

	fan_2d::graphics::gui::text_renderer::set_position(i, position - offset);

	fan_2d::graphics::gui::text_renderer::set_text(i, text);

	inner_rect_t::set_size(i, get_button_size(i));

	auto corners = inner_rect_t::get_corners(i);

	const f32_t t = theme.text_button.outline_thickness;

	corners[1].x -= t;
	corners[3].x -= t;

	corners[2].y -= t;
	corners[3].y -= t;

	outer_rect_t::set_position(i * 4, corners[0]);
	outer_rect_t::set_size(i * 4, fan::vec2(corners[1].x - corners[0].x + t, t));

	outer_rect_t::set_position(i * 4 + 1, corners[1]);
	outer_rect_t::set_size(i * 4 + 1, fan::vec2(t, corners[3].y - corners[1].y + t));

	outer_rect_t::set_position(i * 4 + 2, corners[2]);
	outer_rect_t::set_size(i * 4 + 2, fan::vec2(corners[3].x - corners[2].x + t, t));

	outer_rect_t::set_position(i * 4 + 3, corners[0]);
	outer_rect_t::set_size(i * 4 + 3, fan::vec2(t, corners[2].y - corners[0].y + t));
}

fan::color fan_2d::graphics::gui::rectangle_text_box::get_text_color(uint32_t i) const
{
	return text_renderer::get_text_color(i);
}

void fan_2d::graphics::gui::rectangle_text_box::set_text_color(uint32_t i, const fan::color& color)
{
	text_renderer::set_text_color(i, color);
}

fan::vec2 fan_2d::graphics::gui::rectangle_text_box::get_position(uint32_t i) const {
	return inner_rect_t::get_position(i);
}

fan::vec2 fan_2d::graphics::gui::rectangle_text_box::get_size(uint32_t i) const
{
	return inner_rect_t::get_size(i);
}

fan::vec2 fan_2d::graphics::gui::rectangle_text_box::get_padding(uint32_t i) const {
	return m_properties[i].padding;
}

f32_t fan_2d::graphics::gui::rectangle_text_box::get_font_size(uint32_t i) const
{
	return rectangle_text_box::text_renderer::get_font_size(i);
}

fan_2d::graphics::gui::rectangle_text_box::properties_t fan_2d::graphics::gui::rectangle_text_box::get_property(uint32_t i) const
{
	return m_properties[i];
}

fan::color fan_2d::graphics::gui::rectangle_text_box::get_color(uint32_t i) const
{
	return inner_rect_t::get_color(i);
}

fan_2d::graphics::gui::cursor_src_dst_t fan_2d::graphics::gui::rectangle_text_box::get_cursor(uint32_t i, uint32_t x, uint32_t y)
{
	f32_t converted = fan_2d::graphics::gui::text_renderer::convert_font_size(this->get_font_size(i));

	fan::vec2 src, dst;

	src += this->get_position(i) + this->get_padding(i) / 2 + fan::vec2(outer_rect_t::get_size(i * 4 + 1).x, 0);

	uint32_t offset = 0;

	auto str = this->get_text(i);

	for (int j = 0; j < y; j++) {
		while (str[offset++] != '\n') {
			if (offset >= str.size()) {
				throw std::runtime_error("string didnt have endline");
			}
		}
	}

	for (int j = 0; j < x; j++) {
		wchar_t letter = str[j + offset];
		if (letter == '\n') {
			continue;
		}

		std::wstring wstr;

		wstr.push_back(letter);

		auto letter_info = fan_2d::graphics::gui::text_renderer::get_letter_info(fan::utf16_string(wstr).to_utf8().data(), this->get_font_size(i));

		if (j == x - 1) {
			src.x += letter_info.size.x + (letter_info.advance - letter_info.size.x) / 2 - 1;
		}
		else {
			src.x += letter_info.advance;
		}

	}

	src.y += fan_2d::graphics::gui::text_renderer::font_info.font['\n'].size.y * converted * y;

	dst = src + fan::vec2(0, fan_2d::graphics::gui::text_renderer::font_info.font['\n'].size.y * converted);

	dst = dst - src + fan::vec2(cursor_properties::line_thickness, 0);


	return { src, dst };
}

fan::vec2 fan_2d::graphics::gui::rectangle_text_box::get_text_starting_point(uint32_t i) const
{
	return this->get_position(i) + this->get_padding(i) / 2;
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
	outer_rect_t::edit_data(i * 4, i * 4 + 4); // 4 outlines
	text_renderer::edit_data(i);
}

void fan_2d::graphics::gui::rectangle_text_box::edit_data(uint32_t begin, uint32_t end) {
	inner_rect_t::edit_data(begin, end);
	outer_rect_t::edit_data(begin, end);
	graphics::gui::text_renderer::edit_data(begin, end);
}

fan_2d::graphics::gui::rectangle_text_button::rectangle_text_button(fan::camera* camera, fan_2d::graphics::gui::theme theme_)
	: 
	fan_2d::graphics::gui::rectangle_text_box(camera, theme_),
	rectangle_text_button::mouse(this), rectangle_text_button::text_input<rectangle_text_button>(this) {

	rectangle_text_button::mouse::add_on_input([&] (uint32_t i, fan::key_state state, fan_2d::graphics::gui::mouse_stage mouse_stage) {

		if (mouse_stage == mouse_stage::inside && state == fan::key_state::press) {

			inner_rect_t::set_color(i, theme.text_button.click_color);

			outer_rect_t::set_color(i * 4 + 0, theme.text_button.click_outline_color);
			outer_rect_t::set_color(i * 4 + 1, theme.text_button.click_outline_color);
			outer_rect_t::set_color(i * 4 + 2, theme.text_button.click_outline_color);
			outer_rect_t::set_color(i * 4 + 3, theme.text_button.click_outline_color);

			inner_rect_t::edit_data(i);

			outer_rect_t::edit_data(i * 4, i * 4 + 4);
		}
		else if (mouse_stage == mouse_stage::inside && state == fan::key_state::release) {

			inner_rect_t::set_color(i, theme.text_button.hover_color);

			outer_rect_t::set_color(i * 4 + 0, theme.text_button.hover_outline_color);
			outer_rect_t::set_color(i * 4 + 1, theme.text_button.hover_outline_color);
			outer_rect_t::set_color(i * 4 + 2, theme.text_button.hover_outline_color);
			outer_rect_t::set_color(i * 4 + 3, theme.text_button.hover_outline_color);

			inner_rect_t::edit_data(i);

			outer_rect_t::edit_data(i * 4, i * 4 + 4);

		}
		else if (mouse_stage == mouse_stage::outside && state == fan::key_state::release) {

			inner_rect_t::set_color(i, theme.text_button.color);

			outer_rect_t::set_color(i * 4 + 0, theme.text_button.outline_color);
			outer_rect_t::set_color(i * 4 + 1, theme.text_button.outline_color);
			outer_rect_t::set_color(i * 4 + 2, theme.text_button.outline_color);
			outer_rect_t::set_color(i * 4 + 3, theme.text_button.outline_color);

			inner_rect_t::edit_data(i);

			outer_rect_t::edit_data(i * 4, i * 4 + 4);
		}

	});

	rectangle_text_button::mouse::add_on_mouse_event([&] (uint32_t i, fan_2d::graphics::gui::mouse_stage mouse_stage) {

		if (mouse_stage == mouse_stage::inside) {
			inner_rect_t::set_color(i, theme.text_button.hover_color);

			outer_rect_t::set_color(i * 4 + 0, theme.text_button.hover_outline_color);
			outer_rect_t::set_color(i * 4 + 1, theme.text_button.hover_outline_color);
			outer_rect_t::set_color(i * 4 + 2, theme.text_button.hover_outline_color);
			outer_rect_t::set_color(i * 4 + 3, theme.text_button.hover_outline_color);

			inner_rect_t::edit_data(i);

			outer_rect_t::edit_data(i * 4, i * 4 + 4);
		}
		else if(mouse_stage == mouse_stage::outside_drag) {
			inner_rect_t::set_color(i, theme.text_button.hover_color);

			outer_rect_t::set_color(i * 4 + 0, theme.text_button.hover_outline_color);
			outer_rect_t::set_color(i * 4 + 1, theme.text_button.hover_outline_color);
			outer_rect_t::set_color(i * 4 + 2, theme.text_button.hover_outline_color);
			outer_rect_t::set_color(i * 4 + 3, theme.text_button.hover_outline_color);

			inner_rect_t::edit_data(i);

			outer_rect_t::edit_data(i * 4, i * 4 + 4);
		}
		else {
			inner_rect_t::set_color(i, theme.text_button.color);

			outer_rect_t::set_color(i * 4 + 0, theme.text_button.outline_color);
			outer_rect_t::set_color(i * 4 + 1, theme.text_button.outline_color);
			outer_rect_t::set_color(i * 4 + 2, theme.text_button.outline_color);
			outer_rect_t::set_color(i * 4 + 3, theme.text_button.outline_color);

			inner_rect_t::edit_data(i);

			outer_rect_t::edit_data(i * 4, i * 4 + 4);
		}

	});
}

void fan_2d::graphics::gui::rectangle_text_button::push_back(const properties_t& properties)
{
	rectangle_text_button::rectangle_text_box::push_back(properties);
	rectangle_text_button::text_input::push_back(properties.character_limit, properties.character_width, properties.line_limit);
	rectangle_text_button::mouse::push_back();
}
void fan_2d::graphics::gui::rectangle_text_button::set_place_holder(uint32_t i, const fan::utf16_string& place_holder)
{
	FED_MoveCursorFreeStyleToEndOfLine(&m_wed[i], cursor_reference[i]);

	for (int j = 0; j < m_text[i].size(); j++) {
		FED_DeleteCharacterFromCursor(&m_wed[i], cursor_reference[i]);
	}

	rectangle_text_button::text_renderer::set_text_color(i, fan_2d::graphics::gui::defaults::text_color_place_holder);
	this->set_text(i, place_holder);

	if (m_input_allowed[i]) {
		update_cursor(i);
	}
}
void fan_2d::graphics::gui::rectangle_text_button::draw()
{
	rectangle_text_button::rectangle_text_box::draw();
	input_instance_t::draw();
}

fan_2d::graphics::gui::rectangle_text_button_sized::rectangle_text_button_sized(fan::camera* camera, fan_2d::graphics::gui::theme theme_)
	: 
	fan_2d::graphics::gui::rectangle_text_box_sized(camera, theme_),
	rectangle_text_button_sized::mouse(this), 
	rectangle_text_button_sized::text_input(this) {

	m_box->add_on_input([&](uint32_t i, fan::key_state state, fan_2d::graphics::gui::mouse_stage mouse_stage) {

		if (mouse_stage == decltype(mouse_stage)::inside && state == fan::key_state::release) {

			focus::set_focus(focus::properties_t(m_box->get_camera()->m_window->get_handle(), this, i));
			if (m_input_allowed[i]) {
				render_cursor = true;
				update_cursor(i);
				cursor_timer.restart();
			}
			else {
				render_cursor = false;
			}
		}
		if (mouse_stage == decltype(mouse_stage)::outside && state == fan::key_state::press) {

			focus::set_focus(focus::no_focus);

			if (m_input_allowed[i]) {
				render_cursor = false;
			}
		}

		});

	rectangle_text_button_sized::mouse::add_on_input([&](uint32_t i, fan::key_state state, fan_2d::graphics::gui::mouse_stage mouse_stage) {

		if (mouse_stage == mouse_stage::inside && state == fan::key_state::press) {

			inner_rect_t::set_color(i, theme.text_button.click_color);

			outer_rect_t::set_color(i * 4 + 0, theme.text_button.click_outline_color);
			outer_rect_t::set_color(i * 4 + 1, theme.text_button.click_outline_color);
			outer_rect_t::set_color(i * 4 + 2, theme.text_button.click_outline_color);
			outer_rect_t::set_color(i * 4 + 3, theme.text_button.click_outline_color);

			inner_rect_t::edit_data(i);

			outer_rect_t::edit_data(i * 4, i * 4 + 4);
		}
		else if (mouse_stage == mouse_stage::inside && state == fan::key_state::release) {

			inner_rect_t::set_color(i, theme.text_button.hover_color);

			outer_rect_t::set_color(i * 4 + 0, theme.text_button.hover_outline_color);
			outer_rect_t::set_color(i * 4 + 1, theme.text_button.hover_outline_color);
			outer_rect_t::set_color(i * 4 + 2, theme.text_button.hover_outline_color);
			outer_rect_t::set_color(i * 4 + 3, theme.text_button.hover_outline_color);

			inner_rect_t::edit_data(i);

			outer_rect_t::edit_data(i * 4, i * 4 + 4);

		}
		else if (mouse_stage == mouse_stage::outside && state == fan::key_state::release) {

			inner_rect_t::set_color(i, theme.text_button.color);

			outer_rect_t::set_color(i * 4 + 0, theme.text_button.outline_color);
			outer_rect_t::set_color(i * 4 + 1, theme.text_button.outline_color);
			outer_rect_t::set_color(i * 4 + 2, theme.text_button.outline_color);
			outer_rect_t::set_color(i * 4 + 3, theme.text_button.outline_color);

			inner_rect_t::edit_data(i);

			outer_rect_t::edit_data(i * 4, i * 4 + 4);
		}

		});

	rectangle_text_button_sized::mouse::add_on_mouse_event([&](uint32_t i, fan_2d::graphics::gui::mouse_stage mouse_stage) {

		if (mouse_stage == mouse_stage::inside) {
			inner_rect_t::set_color(i, theme.text_button.hover_color);

			outer_rect_t::set_color(i * 4 + 0, theme.text_button.hover_outline_color);
			outer_rect_t::set_color(i * 4 + 1, theme.text_button.hover_outline_color);
			outer_rect_t::set_color(i * 4 + 2, theme.text_button.hover_outline_color);
			outer_rect_t::set_color(i * 4 + 3, theme.text_button.hover_outline_color);

			inner_rect_t::edit_data(i);

			outer_rect_t::edit_data(i * 4, i * 4 + 4);
		}
		else if (mouse_stage == mouse_stage::outside_drag) {
			inner_rect_t::set_color(i, theme.text_button.hover_color);

			outer_rect_t::set_color(i * 4 + 0, theme.text_button.hover_outline_color);
			outer_rect_t::set_color(i * 4 + 1, theme.text_button.hover_outline_color);
			outer_rect_t::set_color(i * 4 + 2, theme.text_button.hover_outline_color);
			outer_rect_t::set_color(i * 4 + 3, theme.text_button.hover_outline_color);

			inner_rect_t::edit_data(i);

			outer_rect_t::edit_data(i * 4, i * 4 + 4);
		}
		else {
			inner_rect_t::set_color(i, theme.text_button.color);

			outer_rect_t::set_color(i * 4 + 0, theme.text_button.outline_color);
			outer_rect_t::set_color(i * 4 + 1, theme.text_button.outline_color);
			outer_rect_t::set_color(i * 4 + 2, theme.text_button.outline_color);
			outer_rect_t::set_color(i * 4 + 3, theme.text_button.outline_color);

			inner_rect_t::edit_data(i);

			outer_rect_t::edit_data(i * 4, i * 4 + 4);
		}

	});
}

void fan_2d::graphics::gui::rectangle_text_button_sized::push_back(const properties_t& properties)
{
	rectangle_text_button_sized::rectangle_text_box_sized::push_back(properties);
	rectangle_text_button_sized::text_input::push_back(properties.character_limit, properties.character_width, properties.line_limit);
	rectangle_text_button_sized::mouse::push_back();
}

void fan_2d::graphics::gui::rectangle_text_button_sized::set_place_holder(uint32_t i, const fan::utf16_string& place_holder)
{
	FED_MoveCursorFreeStyleToEndOfLine(&m_wed[i], cursor_reference[i]);

	for (int j = 0; j < m_text[i].size(); j++) {
		FED_DeleteCharacterFromCursor(&m_wed[i], cursor_reference[i]);
	}

	rectangle_text_box_sized::text_renderer::set_text_color(i, fan_2d::graphics::gui::defaults::text_color_place_holder);
	this->set_text(i, place_holder);

	if (m_input_allowed[i]) {
		update_cursor(i);
	}

}

void fan_2d::graphics::gui::rectangle_text_button_sized::draw()
{
	rectangle_text_button_sized::rectangle_text_box_sized::draw();
	input_instance_t::draw();
}

void fan_2d::graphics::gui::rectangle_text_button_sized::backspace_callback(uint32_t i)
{
	auto current_string = m_box->get_text(i);
	auto current_property = m_box->get_property(i);

	if (current_string.size() && current_string[0] == '\0' && current_property.place_holder.size()) {
		m_box->set_text(i, current_property.place_holder);
		m_box->set_text_color(i, defaults::text_color_place_holder);
		m_box->write_data();
	}
}

void fan_2d::graphics::gui::rectangle_text_button_sized::text_callback(uint32_t i)
{
	if (m_box->get_text_color(i) != m_box->theme.text_button.text_color) {
		m_box->set_text_color(i, m_box->theme.text_button.text_color);
	}
}

fan_2d::graphics::gui::rectangle_text_button_sized::rectangle_text_button_sized(bool custom, fan::camera* camera, fan_2d::graphics::gui::theme theme_)
	: 
	fan_2d::graphics::gui::rectangle_text_box_sized(camera, theme_),
	rectangle_text_button_sized::mouse(this), 
	rectangle_text_button_sized::text_input(this) {}

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

	fan_2d::graphics::gui::text_renderer::push_back(
		properties.text,
		properties.font_size,
		fan::vec2(properties.position.x + properties.padding.x * 0.5, properties.position.y + properties.padding.y * 0.5),
		fan_2d::graphics::gui::defaults::text_color
	);


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
	sprite_text_button::mouse(this) {}

fan_2d::graphics::gui::scrollbar::scrollbar(fan::camera* camera) :
	fan_2d::graphics::rectangle(camera), scrollbar::mouse(this) {

	scrollbar::mouse::add_on_input([&](uint32_t i, fan::key_state state, mouse_stage mouse_stage) {
		
		if (state == fan::key_state::press && mouse_stage == mouse_stage::inside) {

			if (i % 2 == 0) {
				i += 1;
			}

			fan::vec2 offset;

			fan::vec2 current = rectangle::m_camera->m_window->get_mouse_position();

			switch (m_properties[i / 2].scroll_direction) {
				case scroll_direction_e::horizontal: {

					f32_t current_x = current.x - rectangle::get_size(i).x / 2;

					f32_t min_x = rectangle::get_position(i - 1).x + m_properties[i / 2].outline_thickness;
					f32_t max_x = rectangle::get_position(i - 1).x + rectangle::get_size(i - 1).x - rectangle::get_size(i).x - m_properties[i / 2].outline_thickness;

					offset.y = rectangle::get_position(i).y;

					offset.x = fan::clamp(
						current_x,
						min_x,
						max_x
					);

					if (fan_2d::graphics::rectangle::get_position(i) == offset) {
						return;
					}

					m_properties[i / 2].current = offset.x - min_x;

					fan_2d::graphics::rectangle::set_position(i, offset);

					break;
				}
				case scroll_direction_e::vertical: {

					f32_t current_y = current.y - rectangle::get_size(i).y / 2;

					f32_t min_y = rectangle::get_position(i - 1).y + m_properties[i / 2].outline_thickness;
					f32_t max_y = rectangle::get_position(i - 1).y + rectangle::get_size(i - 1).y - rectangle::get_size(i).y - m_properties[i / 2].outline_thickness;

					offset.x = rectangle::get_position(i).x;

					offset.y = fan::clamp(
						current_y,
						min_y,
						max_y
					);

					if (fan_2d::graphics::rectangle::get_position(i) == offset) {
						return;
					}

					m_properties[i / 2].current = offset.y - min_y;

					fan_2d::graphics::rectangle::set_position(i, offset);

					break;
				}
			}

			for (int j = 0; j < m_on_scroll.size(); j++) {
				if (m_on_scroll[j]) {
					m_on_scroll[j](i / 2, m_properties[j].current);
				}
			}

			rectangle::edit_data(i);
		}
		
	}, false);

	scrollbar::mouse::add_on_mouse_event([&](uint32_t i, mouse_stage mouse_stage) {
		
		if (scrollbar::holding_button(i) != fan::uninitialized && (mouse_stage == mouse_stage::inside || 
			mouse_stage == mouse_stage::outside_drag)) {

			if (i % 2 == 0) {
				i += 1;
			}

			fan::vec2 current = rectangle::m_camera->m_window->get_mouse_position();

			fan::vec2 offset;

			switch (m_properties[i / 2].scroll_direction) {
				case scroll_direction_e::horizontal: {

					offset.x += current.x - (rectangle::get_position(i - 1).x + m_properties[i / 2].outline_thickness);

					f32_t current_x = rectangle::get_position(i - 1).x + offset.x - rectangle::get_size(i).x / 2;
					//fan::print(current_x);
					f32_t min_x = rectangle::get_position(i - 1).x + m_properties[i / 2].outline_thickness;
					f32_t max_x = rectangle::get_position(i - 1).x + rectangle::get_size(i - 1).x - rectangle::get_size(i).x - m_properties[i / 2].outline_thickness;

					offset.y = rectangle::get_position(i).y;

					offset.x = fan::clamp(
						current_x,
						min_x,
						max_x
					);

					if (fan_2d::graphics::rectangle::get_position(i) == offset) {
						return;
					}

					m_properties[i / 2].current = offset.x - min_x;

					fan_2d::graphics::rectangle::set_position(i, offset);

					break;
				}
				case scroll_direction_e::vertical: {
					offset.y += current.y - (rectangle::get_position(i - 1).y + m_properties[i / 2].outline_thickness);

					f32_t current_y = rectangle::get_position(i - 1).y + offset.y - rectangle::get_size(i).y / 2;
					//fan::print(current_y);
					f32_t min_y = rectangle::get_position(i - 1).y + m_properties[i / 2].outline_thickness;
					f32_t max_y = rectangle::get_position(i - 1).y + rectangle::get_size(i - 1).y - rectangle::get_size(i).y - m_properties[i / 2].outline_thickness;

					offset.x = rectangle::get_position(i).x;

					offset.y = fan::clamp(
						current_y,
						min_y,
						max_y
					);

					if (fan_2d::graphics::rectangle::get_position(i) == offset) {
						return;
					}

					m_properties[i / 2].current = offset.y - min_y;

					fan_2d::graphics::rectangle::set_position(i, offset);

					break;
				}
			}

			for (int j = 0; j < m_on_scroll.size(); j++) {
				if (m_on_scroll[j]) {
					m_on_scroll[j](i / 2, m_properties[j].current);
				}
			}

			rectangle::edit_data(i);
		}

	}, false);

}

void fan_2d::graphics::gui::scrollbar::push_back(const properties_t& instance)
{
	fan_2d::graphics::rectangle::properties_t r_property;
	r_property.position = instance.position - (f32_t)instance.outline_thickness;
	r_property.size = 
		instance.size +
		(instance.scroll_direction == scroll_direction_e::horizontal ?
		fan::vec2(instance.length, 0) :
		fan::vec2(0, instance.length)) + instance.outline_thickness * 2;
	r_property.color = instance.color - 0.5;

	fan_2d::graphics::rectangle::push_back(r_property);

	r_property.position = instance.position;
	r_property.size = instance.size;
	r_property.color = instance.color;

	fan_2d::graphics::rectangle::push_back(r_property);

	scroll_properties_t sbp;
	sbp.length = instance.length;
	sbp.outline_thickness = instance.outline_thickness;
	sbp.scroll_direction = instance.scroll_direction;
	sbp.current = instance.current;

	m_properties.emplace_back(sbp);

	scrollbar::mouse::push_back();
	scrollbar::mouse::push_back();
}

void fan_2d::graphics::gui::scrollbar::draw()
{
	fan_2d::graphics::rectangle::draw();
}

void fan_2d::graphics::gui::scrollbar::write_data()
{
	fan_2d::graphics::rectangle::write_data();
}

fan::camera* fan_2d::graphics::gui::scrollbar::get_camera()
{
	return fan_2d::graphics::rectangle::m_camera;
}

void fan_2d::graphics::gui::scrollbar::add_on_scroll(on_scroll_t function)
{
	m_on_scroll.emplace_back(function);
}

fan_2d::graphics::gui::checkbox::checkbox(fan::camera* camera, fan_2d::graphics::gui::theme theme)
	: checkbox::rectangle_t(camera), 
	checkbox::line_t(camera), 
	checkbox::text_renderer_t(camera),
	checkbox::mouse(this),
	 m_theme(theme) {

	checkbox::mouse::add_on_mouse_event([&] (uint32_t i, fan_2d::graphics::gui::mouse_stage mouse_stage) {

		if (mouse_stage == mouse_stage::inside) {
			checkbox::rectangle_t::set_color(i, m_theme.checkbox.hover_color);
			this->edit_data(i);
		}
		else {
			checkbox::rectangle_t::set_color(i, m_theme.checkbox.color);
			this->edit_data(i);
		}
	});

	checkbox::mouse::add_on_input([&](uint32_t i, fan::key_state state, fan_2d::graphics::gui::mouse_stage mouse_stage) {

		if (mouse_stage == mouse_stage::inside && state == fan::key_state::press) {
			checkbox::rectangle_t::set_color(i, m_theme.checkbox.click_color);
			this->edit_data(i);
		}
		else if (mouse_stage == mouse_stage::inside && state == fan::key_state::press) {
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
		}
		else if (mouse_stage == mouse_stage::outside && state == fan::key_state::release) {
			checkbox::rectangle_t::set_color(i, m_theme.checkbox.color);
			this->edit_data(i);
		}
		
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

fan_2d::graphics::gui::rectangle_selectable_button_sized::rectangle_selectable_button_sized(fan::camera* camera, fan_2d::graphics::gui::theme theme_) :
	rectangle_text_button_sized(true, camera, theme_)
{
	rectangle_text_button_sized::mouse::add_on_input([&](uint32_t i, fan::key_state state, fan_2d::graphics::gui::mouse_stage mouse_stage) {

		if (mouse_stage == mouse_stage::inside && state == fan::key_state::press) {

			inner_rect_t::set_color(i, theme.text_button.click_color);

			outer_rect_t::set_color(i * 4 + 0, theme.text_button.click_outline_color);
			outer_rect_t::set_color(i * 4 + 1, theme.text_button.click_outline_color);
			outer_rect_t::set_color(i * 4 + 2, theme.text_button.click_outline_color);
			outer_rect_t::set_color(i * 4 + 3, theme.text_button.click_outline_color);

			inner_rect_t::edit_data(i);

			outer_rect_t::edit_data(i * 4, i * 4 + 4);
		}
		else if (mouse_stage == mouse_stage::inside && state == fan::key_state::release) {

			inner_rect_t::set_color(i, theme.text_button.hover_color);

			outer_rect_t::set_color(i * 4 + 0, theme.text_button.hover_outline_color);
			outer_rect_t::set_color(i * 4 + 1, theme.text_button.hover_outline_color);
			outer_rect_t::set_color(i * 4 + 2, theme.text_button.hover_outline_color);
			outer_rect_t::set_color(i * 4 + 3, theme.text_button.hover_outline_color);

			inner_rect_t::edit_data(i);

			outer_rect_t::edit_data(i * 4, i * 4 + 4);

			// default previous selected
			if (m_selected != i && m_selected != (uint32_t)fan::uninitialized) {
				inner_rect_t::set_color(m_selected, theme.text_button.color);

				outer_rect_t::set_color(m_selected * 4 + 0, theme.text_button.outline_color);
				outer_rect_t::set_color(m_selected * 4 + 1, theme.text_button.outline_color);
				outer_rect_t::set_color(m_selected * 4 + 2, theme.text_button.outline_color);
				outer_rect_t::set_color(m_selected * 4 + 3, theme.text_button.outline_color);

				inner_rect_t::edit_data(m_selected);

				outer_rect_t::edit_data(m_selected * 4, m_selected * 4 + 4);
			}

			if (m_selected != i) {
				for (const auto& f : m_on_select) {
					if (f) {
						f(i);
					}
				}
			}

			m_selected = i;

		}
		else if (mouse_stage == mouse_stage::outside && state == fan::key_state::release) {

			if (m_selected != i) {
				inner_rect_t::set_color(i, theme.text_button.color);

				outer_rect_t::set_color(i * 4 + 0, theme.text_button.outline_color);
				outer_rect_t::set_color(i * 4 + 1, theme.text_button.outline_color);
				outer_rect_t::set_color(i * 4 + 2, theme.text_button.outline_color);
				outer_rect_t::set_color(i * 4 + 3, theme.text_button.outline_color);

				inner_rect_t::edit_data(i);

				outer_rect_t::edit_data(i * 4, i * 4 + 4);
			}
		}

	});

	rectangle_text_button_sized::mouse::add_on_mouse_event([&](uint32_t i, fan_2d::graphics::gui::mouse_stage mouse_stage) {

		if (mouse_stage == mouse_stage::inside) {
			inner_rect_t::set_color(i, theme.text_button.hover_color);

			outer_rect_t::set_color(i * 4 + 0, theme.text_button.hover_outline_color);
			outer_rect_t::set_color(i * 4 + 1, theme.text_button.hover_outline_color);
			outer_rect_t::set_color(i * 4 + 2, theme.text_button.hover_outline_color);
			outer_rect_t::set_color(i * 4 + 3, theme.text_button.hover_outline_color);

			inner_rect_t::edit_data(i);

			outer_rect_t::edit_data(i * 4, i * 4 + 4);
		}
		else if (mouse_stage == mouse_stage::outside_drag) {
			if (m_selected != i) {

				inner_rect_t::set_color(i, theme.text_button.hover_color);

				outer_rect_t::set_color(i * 4 + 0, theme.text_button.hover_outline_color);
				outer_rect_t::set_color(i * 4 + 1, theme.text_button.hover_outline_color);
				outer_rect_t::set_color(i * 4 + 2, theme.text_button.hover_outline_color);
				outer_rect_t::set_color(i * 4 + 3, theme.text_button.hover_outline_color);

				inner_rect_t::edit_data(i);

				outer_rect_t::edit_data(i * 4, i * 4 + 4);
			}
		}
		else {

			if (m_selected != i) {
				inner_rect_t::set_color(i, theme.text_button.color);

				outer_rect_t::set_color(i * 4 + 0, theme.text_button.outline_color);
				outer_rect_t::set_color(i * 4 + 1, theme.text_button.outline_color);
				outer_rect_t::set_color(i * 4 + 2, theme.text_button.outline_color);
				outer_rect_t::set_color(i * 4 + 3, theme.text_button.outline_color);

				inner_rect_t::edit_data(i);

				outer_rect_t::edit_data(i * 4, i * 4 + 4);
			}
		}

	});
}

uint32_t fan_2d::graphics::gui::rectangle_selectable_button_sized::get_selected(uint32_t i) const
{
	return this->m_selected;
}

void fan_2d::graphics::gui::rectangle_selectable_button_sized::set_selected(uint32_t i)
{
	this->m_selected = i;

	inner_rect_t::set_color(i, theme.text_button.hover_color);

	outer_rect_t::set_color(i * 4 + 0, theme.text_button.hover_outline_color);
	outer_rect_t::set_color(i * 4 + 1, theme.text_button.hover_outline_color);
	outer_rect_t::set_color(i * 4 + 2, theme.text_button.hover_outline_color);
	outer_rect_t::set_color(i * 4 + 3, theme.text_button.hover_outline_color);

	inner_rect_t::edit_data(i);

	outer_rect_t::edit_data(i * 4, i * 4 + 4);
}

void fan_2d::graphics::gui::rectangle_selectable_button_sized::add_on_select(std::function<void(uint32_t)> function)
{
	m_on_select.push_back(function);
}