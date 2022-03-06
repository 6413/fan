#include <fan/graphics/graphics.hpp>

#include <fan/graphics/themes.hpp>

#include <fan/graphics/shared_gui.hpp>

void fan_2d::graphics::gui::rectangle_text_box_sized::rectangle_text_box_sized::open(fan::opengl::context_t* context)
{
	outer_rect_t::open(context);
	inner_rect_t::open(context);
	fan_2d::graphics::gui::text_renderer::open(context);
	m_store.open();
}

void fan_2d::graphics::gui::rectangle_text_box_sized::rectangle_text_box_sized::close(fan::opengl::context_t* context)
{
	outer_rect_t::close(context);
	inner_rect_t::close(context);
	fan_2d::graphics::gui::text_renderer::close(context);
	for (int i = 0; i < m_store.size(); i++) {
		m_store[i].m_properties.text.close();
		m_store[i].m_properties.place_holder.close();
		m_store[i].m_properties.theme.close();
	}
	m_store.close();
}

void fan_2d::graphics::gui::rectangle_text_box_sized::push_back(fan::opengl::context_t* context, const fan_2d::graphics::gui::rectangle_text_box_sized::properties_t& property)
{
	store_t store;

	store.m_properties.place_holder.open();
	store.m_properties.text.open();
	store.m_properties.theme.open();

	store.m_properties.font_size = property.font_size;
	store.m_properties.offset = property.offset;
	*store.m_properties.place_holder = property.font_size;
	store.m_properties.position = property.position;
	store.m_properties.size = property.size;
	*store.m_properties.text = property.text;
	store.m_properties.text_position = property.text_position;
	*store.m_properties.theme = property.theme;

	m_store.push_back(store);

	const auto str = property.place_holder.empty() ? property.text : property.place_holder;

	fan_2d::graphics::gui::text_renderer::properties_t text_properties;

	switch (property.text_position) {
	case text_position_e::left:
	{

		text_properties.text = str;
		text_properties.font_size = property.font_size;
		text_properties.position = fan::vec2(
			property.position.x + property.theme.button.outline_thickness - property.size.x * 0.5,
			property.position.y + property.theme.button.outline_thickness
		) + property.offset;
		text_properties.text_color = property.place_holder.empty() ? property.theme.button.text_color : defaults::text_color_place_holder;
		text_properties.outline_color = property.theme.button.text_outline_color;
		text_properties.outline_size = property.theme.button.text_outline_size;

		fan_2d::graphics::gui::text_renderer::push_back(context, text_properties);

		break;
	}
	case text_position_e::middle:
	{
		text_properties.text = str;
		text_properties.font_size = property.font_size;
		text_properties.position = property.position + property.offset;
		text_properties.text_color = property.text.size() && property.text[0] != '\0' ? property.theme.button.text_color : defaults::text_color_place_holder;

		text_properties.outline_color = property.theme.button.text_outline_color;
		text_properties.outline_size = property.theme.button.text_outline_size;

		fan_2d::graphics::gui::text_renderer::push_back(context, text_properties);

		break;
	}
	}

	inner_rect_t::properties_t rect_properties;
	rect_properties.position = property.position;
	rect_properties.size = property.size;
	//rect_properties.rotation_point = rect_properties.position;
	rect_properties.color = property.theme.button.color;

	inner_rect_t::push_back(context, rect_properties);

	auto corners = inner_rect_t::get_corners(context, m_store.size() - 1);

	const f32_t t = property.theme.button.outline_thickness;

	corners[1].x -= t;
	corners[3].x -= t;

	corners[2].y -= t;
	corners[3].y -= t;

	outer_rect_t::properties_t properties;
	properties.position = corners[0] + fan::vec2(rect_properties.size.x, 0);
	properties.rotation_point = properties.position;
	properties.size = fan::vec2(corners[1].x - corners[0].x + t, t) / 2;
	properties.color = property.theme.button.outline_color;

	outer_rect_t::push_back(context, properties);
	properties.position = corners[1] + fan::vec2(0, rect_properties.size.y) + fan::vec2(1, -1);
	properties.size = fan::vec2(t, corners[3].y - corners[1].y + t) / 2 - fan::vec2(0, 0.5);
	if (property.size == 0) {
		properties.size = 0;
	}

	outer_rect_t::push_back(context, properties);

	properties.position = corners[2] + fan::vec2(rect_properties.size.x, 0) + fan::vec2(0.5, 1);
	properties.size = fan::vec2(corners[3].x - corners[2].x + t, t) / 2 - fan::vec2(0.5, 0);
	if (property.size == 0) {
		properties.size = 0;
	}

	outer_rect_t::push_back(context, properties);

	properties.position = corners[0] + fan::vec2(0, rect_properties.size.y - 0.5);
	properties.size = fan::vec2(t, corners[2].y - corners[0].y + t) / 2 + fan::vec2(0, 0.5);
	if (property.size == 0) {
		properties.size = 0;
	}

	outer_rect_t::push_back(context, properties);
}

void fan_2d::graphics::gui::rectangle_text_box_sized::draw(fan::opengl::context_t* context)
{
	// depth test
	inner_rect_t::draw(context);
	outer_rect_t::draw(context);

	fan_2d::graphics::gui::text_renderer::draw(context);
}

void fan_2d::graphics::gui::rectangle_text_box_sized::set_position(fan::opengl::context_t* context, uint32_t i, const fan::vec2& position)
{
	const auto offset = position - rectangle_text_box_sized::get_position(context, i);

	for (int j = 0; j < 4; j++) {
		rectangle_text_box_sized::outer_rect_t::set_position(context, i * 4 + j, rectangle_text_box_sized::outer_rect_t::get_position(context, i * 4 + j) + offset);
	}

	rectangle_text_box_sized::inner_rect_t::set_position(context, i, inner_rect_t::get_position(context, i) + offset);
	rectangle_text_box_sized::text_renderer::set_position(context, i, rectangle_text_box_sized::text_renderer::get_position(context, i) + offset);
}

bool fan_2d::graphics::gui::rectangle_text_box_sized::inside(fan::opengl::context_t* context, uintptr_t i, const fan::vec2& position) const
{
	return inner_rect_t::inside(context, i, position);
}

void fan_2d::graphics::gui::rectangle_text_box_sized::set_text(fan::opengl::context_t* context, uint32_t i, const fan::utf16_string& text) {

	switch (m_store[i].m_properties.text_position) {
	case text_position_e::left:
	{
		fan_2d::graphics::gui::text_renderer::set_text(context, i, text);
		break;
	}
	case text_position_e::middle:
	{
		fan_2d::graphics::gui::text_renderer::set_text(context, i, text);

		break;
	}
	}

	auto corners = inner_rect_t::get_corners(context, i);

	const f32_t t = m_store[i].m_properties.theme->button.outline_thickness;

	corners[1].x -= t;
	corners[3].x -= t;

	corners[2].y -= t;
	corners[3].y -= t;

	fan::vec2 rect_size = inner_rect_t::get_size(context, i);

	outer_rect_t::set_position(context, i * 4, corners[0] + fan::vec2(rect_size.x, 0));
	outer_rect_t::set_size(context, i * 4, fan::vec2(corners[1].x - corners[0].x + t, t) / 2);

	outer_rect_t::set_position(context, i * 4 + 1, corners[1] + fan::vec2(0, rect_size.y) + fan::vec2(1, -1));
	outer_rect_t::set_size(context, i * 4 + 1, fan::vec2(t, corners[3].y - corners[1].y + t) / 2 - fan::vec2(0, 0.5));

	outer_rect_t::set_position(context, i * 4 + 2, corners[2] + fan::vec2(rect_size.x, 0) + fan::vec2(0.5, 1));
	outer_rect_t::set_size(context, i * 4 + 2, fan::vec2(corners[3].x - corners[2].x + t, t) / 2 - fan::vec2(0.5, 0));

	outer_rect_t::set_position(context, i * 4 + 3, corners[0] + fan::vec2(0, rect_size.y - 0.5));
	outer_rect_t::set_size(context, i * 4 + 3, fan::vec2(t, corners[2].y - corners[0].y + t) / 2 + fan::vec2(0, 0.5));
}

void fan_2d::graphics::gui::rectangle_text_box_sized::set_size(fan::opengl::context_t* context, uint32_t i, const fan::vec2& size)
{

	const auto offset = size - rectangle_text_box_sized::get_size(context, i);
	rectangle_text_box_sized::inner_rect_t::set_size(context, i, inner_rect_t::get_size(context, i) + offset);

	auto corners = inner_rect_t::get_corners(context, i);

	const f32_t t = m_store[i].m_properties.theme->button.outline_thickness;

	corners[1].x -= t;
	corners[3].x -= t;

	corners[2].y -= t;
	corners[3].y -= t;

	fan::vec2 rect_size = inner_rect_t::get_size(context, i);

	outer_rect_t::set_position(context, i * 4, corners[0] + fan::vec2(rect_size.x, 0));
	outer_rect_t::set_size(context, i * 4, fan::vec2(corners[1].x - corners[0].x + t, t) / 2);

	outer_rect_t::set_position(context, i * 4 + 1, corners[1] + fan::vec2(0, rect_size.y) + fan::vec2(1, -1));
	outer_rect_t::set_size(context, i * 4 + 1, size == 0 ? 0 : (fan::vec2(t, corners[3].y - corners[1].y + t) / 2 - fan::vec2(0, 0.5)));

	outer_rect_t::set_position(context, i * 4 + 2, corners[2] + fan::vec2(rect_size.x, 0) + fan::vec2(0.5, 1));
	outer_rect_t::set_size(context, i * 4 + 2, size == 0 ? 0 : (fan::vec2(corners[3].x - corners[2].x + t, t) / 2 - fan::vec2(0.5, 0)));

	outer_rect_t::set_position(context, i * 4 + 3, corners[0] + fan::vec2(0, rect_size.y - 0.5));
	outer_rect_t::set_size(context, i * 4 + 3, size == 0 ? 0 : (fan::vec2(t, corners[2].y - corners[0].y + t) / 2 + fan::vec2(0, 0.5)));
}

fan::color fan_2d::graphics::gui::rectangle_text_box_sized::get_text_color(fan::opengl::context_t* context, uint32_t i) const
{
	return text_renderer::get_text_color(context, i);
}

void fan_2d::graphics::gui::rectangle_text_box_sized::set_text_color(fan::opengl::context_t* context, uint32_t i, const fan::color& color)
{
	text_renderer::set_text_color(context, i, color);
}

fan::vec2 fan_2d::graphics::gui::rectangle_text_box_sized::get_position(fan::opengl::context_t* context, uint32_t i) const
{
	return inner_rect_t::get_position(context, i);
}

fan::vec2 fan_2d::graphics::gui::rectangle_text_box_sized::get_size(fan::opengl::context_t* context, uint32_t i) const
{
	return inner_rect_t::get_size(context, i);
}

f32_t fan_2d::graphics::gui::rectangle_text_box_sized::get_font_size(fan::opengl::context_t* context, uint32_t i) const
{
	return text_renderer::get_font_size(context, i);
}

fan::color fan_2d::graphics::gui::rectangle_text_box_sized::get_color(fan::opengl::context_t* context, uint32_t i) const
{
	return inner_rect_t::get_color(context, i);
}

fan_2d::graphics::gui::src_dst_t fan_2d::graphics::gui::rectangle_text_box_sized::get_cursor(fan::opengl::context_t* context, uint32_t i, uint32_t x, uint32_t y)
{
	f32_t font_size = get_font_size(context, i);
	f32_t line_height = fan_2d::graphics::gui::text_renderer::get_line_height(context, font_size);

	fan::vec2 src, dst;

	if (m_store[i].m_properties.text_position == fan_2d::graphics::gui::text_position_e::left) {

		src += (this->get_position(context, i) + fan::vec2(0, this->get_size(context, i).y / 2) + m_store[i].m_properties.theme->button.outline_thickness);

		src.y -= line_height / 2;

		src += m_store[i].m_properties.offset;
	}
	else if (m_store[i].m_properties.text_position == fan_2d::graphics::gui::text_position_e::middle) {
		auto text_size = get_text_size(context, i);
		src += this->get_position(context, i) + fan::vec2(-text_size.x / 2, 0);
	}

	uint32_t offset = 0;

	auto str = this->get_text(context, i);

	for (int j = 0; j < y; j++) {
		while (str[offset++] != '\n') {
			if (offset >= str.size()) {
				throw std::runtime_error("string didnt have endline");
			}
		}
	}

	for (int j = 0; j < x; j++) {
		wchar_t wc = str[j + offset];
		if (wc == '\n') {
			continue;
		}

		auto letter_info = fan_2d::graphics::gui::text_renderer::get_letter_info(context, fan::utf16_string(wc).to_utf8().data(), font_size);
		src.x += letter_info.metrics.advance;
	}

	src.y += line_height * y;

	dst = fan::vec2(cursor_properties::line_thickness / 2, line_height / 2);

	return { src, dst };
}

fan::vec2 fan_2d::graphics::gui::rectangle_text_box_sized::get_text_starting_point(fan::opengl::context_t* context, uint32_t i) const
{
	fan::vec2 src;

	if (m_store[i].m_properties.text_position == fan_2d::graphics::gui::text_position_e::left) {
		src = this->get_position(context, i);
		src.y += font.characters['\n'].metrics.size.y * convert_font_size(context, get_font_size(context, i));
		src += m_store[i].m_properties.offset;
	}
	else if (m_store[i].m_properties.text_position == fan_2d::graphics::gui::text_position_e::middle) {
		auto text_size = text_renderer::get_text_size(context, get_text(context, i), text_renderer::get_font_size(context, i));
		text_size.y = fan_2d::graphics::gui::text_renderer::get_line_height(context, get_font_size(context, i));
		src = this->get_position(context, i) - text_size * 0.5;
	}

	return src;
}

fan_2d::graphics::gui::rectangle_text_box_sized::properties_t fan_2d::graphics::gui::rectangle_text_box_sized::get_property(fan::opengl::context_t* context, uint32_t i) const
{
	return *(fan_2d::graphics::gui::rectangle_text_box_sized::properties_t*)&m_store[i].m_properties;
}

void fan_2d::graphics::gui::rectangle_text_box_sized::set_offset(fan::opengl::context_t* context, uint32_t i, const fan::vec2& offset)
{
	m_store[i].m_properties.offset = offset;

	auto new_position = this->get_position(context, i) + offset;

	fan_2d::graphics::gui::text_renderer::set_position(context, i, new_position);
}

uintptr_t fan_2d::graphics::gui::rectangle_text_box_sized::size(fan::opengl::context_t* context) const
{
	return inner_rect_t::size(context);
}

void fan_2d::graphics::gui::rectangle_text_box_sized::erase(fan::opengl::context_t* context, uint32_t i)
{
	inner_rect_t::erase(context, i);
	outer_rect_t::erase(context, i * 4, i * 4 + 4);
	fan_2d::graphics::gui::text_renderer::erase(context, i);

	m_store.erase(i);
}

void fan_2d::graphics::gui::rectangle_text_box_sized::erase(fan::opengl::context_t* context, uint32_t begin, uint32_t end)
{
	inner_rect_t::erase(context, begin, end);
	outer_rect_t::erase(context, begin * 4, end * 4);
	fan_2d::graphics::gui::text_renderer::erase(context, begin, end);
	m_store.erase(begin, end);
}

void fan_2d::graphics::gui::rectangle_text_box_sized::clear(fan::opengl::context_t* context)
{
	fan::class_duplicator<fan_2d::graphics::rectangle, 0>::clear(context);
	fan::class_duplicator<fan_2d::graphics::rectangle, 1>::clear(context);
	fan_2d::graphics::gui::text_renderer::clear(context);
	m_store.clear();
}

//void fan_2d::graphics::gui::rectangle_text_box_sized::set_theme(fan_2d::graphics::gui::theme theme_)
//{
//	m_store[i].m_properties.theme = theme_;
//
//	for (int i = 0; i < this->size(); i++) {
//		this->set_theme(i, theme);
//	}
//}

void fan_2d::graphics::gui::rectangle_text_box_sized::update_theme(fan::opengl::context_t* context, uint32_t i)
{
	inner_rect_t::set_color(context, i, m_store[i].m_properties.theme->button.color);
	outer_rect_t::set_color(context, i * 4 + 0, m_store[i].m_properties.theme->button.outline_color);
	outer_rect_t::set_color(context, i * 4 + 1, m_store[i].m_properties.theme->button.outline_color);
	outer_rect_t::set_color(context, i * 4 + 2, m_store[i].m_properties.theme->button.outline_color);
	outer_rect_t::set_color(context, i * 4 + 3, m_store[i].m_properties.theme->button.outline_color);
}

void fan_2d::graphics::gui::rectangle_text_box_sized::set_theme(fan::opengl::context_t* context, uint32_t i, const fan_2d::graphics::gui::theme& theme_)
{
	inner_rect_t::set_color(context, i, theme_.button.color);
	outer_rect_t::set_color(context, i * 4 + 0, theme_.button.outline_color);
	outer_rect_t::set_color(context, i * 4 + 1, theme_.button.outline_color);
	outer_rect_t::set_color(context, i * 4 + 2, theme_.button.outline_color);
	outer_rect_t::set_color(context, i * 4 + 3, theme_.button.outline_color);
}

void fan_2d::graphics::gui::rectangle_text_box_sized::enable_draw(fan::opengl::context_t* context)
{
	inner_rect_t::enable_draw(context);
	outer_rect_t::enable_draw(context);
	fan_2d::graphics::gui::text_renderer::enable_draw(context);
}

void fan_2d::graphics::gui::rectangle_text_box_sized::disable_draw(fan::opengl::context_t* context)
{
	fan::class_duplicator<fan_2d::graphics::rectangle, 0>::disable_draw(context);
	fan::class_duplicator<fan_2d::graphics::rectangle, 1>::disable_draw(context);
	fan_2d::graphics::gui::text_renderer::disable_draw(context);
}

//void fan_2d::graphics::gui::rectangle_text_box_sized::set_draw_order(uint32_t i)
//{
//	fan::class_duplicator<fan_2d::graphics::rectangle, 0>::set_draw_order(i);
//	fan::class_duplicator<fan_2d::graphics::rectangle, 1>::set_draw_order(i);
//	fan_2d::graphics::gui::text_renderer::set_draw_order(i);
//}

//fan_2d::graphics::gui::rectangle_text_box::rectangle_text_box(fan::camera* camera, fan_2d::graphics::gui::theme theme)
//	:
//	inner_rect_t(camera),
//	outer_rect_t(camera),
//	fan_2d::graphics::gui::text_renderer(camera), theme(theme) {}
//
//void fan_2d::graphics::gui::rectangle_text_box::push_back(const properties_t& property)
//{
//	m_properties.emplace_back(property);
//
//	inner_rect_t::properties_t rect_properties;
//	rect_properties.position = property.position;
//	rect_properties.size = get_button_size(
//		property.text, 
//		property.font_size, 
//		text_renderer::get_new_lines(property.text), 
//		property.padding
//	);
//	rect_properties.rotation_point = rect_properties.position;
//	rect_properties.color = theme.button.color;
//
//	fan_2d::graphics::gui::text_renderer::properties_t text_properties;
//	text_properties.text = property.text;
//	text_properties.font_size = property.font_size;
//	text_properties.position = property.position + property.padding * 0.5 + fan::vec2(theme.button.outline_thickness, 0);
//	text_properties.text_color = theme.button.text_color;
//	text_properties.outline_color = theme.button.outline_color;
//	text_properties.outline_size = theme.button.text_outline_size;
//	fan_2d::graphics::gui::text_renderer::push_back(text_properties);
//
//	inner_rect_t::push_back(rect_properties);
//
//	auto corners = inner_rect_t::get_corners(m_properties.size() - 1);
//
//	const f32_t t = theme.button.outline_thickness;
//
//	corners[1].x -= t;
//	corners[3].x -= t;
//
//	corners[2].y -= t;
//	corners[3].y -= t;
//
//	outer_rect_t::properties_t properties;
//	properties.position = corners[0] + fan::vec2(rect_properties.size.x, 0);
//	properties.rotation_point = properties.position;
//	properties.size = fan::vec2(corners[1].x - corners[0].x + t, t) / 2;
//	properties.color = theme.button.outline_color;
//
//	outer_rect_t::push_back(properties);
//	properties.position = corners[1] + fan::vec2(0, rect_properties.size.y);
//	properties.size = fan::vec2(t, corners[3].y - corners[1].y + t) / 2;
//
//	outer_rect_t::push_back(properties);
//
//	properties.position = corners[2] + fan::vec2(rect_properties.size.x, 0);
//	properties.size = fan::vec2(corners[3].x - corners[2].x + t, t) / 2;
//
//	outer_rect_t::push_back(properties);
//
//	properties.position = corners[0] + fan::vec2(0, rect_properties.size.y);
//	properties.size = fan::vec2(t, corners[2].y - corners[0].y + t) / 2;
//
//	outer_rect_t::push_back(properties);
//}
//
//void fan_2d::graphics::gui::rectangle_text_box::draw(uint32_t begin, uint32_t end)
//{
//	// depth test
//	inner_rect_t::draw(begin == (uint32_t)-1 ? 0 : begin, end == (uint32_t)-1 ? outer_rect_t::size() : end);		
//	outer_rect_t::draw(begin == (uint32_t)-1 ? 0 : begin, end == (uint32_t)-1 ? outer_rect_t::size() : end);
//
//	fan_2d::graphics::gui::text_renderer::draw();
//}
//
//void fan_2d::graphics::gui::rectangle_text_box::set_position(uint32_t i, const fan::vec2& position)
//{
//	const auto offset = position - rectangle_text_box::get_position(i);
//
//	for (int j = 0; j < 4; j++) {
//		rectangle_text_box::outer_rect_t::set_position(i + j, outer_rect_t::get_position(i + j) + offset);
//	}
//
//	rectangle_text_box::inner_rect_t::set_position(i, rectangle_text_box::inner_rect_t::get_position(i) + offset);
//	rectangle_text_box::text_renderer::set_position(i, rectangle_text_box::text_renderer::get_position(i) + offset);
//}
//
//bool fan_2d::graphics::gui::rectangle_text_box::inside(uintptr_t i, const fan::vec2& position) const
//{
//	return inner_rect_t::inside(i, position);
//}
//
//fan::utf16_string fan_2d::graphics::gui::rectangle_text_box::get_text(uint32_t i) const {
//	return fan_2d::graphics::gui::rectangle_text_box::text_renderer::get_text(i);
//}
//
//void fan_2d::graphics::gui::rectangle_text_box::set_text(uint32_t i, const fan::utf16_string& text) {
//	
//
//	auto rect_size = get_button_size(
//		text,
//		text_renderer::get_font_size(i),
//		0,
//		m_store[i].m_properties.padding
//	);
//
//	inner_rect_t::set_position(i, inner_rect_t::get_position(i) + (rect_size - inner_rect_t::get_size()) * 0.5);
//
//	inner_rect_t::set_size(i, rect_size);
//
//	fan_2d::graphics::gui::text_renderer::set_position(i, inner_rect_t::get_position(i) + m_store[i].m_properties.padding * 0.5);
//
//	fan_2d::graphics::gui::text_renderer::set_text(i, text);
//
//	auto corners = inner_rect_t::get_corners(i);
//
//	const f32_t t = theme.button.outline_thickness;
//
//	corners[1].x -= t;
//	corners[3].x -= t;
//
//	corners[2].y -= t;
//	corners[3].y -= t;
//
//	outer_rect_t::set_position(i * 4, corners[0] + fan::vec2(rect_size.x, 0));
//	outer_rect_t::set_size(i * 4, fan::vec2(corners[1].x - corners[0].x + t, t) / 2);
//
//	outer_rect_t::set_position(i * 4 + 1, corners[1] + fan::vec2(0, rect_size.y));
//	outer_rect_t::set_size(i * 4 + 1, fan::vec2(t, corners[3].y - corners[1].y + t) / 2);
//
//	outer_rect_t::set_position(i * 4 + 2, corners[2] + fan::vec2(rect_size.x, 0));
//	outer_rect_t::set_size(i * 4 + 2, fan::vec2(corners[3].x - corners[2].x + t, t) / 2);
//
//	outer_rect_t::set_position(i * 4 + 3, corners[0] + fan::vec2(0, rect_size.y));
//	outer_rect_t::set_size(i * 4 + 3, fan::vec2(t, corners[2].y - corners[0].y + t) / 2);
//}
//
//fan::color fan_2d::graphics::gui::rectangle_text_box::get_text_color(uint32_t i) const
//{
//	return text_renderer::get_text_color(i);
//}
//
//void fan_2d::graphics::gui::rectangle_text_box::set_text_color(uint32_t i, const fan::color& color)
//{
//	text_renderer::set_text_color(i, color);
//}
//
//fan::vec2 fan_2d::graphics::gui::rectangle_text_box::get_position(uint32_t i) const {
//	return inner_rect_t::get_position(i);
//}
//
//fan::vec2 fan_2d::graphics::gui::rectangle_text_box::get_size(uint32_t i) const
//{
//	return inner_rect_t::get_size(i);
//}
//
//fan::vec2 fan_2d::graphics::gui::rectangle_text_box::get_padding(uint32_t i) const {
//	return m_store[i].m_properties.padding;
//}
//
//f32_t fan_2d::graphics::gui::rectangle_text_box::get_font_size(uint32_t i) const
//{
//	return rectangle_text_box::text_renderer::get_font_size(i);
//}
//
//fan_2d::graphics::gui::rectangle_text_box::properties_t fan_2d::graphics::gui::rectangle_text_box::get_property(uint32_t i) const
//{
//	return m_store[i].m_properties;
//}
//
//fan::color fan_2d::graphics::gui::rectangle_text_box::get_color(uint32_t i) const
//{
//	return inner_rect_t::get_color(i);
//}
//
//fan_2d::graphics::gui::src_dst_t fan_2d::graphics::gui::rectangle_text_box::get_cursor(uint32_t i, uint32_t x, uint32_t y)
//{
//	f32_t converted = fan_2d::graphics::gui::text_renderer::convert_font_size(this->get_font_size(i));
//
//	fan::vec2 src, dst;
//
//	src += this->get_position(i) + this->get_padding(i) / 2 + fan::vec2(outer_rect_t::get_size(i * 4 + 1).x, 0);
//
//	uint32_t offset = 0;
//
//	auto str = this->get_text(i);
//
//	for (int j = 0; j < y; j++) {
//		while (str[offset++] != '\n') {
//			if (offset >= str.size()) {
//				throw std::runtime_error("string didnt have endline");
//			}
//		}
//	}
//
//	for (int j = 0; j < x; j++) {
//		wchar_t letter = str[j + offset];
//		if (letter == '\n') {
//			continue;
//		}
//
//		std::wstring wstr;
//
//		wstr.push_back(letter);
//
//		auto letter_info = fan_2d::graphics::gui::text_renderer::get_letter_info(fan::utf16_string(wstr).to_utf8().data(), this->get_font_size(i));
//
//		if (j == x - 1) {
//			src.x += letter_info.metrics.size.x + (letter_info.metrics.advance - letter_info.metrics.size.x) / 2 - 1;
//		}
//		else {
//			src.x += letter_info.metrics.advance;
//		}
//
//	}
//
//	src.y += fan_2d::graphics::gui::text_renderer::font.font['\n'].metrics.size.y * converted * y;
//
//	dst = src + fan::vec2(0, fan_2d::graphics::gui::text_renderer::font.font['\n'].metrics.size.y * converted);
//
//	dst = dst - src + fan::vec2(cursor_properties::line_thickness, 0);
//
//
//	return { src, dst };
//}
//
//fan::vec2 fan_2d::graphics::gui::rectangle_text_box::get_text_starting_point(uint32_t i) const
//{
//	return this->get_position(i) + this->get_padding(i) / 2;
//}
//
//fan::camera* fan_2d::graphics::gui::rectangle_text_box::get_camera()
//{
//	return inner_rect_t::m_camera;
//}
//
//uintptr_t fan_2d::graphics::gui::rectangle_text_box::size() const
//{
//	return inner_rect_t::size();
//}
//
//void fan_2d::graphics::gui::rectangle_text_box::write_data() {
//	inner_rect_t::write_data();
//	outer_rect_t::write_data();
//	graphics::gui::text_renderer::write_data();
//}
//
//void fan_2d::graphics::gui::rectangle_text_box::edit_data(uint32_t i) {
//	inner_rect_t::edit_data(i);
//	outer_rect_t::edit_data(i * 4, i * 4 + 4); // 4 outlines
//	text_renderer::edit_data(i);
//}
//
//void fan_2d::graphics::gui::rectangle_text_box::edit_data(uint32_t begin, uint32_t end) {
//	inner_rect_t::edit_data(begin, end);
//	outer_rect_t::edit_data(begin, end);
//	graphics::gui::text_renderer::edit_data(begin, end);
//}
//
//void fan_2d::graphics::gui::rectangle_text_box::erase(uint32_t i)
//{
//	fan::class_duplicator<fan_2d::graphics::rectangle, 0>::erase(i);
//	fan::class_duplicator<fan_2d::graphics::rectangle, 1>::erase(i);
//	fan_2d::graphics::gui::text_renderer::erase(i);
//
//	m_properties.erase(m_properties.begin() + i);
//}
//
//void fan_2d::graphics::gui::rectangle_text_box::erase(uint32_t begin, uint32_t end)
//{
//	fan::class_duplicator<fan_2d::graphics::rectangle, 0>::erase(begin, end);
//	fan::class_duplicator<fan_2d::graphics::rectangle, 1>::erase(begin, end);
//	fan_2d::graphics::gui::text_renderer::erase(begin, end);
//
//	m_properties.erase(m_properties.begin() + begin, m_properties.begin() + end);
//}
//
//void fan_2d::graphics::gui::rectangle_text_box::clear()
//{
//	fan::class_duplicator<fan_2d::graphics::rectangle, 0>::clear();
//	fan::class_duplicator<fan_2d::graphics::rectangle, 1>::clear();
//	fan_2d::graphics::gui::text_renderer::clear();
//
//	m_properties.clear();
//}
//
//void fan_2d::graphics::gui::rectangle_text_box::set_theme(fan_2d::graphics::gui::theme theme_)
//{
//	theme = theme_;
//
//	for (int i = 0; i < this->size(); i++) {
//		this->set_theme(i, theme);
//	}
//}
//
//void fan_2d::graphics::gui::rectangle_text_box::set_theme(uint32_t i, fan_2d::graphics::gui::theme theme_)
//{
//	theme = theme_;
//
//	inner_rect_t::set_color(i, theme.button.color);
//
//	outer_rect_t::set_color(i * 4 + 0, theme.button.outline_color);
//	outer_rect_t::set_color(i * 4 + 1, theme.button.outline_color);
//	outer_rect_t::set_color(i * 4 + 2, theme.button.outline_color);
//	outer_rect_t::set_color(i * 4 + 3, theme.button.outline_color);
//}
//
//void fan_2d::graphics::gui::rectangle_text_box::enable_draw()
//{
//	inner_rect_t::enable_draw();
//	outer_rect_t::enable_draw();
//	fan_2d::graphics::gui::text_renderer::enable_draw();
//}
//
//void fan_2d::graphics::gui::rectangle_text_box::disable_draw()
//{
//	inner_rect_t::disable_draw();
//	outer_rect_t::disable_draw();
//	fan_2d::graphics::gui::text_renderer::disable_draw();
//}
//
//fan_2d::graphics::gui::circle_button::circle_button(fan::camera* camera) :
//	fan_2d::graphics::circle(camera),
//	fan_2d::graphics::gui::base::mouse<fan_2d::graphics::gui::circle_button>(this)
//{
//
//}
//fan_2d::graphics::gui::circle_button::~circle_button() {
//	if (pointer_remove_flag == 1) {
//		pointer_remove_flag = 0;
//	}
//	else {
//
//	}
//}
//
//void fan_2d::graphics::gui::circle_button::push_back(properties_t properties) {
//	m_reserved.emplace_back((uint32_t)properties.button_state);
//	if (properties.button_state == button_states_e::locked) {
//
//		properties.theme = fan_2d::graphics::gui::themes::locked();
//	}
//
//	fan_2d::graphics::circle::properties_t p;
//	p.position = properties.position;
//	p.radius = properties.radius;
//	p.color = properties.theme.button.color;
//
//	fan_2d::graphics::circle::push_back(p);
//
//	m_theme.emplace_back(properties.theme);
//
//	if (inside(size() - 1) && properties.button_state != button_states_e::locked) {
//		circle_button::mouse::m_focused_button_id = size() - 1;
//		lib_add_on_mouse_move(m_camera->m_window, circle_button::mouse::m_focused_button_id, fan_2d::graphics::gui::mouse_stage::inside);
//	}
//}
//
//void fan_2d::graphics::gui::circle_button::erase(uint32_t i) {
//	circle::erase(i);
//	m_theme.erase(m_theme.begin() + i);
//	m_reserved.erase(m_reserved.begin() + i);
//}
//void fan_2d::graphics::gui::circle_button::erase(uint32_t begin, uint32_t end) {
//	circle::erase(begin, end);
//	m_theme.erase(m_theme.begin() + begin, m_theme.begin() + end);
//	m_reserved.erase(m_reserved.begin() + begin, m_reserved.begin() + end);
//}
//void fan_2d::graphics::gui::circle_button::clear() {
//	circle::clear();
//	m_theme.clear();
//	m_reserved.clear();
//}
//
//void fan_2d::graphics::gui::circle_button::set_locked(uint32_t i, bool flag) {
//	if (flag) {
//		if (m_focused_button_id == i) {
//			m_focused_button_id = fan::uninitialized;
//		}
//		m_reserved[i] |= (uint32_t)button_states_e::locked;
//		m_m_store[i].m_properties.theme = fan_2d::graphics::gui::themes::locked();
//		update_theme(i);
//	}
//	else {
//		m_reserved[i] &= ~(uint32_t)button_states_e::locked;
//		this->update_theme(i);
//		if (inside(i)) {
//			m_focused_button_id = i;
//		}
//	}
//}
//
//bool fan_2d::graphics::gui::circle_button::locked(uint32_t i) const {
//	return m_reserved[i] & (uint32_t)button_states_e::locked;
//}
//
//void fan_2d::graphics::gui::circle_button::enable_draw() {
//	if (circle::m_draw_index == -1 || circle::m_camera->m_window->m_draw_queue[circle::m_draw_index].first != this) {
//		circle::m_draw_index = circle::m_camera->m_window->push_draw_call(this, [&] {
//			this->draw();
//			});
//	}
//	else {
//		circle::m_camera->m_window->edit_draw_call(circle::m_draw_index, this, [&] {
//			this->draw();
//		});
//	}
//}
//void fan_2d::graphics::gui::circle_button::disable_draw() {
//	if (m_draw_index == -1) {
//		return;
//	}
//
//	m_camera->m_window->erase_draw_call(m_draw_index);
//}
//
//void fan_2d::graphics::gui::circle_button::update_theme(uint32_t i) {
//	circle::set_color(i, m_m_store[i].m_properties.theme->button.color);
//}
//
//void fan_2d::graphics::gui::circle_button::lib_add_on_input(fan::window *window, uint32_t i, uint16_t key, fan::key_state state, fan_2d::graphics::gui::mouse_stage stage) {
//
//	if (key != fan::mouse_left) {
//		return;
//	}
//
//	if (stage == mouse_stage::inside && state == fan::key_state::press) {
//
//		circle::set_color(i, m_m_store[i].m_properties.theme->button.click_color);
//
//	}
//	else if (stage == mouse_stage::inside && state == fan::key_state::release) {
//
//		circle::set_color(i, m_m_store[i].m_properties.theme->button.hover_color);
//
//	}
//	else if (stage == mouse_stage::outside && state == fan::key_state::release) {
//
//		circle::set_color(i, m_m_store[i].m_properties.theme->button.color);
//
//		for (int j = 0; j < this->size(); j++) {
//			if (this->inside(j)) {
//
//				circle::set_color(j, m_m_store[i].m_properties.theme->button.hover_color);
//
//				break;
//
//			}
//		}
//	}
//}
//
//void fan_2d::graphics::gui::circle_button::lib_add_on_mouse_move(fan::window *window, uint32_t i, fan_2d::graphics::gui::mouse_stage stage) {
//	if (stage == mouse_stage::inside) {
//		circle::set_color(i, m_m_store[i].m_properties.theme->button.hover_color);
//	}
//	else {
//		circle::set_color(i, m_m_store[i].m_properties.theme->button.color);
//	}
//}
//
//fan::camera* fan_2d::graphics::gui::circle_button::get_camera() {
//	return m_camera;
//}
//
//fan_2d::graphics::gui::rectangle_text_button::rectangle_text_button(fan::camera* camera, fan_2d::graphics::gui::theme theme_)
//	: 
//	fan_2d::graphics::gui::rectangle_text_box(camera, theme_),
//	rectangle_text_button::mouse(this), rectangle_text_button::text_input<rectangle_text_button>(this) {
//}
//
//void fan_2d::graphics::gui::rectangle_text_button::push_back(const properties_t& properties)
//{
//	rectangle_text_button::rectangle_text_box::push_back(properties);
//	rectangle_text_button::text_input::push_back(properties.character_limit, properties.character_width, properties.line_limit);
//}
//void fan_2d::graphics::gui::rectangle_text_button::set_place_holder(uint32_t i, const fan::utf16_string& place_holder)
//{
//	FED_MoveCursorFreeStyleToEndOfLine(&m_wed[i], cursor_reference[i]);
//
//	for (int j = 0; j < m_text[i].size(); j++) {
//		FED_DeleteCharacterFromCursor(&m_wed[i], cursor_reference[i]);
//	}
//
//	rectangle_text_button::text_renderer::set_text_color(i, fan_2d::graphics::gui::defaults::text_color_place_holder);
//	this->set_text(i, place_holder);
//
//	if (m_input_allowed[i]) {
//		update_cursor(i);
//	}
//}
//void fan_2d::graphics::gui::rectangle_text_button::draw()
//{
//	rectangle_text_button::rectangle_text_box::draw();
//	input_instance_t::draw();
//}
//
//void fan_2d::graphics::gui::rectangle_text_button::backspace_callback(uint32_t i)
//{
//	auto current_string = m_box->get_text(i);
//	auto current_property = m_box->get_property(i);
//
//	if (current_string.size() && current_string[0] == '\0' && current_property.place_holder.size()) {
//		m_box->set_text(i, current_property.place_holder);
//		m_box->set_text_color(i, defaults::text_color_place_holder);
//	}
//}
//
//void fan_2d::graphics::gui::rectangle_text_button::text_callback(uint32_t i)
//{
//	if (m_box->get_text_color(i) != m_box->theme.button.text_color) {
//		m_box->set_text_color(i, m_box->theme.button.text_color);
//	}
//}
//
//void fan_2d::graphics::gui::rectangle_text_button::erase(uint32_t i)
//{
//	rectangle_text_button::rectangle_text_box::erase(i);
//	/*rectangle_text_button::mouse::erase(i);
//	rectangle_text_button::text_input::erase(i);*/
//}
//
//void fan_2d::graphics::gui::rectangle_text_button::erase(uint32_t begin, uint32_t end)
//{
//	rectangle_text_button::rectangle_text_box::erase(begin, end);
//	//rectangle_text_button::mouse::erase(begin, end);
//	//rectangle_text_button::text_input::erase(begin, end);
//}
//
//void fan_2d::graphics::gui::rectangle_text_button::clear()
//{
//	rectangle_text_button::rectangle_text_box::clear();
//	//rectangle_text_button::mouse::clear();
//	//rectangle_text_button::text_input::clear();
//}
//
//void fan_2d::graphics::gui::rectangle_text_button::lib_add_on_input(fan::window *window, uint32_t i, uint16_t key, fan::key_state state, fan_2d::graphics::gui::mouse_stage stage)
//{
//	if (key != fan::mouse_left) {
//		return;
//	}
//
//	if (stage == mouse_stage::inside && state == fan::key_state::press) {
//
//		inner_rect_t::set_color(i, theme.button.click_color);
//
//		outer_rect_t::set_color(i * 4 + 0, theme.button.click_outline_color);
//		outer_rect_t::set_color(i * 4 + 1, theme.button.click_outline_color);
//		outer_rect_t::set_color(i * 4 + 2, theme.button.click_outline_color);
//		outer_rect_t::set_color(i * 4 + 3, theme.button.click_outline_color);
//
//	}
//	else if (stage == mouse_stage::inside && state == fan::key_state::release) {
//
//		inner_rect_t::set_color(i, theme.button.hover_color);
//
//		outer_rect_t::set_color(i * 4 + 0, theme.button.hover_outline_color);
//		outer_rect_t::set_color(i * 4 + 1, theme.button.hover_outline_color);
//		outer_rect_t::set_color(i * 4 + 2, theme.button.hover_outline_color);
//		outer_rect_t::set_color(i * 4 + 3, theme.button.hover_outline_color);
//
//	}
//	else if (stage == mouse_stage::outside && state == fan::key_state::release) {
//
//		inner_rect_t::set_color(i, theme.button.color);
//
//		outer_rect_t::set_color(i * 4 + 0, theme.button.outline_color);
//		outer_rect_t::set_color(i * 4 + 1, theme.button.outline_color);
//		outer_rect_t::set_color(i * 4 + 2, theme.button.outline_color);
//		outer_rect_t::set_color(i * 4 + 3, theme.button.outline_color);
//
//		for (int j = 0; j < this->size(); j++) {
//			if (this->inside(j)) {
//
//				inner_rect_t::set_color(j, theme.button.hover_color);
//
//				outer_rect_t::set_color(j * 4 + 0, theme.button.hover_outline_color);
//				outer_rect_t::set_color(j * 4 + 1, theme.button.hover_outline_color);
//				outer_rect_t::set_color(j * 4 + 2, theme.button.hover_outline_color);
//				outer_rect_t::set_color(j * 4 + 3, theme.button.hover_outline_color);
//
//				break;
//
//			}
//		}
//
//	}
//}
//
//void fan_2d::graphics::gui::rectangle_text_button::lib_add_on_mouse_move(fan::window *window, uint32_t i, fan_2d::graphics::gui::mouse_stage stage)
//{
//	
//	switch (stage) {
//    case mouse_stage::inside: {
//      
//			inner_rect_t::set_color(i, theme.button.hover_color);
//
//			outer_rect_t::set_color(i * 4 + 0, theme.button.hover_outline_color);
//			outer_rect_t::set_color(i * 4 + 1, theme.button.hover_outline_color);
//			outer_rect_t::set_color(i * 4 + 2, theme.button.hover_outline_color);
//			outer_rect_t::set_color(i * 4 + 3, theme.button.hover_outline_color);
//
//      break;
//    }
//    default: { // outside, outside drag
//
//			inner_rect_t::set_color(i, theme.button.color);
//
//			outer_rect_t::set_color(i * 4 + 0, theme.button.outline_color);
//			outer_rect_t::set_color(i * 4 + 1, theme.button.outline_color);
//			outer_rect_t::set_color(i * 4 + 2, theme.button.outline_color);
//			outer_rect_t::set_color(i * 4 + 3, theme.button.outline_color);
//
//      break;
//    }
//  }
//}
//
//bool fan_2d::graphics::gui::rectangle_text_button::locked(uint32_t i) const
//{
//	return false; // ?
//}
//
void fan_2d::graphics::gui::text_renderer_clickable::open(fan::window* window, fan::opengl::context_t* context) {
	text_renderer::open(context);
	m_button_event.open(window, context);
	m_store.open();
}

void fan_2d::graphics::gui::text_renderer_clickable::close(fan::window* window, fan::opengl::context_t* context) {
	text_renderer::close(context);
	m_button_event.close(window, context);
	m_store.close();
}

void fan_2d::graphics::gui::text_renderer_clickable::lib_add_on_input(fan::window* window, fan::opengl::context_t* context, uint32_t i, uint16_t key, fan::key_state state, fan_2d::graphics::gui::mouse_stage stage) {

	if (key != fan::mouse_left) {
		return;
	}

	switch (stage) {
	case mouse_stage::inside: {
		switch (state) {
		case fan::key_state::press: {

			if (m_store[i].previous_states == 2) {
				text_renderer::set_text_color(context, i, text_renderer::get_text_color(context, i) + (click_strength - hover_strength));
				fan::color c;
				if ((c = text_renderer::get_outline_color(context, i)) != 0) {
					text_renderer::set_outline_color(context, i, text_renderer::get_outline_color(context, i) + (click_strength - hover_strength));
				}
			}
			else {
				text_renderer::set_text_color(context, i, text_renderer::get_text_color(context, i) + click_strength);
				fan::color c;
				if ((c = text_renderer::get_outline_color(context, i)) != 0) {
					text_renderer::set_outline_color(context, i, text_renderer::get_outline_color(context, i) + click_strength);
				}
			}

			m_store[i].previous_states = 1;
			// click
			break;
		}
		case fan::key_state::release: {

			text_renderer::set_text_color(context, i, text_renderer::get_text_color(context, i) - (click_strength - hover_strength));
			fan::color c;
			if ((c = text_renderer::get_outline_color(context, i)) != 0) {
				text_renderer::set_outline_color(context, i, text_renderer::get_outline_color(context, i) - (click_strength - hover_strength));
			}

			m_store[i].previous_states = 2;
			// hover
			break;
		}
		}
		break;
	}
	case mouse_stage::outside: {

		switch (state) {
		case fan::key_state::release: {

			text_renderer::set_text_color(context, i, text_renderer::get_text_color(context, i) - click_strength);
			fan::color c;
			if ((c = text_renderer::get_outline_color(context, i)) != 0) {
				text_renderer::set_outline_color(context, i, text_renderer::get_outline_color(context, i) - click_strength);
			}

			m_store[i].previous_states = 0;

			// return original
			break;
		}
		}

		break;
	}
	case mouse_stage::inside_drag: {

		if (m_store[i].previous_states == 2) {
			return;
		}

		text_renderer::set_text_color(context, i, text_renderer::get_text_color(context, i) + hover_strength);
		fan::color c;
		if ((c = text_renderer::get_outline_color(context, i)) != 0) {
			text_renderer::set_outline_color(context, i, text_renderer::get_outline_color(context, i) + hover_strength);
		}

		m_store[i].previous_states = 2;

		break;
	}
	}

}

void fan_2d::graphics::gui::text_renderer_clickable::lib_add_on_mouse_move(fan::window* window, fan::opengl::context_t* context, uint32_t i, fan_2d::graphics::gui::mouse_stage stage) {

	switch (stage) {
	case mouse_stage::inside: {

		if (m_store[i].previous_states == 2) {
			return;
		}

		text_renderer::set_text_color(context, i, text_renderer::get_text_color(context, i) + hover_strength);
		fan::color c;
		if ((c = text_renderer::get_outline_color(context, i)) != 0) {
			text_renderer::set_outline_color(context, i, text_renderer::get_outline_color(context, i) + hover_strength);
		}

		m_store[i].previous_states = 2;

		break;
	}
	default: { // outside, outside drag

		if (m_button_event.holding_button() == (uint32_t)-1 || m_store[i].previous_states == 0) {
			return;
		}

		switch (m_store[i].previous_states) {
		case 1: {
			text_renderer::set_text_color(context, i, text_renderer::get_text_color(context, i) - click_strength);
			fan::color c;
			if ((c = text_renderer::get_outline_color(context, i)) != 0) {
				text_renderer::set_outline_color(context, i, text_renderer::get_outline_color(context, i) - click_strength);
			}
			break;
		}
		case 2: {
			text_renderer::set_text_color(context, i, text_renderer::get_text_color(context, i) - hover_strength);
			fan::color c;
			if ((c = text_renderer::get_outline_color(context, i)) != 0) {
				text_renderer::set_outline_color(context, i, text_renderer::get_outline_color(context, i) - hover_strength);
			}
			break;
		}
		}

		m_store[i].previous_states = 0;

		break;
	}
	}
}

void fan_2d::graphics::gui::text_renderer_clickable::push_back(fan::window* window, fan::opengl::context_t* context, const text_renderer_clickable::properties_t& properties)
{
	store_t store;
	store.m_hitbox = hitbox_t{ properties.hitbox_position, properties.hitbox_size };
	store.previous_states = 0;
	m_store.push_back(store);

	text_renderer::push_back(context, properties);
}

void fan_2d::graphics::gui::text_renderer_clickable::set_hitbox(fan::window* window, fan::opengl::context_t* context, uint32_t i, const fan::vec2& hitbox_position, const fan::vec2& hitbox_size)
{
	m_store[i].m_hitbox.hitbox_position = hitbox_position;
	m_store[i].m_hitbox.hitbox_size = hitbox_size;
}

fan::vec2 fan_2d::graphics::gui::text_renderer_clickable::get_hitbox_position(fan::window* window, fan::opengl::context_t* context, uint32_t i) const
{
	return m_store[i].m_hitbox.hitbox_position;
}

void fan_2d::graphics::gui::text_renderer_clickable::set_hitbox_position(fan::window* window, fan::opengl::context_t* context, uint32_t i, const fan::vec2& hitbox_position)
{
	m_store[i].m_hitbox.hitbox_position = hitbox_position;
}

fan::vec2 fan_2d::graphics::gui::text_renderer_clickable::get_hitbox_size(fan::window* window, fan::opengl::context_t* context, uint32_t i) const
{
	return m_store[i].m_hitbox.hitbox_size;
}

void fan_2d::graphics::gui::text_renderer_clickable::set_hitbox_size(fan::window* window, fan::opengl::context_t* context, uint32_t i, const fan::vec2& hitbox_size)
{
	m_store[i].m_hitbox.hitbox_size = hitbox_size;
}

bool fan_2d::graphics::gui::text_renderer_clickable::inside(fan::opengl::context_t* context, uint32_t i, const fan::vec2& p) const
{
	const fan::vec2 src = m_store[i].m_hitbox.hitbox_position - m_store[i].m_hitbox.hitbox_size;
	const fan::vec2 dst = m_store[i].m_hitbox.hitbox_position + m_store[i].m_hitbox.hitbox_size;

	return fan_2d::collision::rectangle::point_inside_no_rotation(p, src, dst);
}


void fan_2d::graphics::gui::rectangle_text_button_sized::open(fan::window* window, fan::opengl::context_t* context)
{
	fan_2d::graphics::gui::rectangle_text_box_sized::open(context);
	m_key_event.open(window, context);
	m_button_event.open(window, context);
	m_reserved.open();
}

void fan_2d::graphics::gui::rectangle_text_button_sized::close(fan::window* window, fan::opengl::context_t* context)
{
	fan_2d::graphics::gui::rectangle_text_box_sized::close(context);
	m_key_event.close(window, context);
	m_button_event.close(window, context);
	m_reserved.close();

	if (base::button_event_t<rectangle_text_button_sized>::pointer_remove_flag == 1) {
		base::button_event_t<rectangle_text_button_sized>::pointer_remove_flag = 0;
	}
	else {

	}
}

void fan_2d::graphics::gui::rectangle_text_button_sized::push_back(fan::window* window, fan::opengl::context_t* context, properties_t properties)
{
	m_reserved.emplace_back((uint32_t)properties.button_state);
	if (properties.button_state == button_states_e::locked) {

		properties.theme = fan_2d::graphics::gui::themes::locked();
	}

	rectangle_text_button_sized::rectangle_text_box_sized::push_back(context, properties);

	m_key_event.push_back(window, context, properties.character_limit, properties.character_width, properties.line_limit);

	if (inside(context, size(context) - 1, window->get_mouse_position()) && properties.button_state != button_states_e::locked) {
		m_button_event.m_focused_button_id = size(context) - 1;
		lib_add_on_mouse_move(window, context, m_button_event.m_focused_button_id, fan_2d::graphics::gui::mouse_stage::inside);
	}
}

void fan_2d::graphics::gui::rectangle_text_button_sized::set_place_holder(fan::window* window, fan::opengl::context_t* context, uint32_t i, const fan::utf16_string& place_holder)
{
	FED_MoveCursorFreeStyleToEndOfLine(&m_key_event.m_store[i].m_wed, m_key_event.m_store[i].cursor_reference);

	for (int j = 0; j < ((text_renderer::store_t*)&m_store[i])->m_text->size(); j++) {
		FED_DeleteCharacterFromCursor(&m_key_event.m_store[i].m_wed, m_key_event.m_store[i].cursor_reference);
	}

	rectangle_text_box_sized::text_renderer::set_text_color(context, i, fan_2d::graphics::gui::defaults::text_color_place_holder);
	this->set_text(context, i, place_holder);

	if (m_key_event.m_store[i].m_input_allowed) {
		m_key_event.update_cursor(window, context, i);
	}

}

void fan_2d::graphics::gui::rectangle_text_button_sized::draw(fan::window* window, fan::opengl::context_t* context)
{
	rectangle_text_button_sized::rectangle_text_box_sized::draw(context);
	m_key_event.draw(window, context);
}

void fan_2d::graphics::gui::rectangle_text_button_sized::backspace_callback(fan::window* window, fan::opengl::context_t* context, uint32_t i)
{
	auto current_string = this->get_text(context, i);
	auto current_property = this->get_property(context, i);

	if (current_string.size() && current_string[0] == '\0' && current_property.place_holder.size()) {
		this->set_text(context, i, current_property.place_holder);
		this->set_text_color(context, i, defaults::text_color_place_holder);
	}
}

void fan_2d::graphics::gui::rectangle_text_button_sized::text_callback(fan::window* window, fan::opengl::context_t* context, uint32_t i)
{
	if (this->get_text_color(context, i) != this->m_store[i].m_properties.theme->button.text_color) {
		this->set_text_color(context, i, this->m_store[i].m_properties.theme->button.text_color);
	}
}

void fan_2d::graphics::gui::rectangle_text_button_sized::erase(fan::window* window, fan::opengl::context_t* context, uint32_t i)
{
	rectangle_text_button_sized::rectangle_text_box_sized::erase(context, i);
	m_reserved.erase(i);
	m_key_event.set_focus(window, context, -1);
	m_key_event.erase(window, context, i);
}

void fan_2d::graphics::gui::rectangle_text_button_sized::erase(fan::window* window, fan::opengl::context_t* context, uint32_t begin, uint32_t end)
{
	rectangle_text_button_sized::rectangle_text_box_sized::erase(context, begin, end);
	m_reserved.erase(begin, end);
	m_key_event.set_focus(window, context, -1);
	m_key_event.erase(window, context, begin, end);
}

void fan_2d::graphics::gui::rectangle_text_button_sized::clear(fan::window* window, fan::opengl::context_t* context)
{
	rectangle_text_button_sized::rectangle_text_box_sized::clear(context);
	m_reserved.clear();

	m_key_event.set_focus(window, context, fan::uninitialized);

	// otherwise default add_inputs in constructor will be erased as well
	//rectangle_text_button_sized::mouse::clear();
	//rectangle_text_button_sized::text_input::clear();
}

void fan_2d::graphics::gui::rectangle_text_button_sized::set_locked(fan::window* window, fan::opengl::context_t* context, uint32_t i, bool flag, bool change_theme) {
	if (flag) {
		if (m_button_event.m_focused_button_id == i) {
			m_button_event.m_focused_button_id = fan::uninitialized;
		}
		m_reserved[i] |= (uint32_t)button_states_e::locked;
		if (change_theme) {
			*m_store[i].m_properties.theme = fan_2d::graphics::gui::themes::locked();
			update_theme(context, i);
		}
	}
	else {
		m_reserved[i] &= ~(uint32_t)button_states_e::locked;
		if (inside(context, i, window->get_mouse_position())) {
			m_button_event.m_focused_button_id = i;
		}
	}

}

void fan_2d::graphics::gui::rectangle_text_button_sized::lib_add_on_input(fan::window* window, fan::opengl::context_t* context, uint32_t i, uint16_t key, fan::key_state state, fan_2d::graphics::gui::mouse_stage stage)
{
	if (this->locked(window, context, i)) {
		return;
	}

	m_store[i].m_properties.theme->button.m_click_callback(window, i, key, state, stage);

	if (key != fan::mouse_left) {
		return;
	}

	static auto change_color_f = [context](fan_2d::graphics::gui::rectangle_text_button_sized* rtbs, uint32_t i, const fan::color& in, const fan::color& out) {

		((fan_2d::graphics::gui::rectangle_text_button_sized::inner_rect_t*)rtbs)->set_color(context, i, in);
		((fan_2d::graphics::gui::rectangle_text_button_sized::outer_rect_t*)rtbs)->set_color(context, i * 4 + 0, out);
		((fan_2d::graphics::gui::rectangle_text_button_sized::outer_rect_t*)rtbs)->set_color(context, i * 4 + 1, out);
		((fan_2d::graphics::gui::rectangle_text_button_sized::outer_rect_t*)rtbs)->set_color(context, i * 4 + 2, out);
		((fan_2d::graphics::gui::rectangle_text_button_sized::outer_rect_t*)rtbs)->set_color(context, i * 4 + 3, out);
	};

	switch (stage) {
	case fan_2d::graphics::gui::mouse_stage::inside: {

		switch (state) {
		case fan::key_state::press: {
			change_color_f(this, i, m_store[i].m_properties.theme->button.click_color, m_store[i].m_properties.theme->button.click_outline_color);
			break;
		}
		case fan::key_state::release: {
			focus::set_focus(focus::properties_t(window->get_handle(), this, i));
			if (m_key_event.m_store[i].m_input_allowed) {
				m_key_event.render_cursor = true;
				m_key_event.update_cursor(window, context, i);
				m_key_event.cursor_timer.restart();
				m_key_event.m_cursor.enable_draw(context);
			}
			else {
				m_key_event.render_cursor = false;
			}
			change_color_f(this, i, m_store[i].m_properties.theme->button.hover_color, m_store[i].m_properties.theme->button.hover_outline_color);
			break;
		}
		}
		break;
	}
	case fan_2d::graphics::gui::mouse_stage::outside: {
		switch (state) {
		case fan::key_state::press: {
			focus::set_focus(focus::no_focus);
			break;
		}
		case fan::key_state::release: {
			change_color_f(this, i, m_store[i].m_properties.theme->button.color, m_store[i].m_properties.theme->button.outline_color);
			for (int j = 0; j < this->size(context); j++) {
				if (this->inside(context, j, window->get_mouse_position()) && !this->locked(window, context, j)) {
					change_color_f(this, j, m_store[i].m_properties.theme->button.hover_color, m_store[i].m_properties.theme->button.hover_outline_color);
					break;
				}
			}
			break;
		}
		}
		break;
	}
	}
}

void fan_2d::graphics::gui::rectangle_text_button_sized::lib_add_on_mouse_move(fan::window* window, fan::opengl::context_t* context, uint32_t i, fan_2d::graphics::gui::mouse_stage stage)
{
	if (this->locked(window, context, i)) {
		return;
	}

	m_store[i].m_properties.theme->button.m_hover_callback(window, i, stage);

	switch (stage) {
	case mouse_stage::inside: {

		inner_rect_t::set_color(context, i, m_store[i].m_properties.theme->button.hover_color);
		outer_rect_t::set_color(context, i * 4 + 0, m_store[i].m_properties.theme->button.hover_outline_color);
		outer_rect_t::set_color(context, i * 4 + 1, m_store[i].m_properties.theme->button.hover_outline_color);
		outer_rect_t::set_color(context, i * 4 + 2, m_store[i].m_properties.theme->button.hover_outline_color);
		outer_rect_t::set_color(context, i * 4 + 3, m_store[i].m_properties.theme->button.hover_outline_color);

		break;
	}
	default: { // outside, outside drag

		inner_rect_t::set_color(context, i, m_store[i].m_properties.theme->button.color);
		outer_rect_t::set_color(context, i * 4 + 0, m_store[i].m_properties.theme->button.outline_color);
		outer_rect_t::set_color(context, i * 4 + 1, m_store[i].m_properties.theme->button.outline_color);
		outer_rect_t::set_color(context, i * 4 + 2, m_store[i].m_properties.theme->button.outline_color);
		outer_rect_t::set_color(context, i * 4 + 3, m_store[i].m_properties.theme->button.outline_color);

		break;
	}
	}

}

bool fan_2d::graphics::gui::rectangle_text_button_sized::locked(fan::window* window, fan::opengl::context_t* context, uint32_t i) const
{
	return m_reserved[i] & (uint32_t)button_states_e::locked;
}

void fan_2d::graphics::gui::rectangle_text_button_sized::enable_draw(fan::window* window, fan::opengl::context_t* context)
{
	rectangle_text_box_sized::enable_draw(context);
	m_key_event.enable_draw(window, context);
}

void fan_2d::graphics::gui::rectangle_text_button_sized::disable_draw(fan::window* window, fan::opengl::context_t* context)
{
	rectangle_text_box_sized::disable_draw(context);
	m_key_event.disable_draw(window, context);
}
//
//fan_2d::graphics::gui::sprite_text_box::sprite_text_box(fan::camera* camera, const std::string& path)
//	:  sprite_t(camera), fan_2d::graphics::gui::text_renderer(camera), image(fan_2d::graphics::load_image(camera->m_window, path)) { }
//
//fan::camera* fan_2d::graphics::gui::sprite_text_box::get_camera()
//{
//	return sprite_t::m_camera;
//}
//
//uint64_t fan_2d::graphics::gui::sprite_text_box::size() const {
//	return sprite_t::size();
//}
//
//bool fan_2d::graphics::gui::sprite_text_box::inside(uint32_t i, const fan::vec2& position) const {
//	return sprite_t::inside(i, position);
//}
//
//void fan_2d::graphics::gui::sprite_text_box::push_back(const properties_t& properties)
//{
//	m_properties.emplace_back(properties);
//
//	sprite_t::properties_t s_properties;
//	s_properties.image = image;
//	s_properties.position = properties.position;
//	s_properties.size = get_button_size(
//		properties.text,
//		properties.font_size,
//		text_renderer::get_new_lines(properties.text),
//		properties.padding
//	);
//
//	fan_2d::graphics::gui::text_renderer::properties_t text_properties;
//	text_properties.text = properties.text;
//	text_properties.font_size = properties.font_size;
//	text_properties.position = fan::vec2(properties.position.x + properties.padding.x * 0.5, properties.position.y + properties.padding.y * 0.5);
//	text_properties.text_color = fan_2d::graphics::gui::defaults::text_color;
//
//	fan_2d::graphics::gui::text_renderer::push_back(text_properties);
//
//
//	sprite_t::push_back(s_properties);
//}
//
//void fan_2d::graphics::gui::sprite_text_box::draw(uint32_t begin, uint32_t end)
//{
//	sprite_t::draw(begin, end);
//	fan_2d::graphics::gui::text_renderer::draw();
//}
//
//fan_2d::graphics::gui::sprite_text_button::sprite_text_button(fan::camera* camera, const std::string& path)
//	: fan_2d::graphics::gui::sprite_text_box(camera, path),
//	sprite_text_button::mouse(this) {}
//
//void fan_2d::graphics::gui::sprite_text_button::lib_add_on_input(fan::window *window, uint32_t i, uint16_t key, fan::key_state state, fan_2d::graphics::gui::mouse_stage stage)
//{
//}
//
//void fan_2d::graphics::gui::sprite_text_button::lib_add_on_mouse_move(fan::window *window, uint32_t i, fan_2d::graphics::gui::mouse_stage stage)
//{
//}
//
////fan_2d::graphics::gui::scrollbar::scrollbar(fan::camera* camera) :
////	fan_2d::graphics::rectangle(camera), scrollbar::mouse(this) {
////
////	scrollbar::mouse::add_on_input([&](uint32_t i, fan::key_state state, mouse_stage mouse_stage) {
////		
////		if (state == fan::key_state::press && mouse_stage == mouse_stage::inside) {
////
////			if (i % 2 == 0) {
////				i += 1;
////			}
////
////			fan::vec2 offset;
////
////			fan::vec2 current = rectangle::m_camera->m_window->get_mouse_position();
////
////			switch (m_properties[i / 2].scroll_direction) {
////				case scroll_direction_e::horizontal: {
////
////					f32_t current_x = current.x - rectangle::get_size(i).x / 2;
////
////					f32_t min_x = rectangle::get_position(i - 1).x + m_properties[i / 2].outline_thickness;
////					f32_t max_x = rectangle::get_position(i - 1).x + rectangle::get_size(i - 1).x - rectangle::get_size(i).x - m_properties[i / 2].outline_thickness;
////
////					offset.y = rectangle::get_position(i).y;
////
////					offset.x = fan::clamp(
////						current_x,
////						min_x,
////						max_x
////					);
////
////					if (fan_2d::graphics::rectangle::get_position(i) == offset) {
////						return;
////					}
////
////					m_properties[i / 2].current = offset.x - min_x;
////
////					fan_2d::graphics::rectangle::set_position(i, offset);
////
////					break;
////				}
////				case scroll_direction_e::vertical: {
////
////					f32_t current_y = current.y - rectangle::get_size(i).y / 2;
////
////					f32_t min_y = rectangle::get_position(i - 1).y + m_properties[i / 2].outline_thickness;
////					f32_t max_y = rectangle::get_position(i - 1).y + rectangle::get_size(i - 1).y - rectangle::get_size(i).y - m_properties[i / 2].outline_thickness;
////
////					offset.x = rectangle::get_position(i).x;
////
////					offset.y = fan::clamp(
////						current_y,
////						min_y,
////						max_y
////					);
////
////					if (fan_2d::graphics::rectangle::get_position(i) == offset) {
////						return;
////					}
////
////					m_properties[i / 2].current = offset.y - min_y;
////
////					fan_2d::graphics::rectangle::set_position(i, offset);
////
////					break;
////				}
////			}
////
////			for (int j = 0; j < m_on_scroll.size(); j++) {
////				if (m_on_scroll[j]) {
////					m_on_scroll[j](i / 2, m_properties[j].current);
////				}
////			}
////
////			rectangle::edit_data(i);
////		}
////		
////	}, false);
////
////	scrollbar::mouse::add_on_mouse_event([&](uint32_t i, mouse_stage mouse_stage) {
////		
////		if (scrollbar::holding_button() != fan::uninitialized && (mouse_stage == mouse_stage::inside || 
////			mouse_stage == mouse_stage::outside_drag)) {
////
////			if (i % 2 == 0) {
////				i += 1;
////			}
////
////			fan::vec2 current = rectangle::m_camera->m_window->get_mouse_position();
////
////			fan::vec2 offset;
////
////			switch (m_properties[i / 2].scroll_direction) {
////				case scroll_direction_e::horizontal: {
////
////					offset.x += current.x - (rectangle::get_position(i - 1).x + m_properties[i / 2].outline_thickness);
////
////					f32_t current_x = rectangle::get_position(i - 1).x + offset.x - rectangle::get_size(i).x / 2;
////					//fan::print(current_x);
////					f32_t min_x = rectangle::get_position(i - 1).x + m_properties[i / 2].outline_thickness;
////					f32_t max_x = rectangle::get_position(i - 1).x + rectangle::get_size(i - 1).x - rectangle::get_size(i).x - m_properties[i / 2].outline_thickness;
////
////					offset.y = rectangle::get_position(i).y;
////
////					offset.x = fan::clamp(
////						current_x,
////						min_x,
////						max_x
////					);
////
////					if (fan_2d::graphics::rectangle::get_position(i) == offset) {
////						return;
////					}
////
////					m_properties[i / 2].current = offset.x - min_x;
////
////					fan_2d::graphics::rectangle::set_position(i, offset);
////
////					break;
////				}
////				case scroll_direction_e::vertical: {
////					offset.y += current.y - (rectangle::get_position(i - 1).y + m_properties[i / 2].outline_thickness);
////
////					f32_t current_y = rectangle::get_position(i - 1).y + offset.y - rectangle::get_size(i).y / 2;
////					//fan::print(current_y);
////					f32_t min_y = rectangle::get_position(i - 1).y + m_properties[i / 2].outline_thickness;
////					f32_t max_y = rectangle::get_position(i - 1).y + rectangle::get_size(i - 1).y - rectangle::get_size(i).y - m_properties[i / 2].outline_thickness;
////
////					offset.x = rectangle::get_position(i).x;
////
////					offset.y = fan::clamp(
////						current_y,
////						min_y,
////						max_y
////					);
////
////					if (fan_2d::graphics::rectangle::get_position(i) == offset) {
////						return;
////					}
////
////					m_properties[i / 2].current = offset.y - min_y;
////
////					fan_2d::graphics::rectangle::set_position(i, offset);
////
////					break;
////				}
////			}
////
////			for (int j = 0; j < m_on_scroll.size(); j++) {
////				if (m_on_scroll[j]) {
////					m_on_scroll[j](i / 2, m_properties[j].current);
////				}
////			}
////
////			rectangle::edit_data(i);
////		}
////
////	}, false);
////
////}
////
////void fan_2d::graphics::gui::scrollbar::push_back(const properties_t& instance)
////{
////	fan_2d::graphics::rectangle::properties_t r_property;
////	r_property.position = instance.position - (f32_t)instance.outline_thickness;
////	r_property.size = 
////		instance.size +
////		(instance.scroll_direction == scroll_direction_e::horizontal ?
////		fan::vec2(instance.length, 0) :
////		fan::vec2(0, instance.length)) + instance.outline_thickness * 2;
////	r_property.color = instance.color - 0.5;
////
////	fan_2d::graphics::rectangle::push_back(r_property);
////
////	r_property.position = instance.position;
////	r_property.size = instance.size;
////	r_property.color = instance.color;
////
////	fan_2d::graphics::rectangle::push_back(r_property);
////
////	scroll_properties_t sbp;
////	sbp.length = instance.length;
////	sbp.outline_thickness = instance.outline_thickness;
////	sbp.scroll_direction = instance.scroll_direction;
////	sbp.current = instance.current;
////
////	m_properties.emplace_back(sbp);
////
////	scrollbar::mouse::push_back();
////	scrollbar::mouse::push_back();
////}
////
////void fan_2d::graphics::gui::scrollbar::draw()
////{
////	fan_2d::graphics::rectangle::draw();
////}
////
////void fan_2d::graphics::gui::scrollbar::write_data()
////{
////	fan_2d::graphics::rectangle::write_data();
////}
////
////fan::camera* fan_2d::graphics::gui::scrollbar::get_camera()
////{
////	return fan_2d::graphics::rectangle::m_camera;
////}
////
////void fan_2d::graphics::gui::scrollbar::add_on_scroll(on_scroll_t function)
////{
////	m_on_scroll.emplace_back(function);
////}
//
//fan_2d::graphics::gui::checkbox::checkbox(fan::camera* camera, fan_2d::graphics::gui::theme theme)
//	: checkbox::rectangle_t(camera), 
//	checkbox::line_t(camera), 
//	checkbox::text_renderer_t(camera),
//	checkbox::mouse(this),
//	 m_theme(theme) {
//
//}
//
//void fan_2d::graphics::gui::checkbox::push_back(const checkbox::properties_t& property)
//{
//	m_properties.emplace_back(property);
//
//	m_visible.emplace_back(property.checked);
//	
//	f32_t text_middle_height = text_renderer_t::get_line_height(property.font_size);
//
//	fan_2d::graphics::rectangle::properties_t properties;
//	properties.position = property.position;
//	properties.size = text_middle_height / 2 * property.box_size_multiplier;
//	properties.color = m_theme.checkbox.color;
//	properties.rotation_point = properties.position;
//
//	checkbox::rectangle_t::push_back(properties); 
//
//	auto corners = checkbox::rectangle_t::get_corners(checkbox::rectangle_t::size() - 1);
//
//	checkbox::line_t::push_back(corners.top_left, corners.bottom_right, m_theme.checkbox.check_color, property.line_thickness / 2); // might be varying position
//	checkbox::line_t::push_back(corners.bottom_left, corners.top_right, m_theme.checkbox.check_color, property.line_thickness / 2); // might be varying position
//
//	auto text_size = text_renderer_t::get_text_size(property.text, property.font_size);
//
//	fan_2d::graphics::gui::text_renderer::properties_t text_properties;
//
//	text_properties.text = property.text;
//	text_properties.font_size = property.font_size;
//	text_properties.position = property.position + fan::vec2(text_size.x / 2 + properties.size.x + 5 * property.box_size_multiplier, 0);
//	text_properties.text_color = m_theme.checkbox.text_color;
//}
//
//void fan_2d::graphics::gui::checkbox::draw()
//{
//	// depth test
//	//fan_2d::graphics::draw([&] {
//		
//		for (int i = 0; i < fan_2d::graphics::line::size() / 2; i++) {
//			checkbox::rectangle_t::draw();
//
//			if (m_visible[i]) {
//				checkbox::line_t::draw();
//				//fan_2d::graphics::line::draw(i * 2 + 1, i * 2 + 2);
//			}
//		}
//
//		checkbox::text_renderer_t::draw();
//	//});
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
//uint32_t fan_2d::graphics::gui::checkbox::size() const
//{
//	return rectangle_t::size();
//}
//
//bool fan_2d::graphics::gui::checkbox::inside(uint32_t i, const fan::vec2& position) const
//{
//	return rectangle_t::inside(i, position);
//}
//
//fan::camera* fan_2d::graphics::gui::checkbox::get_camera()
//{
//	return checkbox::rectangle_t::m_camera;
//}
//
//void fan_2d::graphics::gui::checkbox::write_data() {
//	checkbox::line_t::write_data();
//	checkbox::rectangle_t::write_data();
//	checkbox::text_renderer_t::write_data();
//}
//
//void fan_2d::graphics::gui::checkbox::edit_data(uint32_t i) {
//	checkbox::line_t::edit_data(i);
//	checkbox::rectangle_t::edit_data(i);
//	checkbox::text_renderer_t::edit_data(i);
//}
//void fan_2d::graphics::gui::checkbox::edit_data(uint32_t begin, uint32_t end) {
//	checkbox::line_t::edit_data(begin, end);
//	checkbox::rectangle_t::edit_data(begin, end);
//	checkbox::text_renderer_t::edit_data(begin, end);
//}
//
//void fan_2d::graphics::gui::checkbox::lib_add_on_input(fan::window *window, uint32_t i, uint16_t key, fan::key_state state, fan_2d::graphics::gui::mouse_stage stage)
//{
//	if (key != fan::mouse_left) {
//		return;
//	}
//
//	if (stage == mouse_stage::inside && state == fan::key_state::release) {
//		checkbox::rectangle_t::set_color(i, m_theme.checkbox.click_color);
//
//		m_visible[i] = !m_visible[i];
//
//		if (m_visible[i]) {
//			if (m_on_check) {
//				m_on_check(i);
//			}
//		}
//		else {
//			if (m_on_uncheck) {
//				m_on_uncheck(i);
//			}
//		}
//		this->edit_data(i);
//	}
//	else if (stage == mouse_stage::outside && state == fan::key_state::release) {
//		checkbox::rectangle_t::set_color(i, m_theme.checkbox.color);
//		this->edit_data(i);
//	}
//}
//
//void fan_2d::graphics::gui::checkbox::lib_add_on_mouse_move(fan::window *window, uint32_t i, fan_2d::graphics::gui::mouse_stage stage)
//{
//	switch (stage) {
//		case mouse_stage::inside: {
//			
//			checkbox::rectangle_t::set_color(i, m_theme.checkbox.hover_color);
//			this->edit_data(i);
//
//			break;
//		}
//		default: { // outside, outside drag
//
//			checkbox::rectangle_t::set_color(i, m_theme.checkbox.color);
//			this->edit_data(i);
//
//			break;
//		}
//	}
//}
//
//void fan_2d::graphics::gui::checkbox::enable_draw()
//{
//	if (checkbox::text_renderer_t::m_draw_index == -1 || checkbox::text_renderer_t::m_camera->m_window->m_draw_queue[checkbox::text_renderer_t::m_draw_index].first != this) {
//		checkbox::text_renderer_t::m_draw_index = checkbox::text_renderer_t::m_camera->m_window->push_draw_call(this, [&] {
//			this->draw();
//			});
//	}
//	else {
//		checkbox::text_renderer_t::m_camera->m_window->edit_draw_call(checkbox::text_renderer_t::m_draw_index, this, [&] {
//			this->draw();
//		});
//	}
//}
//
//void fan_2d::graphics::gui::checkbox::disable_draw()
//{
//	checkbox::line_t::disable_draw();
//	checkbox::rectangle_t::disable_draw();
//	checkbox::text_renderer_t::disable_draw();
//}
//
//fan_2d::graphics::gui::rectangle_selectable_button_sized::rectangle_selectable_button_sized(fan::camera* camera) :
//	rectangle_text_button_sized(true, camera)
//{
//
//}
//
//uint32_t fan_2d::graphics::gui::rectangle_selectable_button_sized::get_selected(uint32_t i) const
//{
//	return this->m_selected;
//}
//
//void fan_2d::graphics::gui::rectangle_selectable_button_sized::set_selected(uint32_t i)
//{
//	this->m_selected = i;
//
//	inner_rect_t::set_color(i, m_store[i].m_properties.theme->button.hover_color);
//
//	outer_rect_t::set_color(i * 4 + 0, m_store[i].m_properties.theme->button.hover_outline_color);
//	outer_rect_t::set_color(i * 4 + 1, m_store[i].m_properties.theme->button.hover_outline_color);
//	outer_rect_t::set_color(i * 4 + 2, m_store[i].m_properties.theme->button.hover_outline_color);
//	outer_rect_t::set_color(i * 4 + 3, m_store[i].m_properties.theme->button.hover_outline_color);
//
//	inner_rect_t::edit_data(i);
//
//	outer_rect_t::edit_data(i * 4, i * 4 + 4);
//}
//
//void fan_2d::graphics::gui::rectangle_selectable_button_sized::add_on_select(std::function<void(uint32_t)> function)
//{
//	m_on_select.push_back(function);
//}
//
//void fan_2d::graphics::gui::rectangle_selectable_button_sized::lib_add_on_input(fan::window *window, uint32_t i, uint16_t key, fan::key_state state, fan_2d::graphics::gui::mouse_stage stage)
//{
//	if (stage == mouse_stage::inside && state == fan::key_state::press) {
//
//		inner_rect_t::set_color(i, m_store[i].m_properties.theme->button.click_color);
//
//		outer_rect_t::set_color(i * 4 + 0, m_store[i].m_properties.theme->button.click_outline_color);
//		outer_rect_t::set_color(i * 4 + 1, m_store[i].m_properties.theme->button.click_outline_color);
//		outer_rect_t::set_color(i * 4 + 2, m_store[i].m_properties.theme->button.click_outline_color);
//		outer_rect_t::set_color(i * 4 + 3, m_store[i].m_properties.theme->button.click_outline_color);
//
//		inner_rect_t::edit_data(i);
//
//		outer_rect_t::edit_data(i * 4, i * 4 + 4);
//	}
//	else if (stage == mouse_stage::inside && state == fan::key_state::release) {
//
//		inner_rect_t::set_color(i, m_store[i].m_properties.theme->button.hover_color);
//
//		outer_rect_t::set_color(i * 4 + 0, m_store[i].m_properties.theme->button.hover_outline_color);
//		outer_rect_t::set_color(i * 4 + 1, m_store[i].m_properties.theme->button.hover_outline_color);
//		outer_rect_t::set_color(i * 4 + 2, m_store[i].m_properties.theme->button.hover_outline_color);
//		outer_rect_t::set_color(i * 4 + 3, m_store[i].m_properties.theme->button.hover_outline_color);
//
//		inner_rect_t::edit_data(i);
//
//		outer_rect_t::edit_data(i * 4, i * 4 + 4);
//
//		// default previous selected
//		if (m_selected != i && m_selected != (uint32_t)fan::uninitialized) {
//			inner_rect_t::set_color(m_selected, m_store[i].m_properties.theme->button.color);
//
//			outer_rect_t::set_color(m_selected * 4 + 0, m_store[i].m_properties.theme->button.outline_color);
//			outer_rect_t::set_color(m_selected * 4 + 1, m_store[i].m_properties.theme->button.outline_color);
//			outer_rect_t::set_color(m_selected * 4 + 2, m_store[i].m_properties.theme->button.outline_color);
//			outer_rect_t::set_color(m_selected * 4 + 3, m_store[i].m_properties.theme->button.outline_color);
//
//			inner_rect_t::edit_data(m_selected);
//
//			outer_rect_t::edit_data(m_selected * 4, m_selected * 4 + 4);
//		}
//
//		if (m_selected != i) {
//			for (const auto& f : m_on_select) {
//				if (f) {
//					f(i);
//				}
//			}
//		}
//
//		m_selected = i;
//
//	}
//	else if (stage == mouse_stage::outside && state == fan::key_state::release) {
//
//		if (m_selected != i) {
//			inner_rect_t::set_color(i, m_store[i].m_properties.theme->button.color);
//
//			outer_rect_t::set_color(i * 4 + 0, m_store[i].m_properties.theme->button.outline_color);
//			outer_rect_t::set_color(i * 4 + 1, m_store[i].m_properties.theme->button.outline_color);
//			outer_rect_t::set_color(i * 4 + 2, m_store[i].m_properties.theme->button.outline_color);
//			outer_rect_t::set_color(i * 4 + 3, m_store[i].m_properties.theme->button.outline_color);
//
//			inner_rect_t::edit_data(i);
//
//			outer_rect_t::edit_data(i * 4, i * 4 + 4);
//		}
//	}
//}
//
//void fan_2d::graphics::gui::rectangle_selectable_button_sized::lib_add_on_mouse_move(fan::window *window, uint32_t i, fan_2d::graphics::gui::mouse_stage stage)
//{
//	switch (stage) {
//    case mouse_stage::inside: {
//      
//			inner_rect_t::set_color(i, m_store[i].m_properties.theme->button.hover_color);
//
//			outer_rect_t::set_color(i * 4 + 0, m_store[i].m_properties.theme->button.hover_outline_color);
//			outer_rect_t::set_color(i * 4 + 1, m_store[i].m_properties.theme->button.hover_outline_color);
//			outer_rect_t::set_color(i * 4 + 2, m_store[i].m_properties.theme->button.hover_outline_color);
//			outer_rect_t::set_color(i * 4 + 3, m_store[i].m_properties.theme->button.hover_outline_color);
//
//			inner_rect_t::edit_data(i);
//
//			outer_rect_t::edit_data(i * 4, i * 4 + 4);
//
//      break;
//    }
//    case mouse_stage::outside_drag: {
//
//			if (m_selected != i) {
//
//				inner_rect_t::set_color(i, m_store[i].m_properties.theme->button.hover_color);
//
//				outer_rect_t::set_color(i * 4 + 0, m_store[i].m_properties.theme->button.hover_outline_color);
//				outer_rect_t::set_color(i * 4 + 1, m_store[i].m_properties.theme->button.hover_outline_color);
//				outer_rect_t::set_color(i * 4 + 2, m_store[i].m_properties.theme->button.hover_outline_color);
//				outer_rect_t::set_color(i * 4 + 3, m_store[i].m_properties.theme->button.hover_outline_color);
//
//				inner_rect_t::edit_data(i);
//
//				outer_rect_t::edit_data(i * 4, i * 4 + 4);
//			}
//
//      break;
//    }
//		case mouse_stage::outside: {
//
//			if (m_selected != i) {
//				inner_rect_t::set_color(i, m_store[i].m_properties.theme->button.color);
//
//				outer_rect_t::set_color(i * 4 + 0, m_store[i].m_properties.theme->button.outline_color);
//				outer_rect_t::set_color(i * 4 + 1, m_store[i].m_properties.theme->button.outline_color);
//				outer_rect_t::set_color(i * 4 + 2, m_store[i].m_properties.theme->button.outline_color);
//				outer_rect_t::set_color(i * 4 + 3, m_store[i].m_properties.theme->button.outline_color);
//
//				inner_rect_t::edit_data(i);
//
//				outer_rect_t::edit_data(i * 4, i * 4 + 4);
//			}
//
//			break;
//		}
//  }
//}
//
////fan_2d::graphics::gui::dropdown_menu::dropdown_menu(fan::camera* camera, const fan_2d::graphics::gui::theme& theme) :
////	fan_2d::graphics::gui::rectangle_text_button_sized(camera, theme)
////{
////
////	add_on_input([&](uint32_t i, fan::key_state state, mouse_stage stage) {
////		
////		switch (stage) {
////			case mouse_stage::inside: {
////
////				if (state != fan::key_state::release) {
////					return;
////				}
////
////				uint32_t offset = 0;
////
////				int j = 0;
////
////				for (; offset < i; j++) {
////					offset += m_amount_per_menu[j];
////				}
////
////				if (offset < m_amount_per_menu.size()) {
////					m_hovered = offset;
////				}
////				else {
////					fan::print("selected", i);
////					m_hovered = fan::uninitialized;
////				}
////
////				break;
////			}
////		}
////
////	});
////
////	add_on_mouse_event([&] (uint32_t i, mouse_stage mouse_stage) {
////
////		switch (mouse_stage) {
////			case mouse_stage::inside: {
////
////				uint32_t offset = 0;
////
////				int j = 0;
////
////				for (; offset < i; j++) {
////					offset += m_amount_per_menu[j];
////				}
////
////				if (offset < m_amount_per_menu.size()) {
////					m_hovered = offset;
////				}
////
////				break;
////			}
////			default: {
////
////				uint32_t offset = 0;
////
////				int j = 0;
////
////				for (; offset < i; j++) {
////					offset += m_amount_per_menu[j];
////				}
////
////				src_dst_t hitbox = m_hitboxes[offset - (i > 0 ? m_amount_per_menu[j - 1] : 0)];
////
////				if (!fan_2d::collision::rectangle::point_inside_no_rotation(get_camera()->m_window->get_mouse_position(), hitbox.src, hitbox.dst)) {
////					m_hovered = fan::uninitialized;
////				}
////
////				break;
////			}
////		}
////
////	}, false);
////}
////
////void fan_2d::graphics::gui::dropdown_menu::push_back(const properties_t& property)
////{
////	m_amount_per_menu.emplace_back(property.dropdown_texts.size() + 1);
////
////	rectangle_text_button_sized::properties_t rp;
////	rp.position = property.position;
////	rp.size = property.size;
////	rp.text_position = property.text_position;
////	rp.font_size = property.font_size;
////	rp.text = property.text;
////	rp.advance = property.advance;
////
////	src_dst_t hitbox;
////	hitbox.src = rp.position - rp.size * 0.5;
////
////	rectangle_text_button_sized::push_back(rp);
////
////	for (int i = 0; i < property.dropdown_texts.size(); i++) {
////		rp.text = property.dropdown_texts[i];
////		rp.position.y += rp.size.y;
////		rectangle_text_button_sized::push_back(rp);
////	}
////
////	hitbox.dst = rp.position + fan::vec2(rp.size.x * 0.5, rp.size.y * 0.5);
////
////	m_hitboxes.emplace_back(hitbox);
////}
////
////void fan_2d::graphics::gui::dropdown_menu::draw()
////{
////	uint32_t offset = 0;
////	for (int i = 0; i < m_amount_per_menu.size(); i++) {
////		rectangle_text_button_sized::draw(offset, (i == m_hovered) ? m_amount_per_menu[i] : 1);
////		offset += m_amount_per_menu[i];
////	}
////}
//
//fan_2d::graphics::gui::progress_bar::progress_bar(fan::camera* camera)
//	: fan_2d::graphics::rectangle(camera)
//{
//}
//
//void fan_2d::graphics::gui::progress_bar::push_back(const properties_t& properties)
//{
//	rectangle_t::properties_t p;
//	p.position = properties.position;
//	p.size = properties.size;
//	p.color = properties.back_color;
//
//	progress_bar_t bp;
//	bp.progress_x = properties.progress_x;
//	bp.inner_size_multipliers = properties.inner_size_multiplier;
//	bp.progress = properties.progress;
//
//	if (properties.progress_x) {
//		f32_t original_x_size = p.size.x;
//
//		rectangle_t::push_back(p);
//
//		f32_t left_gap = original_x_size - original_x_size * properties.inner_size_multiplier;
//
//		p.size -= left_gap;
//	 
//		f32_t x = p.size.x;
//
//		p.size.x *= properties.progress / 100;
//
//		p.position.x = properties.position.x - original_x_size + p.size.x + left_gap;
//
//		p.color = properties.front_color;
//
//		rectangle_t::push_back(p);
//	}
//	else {
//		f32_t original_y_size = p.size.y;
//
//		rectangle_t::push_back(p);
//
//		f32_t top_gap = original_y_size - original_y_size * properties.inner_size_multiplier;
//
//		p.size -= top_gap;
//	 
//		f32_t y = p.size.y;
//
//		p.size.y *= properties.progress / 100;
//
//		p.position.y = properties.position.y + original_y_size - p.size.y + top_gap;
//
//		p.color = properties.front_color;
//
//		rectangle_t::push_back(p);
//	}
//	
//
//	m_progress_bar_properties.emplace_back(bp);
//}
//
//void fan_2d::graphics::gui::progress_bar::clear()
//{
//	rectangle_t::clear();
//	m_progress_bar_properties.clear();
//}
//
//f32_t fan_2d::graphics::gui::progress_bar::get_progress(uint32_t i) const
//{
//	return m_progress_bar_properties[i].progress;
//}
//
//void fan_2d::graphics::gui::progress_bar::set_progress(uint32_t i, f32_t progress)
//{
//	progress = fan::clamp(progress, 0.0f, 100.0f);
//
//	m_progress_bar_properties[i].progress = progress;
//
//	if (m_progress_bar_properties[i].progress_x) {
//		fan::vec2 background_size = rectangle_t::get_size(i * 2);
//
//		f32_t original_x_size = background_size.x;
//
//		f32_t left_gap = original_x_size - original_x_size * m_progress_bar_properties[i].inner_size_multipliers;
//
//		background_size -= left_gap;
//	 
//		f32_t x = background_size.x;
//
//		background_size.x *= m_progress_bar_properties[i].progress / 100;
//	
//		fan::vec2 position = rectangle_t::get_position(i * 2);
//
//		rectangle_t::set_position(i * 2 + 1, fan::vec2(position.x - original_x_size + background_size.x + left_gap, position.y));
//		rectangle_t::set_size(i * 2 + 1, background_size);
//
//	}
//	else {
//		fan::vec2 background_size = rectangle_t::get_size(i * 2);
//
//		f32_t original_y_size = background_size.y;
//
//		f32_t top_gap = original_y_size - original_y_size * m_progress_bar_properties[i].inner_size_multipliers;
//
//		background_size -= top_gap;
//	 
//		f32_t y = background_size.y;
//
//		background_size.y *= m_progress_bar_properties[i].progress / 100;
//	
//		fan::vec2 position = rectangle_t::get_position(i * 2);
//
//		rectangle_t::set_position(i * 2 + 1, fan::vec2(position.x, position.y + original_y_size - background_size.y + top_gap));
//		rectangle_t::set_size(i * 2 + 1, background_size);
//	}
//
//}
//
//fan::vec2 fan_2d::graphics::gui::progress_bar::get_position(uint32_t i) const
//{
//	return rectangle_t::get_position(i * 2);
//}
//
//void fan_2d::graphics::gui::progress_bar::set_position(uint32_t i, const fan::vec2& position)
//{
//	const fan::vec2 offset = position - rectangle_t::get_position(i * 2);
//
//	rectangle_t::set_position(i * 2, rectangle_t::get_position(i * 2) + offset);
//	rectangle_t::set_position(i * 2 + 1, rectangle_t::get_position(i * 2 + 1) + offset);
//}


