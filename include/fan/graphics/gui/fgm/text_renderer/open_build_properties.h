case builder_draw_type_t::text_renderer: {

  fan::vec2 position = pile->builder.tr.get_position(&pile->context, click_collision_.builder_draw_type_index);
  fan::vec2 size = pile->builder.tr.get_text_size(&pile->context, click_collision_.builder_draw_type_index);

  decltype(pile->editor.properties_button)::properties_t properties_button_p;
  properties_button_p.text = fan::to_wstring(position.x, 3) + L", " + fan::to_wstring(position.y, 3);
  properties_button_p.size = fan::vec2(constants::gui_size * 3, constants::gui_size);
  properties_button_p.font_size = constants::gui_size;
  properties_button_p.theme = fan_2d::graphics::gui::themes::gray();
  properties_button_p.theme.button.outline_size = 0.001;
  properties_button_p.text_position = fan_2d::graphics::gui::text_position_e::left;
  properties_button_p.allow_input = true;
  properties_button_p.position = fan::vec2(
    constants::properties_box_pad,
    fan::vec2(pile->editor.builder_types.get_size(&pile->window, &pile->context, click_collision_.builder_draw_type)).y
  ) + fan::vec2(0, constants::gui_size);

  pile->editor.properties_button.push_back(&pile->window, &pile->context, properties_button_p);

  decltype(pile->editor.properties_button_text)::properties_t properties_text_p;

  auto calculate_text_position = [&]() {
    properties_text_p.position.x = 0;
    properties_text_p.position.x += constants::properties_text_pad;
    properties_text_p.position.x += fan_2d::graphics::gui::text_renderer_t::get_text_size(
      &pile->context,
      properties_text_p.text,
      properties_text_p.font_size
    ).x * 0.5;
    properties_text_p.position.y = properties_button_p.position.y;
  };

  properties_text_p.text = "position";
  properties_text_p.font_size = constants::gui_size;
  properties_text_p.text_color = fan::colors::white;
  calculate_text_position();

  pile->editor.properties_button_text.push_back(&pile->context, properties_text_p);

  properties_button_p.position.y += constants::matrix_multiplier * 50;

  properties_button_p.text = fan::to_wstring(pile->builder.tr.get_font_size(&pile->context, click_collision_.builder_draw_type_index));
  pile->editor.properties_button.push_back(&pile->window, &pile->context, properties_button_p);

  properties_text_p.text = "font size";
  calculate_text_position();

  pile->editor.properties_button_text.push_back(&pile->context, properties_text_p);

  properties_button_p.position.y += constants::matrix_multiplier * 50;

  properties_button_p.text = fan::to_wstring(pile->builder.tr.get_outline_size(&pile->context, click_collision_.builder_draw_type_index));
  pile->editor.properties_button.push_back(&pile->window, &pile->context, properties_button_p);

  properties_text_p.text = "outline size";
  calculate_text_position();

  pile->editor.properties_button_text.push_back(&pile->context, properties_text_p);

  properties_button_p.position.y += constants::matrix_multiplier * 50;

  properties_button_p.text = pile->builder.tr.get_text(&pile->context, click_collision_.builder_draw_type_index);
  pile->editor.properties_button.push_back(&pile->window, &pile->context, properties_button_p);

  properties_text_p.text = "text";
  calculate_text_position();

  pile->editor.properties_button_text.push_back(&pile->context, properties_text_p);

  properties_button_p.position.y += constants::matrix_multiplier * 50;

  fan::color text_color = pile->builder.tr.get_text_color(&pile->context, click_collision_.builder_draw_type_index);
  uint32_t r = text_color.r * 255, g = text_color.g * 255, b = text_color.b * 255, a = text_color.a * 255;
  r = fan::clamp(r, 0u, 255u);
  g = fan::clamp(g, 0u, 255u);
  b = fan::clamp(b, 0u, 255u);
  a = fan::clamp(a, 0u, 255u);
  char hexcol[24];
  snprintf(hexcol, sizeof hexcol, "%02x%02x%02x%02x", r & 0xff, g & 0xff, b & 0xff, a & 0xff);
  properties_button_p.text = hexcol;
  pile->editor.properties_button.push_back(&pile->window, &pile->context, properties_button_p);

  properties_text_p.text = "text color";
  calculate_text_position();

  pile->editor.properties_button_text.push_back(&pile->context, properties_text_p);

  properties_button_p.position.y += constants::matrix_multiplier * 50;

  fan::color outline_color = pile->builder.tr.get_outline_color(&pile->context, click_collision_.builder_draw_type_index);
  r = fan::clamp((uint32_t)(outline_color.r * 255), 0u, 255u);
  g = fan::clamp((uint32_t)(outline_color.g * 255), 0u, 255u);
  b = fan::clamp((uint32_t)(outline_color.b * 255), 0u, 255u);
  a = fan::clamp((uint32_t)(outline_color.a * 255), 0u, 255u);
  snprintf(hexcol, sizeof hexcol, "%02x%02x%02x%02x", r & 0xff, g & 0xff, b & 0xff, a & 0xff);
  properties_button_p.text = hexcol;
  pile->editor.properties_button.push_back(&pile->window, &pile->context, properties_button_p);

  properties_text_p.text = "outline color";
  calculate_text_position();

  pile->editor.properties_button_text.push_back(&pile->context, properties_text_p);

  properties_button_p.position.y += constants::matrix_multiplier * 50;

  properties_text_p.text = " ";
  calculate_text_position();

  properties_button_p.allow_input = false;
  properties_button_p.text = "erase";
  properties_button_p.theme = fan_2d::graphics::gui::themes::deep_red();
  properties_button_p.theme.button.outline_size = 0.001;
  properties_button_p.size /= 1.5;
  properties_button_p.position.x = properties_button_p.size.x + constants::properties_text_pad;
  properties_button_p.text_position = fan_2d::graphics::gui::text_position_e::middle;
  pile->editor.properties_button.push_back(&pile->window, &pile->context, properties_button_p);

  pile->editor.properties_button_text.push_back(&pile->context, properties_text_p);

  pile->editor.properties_button.m_button_event.set_on_input(pile, [](
    fan::window_t* window, fan::opengl::context_t* context, uint32_t i, uint16_t key, fan::key_state key_state, mouse_stage mouse_stage, void* userptr) {
    pile_t* pile = (pile_t*)userptr;

    if (key != fan::mouse_left) {
      return;
    }

    if (key_state != fan::key_state::release) {
      return;
    }
    if (mouse_stage != fan_2d::graphics::gui::mouse_stage::inside) {
      return;
    }

    switch (pile->editor.selected_type) {
      case builder_draw_type_t::text_renderer: {
        switch (i) {
          case 6: {
            switch (pile->editor.selected_type) {
              #include "erase_active.h"
            }
            break;
          }
        }
      }
    }
  });

  pile->editor.properties_button.m_key_event.set_on_focus_loss_callback(pile,
    [](fan::window_t* window, fan::graphics::context_t* context, uint32_t i, void* userptr) {
    pile_t* pile = (pile_t*)userptr;

    // already updated through move release
    if (pile->editor.properties_button.size(window, context) == 0) {
      return;
    }

    switch (pile->editor.selected_type) {
      case builder_draw_type_t::text_renderer: {
        // position, size, etc...
        switch (i) {
          case 0: {
            std::vector<int> values = fan::string_to_values<int>(
              pile->editor.properties_button.get_text(
              window,
              context,
              i
              )
            );

            fan::vec2 position;

            if (values.size() != 2) {
              fan::print("invalid position, usage: x, y");
              position = 0;
            }
            else {
              position = *(fan::vec2i*)values.data();
            }

            pile->builder.tr.set_position(
              context,
              pile->editor.selected_type_index,
              position
            );

            pile->editor.update_resize_rectangles(pile);
            break;
          }
          case 1: {
            f32_t font_size = std::stof(
              pile->editor.properties_button.get_text(
              window,
              context,
              i
              )
            );

            pile->builder.tr.set_font_size(
              context,
              pile->editor.selected_type_index,
              font_size
            );

            pile->editor.update_resize_rectangles(pile);

            break;
          }
          case 2: {
            f32_t outline_size = std::stof(
              pile->editor.properties_button.get_text(
                window,
                context,
                i
              )
            );

            pile->builder.tr.set_outline_size(
              context,
              pile->editor.selected_type_index,
              outline_size
            );

            break;
          }
          case 3: {
            pile->builder.tr.set_text(
              context,
              pile->editor.selected_type_index,
              pile->editor.properties_button.get_text(window, context, i)
            );
            break;
          }
          case 4: {

            pile->builder.tr.set_text_color(
              context,
              pile->editor.selected_type_index,
              fan::color::hex(std::stoul(pile->editor.properties_button.get_text(
              window,
              context,
              i
            ), nullptr, 16))
            );
            break;
          }
          case 5: {

            pile->builder.tr.set_outline_color(
              context,
              pile->editor.selected_type_index,
              fan::color::hex(std::stoul(pile->editor.properties_button.get_text(
                window,
                context,
                i
              ), nullptr, 16))
            );
            break;
          }
        }
        break;
      }
    }
  });

  break;
}