case builder_draw_type_t::sprite: {

  fan::vec2 position = fan::vec2(pile->builder.sprite.get_position(&pile->context, click_collision_.builder_draw_type_index));
  fan::vec2 size = fan::vec2(pile->builder.sprite.get_size(&pile->context, click_collision_.builder_draw_type_index));

  decltype(pile->editor.properties_button)::properties_t properties_button_p;
  properties_button_p.text = fan::to_wstring(position.x, 0) + L", " + fan::to_wstring(position.y, 0);
  properties_button_p.size = fan::vec2(constants::gui_size * 5, constants::gui_size);
  properties_button_p.font_size = constants::gui_size;
  properties_button_p.theme = fan_2d::graphics::gui::themes::gray();
  properties_button_p.text_position = fan_2d::graphics::gui::text_position_e::left;
  properties_button_p.allow_input = true;
  properties_button_p.position = fan::vec2(
    constants::properties_box_pad,
    fan::vec2(pile->editor.builder_types.get_size(&pile->window, &pile->context, click_collision_.builder_draw_type)).y
  ) + 10;

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

  properties_button_p.position.y += 50;

  properties_button_p.text = fan::to_wstring(size.x, 0) + L", " + fan::to_wstring(size.y, 0);
  pile->editor.properties_button.push_back(&pile->window, &pile->context, properties_button_p);

  properties_text_p.text = "size";
  calculate_text_position();

  pile->editor.properties_button_text.push_back(&pile->context, properties_text_p);
  
 /* properties_button_p.position.y += 50;

  properties_button_p.userptr = pile->builder.rtbs.get_userptr(&pile->window, &pile->context, click_collision_.builder_draw_type_index);
  properties_button_p.text = std::to_string(*(uint32_t*)&properties_button_p.userptr);
  pile->editor.properties_button.push_back(&pile->window, &pile->context, properties_button_p);

  properties_text_p.text = "id";
  calculate_text_position();

  pile->editor.properties_button_text.push_back(&pile->context, properties_text_p);*/

  properties_button_p.position.y += 50;

  properties_button_p.text = pile->editor.sprite_image_names[pile->editor.selected_type_index];
  calculate_text_position();

  pile->editor.properties_button.push_back(&pile->window, &pile->context, properties_button_p);

  properties_text_p.text = "image name:";
  calculate_text_position();

  pile->editor.properties_button_text.push_back(&pile->context, properties_text_p);

  properties_button_p.position.y += 50;

  properties_button_p.allow_input = false;
  properties_button_p.text = "erase";
  properties_button_p.theme = fan_2d::graphics::gui::themes::deep_red();
  properties_button_p.size.x = 30;
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
      case builder_draw_type_t::sprite: {
        switch (i) {
          case 3: {
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
      case builder_draw_type_t::sprite: {
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

            pile->builder.sprite.set_position(
              context,
              pile->editor.selected_type_index,
              position
            );

            pile->editor.update_resize_rectangles(pile);

            break;
          }
          case 1: {
            std::vector<int> values = fan::string_to_values<int>(
              pile->editor.properties_button.get_text(
              window,
              context,
              i
              )
            );

            fan::vec2 size;

            if (values.size() != 2) {
              fan::print("invalid size, usage: x, y");
              size = 0;
            }
            else {
              size = *(fan::vec2i*)values.data();
            }

            pile->builder.sprite.set_size(
              context,
              pile->editor.selected_type_index,
              size
            );

            pile->editor.update_resize_rectangles(pile);
            break;
          }
          case 2: {
            fan::utf16_string wpath = pile->editor.properties_button.get_text(
              window,
              context,
              i
            );

            std::string path(wpath.begin(), wpath.end());

            fan::opengl::image_t image;
            fan::tp::ti_t ti;
            fan::tp::texture_packe0::texture_properties_t tp;
            tp.name = path;

            if (path == "\\null") {
              image.create_missing_texture(&pile->context);
              pile->builder.sprite.reload_sprite(
                context,
                pile->editor.selected_type_index,
                image
              );
              pile->builder.sprite.set_texture_coordinates(
                context,
                pile->editor.selected_type_index,
                image.calculate_aspect_ratio(
                  pile->builder.sprite.get_size(context, pile->editor.selected_type_index),
                  3
                )
              );
              pile->editor.sprite_image_names[pile->editor.selected_type_index] = "";
            }
            else if (!pile->tp.qti(path, &ti)) {
              pile->editor.properties_button.set_text(window, context, 2, path);
              pile->builder.sprite.load_texturepack(context, pile->editor.selected_type_index, &pile->tp, &ti);
              pile->editor.sprite_image_names[pile->editor.selected_type_index] = path;
            }
            else {
              if (pile->tp.push_texture(path, tp)) {
                pile->editor.print(pile, std::string("failed to load image:") + path);
              }
              pile->tp.process();
              pile->tp.qti(path, &ti);
              pile->editor.properties_button.set_text(window, context, 2, path);
              pile->builder.sprite.load_texturepack(context, pile->editor.selected_type_index, &pile->tp, &ti);
              pile->editor.sprite_image_names[pile->editor.selected_type_index] = path;
            }
            break;
          }
        }
        break;
      }
    }
  });

  break;
}