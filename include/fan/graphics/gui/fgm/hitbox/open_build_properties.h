case builder_draw_type_t::hitbox: {

  fan::vec2 position = fan::vec2(pile->builder.hitbox.get_position(&pile->context, click_collision_.builder_draw_type_index));
  fan::vec2 size = fan::vec2(pile->builder.hitbox.get_size(&pile->context, click_collision_.builder_draw_type_index));

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


  properties_button_p.position.y += 50;

  properties_button_p.text = "";
  pile->editor.properties_button.push_back(&pile->window, &pile->context, properties_button_p);

  properties_text_p.text = "on click";
  calculate_text_position();

  pile->editor.properties_button_text.push_back(&pile->context, properties_text_p);

  properties_button_p.position.y += 50;

  properties_button_p.text = "";
  pile->editor.properties_button.push_back(&pile->window, &pile->context, properties_button_p);

  properties_text_p.text = "on release";
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
        case builder_draw_type_t::hitbox: {
          switch (i) {
            case 4: {
              #include "erase_active.h"
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

      static auto code_builder = [&](const fan::utf16_string& wpath, const std::string& save_path) {
        std::string path(wpath.begin(), wpath.end());

        if (!fan::io::file::exists(path)) {
          fan::print_warning("path does not exist:" + path);
        }

        std::string data;
        fan::io::file::read(save_path, &data);
        // selected type and bll? + 1 for bll nodes xd
        std::string index = std::to_string(pile->editor.selected_type_index + 1);
        std::string case_line = "case " + index + ":" + " {";

        std::size_t begin = data.find(case_line);

        path.insert(0, "\n#include <");
        path.insert(path.size(), ">\n");
        path.insert(0, case_line);
        path.insert(path.size(), "}");
        path.insert(path.size(), "\n");

        if (begin != std::string::npos) {
          std::size_t end = data.find_first_of("}", begin);
          if (end == std::string::npos) {
            fan::throw_error("corrupted:" + save_path);
          }
          end += 2;
          data.erase(begin, end - begin);
          data.insert(begin, path);

        }
        // if case does not exist, create new
        else {
          std::size_t it = data.find_last_of("}");
          if (it == std::string::npos) {
            fan::throw_error("corrupted:" + save_path);
          }
          it -= 1;
          data.insert(it, path);
        }

        fan::io::file::write(save_path, data, std::ios_base::binary);
      };

      switch (pile->editor.selected_type) {
        case builder_draw_type_t::hitbox: {
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

              pile->builder.hitbox.set_position(
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

              pile->builder.hitbox.set_size(
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
              code_builder(wpath, std::string(STRINGIFY_DEFINE(FAN_INCLUDE_PATH)) + "/fan/" + "gui_maker/on_click_cb");

              break;
            }
            case 3: {
              fan::utf16_string wpath = pile->editor.properties_button.get_text(
                window,
                context,
                i
              );

              code_builder(wpath, std::string(STRINGIFY_DEFINE(FAN_INCLUDE_PATH)) + "/fan/" + "gui_maker/on_release_cb");

              break;
            }
          }
          break;
        }
      }
    });

  break;
}