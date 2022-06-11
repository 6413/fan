#pragma once

#include _FAN_PATH(graphics/gui/gui.h)
#include _FAN_PATH(graphics/gui/be.h)
#include _FAN_PATH(tp/tp.h)

namespace fan_2d {
  namespace graphics {
    namespace gui {
      namespace fgm {

        struct pile_t;

        struct editor_t {

          fan::opengl::matrices_t gui_matrices;
          fan::opengl::matrices_t gui_properties_matrices;
          fan::graphics::viewport_t builder_viewport;
          fan::graphics::viewport_t properties_viewport;

          fan::vec2 properties_camera;

          struct constants {
            static constexpr fan::vec2 window_size = fan::vec2(900, 700);
            static constexpr fan::vec2 builder_viewport_size = fan::vec2(600, window_size.y);
            static constexpr f32_t gui_size = 16;
            static constexpr f32_t properties_text_pad = 10;
            static constexpr f32_t scroll_speed = 10;

            static constexpr f32_t properties_box_pad = 180;

            static constexpr f32_t resize_rectangle_size = 5;
          };

          struct flags_t {
            static constexpr uint32_t moving = (1 << 0);
            static constexpr uint32_t resizing = (1 << 1);
            static constexpr uint32_t ignore_properties_close = (1 << 3);
            static constexpr uint32_t ignore_moving = (1 << 4);
          };

          struct builder_draw_type_t {
            static constexpr uint32_t sprite = 0;
            static constexpr uint32_t text_renderer = 1;
            static constexpr uint32_t hitbox = 2;
            static constexpr uint32_t button = 3;
          };

          struct click_collision_t {
            uint32_t builder_draw_type;
            uint32_t builder_draw_type_index;
          };

          struct userptr_t {
            uint32_t depth_nodereference;
            uint32_t id;
          };

          bool is_inside_builder_viewport(pile_t* pile, const fan::vec2& position);
          bool is_inside_types_viewport(pile_t* pile, const fan::vec2& position);
          bool is_inside_properties_viewport(pile_t* pile, const fan::vec2& position);

          bool click_collision(pile_t* pile, click_collision_t* click_collision_);

          void open_build_properties(pile_t* pile, click_collision_t click_collision_);
          void close_build_properties(pile_t* pile);

          void open(pile_t* pile);

          void update_resize_rectangles(pile_t* pile);

          void depth_map_push(pile_t* pile, uint32_t type, uint32_t index);
          void editor_erase_active(pile_t* pile);
          void print(pile_t* pile, const std::string& message);

          bool check_for_colliding_hitbox_ids(const std::string& id);
          bool check_for_colliding_button_ids(const std::string& id);

          fan::vec2 builder_viewport_size;
          fan::vec2 origin_shapes;
          fan::vec2 origin_properties;

          uint32_t builder_draw_type;
          uint32_t builder_draw_type_index;

          uint32_t selected_type;
          uint32_t selected_type_index;

          uint32_t copied_type;
          uint32_t copied_type_index;

          fan::vec2 click_position;
          fan::vec2 resize_position;
          fan::vec2 resize_size;

          fan::vec2 move_offset;

          uint8_t flags;

          struct depth_t {
            uint32_t depth;
            uint32_t type;
            uint32_t index;
          };

          fan::hector_t<depth_t> depth_map;
          std::vector<std::string> hitbox_ids;
          std::vector<std::string> button_ids;
          std::vector<std::string> sprite_image_names;

          fan_2d::graphics::line_t outline;
          fan_2d::graphics::gui::rectangle_text_button_sized_t builder_types;

          fan_2d::graphics::gui::text_renderer_t builder_text;

          fan_2d::graphics::gui::text_renderer_t properties_button_text;
          fan_2d::graphics::gui::rectangle_text_button_sized_t properties_button;

          fan_2d::graphics::gui::rectangle_button_sized_t resize_rectangles;

          uint32_t depth_index;

          uint8_t resize_stage;
        };

        struct builder_t {

          void open(pile_t* pile);

          fan_2d::graphics::sprite_t sprite;
          fan_2d::graphics::gui::text_renderer_t tr;
          fan_2d::graphics::sprite_t hitbox;
          fan_2d::graphics::gui::rectangle_text_button_sized_t button;
        };

        struct pile_t {

          fan::window_t window;
          fan::opengl::context_t context;

          editor_t editor;
          builder_t builder;

          fan::tp::texture_packe0 tp;

          std::string tp_project_name;
          std::string compiled_tp_name;

          void open(int argc, char** argv) {
            window.open(editor_t::constants::window_size);
            context.init();
            context.bind_to_window(&window);
            context.set_viewport(0, window.get_size());
            fan::tp::texture_packe0::open_properties_t op;
            tp.open(op);
            tp_project_name = argv[1];
            tp.load(tp_project_name.data());
            tp.process();
            compiled_tp_name = argv[2];

            window.add_resize_callback(this, [](fan::window_t*, const fan::vec2i& size, void* userptr) {
              pile_t* pile = (pile_t*)userptr;

              pile->context.set_viewport(0, size);

              fan::vec2 window_size = pile->window.get_size();
              pile->editor.gui_matrices.set_ortho(&pile->context, fan::vec2(0, window_size.x), fan::vec2(0, window_size.y));
              pile->editor.gui_properties_matrices.set_ortho(
                &pile->context, 
                fan::vec2(
                  0, 
                  window_size.x - pile->editor.origin_properties.x
                ), 
                fan::vec2(
                0, 
                window_size.y - pile->editor.origin_properties.y)
              );

              fan::graphics::viewport_t::properties_t vp;

              vp.size = fan::vec2(window_size.x, window_size.y) - pile->editor.origin_properties; 
              vp.position = fan::vec2(pile->editor.origin_properties.x, 0);

              pile->editor.properties_viewport.set(&pile->context, vp);
              vp.position = 0;
              vp.size = window_size;
              pile->editor.builder_viewport.set(&pile->context, vp);
            });

            editor.gui_matrices.open();
            editor.gui_properties_matrices.open();

            builder.open(this);
            editor.open(this);

            fan::vec2 window_size = window.get_size();
            editor.gui_matrices.set_ortho(&context, fan::vec2(0, window_size.x), fan::vec2(0, window_size.y));
            editor.gui_properties_matrices.set_ortho(&context, fan::vec2(0, window_size.x - editor.origin_properties.x), fan::vec2(0, window_size.y - editor.origin_properties.y));
          }

          void save(const char* filename) {
            fan::io::file::file_t* f;
            fan::io::file::properties_t fp;
            fp.mode = "w+b";
            if (fan::io::file::open(&f, filename, fp)) {
              fan::throw_error(std::string("failed to open file:") + filename);
            }

            builder.sprite.write_out_texturepack(&context, f);
            for (uint32_t i = 0; i < editor.sprite_image_names.size(); i++) {
              uint32_t count = editor.sprite_image_names[i].size();
              fan::io::file::write(f, &count, sizeof(count), 1);
              fan::io::file::write(f, editor.sprite_image_names[i].data(), count, 1);
            }

            builder.tr.write_out(&context, f);
            builder.button.write_out(&context, f);
            fan_2d::graphics::gui::be_t be;
            be.open();
            fan_2d::graphics::gui::be_t::properties_t p;
            for (uint32_t i = 0; i < builder.hitbox.size(&context); i++) {
              // FIX LATER ALLOW CIRCLE TOO
              p.hitbox_type = fan_2d::graphics::gui::be_t::hitbox_type_t::rectangle;
              p.hitbox_rectangle.position = builder.hitbox.get_position(&context, i);
              p.hitbox_rectangle.size = builder.hitbox.get_size(&context, i);
              be.push_back(p);
            }

            be.write_out(f);
            uint32_t count = editor.hitbox_ids.size();
            fan::io::file::write(f, &count, sizeof(count), 1);
            for (uint32_t i = 0; i < count; i++) {
              uint32_t s = editor.hitbox_ids[i].size();
              fan::io::file::write(f, &s, sizeof(s), 1);
              fan::io::file::write(f, editor.hitbox_ids[i].data(), editor.hitbox_ids[i].size(), 1);
            }
            count = editor.button_ids.size();
            fan::io::file::write(f, &count, sizeof(count), 1);
            for (uint32_t i = 0; i < count; i++) {
              uint32_t s = editor.button_ids[i].size();
              fan::io::file::write(f, &s, sizeof(s), 1);
              fan::io::file::write(f, editor.button_ids[i].data(), editor.button_ids[i].size(), 1);
            }
            editor.depth_map.write_out(f);

            fan::io::file::close(f);

            tp.save(tp_project_name.data());
            tp.save_compiled(compiled_tp_name.data());
          }

          void load_file(const char* filename) {
            fan::io::file::file_t* f;
            fan::io::file::properties_t fp;
            fp.mode = "r+b";
            if (fan::io::file::open(&f, filename, fp)) {
              fan::throw_error(std::string("failed to open file:") + filename);
            }

            builder.sprite.write_in_texturepack(&context, f, &tp, &editor.sprite_image_names);
            builder.tr.write_in(&context, f);
            builder.button.write_in(&context, f);
            fan_2d::graphics::gui::be_t be;
            be.open();
            be.write_in(f);

            uint32_t count;
            fan::io::file::read(f, &count, sizeof(count), 1);
            editor.hitbox_ids.resize(count);
            for (uint32_t i = 0; i < count; i++) {
              uint32_t s;
              fan::io::file::read(f, &s, sizeof(s), 1);
              editor.hitbox_ids[i].resize(s);
              fan::io::file::read(f, editor.hitbox_ids[i].data(), s, 1);
            }

            for (uint32_t i = 0; i < count; i++) {
              fan_2d::graphics::sprite_t::properties_t sprite_p;
              fan::color colors[9];
              colors[0] = fan::colors::gray - fan::color(0, 0, 0, 0.1);
              colors[1] = fan::colors::gray - fan::color(0, 0, 0, 0.1);
              colors[2] = fan::colors::gray - fan::color(0, 0, 0, 0.1);
              colors[3] = fan::colors::gray - fan::color(0, 0, 0, 0.1);
              colors[4] = fan::colors::transparent;
              colors[5] = fan::colors::gray - fan::color(0, 0, 0, 0.1);
              colors[6] = fan::colors::gray - fan::color(0, 0, 0, 0.1);
              colors[7] = fan::colors::gray - fan::color(0, 0, 0, 0.1);
              colors[8] = fan::colors::gray - fan::color(0, 0, 0, 0.1);
              sprite_p.image.load(&context, colors, fan::vec2ui(3, 3));
              switch (be.m_button_data[i + 1].properties.hitbox_type) {
                case fan_2d::graphics::gui::be_t::hitbox_type_t::rectangle: {
                  // BLL + 1
                  sprite_p.position = be.m_button_data[i + 1].properties.hitbox_rectangle.position;
                  sprite_p.size = be.m_button_data[i + 1].properties.hitbox_rectangle.size;
                  
                  break;
                }
                case fan_2d::graphics::gui::be_t::hitbox_type_t::circle: {
                  break;
                }
              }
              builder.hitbox.push_back(&context, sprite_p);
            }

            fan::io::file::read(f, &count, sizeof(count), 1);
            editor.button_ids.resize(count);

            for (uint32_t i = 0; i < count; i++) {
              uint32_t s;
              fan::io::file::read(f, &s, sizeof(s), 1);
              editor.button_ids[i].resize(s);
              fan::io::file::read(f, editor.button_ids[i].data(), s, 1);
            }

            editor.depth_map.write_in(f);

            fan::io::file::close(f);
          }

        };

        #include _FAN_PATH(graphics/gui/fgm/editor/editor.h)
        #include _FAN_PATH(graphics/gui/fgm/builder.h)

      }
    }
  }
}