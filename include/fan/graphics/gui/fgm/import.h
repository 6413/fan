#include _FAN_PATH(graphics/gui/gui.h)
#include _FAN_PATH(graphics/gui/be.h)

namespace fan_2d {
  namespace graphics {
    namespace gui {
      namespace fgm {

        struct load_t {

          typedef void(*on_input_cb)(const std::string& id, be_t*, uint32_t index, uint16_t key, fan::key_state key_state, fan_2d::graphics::gui::mouse_stage mouse_stage);
          typedef void(*on_mouse_move_cb)(const std::string& id, be_t*, uint32_t index, mouse_stage mouse_stage);

          void open(fan::window_t* window, fan::opengl::context_t* context_) {

            on_input_function = [](const std::string& id, be_t*, uint32_t index, uint16_t key, fan::key_state key_state, fan_2d::graphics::gui::mouse_stage mouse_stage) {};
            on_mouse_event_function = [](const std::string& id, be_t*, uint32_t index, mouse_stage mouse_stage) {};

            context = context_;
            sprite.open(context);
            tr.open(context);
            button.open(window, context);

            hitbox_ids = new std::vector<std::string>();
            button_ids = new std::vector<std::string>();
            sprite_image_names = new std::vector<std::string>();

            fan::vec2 window_size = window->get_size();

            matrices.open();
            matrices.set_ortho(context, fan::vec2(0, window_size.x), fan::vec2(0, window_size.y));

            sprite.bind_matrices(context, &matrices);
            tr.bind_matrices(context, &matrices);
            button.bind_matrices(context, &matrices);

            window->add_resize_callback(this, [](fan::window_t* window, const fan::vec2i& size, void* userptr) {
              load_t* pile = (load_t*)userptr;

              pile->context->set_viewport(0, size);

              fan::vec2 window_size = window->get_size();
              pile->matrices.set_ortho(pile->context, fan::vec2(0, window_size.x), fan::vec2(0, window_size.y));
            });
            be.open();
          }
          void close(fan::window_t* window, fan::opengl::context_t* context) {
            sprite.close(context);
            tr.close(context);
            be.close();
            button.close(window, context);
            matrices.close();
            delete hitbox_ids;
            delete button_ids;
            delete sprite_image_names;
          }

          void load(fan::window_t* window, fan::opengl::context_t* context, const char* path, fan::opengl::texturepack* tp) {
            fan::io::file::file_t* f;
            fan::io::file::properties_t fp;
            fp.mode = "r+b";
            if (fan::io::file::open(f, filename, fp)) {
              fan::throw_error(std::string("failed to open file:") + filename);
            }

            sprite.write_in_texturepack(context, f, tp, sprite_image_names);
            tr.write_in(context, f);
            button.write_in(context, f);
            be.write_in(f);

            uint32_t count;
            fan::io::file::read(f, &count, sizeof(count), 1);
            hitbox_ids->resize(count);
            for (uint32_t i = 0; i < count; i++) {
              uint32_t s;
              fan::io::file::read(f, &s, sizeof(s), 1);
              (*hitbox_ids)[i].resize(s);
              fan::io::file::read(f, (*hitbox_ids)[i].data(), s, 1);
            }
            fan::io::file::read(f, &count, sizeof(count), 1);
            button_ids->resize(count);
            for (uint32_t i = 0; i < count; i++) {
              uint32_t s;
              fan::io::file::read(f, &s, sizeof(s), 1);
              (*button_ids)[i].resize(s);
              fan::io::file::read(f, (*button_ids)[i].data(), s, 1);
            }

            fclose(f);
            be.set_userptr(this);
            be.set_on_input([](fan_2d::graphics::gui::be_t* be, uint32_t index, uint16_t key, fan::key_state 
              key_state, fan_2d::graphics::gui::mouse_stage mouse_stage) {
              load_t* load = (load_t*)be->get_userptr();
              load->on_input_function((*load->hitbox_ids)[index - 1], be, index, key, key_state, mouse_stage);
            });
            be.set_on_mouse_event([](be_t* be, uint32_t index, mouse_stage mouse_stage) {
              load_t* load = (load_t*)be->get_userptr();
              load->on_mouse_event_function((*load->hitbox_ids)[index - 1], be, index, mouse_stage);
            });
            be.bind_to_window(window);
          }

          void* get_userptr() {
            return userptr;
          }
          void set_userptr(void* userptr_) {
            userptr = userptr_;
          }

          void set_on_input(on_input_cb function) {
            on_input_function = function;
          }

          void set_on_mouse_event(on_mouse_move_cb function) {
            on_mouse_event_function = function;
          }

          void enable_draw(fan::window_t* window, fan::opengl::context_t* context) {
            sprite.enable_draw(context);
            tr.enable_draw(context);
            button.enable_draw(window, context);
          }

          fan_2d::graphics::sprite_t sprite;
          fan_2d::graphics::gui::text_renderer_t tr;

          fan::opengl::matrices_t matrices;
          fan::opengl::context_t* context;
          fan_2d::graphics::gui::be_t be;

          fan_2d::graphics::gui::rectangle_text_button_sized_t button;

          std::vector<std::string>* hitbox_ids;
          std::vector<std::string>* button_ids;
          std::vector<std::string>* sprite_image_names;

          on_input_cb on_input_function;
          on_mouse_move_cb on_mouse_event_function;
          
          void* userptr;
        };
      }
    }
  }
}