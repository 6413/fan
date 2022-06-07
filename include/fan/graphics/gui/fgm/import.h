#include _FAN_PATH(graphics/gui/gui.h)
#include _FAN_PATH(graphics/gui/be.h)

namespace fan_2d {
  namespace graphics {
    namespace gui {
      namespace fgm {

        struct load_t {
          void open(fan::window_t* window, fan::opengl::context_t* context_) {
            context = context_;
            sprite.open(context);

            fan::vec2 window_size = window->get_size();

            matrices.open();
            matrices.set_ortho(context, fan::vec2(0, window_size.x), fan::vec2(0, window_size.y));

            sprite.bind_matrices(context, &matrices);
            tr.bind_matrices(context, &matrices);

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
            matrices.close();
          }

          void load(fan::window_t* window, fan::opengl::context_t* context, const char* path, fan::opengl::texturepack* tp) {
            FILE* f = fopen(path, "r+b");
            if (!f) {
              fan::throw_error("failed to open file stream");
            }

            sprite.write_in_texturepack(context, f, tp, 0);
            be.write_in(context, f);
            fclose(f);
            be.set_on_input([](fan_2d::graphics::gui::be_t*, uint32_t index, uint16_t key, fan::key_state key_state, fan_2d::graphics::gui::mouse_stage mouse_stage) {
              if (key != fan::mouse_left) {
                return;
              }
              switch (key_state) {
                case fan::key_state::press: {
                  #include <gui_maker/on_click_cb>
                  break;
                }
                case fan::key_state::release: {
                  #include <gui_maker/on_release_cb>
                  break;
                }
              }
            });
            be.bind_to_window(window);
          }

          void enable_draw(fan::window_t* window, fan::opengl::context_t* context) {
            sprite.enable_draw(context);
          }

          fan_2d::graphics::sprite_t sprite;
          fan_2d::graphics::gui::text_renderer_t tr;

          fan::opengl::matrices_t matrices;
          fan::opengl::context_t* context;
          fan_2d::graphics::gui::be_t be;
        };
      }
    }
  }
}