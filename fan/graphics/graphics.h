#pragma once
// extension to loco.h

#include <fan/graphics/loco.h>

// user friendly functions
/***************************************/

namespace fan {
  namespace graphics {

    bool is_mouse_clicked(int button = fan::mouse_left);
    bool is_mouse_down(int button = fan::mouse_left);
    bool is_mouse_released(int button = fan::mouse_left);
    fan::vec2 get_mouse_drag(int button = fan::mouse_left);

    void set_window_size(const fan::vec2& size);

#if defined(loco_imgui)
    using imgui_element_t = loco_t::imgui_element_t;

    void text(const std::string& text, const fan::vec2& position = 0, const fan::color& color = fan::colors::white);
    void text_bottom_right(const std::string& text, const fan::color& color = fan::colors::white, const fan::vec2& offset = 0);
#endif

    struct light_properties_t {
      camera_impl_t* camera = &gloco->orthographic_camera;
      fan::vec3 position = fan::vec3(0, 0, 0);
      f32_t parallax_factor = 0;
      fan::vec2 size = fan::vec2(0.1, 0.1);
      fan::vec2 rotation_point = fan::vec2(0, 0);
      fan::color color = fan::color(1, 1, 1, 1);
      fan::vec3 rotation_vector = fan::vec3(0, 0, 1);
      uint32_t flags = 0;
      fan::vec3 angle = fan::vec3(0, 0, 0);
    };

    struct light_t : loco_t::shape_t {
      light_t(light_properties_t p = light_properties_t()) {
        *(loco_t::shape_t*)this = loco_t::shape_t(
          fan_init_struct(
            typename loco_t::light_t::properties_t,
            .camera = p.camera->camera,
            .viewport = p.camera->viewport,
            .position = p.position,
            .parallax_factor = p.parallax_factor,
            .size = p.size,
            .rotation_point = p.rotation_point,
            .color = p.color,
            .rotation_vector = p.rotation_vector,
            .flags = p.flags,
            .angle = p.angle
          ));
      }
    };

    #if defined(loco_line)

      struct line_properties_t {
        camera_impl_t* camera = &gloco->orthographic_camera;
        fan::vec3 src = fan::vec3(0, 0, 0);
        fan::vec2 dst = fan::vec2(1, 1);
        fan::color color = fan::color(1, 1, 1, 1);
        bool blending = false;
      };

      struct line_t : loco_t::shape_t {
        line_t(line_properties_t p = line_properties_t()) {
          *(loco_t::shape_t*)this = loco_t::shape_t(
            fan_init_struct(
              typename loco_t::line_t::properties_t,
              .camera = p.camera->camera,
              .viewport = p.camera->viewport,
              .src = p.src,
              .dst = p.dst,
              .color = p.color,
              .blending = p.blending
            ));
        }
      };
    #endif

//#if defined(loco_rectangle)
    struct rectangle_properties_t {
      camera_impl_t* camera = &gloco->orthographic_camera;
      fan::vec3 position = fan::vec3(0, 0, 0);
      fan::vec2 size = fan::vec2(32, 32);
      fan::color color = fan::color(1, 1, 1, 1);
      fan::color outline_color = color;
      fan::vec3 angle = 0;
      fan::vec2 rotation_point = 0;
      bool blending = false;
    };

    // make sure you dont do position = vec2
    struct rectangle_t : loco_t::shape_t {
      rectangle_t(rectangle_properties_t p = rectangle_properties_t()) {
        *(loco_t::shape_t*)this = loco_t::shape_t(
          fan_init_struct(
            typename loco_t::rectangle_t::properties_t,
            .camera = p.camera->camera,
            .viewport = p.camera->viewport,
            .position = p.position,
            .size = p.size,
            .color = p.color,
            .outline_color = p.outline_color,
            .angle = p.angle,
            .rotation_point = p.rotation_point,
            .blending = p.blending
          )
        );
      }
    };

    struct sprite_properties_t {
      camera_impl_t* camera = &gloco->orthographic_camera;
      fan::vec3 position = fan::vec3(0, 0, 0);
      fan::vec2 size = fan::vec2(32, 32);
      fan::vec3 angle = 0;
      fan::color color = fan::color(1, 1, 1, 1);
      fan::vec2 rotation_point = 0;
      loco_t::image_t image = gloco->default_texture;
      std::array<loco_t::image_t, 30> images;
      f32_t parallax_factor = 0;
      bool blending = false;
      uint32_t flags = 0;
    };


    struct sprite_t : loco_t::shape_t {
      sprite_t(sprite_properties_t p = sprite_properties_t()) {
        *(loco_t::shape_t*)this = loco_t::shape_t(
          fan_init_struct(
            typename loco_t::sprite_t::properties_t,
            .camera = p.camera->camera,
            .viewport = p.camera->viewport,
            .parallax_factor = p.parallax_factor,
            .position = p.position,
            .size = p.size,
            .angle = p.angle,
            .image = p.image,
            .images = p.images,
            .color = p.color,
            .rotation_point = p.rotation_point,
            .blending = p.blending,
            .flags = p.flags
          ));
      }
    };

    struct unlit_sprite_properties_t {
      camera_impl_t* camera = &gloco->orthographic_camera;
      fan::vec3 position = fan::vec3(0, 0, 0);
      fan::vec2 size = fan::vec2(0.1, 0.1);
      fan::vec3 angle = 0;
      fan::color color = fan::color(1, 1, 1, 1);
      fan::vec2 rotation_point = 0;
      loco_t::image_t image = gloco->default_texture;
      std::array<loco_t::image_t, 30> images;
      fan::vec2 tc_position = 0;
      fan::vec2 tc_size = 1;
      bool blending = false;
    };

    struct unlit_sprite_t : loco_t::shape_t {
      unlit_sprite_t(unlit_sprite_properties_t p = unlit_sprite_properties_t()) {
        *(loco_t::shape_t*)this = loco_t::shape_t(
          fan_init_struct(
            typename loco_t::unlit_sprite_t::properties_t,
            .camera = p.camera->camera,
            .viewport = p.camera->viewport,
            .position = p.position,
            .size = p.size,
            .angle = p.angle,
            .image = p.image,
            .images = p.images,
            .color = p.color,
            .tc_position = p.tc_position,
            .tc_size = p.tc_size,
            .rotation_point = p.rotation_point,
            .blending = p.blending
          ));
      }
    };
#if defined(loco_circle)
    struct circle_properties_t {
      camera_impl_t* camera = &gloco->orthographic_camera;
      fan::vec3 position = fan::vec3(0, 0, 0);
      f32_t radius = 32.f;
      fan::color color = fan::color(1, 1, 1, 1);
      bool blending = false;
      uint32_t flags = 0;
    };

    struct circle_t : loco_t::shape_t {
      circle_t(circle_properties_t p = circle_properties_t()) {
        *(loco_t::shape_t*)this = loco_t::shape_t(
          fan_init_struct(
            typename loco_t::circle_t::properties_t,
            .camera = p.camera->camera,
            .viewport = p.camera->viewport,
            .position = p.position,
            .radius = p.radius,
            .color = p.color,
            .blending = p.blending,
            .flags = p.flags
          ));
      }
    };
#endif

    struct capsule_properties_t {
      camera_impl_t* camera = &gloco->orthographic_camera;
      fan::vec3 position = fan::vec3(0, 0, 0);
      fan::vec2 center0 = 0;
      fan::vec2 center1{0, 128.f};
      f32_t radius = 64.0f;
      fan::color color = fan::color(1, 1, 1, 1);
      fan::color outline_color = color;
      bool blending = true;
      uint32_t flags = 0;
    };

    struct capsule_t : loco_t::shape_t {
      capsule_t(capsule_properties_t p = capsule_properties_t()) {
        *(loco_t::shape_t*)this = loco_t::shape_t(
          fan_init_struct(
            typename loco_t::capsule_t::properties_t,
            .camera = p.camera->camera,
            .viewport = p.camera->viewport,
            .position = p.position,
            .center0 = p.center0,
            .center1 = p.center1,
            .radius = p.radius,
            .color = p.color,
            .outline_color = p.outline_color,
            .blending = p.blending,
            .flags = p.flags
          ));
      }
    };

    using vertex_t = loco_t::vertex_t;
    struct polygon_properties_t {
      camera_impl_t* camera = &gloco->orthographic_camera;
      fan::vec3 position = 0;
      std::vector<vertex_t> vertices;
      fan::vec3 angle = 0;
      fan::vec2 rotation_point = 0;
      bool blending = true;
    };

    struct polygon_t : loco_t::shape_t {
      polygon_t() = default;
      polygon_t(polygon_properties_t p) {
        *(loco_t::shape_t*)this = loco_t::shape_t(
          fan_init_struct(
            typename loco_t::polygon_t::properties_t,
            .camera = p.camera->camera,
            .viewport = p.camera->viewport,
            .position = p.position,
            .vertices = p.vertices,
            .angle = p.angle,
            .rotation_point = p.rotation_point,
            .blending = p.blending,
          ));
      }
    };

    struct grid_properties_t {
      camera_impl_t* camera = &gloco->orthographic_camera;
      fan::vec3 position = fan::vec3(0, 0, 0);
      fan::vec2 size = fan::vec2(0.1, 0.1);
      fan::vec2 grid_size = fan::vec2(1, 1);
      fan::vec2 rotation_point = fan::vec2(0, 0);
      fan::color color = fan::color(1, 1, 1, 1);
      fan::vec3 angle = fan::vec3(0, 0, 0);
    };
    struct grid_t : loco_t::shape_t {
      grid_t(grid_properties_t p = grid_properties_t()) {
        *(loco_t::shape_t*)this = loco_t::shape_t(
          fan_init_struct(
            typename loco_t::grid_t::properties_t,
            .camera = p.camera->camera,
            .viewport = p.camera->viewport,
            .position = p.position,
            .size = p.size,
            .grid_size = p.grid_size,
            .rotation_point = p.rotation_point,
            .color = p.color,
            .angle = p.angle
          ));
      }
    };

    struct line3d_properties_t {
      camera_impl_t* camera = &gloco->perspective_camera;
      fan::vec3 src = fan::vec3(0, 0, 0);
      fan::vec3 dst = fan::vec3(10, 10, 10);
      fan::color color = fan::color(1, 1, 1, 1);
      bool blending = false;
    };

    struct line3d_t : loco_t::shape_t {
      line3d_t(line3d_properties_t p = line3d_properties_t()) {
        *(loco_t::shape_t*)this = loco_t::shape_t(
          fan_init_struct(
            typename loco_t::line3d_t::properties_t,
            .camera = p.camera->camera,
            .viewport = p.camera->viewport,
            .src = p.src,
            .dst = p.dst,
            .color = p.color,
            .blending = p.blending
          ));
      }
    };

    loco_t::polygon_t::properties_t create_hexagon(f32_t radius, const fan::color& color);

#if defined(loco_vfi)

    // for line
    fan::line3 get_highlight_positions(const fan::vec3& op_, const fan::vec2& os, int index);

    // REQUIRES to be allocated by new since lambda captures this
    // also container that it's stored in, must not change pointers
    template <typename T>
    struct vfi_root_custom_t {
      void create_highlight() {
        fan::vec3 op = children[0].get_position();
        fan::vec2 os = children[0].get_size();
        loco_t::camera_t c = children[0].get_camera();
        loco_t::viewport_t v = children[0].get_viewport();
        fan::graphics::camera_t cam;
        cam.camera = c;
        cam.viewport = v;
        for (std::size_t j = 0; j < highlight.size(); ++j) {
          for (std::size_t i = 0; i < highlight[0].size(); ++i) {
            fan::line3 line = get_highlight_positions(op, os, i);
            highlight[j][i] = fan::graphics::line_t{ {
              .camera = &cam,
              .src = line[0],
              .dst = line[1],
              .color = fan::color(1, 0.5, 0, 1)
            } };
          }
        }
      }
      void disable_highlight() {
        fan::vec3 op = children[0].get_position();
        fan::vec2 os = children[0].get_size();
        for (std::size_t j = 0; j < highlight.size(); ++j) {
          for (std::size_t i = 0; i < highlight[j].size(); ++i) {
            fan::line3 line = get_highlight_positions(op, os, i);
            if (highlight[j][i].iic() == false) {
              highlight[j][i].set_line(0, 0);
            }
          }
        }
      }

      void set_root(const loco_t::vfi_t::properties_t& p) {
        fan::graphics::vfi_t::properties_t in = p;
        in.shape_type = loco_t::vfi_t::shape_t::rectangle;
        in.shape.rectangle->viewport = p.shape.rectangle->viewport;
        in.shape.rectangle->camera = p.shape.rectangle->camera;
        in.keyboard_cb = [this, user_cb = p.keyboard_cb](const auto& d) -> int {
          if (d.key == fan::key_c &&
            (d.keyboard_state == fan::keyboard_state::press ||
              d.keyboard_state == fan::keyboard_state::repeat)) {
            this->resize = true;
            return user_cb(d);
          }
          this->resize = false;
          return 0;
          };
        in.mouse_button_cb = [this, user_cb = p.mouse_button_cb](const auto& d) -> int {
          if (g_ignore_mouse) {
            return 0;
          }

          if (d.button != fan::mouse_left) {
            return 0;
          }
          if (d.button_state != fan::mouse_state::press) {
            this->move = false;
            moving_object = false;
            d.flag->ignore_move_focus_check = false;
              if (previous_click_position == d.position) {
                for (auto it = selected_objects.begin(); it != selected_objects.end(); ) {
                    (*it)->disable_highlight();
                    if (*it != this) {
                      it = selected_objects.erase(it);
                    } else {
                      ++it;
                    }
                  }
              }
            return 0;
          }
          if (d.mouse_stage != loco_t::vfi_t::mouse_stage_e::viewport_inside) {
            return 0;
          }

          if (previous_focus && previous_focus != this) {
            for (std::size_t i = 0; i < previous_focus->highlight[0].size(); ++i) {
              if (previous_focus->highlight[0][i].iic() == false) {
                previous_focus->highlight[0][i].set_line(0, 0);
              }
            }
          }
          //selected_objects.clear();
          if (std::find(selected_objects.begin(), selected_objects.end(), this) == selected_objects.end()) {
            selected_objects.push_back(this);
          }
          //selected_objects.push_back(this);
          create_highlight();
          previous_focus = this;

          if (move_and_resize_auto) {
            previous_click_position = d.position;
            d.flag->ignore_move_focus_check = true;
            this->move = true;
            moving_object = true;
            this->click_offset = get_position() - d.position;
            
            gloco->vfi.set_focus_keyboard(d.vfi->focus.mouse);
          }
          return user_cb(d);
          };
        in.mouse_move_cb = [this, user_cb = p.mouse_move_cb](const auto& d) -> int {
          if (g_ignore_mouse) {
            return 0;
          }

          if (move_and_resize_auto) {
            if (this->resize && this->move) {
              fan::vec2 new_size = (d.position - get_position());
              static constexpr fan::vec2 min_size(10, 10);
              this->set_size(new_size.x);
              fan::vec3 op = children[0].get_position();
              fan::vec2 os = children[0].get_size();
              for (std::size_t j = 0; j < highlight.size(); ++j) {
                for (std::size_t i = 0; i < highlight[j].size(); ++i) {
                  fan::line3 line = get_highlight_positions(op, os, i);
                  if (highlight[j][i].iic() == false) {
                    highlight[j][i].set_line(line[0], line[1]);
                  }
                }
              }
              if (previous_focus && previous_focus != this) {
                for (std::size_t i = 0; i < previous_focus->highlight[0].size(); ++i) {
                  if (previous_focus->highlight[0][i].iic() == false) {
                    previous_focus->highlight[0][i].set_line(0, 0);
                  }
                }
                previous_focus = this;
              }
              return user_cb(d);
            }
            else if (this->move) {
              fan::vec3 p = get_position();
              p = fan::vec3(d.position + click_offset, p.z);
              p.x = std::round(p.x / 32.0f) * 32.0f;
              p.y = std::round(p.y / 32.0f) * 32.0f;
              this->set_position(p);
              return user_cb(d);
            }
          }
          else {
            return user_cb(d);
          }
          return 0;
          };
        vfi_root = in;
      }
      void push_child(const loco_t::shape_t& shape) {
        children.push_back({ shape });
      }
      fan::vec3 get_position() {
        return vfi_root.get_position();
      }

      static void update_highlight_position(vfi_root_custom_t<T>* instance) {
        fan::vec3 op = instance->children[0].get_position();
        fan::vec2 os = instance->children[0].get_size();
        for (std::size_t j = 0; j < instance->highlight.size(); ++j) {
          for (std::size_t i = 0; i < instance->highlight[j].size(); ++i) {
            fan::line3 line = get_highlight_positions(op, os, i);
            if (instance->highlight[j][i].iic() == false) {
              instance->highlight[j][i].set_line(line[0], line[1]);
            }
          }
        }
      }

      void set_position(const fan::vec3& position) {
        fan::vec2 root_pos = vfi_root.get_position();
        fan::vec2 offset = position - root_pos;
        vfi_root.set_position(fan::vec3(root_pos + offset, position.z));

        for (auto& child : children) {
          child.set_position(fan::vec3(fan::vec2(child.get_position()) + offset, position.z));
        }
        update_highlight_position(this);

        if (previous_focus && previous_focus != this) {
          for (std::size_t i = 0; i < previous_focus->highlight[0].size(); ++i) {
            if (previous_focus->highlight[0][i].iic() == false) {
              previous_focus->highlight[0][i].set_line(0, 0);
            }
          }
          previous_focus = this;
        }

        for (auto* i : selected_objects) {
          if (i == this) {
            continue;
          }
          fan::vec2 root_pos = i->vfi_root.get_position();
          i->vfi_root.set_position(fan::vec3(root_pos + offset, position.z));

          for (auto& child : i->children) {
            child.set_position(fan::vec3(fan::vec2(child.get_position()) + offset, position.z));
          }
          update_highlight_position(i);
        }
      }
      fan::vec2 get_size() {
        return vfi_root.get_size();
      }
      void set_size(const fan::vec2& size) {
        fan::vec2 root_pos = vfi_root.get_size();
        fan::vec2 offset = size - root_pos;
        vfi_root.set_size(root_pos + offset);
        for (auto& child : children) {
          child.set_size(fan::vec2(child.get_size()) + offset);
        }
      }

      fan::color get_color() {
        if (children.size()) {
          return children[0].get_color();
        }
        return fan::color(1);
      }
      void set_color(const fan::color& color) {
        for (auto& child : children) {
          child.set_color(color);
        }
      }

      inline static bool g_ignore_mouse = false;
      inline static bool moving_object = false;

      fan::vec2 click_offset = 0;
      fan::vec2 previous_click_position;
      bool move = false;
      bool resize = false;

      bool move_and_resize_auto = true;

      loco_t::shape_t vfi_root;
      struct child_data_t : loco_t::shape_t, T {

      };
      std::vector<child_data_t> children;

      inline static std::vector<vfi_root_custom_t<T>*> selected_objects;

      inline static vfi_root_custom_t<T>* previous_focus = nullptr;

      // 4 lines for square
      std::vector<std::array<loco_t::shape_t, 4>> highlight{ 1 };
    };

    using vfi_root_t = vfi_root_custom_t<__empty_struct>;


    template <typename T>
    struct vfi_multiroot_custom_t {
      void push_root(const loco_t::vfi_t::properties_t& p) {
        loco_t::vfi_t::properties_t in = p;
        in.shape_type = loco_t::vfi_t::shape_t::rectangle;
        in.shape.rectangle->viewport = p.shape.rectangle->viewport;
        in.shape.rectangle->camera = p.shape.rectangle->camera;
        in.keyboard_cb = [this, user_cb = p.keyboard_cb](const auto& d) -> int {
          if (d.key == fan::key_c &&
            (d.keyboard_state == fan::keyboard_state::press ||
              d.keyboard_state == fan::keyboard_state::repeat)) {
            this->resize = true;
            return 0;
          }
          this->resize = false;
          return user_cb(d);
          };
        in.mouse_button_cb = [this, root_reference = vfi_root.empty() ? 0 : vfi_root.size() - 1, user_cb = p.mouse_button_cb](const auto& d) -> int {
          if (g_ignore_mouse) {
            return 0;
          }

          if (d.button != fan::mouse_left) {
            return user_cb(d);
          }

          if (d.button_state == fan::mouse_state::press && move_and_resize_auto) {
            this->move = true;
            gloco->vfi.focus.method.mouse.flags.ignore_move_focus_check = true;
          }
          else if (d.button_state == fan::mouse_state::release && move_and_resize_auto) {
            this->move = false;
            gloco->vfi.focus.method.mouse.flags.ignore_move_focus_check = false;
          }

          if (d.button_state == fan::mouse_state::release) {
            for (auto& root : vfi_root) {
              auto position = root->get_position();
              auto p = fan::vec3(fan::vec2(position), position.z);
              if (grid_size.x > 0) {
                p.x = floor(p.x / grid_size.x) * grid_size.x + grid_size.x / 2;
              }
              if (grid_size.y > 0) {
                p.y = floor(p.y / grid_size.y) * grid_size.y + grid_size.y / 2;
              }
              root->set_position(p);
            }
            for (auto& child : children) {
              auto position = child.get_position();
              auto p = fan::vec3(fan::vec2(position), position.z);
              if (grid_size.x > 0) {
                p.x = floor(p.x / grid_size.x) * grid_size.x + grid_size.x / 2;
              }
              if (grid_size.y > 0) {
                p.y = floor(p.y / grid_size.y) * grid_size.y + grid_size.y / 2;
              }
              child.set_position(p);
            }
          }
          if (d.button_state != fan::mouse_state::press) {
            return user_cb(d);
          }
          if (d.mouse_stage != loco_t::vfi_t::mouse_stage_e::viewport_inside) {
            return user_cb(d);
          }

          if (move_and_resize_auto) {
            this->click_offset = get_position(root_reference) - d.position;
            gloco->vfi.set_focus_keyboard(d.vfi->focus.mouse);
          }
          return user_cb(d);
          };
        in.mouse_move_cb = [this, root_reference = vfi_root.empty() ? 0 : vfi_root.size() - 1, user_cb = p.mouse_move_cb](const auto& d) -> int {
          if (g_ignore_mouse) {
            return 0;
          }

          if (move_and_resize_auto) {
            if (this->resize && this->move) {
              return user_cb(d);
            }
            else if (this->move) {
              fan::vec3 p = get_position(root_reference);
              p = fan::vec3(d.position + click_offset, p.z);
              this->set_position(root_reference, p);
              return user_cb(d);
            }
          }
          else {
            return user_cb(d);
          }
          return 0;
          };
        vfi_root.push_back(std::make_unique<loco_t::shape_t>(in));
      }
      void push_child(const loco_t::shape_t& shape) {
        children.push_back({ shape });
      }
      fan::vec3 get_position(uint32_t index) {
        return vfi_root[index]->get_position();
      }
      void set_position(uint32_t root_reference, const fan::vec3& position) {
        fan::vec2 root_pos = vfi_root[root_reference]->get_position();
        fan::vec2 offset = position - root_pos;
        for (auto& root : vfi_root) {
          auto p = fan::vec3(fan::vec2(root->get_position()) + offset, position.z);
          root->set_position(fan::vec3(p.x, p.y, p.z));
        }
        for (auto& child : children) {
          auto p = fan::vec3(fan::vec2(child.get_position()) + offset, position.z);
          child.set_position(p);
        }
      }

      inline static bool g_ignore_mouse = false;

      fan::vec2 click_offset = 0;
      bool move = false;
      bool resize = false;
      fan::vec2 grid_size = 0;

      bool move_and_resize_auto = true;

      std::vector<std::unique_ptr<loco_t::shape_t>> vfi_root;
      struct child_data_t : loco_t::shape_t, T {

      };
      std::vector<child_data_t> children;
    };

    using vfi_multiroot_t = vfi_multiroot_custom_t<__empty_struct>;

  #endif
//#endif
  }
}

// Imgui extensions
#if defined(loco_imgui)
namespace ImGui {
  IMGUI_API void Image(loco_t::image_t img, const ImVec2& size, const ImVec2& uv0 = ImVec2(0, 0), const ImVec2& uv1 = ImVec2(1, 1), const ImVec4& tint_col = ImVec4(1, 1, 1, 1), const ImVec4& border_col = ImVec4(0, 0, 0, 0));
  IMGUI_API bool ImageButton(const std::string& str_id, loco_t::image_t img, const ImVec2& size, const ImVec2& uv0 = ImVec2(0, 0), const ImVec2& uv1 = ImVec2(1, 1), int frame_padding = -1, const ImVec4& bg_col = ImVec4(0, 0, 0, 0), const ImVec4& tint_col = ImVec4(1, 1, 1, 1));
  IMGUI_API bool ImageTextButton(loco_t::image_t img, const std::string& text, const fan::color& color, const ImVec2& size, const ImVec2& uv0 = ImVec2(0, 0), const ImVec2& uv1 = ImVec2(1, 1), int frame_padding = -1, const ImVec4& bg_col = ImVec4(0, 0, 0, 0), const ImVec4& tint_col = ImVec4(1, 1, 1, 1));

  bool ToggleButton(const std::string& str, bool* v);
  bool ToggleImageButton(const std::string& char_id, loco_t::image_t image, const ImVec2& size, bool* toggle);
  
  void DrawTextBottomRight(const char* text, uint32_t reverse_yoffset = 0);


  template <std::size_t N>
  bool ToggleImageButton(const std::array<loco_t::image_t, N>& images, const ImVec2& size, int* selectedIndex)
  {
    f32_t y_pos = ImGui::GetCursorPosY() + ImGui::GetStyle().WindowPadding.y -  ImGui::GetStyle().FramePadding.y / 2;
    
    bool clicked = false;
    bool pushed = false;

    for (std::size_t i = 0; i < images.size(); ++i) {
      ImVec4 tintColor = ImVec4(0.2, 0.2, 0.2, 1.0);
      if (*selectedIndex == i) {
        tintColor = ImVec4(0.2, 0.2, 0.2, 1.0f);
        ImGui::PushStyleColor(ImGuiCol_Button, tintColor);
        pushed = true;
      }
      /*if (ImGui::IsItemHovered()) {
        tintColor = ImVec4(1, 1, 1, 1.0f);
      }*/
      ImGui::SetCursorPosY(y_pos);
      if (ImGui::ImageButton("##toggle_image_button" + std::to_string(i) + std::to_string((uint64_t)&clicked), images[i], size)) {
        *selectedIndex = i;
        clicked = true;
      }
      if (pushed) {
        ImGui::PopStyleColor();
        pushed = false;
      }

      ImGui::SameLine();
    }

    return clicked;
  }


  ImVec2 GetPositionBottomCorner(const char* text = "", uint32_t reverse_yoffset = 0);

  void ImageRotated(ImTextureID user_texture_id, const ImVec2& size, int angle, const ImVec2& uv0 = ImVec2(0, 0), const ImVec2& uv1 = ImVec2(1,1), const ImVec4& tint_col = ImVec4(1,1,1,1), const ImVec4& border_col = ImVec4(0,0,0,0));
}
// Imgui extensions

#include <fan/io/directory.h>

namespace fan {
  namespace graphics {
    struct imgui_content_browser_t {
      struct file_info_t {
        std::string filename;
        std::filesystem::path some_path; //?
        std::wstring item_path;
        bool is_directory;
        loco_t::image_t preview_image;
        //std::string 
      };

      std::vector<file_info_t> directory_cache;

      loco_t::image_t icon_arrow_left = gloco->image_load("images_content_browser/arrow_left.webp");
      loco_t::image_t icon_arrow_right = gloco->image_load("images_content_browser/arrow_right.webp");

      loco_t::image_t icon_file = gloco->image_load("images_content_browser/file.webp");
      loco_t::image_t icon_directory = gloco->image_load("images_content_browser/folder.webp");

      loco_t::image_t icon_files_list = gloco->image_load("images_content_browser/files_list.webp");
      loco_t::image_t icon_files_big_thumbnail = gloco->image_load("images_content_browser/files_big_thumbnail.webp");

      bool item_right_clicked = false;
      std::string item_right_clicked_name;

      std::wstring asset_path = L"./";

      std::filesystem::path current_directory;
      enum viewmode_e {
        view_mode_list,
        view_mode_large_thumbnails,
      };
      viewmode_e current_view_mode = view_mode_list;
      float thumbnail_size = 128.0f;
      f32_t padding = 16.0f;
      std::string search_buffer;

      imgui_content_browser_t();
      void update_directory_cache();
      void render();
      void render_large_thumbnails_view();
      void render_list_view();
      void handle_item_interaction(const file_info_t& file_info);
      // [](const std::filesystem::path& path) {}
      void receive_drag_drop_target(auto receive_func) {
        ImGui::Dummy(ImGui::GetContentRegionAvail());

        if (ImGui::BeginDragDropTarget()) {
          if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("CONTENT_BROWSER_ITEM")) {
            const wchar_t* path = (const wchar_t*)payload->Data;
            receive_func(std::filesystem::path(path));
            //fan::print(std::filesystem::path(path));
          }
          ImGui::EndDragDropTarget();
        }
      }
    };
  }
}
#endif

void init_imgui();

void shape_keypack_traverse(loco_t::shaper_t::KeyTraverse_t& KeyTraverse, fan::opengl::context_t& context);

#if defined(loco_box2d)
  #include <fan/graphics/physics_shapes.hpp>
#endif

namespace fan {
  namespace graphics {
    struct interactive_camera_t {
      loco_t::update_callback_nr_t uc_nr;
      f32_t zoom = 2;
      bool hovered = false;
      loco_t::camera_t reference_camera;
      loco_t::viewport_t reference_viewport;
      fan::window_t::buttons_callback_t::nr_t button_cb_nr;

      interactive_camera_t(
        loco_t::camera_t camera_nr = gloco->orthographic_camera.camera, 
        loco_t::viewport_t viewport_nr = gloco->orthographic_camera.viewport
      );
      ~interactive_camera_t();

      // called in loop
      void move_by_cursor();
    };

#if defined(loco_imgui)
    struct dialogue_box_t {

      dialogue_box_t();

      // 0-1
      void set_cursor_position(const fan::vec2& pos);
      fan::ev::task_t text(const std::string& text);

      fan::ev::task_t button(const std::string& text, const fan::vec2& position = -1, const fan::vec2& size = {0, 0});
      int get_button_choice() const;

      fan::ev::task_t wait_user_input();

      void render(const std::string& window_name, ImFont* font, const fan::vec2& window_size, f32_t wrap_width, f32_t line_spacing);

      bool finish_dialog = false; // for skipping
      bool wait_user = false;
      std::string active_dialogue;

      uint64_t character_per_s = 20;
      std::size_t render_pos = 0;
      fan::vec2 cursor_position = -1;
      struct button_t {
        fan::vec2 position = -1;
        fan::vec2 size = 0;
        std::string text;
      };
      std::vector<button_t> buttons;
      int button_choice = -1;
    };

#endif

    struct animator_t {
      fan::vec2 prev_dir = 0;

      uint64_t animation_update_time = 150;//ms
      uint16_t i_down = 0, i_up = 0, i_left = 0, i_right = 0;

      template <std::size_t images_per_action>
      void process_walk(loco_t::shape_t& shape,
        const fan::vec2& vel,
        const std::array<loco_t::image_t, 4>& img_idle,
        const std::array<loco_t::image_t, images_per_action>& img_movement_left,
        const std::array<loco_t::image_t, images_per_action>& img_movement_right,
        const std::array<loco_t::image_t, images_per_action>& img_movement_up,
        const std::array<loco_t::image_t, images_per_action>& img_movement_down
      ) {
        f32_t animation_velocity_threshold = 10.f;
        fan_ev_timer_loop_init(animation_update_time,
          0/*vel.y*/,
          {
          if (vel.y > animation_velocity_threshold) {
            shape.set_image(img_movement_down[i_down % images_per_action]);
            prev_dir.y = 1;
            prev_dir.x = 0;
            ++i_down;
          }
          else if (vel.y < -animation_velocity_threshold) {
            static int i = 0;
            shape.set_image(img_movement_up[i_up % images_per_action]);
            prev_dir.y = -1;
            prev_dir.x = 0;
            ++i_up;
          }
          else {
            if (prev_dir.y < 0) {
              shape.set_image(img_idle[2]);
            }
            else if (prev_dir.y > 0) {
              shape.set_image(img_idle[3]);
            }
            prev_dir.y = 0;
          }
          });

        if (prev_dir.y == 0) {
          fan_ev_timer_loop_init(animation_update_time,
            0/*vel.x*/,
            {
            if (vel.x > animation_velocity_threshold) {
              static int i = 0;
              shape.set_image(img_movement_right[i_right % images_per_action]);
              prev_dir.y = 0;
              prev_dir.x = 1;
              ++i_right;
            }
            else if (vel.x < -animation_velocity_threshold) {
              static int i = 0;
              shape.set_image(img_movement_left[i_left % images_per_action]);
              prev_dir.y = 0;
              prev_dir.x = -1;
              ++i_left;
            }
            else {
              if (prev_dir.x < 0) {
                shape.set_image(img_idle[0]);
              }
              else if (prev_dir.x > 0) {
                shape.set_image(img_idle[1]);
              }
              prev_dir.x = 0;
            }
            });
        }
      }
    };

  }

  struct movement_e {
    fan_enum_string(
      ,
      left,
      right,
      up,
      down
    );
  };

#if defined(loco_box2d)
  namespace physics {
    bool is_on_sensor(fan::physics::body_id_t test_id, fan::physics::body_id_t sensor_id);
    fan::physics::ray_result_t raycast(const fan::vec2& src, const fan::vec2& dst);
  }
#endif
}

// makes shorter code
#define fan_language \
  void main_entry(); \
  using namespace fan::graphics; \
  using namespace fan; \
  int main() { \
    fan::graphics::engine_t engine; \
    main_entry(); \
    return 0; \
  } \
  void main_entry()

#define fan_window_loop \
  static std::function<void()> loop_entry=[]{}; \
  /*destructor_magic*/ \
  struct dm_t { \
    ~dm_t() { \
      gloco->loop(loop_entry); \
    } \
  }dm; \
  loop_entry = [&]()