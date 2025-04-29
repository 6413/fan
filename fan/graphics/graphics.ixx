module;
// extension to loco.h

#if defined(fan_gui)
  #include <fan/imgui/imgui_internal.h>
  #include <fan/graphics/gui/imgui_themes.h>
#endif

#include <fan/types/types.h>
#include <fan/graphics/opengl/init.h>
#include <fan/ev/types.h>

#include <fan/math/math.h>

import fan.types.vector;

import fan.window;
import fan.graphics.image_load;
import fan.graphics.opengl.core;
import fan.graphics.common_context;
import fan.graphics.loco;

import fan.io.directory;

import fan.camera;
import fan.types.color;
import fan.random;
import fan.io.file;

export module fan.graphics;

// user friendly functions
/***************************************/

export namespace fan {
  namespace window {
    fan::vec2 get_size() {
      return gloco->window.get_size();
    }
    void set_size(const fan::vec2& size) {
      gloco->window.set_size(size);
      gloco->viewport_set(gloco->orthographic_camera.viewport, fan::vec2(0, 0), size, size);
      gloco->camera_set_ortho(
        gloco->orthographic_camera.camera,
        fan::vec2(0, size.x),
        fan::vec2(0, size.y)
      );
      gloco->viewport_set(gloco->perspective_camera.viewport, fan::vec2(0, 0), size, size);
      gloco->camera_set_ortho(
        gloco->perspective_camera.camera,
        fan::vec2(0, size.x),
        fan::vec2(0, size.y)
      );
    }

    bool is_mouse_clicked(int button = fan::mouse_left) {
      return gloco->is_mouse_clicked(button);
    }
    bool is_mouse_down(int button = fan::mouse_left) {
      return gloco->is_mouse_down(button);
    }
    bool is_mouse_released(int button = fan::mouse_left) {
      return gloco->is_mouse_released(button);
    }
    fan::vec2 get_mouse_drag(int button = fan::mouse_left) {
      return gloco->get_mouse_drag(button);
    }

    bool is_key_pressed(int key) {
      return gloco->is_key_pressed(key);
    }
    bool is_key_down(int key) {
      return gloco->is_key_down(key);
    }
    bool is_key_released(int key) {
      return gloco->is_key_released(key);
    }
  }
}

export namespace fan {
  namespace graphics {
    using engine_t = fan::graphics::engine_t;
    fan::graphics::image_nr_t image_load(const fan::image::image_info_t& image_info) {
      return gloco->image_load(image_info);
    }
    fan::graphics::image_nr_t image_load(const fan::image::image_info_t& image_info, const fan::graphics::image_load_properties_t& p) {
      return gloco->image_load(image_info, p);
    }
    fan::graphics::image_nr_t image_load(const std::string& path) {
      return gloco->image_load(path);
    }
    fan::graphics::image_nr_t image_load(const std::string& path, const fan::graphics::image_load_properties_t& p) {
      return gloco->image_load(path, p);
    }
    fan::graphics::image_nr_t image_load(fan::color* colors, const fan::vec2ui& size) {
      return gloco->image_load(colors, size);
    }
    fan::graphics::image_nr_t image_load(fan::color* colors, const fan::vec2ui& size, const fan::graphics::image_load_properties_t& p) {
      return gloco->image_load(colors, size, p);
    }
    void image_unload(fan::graphics::image_nr_t nr) {
      return gloco->image_unload(nr);
    }

    using light_flags_e = loco_t::light_flags_e;

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
      uint32_t flags = light_flags_e::circle | light_flags_e::additive;
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
      fan::vec3 angle = 0;
      fan::color color = fan::color(1, 1, 1, 1);
      bool blending = true;
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
            .angle = p.angle,
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
      uint8_t draw_mode = fan::graphics::primitive_topology_t::triangle_strip;
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
            .draw_mode = p.draw_mode
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

    struct universal_image_renderer_properties_t {
      camera_impl_t* camera = &gloco->orthographic_camera;
      fan::vec3 position = 0;
      fan::vec2 size = 0;
      fan::vec2 tc_position = 0;
      fan::vec2 tc_size = 1;

      bool blending = false;

      std::array<loco_t::image_t, 4> images = {
        gloco->default_texture,
        gloco->default_texture,
        gloco->default_texture,
        gloco->default_texture
      };
      uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
    };

    struct universal_image_renderer_t : loco_t::shape_t {
      universal_image_renderer_t(const universal_image_renderer_properties_t& p = universal_image_renderer_properties_t()) {
         *(loco_t::shape_t*)this = loco_t::shape_t(
          fan_init_struct(
            typename loco_t::universal_image_renderer_t::properties_t,
            .camera = p.camera->camera,
            .viewport = p.camera->viewport,
            .position = p.position,
            .size = p.size,
            .tc_position = p.tc_position,
            .tc_size = p.tc_size,
            .blending = p.blending,
            .images = p.images,
            .draw_mode = p.draw_mode,
          )
        );
      }
    };

    struct gradient_properties_t {
      camera_impl_t* camera = &gloco->perspective_camera;

      fan::vec3 position = 0;
      fan::vec2 size = 0;
      std::array<fan::color, 4> color = {
        fan::random::color(),
        fan::random::color(),
        fan::random::color(),
        fan::random::color()
      };
      bool blending = true;
      fan::vec3 angle = 0;
      fan::vec2 rotation_point = 0;

      uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
    };

    struct gradient_t : loco_t::shape_t{
      gradient_t(const gradient_properties_t& p = gradient_properties_t()) {
        *(loco_t::shape_t*)this = loco_t::shape_t(
          fan_init_struct(
            typename loco_t::gradient_t::properties_t,
            .camera = p.camera->camera,
            .viewport = p.camera->viewport,
            .position = p.position,
            .size = p.size,
            .color = p.color,
            .blending = p.blending,
            .angle = p.angle,
            .rotation_point = p.rotation_point
          )
        );
      }
    };

    struct shader_shape_properties_t {
      camera_impl_t* camera = &gloco->perspective_camera;

      fan::vec3 position = 0;
      fan::vec2 size = 0;
      fan::vec2 rotation_point = 0;
      fan::color color = fan::colors::white;
      fan::vec3 angle = fan::vec3(0);
      uint32_t flags = 0;
      fan::vec2 tc_position = 0;
      fan::vec2 tc_size = 1;
      loco_t::shader_t shader;
      bool blending = true;

      loco_t::image_t image = gloco->default_texture;
      std::array<loco_t::image_t, 30> images;

      uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
    };

    struct shader_shape_t : loco_t::shape_t {
      shader_shape_t(const shader_shape_properties_t& p = shader_shape_properties_t()) {
       *(loco_t::shape_t*)this = loco_t::shape_t(
          fan_init_struct(
            typename loco_t::shader_shape_t::properties_t,
            .camera = p.camera->camera,
            .viewport = p.camera->viewport,
            .position = p.position,
            .size = p.size,
            .rotation_point = p.rotation_point,
            .color = p.color,
            .angle = p.angle,
            .flags = p.flags,
            .tc_position = p.tc_position,
            .tc_size = p.tc_size,
            .shader = p.shader,
            .blending = p.blending,
            .image = p.image,
            .images = p.images,
            .draw_mode = p.draw_mode
          )
        );
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

    loco_t::polygon_t::properties_t create_hexagon(f32_t radius, const fan::color& color) {
  loco_t::polygon_t::properties_t pp;
  // for triangle strip
  for (int i = 0; i < 6; ++i) {
    pp.vertices.push_back(fan::graphics::vertex_t{ fan::vec3(0, 0, 0), color });

    f32_t angle1 = 2 * fan::math::pi * i / 6;
    f32_t x1 = radius * std::cos(angle1);
    f32_t y1 = radius * std::sin(angle1);
    pp.vertices.push_back(fan::graphics::vertex_t{ fan::vec3(fan::vec2(x1, y1), 0), color });

    f32_t angle2 = 2 * fan::math::pi * ((i + 1) % 6) / 6;
    f32_t x2 = radius * std::cos(angle2);
    f32_t y2 = radius * std::sin(angle2);
    pp.vertices.push_back(fan::graphics::vertex_t{ fan::vec3(fan::vec2(x2, y2), 0), color });
  }

  return pp;
}

#if defined(loco_vfi)

    // for line
    fan::line3 get_highlight_positions(const fan::vec3& op_, const fan::vec2& os, int index) {
  fan::line3 positions;
  fan::vec2 op = op_;
  switch (index) {
  case 0:
    positions[0] = fan::vec3(op - os, op_.z + 1);
    positions[1] = fan::vec3(op + fan::vec2(os.x, -os.y), op_.z + 1);
    break;
  case 1:
    positions[0] = fan::vec3(op + fan::vec2(os.x, -os.y), op_.z + 1);
    positions[1] = fan::vec3(op + os, op_.z + 1);
    break;
  case 2:
    positions[0] = fan::vec3(op + os, op_.z + 1);
    positions[1] = fan::vec3(op + fan::vec2(-os.x, os.y), op_.z + 1);
    break;
  case 3:
    positions[0] = fan::vec3(op + fan::vec2(-os.x, os.y), op_.z + 1);
    positions[1] = fan::vec3(op - os, op_.z + 1);
    break;
  default:
    // Handle invalid index if necessary
    break;
  }

  return positions;
}

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
#if defined(fan_gui)
namespace fan {
  namespace graphics {
    namespace gui {
      void image(loco_t::image_t img, const fan::vec2& size, const fan::vec2& uv0 = fan::vec2(0, 0), const fan::vec2& uv1 = fan::vec2(1, 1), const fan::color& tint_col = fan::color(1, 1, 1, 1), const fan::color& border_col = fan::color(0, 0, 0, 0)) {
        ImGui::Image((ImTextureID)gloco->image_get_handle(img), size, uv0, uv1, tint_col, border_col);
      }
      bool image_button(const std::string& str_id, loco_t::image_t img, const fan::vec2& size, const fan::vec2& uv0 = fan::vec2(0, 0), const fan::vec2& uv1 = fan::vec2(1, 1), int frame_padding = -1, const fan::color& bg_col = fan::color(0, 0, 0, 0), const fan::color& tint_col = fan::color(1, 1, 1, 1)) {
        return ImGui::ImageButton(str_id.c_str(), (ImTextureID)gloco->image_get_handle(img), size, uv0, uv1, bg_col, tint_col);
      }
      bool image_text_button(
        loco_t::image_t img,
        const std::string& text,
        const fan::color& color,
        const fan::vec2& size,
        const fan::vec2& uv0 = fan::vec2(0, 0),
        const fan::vec2& uv1 = fan::vec2(1, 1),
        int frame_padding = -1,
        const fan::color& bg_col = fan::color(0, 0, 0, 0),
        const fan::color& tint_col = fan::color(1, 1, 1, 1)
      ) {
        bool ret = ImGui::ImageButton(text.c_str(), (ImTextureID)gloco->image_get_handle(img), size, uv0, uv1, bg_col, tint_col);
        ImVec2 text_size = ImGui::CalcTextSize(text.c_str());
        ImVec2 pos = ImGui::GetItemRectMin();
        pos.x += (size.x - text_size.x) * 0.5f;
        pos.y += (size.y - text_size.y) * 0.5f;
        ImGui::GetWindowDrawList()->AddText(pos, ImGui::GetColorU32(color), text.c_str());
        return ret;
      }

      bool toggle_button(const std::string& str, bool* v) {
        ImGui::Text("%s", str.c_str());
        ImGui::SameLine();

        ImVec2 p = ImGui::GetCursorScreenPos();
        ImDrawList* draw_list = ImGui::GetWindowDrawList();

        float height = ImGui::GetFrameHeight();
        float width = height * 1.55f;
        float radius = height * 0.50f;

        bool changed = ImGui::InvisibleButton(("##" + str).c_str(), ImVec2(width, height));
        if (changed)
          *v = !*v;
        ImU32 col_bg;
        if (ImGui::IsItemHovered())
          col_bg = *v ? IM_COL32(145 + 20, 211, 68 + 20, 255) : IM_COL32(218 - 20, 218 - 20, 218 - 20, 255);
        else
          col_bg = *v ? IM_COL32(145, 211, 68, 255) : IM_COL32(218, 218, 218, 255);

        draw_list->AddRectFilled(p, ImVec2(p.x + width, p.y + height), col_bg, height * 0.5f);
        draw_list->AddCircleFilled(ImVec2(*v ? (p.x + width - radius) : (p.x + radius), p.y + radius), radius - 1.5f, IM_COL32(255, 255, 255, 255));

        return changed;
      }
      bool toggle_image_button(const std::string& char_id, loco_t::image_t image, const fan::vec2& size, bool* toggle) {
        bool clicked = false;

        ImVec4 tintColor = ImVec4(1, 1, 1, 1);
        if (*toggle) {
          tintColor = ImVec4(0.3f, 0.3f, 0.3f, 1.0f);
        }

        if (fan::graphics::gui::image_button(char_id, image, size, ImVec2(0, 0), ImVec2(1, 1), -1, ImVec4(0, 0, 0, 0), tintColor)) {
          *toggle = !(*toggle);
          clicked = true;
        }

        return clicked;
      }

      void text_bottom_right(const char* text, uint32_t reverse_yoffset = 0) {
        ImDrawList* draw_list = ImGui::GetWindowDrawList();

        ImVec2 window_pos = ImGui::GetWindowPos();
        ImVec2 window_size = ImGui::GetWindowSize();

        ImVec2 text_size = ImGui::CalcTextSize(text);

        ImVec2 text_pos;
        text_pos.x = window_pos.x + window_size.x - text_size.x - ImGui::GetStyle().WindowPadding.x;
        text_pos.y = window_pos.y + window_size.y - text_size.y - ImGui::GetStyle().WindowPadding.y;

        text_pos.y -= reverse_yoffset * ImGui::GetTextLineHeightWithSpacing();

        draw_list->AddText(text_pos, IM_COL32(255, 255, 255, 255), text);
      }


      template <std::size_t N>
      bool toggle_image_button(const std::array<loco_t::image_t, N>& images, const fan::vec2& size, int* selectedIndex)
      {
        f32_t y_pos = ImGui::GetCursorPosY() + ImGui::GetStyle().WindowPadding.y - ImGui::GetStyle().FramePadding.y / 2;

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
          if (fan::graphics::gui::image_button("##toggle_image_button" + std::to_string(i) + std::to_string((uint64_t)&clicked), images[i], size)) {
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


      fan::vec2 get_position_bottom_corner(const std::string& text = "", uint32_t reverse_yoffset = 0) {
        fan::vec2 window_pos = ImGui::GetWindowPos();
        fan::vec2 window_size = ImGui::GetWindowSize();

        fan::vec2 text_size = ImGui::CalcTextSize(text.c_str());

        fan::vec2 text_pos;
        text_pos.x = window_pos.x + window_size.x - text_size.x - ImGui::GetStyle().WindowPadding.x;
        text_pos.y = window_pos.y + window_size.y - text_size.y - ImGui::GetStyle().WindowPadding.y;

        text_pos.y -= reverse_yoffset * ImGui::GetTextLineHeightWithSpacing();

        return text_pos;
      }

      // untested
      void image_rotated(
        loco_t::image_t image,
        const fan::vec2& size,
        int angle,
        const fan::vec2& uv0 = fan::vec2(0, 0),
        const fan::vec2& uv1 = fan::vec2(1, 1),
        const fan::color& tint_col = fan::color(1, 1, 1, 1),
        const fan::color& border_col = fan::color(0, 0, 0, 0)
      ) {
        IM_ASSERT(angle % 90 == 0);
        fan::vec2 _uv0, _uv1, _uv2, _uv3;

        switch (angle % 360) {
        case 0:
          fan::graphics::gui::image(image, size, uv0, uv1, tint_col, border_col);
          return;
        case 180:
          fan::graphics::gui::image(image, size, uv1, uv0, tint_col, border_col);
          return;
        case 90:
          _uv3 = uv0;
          _uv1 = uv1;
          _uv0 = fan::vec2(uv1.x, uv0.y);
          _uv2 = fan::vec2(uv0.x, uv1.y);
          break;
        case 270:
          _uv1 = uv0;
          _uv3 = uv1;
          _uv0 = fan::vec2(uv0.x, uv1.y);
          _uv2 = fan::vec2(uv1.x, uv0.y);
          break;
        }

        ImGuiWindow* window = ImGui::GetCurrentWindow();
        if (window->SkipItems)
          return;

        fan::vec2 _size(size.y, size.x); // swapped for rotation
        fan::vec2 cursor_pos = *(fan::vec2*)&window->DC.CursorPos;
        fan::vec2 bb_max = cursor_pos + _size;
        if (border_col.a > 0.0f) {
          bb_max += fan::vec2(2, 2);
        }

        ImRect bb(*(ImVec2*)&cursor_pos, *(ImVec2*)&bb_max);
        ImGui::ItemSize(bb);
        if (!ImGui::ItemAdd(bb, 0))
          return;

        if (border_col.a > 0.0f) {
          window->DrawList->AddRect(*(ImVec2*)&bb.Min, *(ImVec2*)&bb.Max, ImGui::GetColorU32(border_col), 0.0f);
          fan::vec2 x0 = cursor_pos + fan::vec2(1, 1);
          fan::vec2 x2 = bb_max - fan::vec2(1, 1);
          fan::vec2 x1 = fan::vec2(x2.x, x0.y);
          fan::vec2 x3 = fan::vec2(x0.x, x2.y);

          window->DrawList->AddImageQuad(
            (ImTextureID)gloco->image_get_handle(image),
            *(ImVec2*)&x0, *(ImVec2*)&x1, *(ImVec2*)&x2, *(ImVec2*)&x3,
            *(ImVec2*)&_uv0, *(ImVec2*)&_uv1, *(ImVec2*)&_uv2, *(ImVec2*)&_uv3,
            ImGui::GetColorU32(tint_col)
          );
        }
        else {
          fan::vec2 x0 = cursor_pos;
          fan::vec2 x1 = fan::vec2(bb_max.x, cursor_pos.y);
          fan::vec2 x2 = bb_max;
          fan::vec2 x3 = fan::vec2(cursor_pos.x, bb_max.y);

          window->DrawList->AddImageQuad(
            (ImTextureID)gloco->image_get_handle(image),
            *(ImVec2*)&x0, *(ImVec2*)&x1, *(ImVec2*)&x2, *(ImVec2*)&x3,
            *(ImVec2*)&_uv0, *(ImVec2*)&_uv1, *(ImVec2*)&_uv2, *(ImVec2*)&_uv3,
            ImGui::GetColorU32(tint_col)
          );
        }
      }
    }
  }
}

export namespace fan {
  namespace graphics {
    namespace gui {
      struct content_browser_t {
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

        content_browser_t() {
          search_buffer.resize(32);
          asset_path = std::filesystem::absolute(std::filesystem::path(asset_path)).wstring();
          current_directory = std::filesystem::path(asset_path);
          update_directory_cache();
        }
        content_browser_t(bool no_init) {

        }
        content_browser_t(const std::wstring& path) {
          search_buffer.resize(32);
          asset_path = std::filesystem::absolute(std::filesystem::path(asset_path)).wstring();
          current_directory = asset_path / std::filesystem::path(path);
          update_directory_cache();
        }
        void update_directory_cache() {
          for (auto& img : directory_cache) {
            if (img.preview_image.iic() == false) {
              gloco->image_unload(img.preview_image);
            }
          }
          directory_cache.clear();
          fan::io::iterate_directory_sorted_by_name(current_directory, [this](const std::filesystem::directory_entry& path) {
            file_info_t file_info;
            // SLOW
            auto relative_path = std::filesystem::relative(path, asset_path);
            file_info.filename = relative_path.filename().string();
            file_info.item_path = relative_path.wstring();
            file_info.is_directory = path.is_directory();
            file_info.some_path = path.path().filename();//?
            //fan::print(get_file_extension(path.path().string()));
            if (fan::io::file::extension(path.path().string()) == ".webp" || fan::io::file::extension(path.path().string()) == ".png") {
              file_info.preview_image = gloco->image_load(std::filesystem::absolute(path.path()).string());
            }
            directory_cache.push_back(file_info);
            });
        }
        void render() {
          item_right_clicked = false;
          item_right_clicked_name.clear();
          ImGuiStyle& style = ImGui::GetStyle();
          ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10.0f, 16.0f));
          ImGuiWindowClass window_class;
          //window_class.DockNodeFlagsOverrideSet = ImGuiDockNodeFlags_NoTabBar; TODO ?
          ImGui::SetNextWindowClass(&window_class);
          if (ImGui::Begin("Content Browser", 0, ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoTitleBar)) {
            if (ImGui::BeginMenuBar()) {
              ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.f, 0.f, 0.f, 0.f));
              ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.f, 0.f, 0.f, 0.f));
              ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.3f, 0.3f, 0.3f));

              if (fan::graphics::gui::image_button("##icon_arrow_left", icon_arrow_left, fan::vec2(32))) {
                if (std::filesystem::equivalent(current_directory, asset_path) == false) {
                  current_directory = current_directory.parent_path();
                }
                update_directory_cache();
              }
              ImGui::SameLine();
              fan::graphics::gui::image_button("##icon_arrow_right", icon_arrow_right, fan::vec2(32));
              ImGui::SameLine();
              ImGui::PopStyleColor(3);

              ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.f, 0.f, 0.f, 0.f));
              ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.f, 0.f, 0.f, 0.f));
              ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.3f, 0.3f, 0.3f));

              auto image_list = std::to_array({ icon_files_list, icon_files_big_thumbnail });

              fan::vec2 bc = fan::graphics::gui::get_position_bottom_corner();

              bc.x -= ImGui::GetWindowPos().x;
              ImGui::SetCursorPosX(bc.x / 2);

              fan::vec2 button_sizes = 32;

              ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - (button_sizes.x * 2 + style.ItemSpacing.x) * image_list.size());

              ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 20.0f);
              ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10.0f, 7.0f));
              f32_t y_pos = ImGui::GetCursorPosY() + ImGui::GetStyle().WindowPadding.y;
              ImGui::SetCursorPosY(y_pos);


              if (ImGui::InputText("##content_browser_search", search_buffer.data(), search_buffer.size())) {

              }
              ImGui::PopStyleVar(2);

              fan::graphics::gui::toggle_image_button(image_list, button_sizes, (int*)&current_view_mode);

              ImGui::PopStyleColor(3);

              ///ImGui::InputText("Search", search_buffer.data(), search_buffer.size());

              ImGui::EndMenuBar();
            }
            switch (current_view_mode) {
            case view_mode_large_thumbnails:
              render_large_thumbnails_view();
              break;
            case view_mode_list:
              render_list_view();
              break;
            default:
              break;
            }
          }

          ImGui::PopStyleVar(1);
          ImGui::End();
        }
        void render_large_thumbnails_view() {
          float thumbnail_size = 128.0f;
          float panel_width = ImGui::GetContentRegionAvail().x;
          int column_count = std::max((int)(panel_width / (thumbnail_size + padding)), 1);

          ImGui::Columns(column_count, 0, false);

          int pressed_key = -1;

          auto& style = ImGui::GetStyle();
          // basically bad way to check if gui is disabled. I couldn't find other way
          if (style.DisabledAlpha != style.Alpha) {
            if (ImGui::IsWindowFocused()) {
              for (int i = ImGuiKey_A; i != ImGuiKey_Z + 1; ++i) {
                if (ImGui::IsKeyPressed((ImGuiKey)i, false)) {
                  pressed_key = (i - ImGuiKey_A) + 'A';
                  break;
                }
              }
            }
          }

          // Render thumbnails or icons
          for (std::size_t i = 0; i < directory_cache.size(); ++i) {
            // reference somehow corrupts
            auto file_info = directory_cache[i];
            if (std::string(search_buffer.c_str()).size() && file_info.filename.find(search_buffer) == std::string::npos) {
              continue;
            }

            if (pressed_key != -1 && ImGui::IsWindowFocused()) {
              if (!file_info.filename.empty() && std::tolower(file_info.filename[0]) == std::tolower(pressed_key)) {
                ImGui::SetScrollHereY();
              }
            }

            ImGui::PushID(file_info.filename.c_str());
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
            fan::graphics::gui::image_button("##" + file_info.filename, file_info.preview_image.iic() == false ? file_info.preview_image : file_info.is_directory ? icon_directory : icon_file, ImVec2(thumbnail_size, thumbnail_size));

            bool item_hovered = ImGui::IsItemHovered();
            if (item_hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
              item_right_clicked = true;
              item_right_clicked_name = file_info.filename;
              item_right_clicked_name.erase(std::remove_if(item_right_clicked_name.begin(), item_right_clicked_name.end(), isspace), item_right_clicked_name.end());
            }

            // Handle drag and drop, double click, etc.
            handle_item_interaction(file_info);

            ImGui::PopStyleColor();
            ImGui::TextWrapped("%s", file_info.filename.c_str());
            ImGui::NextColumn();
            ImGui::PopID();
          }

          ImGui::Columns(1);
        }
        void render_list_view() {
          if (ImGui::BeginTable("##FileTable", 1, ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY
            | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV
            | ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable | ImGuiTableFlags_Sortable)) {
            ImGui::TableSetupColumn("##Filename", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableHeadersRow();

            int pressed_key = -1;
            ImGuiStyle& style = ImGui::GetStyle();
            if (style.DisabledAlpha != style.Alpha) {
              if (ImGui::IsWindowFocused()) {
                for (int i = ImGuiKey_A; i != ImGuiKey_Z + 1; ++i) {
                  if (ImGui::IsKeyPressed((ImGuiKey)i, false)) {
                    pressed_key = (i - ImGuiKey_A) + 'A';
                    break;
                  }
                }
              }
            }

            // Render table view
            for (std::size_t i = 0; i < directory_cache.size(); ++i) {

              // reference somehow corrupts
              auto file_info = directory_cache[i];

              if (pressed_key != -1 && ImGui::IsWindowFocused()) {
                if (!file_info.filename.empty() && std::tolower(file_info.filename[0]) == std::tolower(pressed_key)) {
                  ImGui::SetScrollHereY();
                }
              }

              if (search_buffer.size() && strstr(file_info.filename.c_str(), search_buffer.c_str()) == nullptr) {
                continue;
              }
              ImGui::TableNextRow();
              ImGui::TableSetColumnIndex(0); // Icon column
              fan::vec2 cursor_pos = fan::vec2(ImGui::GetWindowPos()) + fan::vec2(ImGui::GetCursorPos()) + fan::vec2(ImGui::GetScrollX(), -ImGui::GetScrollY());
              fan::vec2 image_size = ImVec2(thumbnail_size / 4, thumbnail_size / 4);
              ImGuiStyle& style = ImGui::GetStyle();
              std::string space = "";
              while (ImGui::CalcTextSize(space.c_str()).x < image_size.x) {
                space += " ";
              }
              auto str = space + file_info.filename;

              ImGui::Selectable(str.c_str());
              bool item_hovered = ImGui::IsItemHovered();
              if (item_hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
                item_right_clicked_name = str;
                item_right_clicked_name.erase(std::remove_if(item_right_clicked_name.begin(), item_right_clicked_name.end(), isspace), item_right_clicked_name.end());
                item_right_clicked = true;
              }
              if (file_info.preview_image.iic() == false) {
                ImGui::GetWindowDrawList()->AddImage((ImTextureID)gloco->image_get_handle(file_info.preview_image), cursor_pos, cursor_pos + image_size);
              }
              else if (file_info.is_directory) {
                ImGui::GetWindowDrawList()->AddImage((ImTextureID)gloco->image_get_handle(icon_directory), cursor_pos, cursor_pos + image_size);
              }
              else {
                ImGui::GetWindowDrawList()->AddImage((ImTextureID)gloco->image_get_handle(icon_file), cursor_pos, cursor_pos + image_size);
              }

              handle_item_interaction(file_info);
            }

            ImGui::EndTable();
          }
        }
        void handle_item_interaction(const file_info_t& file_info) {
          if (file_info.is_directory == false) {

            if (ImGui::BeginDragDropSource()) {
              ImGui::SetDragDropPayload("CONTENT_BROWSER_ITEM", file_info.item_path.data(), (file_info.item_path.size() + 1) * sizeof(wchar_t));
              ImGui::Text("%s", file_info.filename.c_str());
              ImGui::EndDragDropSource();
            }
          }

          if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
            if (file_info.is_directory) {
              current_directory /= file_info.some_path;
              update_directory_cache();
            }
          }
        }

        // [](const std::filesystem::path& path) {}
        void receive_drag_drop_target(auto receive_func) {
          ImGui::Dummy(ImGui::GetContentRegionAvail());

          if (ImGui::BeginDragDropTarget()) {
            if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("CONTENT_BROWSER_ITEM")) {
              const wchar_t* path = (const wchar_t*)payload->Data;
              receive_func(std::filesystem::absolute(std::filesystem::path(asset_path) / path));
              //fan::print(std::filesystem::path(path));
            }
            ImGui::EndDragDropTarget();
          }
        }
      };
    }
  }
}
#endif

void init_imgui();

void shape_keypack_traverse(loco_t::shaper_t::KeyTraverse_t& KeyTraverse, fan::opengl::context_t& context);

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

#if defined(fan_gui)
    //struct dialogue_box_t {

    //  dialogue_box_t();

    //  // 0-1
    //  void set_cursor_position(const fan::vec2& pos);
    //  fan::ev::task_t text(const std::string& text);

    //  fan::ev::task_t button(const std::string& text, const fan::vec2& position = -1, const fan::vec2& size = {0, 0});
    //  int get_button_choice() const;

    //  fan::ev::task_t wait_user_input();

    //  void render(const std::string& window_name, ImFont* font, const fan::vec2& window_size, f32_t wrap_width, f32_t line_spacing);

    //  bool finish_dialog = false; // for skipping
    //  bool wait_user = false;
    //  std::string active_dialogue;

    //  uint64_t character_per_s = 20;
    //  std::size_t render_pos = 0;
    //  fan::vec2 cursor_position = -1;
    //  struct button_t {
    //    fan::vec2 position = -1;
    //    fan::vec2 size = 0;
    //    std::string text;
    //  };
    //  std::vector<button_t> buttons;
    //  int button_choice = -1;
    //};

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

    loco_t::polygon_t::properties_t create_sine_ground(const fan::vec2& position, f32_t amplitude, f32_t frequency, f32_t width, f32_t groundWidth) {
      loco_t::polygon_t::properties_t pp;
      // for triangle strip
      for (f32_t x = 0; x < groundWidth - width; x += width) {
        f32_t y1 = position.y / 2 + amplitude * std::sin(frequency * x);
        f32_t y2 = position.y / 2 + amplitude * std::sin(frequency * (x + width));

        // top
        pp.vertices.push_back({ fan::vec2(position.x + x, y1), fan::colors::red });
        // bottom
        pp.vertices.push_back({ fan::vec2(position.x + x, position.y), fan::colors::white });
        // next top
        pp.vertices.push_back({ fan::vec2(position.x + x + width, y2), fan::colors::red });
        // next bottom
        pp.vertices.push_back({ fan::vec2(position.x + x + width, position.y), fan::colors::white });
      }

      return pp;
    }
    std::vector<fan::vec2> ground_points(const fan::vec2& position, f32_t amplitude, f32_t frequency, f32_t width, f32_t groundWidth) {
      std::vector<fan::vec2> outline_points;
      for (f32_t x = 0; x <= groundWidth; x += width) {
        f32_t y = position.y / 2 + amplitude * std::sin(frequency * x);
        outline_points.push_back(fan::vec2(position.x + x, y));
      }
      outline_points.push_back(fan::vec2(position.x + groundWidth, position.y));
      outline_points.push_back(fan::vec2(position.x, position.y));
      return outline_points;
    }
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

#if defined(fan_physics)
  export namespace physics {
    bool is_on_sensor(fan::physics::body_id_t test_id, fan::physics::body_id_t sensor_id) {
      return gloco->physics_context.is_on_sensor(test_id, sensor_id);
    }
    fan::physics::ray_result_t raycast(const fan::vec2& src, const fan::vec2& dst) {
      return gloco->physics_context.raycast(src, dst);
    }
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