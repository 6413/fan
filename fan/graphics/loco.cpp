#include "loco.h"

#include <fan/time/time.h>
#include <fan/memory/memory.hpp>

#ifndef __generic_malloc
  #define __generic_malloc(n) malloc(n)
#endif

#ifndef __generic_realloc
  #define __generic_realloc(ptr, n) realloc(ptr, n)
#endif

#ifndef __generic_free
  #define __generic_free(ptr) free(ptr)
#endif

#if defined(loco_imgui)
  #include <fan/imgui/imgui_internal.h>
  #include <fan/graphics/gui/imgui_themes.h>
#endif

#define loco_framebuffer
#define loco_post_process
//
//#define depth_debug
//
global_loco_t::operator loco_t* () {
  return loco;
}

global_loco_t& global_loco_t::operator=(loco_t* l) {
  loco = l;
  return *this;
}

uint32_t fan::graphics::get_draw_mode(uint8_t internal_draw_mode) {
  if (gloco->get_renderer() == loco_t::renderer_t::opengl) {
#if defined(loco_opengl)
    return fan::opengl::core::get_draw_mode(internal_draw_mode);
#endif
  }
  else if (gloco->get_renderer() == loco_t::renderer_t::vulkan) {
#if defined(loco_vulkan)
    return fan::vulkan::core::get_draw_mode(internal_draw_mode);
#endif
  }
#if fan_debug >= fan_debug_medium
  fan::throw_error("invalid get");
#endif
  return -1;
}

void shaper_deep_copy(loco_t::shape_t* dst, const loco_t::shape_t* const src, loco_t::shaper_t::ShapeTypeIndex_t sti) {
  // alloc can be avoided inside switch
  uint8_t* KeyPack = new uint8_t[gloco->shaper.GetKeysSize(*src)];
  gloco->shaper.WriteKeys(*src, KeyPack);

  auto _vi = src->GetRenderData(gloco->shaper);
  auto vlen = gloco->shaper.GetRenderDataSize(sti);
  uint8_t* vi = new uint8_t[vlen];
  std::memcpy(vi, _vi, vlen);

  auto _ri = src->GetData(gloco->shaper);
  auto rlen = gloco->shaper.GetDataSize(sti);

  uint8_t* ri = new uint8_t[rlen];
  std::memcpy(ri, _ri, rlen);

  *dst = gloco->shaper.add(
    sti,
    KeyPack,
    gloco->shaper.GetKeysSize(*src),
    vi,
    ri
  );
#if defined(debug_shape_t)
  fan::print("+", NRI);
#endif

  delete[] KeyPack;
  delete[] vi;
  delete[] ri;
}

//thread_local global_loco_t gloco;

fan::graphics::shader_nr_t loco_t::shader_create() {
  return context_functions.shader_create(&context);
}

// warning does deep copy, addresses can die
fan::graphics::context_shader_t loco_t::shader_get(fan::graphics::shader_nr_t nr) {
  fan::graphics::context_shader_t context_shader;
  if (window.renderer == renderer_t::opengl) {
    context_shader = *(fan::opengl::context_t::shader_t*)context_functions.shader_get(&context, nr);
  }
  else if (window.renderer == renderer_t::vulkan) {
    context_shader = *(fan::vulkan::context_t::shader_t*)context_functions.shader_get(&context, nr);
  }
  return context_shader;
}

void loco_t::shader_erase(fan::graphics::shader_nr_t nr) {
  context_functions.shader_erase(&context, nr);
}

void loco_t::shader_use(fan::graphics::shader_nr_t nr) {
  context_functions.shader_use(&context, nr);
}

void loco_t::shader_set_vertex(fan::graphics::shader_nr_t nr, const std::string& vertex_code) {
  context_functions.shader_set_vertex(&context, nr, vertex_code);
}

void loco_t::shader_set_fragment(fan::graphics::shader_nr_t nr, const std::string& fragment_code) {
  context_functions.shader_set_fragment(&context, nr, fragment_code);
}

bool loco_t::shader_compile(fan::graphics::shader_nr_t nr) {
  return context_functions.shader_compile(&context, nr);
}

fan::graphics::image_nr_t loco_t::image_create() {
  return context_functions.image_create(&context);
}

fan::graphics::context_image_t loco_t::image_get(fan::graphics::image_nr_t nr) {
  fan::graphics::context_image_t img;
  if (window.renderer == renderer_t::opengl) {
    img = *(fan::opengl::context_t::image_t*)context_functions.image_get(&context, nr);
  }
  else if (window.renderer == renderer_t::vulkan) {
    img = *(fan::vulkan::context_t::image_t*)context_functions.image_get(&context, nr);
  }
  return img;
}

uint64_t loco_t::image_get_handle(fan::graphics::image_nr_t nr) {
  return context_functions.image_get_handle(&context, nr);
}

void loco_t::image_erase(fan::graphics::image_nr_t nr) {
  context_functions.image_erase(&context, nr);
}

void loco_t::image_bind(fan::graphics::image_nr_t nr) {
  context_functions.image_bind(&context, nr);
}

void loco_t::image_unbind(fan::graphics::image_nr_t nr) {
  context_functions.image_unbind(&context, nr);
}

void loco_t::image_set_settings(const fan::graphics::image_load_properties_t& settings) {
  context_functions.image_set_settings(&context, settings);
}

fan::graphics::image_nr_t loco_t::image_load(const fan::image::image_info_t& image_info) {
  return context_functions.image_load_info(&context, image_info);
}

fan::graphics::image_nr_t loco_t::image_load(const fan::image::image_info_t& image_info, const fan::graphics::image_load_properties_t& p) {
  return context_functions.image_load_info_props(&context, image_info, p);
}

fan::graphics::image_nr_t loco_t::image_load(const fan::string& path) {
  return context_functions.image_load_path(&context, path);
}

fan::graphics::image_nr_t loco_t::image_load(const fan::string& path, const fan::graphics::image_load_properties_t& p) {
  return context_functions.image_load_path_props(&context, path, p);
}

fan::graphics::image_nr_t loco_t::image_load(fan::color* colors, const fan::vec2ui& size) {
  return context_functions.image_load_colors(&context, colors, size);
}

fan::graphics::image_nr_t loco_t::image_load(fan::color* colors, const fan::vec2ui& size, const fan::graphics::image_load_properties_t& p) {
  return context_functions.image_load_colors_props(&context, colors, size, p);
}

void loco_t::image_unload(fan::graphics::image_nr_t nr) {
  context_functions.image_unload(&context, nr);
}

fan::graphics::image_nr_t loco_t::create_missing_texture() {
  return context_functions.create_missing_texture(&context);
}

fan::graphics::image_nr_t loco_t::create_transparent_texture() {
  return context_functions.create_transparent_texture(&context);
}

void loco_t::image_reload(fan::graphics::image_nr_t nr, const fan::image::image_info_t& image_info) {
  context_functions.image_reload_image_info(&context, nr, image_info);
}
void loco_t::image_reload(fan::graphics::image_nr_t nr, const fan::image::image_info_t& image_info, const fan::graphics::image_load_properties_t& p) {
  context_functions.image_reload_image_info_props(&context, nr, image_info, p);
}
void loco_t::image_reload(fan::graphics::image_nr_t nr, const std::string& path) {
  context_functions.image_reload_path(&context, nr, path);
}
void loco_t::image_reload(fan::graphics::image_nr_t nr, const std::string& path, const fan::graphics::image_load_properties_t& p) {
  context_functions.image_reload_path_props(&context, nr, path, p);
}

fan::graphics::image_nr_t loco_t::image_create(const fan::color& color) {
  return context_functions.image_create_color(&context, color);
}

fan::graphics::image_nr_t loco_t::image_create(const fan::color& color, const fan::graphics::image_load_properties_t& p) {
  return context_functions.image_create_color_props(&context, color, p);
}

fan::graphics::camera_nr_t loco_t::camera_create() {
  return context_functions.camera_create(&context);
}

fan::graphics::context_camera_t& loco_t::camera_get(fan::graphics::camera_nr_t nr) {
  return context_functions.camera_get(&context, nr);
}

void loco_t::camera_erase(fan::graphics::camera_nr_t nr) {
  context_functions.camera_erase(&context, nr);
}

fan::graphics::camera_nr_t loco_t::camera_open(const fan::vec2& x, const fan::vec2& y) {
  return context_functions.camera_open(&context, x, y);
}

fan::vec3 loco_t::camera_get_position(fan::graphics::camera_nr_t nr) {
  return context_functions.camera_get_position(&context, nr);
}

void loco_t::camera_set_position(fan::graphics::camera_nr_t nr, const fan::vec3& cp) {
  context_functions.camera_set_position(&context, nr, cp);
}

fan::vec2 loco_t::camera_get_size(fan::graphics::camera_nr_t nr) {
  return context_functions.camera_get_size(&context, nr);
}

void loco_t::camera_set_ortho(fan::graphics::camera_nr_t nr, fan::vec2 x, fan::vec2 y) {
  context_functions.camera_set_ortho(&context, nr, x, y);
}

void loco_t::camera_set_perspective(fan::graphics::camera_nr_t nr, f32_t fov, const fan::vec2& window_size) {
  context_functions.camera_set_perspective(&context, nr, fov, window_size);
}

void loco_t::camera_rotate(fan::graphics::camera_nr_t nr, const fan::vec2& offset) {
  context_functions.camera_rotate(&context, nr, offset);
}

fan::graphics::viewport_nr_t loco_t::viewport_create() {
  return context_functions.viewport_create(&context);
}

fan::graphics::context_viewport_t& loco_t::viewport_get(fan::graphics::viewport_nr_t nr) {
  return context_functions.viewport_get(&context, nr);
}

void loco_t::viewport_erase(fan::graphics::viewport_nr_t nr) {
  context_functions.viewport_erase(&context, nr);
}

fan::vec2 loco_t::viewport_get_position(fan::graphics::viewport_nr_t nr) {
  return context_functions.viewport_get_position(&context, nr);
}

fan::vec2 loco_t::viewport_get_size(fan::graphics::viewport_nr_t nr) {
  return context_functions.viewport_get_size(&context, nr);
}

void loco_t::viewport_set(const fan::vec2& viewport_position, const fan::vec2& viewport_size, const fan::vec2& window_size) {
  context_functions.viewport_set(&context, viewport_position, viewport_size, window_size);
}

void loco_t::viewport_set(fan::graphics::viewport_nr_t nr, const fan::vec2& viewport_position, const fan::vec2& viewport_size, const fan::vec2& window_size) {
  context_functions.viewport_set_nr(&context, nr, viewport_position, viewport_size, window_size);
}

void loco_t::viewport_zero(fan::graphics::viewport_nr_t nr) {
  context_functions.viewport_zero(&context, nr);
}

bool loco_t::inside(fan::graphics::viewport_nr_t nr, const fan::vec2& position) {
  return context_functions.viewport_inside(&context, nr, position);
}

bool loco_t::inside_wir(fan::graphics::viewport_nr_t nr, const fan::vec2& position) {
  return context_functions.viewport_inside_wir(&context, nr, position);
}

uint8_t* loco_t::A_resize(void* ptr, uintptr_t size) {
  if (ptr) {
    if (size) {
      void* rptr = (void*)__generic_realloc(ptr, size);
      if (rptr == 0) {
        fan::throw_error_impl();
      }
      return (uint8_t*)rptr;
    }
    else {
      __generic_free(ptr);
      return 0;
    }
  }
  else {
    if (size) {
      void* rptr = (void*)__generic_malloc(size);
      if (rptr == 0) {
        fan::throw_error_impl();
      }
      return (uint8_t*)rptr;
    }
    else {
      return 0;
    }
  }
}

void loco_t::use() {
  gloco = this;
}

void loco_t::camera_move(fan::graphics::context_camera_t& camera, f64_t dt, f32_t movement_speed, f32_t friction) {
  camera.velocity /= friction * dt + 1;
  static constexpr auto minimum_velocity = 0.001;
  static constexpr f32_t camera_rotate_speed = 100;
  if (camera.velocity.x < minimum_velocity && camera.velocity.x > -minimum_velocity) {
    camera.velocity.x = 0;
  }
  if (camera.velocity.y < minimum_velocity && camera.velocity.y > -minimum_velocity) {
    camera.velocity.y = 0;
  }
  if (camera.velocity.z < minimum_velocity && camera.velocity.z > -minimum_velocity) {
    camera.velocity.z = 0;
  }

  f64_t msd = (movement_speed * dt);
  if (gloco->window.key_pressed(fan::input::key_w)) {
    camera.velocity += camera.m_front * msd;
  }
  if (gloco->window.key_pressed(fan::input::key_s)) {
    camera.velocity -= camera.m_front * msd;
  }
  if (gloco->window.key_pressed(fan::input::key_a)) {
    camera.velocity -= camera.m_right * msd;
  }
  if (gloco->window.key_pressed(fan::input::key_d)) {
    camera.velocity += camera.m_right * msd;
  }

  if (gloco->window.key_pressed(fan::input::key_space)) {
    camera.velocity.y += movement_speed * gloco->delta_time;
  }
  if (gloco->window.key_pressed(fan::input::key_left_shift)) {
    camera.velocity.y -= movement_speed * gloco->delta_time;
  }

  f64_t rotate = camera.sensitivity * camera_rotate_speed * gloco->delta_time;
  if (gloco->window.key_pressed(fan::input::key_left)) {
    camera.set_yaw(camera.get_yaw() - rotate);
  }
  if (gloco->window.key_pressed(fan::input::key_right)) {
    camera.set_yaw(camera.get_yaw() + rotate);
  }
  if (gloco->window.key_pressed(fan::input::key_up)) {
    camera.set_pitch(camera.get_pitch() + rotate);
  }
  if (gloco->window.key_pressed(fan::input::key_down)) {
    camera.set_pitch(camera.get_pitch() - rotate);
  }

  camera.position += camera.velocity * gloco->delta_time;
  camera.update_view();

  camera.m_view = camera.get_view_matrix();
}

#define shaper_get_key_safe(return_type, kps_type, variable) \
  [KeyPack] ()-> auto& { \
    auto o = gloco->shaper.GetKeyOffset( \
      offsetof(kps_t::CONCAT(_, kps_type), variable), \
      offsetof(kps_t::kps_type, variable) \
    );\
    static_assert(std::is_same_v<decltype(kps_t::kps_type::variable), return_type>, "possibly unwanted behaviour"); \
    return *(return_type*)&KeyPack[o];\
  }()

template <typename T>
loco_t::functions_t loco_t::get_functions() {
  functions_t funcs{
    .get_position = [](shape_t* shape) {
      if constexpr (fan_has_variable(T, position)) {
        return get_render_data(shape, &T::position);
      }
      else if constexpr (fan_has_variable(T, vertices)) {
        return get_render_data(shape, &T::vertices)[0].offset;
      }
      else {
        if (shape->get_shape_type() == shape_type_t::polygon) {
          auto ri = (polygon_t::ri_t*)shape->GetData(gloco->shaper);
          fan::vec3 position = 0;
          fan::opengl::core::get_glbuffer(
            gloco->context.gl,
            &position,
            ri->vbo.m_buffer,
            sizeof(position),
            sizeof(polygon_vertex_t) * 0 + fan::member_offset(&polygon_vertex_t::offset),
            ri->vbo.m_target
          );
          return position;
        }
        else {
          fan::throw_error("unimplemented get - for line use get_src()");
        }
        
        return fan::vec3();
      }
    },
    .set_position2 = [](shape_t* shape, const fan::vec2& position) {
      if constexpr (fan_has_variable(T, position)) {
        modify_render_data_element(shape, &T::position, position);
      }
      else {
        if (shape->get_shape_type() == shape_type_t::polygon) {
          auto ri = (polygon_t::ri_t*)shape->GetData(gloco->shaper);
          ri->vao.bind(gloco->context.gl);
          ri->vbo.bind(gloco->context.gl);
          uint32_t vertex_count = ri->buffer_size / sizeof(polygon_vertex_t);
          for (uint32_t i = 0; i < vertex_count; ++i) {
            fan::opengl::core::edit_glbuffer(
              gloco->context.gl, 
              ri->vbo.m_buffer, 
              &position, 
              sizeof(polygon_vertex_t) * i + fan::member_offset(&polygon_vertex_t::offset),
              sizeof(position),
              ri->vbo.m_target
            );
          }
        }
        else {
          fan::throw_error("unimplemented set - for line use set_src()");
        }
      }
    },
    .set_position3 = [](shape_t* shape, const fan::vec3& position) {
      if constexpr (fan_has_variable(T, position)) {
          auto sti = gloco->shaper.ShapeList[*shape].sti;

          // alloc can be avoided inside switch
          auto KeyPackSize = gloco->shaper.GetKeysSize(*shape);
          uint8_t* KeyPack = new uint8_t[KeyPackSize];
          gloco->shaper.WriteKeys(*shape, KeyPack);

          switch (sti) {       
          case loco_t::shape_type_t::light: {
            break;
          }
          // common
          case loco_t::shape_type_t::capsule:
          case loco_t::shape_type_t::gradient:
          case loco_t::shape_type_t::grid:
          case loco_t::shape_type_t::circle:
          case loco_t::shape_type_t::rectangle:
          case loco_t::shape_type_t::rectangle3d:
          case loco_t::shape_type_t::line: {
            shaper_get_key_safe(depth_t, common_t, depth) = position.z;
            break;
          }
                                          // texture
          case loco_t::shape_type_t::particles:
          case loco_t::shape_type_t::universal_image_renderer:
          case loco_t::shape_type_t::unlit_sprite:
          case loco_t::shape_type_t::sprite: {
            shaper_get_key_safe(depth_t, texture_t, depth) = position.z;
            break;
          }
          default: {
            fan::throw_error("unimplemented");
          }
          }

    
          auto _vi = shape->GetRenderData(gloco->shaper);
          auto vlen = gloco->shaper.GetRenderDataSize(sti);
          uint8_t* vi = new uint8_t[vlen];
          std::memcpy(vi, _vi, vlen);
          ((T*)vi)->position = position;

          auto _ri = shape->GetData(gloco->shaper);
          auto rlen = gloco->shaper.GetDataSize(sti);
          uint8_t* ri = new uint8_t[rlen];
          std::memcpy(ri, _ri, rlen);

          shape->remove();
          *shape = gloco->shaper.add(
            sti,
            KeyPack,
            KeyPackSize,
            vi,
            ri
          );
#if defined(debug_shape_t)
          fan::print("+", shape->NRI);
#endif
          delete[] KeyPack;
          delete[] vi;
          delete[] ri;
      }
      else {
        fan::throw_error("unimplemented set - for line use set_src()");
      }
      },
      .get_size = [](shape_t* shape) {
        if constexpr (fan_has_variable(T, size)) {
          if constexpr (sizeof(T::size) == sizeof(fan::vec2)) {
            return get_render_data(shape, &T::size);
          }
          else {
            fan::throw_error("unimplemented get");
            return fan::vec2();
          }
        }
        else if constexpr (fan_has_variable(T, radius)) {
          return fan::vec2(get_render_data(shape, &T::radius));
        }
        else {
          fan::throw_error("unimplemented get");
          return fan::vec2();
        }
      },
      .get_size3 = [](shape_t* shape) {
        if constexpr (fan_has_variable(T, size)) {
          if constexpr (sizeof(T::size) == sizeof(fan::vec3)) {
            return get_render_data(shape, &T::size);
          }
          else {
            fan::throw_error("unimplemented get");
            return fan::vec3();
          }
        }
        else if constexpr (fan_has_variable(T, radius)) {
          return fan::vec3(get_render_data(shape, &T::radius));
        }
        else {
          fan::throw_error("unimplemented get");
          return fan::vec3();
        }
      },
      .set_size = [](shape_t* shape, const fan::vec2& size) {
        if constexpr (fan_has_variable(T, size)) {
          modify_render_data_element(shape, &T::size, size);
        }
        else if constexpr (fan_has_variable(T, radius)) {
          modify_render_data_element(shape, &T::radius, size.x);
        }
        else {
          fan::throw_error("unimplemented set");
        }
      },
      .set_size3 = [](shape_t* shape, const fan::vec3& size) {
        if constexpr (fan_has_variable(T, size)) {
          modify_render_data_element(shape, &T::size, size);
        }
        else if constexpr (fan_has_variable(T, radius)) {
          modify_render_data_element(shape, &T::radius, size.x);
        }
        else {
          fan::throw_error("unimplemented set");
        }
      },
      .get_rotation_point = [](shape_t* shape) {
        if constexpr (fan_has_variable(T, rotation_point)) {
          return get_render_data(shape, &T::rotation_point);
        }
        else {
          fan::throw_error("unimplemented get");
          return fan::vec2();
        }
      },
      .set_rotation_point = [](shape_t* shape, const fan::vec2& rotation_point) {
        if constexpr (fan_has_variable(T, rotation_point)) {
          modify_render_data_element(shape, &T::rotation_point, rotation_point);
        }
        else if constexpr (fan_has_variable(T, vertices)) {
          
        }
        else {
          fan::throw_error("unimplemented set");
        }
            },
      .get_color = [](shape_t* shape) -> fan::color{
        if constexpr (fan_has_variable(T, color)) {
          return *(fan::color*)&get_render_data(shape, &T::color);
        }
        else {
          fan::throw_error("unimplemented get");
          return fan::color();
        }
      },
      .set_color = [](shape_t* shape, const fan::color& color) {
        if constexpr (fan_has_variable(T, color)) {
          if constexpr (!std::is_same_v<T, loco_t::gradient_t::vi_t>) {
            modify_render_data_element(shape, &T::color, color);
          }
        }
        else {
          fan::throw_error("unimplemented set");
        }
      },
      .get_angle = [](shape_t* shape) {
        if constexpr (fan_has_variable(T, angle)) {
          return get_render_data(shape, &T::angle);
        }
        else {
          fan::throw_error("unimplemented get");
          return fan::vec3();
        }
      },
      .set_angle = [](shape_t* shape, const fan::vec3& angle) {
        if constexpr (fan_has_variable(T, angle)) {
          modify_render_data_element(shape, &T::angle, angle);
        }
        else {
          if (shape->get_shape_type() == shape_type_t::polygon) {
            auto ri = (polygon_t::ri_t*)shape->GetData(gloco->shaper);
            ri->vao.bind(gloco->context.gl);
            ri->vbo.bind(gloco->context.gl);
            uint32_t vertex_count = ri->buffer_size / sizeof(polygon_vertex_t);
            for (uint32_t i = 0; i < vertex_count; ++i) {
              fan::opengl::core::edit_glbuffer(
                gloco->context.gl,
                ri->vbo.m_buffer,
                &angle,
                sizeof(polygon_vertex_t) * i + fan::member_offset(&polygon_vertex_t::angle),
                sizeof(angle),
                ri->vbo.m_target
              );
            }
          }
          else {
            fan::throw_error("unimplemented set");
          }
        }
      },
      .get_tc_position = [](shape_t* shape) {
        if constexpr (fan_has_variable(T, tc_position)) {
          return get_render_data(shape, &T::tc_position);
        }
        else {
          fan::throw_error("unimplemented get");
          return fan::vec2();
        }
      },
      .set_tc_position = [](shape_t* shape, const fan::vec2& tc_position) {
        if constexpr (fan_has_variable(T, tc_position)) {
          modify_render_data_element(shape, &T::tc_position, tc_position);
        }
        else {
          fan::throw_error("unimplemented set");
        }
      },
      .get_tc_size = [](shape_t* shape) {
        if constexpr (fan_has_variable(T, tc_size)) {
          return get_render_data(shape, &T::tc_size);
        }
        else {
          fan::throw_error("unimplemented get");
          return fan::vec2();
        }
      },
      .set_tc_size = [](shape_t* shape, const fan::vec2& tc_size) {
        if constexpr (fan_has_variable(T, tc_size)) {
          modify_render_data_element(shape, &T::tc_size, tc_size);
        }
        else {
          fan::throw_error("unimplemented set");
        }
      },
      .load_tp = [](shape_t* shape, loco_t::texturepack_t::ti_t* ti) -> bool {
        if constexpr(std::is_same_v<T, loco_t::sprite_t::vi_t> ||
        std::is_same_v<T, loco_t::unlit_sprite_t::vi_t>) {
          auto sti = gloco->shaper.ShapeList[*shape].sti;
            
          auto KeyPackSize = gloco->shaper.GetKeysSize(*shape);
          uint8_t* KeyPack = new uint8_t[KeyPackSize];
          gloco->shaper.WriteKeys(*shape, KeyPack);
          switch (sti) {
            // texture
            case loco_t::shape_type_t::particles:
            case loco_t::shape_type_t::universal_image_renderer:
            case loco_t::shape_type_t::unlit_sprite:
            case loco_t::shape_type_t::sprite: {
              shaper_get_key_safe(image_t, texture_t, image) = *ti->image;
              break;
            }
            default: {
              fan::throw_error("unimplemented");
            }
          }

          auto& im = *ti->image;
          auto img = gloco->image_get(im);

          auto _vi = shape->GetRenderData(gloco->shaper);
          auto vlen = gloco->shaper.GetRenderDataSize(sti);
          uint8_t* vi = new uint8_t[vlen];
          std::memcpy(vi, _vi, vlen);
          std::visit([vi, ti] (const auto& v) {
            ((T*)vi)->tc_position = ti->position / v.size;
            ((T*)vi)->tc_size = ti->size / v.size;
          }, img);

          auto _ri = shape->GetData(gloco->shaper);
          auto rlen = gloco->shaper.GetDataSize(sti);
          uint8_t* ri = new uint8_t[rlen];
          std::memcpy(ri, _ri, rlen);

          shape->remove();
          *shape = gloco->shaper.add(
            sti,
            KeyPack,
            KeyPackSize,
            vi,
            ri
          );
#if defined(debug_shape_t)
          fan::print("+", shape->NRI);
#endif
          delete[] KeyPack;
          delete[] vi;
          delete[] ri;
          }
        return 0;
      },
      .get_grid_size = [](shape_t* shape) {
        if constexpr (fan_has_variable(T, grid_size)) {
          return get_render_data(shape, &T::grid_size);
        }
        else {
          fan::throw_error("unimplemented get");
          return fan::vec2();
        }
      },
      .set_grid_size = [](shape_t* shape, const fan::vec2& grid_size) {
        if constexpr (fan_has_variable(T, grid_size)) {
          modify_render_data_element(shape, &T::grid_size, grid_size);
        }
        else {
          fan::throw_error("unimplemented set");
        }
      },
      .get_camera = [](shape_t* shape) {
        auto sti = gloco->shaper.ShapeList[*shape].sti;

        // alloc can be avoided inside switch
        uint8_t* KeyPack = gloco->shaper.GetKeys(*shape);

        switch (sti) {
          // light
        case loco_t::shape_type_t::light: {
          return shaper_get_key_safe(camera_t, light_t, camera);
        }
                                        // common
        case loco_t::shape_type_t::capsule:
        case loco_t::shape_type_t::gradient:
        case loco_t::shape_type_t::grid:
        case loco_t::shape_type_t::circle:
        case loco_t::shape_type_t::rectangle:
        case loco_t::shape_type_t::line: {
          return shaper_get_key_safe(camera_t, common_t, camera);
        }
                                        // texture
        case loco_t::shape_type_t::particles:
        case loco_t::shape_type_t::universal_image_renderer:
        case loco_t::shape_type_t::unlit_sprite:
        case loco_t::shape_type_t::sprite: {
          return shaper_get_key_safe(camera_t, texture_t, camera);
        }
        default: {
          fan::throw_error("unimplemented");
        }
        }
        return loco_t::camera_t();
      },
      .set_camera = [](shape_t* shape, loco_t::camera_t camera) {
        {
            auto sti = gloco->shaper.ShapeList[*shape].sti;

          // alloc can be avoided inside switch
          auto KeyPackSize = gloco->shaper.GetKeysSize(*shape);
          uint8_t* KeyPack = new uint8_t[KeyPackSize];
          gloco->shaper.WriteKeys(*shape, KeyPack);
          switch(sti) {
            // light
            case loco_t::shape_type_t::light: {
              shaper_get_key_safe(camera_t, light_t, camera) = camera;
              break;
            }
            // common
            case loco_t::shape_type_t::capsule:
            case loco_t::shape_type_t::gradient:
            case loco_t::shape_type_t::grid:
            case loco_t::shape_type_t::circle:
            case loco_t::shape_type_t::rectangle:
            case loco_t::shape_type_t::rectangle3d:
            case loco_t::shape_type_t::line: {
              shaper_get_key_safe(camera_t, common_t, camera) = camera;
              break;
            }
            // texture
            case loco_t::shape_type_t::particles:
            case loco_t::shape_type_t::universal_image_renderer:
            case loco_t::shape_type_t::unlit_sprite:
            case loco_t::shape_type_t::sprite: {
              shaper_get_key_safe(camera_t, texture_t, camera) = camera;
              break;
            }
            default: {
              fan::throw_error("unimplemented");
            }
          }

          auto _vi = shape->GetRenderData(gloco->shaper);
          auto vlen = gloco->shaper.GetRenderDataSize(sti);
          uint8_t* vi = new uint8_t[vlen];
          std::memcpy(vi, _vi, vlen);

          auto _ri = shape->GetData(gloco->shaper);
          auto rlen = gloco->shaper.GetDataSize(sti);
          uint8_t* ri = new uint8_t[rlen];
          std::memcpy(ri, _ri, rlen);

          shape->remove();
          *shape = gloco->shaper.add(
            sti,
            KeyPack,
            KeyPackSize,
            vi,
            ri
          );
#if defined(debug_shape_t)
          fan::print("+", shape->NRI);
#endif
          delete[] KeyPack;
          delete[] vi;
          delete[] ri;
        }
      },
      .get_viewport = [](shape_t* shape) {
        uint8_t* KeyPack = gloco->shaper.GetKeys(*shape);

        auto sti = gloco->shaper.ShapeList[*shape].sti;

        switch(sti) {
          // light
          case loco_t::shape_type_t::light: {
            return shaper_get_key_safe(viewport_t, light_t, viewport);
          }
          // common
          case loco_t::shape_type_t::capsule:
          case loco_t::shape_type_t::gradient:
          case loco_t::shape_type_t::grid:
          case loco_t::shape_type_t::circle:
          case loco_t::shape_type_t::rectangle:
          case loco_t::shape_type_t::line: {
            return shaper_get_key_safe(viewport_t, common_t, viewport);
          }
          // texture
          case loco_t::shape_type_t::particles:
          case loco_t::shape_type_t::universal_image_renderer:
          case loco_t::shape_type_t::unlit_sprite:
          case loco_t::shape_type_t::sprite: {
            return shaper_get_key_safe(viewport_t, texture_t, viewport);
          }
          default: {
            fan::throw_error("unimplemented");
          }
        }
        return loco_t::viewport_t();
      },
      .set_viewport = [](shape_t* shape, loco_t::viewport_t viewport) {
        {
          auto sti = gloco->shaper.ShapeList[*shape].sti;

          // alloc can be avoided inside switch
          auto KeyPackSize = gloco->shaper.GetKeysSize(*shape);
          uint8_t* KeyPack = new uint8_t[KeyPackSize];
          gloco->shaper.WriteKeys(*shape, KeyPack);
          switch(sti) {
            // light
            case loco_t::shape_type_t::light: {
              shaper_get_key_safe(viewport_t, light_t, viewport) = viewport;
              break;
            }
            // common
            case loco_t::shape_type_t::capsule:
            case loco_t::shape_type_t::gradient:
            case loco_t::shape_type_t::grid:
            case loco_t::shape_type_t::circle:
            case loco_t::shape_type_t::rectangle:
            case loco_t::shape_type_t::line: {
              shaper_get_key_safe(viewport_t, common_t, viewport) = viewport;
              break;
            }
            // texture
            case loco_t::shape_type_t::particles:
            case loco_t::shape_type_t::universal_image_renderer:
            case loco_t::shape_type_t::unlit_sprite:
            case loco_t::shape_type_t::sprite: {
              shaper_get_key_safe(viewport_t, texture_t, viewport) = viewport;
              break;
            }
            default: {
              fan::throw_error("unimplemented");
            }
          }

          auto _vi = shape->GetRenderData(gloco->shaper);
          auto vlen = gloco->shaper.GetRenderDataSize(sti);
          uint8_t* vi = new uint8_t[vlen];
          std::memcpy(vi, _vi, vlen);

          auto _ri = shape->GetData(gloco->shaper);
          auto rlen = gloco->shaper.GetDataSize(sti);
          uint8_t* ri = new uint8_t[rlen];
          std::memcpy(ri, _ri, rlen);

          shape->remove();
          *shape = gloco->shaper.add(
            sti,
            KeyPack,
            KeyPackSize,
            vi,
            ri
          );
#if defined(debug_shape_t)
          fan::print("+", shape->NRI);
#endif
          delete[] KeyPack;
          delete[] vi;
          delete[] ri;
        }
      },

      .get_image = [](shape_t* shape) -> loco_t::image_t {
        auto sti = gloco->shaper.ShapeList[*shape].sti;
        uint8_t* KeyPack = gloco->shaper.GetKeys(*shape);
        switch (sti) {
        // texture
        case loco_t::shape_type_t::particles:
        case loco_t::shape_type_t::universal_image_renderer:
        case loco_t::shape_type_t::unlit_sprite:
        case loco_t::shape_type_t::sprite: {
          return shaper_get_key_safe(image_t, texture_t, image);
        }
        default: {
          fan::throw_error("unimplemented");
        }
        }
        return loco_t::image_t();
      },
      .set_image = [](shape_t* shape, loco_t::image_t image) {
         
        auto sti = gloco->shaper.ShapeList[*shape].sti;

        // alloc can be avoided inside switch
        auto KeyPackSize = gloco->shaper.GetKeysSize(*shape);
        uint8_t* KeyPack = new uint8_t[KeyPackSize];
        gloco->shaper.WriteKeys(*shape, KeyPack);
        switch (sti) {
        // texture
        case loco_t::shape_type_t::particles:
        case loco_t::shape_type_t::universal_image_renderer:
        case loco_t::shape_type_t::unlit_sprite:
        case loco_t::shape_type_t::sprite: 
        case loco_t::shape_type_t::shader_shape:
        {
          shaper_get_key_safe(image_t, texture_t, image) = image;
          break;
        }
        default: {
          fan::throw_error("unimplemented");
        }
        }
            
        auto _vi = shape->GetRenderData(gloco->shaper);
        auto vlen = gloco->shaper.GetRenderDataSize(sti);
        uint8_t* vi = new uint8_t[vlen];
        std::memcpy(vi, _vi, vlen);

        auto _ri = shape->GetData(gloco->shaper);
        auto rlen = gloco->shaper.GetDataSize(sti);
        uint8_t* ri = new uint8_t[rlen];
        std::memcpy(ri, _ri, rlen);

        shape->remove();
        *shape = gloco->shaper.add(
          sti,
          KeyPack,
          KeyPackSize,
          vi,
          ri
        );
#if defined(debug_shape_t)
        fan::print("+", shape->NRI);
#endif
        delete[] KeyPack;
        delete[] vi;
        delete[] ri;
      },
      .get_parallax_factor = [](shape_t* shape) {
        if constexpr (fan_has_variable(T, parallax_factor)) {
          return get_render_data(shape, &T::parallax_factor);
        }
        else {
          fan::throw_error("unimplemented get");
          return 0.0f;
        }
      },
      .set_parallax_factor = [](shape_t* shape, f32_t parallax_factor) {
        if constexpr (fan_has_variable(T, parallax_factor)) {
          modify_render_data_element(shape, &T::parallax_factor, parallax_factor);
        }
        else {
          fan::throw_error("unimplemented set");
        }
      },
      .get_rotation_vector = [](shape_t* shape) {
        if constexpr (fan_has_variable(T, rotation_vector)) {
          return get_render_data(shape, &T::rotation_vector);
        }
        else {
          fan::throw_error("unimplemented get");
          return fan::vec3();
        }
      },
      .get_flags = [](shape_t* shape) -> uint32_t {
        if constexpr (fan_has_variable(T, flags)) {
          return get_render_data(shape, &T::flags);
        }
        else {
          fan::throw_error("unimplemented get");
          return 0;
        }
      },
      .set_flags = [](shape_t* shape, uint32_t flags) {
        if constexpr (fan_has_variable(T, flags)) {
          modify_render_data_element(shape, &T::flags, flags);
        }
        else {
          fan::throw_error("unimplemented set");
        }
      },
      .get_radius = [](shape_t* shape) {
        if constexpr (fan_has_variable(T, radius)) {
          return get_render_data(shape, &T::radius);
        }
        else {
          fan::throw_error("unimplemented get");
          return 0.0f;
        }
      },
      .get_src = [](shape_t* shape) {
        if constexpr (fan_has_variable(T, src)) {
          return get_render_data(shape, &T::src);
        }
        else {
          fan::throw_error("unimplemented get");
          return fan::vec3();
        }
      },
      .get_dst = [](shape_t* shape) {
        if constexpr (fan_has_variable(T, dst)) {
          return get_render_data(shape, &T::dst);
        }
        else {
          fan::throw_error("unimplemented get");
          return fan::vec3();
        }
      },
      .get_outline_size = [](shape_t* shape) {
        if constexpr (fan_has_variable(T, outline_size)) {
          return get_render_data(shape, &T::outline_size);
        }
        else {
          fan::throw_error("unimplemented get");
          return 0.0f;
        }
      },
      .get_outline_color = [](shape_t* shape) {
        if constexpr (fan_has_variable(T, outline_color)) {
          return get_render_data(shape, &T::outline_color);
        }
        else {
          fan::throw_error("unimplemented get");
          return fan::color();
        }
      },
      .reload = [](shape_t* shape, uint8_t format, void** image_data, const fan::vec2& image_size, uint32_t filter) {
        if (shape->get_shape_type() != loco_t::shape_type_t::universal_image_renderer) {
          fan::throw_error("only meant to be used with universal_image_renderer");
        }
        loco_t::universal_image_renderer_t::ri_t& ri = *(loco_t::universal_image_renderer_t::ri_t*)shape->GetData(gloco->shaper);
        if (format != ri.format) {
          auto sti = gloco->shaper.ShapeList[*shape].sti;
          uint8_t* KeyPack = gloco->shaper.GetKeys(*shape);
          loco_t::image_t vi_image = shaper_get_key_safe(loco_t::image_t, texture_t, image);


          auto shader = gloco->shaper.GetShader(sti);
          gloco->shader_set_vertex(
            shader,
            read_shader("shaders/opengl/2D/objects/pixel_format_renderer.vs")
          );
          {
            fan::string fs;
            switch(format) {
              case fan::pixel_format::yuv420p: {
                fs = read_shader("shaders/opengl/2D/objects/yuv420p.fs");
                break;
              }
              case fan::pixel_format::nv12: {
                fs = read_shader("shaders/opengl/2D/objects/nv12.fs");
                break;
              }
              default: {
                fan::throw_error("unimplemented format");
              }
            }
            gloco->shader_set_fragment(shader, fs);
            gloco->shader_compile(shader);
          }

          uint8_t image_count_old = fan::pixel_format::get_texture_amount(ri.format);
          uint8_t image_count_new = fan::pixel_format::get_texture_amount(format);
          if (image_count_new < image_count_old) {
            // -1 ? 
            for (uint32_t i = image_count_old - 1; i > image_count_new; --i) {
              if (i == 0) {
                gloco->image_erase(vi_image);
              }
              else {
                gloco->image_erase(ri.images_rest[i - 1]);
              }
            }
          }
          else if (image_count_new > image_count_old) {
            loco_t::image_t images[4];
            for (uint32_t i = image_count_old; i < image_count_new; ++i) {
              images[i] = gloco->image_create();
            }
            shape->set_image(images[0]);
            std::copy(&images[1], &images[0] + ri.images_rest.size(), ri.images_rest.data());
          }
        }

        auto vi_image = shape->get_image();

        uint8_t image_count_new = fan::pixel_format::get_texture_amount(format);
        for (uint32_t i = 0; i < image_count_new; i++) {
          fan::image::image_info_t image_info;
          image_info.data = image_data[i];
          image_info.size = fan::pixel_format::get_image_sizes(format, image_size)[i];
          auto lp = fan::pixel_format::get_image_properties<loco_t::image_load_properties_t>(format)[i];
          lp.min_filter = filter;
          lp.mag_filter = filter;
          if (i == 0) {
            gloco->image_reload(
              vi_image,
              image_info,
              lp
            );
          }
          else {
            gloco->image_reload(
              ri.images_rest[i - 1],
              image_info,
              lp
            );
          }
        }
        ri.format = format;
      },
      .draw = [](uint8_t draw_range) {
        // Implement draw function
      },
      .set_line = [](shape_t* shape, const fan::vec2& src, const fan::vec2& dst) {
        if constexpr (fan_has_variable(T, src) && fan_has_variable(T, dst)) {
          modify_render_data_element(shape, &T::src, src);
          modify_render_data_element(shape, &T::dst, dst);
        }
        else {
          fan::throw_error("unimplemented set");
        }
      },
      .set_line3 = [](shape_t* shape, const fan::vec3& src, const fan::vec3& dst) {
        if constexpr (fan_has_variable(T, src) && fan_has_variable(T, dst)) {
          modify_render_data_element(shape, &T::src, src);
          modify_render_data_element(shape, &T::dst, dst);
        }
        else {
          fan::throw_error("unimplemented set");
        }
      }
    };
  return funcs;
}

#undef shaper_get_key_safe

void loco_t::close() {
  window.close();
}

void generate_commands(loco_t* loco) {
#if defined(loco_imgui)
  loco->console.open();

  loco->console.commands.add("echo", [](const fan::commands_t::arg_t& args) {
    fan::commands_t::output_t out;
    out.text = fan::append_args(args) + "\n";
    out.highlight = fan::graphics::highlight_e::info;
    gloco->console.commands.output_cb(out);
  }).description = "prints something - usage echo [args]";

  loco->console.commands.add("help", [](const fan::commands_t::arg_t& args) {
    if (args.empty()) {
      fan::commands_t::output_t out;
      out.highlight = fan::graphics::highlight_e::info;
      std::string out_str;
      out_str += "{\n";
      for (const auto& i : gloco->console.commands.func_table) {
        out_str += "\t" + i.first + ",\n";
      }
      out_str += "}\n";
      out.text = out_str;
      gloco->console.commands.output_cb(out);
      return;
    }
    else if (args.size() == 1) {
      auto found = gloco->console.commands.func_table.find(args[0]);
      if (found == gloco->console.commands.func_table.end()) {
        gloco->console.commands.print_command_not_found(args[0]);
        return;
      }
      fan::commands_t::output_t out;
      out.text = found->second.description + "\n";
      out.highlight = fan::graphics::highlight_e::info;
      gloco->console.commands.output_cb(out);
    }
    else {
      gloco->console.commands.print_invalid_arg_count();
    }
  }).description = "get info about specific command - usage help command";

  loco->console.commands.add("list", [](const fan::commands_t::arg_t& args) {
    std::string out_str;
    for (const auto& i : gloco->console.commands.func_table) {
      out_str += i.first + "\n";
    }

    fan::commands_t::output_t out;
    out.text = out_str;
    out.highlight = fan::graphics::highlight_e::info;

    gloco->console.commands.output_cb(out);
  }).description = "lists all commands - usage list";

  loco->console.commands.add("alias", [](const fan::commands_t::arg_t& args) {
    if (args.size() < 2 || args[1].empty()) {
      gloco->console.commands.print_invalid_arg_count();
      return;
    }
    if (gloco->console.commands.insert_to_command_chain(args)) {
      return;
    }
    gloco->console.commands.func_table[args[0]] = gloco->console.commands.func_table[args[1]];
  }).description = "can create alias commands - usage alias [cmd name] [cmd]";


  loco->console.commands.add("show_fps", [](const fan::commands_t::arg_t& args) {
    if (args.size() != 1) {
      gloco->console.commands.print_invalid_arg_count();
      return;
    }
    gloco->toggle_fps = std::stoi(args[0]);
  }).description = "toggles fps - usage show_fps [value]";

  loco->console.commands.add("quit", [](const fan::commands_t::arg_t& args) {
    exit(0);
  }).description = "quits program - usage quit";

  loco->console.commands.add("clear", [](const fan::commands_t::arg_t& args) {
    gloco->console.output_buffer.clear();
    gloco->console.editor.SetText("");
  }).description = "clears output buffer - usage clear";

  loco->console.commands.add("set_gamma", [](const fan::commands_t::arg_t& args) {
    if (args.size() != 1) {
      gloco->console.commands.print_invalid_arg_count();
      return;
    }
    gloco->shader_set_value(gloco->gl.m_fbo_final_shader, "gamma", std::stof(args[0]));
  }).description = "sets gamma for postprocessing shader";

  loco->console.commands.add("set_gamma", [](const fan::commands_t::arg_t& args) {
    if (args.size() != 1) {
      gloco->console.commands.print_invalid_arg_count();
      return;
    }
    gloco->shader_set_value(gloco->gl.m_fbo_final_shader, "gamma", std::stof(args[0]));
  }).description = "sets gamma for postprocessing shader";

  loco->console.commands.add("set_exposure", [](const fan::commands_t::arg_t& args) {
    if (args.size() != 1) {
      gloco->console.commands.print_invalid_arg_count();
      return;
    }
    gloco->shader_set_value(gloco->gl.m_fbo_final_shader, "exposure", std::stof(args[0]));
  }).description = "sets exposure for postprocessing shader";

  loco->console.commands.add("set_bloom_strength", [](const fan::commands_t::arg_t& args) {
    if (args.size() != 1) {
      gloco->console.commands.print_invalid_arg_count();
      return;
    }
    gloco->shader_set_value(gloco->gl.m_fbo_final_shader, "bloom_strength", std::stof(args[0]));
  }).description = "sets bloom strength for postprocessing shader";

  loco->console.commands.add("set_vsync", [](const fan::commands_t::arg_t& args) {
    if (args.size() != 1) {
      gloco->console.commands.print_invalid_arg_count();
      return;
    }
    gloco->set_vsync(std::stoi(args[0]));
    }).description = "sets vsync";

  loco->console.commands.add("set_target_fps", [](const fan::commands_t::arg_t& args) {
    if (args.size() != 1) {
      gloco->console.commands.print_invalid_arg_count();
      return;
    }
    gloco->set_target_fps(std::stoi(args[0]));
  }).description = "sets target fps";

  loco->console.commands.add("debug_memory", [loco, nr = fan::console_t::frame_cb_t::nr_t()](const fan::commands_t::arg_t& args) mutable {
    if (args.size() != 1) {
      loco->console.commands.print_invalid_arg_count();
      return;
    }
    if (nr.iic() && std::stoi(args[0])) {
      nr = loco->console.push_frame_process([] {
         ImGui::SetNextWindowBgAlpha(0.9f);
          static int init = 0;
          ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoFocusOnAppearing ;
          if (init == 0) {
            ImGui::SetNextWindowSize(fan::vec2(600, 300));
            //window_flags |= ImGuiWindowFlags_AlwaysAutoResize;
            init = 1;
          }
        ImGui::Begin("fan_memory_dbg_wnd", 0, window_flags);
        fan::graphics::render_allocations_plot();
        ImGui::End();
      });
    }
    else if (!nr.iic() && !std::stoi(args[0])){
      loco->console.erase_frame_process(nr);
    }
  }).description = "opens memory debug window";

  /*loco->console.commands.add("console_transparency", [](const fan::commands_t::arg_t& args) {
    if (args.size() != 1) {
      gloco->console.commands.print_invalid_arg_count();
      return;
    }
    gloco->console.transparency = std::stoull(args[0]);
    for (int i = 0; i < 21; ++i) {
      (gloco->console.editor.GetPalette().data() + i = gloco->console.transparency;
    }
    }).description = "";*/

#endif
}

#if defined(loco_imgui)
void loco_t::load_fonts(auto& fonts, ImGuiIO& io, const std::string& name, f32_t font_size) {
  for (std::size_t i = 0; i < std::size(fonts); ++i) {
    fonts[i] = io.Fonts->AddFontFromFileTTF(name.c_str(), (int)(font_size * (1 << i)) * 1.5);

    if (fonts[i] == nullptr) {
      fan::throw_error(fan::string("failed to load font:") + name);
    }
  }
  io.Fonts->Build();
}
#endif

void check_vk_result(VkResult err) {
  if (err != VK_SUCCESS) {
    fan::print("vkerr", (int)err);
  }
}

#if defined(loco_imgui)
void loco_t::init_imgui() {
  ImGui::CreateContext();
  ImPlot::CreateContext();
  auto& input_map = ImPlot::GetInputMap();
  input_map.Pan = ImGuiMouseButton_Middle;

  ImGuiIO& io = ImGui::GetIO(); (void)io;
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
  ///    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

  ImGuiStyle& style = ImGui::GetStyle();
  if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
  {
    style.WindowRounding = 0.;
  }
  style.FrameRounding = 5.f;
  style.FramePadding = ImVec2(12.f, 5.f);
  style.Colors[ImGuiCol_WindowBg].w = 1.0f;

  imgui_themes::dark();

  if (window.renderer == renderer_t::opengl) {
    glfwMakeContextCurrent(window);
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    const char* glsl_version = "#version 120";
    ImGui_ImplOpenGL3_Init(glsl_version);
  }
  else if (window.renderer == renderer_t::vulkan) {
    ImGui_ImplGlfw_InitForVulkan(window, true);
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = context.vk.instance;
    init_info.PhysicalDevice = context.vk.physical_device;
    init_info.Device = context.vk.device;
    init_info.QueueFamily = context.vk.queue_family;
    init_info.Queue = context.vk.graphics_queue;
    init_info.DescriptorPool = context.vk.descriptor_pool.m_descriptor_pool;
    init_info.RenderPass = context.vk.MainWindowData.RenderPass;
    init_info.Subpass = 0;
    init_info.MinImageCount = context.vk.MinImageCount;
    init_info.ImageCount = context.vk.MainWindowData.ImageCount;
    init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    init_info.CheckVkResultFn = check_vk_result;

    ImGui_ImplVulkan_Init(&init_info);
  }

  load_fonts(fonts, io, "fonts/SourceCodePro-Regular.ttf", 4.f);
  load_fonts(fonts_bold, io, "fonts/SourceCodePro-Bold.ttf", 4.f);
  
  io.FontDefault = fonts[2];

  fan::graphics::add_input_action(fan::key_escape, "open_settings");

}
void loco_t::destroy_imgui() {
  if (window.renderer == renderer_t::opengl) {
    ImGui_ImplOpenGL3_Shutdown();
  }
  else if (window.renderer == renderer_t::vulkan) {
    vkDeviceWaitIdle(context.vk.device);
    ImGui_ImplVulkan_Shutdown();
  }
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  ImPlot::DestroyContext();
  if (window.renderer == renderer_t::vulkan) {
    context.vk.imgui_close();
  }

}
#endif


void loco_t::init_framebuffer() {
  if (window.renderer == renderer_t::opengl) {
    gl.init_framebuffer();
  }
}

loco_t::loco_t() : loco_t(properties_t()) {

}

loco_t::loco_t(const properties_t& p) {
  if (fan::init_manager_t::initialized() == false) {
    fan::init_manager_t::initialize();
  }
  render_shapes_top = p.render_shapes_top;
  window.renderer = p.renderer;
  if (window.renderer == renderer_t::opengl) {
    new (&context.gl) fan::opengl::context_t();
    context_functions = fan::graphics::get_gl_context_functions();
    gl.open();
  }

  window.open(p.window_size, fan::window_t::default_window_name, p.window_flags);
  gloco = this;

  if(window.renderer == renderer_t::vulkan) {
    context_functions = fan::graphics::get_vk_context_functions();
    new (&context.vk) fan::vulkan::context_t();
    //context.vk.enable_clear = !render_shapes_top;
    context.vk.shapes_top = render_shapes_top;
    context.vk.open(window);
  }

  start_time = fan::time::clock::now();

  set_vsync(false); // using libuv
  //fan::print("less pain", this, (void*)&lighting, (void*)((uint8_t*)&lighting - (uint8_t*)this), sizeof(*this), lighting.ambient);
  if (window.renderer == renderer_t::opengl) {
    glfwMakeContextCurrent(window);

#if fan_debug >= fan_debug_high
    get_context().gl.set_error_callback();
#endif

    gl.initialize_fb_vaos();
  }

#if defined(loco_vfi)
  window.add_buttons_callback([this](const fan::window_t::mouse_buttons_cb_data_t& d) {
    fan::vec2 window_size = window.get_size();
    vfi.feed_mouse_button(d.button, d.state);
  });

  window.add_keys_callback([&](const fan::window_t::keyboard_keys_cb_data_t& d) {
    vfi.feed_keyboard(d.key, d.state);
  });

  window.add_mouse_move_callback([&](const fan::window_t::mouse_move_cb_data_t& d) {
    vfi.feed_mouse_move(d.position);
  });

  window.add_text_callback([&](const fan::window_t::text_cb_data_t& d) {
    vfi.feed_text(d.character);
  });
#endif

  default_texture = create_missing_texture();

  shaper.Open();

  {

    // filler
    shaper.AddKey(Key_e::light, sizeof(uint8_t), shaper_t::KeyBitOrderAny);
    shaper.AddKey(Key_e::light_end, sizeof(uint8_t), shaper_t::KeyBitOrderAny);
    shaper.AddKey(Key_e::depth, sizeof(loco_t::depth_t), shaper_t::KeyBitOrderLow);
    shaper.AddKey(Key_e::blending, sizeof(loco_t::blending_t), shaper_t::KeyBitOrderLow);
    shaper.AddKey(Key_e::image, sizeof(loco_t::image_t), shaper_t::KeyBitOrderLow);
    shaper.AddKey(Key_e::viewport, sizeof(loco_t::viewport_t), shaper_t::KeyBitOrderAny);
    shaper.AddKey(Key_e::camera, sizeof(loco_t::camera_t), shaper_t::KeyBitOrderAny);
    shaper.AddKey(Key_e::ShapeType, sizeof(shaper_t::ShapeTypeIndex_t), shaper_t::KeyBitOrderAny);
    shaper.AddKey(Key_e::filler, sizeof(uint8_t), shaper_t::KeyBitOrderAny);
    shaper.AddKey(Key_e::draw_mode, sizeof(uint8_t), shaper_t::KeyBitOrderAny);
    shaper.AddKey(Key_e::vertex_count, sizeof(uint32_t), shaper_t::KeyBitOrderAny);

    //gloco->shaper.AddKey(Key_e::image4, sizeof(loco_t::image_t) * 4, shaper_t::KeyBitOrderLow);
  }

  //{
  //  shaper_t::KeyTypeIndex_t ktia[] = {
  //    Key_e::depth,
  //    Key_e::blending,
  //    Key_e::image,
  //    Key_e::image,
  //    Key_e::multitexture,
  //    Key_e::viewport,
  //    Key_e::camera,
  //    Key_e::ShapeType
  //  };
  //  gloco->shaper.AddKeyPack(kp::multitexture, sizeof(ktia) / sizeof(ktia[0]), ktia);
  //}

  // order of open needs to be same with shapes enum

  {
    fan::vec2 window_size = window.get_size();
    {
      orthographic_camera.camera = open_camera(
        fan::vec2(0, window_size.x),
        fan::vec2(0, window_size.y)
      );
      orthographic_camera.viewport = open_viewport(
        fan::vec2(0, 0),
        window_size
      );
    }
    {
      perspective_camera.camera = open_camera_perspective();
      perspective_camera.viewport = open_viewport(
        fan::vec2(0, 0),
        window_size
      );
    }
  }

  if (window.renderer == renderer_t::opengl) {
    gl.shapes_open();
  }
  else if (window.renderer == renderer_t::vulkan) {
    vk.shapes_open();
  }


#if defined(loco_physics)
  fan::graphics::open_bcol();
#endif

#if defined(loco_imgui)
  init_imgui();
  generate_commands(this);
#endif

  bool windowed = true;
  // free this xd
  gloco->window.add_keys_callback(
    [windowed](const fan::window_t::keyboard_keys_cb_data_t& data) mutable {
      if (data.key == fan::key_enter && data.state == fan::keyboard_state::press && gloco->window.key_pressed(fan::key_left_alt)) {
        windowed = !windowed;
        gloco->window.set_display_mode(windowed ? fan::window_t::mode::windowed : fan::window_t::mode::borderless);
      }
    }
  );
  #if defined(loco_imgui)
  settings_menu.open();
  #endif

  auto it = fan::graphics::engine_init_cbs.GetNodeFirst();
  while (it != fan::graphics::engine_init_cbs.dst) {
    fan::graphics::engine_init_cbs.StartSafeNext(it);
    fan::graphics::engine_init_cbs[it](this);
    it = fan::graphics::engine_init_cbs.EndSafeNext();
  }
}

void loco_t::destroy() {
  if (window == nullptr) {
    return;
  }
#if defined(loco_imgui)
  console.commands.func_table.clear();
  console.close();
#endif
  fan::graphics::close_bcol();
  if (window.renderer == loco_t::renderer_t::vulkan) {
    vkDeviceWaitIdle(context.vk.device);
    vkDestroySampler(context.vk.device, vk.post_process_sampler, nullptr);
    vk.d_attachments.close(context.vk);
    vk.post_process.close(context.vk);
  }
  shaper.Close();
#if defined(loco_imgui)
  destroy_imgui();
#endif
  window.close();
}

loco_t::~loco_t() {
  destroy();
}

void loco_t::switch_renderer(uint8_t renderer) {
  std::vector<std::string> image_paths;
  fan::vec2 window_size = window.get_size();
  fan::vec2 window_position = window.get_position();
  uint64_t flags = window.flags;

  {// close
    if (window.renderer == loco_t::renderer_t::vulkan) {
      // todo wrap to vk.
      vkDeviceWaitIdle(context.vk.device);
      vkDestroySampler(context.vk.device, vk.post_process_sampler, nullptr);
      vk.d_attachments.close(context.vk);
      vk.post_process.close(context.vk);
      //CLOOOOSEEE POSTPROCESSS IMAGEEES
    }
    else if (window.renderer == loco_t::renderer_t::opengl) {
      for(auto &st : shaper.ShapeTypes){
        if (std::holds_alternative<loco_t::shaper_t::ShapeType_t::vk_t>(st.renderer)) {
          auto& str = std::get<loco_t::shaper_t::ShapeType_t::vk_t>(st.renderer);
          str.shape_data.close(context.vk);
          str.pipeline.close(context.vk);
        }
        //st.BlockList.Close();
      }
      glDeleteVertexArrays(1, &gl.fb_vao);
      glDeleteBuffers(1, &gl.fb_vbo);
      context.gl.internal_close();
    }
#if defined(loco_imgui)
    destroy_imgui();
#endif
    window.close();
  }
  {// reopen
    window.renderer = reload_renderer_to; // i dont like this {window.renderer = ...}
    if (window.renderer == renderer_t::opengl) {
      context_functions = fan::graphics::get_gl_context_functions();
      new (&context.gl) fan::opengl::context_t();
      gl.open();
    }

    window.open(window_size, fan::window_t::default_window_name, flags | fan::window_t::flags::no_visible);
    window.set_position(window_position);
    window.set_position(window_position);
    glfwShowWindow(window);
    window.flags = flags;
    if(window.renderer == renderer_t::vulkan) {
      new (&context.vk) fan::vulkan::context_t();
      context_functions = fan::graphics::get_vk_context_functions();
      context.vk.open(window);
    }
  }
  {// reload
    {
      {
        fan::graphics::camera_list_t::nrtra_t nrtra;
        fan::graphics::camera_nr_t nr;
        nrtra.Open(&camera_list, &nr);
        while (nrtra.Loop(&camera_list, &nr)) {
          auto& cam = camera_list[nr];
          camera_set_ortho(
            nr,
            fan::vec2(cam.coordinates.left, cam.coordinates.right),
            fan::vec2(cam.coordinates.up, cam.coordinates.down)
          );
        }
        nrtra.Close(&camera_list);
      }
      {
        fan::graphics::viewport_list_t::nrtra_t nrtra;
        fan::graphics::viewport_nr_t nr;
        nrtra.Open(&viewport_list, &nr);
        while (nrtra.Loop(&viewport_list, &nr)) {
          auto& viewport = viewport_list[nr];
          viewport_set(
            nr,
            viewport.viewport_position,
            viewport.viewport_size,
            window.get_size()
          );
        }
        nrtra.Close(&viewport_list);
      }
    }

    {
      {
        {
          fan::graphics::image_list_t::nrtra_t nrtra;
          fan::graphics::image_nr_t nr;
          nrtra.Open(&image_list, &nr);
          while (nrtra.Loop(&image_list, &nr)) {
            
            if(window.renderer == renderer_t::opengl) {
              // illegal
              image_list[nr].internal = new fan::opengl::context_t::image_t;
              fan_opengl_call(glGenTextures(1, &((fan::opengl::context_t::image_t*)context_functions.image_get(&context.vk, nr))->texture_id));
            }
            else if(window.renderer == renderer_t::vulkan) {
              // illegal
              
              image_list[nr].internal = new fan::vulkan::context_t::image_t;
            }
            // handle blur?
            auto image_path = image_list[nr].image_path;
            if (image_path.empty()) {
              fan::image::image_info_t info;
              info.data = (void*)fan::image::missing_texture_pixels;
              info.size = 2;
              info.channels = 4;
              fan::graphics::image_load_properties_t lp;
              lp.min_filter = fan::graphics::image_filter::nearest;
              lp.mag_filter = fan::graphics::image_filter::nearest;
              lp.visual_output = fan::graphics::image_sampler_address_mode::repeat;
              image_reload(nr, info, lp);
            }
            else {
              image_reload(nr, image_list[nr].image_path);
            }
          }
          nrtra.Close(&image_list);
        }
        {
          fan::graphics::shader_list_t::nrtra_t nrtra;
          fan::graphics::shader_nr_t nr;
          nrtra.Open(&shader_list, &nr);
          while (nrtra.Loop(&shader_list, &nr)) {
            if(window.renderer == renderer_t::opengl) {
              shader_list[nr].internal = new fan::opengl::context_t::shader_t;
            }
            else if(window.renderer == renderer_t::vulkan) {
              shader_list[nr].internal = new fan::vulkan::context_t::shader_t;
            }
          }
          nrtra.Close(&shader_list);
        }
      }
      fan::image::image_info_t info;
      info.data = (void*)fan::image::missing_texture_pixels;
      info.size = 2;
      info.channels = 4;
      fan::graphics::image_load_properties_t lp;
      lp.min_filter = fan::graphics::image_filter::nearest;
      lp.mag_filter = fan::graphics::image_filter::nearest;
      lp.visual_output = fan::graphics::image_sampler_address_mode::repeat;
      image_reload(default_texture, info, lp);
    }
    shape_functions.clear();
    if (window.renderer == renderer_t::opengl) {
      gl.shapes_open();
      gl.initialize_fb_vaos();
    }
    else if (window.renderer == renderer_t::vulkan) {
      vk.shapes_open();
    }
    init_imgui();
    #if defined(loco_imgui)
      settings_menu.set_settings_theme();
    #endif

    shaper._BlockListCapacityChange(shape_type_t::rectangle, 0, 1);
    shaper._BlockListCapacityChange(shape_type_t::sprite, 0, 1);
  }
  reload_renderer_to = -1;
}

void loco_t::draw_shapes() {
  if (window.renderer == renderer_t::opengl) {
    gl.draw_shapes();
  }
  else if (window.renderer == renderer_t::vulkan) {
    vk.draw_shapes();
  }
}

void loco_t::process_shapes() {

  if (window.renderer == renderer_t::vulkan) {
    if (render_shapes_top == true) {
      vk.begin_render_pass();
    }
  }

  for (const auto& i : m_pre_draw) {
    i();
  }

  draw_shapes();

  for (const auto& i : m_post_draw) {
    i();
  }

  if (window.renderer == renderer_t::vulkan) {
    auto& cmd_buffer = context.vk.command_buffers[context.vk.current_frame];
    if (vk.image_error != (decltype(vk.image_error))-0xfff) {
      vkCmdNextSubpass(cmd_buffer, VK_SUBPASS_CONTENTS_INLINE);
      vkCmdBindPipeline(cmd_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk.post_process);
      vkCmdBindDescriptorSets(
        cmd_buffer,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        vk.post_process.m_layout,
        0,
        1,
        vk.d_attachments.m_descriptor_set,
        0,
        nullptr
      );

      // render post process
      vkCmdDraw(cmd_buffer, 6, 1, 0, 0);
    }
    if (render_shapes_top == true) {
      vkCmdEndRenderPass(cmd_buffer);
    }
  }
}

void loco_t::process_gui() {
#if defined(loco_imgui)
  {
    auto it = m_imgui_draw_cb.GetNodeFirst();
    while (it != m_imgui_draw_cb.dst) {
      m_imgui_draw_cb.StartSafeNext(it);
      m_imgui_draw_cb[it]();
      it = m_imgui_draw_cb.EndSafeNext();
    }
  }

  if (ImGui::IsKeyPressed(ImGuiKey_F3, false)) {
    render_console = !render_console;

    // force focus xd
    console.input.InsertText("a");
    console.input.SetText("");
    console.init_focus = true;
    console.input.IsFocused() = false;
  }
  if (render_console) {
    console.render();
  }  
  if (fan::graphics::is_input_action_active("open_settings")) {
    render_settings_menu = !render_settings_menu;
  }
  if (render_settings_menu) {
    settings_menu.render();
  }
  
  if (toggle_fps) {
    ImGui::SetNextWindowBgAlpha(0.9f);
    static int init = 0;
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoFocusOnAppearing ;
    if (init == 0) {
      window_flags |= ImGuiWindowFlags_AlwaysAutoResize;
      init = 1;
    }
    ImGui::Begin("Performance window", 0, window_flags);
  
    static constexpr int buffer_size = 128;
    static std::array<float, buffer_size> samples = {0};
    static int insert_index = 0;
    static float running_sum = 0.0f;
    static float running_min = std::numeric_limits<float>::max();
    static float running_max = std::numeric_limits<float>::min();
    static fan::time::clock refresh_speed{(uint64_t)0.05e9, true};

    if (refresh_speed.finished()) {
      float old_value = samples[insert_index];
      for (int i = 0; i < buffer_size - 1; ++i) {
          samples[i] = samples[i + 1];
      }

      samples[buffer_size - 1] = delta_time;

      running_sum += samples[buffer_size - 1] - samples[0];

      if (delta_time <= running_min) {
        running_min = delta_time;
      }
      else if (delta_time >= running_max) {
        running_max = delta_time;
      }

      insert_index = (insert_index + 1) % buffer_size;
      refresh_speed.restart();
    }

    float average_frame_time_ms = running_sum / buffer_size;
    float average_fps = 1.0f / average_frame_time_ms;
    float lowest_fps = 1.0f / running_max;
    float highest_fps = 1.0f / running_min;

    ImGui::Text("fps: %d", (int)(1.f / delta_time));
    ImGui::Text("Average Frame Time: %.4f ms", average_frame_time_ms);
    ImGui::Text("Lowest Frame Time: %.4f ms", running_min);
    ImGui::Text("Highest Frame Time: %.4f ms", running_max);
    ImGui::Text("Average fps: %.4f", average_fps);
    ImGui::Text("Lowest fps: %.4f", lowest_fps);
    ImGui::Text("Highest fps: %.4f", highest_fps);
    if (ImGui::Button("Reset lowest&highest")) {
      running_min = std::numeric_limits<float>::max();
      running_max = std::numeric_limits<float>::min();
    }

    if (ImPlot::BeginPlot("frame time", ImVec2(-1, 0), ImPlotFlags_NoFrame | ImPlotFlags_NoLegend)) { 
      ImPlot::SetupAxes("Frame Index", "FPS", ImPlotAxisFlags_NoLabel | ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_NoLabel | ImPlotAxisFlags_AutoFit); //
      ImPlot::PlotLine("FPS", samples.data(), buffer_size, 1.0, 0.0);
      ImPlot::EndPlot();
    }
    ImGui::Text("Current Frame Time: %.4f ms", delta_time);
    ImGui::End();
  }

#if defined(loco_framebuffer)

#endif

  ImGui::Render();

  
  if (window.renderer == renderer_t::opengl) {

    //glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    //glClear(GL_COLOR_BUFFER_BIT);

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
  }
  else if (window.renderer == renderer_t::vulkan) {
    auto& cmd_buffer = context.vk.command_buffers[context.vk.current_frame];
    // did draw
    if (vk.image_error == (decltype(vk.image_error))-0xfff) {
      vk.image_error = VK_SUCCESS;
    }
    if (render_shapes_top == false) {
      vkCmdEndRenderPass(cmd_buffer);
    }

    ImDrawData* draw_data = ImGui::GetDrawData();
    const bool is_minimized = (draw_data->DisplaySize.x <= 0.0f || draw_data->DisplaySize.y <= 0.0f);
    if (!is_minimized) {
     context.vk.ImGuiFrameRender(vk.image_error, clear_color);
    }
  }
#endif
}

void loco_t::process_frame() {

  if (window.renderer == renderer_t::opengl) {
    gl.begin_process_frame();
  }

  {
    auto it = m_update_callback.GetNodeFirst();
    while (it != m_update_callback.dst) {
      m_update_callback.StartSafeNext(it);
      m_update_callback[it](this);
      it = m_update_callback.EndSafeNext();
    }
  }

#if defined(loco_box2d)
  {
    auto it = shape_physics_update_cbs.GetNodeFirst();
    while (it != shape_physics_update_cbs.dst) {
      shape_physics_update_cbs.StartSafeNext(it);
      ((shape_physics_update_cb)shape_physics_update_cbs[it].cb)(shape_physics_update_cbs[it]);
      it = shape_physics_update_cbs.EndSafeNext();
    }
  }
#endif

  for (const auto& i : single_queue) {
    i();
  }

  single_queue.clear();

  #if defined(loco_imgui)
    ImGui::End();
#endif

  shaper.ProcessBlockEditQueue();


  if (window.renderer == renderer_t::vulkan){
    vk.begin_draw();
  }

  viewport_set(0, window.get_size(), window.get_size());

  if (render_shapes_top == false) {
    process_shapes();
    process_gui();
  }
  else {
    process_gui();
    process_shapes();
  }

  if (window.renderer == renderer_t::opengl) {
    glfwSwapBuffers(window);
  }
  else if (window.renderer == renderer_t::vulkan) {
#if !defined(loco_imgui)
    auto& cmd_buffer = context.vk.command_buffers[context.vk.current_frame];
    // did draw
    vkCmdNextSubpass(cmd_buffer, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdEndRenderPass(cmd_buffer);
#endif
    VkResult err = context.vk.end_render();
    context.vk.recreate_swap_chain(&window, err);
  }
}

bool loco_t::should_close() {
  if (window == nullptr) {
    return true;
  }
  return glfwWindowShouldClose(window);
}

bool loco_t::process_loop(const fan::function_t<void()>& lambda) {

#if defined(loco_imgui)
  std::vector<uint8_t> temp;
  struct shape_info_t {
    shape_t shape;
  };
  static std::vector<shape_t> shapes;
  if (reload_renderer_to != (decltype(reload_renderer_to))-1) {
    switch_renderer(reload_renderer_to);
  }

  if (window.renderer == renderer_t::opengl) {
    ImGui_ImplOpenGL3_NewFrame();
  }
  else if (window.renderer == renderer_t::vulkan) {
    ImGui_ImplVulkan_NewFrame();
  }

  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  auto& style = ImGui::GetStyle();
  ImVec4* colors = style.Colors;

  ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0));
  ImGui::PushStyleColor(ImGuiCol_DockingEmptyBg, ImVec4(0, 0, 0, 0));
  ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport());
  ImGui::PopStyleColor(2);

  ImGui::SetNextWindowPos(ImVec2(0, 0));
  ImGui::SetNextWindowSize(window.get_size());
  ImGui::Begin("##global_renderer", 0, ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |  ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar);
#endif

  lambda();

  // user can terminate from main loop
  if (should_close()) {
    close();
    return 1;
  }//

  process_frame();
  window.handle_events();
  delta_time = window.m_delta_time;
  
  // window can also be closed from window cb
  if (should_close()) {
    close();
    return 1;
  }//
    
  return 0;
}

void loco_t::start_timer() {
  double delay;
  if (target_fps <= 0) {
    delay = 0;
  }
  else {
    delay = std::round(1.0 / target_fps * 1000.0);
  }
  if (delay > 0) {
    uv_timer_start(&timer_handle, [](uv_timer_t* handle) {
      loco_t* loco = static_cast<loco_t*>(handle->data);    
      if (loco->process_loop(loco->main_loop)) {
        uv_timer_stop(handle);
        uv_stop(uv_default_loop());
        loco->window.glfw_window = nullptr;
      }
    }, 0, delay);
  }
}

void loco_t::start_idle() {
  uv_idle_start(&idle_handle, [](uv_idle_t* handle) {
    loco_t* loco = static_cast<loco_t*>(handle->data);
    if (loco->process_loop(loco->main_loop)) {
      uv_idle_stop(handle);
      uv_stop(uv_default_loop());
    }
  });
}


void loco_t::loop(const fan::function_t<void()>& lambda) {
  main_loop = lambda;
g_loop:
  double delay = std::round(1.0 / target_fps * 1000.0);

  if (!timer_init)  {
    uv_timer_init(uv_default_loop(), &timer_handle);
    timer_init = true;
  }
  if (!idle_init) {
    uv_idle_init(uv_default_loop(), &idle_handle);
    idle_init = true;
  }

  timer_handle.data = this;
  idle_handle.data = this;

  if (target_fps > 0) {
    start_timer();
  }
  else {
    start_idle();
  }

  uv_run(uv_default_loop(), UV_RUN_DEFAULT);
  if (should_close() == false) {
    goto g_loop;
  }
}

/*
void loco_t::loop(const fan::function_t<void()>&lambda) {
  while (1) {
    if (process_loop(lambda)) {
      break;
    }
  }
}
*/


loco_t::camera_t loco_t::open_camera(const fan::vec2& x, const fan::vec2& y) {
  loco_t::camera_t camera = camera_create();
  camera_set_ortho(camera, fan::vec2(x.x, x.y), fan::vec2(y.x, y.y));
  return camera;
}

loco_t::camera_t loco_t::open_camera_perspective(f32_t fov) {
  loco_t::camera_t camera = camera_create();
  camera_set_perspective(camera, fov, window.get_size());
  return camera;
}

loco_t::viewport_t loco_t::open_viewport(const fan::vec2& viewport_position, const fan::vec2& viewport_size) {
  loco_t::viewport_t viewport = viewport_create();
  viewport_set(viewport, viewport_position, viewport_size, window.get_size());
  return viewport;
}

void loco_t::set_viewport(loco_t::viewport_t viewport, const fan::vec2& viewport_position, const fan::vec2& viewport_size) {
  viewport_set(viewport, viewport_position, viewport_size, window.get_size());
}

//

fan::vec2 loco_t::transform_matrix(const fan::vec2& position) {
  fan::vec2 window_size = window.get_size();
  // not custom ortho friendly - made for -1 1
  return position / window_size * 2 - 1;
}

fan::vec2 loco_t::screen_to_ndc(const fan::vec2& screen_pos) {
  fan::vec2 window_size = window.get_size();
  return screen_pos / window_size * 2 - 1;
}

fan::vec2 loco_t::ndc_to_screen(const fan::vec2& ndc_position) {
  fan::vec2 window_size = window.get_size();
  fan::vec2 normalized_position = (ndc_position + 1) / 2;
  return normalized_position * window_size;
}

void loco_t::set_vsync(bool flag) {
  // vulkan vsync is enabled by presentation mode in swap chain
  if (window.renderer == renderer_t::opengl) {
    context.gl.set_vsync(&window, flag);
  }
}

void loco_t::update_timer_interval() {
  double delay;
  if (target_fps <= 0) {
    delay = 0;
  }
  else {
   delay = std::round(1.0 / target_fps * 1000.0);
  }
  if (delay > 0) {
    if (timer_enabled == false) {
      start_timer();
      timer_enabled = true;
    }
    uv_idle_stop(&idle_handle);
    uv_timer_set_repeat(&timer_handle, delay);
    uv_timer_again(&timer_handle);
  }
  else {
    uv_timer_stop(&timer_handle);
    if (!idle_init) {
      uv_idle_init(uv_default_loop(), &idle_handle);
      idle_handle.data = this;
      idle_init = true;
    }
    start_idle(); 
  }
}

void loco_t::set_target_fps(int32_t new_target_fps) {
  target_fps = new_target_fps;
  update_timer_interval();
}

#if defined(loco_imgui)

template <typename T>
loco_t::imgui_fs_var_t::imgui_fs_var_t(
  loco_t::shader_t shader_nr,
  const fan::string& var_name,
  T initial_,
  f32_t speed,
  f32_t min,
  f32_t max
) {
  //fan::vec_wrap_t < sizeof(T) / fan::conditional_value_t < std::is_class_v<T>, sizeof(T{} [0] ), sizeof(T) > , f32_t > initial = initial_;
  fan::vec_wrap_t<fan::conditional_value_t<std::is_arithmetic_v<T>, 1, sizeof(T) / sizeof(f32_t)>::value, f32_t> 
    initial;
  if constexpr (std::is_arithmetic_v<T>) {
    initial = (f32_t)initial_;
  }
  else {
    initial = initial_;
  }
  fan::opengl::context_t::shader_t shader = std::get<fan::opengl::context_t::shader_t>(gloco->shader_get(shader_nr));
  if (gloco->window.renderer == renderer_t::vulkan) {
    fan::throw_error("");
  }
  auto found = gloco->shader_list[shader_nr].uniform_type_table.find(var_name);
  if (found == gloco->shader_list[shader_nr].uniform_type_table.end()) {
    //fan::print("failed to set uniform value");
    return;
    //fan::throw_error("failed to set uniform value");
  }
  ie = [str = found->second, shader_nr, var_name, speed, min, max, data = initial]() mutable {
    bool modify = false;
    switch(fan::get_hash(str)) {
      case fan::get_hash(std::string_view("float")): {
        modify = ImGui::DragFloat(fan::string(std::move(var_name)).c_str(), &data[0], (f32_t)speed, (f32_t)min, (f32_t)max);
        break;
      }
      case fan::get_hash(std::string_view("vec2")): {
        modify = ImGui::DragFloat2(fan::string(std::move(var_name)).c_str(), ((fan::vec2*)&data)->data(), (f32_t)speed, (f32_t)min, (f32_t)max);
        break;
      }
      case fan::get_hash(std::string_view("vec3")): {
        modify = ImGui::DragFloat3(fan::string(std::move(var_name)).c_str(), ((fan::vec3*)&data)->data(), (f32_t)speed, (f32_t)min, (f32_t)max);
        break;
      }
      case fan::get_hash(std::string_view("vec4")): {
        modify = ImGui::DragFloat4(fan::string(std::move(var_name)).c_str(), ((fan::vec4*)&data)->data(), (f32_t)speed, (f32_t)min, (f32_t)max);
        break;
      }
    }
    if (modify) {
      gloco->shader_set_value(shader_nr, var_name, data);
    }
  };
  gloco->shader_set_value(shader_nr, var_name, initial);
}


template loco_t::imgui_fs_var_t::imgui_fs_var_t(
  loco_t::shader_t shader_nr,
  const fan::string& var_name,
  fan::vec2 initial_,
  f32_t speed,
  f32_t min,
  f32_t max
);
template loco_t::imgui_fs_var_t::imgui_fs_var_t(
  loco_t::shader_t shader_nr,
  const fan::string& var_name,
  double initial_,
  f32_t speed,
  f32_t min,
  f32_t max
);

void loco_t::set_imgui_viewport(loco_t::viewport_t viewport) {
  ImVec2 mainViewportPos = ImGui::GetMainViewport()->Pos;

  ImVec2 windowPos = ImGui::GetWindowPos();

  fan::vec2 windowPosRelativeToMainViewport;
  windowPosRelativeToMainViewport.x = windowPos.x - mainViewportPos.x;
  windowPosRelativeToMainViewport.y = windowPos.y - mainViewportPos.y;

  fan::vec2 window_size = window.get_size();
  fan::vec2 viewport_size = ImGui::GetContentRegionAvail();

  ImVec2 padding = ImGui::GetStyle().WindowPadding;
  viewport_size.x += padding.x * 2;
  viewport_size.y += padding.y * 2;

  fan::vec2 viewport_pos = fan::vec2(windowPosRelativeToMainViewport);
  viewport_set(
    viewport,
    viewport_pos,
    viewport_size,
    window_size
  );
}
#endif

#if defined(loco_imgui)

loco_t::imgui_element_nr_t::imgui_element_nr_t(const imgui_element_nr_t& nr) : imgui_element_nr_t() {
  if (nr.is_invalid()) {
    return;
  }
  init();
}

loco_t::imgui_element_nr_t::imgui_element_nr_t(imgui_element_nr_t&& nr) {
  NRI = nr.NRI;
  nr.invalidate_soft();
}

loco_t::imgui_element_nr_t::~imgui_element_nr_t() {
  invalidate();
}

loco_t::imgui_element_nr_t& loco_t::imgui_element_nr_t::operator=(const imgui_element_nr_t& id) {
  if (!is_invalid()) {
    invalidate();
  }
  if (id.is_invalid()) {
    return *this;
  }

  if (this != &id) {
    init();
  }
  return *this;
}

loco_t::imgui_element_nr_t& loco_t::imgui_element_nr_t::operator=(imgui_element_nr_t&& id) {
  if (!is_invalid()) {
    invalidate();
  }
  if (id.is_invalid()) {
    return *this;
  }

  if (this != &id) {
    if (!is_invalid()) {
      invalidate();
    }
    NRI = id.NRI;

    id.invalidate_soft();
  }
  return *this;
}

void loco_t::imgui_element_nr_t::init() {
  *(base_t*)this = gloco->m_imgui_draw_cb.NewNodeLast();
}

bool loco_t::imgui_element_nr_t::is_invalid() const {
  return loco_t::imgui_draw_cb_inric(*this);
}

void loco_t::imgui_element_nr_t::invalidate_soft() {
  *(base_t*)this = gloco->m_imgui_draw_cb.gnric();
}

void loco_t::imgui_element_nr_t::invalidate() {
  if (is_invalid()) {
    return;
  }
  gloco->m_imgui_draw_cb.unlrec(*this);
  *(base_t*)this = gloco->m_imgui_draw_cb.gnric();
}

#endif


void loco_t::input_action_t::add(const int* keys, std::size_t count, std::string_view action_name) {
  action_data_t action_data;
  action_data.count = (uint8_t)count;
  std::memcpy(action_data.keys, keys, sizeof(int) * count);
  input_actions[action_name] = action_data;
}

void loco_t::input_action_t::add(int key, std::string_view action_name) {
  add(&key, 1, action_name);
}

void loco_t::input_action_t::add(std::initializer_list<int> keys, std::string_view action_name) {
  add(keys.begin(), keys.size(), action_name);
}

void loco_t::input_action_t::edit(int key, std::string_view action_name) {
  auto found = input_actions.find(action_name);
  if (found == input_actions.end()) {
    fan::throw_error("trying to modify non existing action");
  }
  std::memset(found->second.keys, 0, sizeof(found->second.keys));
  found->second.keys[0] = key;
  found->second.count = 1;
  found->second.combo_count = 0;
}

void loco_t::input_action_t::add_keycombo(std::initializer_list<int> keys, std::string_view action_name) {
  action_data_t action_data;
  action_data.combo_count = (uint8_t)keys.size();
  std::memcpy(action_data.key_combos, keys.begin(), sizeof(int) * action_data.combo_count);
  input_actions[action_name] = action_data;
}

bool loco_t::input_action_t::is_active(std::string_view action_name, int pstate) {
  auto found = input_actions.find(action_name);
  if (found != input_actions.end()) {
    action_data_t& action_data = found->second;

    if (action_data.combo_count) {
      int state = none;
      for (int i = 0; i < action_data.combo_count; ++i) {
        int s = gloco->window.key_state(action_data.key_combos[i]);
        if (s == none) {
          return none == loco_t::input_action_t::press;
        }
        if (state == input_action_t::press && s == input_action_t::repeat) {
          state = 1;
        }
        if (state == input_action_t::press_or_repeat) {
          if (state == input_action_t::press && s == input_action_t::repeat) {
          }
        }
        else {
          state = s;
        }
      }
      if (pstate == input_action_t::press_or_repeat) {
        return state == input_action_t::press || 
          state == input_action_t::repeat;
      }
      return state == pstate;
    }
    else if (action_data.count){
      int state = none;
      for (int i = 0; i < action_data.count; ++i) {
        int s = gloco->window.key_state(action_data.keys[i]);
        if (s != none) {
          state = s;
        }
      }
      if (pstate == input_action_t::press_or_repeat) {
        return state == input_action_t::press || 
          state == input_action_t::repeat;
      }
      return state == pstate;
    }
  }
  return none == pstate;
}

bool loco_t::input_action_t::is_action_clicked(std::string_view action_name) {
  return is_active(action_name);
}
bool loco_t::input_action_t::is_action_down(std::string_view action_name) {
  return is_active(action_name, press_or_repeat);
}
bool loco_t::input_action_t::exists(std::string_view action_name) {
  return input_actions.find(action_name) != input_actions.end();
}
void loco_t::input_action_t::insert_or_assign(int key, std::string_view action_name) {
  action_data_t action_data;
  action_data.count = (uint8_t)1;
  std::memcpy(action_data.keys, &key, sizeof(int) * 1);
  input_actions.insert_or_assign(action_name, action_data);
}

fan::vec2 loco_t::transform_position(const fan::vec2& p, loco_t::viewport_t viewport, loco_t::camera_t camera) {

  auto v = gloco->viewport_get(viewport);
  auto c = gloco->camera_get(camera);

  fan::vec2 viewport_position = v.viewport_position;
  fan::vec2 viewport_size = v.viewport_size;

  f32_t l = c.coordinates.left;
  f32_t r = c.coordinates.right;
  f32_t t = c.coordinates.up;
  f32_t b = c.coordinates.down;

  fan::vec2 tp = p - viewport_position;
  fan::vec2 d = viewport_size;
  tp /= d;
  tp = fan::vec2(r * tp.x - l * tp.x + l, b * tp.y - t * tp.y + t);
  tp += c.position;
  return tp;
}

fan::vec2 loco_t::get_mouse_position(const loco_t::camera_t& camera, const loco_t::viewport_t& viewport) {
  return transform_position(get_mouse_position(), viewport, camera);
}

fan::vec2 loco_t::get_mouse_position() {
  return window.get_mouse_position();
  //return get_mouse_position(gloco->default_camera->camera, gloco->default_camera->viewport); behaving oddly
}

fan::vec2 fan::graphics::get_mouse_position(const fan::graphics::camera_t& camera) {
  return loco_t::transform_position(gloco->get_mouse_position(), camera.viewport, camera.camera);
}

fan::vec2 loco_t::translate_position(const fan::vec2& p, loco_t::viewport_t viewport, loco_t::camera_t camera) {

  auto v = gloco->viewport_get(viewport);
  fan::vec2 viewport_position = v.viewport_position;
  fan::vec2 viewport_size = v.viewport_size;

  auto c = gloco->camera_get(camera);

  f32_t l = c.coordinates.left;
  f32_t r = c.coordinates.right;
  f32_t t = c.coordinates.up;
  f32_t b = c.coordinates.down;

  fan::vec2 tp = p - viewport_position;
  fan::vec2 d = viewport_size;
  tp /= d;
  tp = fan::vec2(r * tp.x - l * tp.x + l, b * tp.y - t * tp.y + t);
  return tp;
}

fan::vec2 loco_t::translate_position(const fan::vec2& p) {
  return translate_position(p, orthographic_camera.viewport, orthographic_camera.camera);
}

void loco_t::shape_t::erase() {
  remove();
}

loco_t::shape_t::shape_t() {
  sic();
}

loco_t::shape_t::shape_t(shaper_t::ShapeID_t&& s) {
  //if (s.iic() == false) {
  //  if (((shape_t*)&s)->get_shape_type() == shape_type_t::polygon) {
  //    loco_t::polygon_t::ri_t* src_data = (loco_t::polygon_t::ri_t*)s.GetData(gloco->shaper);
  //    loco_t::polygon_t::ri_t* dst_data = (loco_t::polygon_t::ri_t*)GetData(gloco->shaper);
  //    *dst_data = *src_data;
  //  }
  //}
  NRI = s.NRI;
  s.sic();
}

loco_t::shape_t::shape_t(const shaper_t::ShapeID_t& s) : shape_t() {

  if (s.iic()) {
    return;
  }

  {
    auto sti = gloco->shaper.ShapeList[s].sti;
    shaper_deep_copy(this, (const loco_t::shape_t*)&s, sti);
  }
  if (((shape_t*)&s)->get_shape_type() == shape_type_t::polygon) {
    loco_t::polygon_t::ri_t* src_data = (loco_t::polygon_t::ri_t*)s.GetData(gloco->shaper);
    loco_t::polygon_t::ri_t* dst_data = (loco_t::polygon_t::ri_t*)GetData(gloco->shaper);
    if (gloco->get_renderer() == renderer_t::opengl) {
      dst_data->vao.open(gloco->context.gl);
      dst_data->vbo.open(gloco->context.gl, src_data->vbo.m_target);

      auto& shape_data = std::get<loco_t::shaper_t::ShapeType_t::gl_t>(gloco->shaper.GetShapeTypes(shape_type_t::polygon).renderer);
      fan::graphics::context_shader_t shader;
      if (!shape_data.shader.iic()) {
        shader = gloco->shader_get(shape_data.shader);
      }
      dst_data->vao.bind(gloco->context.gl);
        dst_data->vbo.bind(gloco->context.gl);
      uint64_t ptr_offset = 0;
      for (shape_gl_init_t& location : polygon_t::locations) {
        if ((gloco->context.gl.opengl.major == 2 && gloco->context.gl.opengl.minor == 1) && !shape_data.shader.iic()) {
          location.index.first = fan_opengl_call(glGetAttribLocation(std::get<fan::opengl::context_t::shader_t>(shader).id, location.index.second));
        }
        fan_opengl_call(glEnableVertexAttribArray(location.index.first));
        fan_opengl_call(glVertexAttribPointer(location.index.first, location.size, location.type, GL_FALSE, location.stride, (void*)ptr_offset));
        // instancing
        if ((gloco->context.gl.opengl.major > 3) || (gloco->context.gl.opengl.major == 3 && gloco->context.gl.opengl.minor >= 3)) {
          if (shape_data.instanced) {
            fan_opengl_call(glVertexAttribDivisor(location.index.first, 1));
          }
        }
        switch (location.type) {
        case GL_FLOAT: {
          ptr_offset += location.size * sizeof(GLfloat);
          break;
        }
        case GL_UNSIGNED_INT: {
          ptr_offset += location.size * sizeof(GLuint);
          break;
        }
        default: {
          fan::throw_error_impl();
        }
        }
      }
      fan::opengl::core::write_glbuffer(gloco->context.gl, dst_data->vbo.m_buffer, 0, dst_data->buffer_size, dst_data->vbo.m_usage, dst_data->vbo.m_target);
      glBindBuffer(GL_COPY_READ_BUFFER, src_data->vbo.m_buffer);
      glBindBuffer(GL_COPY_WRITE_BUFFER, dst_data->vbo.m_buffer);
      glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, dst_data->buffer_size);
      loco_t::polygon_vertex_t* ri = new loco_t::polygon_vertex_t[dst_data->buffer_size / sizeof(loco_t::polygon_vertex_t)];
      loco_t::polygon_vertex_t* ri2 = new loco_t::polygon_vertex_t[dst_data->buffer_size / sizeof(loco_t::polygon_vertex_t)];
      fan::opengl::core::get_glbuffer(gloco->context.gl, ri, dst_data->vbo.m_buffer, dst_data->buffer_size, 0, dst_data->vbo.m_target);
      fan::opengl::core::get_glbuffer(gloco->context.gl, ri2, src_data->vbo.m_buffer, src_data->buffer_size, 0, src_data->vbo.m_target);
      delete[] ri;
    }
    else {
      fan::throw_error_impl();
    }
  }
}

loco_t::shape_t::shape_t(shape_t&& s) : shape_t(std::move(*dynamic_cast<shaper_t::ShapeID_t*>(&s))) {

}

loco_t::shape_t::shape_t(const loco_t::shape_t& s) : shape_t(*dynamic_cast<const shaper_t::ShapeID_t*>(&s)) {
  //NRI = s.NRI;
}

loco_t::shape_t& loco_t::shape_t::operator=(const loco_t::shape_t& s) {
  if (iic() == false) {
    remove();
  }
  if (s.iic()) {
    return *this;
  }
  if (this != &s) {
    {
      auto sti = gloco->shaper.ShapeList[s].sti;

      shaper_deep_copy(this, (const loco_t::shape_t*)&s, sti);
    }
    if (((shape_t*)&s)->get_shape_type() == shape_type_t::polygon) {
      loco_t::polygon_t::ri_t* src_data = (loco_t::polygon_t::ri_t*)s.GetData(gloco->shaper);
      loco_t::polygon_t::ri_t* dst_data = (loco_t::polygon_t::ri_t*)GetData(gloco->shaper);
      if (gloco->get_renderer() == renderer_t::opengl) {
        dst_data->vao.open(gloco->context.gl);
        dst_data->vbo.open(gloco->context.gl, src_data->vbo.m_target);

        auto& shape_data = std::get<loco_t::shaper_t::ShapeType_t::gl_t>(gloco->shaper.GetShapeTypes(shape_type_t::polygon).renderer);
        fan::graphics::context_shader_t shader;
        if (!shape_data.shader.iic()) {
          shader = gloco->shader_get(shape_data.shader);
        }
        dst_data->vao.bind(gloco->context.gl);
        dst_data->vbo.bind(gloco->context.gl);
        uint64_t ptr_offset = 0;
        for (shape_gl_init_t& location : polygon_t::locations) {
          if ((gloco->context.gl.opengl.major == 2 && gloco->context.gl.opengl.minor == 1) && !shape_data.shader.iic()) {
            location.index.first = fan_opengl_call(glGetAttribLocation(std::get<fan::opengl::context_t::shader_t>(shader).id, location.index.second));
          }
          fan_opengl_call(glEnableVertexAttribArray(location.index.first));
          fan_opengl_call(glVertexAttribPointer(location.index.first, location.size, location.type, GL_FALSE, location.stride, (void*)ptr_offset));
          // instancing
          if ((gloco->context.gl.opengl.major > 3) || (gloco->context.gl.opengl.major == 3 && gloco->context.gl.opengl.minor >= 3)) {
            if (shape_data.instanced) {
              fan_opengl_call(glVertexAttribDivisor(location.index.first, 1));
            }
          }
          switch (location.type) {
          case GL_FLOAT: {
            ptr_offset += location.size * sizeof(GLfloat);
            break;
          }
          case GL_UNSIGNED_INT: {
            ptr_offset += location.size * sizeof(GLuint);
            break;
          }
          default: {
            fan::throw_error_impl();
          }
          }
          fan::opengl::core::write_glbuffer(gloco->context.gl, dst_data->vbo.m_buffer, 0, dst_data->buffer_size, dst_data->vbo.m_usage, dst_data->vbo.m_target);
          glBindBuffer(GL_COPY_READ_BUFFER, src_data->vbo.m_buffer);
          glBindBuffer(GL_COPY_WRITE_BUFFER, dst_data->vbo.m_buffer);
          glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, dst_data->buffer_size);
        }
      }
      else {
        fan::throw_error_impl();
      }
    }
    //fan::print("i dont know what to do");
    //NRI = s.NRI;
  }
  return *this;
}

loco_t::shape_t& loco_t::shape_t::operator=(loco_t::shape_t&& s) {
  if (iic() == false) {
    remove();
  }
  if (s.iic()) {
    return *this;
  }

  if (this != &s) {
    if (s.iic() == false) {

    }
    NRI = s.NRI;
    s.sic();
  }
  return *this;
}

loco_t::shape_t::~shape_t() {
  remove();
}

void loco_t::shape_t::remove() {
  if (iic()) {
    return;
  }
#if defined(debug_shape_t)
  fan::print("-", NRI);
#endif
  if (gloco->shaper.ShapeList.Usage() == 0) {
    return;
  }
  auto shape_type = get_shape_type();
  if (shape_type == loco_t::shape_type_t::vfi) {
    gloco->vfi.erase(*this);
    sic();
    return;
  }
  if (shape_type == loco_t::shape_type_t::polygon) {
     auto ri = (polygon_t::ri_t*)GetData(gloco->shaper);
     ri->vbo.close(gloco->context.gl);
     ri->vao.close(gloco->context.gl);
  }
  gloco->shaper.remove(*this);
  sic();
}


// many things assume uint16_t so thats why not shaper_t::ShapeTypeIndex_t

uint16_t loco_t::shape_t::get_shape_type() const {
  return gloco->shaper.ShapeList[*this].sti;
}

void loco_t::shape_t::set_position(const fan::vec3& position) {
  gloco->shape_functions[get_shape_type()].set_position3(this, position);
}

fan::vec3 loco_t::shape_t::get_position() {
  return gloco->shape_functions[get_shape_type()].get_position(this);
}

void loco_t::shape_t::set_size(const fan::vec2& size) {
  gloco->shape_functions[get_shape_type()].set_size(this, size);
}

void loco_t::shape_t::set_size3(const fan::vec3& size) {
  gloco->shape_functions[get_shape_type()].set_size3(this, size);
}

fan::vec2 loco_t::shape_t::get_size() {
  return gloco->shape_functions[get_shape_type()].get_size(this);
}

fan::vec3 loco_t::shape_t::get_size3() {
  return gloco->shape_functions[get_shape_type()].get_size3(this);
}

void loco_t::shape_t::set_rotation_point(const fan::vec2& rotation_point) {
  gloco->shape_functions[get_shape_type()].set_rotation_point(this, rotation_point);
}

fan::vec2 loco_t::shape_t::get_rotation_point() {
  return gloco->shape_functions[get_shape_type()].get_rotation_point(this);
}

void loco_t::shape_t::set_color(const fan::color& color) {
  gloco->shape_functions[get_shape_type()].set_color(this, color);
}

fan::color loco_t::shape_t::get_color() {
  return gloco->shape_functions[get_shape_type()].get_color(this);
}

void loco_t::shape_t::set_angle(const fan::vec3& angle) {
  gloco->shape_functions[get_shape_type()].set_angle(this, angle);
}

fan::vec3 loco_t::shape_t::get_angle() {
  return gloco->shape_functions[get_shape_type()].get_angle(this);
}

fan::vec2 loco_t::shape_t::get_tc_position() {
  return gloco->shape_functions[get_shape_type()].get_tc_position(this);
}

void loco_t::shape_t::set_tc_position(const fan::vec2& tc_position) {
  gloco->shape_functions[get_shape_type()].set_tc_position(this, tc_position);
}

fan::vec2 loco_t::shape_t::get_tc_size() {
  return gloco->shape_functions[get_shape_type()].get_tc_size(this);
}

void loco_t::shape_t::set_tc_size(const fan::vec2& tc_size) {
  gloco->shape_functions[get_shape_type()].set_tc_size(this, tc_size);
}

bool loco_t::shape_t::load_tp(loco_t::texturepack_t::ti_t* ti) {
  return gloco->shape_functions[get_shape_type()].load_tp(this, ti);
}

loco_t::texturepack_t::ti_t loco_t::shape_t::get_tp() {
  loco_t::texturepack_t::ti_t ti;
  ti.image = &gloco->default_texture;
  auto img = gloco->image_get(*ti.image);
  std::visit([this, &ti] (const auto& v) {
    ti.position = get_tc_position() * v.size;
    ti.size = get_tc_size() * v.size;
  }, img);

  return ti;
  //return gloco->shape_functions[gloco->shaper.GetSTI(*this)].get_tp(this);
}

bool loco_t::shape_t::set_tp(loco_t::texturepack_t::ti_t* ti) {
  return load_tp(ti);
}

loco_t::camera_t loco_t::shape_t::get_camera() {
  return gloco->shape_functions[get_shape_type()].get_camera(this);
}

void loco_t::shape_t::set_camera(loco_t::camera_t camera) {
  gloco->shape_functions[get_shape_type()].set_camera(this, camera);
}

loco_t::viewport_t loco_t::shape_t::get_viewport() {
  return gloco->shape_functions[get_shape_type()].get_viewport(this);
}

void loco_t::shape_t::set_viewport(loco_t::viewport_t viewport) {
  gloco->shape_functions[get_shape_type()].set_viewport(this, viewport);
}

fan::vec2 loco_t::shape_t::get_grid_size() {
  return gloco->shape_functions[get_shape_type()].get_grid_size(this);
}

void loco_t::shape_t::set_grid_size(const fan::vec2& grid_size) {
  gloco->shape_functions[get_shape_type()].set_grid_size(this, grid_size);
}

loco_t::image_t loco_t::shape_t::get_image() {
  return gloco->shape_functions[get_shape_type()].get_image(this);
}

void loco_t::shape_t::set_image(loco_t::image_t image) {
  gloco->shape_functions[get_shape_type()].set_image(this, image);
}

f32_t loco_t::shape_t::get_parallax_factor() {
  return gloco->shape_functions[get_shape_type()].get_parallax_factor(this);
}

void loco_t::shape_t::set_parallax_factor(f32_t parallax_factor) {
  gloco->shape_functions[get_shape_type()].set_parallax_factor(this, parallax_factor);
}

fan::vec3 loco_t::shape_t::get_rotation_vector() {
  return gloco->shape_functions[get_shape_type()].get_rotation_vector(this);
}

uint32_t loco_t::shape_t::get_flags() {
  return gloco->shape_functions[get_shape_type()].get_flags(this);
}

void loco_t::shape_t::set_flags(uint32_t flag) {
  return gloco->shape_functions[get_shape_type()].set_flags(this, flag);
}

f32_t loco_t::shape_t::get_radius() {
  return gloco->shape_functions[get_shape_type()].get_radius(this);
}

fan::vec3 loco_t::shape_t::get_src() {
  return gloco->shape_functions[get_shape_type()].get_src(this);
}

fan::vec3 loco_t::shape_t::get_dst() {
  return gloco->shape_functions[get_shape_type()].get_dst(this);
}

f32_t loco_t::shape_t::get_outline_size() {
  return gloco->shape_functions[get_shape_type()].get_outline_size(this);
}

fan::color loco_t::shape_t::get_outline_color() {
  return gloco->shape_functions[get_shape_type()].get_outline_color(this);
}

void loco_t::shape_t::reload(uint8_t format, void** image_data, const fan::vec2& image_size, uint32_t filter) {
  gloco->shape_functions[get_shape_type()].reload(this, format, image_data, image_size, filter);
}

void loco_t::shape_t::reload(uint8_t format, const fan::vec2& image_size, uint32_t filter) {
  void* data[4]{};
  gloco->shape_functions[get_shape_type()].reload(this, format, data, image_size, filter);
}

void loco_t::shape_t::set_line(const fan::vec2& src, const fan::vec2& dst) {
  gloco->shape_functions[get_shape_type()].set_line(this, src, dst);
}

/// shapes +
/// shapes +
/// shapes +
/// shapes +

loco_t::shape_t loco_t::light_t::push_back(const properties_t& properties) {
  vi_t vi;
  vi.position = properties.position;
  vi.parallax_factor = properties.parallax_factor;
  vi.size = properties.size;
  vi.rotation_point = properties.rotation_point;
  vi.color = properties.color;
  vi.rotation_vector = properties.rotation_vector;
  vi.flags = properties.flags;
  vi.angle = properties.angle;
  ri_t ri;

  return shape_add(shape_type, vi, ri,
    Key_e::light, (uint8_t)0,
    Key_e::viewport, properties.viewport,
    Key_e::camera, properties.camera,
    Key_e::ShapeType, shape_type,
    Key_e::draw_mode, properties.draw_mode,
    Key_e::vertex_count, properties.vertex_count
  );
}

loco_t::shape_t loco_t::line_t::push_back(const properties_t& properties) {
  vi_t vi;
  vi.src = properties.src;
  vi.dst = properties.dst;
  vi.color = properties.color;
  ri_t ri;

  return shape_add(shape_type, vi, ri,
    Key_e::depth, (uint16_t)properties.src.z,
    Key_e::blending, (uint8_t)properties.blending,
    Key_e::viewport, properties.viewport,
    Key_e::camera, properties.camera,
    Key_e::ShapeType, shape_type,
    Key_e::draw_mode, properties.draw_mode,
    Key_e::vertex_count, properties.vertex_count
  );
}

loco_t::shape_t loco_t::rectangle_t::push_back(const properties_t& properties) {
  vi_t vi;
  vi.position = properties.position;
  vi.size = properties.size;
  vi.color = properties.color;
  vi.outline_color = properties.outline_color;
  vi.angle = properties.angle;
  vi.rotation_point = properties.rotation_point;
  ri_t ri;

  return shape_add(shape_type, vi, ri,
    Key_e::depth, (uint16_t)properties.position.z,
    Key_e::blending, (uint8_t)properties.blending,
    Key_e::viewport, properties.viewport,
    Key_e::camera, properties.camera,
    Key_e::ShapeType, shape_type,
    Key_e::draw_mode, properties.draw_mode,
    Key_e::vertex_count, properties.vertex_count
  );
}

loco_t::shape_t loco_t::circle_t::push_back(const circle_t::properties_t& properties) {
  circle_t::vi_t vi;
  vi.position = properties.position;
  vi.radius = properties.radius;
  vi.rotation_point = properties.rotation_point;
  vi.color = properties.color;
  vi.rotation_vector = properties.rotation_vector;
  vi.angle = properties.angle;
  vi.flags = properties.flags;
  circle_t::ri_t ri;
  return shape_add(shape_type, vi, ri,
    Key_e::depth, (uint16_t)properties.position.z,
    Key_e::blending, (uint8_t)properties.blending,
    Key_e::viewport, properties.viewport,
    Key_e::camera, properties.camera,
    Key_e::ShapeType, shape_type,
    Key_e::draw_mode, properties.draw_mode,
    Key_e::vertex_count, properties.vertex_count
  );
}

loco_t::shape_t loco_t::capsule_t::push_back(const loco_t::capsule_t::properties_t& properties) {
  capsule_t::vi_t vi;
  vi.position = properties.position;
  vi.center0 = properties.center0;
  vi.center1 = properties.center1;
  vi.radius = properties.radius;
  vi.rotation_point = properties.rotation_point;
  vi.color = properties.color;
  vi.outline_color = properties.outline_color;
  vi.rotation_vector = properties.rotation_vector;
  vi.angle = properties.angle;
  vi.flags = properties.flags;
  capsule_t::ri_t ri;
  return shape_add(shape_type, vi, ri,
    Key_e::depth, (uint16_t)properties.position.z,
    Key_e::blending, (uint8_t)properties.blending,
    Key_e::viewport, properties.viewport,
    Key_e::camera, properties.camera,
    Key_e::ShapeType, shape_type,
    Key_e::draw_mode, properties.draw_mode,
    Key_e::vertex_count, properties.vertex_count
  );
}

loco_t::shape_t loco_t::polygon_t::push_back(const loco_t::polygon_t::properties_t& properties) {
  if (properties.vertices.empty()) {
    fan::throw_error("invalid vertices");
  }

  std::vector<loco_t::polygon_vertex_t> polygon_vertices(properties.vertices.size());
  for (std::size_t i = 0; i < properties.vertices.size(); ++i) {
    polygon_vertices[i].position = properties.vertices[i].position;
    polygon_vertices[i].color = properties.vertices[i].color;
    polygon_vertices[i].offset = properties.position;
    polygon_vertices[i].angle = properties.angle;
    polygon_vertices[i].rotation_point = properties.rotation_point;
  }

  vi_t vis;
  ri_t ri;
  ri.buffer_size = sizeof(decltype(polygon_vertices)::value_type) * polygon_vertices.size();
  ri.vao.open(gloco->context.gl);
  ri.vao.bind(gloco->context.gl);
  ri.vbo.open(gloco->context.gl, GL_ARRAY_BUFFER);
  fan::opengl::core::write_glbuffer(
    gloco->context.gl, 
    ri.vbo.m_buffer, 
    polygon_vertices.data(), 
    ri.buffer_size,
    GL_STATIC_DRAW,
    ri.vbo.m_target
  );

  auto& shape_data = std::get<loco_t::shaper_t::ShapeType_t::gl_t>(gloco->shaper.GetShapeTypes(shape_type).renderer);

  fan::graphics::context_shader_t shader;
  if (!shape_data.shader.iic()) {
    shader = gloco->shader_get(shape_data.shader);
  }
  uint64_t ptr_offset = 0;
  for (shape_gl_init_t& location : locations) {
    if ((gloco->context.gl.opengl.major == 2 && gloco->context.gl.opengl.minor == 1) && !shape_data.shader.iic()) {
      location.index.first = fan_opengl_call(glGetAttribLocation(std::get<fan::opengl::context_t::shader_t>(shader).id, location.index.second));
    }
    fan_opengl_call(glEnableVertexAttribArray(location.index.first));
    fan_opengl_call(glVertexAttribPointer(location.index.first, location.size, location.type, GL_FALSE, location.stride, (void*)ptr_offset));
    // instancing
    if ((gloco->context.gl.opengl.major > 3) || (gloco->context.gl.opengl.major == 3 && gloco->context.gl.opengl.minor >= 3)) {
      if (shape_data.instanced) {
        fan_opengl_call(glVertexAttribDivisor(location.index.first, 1));
      }
    }
    switch (location.type) {
    case GL_FLOAT: {
      ptr_offset += location.size * sizeof(GLfloat);
      break;
    }
    case GL_UNSIGNED_INT: {
      ptr_offset += location.size * sizeof(GLuint);
      break;
    }
    default: {
      fan::throw_error_impl();
    }
    }
  }

  return shape_add(shape_type, vis, ri,
    Key_e::depth, (uint16_t)properties.position.z,
    Key_e::blending, (uint8_t)properties.blending,
    Key_e::viewport, properties.viewport,
    Key_e::camera, properties.camera,
    Key_e::ShapeType, shape_type,
    Key_e::draw_mode, properties.draw_mode,
    Key_e::vertex_count, (uint32_t)properties.vertices.size()
  );
}

loco_t::shape_t loco_t::sprite_t::push_back(const properties_t& properties) {

  vi_t vi;
  vi.position = properties.position;
  vi.size = properties.size;
  vi.rotation_point = properties.rotation_point;
  vi.color = properties.color;
  vi.angle = properties.angle;
  vi.flags = properties.flags;
  vi.tc_position = properties.tc_position;
  vi.tc_size = properties.tc_size;
  vi.parallax_factor = properties.parallax_factor;
  vi.seed = properties.seed;

  ri_t ri;
  ri.images = properties.images;

  loco_t& loco = *OFFSETLESS(this, loco_t, sprite);
  if (loco.window.renderer == loco_t::renderer_t::opengl) {

    if ((loco.context.gl.opengl.major > 3) || (loco.context.gl.opengl.major == 3 && loco.context.gl.opengl.minor >= 3)) {
      return shape_add(
        shape_type, vi, ri, 
        Key_e::depth,
        static_cast<uint16_t>(properties.position.z),
        Key_e::blending, static_cast<uint8_t>(properties.blending),
        Key_e::image, properties.image, 
        Key_e::viewport, properties.viewport, 
        Key_e::camera, properties.camera,
        Key_e::ShapeType, shape_type,
        Key_e::draw_mode, properties.draw_mode,
        Key_e::vertex_count, properties.vertex_count
      );
    }
    else {
      // Legacy version requires array of 6 identical vertices
      vi_t vertices[6];
      for (int i = 0; i < 6; i++) {
        vertices[i] = vi;
      }

      return shape_add(
        shape_type, vertices[0], ri, Key_e::depth,
        static_cast<uint16_t>(properties.position.z),
        Key_e::blending, static_cast<uint8_t>(properties.blending),
        Key_e::image, properties.image, Key_e::viewport,
        properties.viewport, Key_e::camera, properties.camera,
        Key_e::ShapeType, shape_type,
        Key_e::draw_mode, properties.draw_mode,
        Key_e::vertex_count, properties.vertex_count
      );
    }
  }
  else if (loco.window.renderer == renderer_t::vulkan) {
    return shape_add(
      shape_type, vi, ri, Key_e::depth,
      static_cast<uint16_t>(properties.position.z),
      Key_e::blending, static_cast<uint8_t>(properties.blending),
      Key_e::image, properties.image, Key_e::viewport,
      properties.viewport, Key_e::camera, properties.camera,
      Key_e::ShapeType, shape_type,
      Key_e::draw_mode, properties.draw_mode,
      Key_e::vertex_count, properties.vertex_count
    );
  }
  
  return {};
}

loco_t::shape_t loco_t::text_t::push_back(const properties_t& properties) {
  return gloco->shaper.add(shape_type_t::text, nullptr, 0, nullptr, nullptr);
}

loco_t::shape_t loco_t::unlit_sprite_t::push_back(const properties_t& properties) {
  //KeyPack.ShapeType = shape_type;
  vi_t vi;
  vi.position = properties.position;
  vi.size = properties.size;
  vi.rotation_point = properties.rotation_point;
  vi.color = properties.color;
  vi.angle = properties.angle;
  vi.flags = properties.flags;
  vi.tc_position = properties.tc_position;
  vi.tc_size = properties.tc_size;
  vi.parallax_factor = properties.parallax_factor;
  vi.seed = properties.seed;
  ri_t ri;
  ri.images = properties.images;
  return shape_add(shape_type, vi, ri,
    Key_e::depth, (uint16_t)properties.position.z,
    Key_e::blending, (uint8_t)properties.blending,
    Key_e::image, properties.image,
    Key_e::viewport, properties.viewport,
    Key_e::camera, properties.camera,
    Key_e::ShapeType, shape_type,
    Key_e::draw_mode, properties.draw_mode,
    Key_e::vertex_count, properties.vertex_count
  );
}

loco_t::shape_t loco_t::grid_t::push_back(const properties_t& properties) {
  vi_t vi;
  vi.position = properties.position;
  vi.size = properties.size;
  vi.grid_size = properties.grid_size;
  vi.rotation_point = properties.rotation_point;
  vi.color = properties.color;
  vi.angle = properties.angle;
  ri_t ri;
  return shape_add(shape_type, vi, ri,
    Key_e::depth, (uint16_t)properties.position.z,
    Key_e::blending, (uint8_t)properties.blending,
    Key_e::viewport, properties.viewport,
    Key_e::camera, properties.camera,
    Key_e::ShapeType, shape_type,
    Key_e::draw_mode, properties.draw_mode,
    Key_e::vertex_count, properties.vertex_count
  );
}

loco_t::shape_t loco_t::particles_t::push_back(const properties_t& properties) {
  //KeyPack.ShapeType = shape_type;
  vi_t vi;
  ri_t ri;
  ri.position = properties.position;
  ri.size = properties.size;
  ri.color = properties.color;

  ri.begin_time = fan::time::clock::now();
  ri.alive_time = properties.alive_time;
  ri.respawn_time = properties.respawn_time;
  ri.count = properties.count;
  ri.position_velocity = properties.position_velocity;
  ri.angle_velocity = properties.angle_velocity;
  ri.begin_angle = properties.begin_angle;
  ri.end_angle = properties.end_angle;
  ri.angle = properties.angle;
  ri.gap_size = properties.gap_size;
  ri.max_spread_size = properties.max_spread_size;
  ri.size_velocity = properties.size_velocity;
  ri.shape = properties.shape;

  return shape_add(shape_type, vi, ri,
    Key_e::depth, (uint16_t)properties.position.z,
    Key_e::blending, (uint8_t)properties.blending,
    Key_e::image, properties.image,
    Key_e::viewport, properties.viewport,
    Key_e::camera, properties.camera,
    Key_e::ShapeType, shape_type,
    Key_e::draw_mode, properties.draw_mode,
    Key_e::vertex_count, properties.vertex_count
  );
}

loco_t::shape_t loco_t::universal_image_renderer_t::push_back(const properties_t& properties) {
  vi_t vi;
  vi.position = properties.position;
  vi.size = properties.size;
  vi.tc_position = properties.tc_position;
  vi.tc_size = properties.tc_size;
  ri_t ri;
  // + 1
  std::copy(&properties.images[1], &properties.images[0] + properties.images.size(), ri.images_rest.data());
  return shape_add(shape_type, vi, ri,
    Key_e::depth, (uint16_t)properties.position.z,
    Key_e::blending, (uint8_t)properties.blending,
    Key_e::image, properties.images[0],
    Key_e::viewport, properties.viewport,
    Key_e::camera, properties.camera,
    Key_e::ShapeType, shape_type,
    Key_e::draw_mode, properties.draw_mode,
    Key_e::vertex_count, properties.vertex_count
  );
}

loco_t::shape_t loco_t::gradient_t::push_back(const properties_t& properties) {
  kps_t::common_t KeyPack;
  KeyPack.ShapeType = shape_type;
  KeyPack.depth = properties.position.z;
  KeyPack.blending = properties.blending;
  KeyPack.camera = properties.camera;
  KeyPack.viewport = properties.viewport;
  vi_t vi;
  vi.position = properties.position;
  vi.size = properties.size;
  vi.color = properties.color;
  vi.angle = properties.angle;
  vi.rotation_point = properties.rotation_point;
  ri_t ri;

  return shape_add(shape_type, vi, ri,
    Key_e::depth, (uint16_t)properties.position.z,
    Key_e::blending, (uint8_t)properties.blending,
    Key_e::viewport, properties.viewport,
    Key_e::camera, properties.camera,
    Key_e::ShapeType, shape_type,
    Key_e::draw_mode, properties.draw_mode,
    Key_e::vertex_count, properties.vertex_count
  );
}

loco_t::shape_t loco_t::shader_shape_t::push_back(const properties_t& properties) {
  //KeyPack.ShapeType = shape_type;
  vi_t vi;
  vi.position = properties.position;
  vi.size = properties.size;
  vi.rotation_point = properties.rotation_point;
  vi.color = properties.color;
  vi.angle = properties.angle;
  vi.flags = properties.flags;
  vi.tc_position = properties.tc_position;
  vi.tc_size = properties.tc_size;
  vi.parallax_factor = properties.parallax_factor;
  vi.seed = properties.seed;
  ri_t ri;
  ri.images = properties.images;
  loco_t::shape_t ret = shape_add(shape_type, vi, ri,
    Key_e::depth, (uint16_t)properties.position.z,
    Key_e::blending, (uint8_t)properties.blending,
    Key_e::image, properties.image,
    Key_e::viewport, properties.viewport,
    Key_e::camera, properties.camera,
    Key_e::ShapeType, shape_type,
    Key_e::draw_mode, properties.draw_mode,
    Key_e::vertex_count, properties.vertex_count
  );
  gloco->shaper.GetShader(shape_type) = properties.shader;
  return ret;
}


loco_t::shape_t loco_t::rectangle3d_t::push_back(const properties_t& properties) {
  vi_t vi;
  vi.position = properties.position;
  vi.size = properties.size;
  vi.color = properties.color;
  //vi.angle = properties.angle;
  ri_t ri;

  loco_t& loco = *OFFSETLESS(this, loco_t, rectangle3d);

  if (loco.window.renderer == loco_t::renderer_t::opengl) {
    if ((loco.context.gl.opengl.major > 3) || (loco.context.gl.opengl.major == 3 && loco.context.gl.opengl.minor >= 3)) {
      // might not need depth
      return shape_add(shape_type, vi, ri,
        Key_e::depth, (uint16_t)properties.position.z,
        Key_e::blending, (uint8_t)properties.blending,
        Key_e::viewport, properties.viewport,
        Key_e::camera, properties.camera,
        Key_e::ShapeType, shape_type,
        Key_e::draw_mode, properties.draw_mode,
        Key_e::vertex_count, properties.vertex_count
      );
    }
    else {
      vi_t vertices[36];
      for (int i = 0; i < 36; i++) {
        vertices[i] = vi;
      }

      return shape_add(shape_type, vertices[0], ri,
        Key_e::depth, (uint16_t)properties.position.z,
        Key_e::blending, (uint8_t)properties.blending,
        Key_e::viewport, properties.viewport,
        Key_e::camera, properties.camera,
        Key_e::ShapeType, shape_type,
        Key_e::draw_mode, properties.draw_mode,
        Key_e::vertex_count, properties.vertex_count
      );
    }
  }
  else if (loco.window.renderer == loco_t::renderer_t::vulkan) {
    
  }
  fan::throw_error();
  return{};
}

loco_t::shape_t loco_t::line3d_t::push_back(const properties_t& properties) {
  vi_t vi;
  vi.src = properties.src;
  vi.dst = properties.dst;
  vi.color = properties.color;
  ri_t ri;

  return shape_add(shape_type, vi, ri,
    Key_e::depth, (uint16_t)properties.src.z,
    Key_e::blending, (uint8_t)properties.blending,
    Key_e::viewport, properties.viewport,
    Key_e::camera, properties.camera,
    Key_e::ShapeType, shape_type,
    Key_e::draw_mode, properties.draw_mode,
    Key_e::vertex_count, properties.vertex_count
  );
}

//-------------------------------------shapes-------------------------------------


std::vector<uint8_t> loco_t::create_noise_image_data(const fan::vec2& image_size, int seed) {
  FastNoiseLite noise;
  noise.SetFractalType(FastNoiseLite::FractalType_FBm);
  noise.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
  noise.SetFrequency(0.010);
  noise.SetFractalGain(0.5);
  noise.SetFractalLacunarity(2.0);
  noise.SetFractalOctaves(5);
  noise.SetSeed(seed);
  noise.SetFractalPingPongStrength(2.0);
  f32_t noise_tex_min = -1;
  f32_t noise_tex_max = 0.1;
  //noise
  // Gather noise data
  std::vector<uint8_t> noiseDataRGB(image_size.multiply() * 4);

  int index = 0;

  float scale = 255 / (noise_tex_max - noise_tex_min);

  for (int y = 0; y < image_size.y; y++)
  {
    for (int x = 0; x < image_size.x; x++)
    {
      float noiseValue = noise.GetNoise((float)x, (float)y);
      unsigned char cNoise = (unsigned char)std::max(0.0f, std::min(255.0f, (noiseValue - noise_tex_min) * scale));
      noiseDataRGB[index * 4 + 0] = cNoise;
      noiseDataRGB[index * 4 + 1] = cNoise;
      noiseDataRGB[index * 4 + 2] = cNoise;
      noiseDataRGB[index * 4 + 3] = 255;
      index++;
    }
  }

  return noiseDataRGB;
}


loco_t::shader_t loco_t::get_sprite_vertex_shader(const fan::string& fragment) {
  loco_t::shader_t shader = shader_create();
  shader_set_vertex(
    shader,
    loco_t::read_shader("shaders/opengl/2D/objects/sprite.vs")
  );
  shader_set_fragment(shader, fragment);
  if (!shader_compile(shader)) {
    shader_erase(shader);
    shader.sic();
  }
  return shader;
}
#if defined(loco_json)
[[nodiscard]]
std::pair<size_t, size_t> fan::json_stream_parser_t::find_next_json_bounds(std::string_view s, size_t pos) const noexcept {
  pos = s.find('{', pos);
  if (pos == std::string::npos) return { pos, pos };

  int depth = 0;
  bool in_str = false;

  for (size_t i = pos; i < s.length(); i++) {
    char c = s[i];
    if (c == '"' && (i == 0 || s[i - 1] != '\\')) in_str = !in_str;
    else if (!in_str) {
      if (c == '{') depth++;
      else if (c == '}' && --depth == 0) return { pos, i + 1 };
    }
  }
  return { pos, std::string::npos };
}

std::vector<fan::json_stream_parser_t::parsed_result> fan::json_stream_parser_t::process(std::string_view chunk) {
  std::vector<parsed_result> results;
  buf += chunk;
  size_t pos = 0;

  while (pos < buf.length()) {
    auto [start, end] = find_next_json_bounds(buf, pos);
    if (start == std::string::npos) break;
    if (end == std::string::npos) {
      buf = buf.substr(start);
      break;
    }

    try {
      results.push_back({ true, fan::json::parse(buf.data() + start, buf.data() + end - start), "" });
    }
    catch (const fan::json::parse_error& e) {
      results.push_back({ false, fan::json{}, e.what() });
    }

    pos = buf.find('{', end);
    if (pos == std::string::npos) pos = end;
  }

  buf = pos < buf.length() ? buf.substr(pos) : "";
  return results;
}

#endif

loco_t::image_t loco_t::create_noise_image(const fan::vec2& image_size) {

  loco_t::image_load_properties_t lp;
  lp.format = GL_RGBA; // Change this to GL_RGB
  lp.internal_format = GL_RGBA; // Change this to GL_RGB
  lp.min_filter = fan::graphics::image_filter::linear;
  lp.mag_filter = fan::graphics::image_filter::linear;
  lp.visual_output = GL_MIRRORED_REPEAT;

  loco_t::image_t image;

  auto noise_data = create_noise_image_data(image_size);

  fan::image::image_info_t ii;
  ii.data = noise_data.data();
  ii.size = image_size;
  ii.channels = 3;

  image = image_load(ii, lp);
  return image;
}

loco_t::image_t loco_t::create_noise_image(const fan::vec2& image_size, const std::vector<uint8_t>& noise_data) {

  loco_t::image_load_properties_t lp;
  lp.format = GL_RGBA; // Change this to GL_RGB
  lp.internal_format = GL_RGBA; // Change this to GL_RGB
  lp.min_filter = fan::graphics::image_filter::linear;
  lp.mag_filter = fan::graphics::image_filter::linear;
  lp.visual_output = GL_MIRRORED_REPEAT;

  loco_t::image_t image;

  fan::image::image_info_t ii;
  ii.data = (void*)noise_data.data();
  ii.size = image_size;

  image = image_load(ii, lp);
  return image;
}

fan::ray3_t loco_t::convert_mouse_to_ray(const fan::vec2i& mouse_position, const fan::vec2& screen_size, const fan::vec3& camera_position, const fan::mat4& projection, const fan::mat4& view) {

  fan::vec4 ray_ndc((2.0f * mouse_position.x) / screen_size.x - 1.0f, 1.0f - (2.0f * mouse_position.y) / screen_size.y, 1.0f, 1.0f);

  fan::mat4 inverted_projection = projection.inverse();

  fan::vec4 ray_clip = inverted_projection * ray_ndc;

  ray_clip.z = -1.0f;
  ray_clip.w = 0.0f;

  fan::mat4 inverted_view = view.inverse();

  fan::vec4 ray_world = inverted_view * ray_clip;

  fan::vec3 ray_dir = fan::vec3(ray_world.x, ray_world.y, ray_world.z).normalize();

  fan::vec3 ray_origin = camera_position;
  return fan::ray3_t(ray_origin, ray_dir);
}

fan::ray3_t loco_t::convert_mouse_to_ray(const fan::vec3& camera_position, const fan::mat4& projection, const fan::mat4& view) {
  return convert_mouse_to_ray(get_mouse_position(), window.get_size(), camera_position, projection, view);
}

fan::ray3_t loco_t::convert_mouse_to_ray(const fan::mat4& projection, const fan::mat4& view) {
  return convert_mouse_to_ray(get_mouse_position(), window.get_size(), camera_get_position(perspective_camera.camera), projection, view);
}

bool loco_t::is_ray_intersecting_cube(const fan::ray3_t& ray, const fan::vec3& position, const fan::vec3& size) {
  fan::vec3 min_bounds = position - size;
  fan::vec3 max_bounds = position + size;

  fan::vec3 t_min = (min_bounds - ray.origin) / (ray.direction + fan::vec3(1e-6f));
  fan::vec3 t_max = (max_bounds - ray.origin) / (ray.direction + fan::vec3(1e-6f));

  fan::vec3 t1 = t_min.min(t_max);
  fan::vec3 t2 = t_min.max(t_max);

  float t_near = fan::max(t1.x, fan::max(t1.y, t1.z));
  float t_far = fan::min(t2.x, fan::min(t2.y, t2.z));

  return t_near <= t_far && t_far >= 0.0f;
}

#if defined(loco_box2d)
loco_t::physics_update_cbs_t::nr_t loco_t::add_physics_update(const physics_update_data_t& data) {
  auto it = shape_physics_update_cbs.NewNodeLast();
  shape_physics_update_cbs[it] = data;
  return it;
}
void loco_t::remove_physics_update(loco_t::physics_update_cbs_t::nr_t nr) {
  shape_physics_update_cbs.unlrec(nr);
}
#endif

#if defined(loco_imgui)
void fan::graphics::text_partial_render(const std::string& text, size_t render_pos, f32_t wrap_width, f32_t line_spacing) {
  static auto find_next_word = [](const std::string& str, std::size_t offset) -> std::size_t {
    std::size_t found = str.find(' ', offset);
    if (found == std::string::npos) {
      found = str.size();
    }
    if (found != std::string::npos) {
    }
    return found;
    };
  static auto find_previous_word = [](const std::string& str, std::size_t offset) -> std::size_t {
    std::size_t found = str.rfind(' ', offset);
    if (found == std::string::npos) {
      found = str.size();
    }
    if (found != std::string::npos) {
    }
    return found;
  };

  std::vector<std::string> lines;
  std::size_t previous_word = 0;
  std::size_t previous_push = 0;
  bool found = false;
  for (std::size_t i = 0; i < text.size(); ++i) {
    std::size_t word_index = text.find(' ', i);
    if (word_index == std::string::npos) {
      word_index = text.size();
    }

    std::string str = text.substr(previous_push, word_index - previous_push);
    f32_t width = ImGui::CalcTextSize(str.c_str()).x;

    if (width >= wrap_width) {
      if (previous_push == previous_word) {
        lines.push_back(text.substr(previous_push, i - previous_push));
        previous_push = i;
      }
      else {
        lines.push_back(text.substr(previous_push, previous_word - previous_push));
        previous_push = previous_word + 1;
        i = previous_word;
      }
    }
    previous_word = word_index;
    i = word_index;
  }

  // Add remaining text as last line
  if (previous_push < text.size()) {
    lines.push_back(text.substr(previous_push));
  }

  std::size_t empty_lines = 0;
  std::size_t character_offset = 0;
  ImVec2 pos = ImGui::GetCursorScreenPos();
  for (const auto& line : lines) {
    if (line.empty()) {
      ++empty_lines;
      continue;
    }
    std::size_t empty = 0;
    if (empty >= line.size()) {
      break;
    }
    while (line[empty] == ' ') {
      if (empty >= line.size()) {
        break;
      }
      ++empty;
    }
    if (character_offset >= render_pos) {
      break;
    }
    std::string render_text = line.substr(empty).c_str();
    ImGui::SetCursorScreenPos(pos);
    if (character_offset + render_text.size() >= render_pos) {
      ImGui::TextUnformatted(render_text.substr(0, render_pos - character_offset).c_str());
      break;
    }
    else {
      ImGui::TextUnformatted(render_text.c_str());
      if (render_text.back() != ' ') {
        character_offset += 1;
      }
      character_offset += render_text.size();
      pos.y += ImGui::GetTextLineHeightWithSpacing() + line_spacing;
    }
  }
  if (empty_lines) {
    ImGui::TextColored(fan::colors::red, "warning empty lines:%zu", empty_lines);
  }
}

#endif

#if defined(loco_imgui)
// fan_track_allocations() must be called in global scope before calling this function
void fan::graphics::render_allocations_plot() {
#if defined(fan_tracking_allocations)
  static std::vector<f32_t> allocation_sizes;
  static std::vector<fan::heap_profiler_t::memory_data_t> allocations;

  allocation_sizes.clear();
  allocations.clear();


  f32_t max_y = 0;
  for (const auto& entry : fan::heap_profiler_t::instance().memory_map) {
    f32_t v = (f32_t)entry.second.n / (1024 * 1024);
    if (v < 0.001) {
      continue;
    }
    allocation_sizes.push_back(v);
    max_y = std::max(max_y, v);
    allocations.push_back(entry.second);
  }
  static fan::heap_profiler_t::stacktrace_t stack;
  if (ImPlot::BeginPlot("Memory Allocations", ImGui::GetWindowSize(), ImPlotFlags_NoFrame | ImPlotFlags_NoLegend)) {
    float max_allocation = *std::max_element(allocation_sizes.begin(), allocation_sizes.end());
    ImPlot::SetupAxis(ImAxis_Y1, "Memory (MB)");
    ImPlot::SetupAxisLimits(ImAxis_Y1, 0, max_y);
    ImPlot::SetupAxis(ImAxis_X1, "Allocations");
    ImPlot::SetupAxisLimits(ImAxis_X1, 0, static_cast<double>(allocation_sizes.size()));

    ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.25f);
    ImPlot::PlotBars("Allocations", allocation_sizes.data(), allocation_sizes.size());
    //if (ImPlot::IsPlotHovered()) {
    //  fan::print("A");
    //}
    ImPlot::PopStyleVar();

    bool hovered = false;
    if (ImPlot::IsPlotHovered()) {
      ImPlotPoint mouse = ImPlot::GetPlotMousePos();
      f32_t half_width = 0.25;
      //mouse.x             = ImPlot::RoundTime(ImPlotTime::FromDouble(mouse.x), ImPlotTimeUnit_Day).ToDouble();
      mouse.x = (int)mouse.x;
      float  tool_l = ImPlot::PlotToPixels(mouse.x - half_width * 1.5, mouse.y).x;
      float  tool_r = ImPlot::PlotToPixels(mouse.x + half_width * 1.5, mouse.y).x;
      float  tool_t = ImPlot::GetPlotPos().y;
      float  tool_b = tool_t + ImPlot::GetPlotSize().y;
      ImPlot::PushPlotClipRect();
      auto draw_list = ImGui::GetWindowDrawList();
      draw_list->AddRectFilled(ImVec2(tool_l, tool_t), ImVec2(tool_r, tool_b), IM_COL32(128, 128, 128, 64));
      ImPlot::PopPlotClipRect();

      if (mouse.x >= 0 && mouse.x < allocation_sizes.size()) {
        if (ImGui::IsMouseClicked(0)) {
          ImGui::OpenPopup("view stack");
        }
        stack = allocations[(int)mouse.x].line_data;
        hovered = true;
      }
    }
    if (hovered) {
      ImGui::BeginTooltip();
      std::ostringstream oss;
      oss << stack;
      std::string stack_str = oss.str();
      std::string final_str;
      std::size_t pos = 0;
      while (true) {
        auto end = stack_str.find(')', pos);
        if (end != std::string::npos) {
          end += 1;
          auto begin = stack_str.rfind('\\', end);
          if (begin != std::string::npos) {
            begin += 1;
            final_str += stack_str.substr(begin, end - begin);
            final_str += "\n";
            pos = end + 1;
          }
          else {
            break;
          }
        }
        else {
          break;
        }
      }
      ImGui::TextUnformatted(final_str.c_str());
      ImGui::EndTooltip();
    }
    if (ImGui::BeginPopup("view stack", ImGuiWindowFlags_AlwaysHorizontalScrollbar)) {
      std::ostringstream oss;
      oss << stack;
      ImGui::TextUnformatted(oss.str().c_str());
      ImGui::EndPopup();
    }
    ImPlot::EndPlot();
  }
#endif
}
#endif

bool loco_t::is_mouse_clicked(int button) {
  return window.key_state(button) == (int)fan::mouse_state::press;
}

bool loco_t::is_mouse_down(int button) {
  int state = window.key_state(button);
  return 
    state == (int)fan::mouse_state::press ||
    state == (int)fan::mouse_state::repeat;
}

bool loco_t::is_mouse_released(int button) {
  return window.key_state(button) == (int)fan::mouse_state::release;
}

fan::vec2 loco_t::get_mouse_drag(int button) {
  if (is_mouse_down(button)) {
    if (window.drag_delta_start != -1) {
      return window.get_mouse_position() - window.drag_delta_start;
    }
  }
  return fan::vec2();
}

void fan::graphics::add_input_action(const int* keys, std::size_t count, std::string_view action_name) {
  gloco->input_action.add(keys, count, action_name);
}
void fan::graphics::add_input_action(std::initializer_list<int> keys, std::string_view action_name) {
  gloco->input_action.add(keys, action_name);
}
void fan::graphics::add_input_action(int key, std::string_view action_name) {
  gloco->input_action.add(key, action_name);
}
bool fan::graphics::is_input_action_active(std::string_view action_name, int pstate) {
  return gloco->input_action.is_active(action_name);
}
#if defined(loco_imgui)
bool fan::graphics::gui::render_blank_window(const std::string& name) {
  ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
  ImGui::SetNextWindowPos(ImVec2(0, 0));
  return ImGui::Begin(name.c_str(), 0,
    ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse | 
    ImGuiWindowFlags_NoResize | 
    ImGuiWindowFlags_NoTitleBar
  );
}
#endif

#if defined(loco_json)
bool fan::graphics::shape_to_json(loco_t::shape_t& shape, fan::json* json) {
  fan::json& out = *json;
  switch (shape.get_shape_type()) {
  case loco_t::shape_type_t::light: {
    out["shape"] = "light";
    out["position"] = shape.get_position();
    out["parallax_factor"] = shape.get_parallax_factor();
    out["size"] = shape.get_size();
    out["rotation_point"] = shape.get_rotation_point();
    out["color"] = shape.get_color();
    out["rotation_vector"] = shape.get_rotation_vector();
    out["flags"] = shape.get_flags();
    out["angle"] = shape.get_angle();
    break;
  }
  case loco_t::shape_type_t::line: {
    out["shape"] = "line";
    out["color"] = shape.get_color();
    out["src"] = shape.get_src();
    out["dst"] = shape.get_dst();
    break;
  }
  case loco_t::shape_type_t::rectangle: {
    out["shape"] = "rectangle";
    out["position"] = shape.get_position();
    out["size"] = shape.get_size();
    out["rotation_point"] = shape.get_rotation_point();
    out["color"] = shape.get_color();
    out["angle"] = shape.get_angle();
    break;
  }
  case loco_t::shape_type_t::sprite: {
    out["shape"] = "sprite";
    out["position"] = shape.get_position();
    out["parallax_factor"] = shape.get_parallax_factor();
    out["size"] = shape.get_size();
    out["rotation_point"] = shape.get_rotation_point();
    out["color"] = shape.get_color();
    out["angle"] = shape.get_angle();
    out["flags"] = shape.get_flags();
    out["tc_position"] = shape.get_tc_position();
    out["tc_size"] = shape.get_tc_size();
    break;
  }
  case loco_t::shape_type_t::unlit_sprite: {
    out["shape"] = "unlit_sprite";
    out["position"] = shape.get_position();
    out["parallax_factor"] = shape.get_parallax_factor();
    out["size"] = shape.get_size();
    out["rotation_point"] = shape.get_rotation_point();
    out["color"] = shape.get_color();
    out["angle"] = shape.get_angle();
    out["flags"] = shape.get_flags();
    out["tc_position"] = shape.get_tc_position();
    out["tc_size"] = shape.get_tc_size();
    break;
  }
  case loco_t::shape_type_t::text: {
    out["shape"] = "text";
    break;
  }
  case loco_t::shape_type_t::circle: {
    out["shape"] = "circle";
    out["position"] = shape.get_position();
    out["radius"] = shape.get_radius();
    out["rotation_point"] = shape.get_rotation_point();
    out["color"] = shape.get_color();
    out["rotation_vector"] = shape.get_rotation_vector();
    out["angle"] = shape.get_angle();
    break;
  }
  case loco_t::shape_type_t::grid: {
    out["shape"] = "grid";
    out["position"] = shape.get_position();
    out["size"] = shape.get_size();
    out["grid_size"] = shape.get_grid_size();
    out["rotation_point"] = shape.get_rotation_point();
    out["color"] = shape.get_color();
    out["angle"] = shape.get_angle();
    break;
  }
  case loco_t::shape_type_t::particles: {
    auto& ri = *(loco_t::particles_t::ri_t*)shape.GetData(gloco->shaper);
    out["shape"] = "particles";
    out["position"] = ri.position;
    out["size"] = ri.size;
    out["color"] = ri.color;
    out["begin_time"] = ri.begin_time;
    out["alive_time"] = ri.alive_time;
    out["respawn_time"] = ri.respawn_time;
    out["count"] = ri.count;
    out["position_velocity"] = ri.position_velocity;
    out["angle_velocity"] = ri.angle_velocity;
    out["begin_angle"] = ri.begin_angle;
    out["end_angle"] = ri.end_angle;
    out["angle"] = ri.angle;
    out["gap_size"] = ri.gap_size;
    out["max_spread_size"] = ri.max_spread_size;
    out["size_velocity"] = ri.size_velocity;
    out["particle_shape"] = ri.shape;
    out["blending"] = ri.blending;
    break;
  }
  default: {
    fan::throw_error("unimplemented shape");
  }
  }
  return false;
}
bool fan::graphics::json_to_shape(const fan::json& in, loco_t::shape_t* shape) {
  std::string shape_type = in["shape"];
  switch (fan::get_hash(shape_type.c_str())) {
  case fan::get_hash("rectangle"): {
    loco_t::rectangle_t::properties_t p;
    p.position = in["position"];
    p.size = in["size"];
    p.rotation_point = in["rotation_point"];
    p.color = in["color"];
    p.angle = in["angle"];
    *shape = p;
    break;
  }
    case fan::get_hash("light"): {
    loco_t::light_t::properties_t p;
    p.position = in["position"];
    p.parallax_factor = in["parallax_factor"];
    p.size = in["size"];
    p.rotation_point = in["rotation_point"];
    p.color = in["color"];
    p.rotation_vector = in["rotation_vector"];
    p.flags = in["flags"];
    p.angle = in["angle"];
    *shape = p;
    break;
  }
  case fan::get_hash("line"): {
    loco_t::line_t::properties_t p;
    p.color = in["color"];
    p.src = in["src"];
    p.dst = in["dst"];
    *shape = p;
    break;
  }
  case fan::get_hash("sprite"): {
    loco_t::sprite_t::properties_t p;
    p.blending = true;
    p.position = in["position"];
    p.parallax_factor = in["parallax_factor"];
    p.size = in["size"];
    p.rotation_point = in["rotation_point"];
    p.color = in["color"];
    p.angle = in["angle"];
    p.flags = in["flags"];
    p.tc_position = in["tc_position"];
    p.tc_size = in["tc_size"];
    *shape = p;
    break;
  }
  case fan::get_hash("unlit_sprite"): {
    loco_t::unlit_sprite_t::properties_t p;
    p.blending = true;
    p.position = in["position"];
    p.parallax_factor = in["parallax_factor"];
    p.size = in["size"];
    p.rotation_point = in["rotation_point"];
    p.color = in["color"];
    p.angle = in["angle"];
    p.flags = in["flags"];
    p.tc_position = in["tc_position"];
    p.tc_size = in["tc_size"];
    *shape = p;
    break;
  }
  case fan::get_hash("circle"): {
    loco_t::circle_t::properties_t p;
    p.position = in["position"];
    p.radius = in["radius"];
    p.rotation_point = in["rotation_point"];
    p.color = in["color"];
    p.rotation_vector = in["rotation_vector"];
    p.angle = in["angle"];
    *shape = p;
    break;
  }
  case fan::get_hash("grid"): {
    loco_t::grid_t::properties_t p;
    p.position = in["position"];
    p.size = in["size"];
    p.grid_size = in["grid_size"];
    p.rotation_point = in["rotation_point"];
    p.color = in["color"];
    p.angle = in["angle"];
    *shape = p;
    break;
  }
  case fan::get_hash("particles"): {
    loco_t::particles_t::properties_t p;
    p.position = in["position"];
    p.size = in["size"];
    p.color = in["color"];
    p.begin_time = in["begin_time"];
    p.alive_time = in["alive_time"];
    p.respawn_time = in["respawn_time"];
    p.count = in["count"];
    p.position_velocity = in["position_velocity"];
    p.angle_velocity = in["angle_velocity"];
    p.begin_angle = in["begin_angle"];
    p.end_angle = in["end_angle"];
    p.angle = in["angle"];
    p.gap_size = in["gap_size"];
    p.max_spread_size = in["max_spread_size"];
    p.size_velocity = in["size_velocity"];
    p.shape = in["particle_shape"];
    p.blending = in["blending"];
    *shape = p;
    break;
  }
  default: {
    fan::throw_error("unimplemented shape");
  }
  }
  return false;
}
bool fan::graphics::shape_serialize(loco_t::shape_t& shape, fan::json* out) {
  return shape_to_json(shape, out);
}
bool fan::graphics::shape_to_bin(loco_t::shape_t& shape, std::vector<uint8_t>* data) {
  std::vector<uint8_t>& out = *data;
  fan::write_to_vector(out, shape.get_shape_type());
  fan::write_to_vector(out, shape.gint());
  switch (shape.get_shape_type()) {
  case loco_t::shape_type_t::light: {
    fan::write_to_vector(out, shape.get_position());
    fan::write_to_vector(out, shape.get_parallax_factor());
    fan::write_to_vector(out, shape.get_size());
    fan::write_to_vector(out, shape.get_rotation_point());
    fan::write_to_vector(out, shape.get_color());
    fan::write_to_vector(out, shape.get_rotation_vector());
    fan::write_to_vector(out, shape.get_flags());
    fan::write_to_vector(out, shape.get_angle());
    break;
  }
  case loco_t::shape_type_t::line: {
    fan::write_to_vector(out, shape.get_color());
    fan::write_to_vector(out, shape.get_src());
    fan::write_to_vector(out, shape.get_dst());
    break;
    case loco_t::shape_type_t::rectangle: {
    fan::write_to_vector(out, shape.get_position());
    fan::write_to_vector(out, shape.get_size());
    fan::write_to_vector(out, shape.get_rotation_point());
    fan::write_to_vector(out, shape.get_color());
    fan::write_to_vector(out, shape.get_angle());
    break;
    }
    case loco_t::shape_type_t::sprite: {
    fan::write_to_vector(out, shape.get_position());
    fan::write_to_vector(out, shape.get_parallax_factor());
    fan::write_to_vector(out, shape.get_size());
    fan::write_to_vector(out, shape.get_rotation_point());
    fan::write_to_vector(out, shape.get_color());
    fan::write_to_vector(out, shape.get_angle());
    fan::write_to_vector(out, shape.get_flags());
    fan::write_to_vector(out, shape.get_tc_position());
    fan::write_to_vector(out, shape.get_tc_size());
    break;
    }
    case loco_t::shape_type_t::unlit_sprite: {
    fan::write_to_vector(out, shape.get_position());
    fan::write_to_vector(out, shape.get_parallax_factor());
    fan::write_to_vector(out, shape.get_size());
    fan::write_to_vector(out, shape.get_rotation_point());
    fan::write_to_vector(out, shape.get_color());
    fan::write_to_vector(out, shape.get_angle());
    fan::write_to_vector(out, shape.get_flags());
    fan::write_to_vector(out, shape.get_tc_position());
    fan::write_to_vector(out, shape.get_tc_size());
    break;
    }
    case loco_t::shape_type_t::circle: {
    fan::write_to_vector(out, shape.get_position());
    fan::write_to_vector(out, shape.get_radius());
    fan::write_to_vector(out, shape.get_rotation_point());
    fan::write_to_vector(out, shape.get_color());
    fan::write_to_vector(out, shape.get_rotation_vector());
    fan::write_to_vector(out, shape.get_angle());
    break;
    }
    case loco_t::shape_type_t::grid: {
    fan::write_to_vector(out, shape.get_position());
    fan::write_to_vector(out, shape.get_size());
    fan::write_to_vector(out, shape.get_grid_size());
    fan::write_to_vector(out, shape.get_rotation_point());
    fan::write_to_vector(out, shape.get_color());
    fan::write_to_vector(out, shape.get_angle());
    break;
    }
    case loco_t::shape_type_t::particles: {
    auto& ri = *(loco_t::particles_t::ri_t*)shape.GetData(gloco->shaper);
    fan::write_to_vector(out, ri.position);
    fan::write_to_vector(out, ri.size);
    fan::write_to_vector(out, ri.color);
    fan::write_to_vector(out, ri.begin_time);
    fan::write_to_vector(out, ri.alive_time);
    fan::write_to_vector(out, ri.respawn_time);
    fan::write_to_vector(out, ri.count);
    fan::write_to_vector(out, ri.position_velocity);
    fan::write_to_vector(out, ri.angle_velocity);
    fan::write_to_vector(out, ri.begin_angle);
    fan::write_to_vector(out, ri.end_angle);
    fan::write_to_vector(out, ri.angle);
    fan::write_to_vector(out, ri.gap_size);
    fan::write_to_vector(out, ri.max_spread_size);
    fan::write_to_vector(out, ri.size_velocity);
    fan::write_to_vector(out, ri.shape);
    fan::write_to_vector(out, ri.blending);
    break;
    }
  }
  case loco_t::shape_type_t::light_end: {
    break;
  }
  default: {
    fan::throw_error("unimplemented shape");
  }
  }
  return false;
}
bool fan::graphics::bin_to_shape(const std::vector<uint8_t>& in, loco_t::shape_t* shape, uint64_t& offset) {
  using sti_t = std::remove_reference_t<decltype(loco_t::shape_t().get_shape_type())>;
  using nr_t = std::remove_reference_t<decltype(loco_t::shape_t().gint())>;
  sti_t shape_type = fan::vector_read_data<sti_t>(in, offset);
  nr_t nri = fan::vector_read_data<nr_t>(in, offset);
  switch (shape_type) {
  case loco_t::shape_type_t::rectangle: {
    loco_t::rectangle_t::properties_t p;
    p.position = fan::vector_read_data<decltype(p.position)>(in, offset);
    p.size = fan::vector_read_data<decltype(p.size)>(in, offset);
    p.rotation_point = fan::vector_read_data<decltype(p.rotation_point)>(in, offset);
    p.color = fan::vector_read_data<decltype(p.color)>(in, offset);
    p.angle = fan::vector_read_data<decltype(p.angle)>(in, offset);
    p.outline_color = p.color;
    *shape = p;
    return false;
  }
  case loco_t::shape_type_t::light: {
    loco_t::light_t::properties_t p;
    p.position = fan::vector_read_data<decltype(p.position)>(in, offset);
    p.parallax_factor = fan::vector_read_data<decltype(p.parallax_factor)>(in, offset);
    p.size = fan::vector_read_data<decltype(p.size)>(in, offset);
    p.rotation_point = fan::vector_read_data<decltype(p.rotation_point)>(in, offset);
    p.color = fan::vector_read_data<decltype(p.color)>(in, offset);
    p.rotation_vector = fan::vector_read_data<decltype(p.rotation_vector)>(in, offset);
    p.flags = fan::vector_read_data<decltype(p.flags)>(in, offset);
    p.angle = fan::vector_read_data<decltype(p.angle)>(in, offset);
    *shape = p;
    break;
  }
  case loco_t::shape_type_t::line: {
    loco_t::line_t::properties_t p;
    p.color = fan::vector_read_data<decltype(p.color)>(in, offset);
    p.src = fan::vector_read_data<decltype(p.src)>(in, offset);
    p.dst = fan::vector_read_data<decltype(p.dst)>(in, offset);
    *shape = p;
    break;
  }
  case loco_t::shape_type_t::sprite: {
    loco_t::sprite_t::properties_t p;
    p.position = fan::vector_read_data<decltype(p.position)>(in, offset);
    p.parallax_factor = fan::vector_read_data<decltype(p.parallax_factor)>(in, offset);
    p.size = fan::vector_read_data<decltype(p.size)>(in, offset);
    p.rotation_point = fan::vector_read_data<decltype(p.rotation_point)>(in, offset);
    p.color = fan::vector_read_data<decltype(p.color)>(in, offset);
    p.angle = fan::vector_read_data<decltype(p.angle)>(in, offset);
    p.flags = fan::vector_read_data<decltype(p.flags)>(in, offset);
    p.tc_position = fan::vector_read_data<decltype(p.tc_position)>(in, offset);
    p.tc_size = fan::vector_read_data<decltype(p.tc_size)>(in, offset);
    *shape = p;
    break;
  }
  case loco_t::shape_type_t::unlit_sprite: {
    loco_t::unlit_sprite_t::properties_t p;
    p.position = fan::vector_read_data<decltype(p.position)>(in, offset);
    p.parallax_factor = fan::vector_read_data<decltype(p.parallax_factor)>(in, offset);
    p.size = fan::vector_read_data<decltype(p.size)>(in, offset);
    p.rotation_point = fan::vector_read_data<decltype(p.rotation_point)>(in, offset);
    p.color = fan::vector_read_data<decltype(p.color)>(in, offset);
    p.angle = fan::vector_read_data<decltype(p.angle)>(in, offset);
    p.flags = fan::vector_read_data<decltype(p.flags)>(in, offset);
    p.tc_position = fan::vector_read_data<decltype(p.tc_position)>(in, offset);
    p.tc_size = fan::vector_read_data<decltype(p.tc_size)>(in, offset);
    *shape = p;
    break;
  }
  case loco_t::shape_type_t::circle: {
    loco_t::circle_t::properties_t p;
    p.position = fan::vector_read_data<decltype(p.position)>(in, offset);
    p.radius = fan::vector_read_data<decltype(p.radius)>(in, offset);
    p.rotation_point = fan::vector_read_data<decltype(p.rotation_point)>(in, offset);
    p.color = fan::vector_read_data<decltype(p.color)>(in, offset);
    p.rotation_vector = fan::vector_read_data<decltype(p.rotation_vector)>(in, offset);
    p.angle = fan::vector_read_data<decltype(p.angle)>(in, offset);
    *shape = p;
    break;
  }
  case loco_t::shape_type_t::grid: {
    loco_t::grid_t::properties_t p;
    p.position = fan::vector_read_data<decltype(p.position)>(in, offset);
    p.size = fan::vector_read_data<decltype(p.size)>(in, offset);
    p.grid_size = fan::vector_read_data<decltype(p.grid_size)>(in, offset);
    p.rotation_point = fan::vector_read_data<decltype(p.rotation_point)>(in, offset);
    p.color = fan::vector_read_data<decltype(p.color)>(in, offset);
    p.angle = fan::vector_read_data<decltype(p.angle)>(in, offset);
    *shape = p;
    break;
  }
  case loco_t::shape_type_t::particles: {
    loco_t::particles_t::properties_t p;
    p.position = fan::vector_read_data<decltype(p.position)>(in, offset);
    p.size = fan::vector_read_data<decltype(p.size)>(in, offset);
    p.color = fan::vector_read_data<decltype(p.color)>(in, offset);
    p.begin_time = fan::vector_read_data<decltype(p.begin_time)>(in, offset);
    p.alive_time = fan::vector_read_data<decltype(p.alive_time)>(in, offset);
    p.respawn_time = fan::vector_read_data<decltype(p.respawn_time)>(in, offset);
    p.count = fan::vector_read_data<decltype(p.count)>(in, offset);
    p.position_velocity = fan::vector_read_data<decltype(p.position_velocity)>(in, offset);
    p.angle_velocity = fan::vector_read_data<decltype(p.angle_velocity)>(in, offset);
    p.begin_angle = fan::vector_read_data<decltype(p.begin_angle)>(in, offset);
    p.end_angle = fan::vector_read_data<decltype(p.end_angle)>(in, offset);
    p.angle = fan::vector_read_data<decltype(p.angle)>(in, offset);
    p.gap_size = fan::vector_read_data<decltype(p.gap_size)>(in, offset);
    p.max_spread_size = fan::vector_read_data<decltype(p.max_spread_size)>(in, offset);
    p.size_velocity = fan::vector_read_data<decltype(p.size_velocity)>(in, offset);
    p.shape = fan::vector_read_data<decltype(p.shape)>(in, offset);
    p.blending = fan::vector_read_data<decltype(p.blending)>(in, offset);
    *shape = p;
    break;
  }
  case loco_t::shape_type_t::light_end: {
    return false;
  }
  default: {
    fan::throw_error("unimplemented");
  }
  }
  if (shape->gint() != nri) {
    fan::throw_error("");
  }
  return false;
}
bool fan::graphics::shape_serialize(loco_t::shape_t& shape, std::vector<uint8_t>* out) {
  return shape_to_bin(shape, out);
}

bool fan::graphics::shape_deserialize_t::iterate(const fan::json& json, loco_t::shape_t* shape) {
  if (init == false) {
    data.it = json.cbegin();
    init = true;
  }
  if (data.it == json.cend()) {
    return 0;
  }
  if (json.type() == fan::json::value_t::object) {
    json_to_shape(json, shape);
    return 0;
  }
  else {
    json_to_shape(*data.it, shape);
    ++data.it;
  }
  return 1;
}

bool fan::graphics::shape_deserialize_t::iterate(const std::vector<uint8_t>& bin_data, loco_t::shape_t* shape) {
  if (bin_data.empty()) {
    return 0;
  }
  else if (data.offset >= bin_data.size()) {
    return 0;
  }
  bin_to_shape(bin_data, shape, data.offset);
  return 1;
}
#endif
