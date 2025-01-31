#include "loco.h"

#include <fan/time/time.h>
//#include <fan/memory/memory.hpp>

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

//thread_local global_loco_t gloco;

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

void loco_t::camera_move(fan::opengl::context_t::camera_t& camera, f64_t dt, f32_t movement_speed, f32_t friction) {
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

void loco_t::render_final_fb() {
  fan_opengl_call(glBindVertexArray(fb_vao));
  fan_opengl_call(glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));
  fan_opengl_call(glBindVertexArray(0));
}


void loco_t::initialize_fb_vaos(uint32_t& vao, uint32_t& vbo) {
  static constexpr f32_t quad_vertices[] = {
     -1.0f, 1.0f, 0, 0.0f, 1.0f,
     -1.0f, -1.0f, 0, 0.0f, 0.0f,
     1.0f, 1.0f, 0, 1.0f, 1.0f,
     1.0f, -1.0f, 0, 1.0f, 0.0f,
  };
  fan_opengl_call(glGenVertexArrays(1, &vao));
  fan_opengl_call(glGenBuffers(1, &vbo));
  fan_opengl_call(glBindVertexArray(vao));
  fan_opengl_call(glBindBuffer(GL_ARRAY_BUFFER, vbo));
  fan_opengl_call(glBufferData(GL_ARRAY_BUFFER, sizeof(quad_vertices), &quad_vertices, GL_STATIC_DRAW));
  fan_opengl_call(glEnableVertexAttribArray(0));
  fan_opengl_call(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0));
  fan_opengl_call(glEnableVertexAttribArray(1));
  fan_opengl_call(glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float))));
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
      else {
        fan::throw_error("unimplemented get - for line use get_src()");
        return fan::vec3();
      }
    },
    .set_position2 = [](shape_t* shape, const fan::vec2& position) {
      if constexpr (fan_has_variable(T, position)) {
        modify_render_data_element(shape, &T::position, position);
      }
      else {
        fan::throw_error("unimplemented set - for line use set_src()");
      }
    },
    .set_position3 = [](shape_t* shape, const fan::vec3& position) {
      if constexpr (fan_has_variable(T, position)) {
          auto sti = gloco->shaper.GetSTI(*shape);

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

    
          auto _vi = gloco->shaper.GetRenderData(*shape);
          auto vlen = gloco->shaper.GetRenderDataSize(sti);
          uint8_t* vi = new uint8_t[vlen];
          std::memcpy(vi, _vi, vlen);
          ((T*)vi)->position = position;

          auto _ri = gloco->shaper.GetData(*shape);
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
          fan::throw_error("unimplemented set");
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
          auto sti = gloco->shaper.GetSTI(*shape);
            
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
          auto& img = gloco->image_get_data(im);

          auto _vi = gloco->shaper.GetRenderData(*shape);
          auto vlen = gloco->shaper.GetRenderDataSize(sti);
          uint8_t* vi = new uint8_t[vlen];
          std::memcpy(vi, _vi, vlen);
          ((T*)vi)->tc_position = ti->position / img.size;
          ((T*)vi)->tc_size = ti->size / img.size;

          auto _ri = gloco->shaper.GetData(*shape);
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
        auto sti = gloco->shaper.GetSTI(*shape);

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
            auto sti = gloco->shaper.GetSTI(*shape);

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

          auto _vi = gloco->shaper.GetRenderData(*shape);
          auto vlen = gloco->shaper.GetRenderDataSize(sti);
          uint8_t* vi = new uint8_t[vlen];
          std::memcpy(vi, _vi, vlen);

          auto _ri = gloco->shaper.GetData(*shape);
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

        auto sti = gloco->shaper.GetSTI(*shape);

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
          auto sti = gloco->shaper.GetSTI(*shape);

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

          auto _vi = gloco->shaper.GetRenderData(*shape);
          auto vlen = gloco->shaper.GetRenderDataSize(sti);
          uint8_t* vi = new uint8_t[vlen];
          std::memcpy(vi, _vi, vlen);

          auto _ri = gloco->shaper.GetData(*shape);
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
        auto sti = gloco->shaper.GetSTI(*shape);
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
         
        auto sti = gloco->shaper.GetSTI(*shape);

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
            
        auto _vi = gloco->shaper.GetRenderData(*shape);
        auto vlen = gloco->shaper.GetRenderDataSize(sti);
        uint8_t* vi = new uint8_t[vlen];
        std::memcpy(vi, _vi, vlen);

        auto _ri = gloco->shaper.GetData(*shape);
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
        loco_t::universal_image_renderer_t::ri_t& ri = *(loco_t::universal_image_renderer_t::ri_t*)gloco->shaper.GetData(*shape);
        if (format != ri.format) {
          auto sti = gloco->shaper.GetSTI(*shape);
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
            std::memcpy(ri.images_rest, &images[1], sizeof(ri.images_rest));
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
            gloco->image_reload_pixels(
              vi_image,
              image_info,
              lp
            );
          }
          else {
            gloco->image_reload_pixels(
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
    gloco->shader_set_value(gloco->m_fbo_final_shader, "gamma", std::stof(args[0]));
  }).description = "sets gamma for postprocessing shader";

  loco->console.commands.add("set_gamma", [](const fan::commands_t::arg_t& args) {
    if (args.size() != 1) {
      gloco->console.commands.print_invalid_arg_count();
      return;
    }
    gloco->shader_set_value(gloco->m_fbo_final_shader, "gamma", std::stof(args[0]));
  }).description = "sets gamma for postprocessing shader";

  loco->console.commands.add("set_exposure", [](const fan::commands_t::arg_t& args) {
    if (args.size() != 1) {
      gloco->console.commands.print_invalid_arg_count();
      return;
    }
    gloco->shader_set_value(gloco->m_fbo_final_shader, "exposure", std::stof(args[0]));
  }).description = "sets exposure for postprocessing shader";

  loco->console.commands.add("set_bloom_strength", [](const fan::commands_t::arg_t& args) {
    if (args.size() != 1) {
      gloco->console.commands.print_invalid_arg_count();
      return;
    }
    gloco->shader_set_value(gloco->m_fbo_final_shader, "bloom_strength", std::stof(args[0]));
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

void init_imgui(loco_t* loco) {
#if defined(loco_imgui)
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
    style.WindowRounding = 0.0f;
    style.Colors[ImGuiCol_WindowBg].w = 1.0f;
  }

  //loco_t::imgui_themes::dark();

  glfwMakeContextCurrent(loco->window);
  ImGui_ImplGlfw_InitForOpenGL(loco->window, true);
  const char* glsl_version = "#version 120";
  ImGui_ImplOpenGL3_Init(glsl_version);

  static constexpr const char* font_name = "fonts/PixelatedEleganceRegular-ovyAA.ttf";
  static constexpr f32_t font_size = 4;


  for (std::size_t i = 0; i < std::size(loco->fonts); ++i) {
    loco->fonts[i] = io.Fonts->AddFontFromFileTTF(font_name, (int)(font_size * (1 << i)) * 1.5);

    if (loco->fonts[i] == nullptr) {
      fan::throw_error(fan::string("failed to load font:") + font_name);
    }
  }
  io.Fonts->Build();
  io.FontDefault = loco->fonts[2];
#endif
}

void destroy_imgui() {
#if defined(loco_imgui)
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  ImPlot::DestroyContext();
#endif
}

void loco_t::init_framebuffer() {
  if (!((opengl.major > 3) || (opengl.major == 3 && opengl.minor >= 3))) {
    window.add_resize_callback([&](const auto& d) {
      viewport_set(orthographic_camera.viewport, fan::vec2(0, 0), d.size, d.size);
      viewport_set(perspective_camera.viewport, fan::vec2(0, 0), d.size, d.size);
    });
    return;
  }

#if defined(loco_opengl)
#if defined(loco_framebuffer)
  m_framebuffer.open(*this);
  // can be GL_RGB16F
  m_framebuffer.bind(*this);
#endif
#endif

#if defined(loco_opengl)

#if defined(loco_framebuffer)
  //
  static auto load_texture = [&](fan::image::image_info_t& image_info, loco_t::image_t& color_buffer, GLenum attachment, bool reload = false) {
    typename fan::opengl::context_t::image_load_properties_t load_properties;
    load_properties.visual_output = GL_REPEAT;
    load_properties.internal_format = GL_RGB;
    load_properties.format = GL_RGB;
    load_properties.type = GL_FLOAT;
    load_properties.min_filter = GL_LINEAR;
    load_properties.mag_filter = GL_LINEAR;
    if (reload == true) {
      image_reload_pixels(color_buffer, image_info, load_properties);
    }
    else {
      color_buffer = image_load(image_info, load_properties);
    }
    fan_opengl_call(glGenerateMipmap(GL_TEXTURE_2D));
    image_bind(color_buffer);
    fan::opengl::core::framebuffer_t::bind_to_texture(*this, image_get(color_buffer), attachment);
  };

  fan::image::image_info_t image_info;
  image_info.data = nullptr;
  image_info.size = window.get_size();
  image_info.channels = 3;

  m_framebuffer.bind(*this);
  for (uint32_t i = 0; i < (uint32_t)std::size(color_buffers); ++i) {
    load_texture(image_info, color_buffers[i], GL_COLOR_ATTACHMENT0 + i);
  }

  window.add_resize_callback([&](const auto& d) {
    fan::image::image_info_t image_info;
    image_info.data = nullptr;
    image_info.size = window.get_size();

    m_framebuffer.bind(*this);
    for (uint32_t i = 0; i < (uint32_t)std::size(color_buffers); ++i) {
      load_texture(image_info, color_buffers[i], GL_COLOR_ATTACHMENT0 + i, true);
    }

    fan::opengl::core::renderbuffer_t::properties_t renderbuffer_properties;
    m_framebuffer.bind(*this);
    renderbuffer_properties.size = image_info.size;
    renderbuffer_properties.internalformat = GL_DEPTH_COMPONENT;
    m_rbo.set_storage(*this, renderbuffer_properties);

    fan::vec2 window_size = gloco->window.get_size();

    viewport_set(orthographic_camera.viewport, fan::vec2(0, 0), d.size, d.size);
    viewport_set(perspective_camera.viewport, fan::vec2(0, 0), d.size, d.size);
  });

  fan::opengl::core::renderbuffer_t::properties_t renderbuffer_properties;
  m_framebuffer.bind(*this);
  renderbuffer_properties.size = image_info.size;
  renderbuffer_properties.internalformat = GL_DEPTH_COMPONENT;
  m_rbo.open(*this);
  m_rbo.set_storage(*this, renderbuffer_properties);
  renderbuffer_properties.internalformat = GL_DEPTH_ATTACHMENT;
  m_rbo.bind_to_renderbuffer(*this, renderbuffer_properties);

  unsigned int attachments[sizeof(color_buffers) / sizeof(color_buffers[0])];

  for (uint8_t i = 0; i < std::size(color_buffers); ++i) {
    attachments[i] = GL_COLOR_ATTACHMENT0 + i;
  }

  fan_opengl_call(glDrawBuffers(std::size(attachments), attachments));

  if (!m_framebuffer.ready(*this)) {
    fan::throw_error("framebuffer not ready");
  }


#if defined(loco_post_process)
  static constexpr uint32_t mip_count = 8;
  blur[0].open(window.get_size(), mip_count);

  bloom.open();
#endif

  m_framebuffer.unbind(*this);

  
  m_fbo_final_shader = shader_create();

  shader_set_vertex(
    m_fbo_final_shader,
    read_shader("shaders/opengl/2D/effects/loco_fbo.vs")
  );
  shader_set_fragment(
    m_fbo_final_shader,
    read_shader("shaders/opengl/2D/effects/loco_fbo.fs")
  );
  shader_compile(m_fbo_final_shader);

#endif
#endif
}

loco_t::loco_t() : loco_t(properties_t()) {

}

loco_t::loco_t(const properties_t& p){
  if (fan::init_manager_t::initialized() == false) {
    fan::init_manager_t::initialize();
  }

  start_time = fan::time::clock::now();
  glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);

  GLFWwindow* dummy_window = glfwCreateWindow(640, 400, "dummy", nullptr, nullptr);
  if (dummy_window == nullptr) {
    fan::throw_error("failed to open dummy window");
  }
  glfwMakeContextCurrent(dummy_window);
  context_t::open();
  {
    if (opengl.major == -1 || opengl.minor == -1) {
      const char* gl_version = (const char*)fan_opengl_call(glGetString(GL_VERSION));
      sscanf(gl_version, "%d.%d", &opengl.major, &opengl.minor);
    }
    glfwMakeContextCurrent(nullptr);
    glfwDestroyWindow(dummy_window);
    glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
  }
  {
    #if 1
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, opengl.major);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, opengl.minor);
    glfwWindowHint(GLFW_SAMPLES, 0);

    if ((opengl.major > 3) || (opengl.major == 3 && opengl.minor > 2)) {
      glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    }

    if ((opengl.major > 3) || (opengl.major == 3 && opengl.minor > 0)) {
      glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, true);
    }
  #else // renderdoc debug
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_SAMPLES, 0);

    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, true);
  #endif

    glfwSetErrorCallback(error_callback);
  }
  window.open(p.window_size, fan::window_t::default_window_name, p.visible, p.window_flags);
  gloco = this;
  set_vsync(false); // using libuv
  //fan::print("less pain", this, (void*)&lighting, (void*)((uint8_t*)&lighting - (uint8_t*)this), sizeof(*this), lighting.ambient);
  glfwMakeContextCurrent(window);

#if fan_debug >= fan_debug_high
  get_context().set_error_callback();
#endif

  default_texture = get_context().create_missing_texture();

#if defined(loco_opengl)
  initialize_fb_vaos(fb_vao, fb_vbo);
#endif


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
  
  shape_functions.resize(shape_functions.size() + 1); // button
  {
    if (opengl.major == 2 && opengl.minor == 1) {
      shape_open<loco_t::sprite_t>(
        &sprite,
        "shaders/opengl/2D/objects/sprite_2_1.vs",
        "shaders/opengl/2D/objects/sprite_2_1.fs",
        6 // set instance count to 6 vertices, in opengl 2.1 there is no instancing,
          // so sending same 6 elements per shape
      );
    }
    else {
      shape_open<loco_t::sprite_t>(
        &sprite,
        "shaders/opengl/2D/objects/sprite.vs",
        "shaders/opengl/2D/objects/sprite.fs"
      );
    }
  }

  shape_functions.resize(shape_functions.size() + 1); // text
  shape_functions.resize(shape_functions.size() + 1); // hitbox
  {
    if (opengl.major == 2 && opengl.minor == 1) {
      // todo implement line
      shape_functions.resize(shape_functions.size() + 1);
    }
    else {
      shape_open<loco_t::line_t>(
        &line,
        "shaders/opengl/2D/objects/line.vs",
        "shaders/opengl/2D/objects/line.fs"
      );
    }
  }

  shape_functions.resize(shape_functions.size() + 1); // mark
  {
    if (opengl.major == 2 && opengl.minor == 1) {
      // todo
      shape_functions.resize(shape_functions.size() + 1);
    }
    else {
      shape_open<loco_t::rectangle_t>(
        &rectangle,
        "shaders/opengl/2D/objects/rectangle.vs",
        "shaders/opengl/2D/objects/rectangle.fs"
      );
    }
  }

  {
    if (opengl.major == 2 && opengl.minor == 1) {
      // todo
      shape_functions.resize(shape_functions.size() + 1);
    }
    else {
      shape_open<loco_t::light_t>(
        &light,
        "shaders/opengl/2D/objects/light.vs",
        "shaders/opengl/2D/objects/light.fs"
      );
    }
  }
  {
    if (opengl.major == 2 && opengl.minor == 1) {
      // todo
      shape_functions.resize(shape_functions.size() + 1);
    }
    else {
      shape_open<loco_t::unlit_sprite_t>(
        &unlit_sprite,
        "shaders/opengl/2D/objects/sprite.vs",
        "shaders/opengl/2D/objects/unlit_sprite.fs"
      );
    }
  }
 
  {
    if (opengl.major == 2 && opengl.minor == 1) {
      shape_functions.resize(shape_functions.size() + 1);
    }
    else {
      shape_open<loco_t::circle_t>(
        &circle,
        "shaders/opengl/2D/objects/circle.vs",
        "shaders/opengl/2D/objects/circle.fs"
      );
    }
  }
  {
    if (opengl.major == 2 && opengl.minor == 1) {
      shape_functions.resize(shape_functions.size() + 1);
    }
    else {
      shape_open<loco_t::capsule_t>(
        &capsule,
        "shaders/opengl/2D/objects/capsule.vs",
        "shaders/opengl/2D/objects/capsule.fs"
      );
    }
  }
  {
    if (opengl.major == 2 && opengl.minor == 1) {
      shape_functions.resize(shape_functions.size() + 1);
    }
    else {
      shape_open<loco_t::polygon_t>(
        &polygon,
        "shaders/opengl/2D/objects/polygon.vs",
        "shaders/opengl/2D/objects/polygon.fs",
        1,
        false
      );
      shaper.GetShapeTypes(loco_t::shape_type_t::polygon).vertex_count = loco_t::polygon_t::max_vertices_per_element;
      shaper.GetShapeTypes(loco_t::shape_type_t::polygon).draw_mode = GL_TRIANGLES;
    }
  }

  {
    if (opengl.major == 2 && opengl.minor == 1) {
      shape_functions.resize(shape_functions.size() + 1);
    }
    else {
      shape_open<loco_t::grid_t>(
        &grid,
        "shaders/opengl/2D/objects/grid.vs",
        "shaders/opengl/2D/objects/grid.fs"
      );
    }
  }

  vfi.open();

  {
    if (opengl.major == 2 && opengl.minor == 1) {
      shape_functions.resize(shape_functions.size() + 1);
    }
    else {
      shape_open<loco_t::particles_t>(
        &particles,
        "shaders/opengl/2D/effects/particles.vs",
        "shaders/opengl/2D/effects/particles.fs"
      );
    }
  }

  {
    if (opengl.major == 2 && opengl.minor == 1) {
      shape_functions.resize(shape_functions.size() + 1);
    }
    else {
      shape_open<loco_t::universal_image_renderer_t>(
        &universal_image_renderer,
        "shaders/opengl/2D/objects/pixel_format_renderer.vs",
        "shaders/opengl/2D/objects/yuv420p.fs"
      );
    }
  }

  {
    if (opengl.major == 2 && opengl.minor == 1) {
      shape_functions.resize(shape_functions.size() + 1);
    }
    else {
      shape_open<loco_t::gradient_t>(
        &gradient,
        "shaders/opengl/2D/effects/gradient.vs",
        "shaders/opengl/2D/effects/gradient.fs"
      );
    }
  }

  shape_functions.resize(shape_functions.size() + 1); // light_end

  {
    if (opengl.major == 2 && opengl.minor == 1) {
      shape_functions.resize(shape_functions.size() + 1);
    }
    else {
      shape_open<loco_t::shader_shape_t>(
        &shader_shape,
        "shaders/opengl/2D/objects/sprite.vs",
        "shaders/opengl/2D/objects/sprite.fs"
      );
    }
  }

  {
    shape_open<loco_t::rectangle3d_t>(
      &rectangle3d,
      "shaders/opengl/3D/objects/rectangle.vs",
      "shaders/opengl/3D/objects/rectangle.fs",
      (opengl.major == 2 && opengl.minor == 1) ? 36 : 1
    );
  }
  {
    if (opengl.major == 2 && opengl.minor == 1) {
      // todo implement line
      shape_functions.resize(shape_functions.size() + 1);
    }
    else {
      shape_open<loco_t::line3d_t>(
        &line3d,
        "shaders/opengl/3D/objects/line.vs",
        "shaders/opengl/3D/objects/line.fs"
      );
    }
  }

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

    //wglMakeCurrent(g_MainWindow.hDC, g_hRC);

#if defined(loco_physics)
    fan::graphics::open_bcol();
#endif


#if defined(loco_imgui)
    init_imgui(this);
    generate_commands(this);
#endif

    init_framebuffer();

    bool windowed = true;
    // free this xd
    gloco->window.add_keys_callback(
      [windowed](const fan::window_t::keyboard_keys_cb_data_t& data) mutable {
        if (data.key == fan::key_enter && data.state == fan::keyboard_state::press && gloco->window.key_pressed(fan::key_left_alt)) {
          windowed = !windowed;
          gloco->window.set_size_mode(windowed ? fan::window_t::mode::windowed : fan::window_t::mode::borderless);
        }
      }
    );

    loco_t::shader_t shader = shader_create();

    shader_set_vertex(shader,
      read_shader("shaders/empty.vs")
    );
      
    shader_set_fragment(shader,
      read_shader("shaders/empty.fs")
    );

    shader_compile(shader);

    gloco->shaper.AddShapeType(
      loco_t::shape_type_t::light_end,
      {
        .MaxElementPerBlock = (shaper_t::MaxElementPerBlock_t)MaxElementPerBlock,
        .RenderDataSize = 0,
        .DataSize = 0,
        .locations = {},
        .shader = shader
      }
    );
    shape_add(
      loco_t::shape_type_t::light_end,
      0,
      0,
      Key_e::light_end, (uint8_t)0,
      Key_e::ShapeType, (loco_t::shaper_t::ShapeTypeIndex_t)loco_t::shape_type_t::light_end
    );
  }
}

loco_t::~loco_t() {
  fan::graphics::close_bcol();
  shaper.Close();
#if defined(loco_imgui)
  destroy_imgui();
#endif
  window.close();
}

void loco_t::draw_shapes() {
  shaper_t::KeyTraverse_t KeyTraverse;
  KeyTraverse.Init(shaper);

  uint32_t texture_count = 0;
  viewport_t viewport;
  viewport.sic();
  camera_t camera;
  camera.sic();

  bool light_buffer_enabled = false;

  { // update 3d view every frame
    auto& camera_perspective = camera_get(perspective_camera.camera);
    camera_perspective.update_view();

    camera_perspective.m_view = camera_perspective.get_view_matrix();
  }

  while (KeyTraverse.Loop(shaper)) {
    
    shaper_t::KeyTypeIndex_t kti = KeyTraverse.kti(shaper);


    switch (kti) {
    case Key_e::blending: {
      uint8_t Key = *(uint8_t*)KeyTraverse.kd();
      if (Key) {
        set_depth_test(false);
        fan_opengl_call(glEnable(GL_BLEND));
        fan_opengl_call(glBlendFunc(blend_src_factor, blend_dst_factor));
        // shaper.SetKeyOrder(Key_e::depth, shaper_t::KeyBitOrderLow);
      }
      else {
        fan_opengl_call(glDisable(GL_BLEND));
        set_depth_test(true);

        //shaper.SetKeyOrder(Key_e::depth, shaper_t::KeyBitOrderHigh);
      }
      break;
    }
    case Key_e::depth: {
#if defined(depth_debug)
      depth_t Key = *(depth_t*)KeyTraverse.kd();
      depth_Key = true;
      fan::print(Key);
#endif
      break;
    }
    case Key_e::image: {
      loco_t::image_t texture = *(loco_t::image_t*)KeyTraverse.kd();
      if (texture.iic() == false) {
        // TODO FIX + 0
        fan_opengl_call(glActiveTexture(GL_TEXTURE0 + 0));
        fan_opengl_call(glBindTexture(GL_TEXTURE_2D, image_get(texture)));
        //++texture_count;
      }
      break;
    }
    case Key_e::viewport: {
      viewport = *(loco_t::viewport_t*)KeyTraverse.kd();
      break;
    }
    case Key_e::camera: {
      camera = *(loco_t::camera_t*)KeyTraverse.kd();
      break;
    }
    case Key_e::ShapeType: {
      // if i remove this why it breaks/corrupts?
      if (*(loco_t::shaper_t::ShapeTypeIndex_t*)KeyTraverse.kd() == loco_t::shape_type_t::light_end) {
        continue;
      }
      break;
    }
    case Key_e::light: {
      if (!((opengl.major > 3) || (opengl.major == 3 && opengl.minor >= 3))) {
        break;
      }
      if (light_buffer_enabled == false) {
#if defined(loco_framebuffer)
        set_depth_test(false);
        fan_opengl_call(glEnable(GL_BLEND));
        fan_opengl_call(glBlendFunc(GL_ONE, GL_ONE));
        unsigned int attachments[sizeof(color_buffers) / sizeof(color_buffers[0])];

        for (uint8_t i = 0; i < std::size(color_buffers); ++i) {
          attachments[i] = GL_COLOR_ATTACHMENT0 + i;
        }

        fan_opengl_call(glDrawBuffers(std::size(attachments), attachments));
        light_buffer_enabled = true;
#endif
      }
      break;
    }
    case Key_e::light_end: {
      if (!((opengl.major > 3) || (opengl.major == 3 && opengl.minor >= 3))) {
        break;
      }
      if (light_buffer_enabled) {
#if defined(loco_framebuffer)
        gloco->get_context().set_depth_test(true);
        unsigned int attachments[sizeof(color_buffers) / sizeof(color_buffers[0])];

        for (uint8_t i = 0; i < std::size(color_buffers); ++i) {
          attachments[i] = GL_COLOR_ATTACHMENT0 + i;
        }

        fan_opengl_call(glDrawBuffers(1, attachments));
        light_buffer_enabled = false;
#endif
        continue;
      }
      break;
    }
    }

    if (KeyTraverse.isbm) {
      
      shaper_t::BlockTraverse_t BlockTraverse;
      shaper_t::ShapeTypeIndex_t shape_type = BlockTraverse.Init(shaper, KeyTraverse.bmid());

      if (shape_type == shape_type_t::light_end) {
        break;
      }
      do {
        auto shader = shaper.GetShader(shape_type);
#if fan_debug >= fan_debug_medium
        if (shape_type == loco_t::shape_type_t::vfi || shape_type == loco_t::shape_type_t::light_end) {
          break;
        }
        else if ((shape_type == 0 || shader.iic())) {
          fan::throw_error("invalid stuff");
        }
#endif
        shader_use(shader);

        if (camera.iic() == false) {
          shader_set_camera(shader, camera);
        }
        else {
          shader_set_camera(shader, orthographic_camera.camera);
        }
        if (viewport.iic() == false) {
          auto& v = viewport_get(viewport);
          viewport_set(v.viewport_position, v.viewport_size, window.get_size());
        }
        shader_set_value(shader, "_t00", 0);
        if ((opengl.major > 3) || (opengl.major == 3 && opengl.minor >= 3)) {
          shader_set_value(shader, "_t01", 1);
        }
#if defined(depth_debug)
        if (depth_Key) {
          auto& ri = *(fan::vec3*)BlockTraverse.GetRenderData(shaper);
          fan::print("d", ri.z);
        }
#endif
#if fan_debug >= fan_debug_high
        switch (shape_type) {
        default: {
          if (camera.iic()) {
            fan::throw_error("failed to get camera");
          }
          if (viewport.iic()) {
            fan::throw_error("failed to get viewport");
          }
          break;
        }
        }
#endif

        if (shape_type == loco_t::shape_type_t::universal_image_renderer) {
          auto shader = shaper.GetShader(shape_type);
          
          auto& ri = *(universal_image_renderer_t::ri_t*)BlockTraverse.GetData(shaper);

          if (ri.images_rest[0].iic() == false) {
            fan_opengl_call(glActiveTexture(GL_TEXTURE0 + 1));
            fan_opengl_call(glBindTexture(GL_TEXTURE_2D, image_get(ri.images_rest[0])));
            shader_set_value(shader, "_t01", 1);
          }
          if (ri.images_rest[1].iic() == false) {
            fan_opengl_call(glActiveTexture(GL_TEXTURE0 + 2));
            fan_opengl_call(glBindTexture(GL_TEXTURE_2D, image_get(ri.images_rest[1])));
            shader_set_value(shader, "_t02", 2);
          }

          if (ri.images_rest[2].iic() == false) {
            fan_opengl_call(glActiveTexture(GL_TEXTURE0 + 3));
            fan_opengl_call(glBindTexture(GL_TEXTURE_2D, image_get(ri.images_rest[2])));
            shader_set_value(shader, "_t03", 3);
          }
          //fan::throw_error("shaper design is changed");
        }
        else if (shape_type == loco_t::shape_type_t::sprite ||
          shape_type == loco_t::shape_type_t::unlit_sprite || 
          shape_type == loco_t::shape_type_t::shader_shape) {
          //fan::print("shaper design is changed");
          auto& ri = *(sprite_t::ri_t*)BlockTraverse.GetData(shaper);
          auto shader = shaper.GetShader(shape_type);
          for (std::size_t i = 2; i < std::size(ri.images) + 2; ++i) {
            if (ri.images[i - 2].iic() == false) {
              shader_set_value(shader, "_t0" + std::to_string(i), i);
              fan_opengl_call(glActiveTexture(GL_TEXTURE0 + i));
              fan_opengl_call(glBindTexture(GL_TEXTURE_2D, image_get(ri.images[i - 2])));
            }
          }
        }

        if (shape_type != loco_t::shape_type_t::light) {

          if (shape_type == loco_t::shape_type_t::sprite || shape_type == loco_t::shape_type_t::unlit_sprite) {
            if ((opengl.major > 3) || (opengl.major == 3 && opengl.minor >= 3)) {
              fan_opengl_call(glActiveTexture(GL_TEXTURE1));
              fan_opengl_call(glBindTexture(GL_TEXTURE_2D, image_get(color_buffers[1])));
            }
          }

          auto& c = camera_get(camera);

          shader_set_value(
            shader,
            "matrix_size",
            fan::vec2(c.coordinates.right - c.coordinates.left, c.coordinates.down - c.coordinates.up).abs()
          );
          shader_set_value(
            shader,
            "viewport",
            fan::vec4(
              viewport_get_position(viewport),
              viewport_get_size(viewport)
            )
          );
          shader_set_value(
            shader,
            "window_size",
            fan::vec2(window.get_size())
          );
          shader_set_value(
            shader,
            "camera_position",
            c.position
          );
          shader_set_value(
            shader,
            "m_time",
            f32_t((fan::time::clock::now() - start_time) / 1e+9)
          );
          //fan::print(fan::time::clock::now() / 1e+9);
          shader_set_value(shader, loco_t::lighting_t::ambient_name, gloco->lighting.ambient);
        }

        auto m_vao = shaper.GetVAO(shape_type);
        auto m_vbo = shaper.GetVAO(shape_type);

        m_vao.bind(*this);
        m_vbo.bind(*this);

        if (opengl.major < 4 || (opengl.major == 4 && opengl.minor < 2)) {
          uintptr_t offset = BlockTraverse.GetRenderDataOffset(shaper);
          std::vector<shape_gl_init_t>& locations = shaper.GetLocations(shape_type);
          for (const auto& location : locations) {
            fan_opengl_call(glVertexAttribPointer(location.index.first, location.size, location.type, GL_FALSE, location.stride, (void*)offset));
            switch (location.type) {
            case GL_FLOAT: {
              offset += location.size * sizeof(GLfloat);
              break;
            }
            case GL_UNSIGNED_INT: {
              offset += location.size * sizeof(GLuint);
              break;
            }
            default: {
              fan::throw_error_impl();
            }
            }
          }
        }

        switch (shape_type) {
        case shape_type_t::rectangle3d: {
          // illegal xd
          set_depth_test(false);
          if ((opengl.major > 4) || (opengl.major == 4 && opengl.minor >= 2)) {
            fan_opengl_call(glDrawArraysInstancedBaseInstance(
              GL_TRIANGLES,
              0,
              36,
              BlockTraverse.GetAmount(shaper),
              BlockTraverse.GetRenderDataOffset(shaper) / shaper.GetRenderDataSize(shape_type)
            ));
          }
          else if ((opengl.major > 3) || (opengl.major == 3 && opengl.minor >= 3)) {
            // this is broken somehow with rectangle3d
            fan_opengl_call(glDrawArraysInstanced(
              GL_TRIANGLES,
              0,
              36,
              BlockTraverse.GetAmount(shaper)
            ));
          }
          else {
            fan_opengl_call(glDrawArrays(
              GL_TRIANGLES,
              0,
              36 * BlockTraverse.GetAmount(shaper)
            ));
          }
          break;
        }
        case shape_type_t::line3d: {
          // illegal xd
          set_depth_test(false);
        }//fallthrough
        case shape_type_t::line: {
          if ((opengl.major > 4) || (opengl.major == 4 && opengl.minor >= 2)) {
            fan_opengl_call(glDrawArraysInstancedBaseInstance(
              GL_LINES,
              0,
              2,
              BlockTraverse.GetAmount(shaper),
              BlockTraverse.GetRenderDataOffset(shaper) / shaper.GetRenderDataSize(shape_type)
            ));
          }
          else {
            fan_opengl_call(glDrawArraysInstanced(
              GL_LINES,
              0,
              2,
              BlockTraverse.GetAmount(shaper)
            ));
          }


          break;
        }
        case shape_type_t::particles: {
          //fan::print("shaper design is changed");
          particles_t::ri_t* pri = (particles_t::ri_t*)BlockTraverse.GetData(shaper);
          loco_t::shader_t shader = shaper.GetShader(shape_type_t::particles);

          for (int i = 0; i < BlockTraverse.GetAmount(shaper); ++i) {
            auto& ri = pri[i];
            shader_set_value(shader, "time", (f32_t)((fan::time::clock::now() - ri.begin_time) / 1e+9));
            shader_set_value(shader, "vertex_count", 6);
            shader_set_value(shader, "count", ri.count);
            shader_set_value(shader, "alive_time", (f32_t)(ri.alive_time / 1e+9));
            shader_set_value(shader, "respawn_time", (f32_t)(ri.respawn_time / 1e+9));
            shader_set_value(shader, "position", *(fan::vec2*)&ri.position);
            shader_set_value(shader, "size", ri.size);
            shader_set_value(shader, "position_velocity", ri.position_velocity);
            shader_set_value(shader, "angle_velocity", ri.angle_velocity);
            shader_set_value(shader, "begin_angle", ri.begin_angle);
            shader_set_value(shader, "end_angle", ri.end_angle);
            shader_set_value(shader, "angle", ri.angle);
            shader_set_value(shader, "color", ri.color);
            shader_set_value(shader, "gap_size", ri.gap_size);
            shader_set_value(shader, "max_spread_size", ri.max_spread_size);
            shader_set_value(shader, "size_velocity", ri.size_velocity);

            shader_set_value(shader, "shape", ri.shape);

            // TODO how to get begin?
            fan_opengl_call(glDrawArrays(
              GL_TRIANGLES,
              0,
              ri.count
            ));
          }

          break;
        }
        default: {
          auto& shape_data = shaper.GetShapeTypes(shape_type);
          if (((opengl.major > 4) || (opengl.major == 4 && opengl.minor >= 2)) && shape_data.instanced) {
           fan_opengl_call(glDrawArraysInstancedBaseInstance(
              shape_data.draw_mode,
              0,
              6,//polygon_t::max_vertices_per_element breaks light
              BlockTraverse.GetAmount(shaper),
              BlockTraverse.GetRenderDataOffset(shaper) / shaper.GetRenderDataSize(shape_type)
            ));
          }
          else if (((opengl.major > 3) || (opengl.major == 3 && opengl.minor >= 3))&& shape_data.instanced) {
            fan_opengl_call(glDrawArraysInstanced(
              shape_data.draw_mode,
              0,
              shape_data.vertex_count,
              BlockTraverse.GetAmount(shaper)
            ));
          }
          else {
            fan_opengl_call(glDrawArrays(
              shape_data.draw_mode,
              (!!!(opengl.major == 2 && opengl.minor == 1)) * (BlockTraverse.GetRenderDataOffset(shaper) / shaper.GetRenderDataSize(shape_type)) * shape_data.vertex_count,
              shape_data.vertex_count * BlockTraverse.GetAmount(shaper)
            ));
          }

          break;
        }
        }
        } while (BlockTraverse.Loop(shaper));
    }
  }
}

void loco_t::process_frame() {

  fan_opengl_call(glViewport(0, 0, window.get_size().x, window.get_size().y));

#if defined(loco_framebuffer)
  if ((opengl.major > 3) || (opengl.major == 3 && opengl.minor >= 3)) {
    m_framebuffer.bind(*this);

    fan_opengl_call(glClearColor(clear_color.r, clear_color.g, clear_color.b, clear_color.a));
    for (std::size_t i = 0; i < std::size(color_buffers); ++i) {
      fan_opengl_call(glActiveTexture(GL_TEXTURE0 + i));
      image_bind(color_buffers[i]);
      fan_opengl_call(glDrawBuffer(GL_COLOR_ATTACHMENT0 + (uint32_t)std::size(color_buffers) - 1 - i));
      if (i + (std::size_t)1 == std::size(color_buffers)) {
        fan_opengl_call(glClearColor(clear_color.r, clear_color.g, clear_color.b, clear_color.a));
      }
      fan_opengl_call(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
    }
  }
  else {
    fan_opengl_call(glClearColor(clear_color.r, clear_color.g, clear_color.b, clear_color.a));
    fan_opengl_call(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
  }
#else
  fan_opengl_call(glClearColor(clear_color.r, clear_color.g, clear_color.b, clear_color.a));
  fan_opengl_call(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
#endif

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

  shaper.ProcessBlockEditQueue();

  viewport_set(0, window.get_size(), window.get_size());

  for (const auto& i : m_pre_draw) {
    i();
  }

  draw_shapes();

#if defined(loco_framebuffer)

  if ((opengl.major > 3) || (opengl.major == 3 && opengl.minor >= 3)) {
    m_framebuffer.unbind(*this);

#if defined(loco_post_process)
  blur[0].draw(&color_buffers[0]);
#endif

  //blur[1].draw(&color_buffers[3]);

  fan_opengl_call(glClearColor(clear_color.r, clear_color.g, clear_color.b, clear_color.a));
  fan_opengl_call(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
  fan::vec2 window_size = window.get_size();
  viewport_set(0, window_size, window_size);

  shader_set_value(m_fbo_final_shader, "_t00", 0);
  shader_set_value(m_fbo_final_shader, "_t01", 1);

  shader_set_value(m_fbo_final_shader, "window_size", window_size);

  fan_opengl_call(glActiveTexture(GL_TEXTURE0));
  image_bind(color_buffers[0]);

#if defined(loco_post_process)
  fan_opengl_call(glActiveTexture(GL_TEXTURE1));
  image_bind(blur[0].mips.front().image);
#endif
  render_final_fb();
#endif
  }

  for (const auto& i : m_post_draw) {
    i();
  }

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

  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

#endif

  glfwSwapBuffers(window);
}

bool loco_t::should_close() {
  return glfwWindowShouldClose(window);
}

bool loco_t::process_loop(const fan::function_t<void()>& lambda) {
#if defined(loco_imgui)
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  auto& style = ImGui::GetStyle();
  ImVec4* colors = style.Colors;
  const ImVec4 bgColor = ImVec4(0.0f, 0.0f, 0.0f, 0.4f);
  colors[ImGuiCol_WindowBg] = bgColor;
  colors[ImGuiCol_ChildBg] = bgColor;
  colors[ImGuiCol_TitleBg] = bgColor;

  ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0));
  ImGui::PushStyleColor(ImGuiCol_DockingEmptyBg, ImVec4(0, 0, 0, 0));
  ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport());
  ImGui::PopStyleColor(2);

  ImGui::SetNextWindowPos(ImVec2(0, 0));
  ImGui::SetNextWindowSize(window.get_size());
  ImGui::Begin("##text_render", 0, ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoResize | ImGuiDockNodeFlags_NoDockingSplit | ImGuiWindowFlags_NoTitleBar);
#endif

  lambda();

#if defined(loco_imgui)
    ImGui::End();
#endif

  process_frame();
  window.handle_events();
  
  if (should_close()) {
    window.close();
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
  double delay = std::round(1.0 / target_fps * 1000.0);

  uv_timer_init(uv_default_loop(), &timer_handle);
  uv_idle_init(uv_default_loop(), &idle_handle);

  timer_handle.data = this;
  idle_handle.data = this;

  if (target_fps > 0) {
    start_timer();
  }
  else {
    start_idle();
  }

  uv_run(uv_default_loop(), UV_RUN_DEFAULT);

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
  get_context().set_vsync(&window, flag);
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
    fan::opengl::context_t::shader_t& shader = gloco->shader_get(shader_nr);
    auto found = shader.uniform_type_table.find(var_name);
    if (found == shader.uniform_type_table.end()) {
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
      gloco->get_context().shader_set_value(shader_nr, var_name, data);
    }
  };
  gloco->get_context().shader_set_value(shader_nr, var_name, initial);
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
  fan::vec2 viewport_pos = fan::vec2(windowPosRelativeToMainViewport + fan::vec2(0, ImGui::GetFontSize() + ImGui::GetStyle().FramePadding.y * 2));
  viewport_set(
    viewport,
    viewport_pos,
    viewport_size,
    window_size
  );
}
#endif

fan::opengl::context_t& loco_t::get_context() {
  return *dynamic_cast<fan::opengl::context_t*>(this);
}



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
  std::memset(found->second.keys, sizeof(found->second.keys), 0);
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

static fan::vec2 transform_position(const fan::vec2& p, loco_t::viewport_t viewport, loco_t::camera_t camera) {

  auto& context = gloco->get_context();
  auto& v = context.viewport_get(viewport);
  auto& c = context.camera_get(camera);

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
  return transform_position(gloco->get_mouse_position(), camera.viewport, camera.camera);
}

fan::vec2 loco_t::translate_position(const fan::vec2& p, loco_t::viewport_t viewport, loco_t::camera_t camera) {

  auto& v = gloco->viewport_get(viewport);
  fan::vec2 viewport_position = v.viewport_position;
  fan::vec2 viewport_size = v.viewport_size;

  auto& c = gloco->camera_get(camera);

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

void loco_t::shape_t::set_position(const fan::vec3& position) {
  gloco->shape_functions[gloco->shaper.GetSTI(*this)].set_position3(this, position);
}

fan::vec3 loco_t::shape_t::get_position() {
  return gloco->shape_functions[gloco->shaper.GetSTI(*this)].get_position(this);
}

void loco_t::shape_t::set_size(const fan::vec2& size) {
  gloco->shape_functions[gloco->shaper.GetSTI(*this)].set_size(this, size);
}

void loco_t::shape_t::set_size3(const fan::vec3& size) {
  gloco->shape_functions[gloco->shaper.GetSTI(*this)].set_size3(this, size);
}

fan::vec2 loco_t::shape_t::get_size() {
  return gloco->shape_functions[gloco->shaper.GetSTI(*this)].get_size(this);
}

fan::vec3 loco_t::shape_t::get_size3() {
  return gloco->shape_functions[gloco->shaper.GetSTI(*this)].get_size3(this);
}

void loco_t::shape_t::set_rotation_point(const fan::vec2& rotation_point) {
  gloco->shape_functions[gloco->shaper.GetSTI(*this)].set_rotation_point(this, rotation_point);
}

fan::vec2 loco_t::shape_t::get_rotation_point() {
  return gloco->shape_functions[gloco->shaper.GetSTI(*this)].get_rotation_point(this);
}

void loco_t::shape_t::set_color(const fan::color& color) {
  gloco->shape_functions[gloco->shaper.GetSTI(*this)].set_color(this, color);
}

fan::color loco_t::shape_t::get_color() {
  return gloco->shape_functions[gloco->shaper.GetSTI(*this)].get_color(this);
}

void loco_t::shape_t::set_angle(const fan::vec3& angle) {
  gloco->shape_functions[gloco->shaper.GetSTI(*this)].set_angle(this, angle);
}

fan::vec3 loco_t::shape_t::get_angle() {
  return gloco->shape_functions[gloco->shaper.GetSTI(*this)].get_angle(this);
}

fan::vec2 loco_t::shape_t::get_tc_position() {
  return gloco->shape_functions[gloco->shaper.GetSTI(*this)].get_tc_position(this);
}

void loco_t::shape_t::set_tc_position(const fan::vec2& tc_position) {
  gloco->shape_functions[gloco->shaper.GetSTI(*this)].set_tc_position(this, tc_position);
}

fan::vec2 loco_t::shape_t::get_tc_size() {
  return gloco->shape_functions[gloco->shaper.GetSTI(*this)].get_tc_size(this);
}

void loco_t::shape_t::set_tc_size(const fan::vec2& tc_size) {
  gloco->shape_functions[gloco->shaper.GetSTI(*this)].set_tc_size(this, tc_size);
}

bool loco_t::shape_t::load_tp(loco_t::texturepack_t::ti_t* ti) {
  return gloco->shape_functions[gloco->shaper.GetSTI(*this)].load_tp(this, ti);
}

loco_t::texturepack_t::ti_t loco_t::shape_t::get_tp() {
  loco_t::texturepack_t::ti_t ti;
  ti.image = &gloco->default_texture;
  auto& img = gloco->image_get_data(*ti.image);
  ti.position = get_tc_position() * img.size;
  ti.size = get_tc_size() * img.size;
  return ti;
  //return gloco->shape_functions[gloco->shaper.GetSTI(*this)].get_tp(this);
}

bool loco_t::shape_t::set_tp(loco_t::texturepack_t::ti_t* ti) {
  return load_tp(ti);
}

loco_t::camera_t loco_t::shape_t::get_camera() {
  return gloco->shape_functions[gloco->shaper.GetSTI(*this)].get_camera(this);
}

void loco_t::shape_t::set_camera(loco_t::camera_t camera) {
  gloco->shape_functions[gloco->shaper.GetSTI(*this)].set_camera(this, camera);
}

loco_t::viewport_t loco_t::shape_t::get_viewport() {
  return gloco->shape_functions[gloco->shaper.GetSTI(*this)].get_viewport(this);
}

void loco_t::shape_t::set_viewport(loco_t::viewport_t viewport) {
  gloco->shape_functions[gloco->shaper.GetSTI(*this)].set_viewport(this, viewport);
}

fan::vec2 loco_t::shape_t::get_grid_size() {
  return gloco->shape_functions[gloco->shaper.GetSTI(*this)].get_grid_size(this);
}

void loco_t::shape_t::set_grid_size(const fan::vec2& grid_size) {
  gloco->shape_functions[gloco->shaper.GetSTI(*this)].set_grid_size(this, grid_size);
}

loco_t::image_t loco_t::shape_t::get_image() {
  return gloco->shape_functions[gloco->shaper.GetSTI(*this)].get_image(this);
}

void loco_t::shape_t::set_image(loco_t::image_t image) {
  gloco->shape_functions[gloco->shaper.GetSTI(*this)].set_image(this, image);
}

f32_t loco_t::shape_t::get_parallax_factor() {
  return gloco->shape_functions[gloco->shaper.GetSTI(*this)].get_parallax_factor(this);
}

void loco_t::shape_t::set_parallax_factor(f32_t parallax_factor) {
  gloco->shape_functions[gloco->shaper.GetSTI(*this)].set_parallax_factor(this, parallax_factor);
}

fan::vec3 loco_t::shape_t::get_rotation_vector() {
  return gloco->shape_functions[gloco->shaper.GetSTI(*this)].get_rotation_vector(this);
}

uint32_t loco_t::shape_t::get_flags() {
  return gloco->shape_functions[gloco->shaper.GetSTI(*this)].get_flags(this);
}

void loco_t::shape_t::set_flags(uint32_t flag) {
  return gloco->shape_functions[gloco->shaper.GetSTI(*this)].set_flags(this, flag);
}

f32_t loco_t::shape_t::get_radius() {
  return gloco->shape_functions[gloco->shaper.GetSTI(*this)].get_radius(this);
}

fan::vec3 loco_t::shape_t::get_src() {
  return gloco->shape_functions[gloco->shaper.GetSTI(*this)].get_src(this);
}

fan::vec3 loco_t::shape_t::get_dst() {
  return gloco->shape_functions[gloco->shaper.GetSTI(*this)].get_dst(this);
}

f32_t loco_t::shape_t::get_outline_size() {
  return gloco->shape_functions[gloco->shaper.GetSTI(*this)].get_outline_size(this);
}

fan::color loco_t::shape_t::get_outline_color() {
  return gloco->shape_functions[gloco->shaper.GetSTI(*this)].get_outline_color(this);
}

void loco_t::shape_t::reload(uint8_t format, void** image_data, const fan::vec2& image_size, uint32_t filter) {
  gloco->shape_functions[gloco->shaper.GetSTI(*this)].reload(this, format, image_data, image_size, filter);
}

void loco_t::shape_t::reload(uint8_t format, const fan::vec2& image_size, uint32_t filter) {
  void* data[4]{};
  gloco->shape_functions[gloco->shaper.GetSTI(*this)].reload(this, format, data, image_size, filter);
}

void loco_t::shape_t::set_line(const fan::vec2& src, const fan::vec2& dst) {
  gloco->shape_functions[gloco->shaper.GetSTI(*this)].set_line(this, src, dst);
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
    Key_e::ShapeType, shape_type
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
    Key_e::ShapeType, shape_type
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
    Key_e::ShapeType, shape_type
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
    Key_e::ShapeType, shape_type
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
    Key_e::ShapeType, shape_type
  );
}

loco_t::shape_t loco_t::polygon_t::push_back(const loco_t::polygon_t::properties_t& properties) {
  if (properties.vertices.empty()) {
    fan::throw_error("invalid vertices");
  }
  if (properties.vertices.size() > loco_t::polygon_t::max_vertices_per_element) {
    fan::throw_error("maximum polygons reached");
  }
  vi_t vis;
  for (std::size_t i = 0; i < properties.vertices.size(); ++i) {
    vis.vertices[i].position = properties.vertices[i].position;
    vis.vertices[i].color = properties.vertices[i].color;
  }
  // fill gaps
  for (std::size_t i = properties.vertices.size(); i < std::size(vis.vertices); ++i) {
    vis.vertices[i].position = vis.vertices[i - 1].position;
    vis.vertices[i].color = vis.vertices[i - 1].color;
  }
  ri_t ri;
  return shape_add(shape_type, vis, ri,
    Key_e::depth, (uint16_t)properties.vertices[0].position.z,
    Key_e::blending, (uint8_t)properties.blending,
    Key_e::viewport, properties.viewport,
    Key_e::camera, properties.camera,
    Key_e::ShapeType, shape_type
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

  if ((gloco->opengl.major > 3) || (gloco->opengl.major == 3 && gloco->opengl.minor >= 3)) {
    return shape_add(
      shape_type, vi, ri, Key_e::depth,
      static_cast<uint16_t>(properties.position.z),
      Key_e::blending, static_cast<uint8_t>(properties.blending),
      Key_e::image, properties.image, Key_e::viewport,
      properties.viewport, Key_e::camera, properties.camera,
      Key_e::ShapeType, shape_type
    );
  } else {
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
      Key_e::ShapeType, shape_type
    );
  }
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
    Key_e::ShapeType, shape_type
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
    Key_e::ShapeType, shape_type
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
    Key_e::ShapeType, shape_type
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
  std::memcpy(ri.images_rest, &properties.images[1], sizeof(ri.images_rest));
  return shape_add(shape_type, vi, ri,
    Key_e::depth, (uint16_t)properties.position.z,
    Key_e::blending, (uint8_t)properties.blending,
    Key_e::image, properties.images[0],
    Key_e::viewport, properties.viewport,
    Key_e::camera, properties.camera,
    Key_e::ShapeType, shape_type
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
  std::memcpy(vi.color, properties.color, sizeof(vi.color));
  vi.angle = properties.angle;
  vi.rotation_point = properties.rotation_point;
  ri_t ri;

  return shape_add(shape_type, vi, ri,
    Key_e::depth, (uint16_t)properties.position.z,
    Key_e::blending, (uint8_t)properties.blending,
    Key_e::viewport, properties.viewport,
    Key_e::camera, properties.camera,
    Key_e::ShapeType, shape_type
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
    Key_e::ShapeType, shape_type
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

  if ((gloco->opengl.major > 3) || (gloco->opengl.major == 3 && gloco->opengl.minor >= 3)) {
    // might not need depth
    return shape_add(shape_type, vi, ri,
      Key_e::depth, (uint16_t)properties.position.z,
      Key_e::blending, (uint8_t)properties.blending,
      Key_e::viewport, properties.viewport,
      Key_e::camera, properties.camera,
      Key_e::ShapeType, shape_type
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
      Key_e::ShapeType, shape_type
    );
  }
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
    Key_e::ShapeType, shape_type
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

fan::line3 fan::graphics::get_highlight_positions(const fan::vec3& op_, const fan::vec2& os, int index) {
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

#if defined(loco_imgui)

void fan::graphics::text(const std::string& text, const fan::vec2& position, const fan::color& color) {
  ImGui::SetCursorPos(position);
  ImGui::PushStyleColor(ImGuiCol_Text, color);
  ImGui::Text("%s", text.c_str());
  ImGui::PopStyleColor();
}
void fan::graphics::text_bottom_right(const std::string& text, const fan::color& color, const fan::vec2& offset) {
  ImVec2 text_pos;
  ImVec2 text_size = ImGui::CalcTextSize(text.c_str());
  ImVec2 window_pos = ImGui::GetWindowPos();
  ImVec2 window_size = ImGui::GetWindowSize();

  text_pos.x = window_pos.x + window_size.x - text_size.x - ImGui::GetStyle().WindowPadding.x;
  text_pos.y = window_pos.y + window_size.y - text_size.y - ImGui::GetStyle().WindowPadding.y;
  fan::graphics::text(text, text_pos + offset, color);
}
IMGUI_API void ImGui::Image(loco_t::image_t img, const ImVec2& size, const ImVec2& uv0, const ImVec2& uv1, const ImVec4& tint_col, const ImVec4& border_col) {
  ImGui::Image((ImTextureID)gloco->image_get(img), size, uv0, uv1, tint_col, border_col);
}
IMGUI_API bool ImGui::ImageButton(const std::string& str_id, loco_t::image_t img, const ImVec2& size, const ImVec2& uv0, const ImVec2& uv1, int frame_padding, const ImVec4& bg_col, const ImVec4& tint_col) {
  return ImGui::ImageButton(str_id.c_str(), (ImTextureID)gloco->image_get(img), size, uv0, uv1, bg_col, tint_col);
}
IMGUI_API bool ImGui::ImageTextButton(loco_t::image_t img, const std::string& text, const fan::color& color, const ImVec2& size, const ImVec2& uv0, const ImVec2& uv1, int frame_padding, const ImVec4& bg_col, const ImVec4& tint_col) {

  bool ret = ImGui::ImageButton(text.c_str(), (ImTextureID)gloco->image_get(img), size, uv0, uv1, bg_col, tint_col);
  ImVec2 text_size = ImGui::CalcTextSize(text.c_str());
  ImVec2 pos = ImGui::GetItemRectMin(); 
  pos.x += (size.x - text_size.x) * 0.5f;
  pos.y += (size.y - text_size.y) * 0.5f;
  ImGui::GetWindowDrawList()->AddText(pos, ImGui::GetColorU32(color), text.c_str());
  return ret;
}
bool ImGui::ToggleButton(const std::string& str, bool* v) {

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
bool ImGui::ToggleImageButton(const std::string& char_id, loco_t::image_t image, const ImVec2& size, bool* toggle)
{
  bool clicked = false;

  ImVec4 tintColor = ImVec4(1, 1, 1, 1);
  if (*toggle) {
  //  tintColor = ImVec4(0.3f, 0.3f, 0.3f, 1.0f);
  }
  if (ImGui::IsItemHovered()) {
  //  tintColor = ImVec4(0.6f, 0.6f, 0.6f, 1.0f);
  }

  if (ImGui::ImageButton(char_id, image, size, ImVec2(0, 0), ImVec2(1, 1), -1, ImVec4(0, 0, 0, 0), tintColor)) {
    *toggle = !(*toggle);
    clicked = true;
  }

  return clicked;
}
ImVec2 ImGui::GetPositionBottomCorner(const char* text, uint32_t reverse_yoffset) {
  ImVec2 window_pos = ImGui::GetWindowPos();
  ImVec2 window_size = ImGui::GetWindowSize();

  ImVec2 text_size = ImGui::CalcTextSize(text);

  ImVec2 text_pos;
  text_pos.x = window_pos.x + window_size.x - text_size.x - ImGui::GetStyle().WindowPadding.x;
  text_pos.y = window_pos.y + window_size.y - text_size.y - ImGui::GetStyle().WindowPadding.y;

  text_pos.y -= reverse_yoffset * ImGui::GetTextLineHeightWithSpacing();

  return text_pos;
}
void ImGui::ImageRotated(ImTextureID user_texture_id, const ImVec2& size, int angle, const ImVec2& uv0, const ImVec2& uv1, const ImVec4& tint_col, const ImVec4& border_col)
{
  IM_ASSERT(angle % 90 == 0);
  ImVec2 _uv0, _uv1, _uv2, _uv3;
  switch (angle % 360)
  {
  case 0:
    Image(user_texture_id, size, uv0, uv1, tint_col, border_col);
    return;
  case 180:
    Image(user_texture_id, size, uv1, uv0, tint_col, border_col);
    return;
  case 90:
    _uv3 = uv0;
    _uv1 = uv1;
    _uv0 = ImVec2(uv1.x, uv0.y);
    _uv2 = ImVec2(uv0.x, uv1.y);
    break;
  case 270:
    _uv1 = uv0;
    _uv3 = uv1;
    _uv0 = ImVec2(uv0.x, uv1.y);
    _uv2 = ImVec2(uv1.x, uv0.y);
    break;
  }
  ImGuiWindow* window = GetCurrentWindow();
  if (window->SkipItems)
    return;
  ImVec2 _size(size.y, size.x);
  ImRect bb(window->DC.CursorPos, window->DC.CursorPos + _size);
  if (border_col.w > 0.0f)
    bb.Max += ImVec2(2, 2);
  ItemSize(bb);
  if (!ItemAdd(bb, 0))
    return;
  if (border_col.w > 0.0f)
  {
    window->DrawList->AddRect(bb.Min, bb.Max, GetColorU32(border_col), 0.0f);
    ImVec2 x0 = bb.Min + ImVec2(1, 1);
    ImVec2 x2 = bb.Max - ImVec2(1, 1);
    ImVec2 x1 = ImVec2(x2.x, x0.y);
    ImVec2 x3 = ImVec2(x0.x, x2.y);
    window->DrawList->AddImageQuad(user_texture_id, x0, x1, x2, x3, _uv0, _uv1, _uv2, _uv3, GetColorU32(tint_col));
  }
  else
  {
    ImVec2 x1 = ImVec2(bb.Max.x, bb.Min.y);
    ImVec2 x3 = ImVec2(bb.Min.x, bb.Max.y);
    window->DrawList->AddImageQuad(user_texture_id, bb.Min, x1, bb.Max, x3, _uv0, _uv1, _uv2, _uv3, GetColorU32(tint_col));
  }
}
void ImGui::DrawTextBottomRight(const char* text, uint32_t reverse_yoffset) {
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
void fan::graphics::imgui_content_browser_t::render() {
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

      if (ImGui::ImageButton("##icon_arrow_left", icon_arrow_left, fan::vec2(32))) {
        if (std::filesystem::equivalent(current_directory, asset_path) == false) {
          current_directory = current_directory.parent_path();
        }
        update_directory_cache();
      }
      ImGui::SameLine();
      ImGui::ImageButton("##icon_arrow_right", icon_arrow_right, fan::vec2(32));
      ImGui::SameLine();
      ImGui::PopStyleColor(3);

      ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.f, 0.f, 0.f, 0.f));
      ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.f, 0.f, 0.f, 0.f));
      ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.3f, 0.3f, 0.3f));

      auto image_list = std::to_array({ icon_files_list, icon_files_big_thumbnail });

      fan::vec2 bc = ImGui::GetPositionBottomCorner();

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

      ImGui::ToggleImageButton(image_list, button_sizes, (int*)&current_view_mode);

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

fan::graphics::imgui_content_browser_t::imgui_content_browser_t() {
  search_buffer.resize(32);
  asset_path = std::filesystem::absolute(std::filesystem::path(asset_path)).wstring();
  current_directory = std::filesystem::path(asset_path);
  update_directory_cache();
}

void fan::graphics::imgui_content_browser_t::update_directory_cache() {
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

void fan::graphics::imgui_content_browser_t::render_large_thumbnails_view() {
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
    ImGui::ImageButton("##" + file_info.filename, file_info.preview_image.iic() == false ? file_info.preview_image : file_info.is_directory ? icon_directory : icon_file, ImVec2(thumbnail_size, thumbnail_size));

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

void fan::graphics::imgui_content_browser_t::render_list_view() {
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
      fan::vec2 cursor_pos = ImGui::GetWindowPos() + ImGui::GetCursorPos() + fan::vec2(ImGui::GetScrollX(), -ImGui::GetScrollY());
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
        ImGui::GetWindowDrawList()->AddImage((ImTextureID)gloco->image_get(file_info.preview_image), cursor_pos, cursor_pos + image_size);
      }
      else if (file_info.is_directory) {
        ImGui::GetWindowDrawList()->AddImage((ImTextureID)gloco->image_get(icon_directory), cursor_pos, cursor_pos + image_size);
      }
      else {
        ImGui::GetWindowDrawList()->AddImage((ImTextureID)gloco->image_get(icon_file), cursor_pos, cursor_pos + image_size);
      }

      handle_item_interaction(file_info);
    }

    ImGui::EndTable();
  }
}

void fan::graphics::imgui_content_browser_t::handle_item_interaction(const file_info_t& file_info) {
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
    auto& ri = *(loco_t::particles_t::ri_t*)gloco->shaper.GetData(shape);
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
bool fan::graphics::shape_to_bin(loco_t::shape_t& shape, std::string* str) {
  std::string& out = *str;
  switch (shape.get_shape_type()) {
  case loco_t::shape_type_t::light: {
    // shape
    fan::write_to_string(out, std::string("light"));
    fan::write_to_string(out, shape.get_position());
    fan::write_to_string(out, shape.get_parallax_factor());
    fan::write_to_string(out, shape.get_size());
    fan::write_to_string(out, shape.get_rotation_point());
    fan::write_to_string(out, shape.get_color());
    fan::write_to_string(out, shape.get_rotation_vector());
    fan::write_to_string(out, shape.get_flags());
    fan::write_to_string(out, shape.get_angle());
    break;
  }
  case loco_t::shape_type_t::line: {
    fan::write_to_string(out, std::string("line"));
    fan::write_to_string(out, shape.get_color());
    fan::write_to_string(out, shape.get_src());
    fan::write_to_string(out, shape.get_dst());
    break;
    case loco_t::shape_type_t::rectangle: {
    fan::write_to_string(out, std::string("rectangle"));
    fan::write_to_string(out, shape.get_position());
    fan::write_to_string(out, shape.get_size());
    fan::write_to_string(out, shape.get_rotation_point());
    fan::write_to_string(out, shape.get_color());
    fan::write_to_string(out, shape.get_angle());
    break;
    }
    case loco_t::shape_type_t::sprite: {
    fan::write_to_string(out, std::string("sprite"));
    fan::write_to_string(out, shape.get_position());
    fan::write_to_string(out, shape.get_parallax_factor());
    fan::write_to_string(out, shape.get_size());
    fan::write_to_string(out, shape.get_rotation_point());
    fan::write_to_string(out, shape.get_color());
    fan::write_to_string(out, shape.get_angle());
    fan::write_to_string(out, shape.get_flags());
    fan::write_to_string(out, shape.get_tc_position());
    fan::write_to_string(out, shape.get_tc_size());
    break;
    }
    case loco_t::shape_type_t::unlit_sprite: {
    fan::write_to_string(out, std::string("unlit_sprite"));
    fan::write_to_string(out, shape.get_position());
    fan::write_to_string(out, shape.get_parallax_factor());
    fan::write_to_string(out, shape.get_size());
    fan::write_to_string(out, shape.get_rotation_point());
    fan::write_to_string(out, shape.get_color());
    fan::write_to_string(out, shape.get_angle());
    fan::write_to_string(out, shape.get_flags());
    fan::write_to_string(out, shape.get_tc_position());
    fan::write_to_string(out, shape.get_tc_size());
    break;
    }
    case loco_t::shape_type_t::circle: {
    fan::write_to_string(out, std::string("circle"));
    fan::write_to_string(out, shape.get_position());
    fan::write_to_string(out, shape.get_radius());
    fan::write_to_string(out, shape.get_rotation_point());
    fan::write_to_string(out, shape.get_color());
    fan::write_to_string(out, shape.get_rotation_vector());
    fan::write_to_string(out, shape.get_angle());
    break;
    }
    case loco_t::shape_type_t::grid: {
    fan::write_to_string(out, std::string("grid"));
    fan::write_to_string(out, shape.get_position());
    fan::write_to_string(out, shape.get_size());
    fan::write_to_string(out, shape.get_grid_size());
    fan::write_to_string(out, shape.get_rotation_point());
    fan::write_to_string(out, shape.get_color());
    fan::write_to_string(out, shape.get_angle());
    break;
    }
    case loco_t::shape_type_t::particles: {
    auto& ri = *(loco_t::particles_t::ri_t*)gloco->shaper.GetData(shape);
    fan::write_to_string(out, std::string("particles"));
    fan::write_to_string(out, ri.position);
    fan::write_to_string(out, ri.size);
    fan::write_to_string(out, ri.color);
    fan::write_to_string(out, ri.begin_time);
    fan::write_to_string(out, ri.alive_time);
    fan::write_to_string(out, ri.respawn_time);
    fan::write_to_string(out, ri.count);
    fan::write_to_string(out, ri.position_velocity);
    fan::write_to_string(out, ri.angle_velocity);
    fan::write_to_string(out, ri.begin_angle);
    fan::write_to_string(out, ri.end_angle);
    fan::write_to_string(out, ri.angle);
    fan::write_to_string(out, ri.gap_size);
    fan::write_to_string(out, ri.max_spread_size);
    fan::write_to_string(out, ri.size_velocity);
    fan::write_to_string(out, ri.shape);
    fan::write_to_string(out, ri.blending);
    break;
    }
  }
  default: {
    fan::throw_error("unimplemented shape");
  }
  }
  return false;
}
bool fan::graphics::bin_to_shape(const std::string& in, loco_t::shape_t* shape, uint64_t& offset) {
  std::string shape_type = fan::read_data<std::string>(in, offset);
  switch (fan::get_hash(shape_type.c_str())) {
  case fan::get_hash("rectangle"): {
    loco_t::rectangle_t::properties_t p;
    p.position = fan::read_data<decltype(p.position)>(in, offset);
    p.size = fan::read_data<decltype(p.size)>(in, offset);
    p.rotation_point = fan::read_data<decltype(p.rotation_point)>(in, offset);
    p.color = fan::read_data<decltype(p.color)>(in, offset);
    p.angle = fan::read_data<decltype(p.angle)>(in, offset);
    *shape = p;
    return false;
  }
  case fan::get_hash("light"): {
    loco_t::light_t::properties_t p;
    p.position = fan::read_data<decltype(p.position)>(in, offset);
    p.parallax_factor = fan::read_data<decltype(p.parallax_factor)>(in, offset);
    p.size = fan::read_data<decltype(p.size)>(in, offset);
    p.rotation_point = fan::read_data<decltype(p.rotation_point)>(in, offset);
    p.color = fan::read_data<decltype(p.color)>(in, offset);
    p.rotation_vector = fan::read_data<decltype(p.rotation_vector)>(in, offset);
    p.flags = fan::read_data<decltype(p.flags)>(in, offset);
    p.angle = fan::read_data<decltype(p.angle)>(in, offset);
    *shape = p;
    break;
  }
  case fan::get_hash("line"): {
    loco_t::line_t::properties_t p;
    p.color = fan::read_data<decltype(p.color)>(in, offset);
    p.src = fan::read_data<decltype(p.src)>(in, offset);
    p.dst = fan::read_data<decltype(p.dst)>(in, offset);
    *shape = p;
    break;
  }
  case fan::get_hash("sprite"): {
    loco_t::sprite_t::properties_t p;
    p.position = fan::read_data<decltype(p.position)>(in, offset);
    p.parallax_factor = fan::read_data<decltype(p.parallax_factor)>(in, offset);
    p.size = fan::read_data<decltype(p.size)>(in, offset);
    p.rotation_point = fan::read_data<decltype(p.rotation_point)>(in, offset);
    p.color = fan::read_data<decltype(p.color)>(in, offset);
    p.angle = fan::read_data<decltype(p.angle)>(in, offset);
    p.flags = fan::read_data<decltype(p.flags)>(in, offset);
    p.tc_position = fan::read_data<decltype(p.tc_position)>(in, offset);
    p.tc_size = fan::read_data<decltype(p.tc_size)>(in, offset);
    *shape = p;
    break;
  }
  case fan::get_hash("unlit_sprite"): {
    loco_t::unlit_sprite_t::properties_t p;
    p.position = fan::read_data<decltype(p.position)>(in, offset);
    p.parallax_factor = fan::read_data<decltype(p.parallax_factor)>(in, offset);
    p.size = fan::read_data<decltype(p.size)>(in, offset);
    p.rotation_point = fan::read_data<decltype(p.rotation_point)>(in, offset);
    p.color = fan::read_data<decltype(p.color)>(in, offset);
    p.angle = fan::read_data<decltype(p.angle)>(in, offset);
    p.flags = fan::read_data<decltype(p.flags)>(in, offset);
    p.tc_position = fan::read_data<decltype(p.tc_position)>(in, offset);
    p.tc_size = fan::read_data<decltype(p.tc_size)>(in, offset);
    *shape = p;
    break;
  }
  case loco_t::shape_type_t::circle: {
    loco_t::circle_t::properties_t p;
    p.position = fan::read_data<decltype(p.position)>(in, offset);
    p.radius = fan::read_data<decltype(p.radius)>(in, offset);
    p.rotation_point = fan::read_data<decltype(p.rotation_point)>(in, offset);
    p.color = fan::read_data<decltype(p.color)>(in, offset);
    p.rotation_vector = fan::read_data<decltype(p.rotation_vector)>(in, offset);
    p.angle = fan::read_data<decltype(p.angle)>(in, offset);
    *shape = p;
    break;
  }
  case loco_t::shape_type_t::grid: {
    loco_t::grid_t::properties_t p;
    p.position = fan::read_data<decltype(p.position)>(in, offset);
    p.size = fan::read_data<decltype(p.size)>(in, offset);
    p.grid_size = fan::read_data<decltype(p.grid_size)>(in, offset);
    p.rotation_point = fan::read_data<decltype(p.rotation_point)>(in, offset);
    p.color = fan::read_data<decltype(p.color)>(in, offset);
    p.angle = fan::read_data<decltype(p.angle)>(in, offset);
    *shape = p;
    break;
  }
  case loco_t::shape_type_t::particles: {
    loco_t::particles_t::properties_t p;
    p.position = fan::read_data<decltype(p.position)>(in, offset);
    p.size = fan::read_data<decltype(p.size)>(in, offset);
    p.color = fan::read_data<decltype(p.color)>(in, offset);
    p.begin_time = fan::read_data<decltype(p.begin_time)>(in, offset);
    p.alive_time = fan::read_data<decltype(p.alive_time)>(in, offset);
    p.respawn_time = fan::read_data<decltype(p.respawn_time)>(in, offset);
    p.count = fan::read_data<decltype(p.count)>(in, offset);
    p.position_velocity = fan::read_data<decltype(p.position_velocity)>(in, offset);
    p.angle_velocity = fan::read_data<decltype(p.angle_velocity)>(in, offset);
    p.begin_angle = fan::read_data<decltype(p.begin_angle)>(in, offset);
    p.end_angle = fan::read_data<decltype(p.end_angle)>(in, offset);
    p.angle = fan::read_data<decltype(p.angle)>(in, offset);
    p.gap_size = fan::read_data<decltype(p.gap_size)>(in, offset);
    p.max_spread_size = fan::read_data<decltype(p.max_spread_size)>(in, offset);
    p.size_velocity = fan::read_data<decltype(p.size_velocity)>(in, offset);
    p.shape = fan::read_data<decltype(p.shape)>(in, offset);
    p.blending = fan::read_data<decltype(p.blending)>(in, offset);
    *shape = p;
    break;
  }
  default: {
    fan::throw_error("unimplemented");
  }
  }
  return false;
}
bool fan::graphics::shape_serialize(loco_t::shape_t& shape, std::string* out) {
  return shape_to_bin(shape, out);
}
#endif

bool fan::graphics::texture_packe0::push_texture(fan::opengl::context_t::image_nr_t image, const texture_properties_t& texture_properties) {

  if (texture_properties.image_name.empty()) {
    fan::print_warning("texture properties name empty");
    return 1;
  }

  for (uint32_t gti = 0; gti < texture_list.size(); gti++) {
    if (texture_list[gti].image_name == texture_properties.image_name) {
      texture_list.erase(texture_list.begin() + gti);
      break;
    }
  }

  auto& context = gloco->get_context();
  auto& img = context.image_get_data(image);

  auto data = context.image_get_pixel_data(image, GL_RGBA, texture_properties.uv_pos, texture_properties.uv_size);
  fan::vec2ui image_size(
    (uint32_t)(img.size.x * texture_properties.uv_size.x),
    (uint32_t)(img.size.y * texture_properties.uv_size.y)
  );


  if ((int)image_size.x % 2 != 0 || (int)image_size.y % 2 != 0) {
    fan::print_warning("failed to load, image size is not divideable by 2");
    fan::print(texture_properties.image_name, image_size);
    return 1;
  }

  texture_t t;
  t.size = image_size;
  t.decoded_data.resize(t.size.multiply() * 4);
  std::memcpy(t.decoded_data.data(), data.get(), t.size.multiply() * 4);
  t.image_name = texture_properties.image_name;
  t.visual_output = texture_properties.visual_output;
  t.min_filter = texture_properties.min_filter;
  t.mag_filter = texture_properties.mag_filter;
  t.group_id = texture_properties.group_id;

  texture_list.push_back(t);
  return 0;
}

#if defined(loco_json)
void fan::graphics::texture_packe0::load_compiled(const char* filename) {
  std::ifstream file(filename);
  fan::json j;
  file >> j;

  loaded_pack.resize(j["pack_amount"]);

  std::vector<loco_t::image_t> images;

  for (std::size_t i = 0; i < j["pack_amount"]; i++) {
    loaded_pack[i].texture_list.resize(j["packs"][i]["count"]);

    for (std::size_t k = 0; k < j["packs"][i]["count"]; k++) {
      pack_t::texture_t* t = &loaded_pack[i].texture_list[k];
      std::string image_name = j["packs"][i]["textures"][k]["image_name"];
      t->position = j["packs"][i]["textures"][k]["position"];
      t->size = j["packs"][i]["textures"][k]["size"];
      t->image_name = image_name;
    }

    std::vector<uint8_t> pixel_data = j["packs"][i]["pixel_data"].get<std::vector<uint8_t>>();
    fan::image::image_info_t image_info;
    image_info.data = WebPDecodeRGBA(
      pixel_data.data(),
      pixel_data.size(),
      &image_info.size.x,
      &image_info.size.y
    );
    loaded_pack[i].pixel_data = std::vector<uint8_t>((uint8_t*)image_info.data, (uint8_t*)image_info.data + image_info.size.x * image_info.size.y * 4);


    loaded_pack[i].visual_output = j["packs"][i]["visual_output"];
    loaded_pack[i].min_filter = j["packs"][i]["min_filter"];
    loaded_pack[i].mag_filter = j["packs"][i]["mag_filter"];
    images.push_back(gloco->image_load(image_info));
    WebPFree(image_info.data);
    for (std::size_t k = 0; k < loaded_pack[i].texture_list.size(); ++k) {
      auto& tl = loaded_pack[i].texture_list[k];
      fan::graphics::texture_packe0::texture_properties_t tp;
      tp.group_id = 0;
      tp.uv_pos = fan::vec2(tl.position) / fan::vec2(image_info.size);
      tp.uv_size = fan::vec2(tl.size) / fan::vec2(image_info.size);
      tp.visual_output = loaded_pack[i].visual_output;
      tp.min_filter = loaded_pack[i].min_filter;
      tp.mag_filter = loaded_pack[i].mag_filter;
      tp.image_name = tl.image_name;
      push_texture(images.back(), tp);
    }
  }
}//

#endif

void fan::camera::move(f32_t movement_speed, f32_t friction) {
  this->velocity /= friction * gloco->delta_time + 1;
  static constexpr auto minimum_velocity = 0.001;
  if (this->velocity.x < minimum_velocity && this->velocity.x > -minimum_velocity) {
    this->velocity.x = 0;
  }
  if (this->velocity.y < minimum_velocity && this->velocity.y > -minimum_velocity) {
    this->velocity.y = 0;
  }
  if (this->velocity.z < minimum_velocity && this->velocity.z > -minimum_velocity) {
    this->velocity.z = 0;
  }
  #if defined(loco_imgui)
  if (!gloco->console.input.IsFocused()) {
#endif
    if (gloco->window.key_pressed(fan::input::key_w)) {
      this->velocity += this->m_front * (movement_speed * gloco->delta_time);
    }
    if (gloco->window.key_pressed(fan::input::key_s)) {
      this->velocity -= this->m_front * (movement_speed * gloco->delta_time);
    }
    if (gloco->window.key_pressed(fan::input::key_a)) {
      this->velocity -= this->m_right * (movement_speed * gloco->delta_time);
    }
    if (gloco->window.key_pressed(fan::input::key_d)) {
      this->velocity += this->m_right * (movement_speed * gloco->delta_time);
    }

    if (gloco->window.key_pressed(fan::input::key_space)) {
      this->velocity.y += movement_speed * gloco->delta_time;
    }
    if (gloco->window.key_pressed(fan::input::key_left_shift)) {
      this->velocity.y -= movement_speed * gloco->delta_time;
    }

    if (gloco->window.key_pressed(fan::input::key_left)) {
      this->set_yaw(this->get_yaw() - sensitivity * 100 * gloco->delta_time);
    }
    if (gloco->window.key_pressed(fan::input::key_right)) {
      this->set_yaw(this->get_yaw() + sensitivity * 100 * gloco->delta_time);
    }
    if (gloco->window.key_pressed(fan::input::key_up)) {
      this->set_pitch(this->get_pitch() + sensitivity * 100 * gloco->delta_time);
    }
    if (gloco->window.key_pressed(fan::input::key_down)) {
      this->set_pitch(this->get_pitch() - sensitivity * 100 * gloco->delta_time);
    }
  #if defined(loco_imgui)
  }
#endif

  this->position += this->velocity * gloco->delta_time;
  this->update_view();
}

loco_t::shader_t loco_t::create_sprite_shader(const fan::string& fragment) {
  loco_t::shader_t shader = shader_create();
  shader_set_vertex(
    shader,
    loco_t::read_shader("shaders/opengl/2D/objects/sprite.vs")
  );
  shader_set_fragment(shader, fragment);
  shader_compile(shader);
  return shader;
}

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

loco_t::image_t loco_t::create_noise_image(const fan::vec2& image_size) {

  loco_t::image_load_properties_t lp;
  lp.format = GL_RGBA; // Change this to GL_RGB
  lp.internal_format = GL_RGBA; // Change this to GL_RGB
  lp.min_filter = loco_t::image_filter::linear;
  lp.mag_filter = loco_t::image_filter::linear;
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
  lp.min_filter = loco_t::image_filter::linear;
  lp.mag_filter = loco_t::image_filter::linear;
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

fan::graphics::interactive_camera_t::interactive_camera_t(loco_t::camera_t camera_nr, loco_t::viewport_t viewport_nr) :
  reference_camera(camera_nr), reference_viewport(viewport_nr)
{
  auto& window = gloco->window;
  static auto update_ortho = [&] (loco_t* loco){
    fan::vec2 s = loco->viewport_get_size(reference_viewport);
    loco->camera_set_ortho(
      reference_camera,
      fan::vec2(-s.x, s.x) / zoom,
      fan::vec2(-s.y, s.y) / zoom
    );
  };

  auto it = gloco->m_update_callback.NewNodeLast();
  gloco->m_update_callback[it] = update_ortho;

  button_cb_nr = window.add_buttons_callback([&](const auto& d) {
    if (d.button == fan::mouse_scroll_up) {
      zoom *= 1.2;
    }
    else if (d.button == fan::mouse_scroll_down) {
      zoom /= 1.2;
    }
  });
}

fan::graphics::interactive_camera_t::~interactive_camera_t() {
  if (button_cb_nr.iic() == false) {
    gloco->window.remove_buttons_callback(button_cb_nr);
    button_cb_nr.sic();
  }
  if (uc_nr.iic() == false) {
    gloco->m_update_callback.unlrec(uc_nr);
    uc_nr.sic();
  }
}

#if defined(loco_imgui)

// called in loop
void fan::graphics::interactive_camera_t::move_by_cursor() {
  if (ImGui::IsMouseDragging(ImGuiMouseButton_Middle)) {
    fan::vec2 drag_delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Middle);
    gloco->camera_set_position(
      gloco->orthographic_camera.camera,
      gloco->camera_get_position(gloco->orthographic_camera.camera) - (drag_delta / zoom) * 2
    );
    ImGui::ResetMouseDragDelta(ImGuiMouseButton_Middle);
  }
}

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

#if defined(loco_imgui)
fan::graphics::dialogue_box_t::dialogue_box_t() {
  gloco->input_action.add(fan::mouse_left, "skip or continue dialog");
}

void fan::graphics::dialogue_box_t::set_cursor_position(const fan::vec2& cursor_position) {
  this->cursor_position = cursor_position;
}

fan::ev::task_t fan::graphics::dialogue_box_t::text(const std::string& text) {
  active_dialogue = text;
  render_pos = 0;
  finish_dialog = false;
  while (render_pos < active_dialogue.size() && !finish_dialog) {
    ++render_pos;
    co_await fan::co_sleep(1000 / character_per_s);
  }
  render_pos = active_dialogue.size();
}

fan::ev::task_t fan::graphics::dialogue_box_t::button(const std::string& text, const fan::vec2& position, const fan::vec2& size) {
  button_choice = -1;
  button_t button;
  button.position = position;
  button.size = size;
  button.text = text;
  buttons.push_back(button);
  co_return;
}

int fan::graphics::dialogue_box_t::get_button_choice() const {
  return button_choice;
}

fan::ev::task_t fan::graphics::dialogue_box_t::wait_user_input() {
  wait_user = true;
  fan::time::clock c;
  c.start(0.5e9);
  int prev_render = render_pos;
  while (wait_user) {
    if (c.finished()) {
      if (prev_render == render_pos) {
        render_pos = std::max(prev_render - 1, 0);
      }
      else {
        render_pos = prev_render;
      }
      c.restart();
    }
    co_await fan::co_sleep(10);
  }
  render_pos = prev_render;
}

void fan::graphics::dialogue_box_t::render(const std::string& window_name, ImFont* font, const fan::vec2& window_size, f32_t wrap_width, f32_t line_spacing) {
  ImGui::PushFont(font);
  fan::vec2 root_window_size = ImGui::GetWindowSize();
  fan::vec2 next_window_pos;
  next_window_pos.x = (root_window_size.x - window_size.x) / 2.0f;
  next_window_pos.y = (root_window_size.y - window_size.y) / 1.1;
  ImGui::SetNextWindowPos(next_window_pos);

  ImGui::SetNextWindowSize(window_size);
  ImGui::Begin(window_name.c_str(), 0, 
    ImGuiWindowFlags_NoNavInputs | ImGuiWindowFlags_NoTitleBar | 
    ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar
  );
  ImGui::SetCursorPos(ImVec2(100.0f, 100.f));
  ImGui::BeginChild((window_name + "child").c_str(), fan::vec2(wrap_width, 0), 0, ImGuiWindowFlags_NoNavInputs | ImGuiWindowFlags_NoTitleBar | 
    ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoBackground);
  if (wait_user == false) {
    ImGui::SetScrollY(ImGui::GetScrollMaxY());
  }
  fan::graphics::text_partial_render(active_dialogue.c_str(), render_pos, wrap_width, line_spacing);
  ImGui::EndChild();
  if (wait_user) {
    fan::vec2 first_pos = -1;


    // calculate biggest button
    fan::vec2 button_size = 0;
    for (std::size_t i = 0; i < buttons.size(); ++i) {
      fan::vec2 text_size = ImGui::CalcTextSize(buttons[i].text.c_str());
      float padding_x = ImGui::GetStyle().FramePadding.x; 
      float padding_y = ImGui::GetStyle().FramePadding.y; 
      ImVec2 bs = ImVec2(text_size.x + padding_x * 2.0f, text_size.y + padding_y * 2.0f);
      button_size = button_size.max(fan::vec2(bs));
    }

    for (std::size_t i = 0; i < buttons.size(); ++i) {
      const auto& button = buttons[i];
      if (button.position != -1) {
        first_pos = button.position;
        ImGui::SetCursorPos((button.position * window_size) - button_size / 2);
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + ImGui::GetScrollY());
      }
      else {
        ImGui::SetCursorPosX(first_pos.x * window_size.x - button_size.x / 2);
      }
      ImGui::PushID(i);

      if (ImGui::ImageTextButton(gloco->default_texture, button.text.c_str(), fan::colors::white, button.size == 0 ? button_size : button.size)) {
        button_choice = i;
        buttons.clear();
        wait_user = false;
        ImGui::PopID();
        break;
      }
      ImGui::PopID();
    }
  }
  if (gloco->input_action.is_active("skip or continue dialog") && ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows | ImGuiHoveredFlags_AllowWhenBlockedByPopup | ImGuiHoveredFlags_AllowWhenBlockedByActiveItem  )) {
    finish_dialog = true;
    wait_user = false;
  }
  ImGui::End();
  ImGui::PopFont();
}

#endif

#endif

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
