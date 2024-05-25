#pragma once

#include <fan/graphics/loco_settings.h>

#define loco_opengl
#define loco_framebuffer
#define loco_post_process
#define loco_vfi
#define loco_physics

#include <fan/window/window.h>
#include <fan/graphics/opengl/gl_core.h>

#if defined(loco_imgui)
#include <fan/imgui/imgui.h>
#include <fan/imgui/imgui_impl_opengl3.h>
#include <fan/imgui/imgui_impl_glfw.h>
#include <fan/imgui/imgui_neo_sequencer.h>
#endif

#include <fan/physics/collision/rectangle.h>

#if defined(loco_imgui)
#include <fan/graphics/console.h>
#endif

#define loco_line
#define loco_rectangle
#define loco_sprite
#define loco_light
#define loco_circle
#define loco_letter
#define loco_responsive_text
#define loco_universal_image_renderer


//#define debug_shape_t

struct loco_t;

// to set new loco use gloco = new_loco;
struct global_loco_t {

  loco_t* loco = nullptr;

  operator loco_t* ();
  global_loco_t& operator=(loco_t* l);
  loco_t* operator->() {
    return loco;
  }
};
inline thread_local global_loco_t gloco;

#if defined(loco_letter)
#include <fan/graphics/font.h>
#endif


static constexpr uint32_t MaxElementPerBlock = 0x100;

#define shaper_set_MaxMaxElementPerBlock 0x100
#include <fan/graphics/shaper.h>

struct loco_t : fan::opengl::context_t {

  loco_t(const loco_t&) = delete;
  loco_t& operator=(const loco_t&) = delete;
  loco_t(loco_t&&) = delete;
  loco_t& operator=(loco_t&&) = delete;

  struct shape_type_t {
    enum {
      invalid = -1,
      // render order
      // make sure shape.open() has same order - TODO remove shape.open - use shape_functions[i].open
      button,
      sprite = 1,
      text,
      hitbox,
      line,
      rectangle,
      light,
      unlit_sprite,
      letter,
      circle,
      grid,
      vfi,
      particles,
      universal_image_renderer,
      last
    };
  };

  struct kp {
    enum {
      light,
      common,
      vfi,
      texture,
      texture4,
    };
  };

  static constexpr const char* shape_names[] = {
    "button",
    "sprite",
    "text",
    "hitbox",
    "line",
    "mark",
    "rectangle",
    "light",
    "unlit_sprite",
    "circle",
    "grid",
    "vfi",
    "particles",
  };

  using blending_t = uint8_t;
  using depth_t = uint16_t;

  void use();

  using camera_t = fan::opengl::context_t::camera_nr_t;
  void camera_move(fan::opengl::context_t::camera_t& camera, f64_t dt, f32_t movement_speed, f32_t friction);

  uint32_t fb_vao;
  uint32_t fb_vbo;
  void render_final_fb();
  void initialize_fb_vaos(uint32_t& vao, uint32_t& vbo);

  using viewport_t = fan::opengl::context_t::viewport_nr_t;
  using image_t = fan::opengl::context_t::image_nr_t;

  using shader_t = fan::opengl::context_t::shader_nr_t;

  struct shape_t;

  #include <fan/graphics/opengl/texture_pack.h>

  using push_back_cb = shape_t (*)(void*);
  using set_position2_cb = void (*)(shape_t*, const fan::vec2&);
  // depth
  using set_position3_cb = void (*)(shape_t*, const fan::vec3&);
  using set_size_cb = void (*)(shape_t*, const fan::vec2&);

  using get_position_cb = fan::vec3 (*)(shape_t*);
  using get_size_cb = fan::vec2 (*)(shape_t*);

  using set_rotation_point_cb = void (*)(shape_t*, const fan::vec2&);
  using get_rotation_point_cb = fan::vec2 (*)(shape_t*);

  using set_color_cb = void (*)(shape_t*, const fan::color&);
  using get_color_cb = fan::color (*)(shape_t*);

  using set_angle_cb = void (*)(shape_t*, const fan::vec3&);
  using get_angle_cb = fan::vec3 (*)(shape_t*);

  using get_tc_position_cb = fan::vec2 (*)(shape_t*);
  using set_tc_position_cb = void (*)(shape_t*, const fan::vec2&);

  using get_tc_size_cb = fan::vec2 (*)(shape_t*);
  using set_tc_size_cb = void (*)(shape_t*, const fan::vec2&);

  using load_tp_cb = bool(*)(shape_t*, loco_t::texturepack_t::ti_t*);

  using get_grid_size_cb = fan::vec2 (*)(shape_t*);
  using set_grid_size_cb = void (*)(shape_t*, const fan::vec2&);

  using get_camera_cb = loco_t::camera_t (*)(shape_t*);
  using set_camera_cb = void (*)(shape_t*, loco_t::camera_t);

  using get_viewport_cb = loco_t::viewport_t (*)(shape_t*);
  using set_viewport_cb = void (*)(shape_t*, loco_t::viewport_t);

  using set_image_cb = void (*)(shape_t*, loco_t::image_t);

  using get_parallax_factor_cb = f32_t (*)(shape_t*);
  using set_parallax_factor_cb = void (*)(shape_t*, f32_t);
  using get_rotation_vector_cb = fan::vec3 (*)(shape_t*);
  using get_flags_cb = uint32_t (*)(shape_t*);
  //
  using get_radius_cb = f32_t (*)(shape_t*);
  using get_src_cb = fan::vec3 (*)(shape_t*);
  using get_dst_cb = fan::vec3 (*)(shape_t*);
  using get_outline_size_cb = f32_t (*)(shape_t*);
  using get_outline_color_cb = fan::color (*)(shape_t*);

  using reload_cb = void (*)(shape_t*, uint8_t format, void** image_data, const fan::vec2& image_size, uint32_t filter); 

  using draw_cb = void (*)(uint8_t draw_range);

  struct functions_t {
    push_back_cb push_back;

    get_position_cb get_position;
    set_position2_cb set_position2;
    set_position3_cb set_position3;

    get_size_cb get_size;
    set_size_cb set_size;

    get_rotation_point_cb get_rotation_point;
    set_rotation_point_cb set_rotation_point;

    get_color_cb get_color;
    set_color_cb set_color;

    get_angle_cb get_angle;
    set_angle_cb set_angle;

    get_tc_position_cb get_tc_position;
    set_tc_position_cb set_tc_position;

    get_tc_size_cb get_tc_size;
    set_tc_size_cb set_tc_size;

    load_tp_cb load_tp;

    get_grid_size_cb get_grid_size;
    set_grid_size_cb set_grid_size;

    get_camera_cb get_camera;
    set_camera_cb set_camera;

    get_viewport_cb get_viewport;
    set_viewport_cb set_viewport;

    set_image_cb set_image;

    get_parallax_factor_cb get_parallax_factor;
    set_parallax_factor_cb set_parallax_factor;
    get_rotation_vector_cb get_rotation_vector;
    get_flags_cb get_flags;

    get_radius_cb get_radius;
    get_src_cb get_src;
    get_dst_cb get_dst;
    get_outline_size_cb get_outline_size;
    get_outline_color_cb get_outline_color;

    reload_cb reload;

    draw_cb draw;
  };

  template <typename T, typename T2>
  static T2& get_render_data(shape_t* shape, T2 T::* attribute) {
    shaper_t::ShapeRenderData_t* data = gloco->shaper.GetRenderData(*shape);
    return ((T*)data)->*attribute;
  }

  template <typename T, typename T2, typename T3>
  static void modify_render_data_element(shape_t* shape, T2 T::* attribute, const T3& value) {
    shaper_t::ShapeRenderData_t* data = gloco->shaper.GetRenderData(*shape);
    ((T*)data)->*attribute = value;
    shaper_t::shape_t shaper_shape = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      shaper_shape.bmid,
      shaper_shape.blid,
      shaper_shape.ElementIndex,
      fan::member_offset(attribute),
      sizeof(T3),
      gloco->shaper.ShapeList[*shape].sti
    );
  };

  template <typename T>
  static functions_t get_functions() {
    functions_t funcs{
      //.push_back = [](void* data) {
      //  // Implement push_back function
      //},
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
          auto sp = gloco->shaper.ShapeList[*shape];

            auto kpi = gloco->shaper.ShapeTypes[sp.sti].KeyPackIndex;
            auto& kps = gloco->shaper.KeyPacks[kpi];

            void* _KeyPack = &((shaper_t::bm_BaseData_t*)kps.bm[sp.bmid])[1];

            // alloc can be avoided inside switch
            uint8_t* KeyPack = new uint8_t[kps.KeySizesSum];
            std::memcpy(KeyPack, _KeyPack, kps.KeySizesSum);
            
            switch(kpi) {
              case loco_t::kp::light: {
              // doesnt have depth
                break;
              }
              case loco_t::kp::common: {
                ((kps_t::common_t*)KeyPack)->depth = position.z;
                break;
              }
              case loco_t::kp::texture: {
                ((kps_t::texture_t*)KeyPack)->depth = position.z;
                break;
              }
              default: {
                fan::throw_error("unimplemented kp");
              }
            }
            auto _vi = gloco->shaper.GetRenderData(*shape);
            auto vlen = gloco->shaper.ShapeTypes[sp.sti].RenderDataSize;
            uint8_t* vi = new uint8_t[vlen];
            std::memcpy(vi, _vi, vlen);
            ((T*)vi)->position = position;

            auto _ri = gloco->shaper.GetData(*shape);
            auto rlen = gloco->shaper.ShapeTypes[sp.sti].DataSize;
            uint8_t* ri = new uint8_t[rlen];
            std::memcpy(ri, _ri, rlen);

            shape->remove();
            *shape = gloco->shaper.add(
              sp.sti,
              KeyPack,
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
            return get_render_data(shape, &T::size);
          }
          else if constexpr (fan_has_variable(T, radius)) {
            return fan::vec2(get_render_data(shape, &T::radius));
          }
          else {
            fan::throw_error("unimplemented get");
            return fan::vec2();
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
            fan::throw_error("unimplemented get");
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
        .get_color = [](shape_t* shape) {
          if constexpr (fan_has_variable(T, color)) {
            return get_render_data(shape, &T::color);
          }
          else {
            fan::throw_error("unimplemented get");
            return fan::color();
          }
              },
        .set_color = [](shape_t* shape, const fan::color& color) {
          if constexpr (fan_has_variable(T, color)) {
            modify_render_data_element(shape, &T::color, color);
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
            auto sp = gloco->shaper.ShapeList[*shape];

            auto kpi = gloco->shaper.ShapeTypes[sp.sti].KeyPackIndex;
            auto& kps = gloco->shaper.KeyPacks[kpi];

            void* _KeyPack = &((shaper_t::bm_BaseData_t*)kps.bm[sp.bmid])[1];

            // alloc can be avoided inside switch
            uint8_t* KeyPack = new uint8_t[kps.KeySizesSum];
            std::memcpy(KeyPack, _KeyPack, kps.KeySizesSum);

            switch(kpi) {
              case loco_t::kp::texture: {
                ((kps_t::texture_t*)KeyPack)->image = *ti->image;
                break;
              }
              default: {
                fan::throw_error("unimplemented kp");
              }
            }
            auto& im = *ti->image;
            auto& img = gloco->image_get_data(im);

            auto _vi = gloco->shaper.GetRenderData(*shape);
            auto vlen = gloco->shaper.ShapeTypes[sp.sti].RenderDataSize;
            uint8_t* vi = new uint8_t[vlen];
            std::memcpy(vi, _vi, vlen);
            ((T*)vi)->tc_position = ti->position / img.size;
            ((T*)vi)->tc_size = ti->size / img.size;

            auto _ri = gloco->shaper.GetData(*shape);
            auto rlen = gloco->shaper.ShapeTypes[sp.sti].DataSize;
            uint8_t* ri = new uint8_t[rlen];
            std::memcpy(ri, _ri, rlen);

            shape->remove();
            *shape = gloco->shaper.add(
              sp.sti,
              KeyPack,
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
          auto& sp = gloco->shaper.ShapeList[*shape];
          auto kpi = gloco->shaper.ShapeTypes[sp.sti].KeyPackIndex;
          void* KeyPack = &((shaper_t::bm_BaseData_t*)gloco->shaper.KeyPacks[
            kpi
          ].bm[sp.bmid])[1];
          switch (kpi) {
          case loco_t::kp::light: {
            return ((kps_t::light_t*)KeyPack)->camera;
          }
          case loco_t::kp::common: {
            return ((kps_t::common_t*)KeyPack)->camera;
          }
          case loco_t::kp::texture: {
            return ((kps_t::texture_t*)KeyPack)->camera;
          }
          default: {
            fan::throw_error("unimplemented kp");
          }
          }
          return loco_t::camera_t();
        },
        .set_camera = [](shape_t* shape, loco_t::camera_t camera) {
          {
            auto sp = gloco->shaper.ShapeList[*shape];

            auto kpi = gloco->shaper.ShapeTypes[sp.sti].KeyPackIndex;
            auto& kps = gloco->shaper.KeyPacks[kpi];

            void* _KeyPack = &((shaper_t::bm_BaseData_t*)kps.bm[sp.bmid])[1];

            // alloc can be avoided inside switch
            uint8_t* KeyPack = new uint8_t[kps.KeySizesSum];
            std::memcpy(KeyPack, _KeyPack, kps.KeySizesSum);

            switch(kpi) {
              case loco_t::kp::light: {
                ((kps_t::light_t*)KeyPack)->camera = camera;
                break;
              }
              case loco_t::kp::common: {
                ((kps_t::common_t*)KeyPack)->camera = camera;
                break;
              }
              case loco_t::kp::texture: {
                ((kps_t::texture_t*)KeyPack)->camera = camera;
                break;
              }
              default: {
                fan::throw_error("unimplemented kp");
              }
            }
            auto _vi = gloco->shaper.GetRenderData(*shape);
            auto vlen = gloco->shaper.ShapeTypes[sp.sti].RenderDataSize;
            uint8_t* vi = new uint8_t[vlen];
            std::memcpy(vi, _vi, vlen);

            auto _ri = gloco->shaper.GetData(*shape);
            auto rlen = gloco->shaper.ShapeTypes[sp.sti].DataSize;
            uint8_t* ri = new uint8_t[rlen];
            std::memcpy(ri, _ri, rlen);

            shape->remove();
            *shape = gloco->shaper.add(
              sp.sti,
              KeyPack,
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
          auto& sp = gloco->shaper.ShapeList[*shape];
          auto kpi = gloco->shaper.ShapeTypes[sp.sti].KeyPackIndex;
          void* KeyPack = &((shaper_t::bm_BaseData_t*)gloco->shaper.KeyPacks[
            kpi
          ].bm[sp.bmid])[1];
          switch (kpi) {
          case loco_t::kp::light: {
            return ((kps_t::light_t*)KeyPack)->viewport;
          }
          case loco_t::kp::common: {
            return ((kps_t::common_t*)KeyPack)->viewport;
          }
          case loco_t::kp::texture: {
            return ((kps_t::texture_t*)KeyPack)->viewport;
          }
          default: {
            fan::throw_error("unimplemented kp");
          }
          }
          return loco_t::viewport_t();
        },
        .set_viewport = [](shape_t* shape, loco_t::viewport_t viewport) {
          {
            auto sp = gloco->shaper.ShapeList[*shape];

            auto kpi = gloco->shaper.ShapeTypes[sp.sti].KeyPackIndex;
            auto& kps = gloco->shaper.KeyPacks[kpi];

            void* _KeyPack = &((shaper_t::bm_BaseData_t*)kps.bm[sp.bmid])[1];

            // alloc can be avoided inside switch
            uint8_t* KeyPack = new uint8_t[kps.KeySizesSum];

            std::memcpy(KeyPack, _KeyPack, kps.KeySizesSum);
            switch (kpi) {
            case loco_t::kp::light: {
              ((kps_t::light_t*)KeyPack)->viewport = viewport;
              break;
            }
            case loco_t::kp::common: {
              ((kps_t::common_t*)KeyPack)->viewport = viewport;
              break;
            }
            case loco_t::kp::texture: {
              ((kps_t::texture_t*)KeyPack)->viewport = viewport;
              break;
            }
            default: {
              fan::throw_error("unimplemented kp");
            }
            }
            
            auto _vi = gloco->shaper.GetRenderData(*shape);
            auto vlen = gloco->shaper.ShapeTypes[sp.sti].RenderDataSize;
            uint8_t* vi = new uint8_t[vlen];
            std::memcpy(vi, _vi, vlen);

            auto _ri = gloco->shaper.GetData(*shape);
            auto rlen = gloco->shaper.ShapeTypes[sp.sti].DataSize;
            uint8_t* ri = new uint8_t[rlen];
            std::memcpy(ri, _ri, rlen);

            shape->remove();
            *shape = gloco->shaper.add(
              sp.sti,
              KeyPack,
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
        .set_image = [](shape_t* shape, loco_t::image_t image) {
          auto sp = gloco->shaper.ShapeList[*shape];

          auto kpi = gloco->shaper.ShapeTypes[sp.sti].KeyPackIndex;
          auto& kps = gloco->shaper.KeyPacks[kpi];

          void* _KeyPack = &((shaper_t::bm_BaseData_t*)kps.bm[sp.bmid])[1];

          // alloc can be avoided inside switch
          uint8_t* KeyPack = new uint8_t[kps.KeySizesSum];

          std::memcpy(KeyPack, _KeyPack, kps.KeySizesSum);
          switch (kpi) {
          case loco_t::kp::texture: {
            ((kps_t::texture_t*)KeyPack)->image = image;
            break;
          }
          default: {
            fan::throw_error("unimplemented kp");
          }
          }
            
          auto _vi = gloco->shaper.GetRenderData(*shape);
          auto vlen = gloco->shaper.ShapeTypes[sp.sti].RenderDataSize;
          uint8_t* vi = new uint8_t[vlen];
          std::memcpy(vi, _vi, vlen);

          auto _ri = gloco->shaper.GetData(*shape);
          auto rlen = gloco->shaper.ShapeTypes[sp.sti].DataSize;
          uint8_t* ri = new uint8_t[rlen];
          std::memcpy(ri, _ri, rlen);

          shape->remove();
          *shape = gloco->shaper.add(
            sp.sti,
            KeyPack,
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
          loco_t::universal_image_renderer_t::vi_t& vi = *(loco_t::universal_image_renderer_t::vi_t*)gloco->shaper.GetRenderData(*shape);
          loco_t::universal_image_renderer_t::ri_t& ri = *(loco_t::universal_image_renderer_t::ri_t*)gloco->shaper.GetData(*shape);
          if (format != ri.format) {
            auto& sp = gloco->shaper.ShapeList[*shape];

            auto kpi = gloco->shaper.ShapeTypes[sp.sti].KeyPackIndex;
            auto& kps = gloco->shaper.KeyPacks[kpi];

            loco_t::kps_t::texture_t& _KeyPack = *(loco_t::kps_t::texture_t*)&((shaper_t::bm_BaseData_t*)kps.bm[sp.bmid])[1];


            auto shader = gloco->shaper.ShapeTypes[sp.sti].shader;
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
              for (uint32_t i = image_count_new; i < image_count_old; ++i) {
                if (i == 0) {
                  gloco->image_erase(_KeyPack.image);
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
               for (uint32_t i = 0; i < image_count_new; i++) {
                  fan::webp::image_info_t image_info;
                  image_info.data = image_data[i];
                  image_info.size = fan::pixel_format::get_image_sizes(format, image_size)[i];
                  auto lp = fan::pixel_format::get_image_properties<loco_t::image_load_properties_t>(format)[i];
                  lp.min_filter = filter;
                  lp.mag_filter = filter;
                  if (i == 0) {
                    gloco->image_reload_pixels(
                      _KeyPack.image,
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
            }
          }
        },
        .draw = [](uint8_t draw_range) {
          // Implement draw function
        },
      };
    return funcs;
  }

#pragma pack(push, 1)
  struct kps_t {
    struct light_t {
      loco_t::viewport_t viewport;
      loco_t::camera_t camera;
      shaper_t::ShapeTypeIndex_t ShapeType;
    };
    struct common_t {
      depth_t depth;
      blending_t blending;
      loco_t::viewport_t viewport;
      loco_t::camera_t camera;
      shaper_t::ShapeTypeIndex_t ShapeType;
    };
    struct vfi_t {
      uint8_t filler = 0;
    };
    struct texture_t {
      depth_t depth;
      blending_t blending;
      loco_t::image_t image;
      loco_t::viewport_t viewport;
      loco_t::camera_t camera;
      shaper_t::ShapeTypeIndex_t ShapeType;
    };
    // for universal_image_renderer
    // struct texture4_t {
    //   blending_t blending;
    //   depth_t depth;
    //   loco_t::image_t image; // 4 - 1
    //   loco_t::viewport_t viewport;
    //   loco_t::camera_t camera;
    //   shaper_t::ShapeTypeIndex_t ShapeType;
    // };
  };
#pragma pack(pop)


  struct shape_info_t {
    functions_t functions;
  };

private:
  std::vector<shape_info_t> shape_info_list;
public:

  struct properties_t {
    bool vsync = true;
    fan::vec2 window_size = -1;
    uint64_t window_flags = 0;
  };

  void init_framebuffer();

  loco_t();
  loco_t(const properties_t& p);

  void process_frame();

  bool process_loop(const fan::function_t<void()>& lambda);
  void loop(const fan::function_t<void()>& lambda);

  loco_t::camera_t open_camera(const fan::vec2 & x, const fan::vec2 & y);
  loco_t::viewport_t open_viewport(const fan::vec2& viewport_position, const fan::vec2& viewport_size);

  void set_viewport(loco_t::viewport_t viewport, const fan::vec2& viewport_position, const fan::vec2& viewport_size);

  // for checking whether you set depth or no
  struct position3_t : public fan::vec3 {
    using fan::vec3::vec3;
    using fan::vec3::operator=;
    position3_t& operator=(const position3_t& p) {
      fan::vec3::operator=(p);
      return *this;
    }
  };


  //
  fan::vec2 transform_matrix(const fan::vec2& position);

  fan::vec2 screen_to_ndc(const fan::vec2& screen_pos);

  fan::vec2 ndc_to_screen(const fan::vec2& ndc_position);
  //

  uint32_t get_fps();
  void set_vsync(bool flag);

  //-----------------------------gui-----------------------------

#if defined(loco_imgui)
protected:
#define BLL_set_CPP_ConstructDestruct
#define BLL_set_CPP_Node_ConstructDestruct
#define BLL_set_SafeNext 1
#define BLL_set_AreWeInsideStruct 1
#define BLL_set_prefix imgui_draw_cb
#include <fan/fan_bll_preset.h>
#define BLL_set_Link 1
#define BLL_set_type_node uint16_t
#define BLL_set_NodeDataType fan::function_t<void()>
#include _FAN_PATH(BLL/BLL.h)
public:

  using imgui_draw_cb_nr_t = imgui_draw_cb_NodeReference_t;
  imgui_draw_cb_t m_imgui_draw_cb;

  struct imgui_element_nr_t : loco_t::imgui_draw_cb_nr_t {
    using base_t = loco_t::imgui_draw_cb_nr_t;

    imgui_element_nr_t() = default;

    imgui_element_nr_t(const imgui_element_nr_t& nr);

    imgui_element_nr_t(imgui_element_nr_t&& nr);
    ~imgui_element_nr_t();


    imgui_element_nr_t& operator=(const imgui_element_nr_t& id);

    imgui_element_nr_t& operator=(imgui_element_nr_t&& id);

    void init();

    bool is_invalid() const;

    void invalidate_soft();

    void invalidate();

    inline void set(const auto& lambda) {
      gloco->m_imgui_draw_cb[*this] = lambda;
    }
  };

  struct imgui_element_t : imgui_element_nr_t {
    imgui_element_t() = default;
    imgui_element_t(const auto& lambda) {
      imgui_element_nr_t::init();
      imgui_element_nr_t::set(lambda);
    }
  };

#define fan_imgui_dragfloat_named(name, variable, speed, m_min, m_max) \
  [=] <typename T5>(T5& var) -> bool{ \
    if constexpr(std::is_same_v<f32_t, T5>)  { \
      return ImGui::DragFloat(fan::string(std::move(name)).c_str(), &var, (f32_t)speed, (f32_t)m_min, (f32_t)m_max); \
    } \
    else if constexpr(std::is_same_v<fan::vec2, T5>)  { \
      return ImGui::DragFloat2(fan::string(std::move(name)).c_str(), var.data(), (f32_t)speed, (f32_t)m_min, (f32_t)m_max); \
    } \
    else if constexpr(std::is_same_v<fan::vec3, T5>)  { \
      return ImGui::DragFloat3(fan::string(std::move(name)).c_str(), var.data(), (f32_t)speed, (f32_t)m_min, (f32_t)m_max); \
    } \
    else if constexpr(std::is_same_v<fan::vec4, T5>)  { \
      return ImGui::DragFloat4(fan::string(std::move(name)).c_str(), var.data(), (f32_t)speed, (f32_t)m_min, (f32_t)m_max); \
    } \
    else {\
      fan::throw_error_impl(); \
      return 0;\
    } \
  }(variable)

#define fan_imgui_dragfloat(variable, speed, m_min, m_max) \
    fan_imgui_dragfloat_named(STRINGIFY(variable), variable, speed, m_min, m_max)

  struct imgui_fs_var_t {
    loco_t::imgui_element_t ie;

    imgui_fs_var_t() = default;

    template <typename T>
    imgui_fs_var_t(
      loco_t::shader_t shader_nr,
      const fan::string& var_name,
      T initial = 0,
      T speed = 1,
      T min = -100000,
      T max = 100000
    ) {
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
            f32_t d = data;
            modify = ImGui::DragFloat(fan::string(std::move(var_name)).c_str(), (f32_t*)&d, (f32_t)speed, (f32_t)min, (f32_t)max);
            data = d;
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
    }
  };

  void set_imgui_viewport(loco_t::viewport_t viewport);

#endif
  //-----------------------------gui-----------------------------

  fan::opengl::context_t& get_context();

  struct camera_impl_t {

    camera_impl_t() = default;
    loco_t::camera_t camera;
    loco_t::viewport_t viewport;
  };

  struct input_action_t {
    enum {
      none = -1,
      release = (int)fan::keyboard_state::release,
      press = (int)fan::keyboard_state::press,
      repeat = (int)fan::keyboard_state::repeat
    };

    struct action_data_t {
      static constexpr int max_key_combos = 5;
      int keys[max_key_combos]{};
      uint8_t count = 0;
      bool combo = 0;
    };

    void add(const int* keys, std::size_t count, std::string_view action_name);
    void add(int key, std::string_view action_name);
    void add(std::initializer_list<int> keys, std::string_view action_name);

    int is_active(std::string_view action_name, int press = 1);

    std::unordered_map<std::string_view, action_data_t> input_actions;
  }input_action;

protected:
  #define BLL_set_CPP_ConstructDestruct
  #define BLL_set_CPP_Node_ConstructDestruct
  #define BLL_set_SafeNext 1
  #define BLL_set_AreWeInsideStruct 1
  #define BLL_set_prefix update_callback
  #include _FAN_PATH(fan_bll_preset.h)
  #define BLL_set_Link 1
  #define BLL_set_type_node uint16_t
  #define BLL_set_NodeDataType fan::function_t<void(loco_t*)>
  #include _FAN_PATH(BLL/BLL.h)
public:

  using update_callback_nr_t = update_callback_NodeReference_t;

  update_callback_t m_update_callback;

  std::vector<fan::function_t<void()>> single_queue;

  image_t default_texture;

  camera_impl_t orthographic_camera;
  camera_impl_t perspective_camera;

  fan::window_t window;

  f64_t& delta_time = window.m_delta_time;

  std::vector<functions_t> shape_functions;

  // needs continous buffer
  std::vector<shaper_t::BlockProperties_t> BlockProperties;

  shaper_t shaper;
  
#pragma pack(push, 1)

  struct Key_e {
    enum {
      blending,
      depth,
      image,
      viewport,
      camera,
      ShapeType,
      filler
    };
  };

#pragma pack(pop)

  fan::vec2 get_mouse_position(const loco_t::camera_t& camera, const loco_t::viewport_t& viewport);
  fan::vec2 get_mouse_position();

  static fan::vec2 translate_position(const fan::vec2& p, loco_t::viewport_t viewport, loco_t::camera_t camera);
  fan::vec2 translate_position(const fan::vec2& p);

  struct shape_t : shaper_t::ShapeID_t{
    shape_t() {
      sic();
    }
    shape_t(shaper_t::ShapeID_t&& s) {
      NRI = s.NRI;
      s.sic();
    }
    shape_t(const shaper_t::ShapeID_t& s) : shape_t() {

      if (s.iic()) {
        return;
      }

      {
        auto& sp = gloco->shaper.ShapeList[s];

        auto kpi = gloco->shaper.ShapeTypes[sp.sti].KeyPackIndex;
        auto kps = gloco->shaper.KeyPacks[kpi];

        void* _KeyPack = &((shaper_t::bm_BaseData_t*)kps.bm[sp.bmid])[1];

        // alloc can be avoided inside switch
        uint8_t* KeyPack = new uint8_t[kps.KeySizesSum];
        std::memcpy(KeyPack, _KeyPack, kps.KeySizesSum);


        auto _vi = gloco->shaper.GetRenderData(s);
        auto vlen = gloco->shaper.ShapeTypes[sp.sti].RenderDataSize;
        uint8_t* vi = new uint8_t[vlen];
        std::memcpy(vi, _vi, vlen);

        auto _ri = gloco->shaper.GetData(s);
        auto rlen = gloco->shaper.ShapeTypes[sp.sti].DataSize;
        uint8_t* ri = new uint8_t[rlen];
        std::memcpy(ri, _ri, rlen);

        *this = gloco->shaper.add(
          sp.sti, 
          KeyPack,
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
    }

    template <typename T>
    requires requires(T t) { typename T::type_t; }
    shape_t(const T& properties) : shape_t() {
      if constexpr (std::is_same_v<T, light_t::properties_t>) {
        *this = gloco->light.push_back(properties);
      }
      else if constexpr (std::is_same_v<T, line_t::properties_t>) {
        *this = gloco->line.push_back(properties);
      }
      else if constexpr (std::is_same_v<T, rectangle_t::properties_t>) {
        *this = gloco->rectangle.push_back(properties);
      }
      else if constexpr (std::is_same_v<T, sprite_t::properties_t>) {
        *this = gloco->sprite.push_back(properties);
      }
      else if constexpr (std::is_same_v<T, unlit_sprite_t::properties_t>) {
        *this = gloco->unlit_sprite.push_back(properties);
      }
      else if constexpr (std::is_same_v<T, letter_t::properties_t>) {
        if constexpr (fan_has_variable(loco_t, letter)) {
          *this = gloco->letter.push_back(properties);
        }
      }
      else if constexpr (std::is_same_v<T, circle_t::properties_t>) {
        if constexpr (fan_has_variable(loco_t, circle)) {
          *this = gloco->circle.push_back(properties);
        }
      }
      else if constexpr (std::is_same_v<T, grid_t::properties_t>) {
        *this = gloco->grid.push_back(properties);
      }
      else if constexpr (std::is_same_v<T, loco_t::vfi_t::common_shape_properties_t>) {
        *this = gloco->vfi.push_back(properties);
      }
      else if constexpr (std::is_same_v<T, loco_t::particles_t::properties_t>) {
        *this = gloco->particles.push_back(properties);
      }
      else if constexpr (std::is_same_v<T, loco_t::universal_image_renderer_t::properties_t>) {
        *this = gloco->universal_image_renderer.push_back(properties);
      }
      else {
        fan::throw_error("failed to find correct shape", typeid(T).name());
      }
#if defined(debug_shape_t)
      fan::print("+", NRI);
#endif
    }
    shape_t(shape_t&& s) : shape_t(std::move(*dynamic_cast<shaper_t::ShapeID_t*>(&s))) {

    }
    shape_t(const shape_t& s) : shape_t(*dynamic_cast<const shaper_t::ShapeID_t*>(&s)) {
      //NRI = s.NRI;
    }
    shape_t& operator=(const shape_t& s) {
      if (iic() == false) {
        remove();
      }
      if (s.iic()) {
        return *this;
      }
      if (this != &s) {
        {
          auto& sp = gloco->shaper.ShapeList[s];

          auto kpi = gloco->shaper.ShapeTypes[sp.sti].KeyPackIndex;
          auto& kps = gloco->shaper.KeyPacks[kpi];

          void* _KeyPack = &((shaper_t::bm_BaseData_t*)kps.bm[sp.bmid])[1];

          // alloc can be avoided inside switch
          uint8_t* KeyPack = new uint8_t[kps.KeySizesSum];
          std::memcpy(KeyPack, _KeyPack, kps.KeySizesSum);


          auto _vi = gloco->shaper.GetRenderData(s);
          auto vlen = gloco->shaper.ShapeTypes[sp.sti].RenderDataSize;
          uint8_t* vi = new uint8_t[vlen];
          std::memcpy(vi, _vi, vlen);

          auto _ri = gloco->shaper.GetData(s);
          auto rlen = gloco->shaper.ShapeTypes[sp.sti].DataSize;
          uint8_t* ri = new uint8_t[rlen];
          std::memcpy(ri, _ri, rlen);

          *this = gloco->shaper.add(
            sp.sti,
            KeyPack,
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
        //fan::print("i dont know what to do");
        //NRI = s.NRI;
      }
      return *this;
    }
    shape_t& operator=(shape_t&& s) {
      if (iic() == false) {
        remove();
      }
      if (s.iic()) {
        return *this;
      }

      if (this != &s) {
        NRI = s.NRI;
        s.sic();
      }
      return *this;
    }
    ~shape_t() {
      remove();
    }

    void remove() {
      if (iic()) {
        return;
      }
#if defined(debug_shape_t)
      fan::print("-", NRI);
#endif
      if (get_shape_type() == loco_t::shape_type_t::vfi) {
        gloco->vfi.erase(*this);
      }
      else {
        gloco->shaper.remove(*this);
      }
      sic();
    }

    void erase() {
      remove();
    }

    // many things assume uint16_t so thats why not shaper_t::ShapeTypeIndex_t
    uint16_t get_shape_type() {
      return gloco->shaper.ShapeList[*this].sti;
    }

    template <typename T>
    void set_position(const fan::vec2_wrap_t<T>& position) {
      gloco->shape_functions[gloco->shaper.ShapeList[*this].sti].set_position2(this, position);
    }

    void set_position(const fan::vec3& position);

    fan::vec3 get_position();

    void set_size(const fan::vec2& size);

    fan::vec2 get_size();

    void set_rotation_point(const fan::vec2& rotation_point);

    fan::vec2 get_rotation_point();

    void set_color(const fan::color& color);

    fan::color get_color();

    void set_angle(const fan::vec3& angle);

    fan::vec3 get_angle();

    fan::vec2 get_tc_position();
    void set_tc_position(const fan::vec2& tc_position);

    fan::vec2 get_tc_size();
    void set_tc_size(const fan::vec2& tc_size);

    bool load_tp(loco_t::texturepack_t::ti_t* ti);
    loco_t::texturepack_t::ti_t get_tp();
    bool set_tp(loco_t::texturepack_t::ti_t* ti);

    fan::vec2 get_grid_size();
    void set_grid_size(const fan::vec2& grid_size);

    loco_t::camera_t get_camera();
    void set_camera(loco_t::camera_t camera);
    loco_t::viewport_t get_viewport();
    void set_viewport(loco_t::viewport_t viewport);

    void set_image(loco_t::image_t image);

    f32_t get_parallax_factor();
    void set_parallax_factor(f32_t parallax_factor);

    fan::vec3 get_rotation_vector();

    uint32_t get_flags();

    f32_t get_radius();
    fan::vec3 get_src();
    fan::vec3 get_dst();
    f32_t get_outline_size();
    fan::color get_outline_color();

    void reload(uint8_t format, void** image_data, const fan::vec2& image_size, uint32_t filter = fan::opengl::GL_LINEAR);

  private:
  };


  struct light_t {

    static constexpr uint16_t shape_type = shape_type_t::light;
    static constexpr int kpi = kp::light;

#pragma pack(push, 1)

    struct vi_t {
      fan::vec3 position;
      f32_t parallax_factor;
      fan::vec2 size;
      fan::vec2 rotation_point;
      fan::color color;
      fan::vec3 rotation_vector;
      uint32_t flags;
      fan::vec3 angle;
    };;
    struct ri_t {

    };

#pragma pack(pop)

    inline static std::vector<shaper_t::ShapeType_t::init_t> locations = {
      shaper_t::ShapeType_t::init_t{0, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position)},
      shaper_t::ShapeType_t::init_t{1, 1, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, parallax_factor))},
      shaper_t::ShapeType_t::init_t{2, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, size))},
      shaper_t::ShapeType_t::init_t{3, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, rotation_point))},
      shaper_t::ShapeType_t::init_t{4, 4, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color))},
      shaper_t::ShapeType_t::init_t{5, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, rotation_vector))},
      shaper_t::ShapeType_t::init_t{6, 1, fan::opengl::GL_UNSIGNED_INT , sizeof(vi_t), (void*)(offsetof(vi_t, flags))},
      shaper_t::ShapeType_t::init_t{7, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, angle))},
    };

    struct properties_t {
      using type_t = light_t;

      fan::vec3 position = 0;
      f32_t parallax_factor = 0;
      fan::vec2 size = 0;
      fan::vec2 rotation_point = 0;
      fan::color color = fan::colors::white;
      fan::vec3 rotation_vector = fan::vec3(0, 0, 1);
      uint32_t flags = 0;
      fan::vec3 angle = 0;

      loco_t::camera_t camera = gloco->orthographic_camera.camera;
      loco_t::viewport_t viewport = gloco->orthographic_camera.viewport;
    };


    shape_t push_back(const properties_t& properties);
  }light;

  struct line_t {

    static constexpr uint16_t shape_type = shape_type_t::line;
    static constexpr int kpi = kp::common;

#pragma pack(push, 1)

    struct vi_t {
      fan::color color;
      fan::vec3 src;
      fan::vec3 dst;
    };
    struct ri_t {

    };

#pragma pack(pop)

  inline static std::vector<shaper_t::ShapeType_t::init_t> locations = {
    shaper_t::ShapeType_t::init_t{0, 4, fan::opengl::GL_FLOAT, sizeof(line_t::vi_t), (void*)offsetof(line_t::vi_t, color)},
    shaper_t::ShapeType_t::init_t{1, 3, fan::opengl::GL_FLOAT, sizeof(line_t::vi_t), (void*)offsetof(line_t::vi_t, src)},
    shaper_t::ShapeType_t::init_t{2, 3, fan::opengl::GL_FLOAT, sizeof(line_t::vi_t), (void*)offsetof(line_t::vi_t, dst)}
  };

    struct properties_t {
      using type_t = line_t;

      fan::color color = fan::colors::white;
      fan::vec3 src;
      fan::vec3 dst;

      bool blending = false;

      loco_t::camera_t camera = gloco->orthographic_camera.camera;
      loco_t::viewport_t viewport = gloco->orthographic_camera.viewport;
    };


    shape_t push_back(const properties_t& properties);

  }line;

  struct rectangle_t {

    static constexpr uint16_t shape_type = shape_type_t::rectangle;
    static constexpr int kpi = kp::common;

#pragma pack(push, 1)

    struct vi_t {
      fan::vec3 position;
      fan::vec2 size;
      fan::vec2 rotation_point;
      fan::color color;
      fan::vec3 angle;
    };
    struct ri_t {
      
    };

#pragma pack(pop)

    inline static  std::vector<shaper_t::ShapeType_t::init_t> locations = {
      shaper_t::ShapeType_t::init_t{0, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position)},
      shaper_t::ShapeType_t::init_t{1, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, size))},
      shaper_t::ShapeType_t::init_t{2, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, rotation_point))},
      shaper_t::ShapeType_t::init_t{3, 4, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color))},
      shaper_t::ShapeType_t::init_t{4, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, angle))}
    };

    struct properties_t {
      using type_t = rectangle_t;

      fan::vec3 position = 0;
      fan::vec2 size = 0;
      fan::color color = fan::colors::white;
      bool blending = false;
      fan::vec3 angle = 0;
      fan::vec2 rotation_point = 0;

      loco_t::camera_t camera = gloco->orthographic_camera.camera;
      loco_t::viewport_t viewport = gloco->orthographic_camera.viewport;
    };


    shape_t push_back(const properties_t& properties);

  }rectangle;

  //----------------------------------------------------------


  struct sprite_t {

    static constexpr uint16_t shape_type = shape_type_t::sprite;
    static constexpr int kpi = kp::texture;

#pragma pack(push, 1)

    struct vi_t {
      fan::vec3 position;
      f32_t parallax_factor;
      fan::vec2 size;
      fan::vec2 rotation_point;
      fan::color color;
      fan::vec3 angle;
      uint32_t flags;
      fan::vec2 tc_position;
      fan::vec2 tc_size;
    };
    struct ri_t {

    };

#pragma pack(pop)

  inline static std::vector<shaper_t::ShapeType_t::init_t> locations = {
    shaper_t::ShapeType_t::init_t{0, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position)},
    shaper_t::ShapeType_t::init_t{1, 1, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, parallax_factor))},
    shaper_t::ShapeType_t::init_t{2, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, size))},
    shaper_t::ShapeType_t::init_t{3, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, rotation_point))},
    shaper_t::ShapeType_t::init_t{4, 4, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color))},
    shaper_t::ShapeType_t::init_t{5, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, angle))},
    shaper_t::ShapeType_t::init_t{6, 1, fan::opengl::GL_UNSIGNED_INT , sizeof(vi_t), (void*)(offsetof(vi_t, flags))},
    shaper_t::ShapeType_t::init_t{7, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, tc_position))},
    shaper_t::ShapeType_t::init_t{8, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, tc_size))}
  };

    struct properties_t {
      using type_t = sprite_t;

      fan::vec3 position = 0;
      f32_t parallax_factor = 0;
      fan::vec2 size = 0;
      fan::vec2 rotation_point = 0;
      fan::color color = fan::colors::white;
      fan::vec3 angle = fan::vec3(0);
      int flags = 0;
      fan::vec2 tc_position = 0;
      fan::vec2 tc_size = 1;
      uint16_t shape_type = loco_t::shape_type_t::sprite;

      bool load_tp(loco_t::texturepack_t::ti_t* ti) {
        auto& im = *ti->image;
        image = im;
        auto& img = gloco->image_get_data(im);
        tc_position = ti->position / img.size;
        tc_size = ti->size / img.size;
        return 0;
      }

      bool blending = false;

      loco_t::image_t image = gloco->default_texture;
      loco_t::camera_t camera = gloco->orthographic_camera.camera;
      loco_t::viewport_t viewport = gloco->orthographic_camera.viewport;
    };

    shape_t push_back(const properties_t& properties);

  }sprite;

  struct unlit_sprite_t {

    static constexpr uint16_t shape_type = shape_type_t::unlit_sprite;
    static constexpr int kpi = kp::texture;

#pragma pack(push, 1)

    struct vi_t {
      fan::vec3 position;
      f32_t parallax_factor;
      fan::vec2 size;
      fan::vec2 rotation_point;
      fan::color color;
      fan::vec3 angle;
      uint32_t flags;
      fan::vec2 tc_position;
      fan::vec2 tc_size;
    };
    struct ri_t {

    };

#pragma pack(pop)

    inline static std::vector<shaper_t::ShapeType_t::init_t> locations = {
      shaper_t::ShapeType_t::init_t{0, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position)},
      shaper_t::ShapeType_t::init_t{1, 1, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, parallax_factor))},
      shaper_t::ShapeType_t::init_t{2, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, size))},
      shaper_t::ShapeType_t::init_t{3, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, rotation_point))},
      shaper_t::ShapeType_t::init_t{4, 4, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color))},
      shaper_t::ShapeType_t::init_t{5, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, angle))},
      shaper_t::ShapeType_t::init_t{6, 1, fan::opengl::GL_UNSIGNED_INT , sizeof(vi_t), (void*)(offsetof(vi_t, flags))},
      shaper_t::ShapeType_t::init_t{7, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, tc_position))},
      shaper_t::ShapeType_t::init_t{8, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, tc_size))}
    };

    struct properties_t {
      using type_t = unlit_sprite_t;

      fan::vec3 position = 0;
      f32_t parallax_factor = 0;
      fan::vec2 size = 0;
      fan::vec2 rotation_point = 0;
      fan::color color = fan::colors::white;
      fan::vec3 angle = fan::vec3(0);
      int flags = 0;
      fan::vec2 tc_position = 0;
      fan::vec2 tc_size = 1;

      bool blending = false;

      loco_t::image_t image = gloco->default_texture;
      loco_t::camera_t camera = gloco->orthographic_camera.camera;
      loco_t::viewport_t viewport = gloco->orthographic_camera.viewport;


      bool load_tp(loco_t::texturepack_t::ti_t* ti) {
        auto& im = *ti->image;
        image = im;
        auto& img = gloco->image_get_data(im);
        tc_position = ti->position / img.size;
        tc_size = ti->size / img.size;
        return 0;
      }
    };

    shape_t push_back(const properties_t& properties);

  }unlit_sprite;


  struct letter_t {

    static constexpr uint16_t shape_type = shape_type_t::letter;
    static constexpr int kpi = kp::common;

#pragma pack(push, 1)

    struct vi_t {
      fan::vec3 position;
      f32_t outline_size;
      fan::vec2 size;
      fan::vec2 tc_position;
      fan::color color;
      fan::color outline_color;
      fan::vec2 tc_size;
      fan::vec3 angle;
    };
    
    struct ri_t {
      uint32_t letter_id;
      f32_t font_size;
    };

#pragma pack(pop)

    inline static std::vector<shaper_t::ShapeType_t::init_t> locations = {
      shaper_t::ShapeType_t::init_t{0, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position)},
      shaper_t::ShapeType_t::init_t{1, 1, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, outline_size))},
      shaper_t::ShapeType_t::init_t{2, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, size))},
      shaper_t::ShapeType_t::init_t{3, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, tc_position))},
      shaper_t::ShapeType_t::init_t{4, 4, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color))},
      shaper_t::ShapeType_t::init_t{5, 4, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, outline_color))},
      shaper_t::ShapeType_t::init_t{6, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, tc_size))},
      shaper_t::ShapeType_t::init_t{7, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, angle))}
    };

    struct properties_t {
      using type_t = letter_t;

      fan::vec3 position;
      f32_t outline_size = 1;
      fan::vec2 size;
      fan::vec2 tc_position;
      fan::color color = fan::colors::white;
      fan::color outline_color;
      fan::vec2 tc_size;
      fan::vec3 angle = 0;

      bool blending = true;

      uint32_t letter_id;
      f32_t font_size;

      loco_t::camera_t camera = gloco->orthographic_camera.camera;
      loco_t::viewport_t viewport = gloco->orthographic_camera.viewport;
    };

    shape_t push_back(const properties_t& properties);

  }letter;

  struct text_t {

    struct vi_t {

    };

    struct ri_t {
      //letter_t::properties_t p;
    };

    struct properties_t {
      using type_t = text_t;

      loco_t::camera_t camera = gloco->orthographic_camera.camera;
      loco_t::viewport_t viewport = gloco->orthographic_camera.viewport;

      fan::vec3 position;
      f32_t outline_size = 1;
      fan::vec2 size;
      fan::vec2 tc_position;
      fan::color color = fan::colors::white;
      fan::color outline_color;
      fan::vec2 tc_size;
      fan::vec3 angle = 0;

      fan::string text;
    };

    shape_t push_back(const properties_t& properties);
  }text;

  struct circle_t {

    static constexpr uint16_t shape_type = shape_type_t::circle;
    static constexpr int kpi = kp::common;

#pragma pack(push, 1)

    struct vi_t {
      fan::vec3 position;
      f32_t radius;
      fan::vec2 rotation_point;
      fan::color color;
      fan::vec3 rotation_vector;
      fan::vec3 angle;
    };
    struct ri_t {

    };

#pragma pack(pop)

    inline static std::vector<shaper_t::ShapeType_t::init_t> locations = {
      shaper_t::ShapeType_t::init_t{ 0, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position) },
      shaper_t::ShapeType_t::init_t{ 1, 1, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, radius)) },
      shaper_t::ShapeType_t::init_t{ 2, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, rotation_point)) },
      shaper_t::ShapeType_t::init_t{ 3, 4, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, color)) },
      shaper_t::ShapeType_t::init_t{ 4, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, rotation_vector)) },
      shaper_t::ShapeType_t::init_t{ 5, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, angle)) }
    };

    struct properties_t {
      using type_t = circle_t;

      fan::vec3 position = 0;
      f32_t radius = 0;
      fan::vec2 rotation_point = 0;
      fan::color color = fan::colors::white;
      fan::vec3 rotation_vector = fan::vec3(0, 0, 1);
      fan::vec3 angle = 0;

      bool blending = false;

      loco_t::camera_t camera = gloco->orthographic_camera.camera;
      loco_t::viewport_t viewport = gloco->orthographic_camera.viewport;
    };


    shaper_t::ShapeID_t push_back(const circle_t::properties_t& properties);

  }circle;

  struct grid_t {

    static constexpr uint16_t shape_type = shape_type_t::grid;
    static constexpr int kpi = kp::common;

#pragma pack(push, 1)

    struct vi_t {
      fan::vec3 position;
      fan::vec2 size;
      fan::vec2 grid_size;
      fan::vec2 rotation_point;
      fan::color color;
      fan::vec3 angle;
    };
    struct ri_t {
      
    };

#pragma pack(pop)

    inline static std::vector<shaper_t::ShapeType_t::init_t> locations = {
      shaper_t::ShapeType_t::init_t{0, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position)},
      shaper_t::ShapeType_t::init_t{1, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, size)},
      shaper_t::ShapeType_t::init_t{2, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, grid_size)},
      shaper_t::ShapeType_t::init_t{3, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, rotation_point)},
      shaper_t::ShapeType_t::init_t{4, 4, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, color)},
      shaper_t::ShapeType_t::init_t{5, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, angle)},
    };

    struct properties_t {
      using type_t = grid_t;

      fan::vec3 position = 0;
      fan::vec2 size = 0;
      fan::vec2 grid_size;
      fan::vec2 rotation_point = 0;
      fan::color color = fan::colors::white;
      fan::vec3 angle = 0;

      bool blending = false;

      loco_t::camera_t camera = gloco->orthographic_camera.camera;
      loco_t::viewport_t viewport = gloco->orthographic_camera.viewport;
    };

    shape_t push_back(const properties_t& properties);
  }grid;


  struct particles_t {

    static constexpr uint16_t shape_type = shape_type_t::particles;
    static constexpr int kpi = kp::texture;

    inline static std::vector<shaper_t::ShapeType_t::init_t> locations = {};

#pragma pack(push, 1)

    struct vi_t {
      
    };

    struct shapes_e {
      enum {
        circle,
        rectangle
      };
    };

    struct ri_t {

      fan::vec3 position;
      fan::vec2 size;
      fan::color color;

      uint64_t begin_time;
      uint64_t alive_time;
      uint64_t respawn_time;
      uint32_t count;
      fan::vec2 position_velocity;
      fan::vec3 angle_velocity;
      f32_t begin_angle;
      f32_t end_angle;

      fan::vec3 angle;

      fan::vec2 gap_size;
      fan::vec2 max_spread_size;
      fan::vec2 size_velocity;

      uint32_t shape;

      bool blending;
    };
#pragma pack(pop)

    struct properties_t {
      using type_t = particles_t;

      fan::vec3 position = 0;
      fan::vec2 size = 100;
      fan::color color = fan::colors::red;

      uint64_t begin_time;
      uint64_t alive_time = (uint64_t)1e+9;
      uint64_t respawn_time = 0;
      uint32_t count = 10;
      fan::vec2 position_velocity = 130;
      fan::vec3 angle_velocity = fan::vec3(0, 0, 0);
      f32_t begin_angle = 0;
      f32_t end_angle = fan::math::pi * 2;

      fan::vec3 angle = 0;

      fan::vec2 gap_size = 1;
      fan::vec2 max_spread_size = 100;
      fan::vec2 size_velocity = 1;

      uint32_t shape = shapes_e::circle;

      bool blending = true;

      loco_t::image_t image = gloco->default_texture;
      loco_t::camera_t camera = gloco->orthographic_camera.camera;
      loco_t::viewport_t viewport = gloco->orthographic_camera.viewport;
    };

    shape_t push_back(const properties_t& properties);

  }particles;

  struct universal_image_renderer_t {

    static constexpr uint16_t shape_type = shape_type_t::universal_image_renderer;
    static constexpr int kpi = kp::texture;

#pragma pack(push, 1)

    struct vi_t {
      fan::vec3 position = 0;
      fan::vec2 size = 0;
      fan::vec2 tc_position = 0;
      fan::vec2 tc_size = 1;
    };
    struct ri_t {
      loco_t::image_t images_rest[3]; // 3 + 1 (pk)
      uint8_t format = fan::pixel_format::undefined;
    };

#pragma pack(pop)

  inline static std::vector<shaper_t::ShapeType_t::init_t> locations = {
    shaper_t::ShapeType_t::init_t{0, 3, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position)},
    shaper_t::ShapeType_t::init_t{1, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, size))},
    shaper_t::ShapeType_t::init_t{2, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, tc_position))},
    shaper_t::ShapeType_t::init_t{3, 2, fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)(offsetof(vi_t, tc_size))}
  };

    struct properties_t {
      using type_t = universal_image_renderer_t;

      fan::vec3 position = 0;
      fan::vec2 size = 0;
      fan::vec2 tc_position = 0;
      fan::vec2 tc_size = 1;

      bool blending = false;

      loco_t::image_t images[4] = {
        gloco->default_texture,
        gloco->default_texture,
        gloco->default_texture,
        gloco->default_texture
      };
      loco_t::camera_t camera = gloco->orthographic_camera.camera;
      loco_t::viewport_t viewport = gloco->orthographic_camera.viewport;
    };

    shape_t push_back(const properties_t& properties);

  }universal_image_renderer;


  template <typename T>
  inline void shape_open(const fan::string& vertex, const fan::string& fragment) {
    auto& context = gloco->get_context();

    loco_t::shader_t shader = context.shader_create();

    context.shader_set_vertex(shader,
      context.read_shader(vertex)
    );

    context.shader_set_fragment(shader,
      context.read_shader(fragment)
    );

    context.shader_compile(shader);

    gloco->shaper.AddShapeType(
      T::shape_type,
      T::kpi,
      {
        .MaxElementPerBlock = (shaper_t::MaxElementPerBlock_t)MaxElementPerBlock,
        .RenderDataSize = sizeof(typename T::vi_t),
        .DataSize = sizeof(typename T::ri_t),
        .locations = T::locations,
        .shader = shader
      }
    );

    loco_t::functions_t functions = loco_t::get_functions<typename T::vi_t>();
    gloco->shape_functions.push_back(functions);
  }


#if defined(loco_vfi)
  #include <fan/graphics/gui/vfi.h>
#endif
  vfi_t vfi;

//#if defined(loco_texture_pack)
//#endif

  #include <fan/tp/tp0.h>

#if defined(loco_post_process)
  #include <fan/graphics/opengl/2D/effects/blur.h>
    blur_t blur[1];
  #include <fan/graphics/opengl/2D/effects/bloom.h>
  bloom_t bloom;
#endif

#if defined(loco_letter)
  fan::graphics::gl_font_impl::font_t font;
#endif

  fan::color clear_color = { 0.10f, 0.10f, 0.131f, 1.f };

#if defined(loco_framebuffer)
#if defined(loco_opengl)

  fan::opengl::core::framebuffer_t m_framebuffer;
  fan::opengl::core::renderbuffer_t m_rbo;
  loco_t::image_t color_buffers[4];
  loco_t::shader_t m_fbo_final_shader;

#endif
#endif

  struct lighting_t {
    static constexpr const char* ambient_name = "lighting_ambient";
    fan::vec3 ambient = fan::vec3(1, 1, 1);
  }lighting;

  //gui
#if defined(loco_imgui)
  fan::console_t console;
  bool toggle_console = false;
  bool toggle_fps = false;

  ImFont* fonts[6];
#endif
  //gui
};

// user friendly functions
/***************************************/
namespace fan {
  namespace graphics {

    using vfi_t = loco_t::vfi_t;

#if defined(loco_imgui)
    using imgui_element_t = loco_t::imgui_element_t;
#endif

    using camera_impl_t = loco_t::camera_impl_t;
    using camera_t = camera_impl_t;

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
      fan::vec2 size = fan::vec2(0.1, 0.1);
      fan::color color = fan::color(1, 1, 1, 1);
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
            .angle = p.angle,
            .rotation_point = p.rotation_point,
            .blending = p.blending
          )
        );
      }
    };

    // a bit bad because if sprite_t::properties or vi change need to update here
    struct sprite_properties_t {
      camera_impl_t* camera = &gloco->orthographic_camera;
      fan::vec3 position = fan::vec3(0, 0, 0);
      fan::vec2 size = fan::vec2(0.1, 0.1);
      fan::vec3 angle = 0;
      fan::color color = fan::color(1, 1, 1, 1);
      fan::vec2 rotation_point = 0;
      loco_t::image_t image = gloco->default_texture;
      f32_t parallax_factor = 0;
      bool blending = false;
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
            .color = p.color,
            .rotation_point = p.rotation_point,
            .blending = p.blending
          ));
      }
    };

    //struct text_properties_t {
    //  camera_impl_t* camera = &gloco->orthographic_camera;
    //  std::string text = "";
    //  fan::color color = fan::colors::white;
    //  fan::color outline_color = fan::colors::black;
    //  fan::vec3 position = fan::vec3(fan::math::inf, -0.9, 0);
    //  fan::vec2 size = -1;
    //};

    //struct text_t : loco_t::shape_t {
    //  text_t(text_properties_t p = text_properties_t()) {
    //    *(loco_t::shape_t*)this = loco_t::shape_t(
    //      fan_init_struct(
    //        typename loco_t::shapes_t::responsive_text_t::properties_t,
    //        .camera = p.camera->camera,
    //        .viewport = p.camera->viewport,
    //        .position = p.position.x == fan::math::inf ? fan::vec3(-1 + 0.025 * p.text.size(), -0.9, 0) : p.position,
    //        .text = p.text,
    //        .line_limit = 1,
    //        .outline_color = p.outline_color,
    //        .letter_size_y_multipler = 1,
    //        .size = p.size == -1 ? fan::vec2(0.025 * p.text.size(), 0.1) : p.size,
    //        .color = p.color
    //      ));
    //  }
    //};

    struct unlit_sprite_properties_t {
      camera_impl_t* camera = &gloco->orthographic_camera;
      fan::vec3 position = fan::vec3(0, 0, 0);
      fan::vec2 size = fan::vec2(0.1, 0.1);
      fan::vec3 angle = 0;
      fan::color color = fan::color(1, 1, 1, 1);
      fan::vec2 rotation_point = 0;
      loco_t::image_t image = gloco->default_texture;
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
            .color = p.color,
            .tc_position = p.tc_position,
            .tc_size = p.tc_size,
            .rotation_point = p.rotation_point,
            .blending = p.blending
          ));
      }
    };

    
  struct letter_properties_t {
    camera_impl_t* camera = &gloco->orthographic_camera;
    fan::vec3 position = fan::vec3(0, 0, 0);
    f32_t outline_size = 1;
    fan::vec2 size = fan::vec2(0.1, 0.1);
    fan::color color = fan::color(1, 1, 1, 1);
    fan::color outline_color = fan::color(0, 0, 0, 1);
    fan::vec3 angle = fan::vec3(0, 0, 0);
    fan::vec2 tc_position = 0;
    fan::vec2 tc_size = 1;

    uint32_t letter_id;
    f32_t font_size;

    bool blending = true;
  };

  struct letter_t : loco_t::shape_t {
    letter_t(letter_properties_t p = letter_properties_t()) {
      *(loco_t::shape_t*)this = loco_t::shape_t(
        fan_init_struct(
          typename loco_t::letter_t::properties_t,
          .camera = p.camera->camera,
          .viewport = p.camera->viewport,
          .position = p.position,
          .outline_size = p.outline_size,
          .size = p.size,
          .color = p.color,
          .outline_color = p.outline_color,
          .angle = p.angle,
          .tc_position = p.tc_position,
          .tc_size = p.tc_size,
          .blending = p.blending,
          .letter_id = p.letter_id,
          .font_size = p.font_size
        ));
    }
  };

#if defined(loco_circle)
    struct circle_properties_t {
      camera_impl_t* camera = &gloco->orthographic_camera;
      fan::vec3 position = fan::vec3(0, 0, 0);
      f32_t radius = 0.1f;
      fan::color color = fan::color(1, 1, 1, 1);
      bool blending = false;
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
            .blending = p.blending
          ));
      }
    };
#endif

    struct grid_properties_t {
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
            .position = p.position,
            .size = p.size,
            .grid_size = p.grid_size,
            .rotation_point = p.rotation_point,
            .color = p.color,
            .angle = p.angle
          ));
      }
    };


#if defined(loco_vfi)

    // REQUIRES to be allocated by new since lambda captures this
    // also container that it's stored in, must not change pointers
    template <typename T>
    struct vfi_root_custom_t {
      void set_root(const loco_t::vfi_t::properties_t& p) {
        fan::graphics::vfi_t::properties_t in = p;
        in.shape_type = loco_t::vfi_t::shape_t::rectangle;
        in.shape.rectangle->viewport = gloco->orthographic_camera.viewport;
        in.shape.rectangle->camera = gloco->orthographic_camera.camera;
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
          if (d.button != fan::mouse_left) {
            return 0;
          }
          if (d.button_state != fan::mouse_state::press) {
            this->move = false;
            d.flag->ignore_move_focus_check = false;
            return 0;
          }
          if (d.mouse_stage != loco_t::vfi_t::mouse_stage_e::inside) {
            return 0;
          }

          if (move_and_resize_auto) {
            d.flag->ignore_move_focus_check = true;
            this->move = true;
            this->click_offset = get_position() - d.position;
            gloco->vfi.set_focus_keyboard(d.vfi->focus.mouse);
          }
          return user_cb(d);
          };
        in.mouse_move_cb = [this, user_cb = p.mouse_move_cb](const auto& d) -> int {
          if (move_and_resize_auto) {
            if (this->resize && this->move) {
              fan::vec2 new_size = (d.position - get_position());
              static constexpr fan::vec2 min_size(10, 10);
              new_size.clamp(min_size);
              this->set_size(new_size.x);
              return user_cb(d);
            }
            else if (this->move) {
              fan::vec3 p = get_position();
              this->set_position(fan::vec3(d.position + click_offset, p.z));
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
      void set_position(const fan::vec3& position) {
        fan::vec2 root_pos = vfi_root.get_position();
        fan::vec2 offset = position - root_pos;
        vfi_root.set_position(fan::vec3(root_pos + offset, position.z));
        for (auto& child : children) {
          child.set_position(fan::vec3(fan::vec2(child.get_position()) + offset, position.z));
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
      fan::vec2 click_offset = 0;
      bool move = false;
      bool resize = false;

      bool move_and_resize_auto = true;

      loco_t::shape_t vfi_root;
      struct child_data_t : loco_t::shape_t, T {

      };
      std::vector<child_data_t> children;
    };

    using vfi_root_t = vfi_root_custom_t<__empty_struct>;


    template <typename T>
    struct vfi_multiroot_custom_t {
      void push_root(const loco_t::vfi_t::properties_t& p) {
        loco_t::vfi_t::properties_t in = p;
        in.shape_type = loco_t::vfi_t::shape_t::rectangle;
        in.shape.rectangle->viewport = gloco->orthographic_camera.viewport;
        in.shape.rectangle->camera = gloco->orthographic_camera.camera;
        in.keyboard_cb = [this, user_cb = p.keyboard_cb](const auto& d) -> int {
          if (d.key == fan::key_c &&
            (d.keyboard_state == fan::keyboard_state::press ||
              d.keyboard_state == fan::keyboard_state::repeat)) {
            this->resize = true;
            return user_cb(d);
          }
          this->resize = false;
          return user_cb(d);
          };
        in.mouse_button_cb = [this, root_reference = vfi_root.empty() ? 0 : vfi_root.size() - 1, user_cb = p.mouse_button_cb](const auto& d) -> int {
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
          if (d.mouse_stage != loco_t::vfi_t::mouse_stage_e::inside) {
            return user_cb(d);
          }

          if (move_and_resize_auto) {
            this->click_offset = get_position(root_reference) - d.position;
            gloco->vfi.set_focus_keyboard(d.vfi->focus.mouse);
          }
          return user_cb(d);
          };
        in.mouse_move_cb = [this, root_reference = vfi_root.empty() ? 0 : vfi_root.size() - 1, user_cb = p.mouse_move_cb](const auto& d) -> int {
          if (move_and_resize_auto) {
            if (this->resize && this->move) {
              return user_cb(d);
            }
            else if (this->move) {
              fan::vec3 p = get_position(root_reference);
              this->set_position(root_reference, fan::vec3(d.position + click_offset, p.z));
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

#if defined(loco_imgui)
namespace ImGui {
  IMGUI_API void Image(loco_t::image_t& img, const ImVec2& size, const ImVec2& uv0 = ImVec2(0, 0), const ImVec2& uv1 = ImVec2(1, 1), const ImVec4& tint_col = ImVec4(1, 1, 1, 1), const ImVec4& border_col = ImVec4(0, 0, 0, 0));
  IMGUI_API bool ImageButton(loco_t::image_t& img, const ImVec2& size, const ImVec2& uv0 = ImVec2(0, 0), const ImVec2& uv1 = ImVec2(1, 1), int frame_padding = -1, const ImVec4& bg_col = ImVec4(0, 0, 0, 0), const ImVec4& tint_col = ImVec4(1, 1, 1, 1));
}
#endif

void init_imgui();

void process_render_data_queue(shaper_t& shaper, fan::opengl::context_t& context);

void shape_keypack_traverse(shaper_t::KeyTraverse_t& KeyTraverse, fan::opengl::context_t& context);


#if defined(loco_json)

#include <fan/io/json_impl.h>

namespace fan {
  using namespace nlohmann;
}

namespace nlohmann {

  //template <> struct adl_serializer<fan::vec2> { static void to_json(json& j, const fan::vec2& v) { j = json{v.x, v.y}; } };
  //template <> struct adl_serializer<fan::vec3> { static void to_json(json& j, const fan::vec3& v) { j = json{v.x, v.y, v.z}; } };
  //template <> struct adl_serializer<fan::vec4> { static void to_json(json& j, const fan::vec4& v) { j = json{v.x, v.y, v.z, v.w}; } };
  //template <> struct adl_serializer<fan::color> { static void to_json(json& j, const fan::color& c) { j = json{c.r, c.g, c.b, c.a}; } };

  template <typename T>
  struct nlohmann::adl_serializer<fan::vec2_wrap_t<T>> {
      static void to_json(nlohmann::json& j, const fan::vec2_wrap_t<T>& v) {
          j = nlohmann::json{ v.x, v.y };
      }
      static void from_json(const nlohmann::json& j, fan::vec2_wrap_t<T>& v) {
          v.x = j[0].get<T>();
          v.y = j[1].get<T>();
      }
  };

  template <typename T>
  struct nlohmann::adl_serializer<fan::vec3_wrap_t<T>> {
      static void to_json(nlohmann::json& j, const fan::vec3_wrap_t<T>& v) {
          j = nlohmann::json{ v.x, v.y, v.z };
      }
      static void from_json(const nlohmann::json& j, fan::vec3_wrap_t<T>& v) {
          v.x = j[0].get<T>();
          v.y = j[1].get<T>();
          v.z = j[2].get<T>();
      }
  };

  template <typename T>
  struct nlohmann::adl_serializer<fan::vec4_wrap_t<T>> {
      static void to_json(nlohmann::json& j, const fan::vec4_wrap_t<T>& v) {
          j = nlohmann::json{ v.x, v.y, v.z, v.w };
      }
      static void from_json(const nlohmann::json& j, fan::vec4_wrap_t<T>& v) {
          v.x = j[0].get<T>();
          v.y = j[1].get<T>();
          v.z = j[2].get<T>();
          v.w = j[3].get<T>();
      }
  };

  template <> struct adl_serializer<fan::color> {
    static void to_json(json& j, const fan::color& c) {
      j = json{ c.r, c.g, c.b, c.a };
    }
    static void from_json(const json& j, fan::color& c) {
      c.r = j[0];
      c.g = j[1];
      c.b = j[2];
      c.a = j[3];
    }
  };

}

namespace fan {
  namespace graphics {
    bool shape_to_json(loco_t::shape_t& shape, fan::json* json);

    bool json_to_shape(const fan::json& in, loco_t::shape_t* shape);

    bool shape_serialize(loco_t::shape_t& shape, fan::json* out);
  }
}

#endif

namespace fan {
  namespace graphics {
    bool shape_to_bin(loco_t::shape_t& shape, std::string* str);

    bool bin_to_shape(const std::string& str, loco_t::shape_t* shape, uint64_t& offset);

    bool shape_serialize(loco_t::shape_t& shape, std::string* out);

    struct shape_deserialize_t {
      struct {
        // json::iterator doesnt support union
        // i dont want to use variant either so i accept few extra bytes
        json::const_iterator it;
        uint64_t offset = 0;
      }data;
      bool init = false;

      bool iterate(const fan::json& json, loco_t::shape_t* shape) {
        if (init == false) {
          data.it = json.cbegin();
          init = true; 
        }
        if (data.it == json.cend()) {
          return 0;
        }

        json_to_shape(*data.it, shape);
        ++data.it;
        return 1;
      }

      bool iterate(const std::string& str, loco_t::shape_t* shape) {
        if (str.empty()) {
          return 0;
        }
        else if (data.offset >= str.size()) {
          return 0;
        }
        bin_to_shape(str, shape, data.offset);
        return 1;
      }
    };
  }
}

#include <fan/graphics/collider.h>