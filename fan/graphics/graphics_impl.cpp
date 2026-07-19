module;

#if defined (FAN_WINDOW)

#include <fan/utility.h>

#include <fan/event/types.h>
#include <coroutine>

#endif

module fan.graphics;

#if defined (FAN_WINDOW)

import fan.print.error;
import fan.print;

#if defined(FAN_JSON)
  import fan.types.json;
#endif
#if defined(FAN_GUI)
  import fan.graphics.gui.base;
#endif

import fan.types.compile_time_string;
import fan.graphics.algorithm.raycast_grid;
import fan.io.file;
import fan.graphics.vulkan.core;

#define POSITION2_WINDOW_CENTER fan::vec2(fan::graphics::ctx().window->get_size() / 2)
#define POSITION3_WINDOW_CENTER fan::vec3(POSITION2_WINDOW_CENTER, 0)

namespace fan::graphics {
  fan::graphics::render_view_t add_render_view() {
    fan::graphics::render_view_t render_view;
    render_view.create();
    return render_view;
  }

  fan::graphics::render_view_t add_render_view(const fan::vec2& ortho_x, const fan::vec2& ortho_y, const fan::vec2& viewport_position, const fan::vec2& viewport_size) {
    fan::graphics::render_view_t render_view;
    render_view.create();
    render_view.set(ortho_x, ortho_y, viewport_position, viewport_size, fan::graphics::get_window().get_size());
    return render_view;
  }

#if defined(FAN_2D)

  fan::graphics::context_image_t image_get(fan::graphics::image_nr_t nr) {
    fan::graphics::context_image_t img;
    img.vk = *(fan::vulkan::context_t::image_t*)fan::graphics::ctx()->image_get(fan::graphics::ctx(), nr);
    return img;
  }

  std::vector<std::uint8_t> image_get_pixel_data(fan::graphics::image_nr_t nr, int image_format, fan::vec2 uvp, fan::vec2 uvs) {
    fan::throw_error_impl("");
    return {};
  }

  fan::graphics::shader_nr_t shader_get_nr(std::uint16_t shape_type) {
    return fan::graphics::get_shapes().shaper.GetShader(shape_type);
  }

  fan::graphics::shader_list_t::nd_t& shader_get_data(std::uint16_t shape_type) {
    return (*fan::graphics::ctx().shader_list)[shader_get_nr(shape_type)];
  }

  bool shader_update_fragment(std::uint16_t shape_type, const std::string_view fragment_file_path, const std::string& fragment) {
    auto shader_nr = shader_get_nr(shape_type);
    auto shader_data = shader_get_data(shape_type);
    shader_set_vertex(shader_nr, shader_data.path_vertex, shader_data.svertex);
    shader_set_fragment(shader_nr, fragment_file_path, fragment);
    return shader_compile(shader_nr);
  }

  light_t::light_t(light_properties_t p) {
    *(fan::graphics::shapes::shape_t*)this = fan::graphics::shapes::shape_t(
      fan_init_struct(
        typename fan::graphics::shapes::light_t::properties_t,
        .camera = p.render_view->camera,
        .viewport = p.render_view->viewport,
        .position = p.position,
        .parallax_factor = p.parallax_factor,
        .size = p.size,
        .rotation_point = p.rotation_point,
        .color = p.color,
        .flags = p.flags,
        .angle = p.angle
      ), p.enable_culling
    );
  }

  light_t::light_t(const fan::vec3& position, const fan::vec2& size, const fan::color& color, render_view_t* render_view)
    : light_t(light_properties_t {
      .render_view = render_view,
      .position = position,
      .size = size,
      .color = color
    }) {}

  line_t::line_t(line_properties_t p) {
    *(fan::graphics::shapes::shape_t*)this = fan::graphics::shapes::shape_t(
      fan_init_struct(
        typename fan::graphics::shapes::line_t::properties_t,
        .camera = p.render_view->camera,
        .viewport = p.render_view->viewport,
        .src = p.src,
        .dst = p.dst,
        .color = p.color,
        .thickness = p.thickness,
        .blending = p.blending,
        .draw_mode = p.draw_mode
      ), p.enable_culling
    );
  }

  line_t::line_t(const fan::vec3& src, const fan::vec3& dst, const fan::color& color, f32_t thickness, render_view_t* render_view)
    : line_t(line_properties_t {
      .render_view = render_view,
      .src = src,
      .dst = dst,
      .color = color,
      .thickness = thickness
    }) {}

  rectangle_t::rectangle_t(rectangle_properties_t p) {
    *(fan::graphics::shapes::shape_t*)this = fan::graphics::shapes::shape_t(
      fan_init_struct(
        typename fan::graphics::shapes::rectangle_t::properties_t,
        .camera = p.render_view->camera,
        .viewport = p.render_view->viewport,
        .position = p.position,
        .size = p.size,
        .color = p.color,
        .outline_color = p.outline_color,
        .angle = p.angle,
        .rotation_point = p.rotation_point,
        .blending = p.blending
      ), p.enable_culling
    );
  }

  rectangle_t::rectangle_t(const fan::vec3& position, const fan::vec2& size, const fan::color& color, render_view_t* render_view)
    : rectangle_t(rectangle_properties_t {
      .render_view = render_view,
      .position = position,
      .size = size,
      .color = color
    }) {}

  sprite_t::sprite_t(sprite_properties_t p) {
    *(fan::graphics::shapes::shape_t*)this = fan::graphics::shapes::shape_t(
      fan_init_struct(
        typename fan::graphics::shapes::sprite_t::properties_t,
        .camera = p.render_view->camera,
        .viewport = p.render_view->viewport,
        .parallax_factor = p.parallax_factor,
        .position = p.position,
        .size = p.size,
        .angle = p.angle,
        .image = p.image,
        .tc_position = p.tc_position,
        .tc_size = p.tc_size,
        .images = p.images,
        .color = p.color,
        .rotation_point = p.rotation_point,
        .blending = p.blending,
        .flags = p.flags,
        .texture_pack_unique_id = p.texture_pack_unique_id
      ), p.enable_culling
    );
  }

  sprite_t::sprite_t(
    const fan::graphics::image_t& image,
    const fan::vec3& position, 
    const fan::vec2& size) : sprite_t(position, size, image) {}

  sprite_t::sprite_t(const fan::vec3& position, const fan::vec2& size, const fan::graphics::image_t& image, render_view_t* render_view)
    : sprite_t(sprite_properties_t {
      .render_view = render_view,
      .position = position,
      .size = size,
      .image = image
    }) {}

  sprite_t::sprite_t(const fan::vec3& position, const fan::vec2& size, fan::vec3 angle, const fan::graphics::image_t& image, render_view_t* render_view)
    : sprite_t(sprite_properties_t {
      .render_view = render_view,
      .position = position,
      .size = size,
      .angle = angle,
      .image = image
    }) {}


  sprite_t::sprite_t(const fan::vec3& position, const fan::vec2& size, const fan::color& color)
  : sprite_t(sprite_properties_t {
    .position = position,
    .size = size,
    .image = fan::graphics::image_load(std::span<const fan::color>(&color, 1), fan::vec2ui(1, 1))
  }) {}

  sprite_t::sprite_t(const fan::vec3& position, const fan::vec2& size, std::initializer_list<fan::color> colors, render_view_t* render_view)
  : sprite_t(sprite_properties_t {
    .render_view = render_view,
    .position = position,
    .size = size,
    .image = fan::graphics::image_load(std::span<const fan::color>(colors.begin(), colors.size()), fan::vec2ui(colors.size(), 1))
  }) {}
sprite_t::sprite_t(const fan::vec3& position, const fan::vec2& size, const std::vector<std::uint8_t>& data, const fan::vec2ui& tex_size, render_view_t* render_view)
  : sprite_t(sprite_properties_t {
    .render_view = render_view,
    .position = position,
    .size = size,
    .image = [&] {
      fan::image::info_t info;
      info.data = const_cast<void*>((const void*)data.data());
      info.size = tex_size;
      info.channels = 4;
      return fan::graphics::image_load(info, image_presets::pixel_art());
    }()
  }) {}

sprite_t::sprite_t(const fan::vec3& position, const fan::vec2& size, const fan::image::info_t& info, const fan::graphics::image_load_properties_t& p, render_view_t* render_view)
  : sprite_t(sprite_properties_t {
    .render_view = render_view,
    .position = position,
    .size = size,
    .image = fan::graphics::image_load(info, p)
  }) {}

  unlit_sprite_t::unlit_sprite_t(unlit_sprite_properties_t p) {
    *(fan::graphics::shapes::shape_t*)this = fan::graphics::shapes::shape_t(
      fan_init_struct(
        typename fan::graphics::shapes::unlit_sprite_t::properties_t,
        .camera = p.render_view->camera,
        .viewport = p.render_view->viewport,
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
      ), p.enable_culling
    );
  }

  unlit_sprite_t::unlit_sprite_t(const fan::vec3& position, const fan::vec2& size, const fan::graphics::image_t& image, render_view_t* render_view)
    : unlit_sprite_t(unlit_sprite_properties_t {
      .render_view = render_view,
      .position = position,
      .size = size,
      .image = image
    }) {}

  unlit_sprite_t::unlit_sprite_t(const fan::vec3& position, const fan::vec2& size, std::initializer_list<fan::color> colors, render_view_t* render_view)
    : unlit_sprite_t(unlit_sprite_properties_t {
        .render_view = render_view,
        .position = position,
        .size = size,
        .image = fan::graphics::image_load(std::span<const fan::color>(colors.begin(), colors.size()), fan::vec2ui(colors.size(), 1))
      }) {}
  unlit_sprite_t::unlit_sprite_t(const fan::vec3& position, const fan::vec2& size, const std::vector<std::uint8_t>& data, const fan::vec2ui& tex_size, render_view_t* render_view)
  : unlit_sprite_t(unlit_sprite_properties_t {
    .render_view = render_view,
    .position = position,
    .size = size,
    .image = [&] {
      fan::image::info_t info;
      info.data = const_cast<void*>((const void*)data.data());
      info.size = tex_size;
      info.channels = 4;
      return fan::graphics::image_load(info, image_presets::pixel_art());
    }()
  }) {}
  unlit_sprite_t::unlit_sprite_t(const fan::vec3& position, const fan::vec2& size, const fan::image::info_t& info, const fan::graphics::image_load_properties_t& p, render_view_t* render_view)
    : unlit_sprite_t(unlit_sprite_properties_t {
      .render_view = render_view,
      .position = position,
      .size = size,
      .image = fan::graphics::image_load(info, p)
    }) {}

  circle_t::circle_t(circle_properties_t p) {
    *(fan::graphics::shapes::shape_t*)this = fan::graphics::shapes::shape_t(
      fan_init_struct(
        typename fan::graphics::shapes::circle_t::properties_t,
        .camera = p.render_view->camera,
        .viewport = p.render_view->viewport,
        .position = p.position,
        .radius = p.radius,
        .color = p.color,
        .outline_color = p.outline_color,
        .outline_width = p.outline_width,
        .angle = p.angle,
        .blending = p.blending,
        .flags = p.flags
      ), p.enable_culling
    );
  }

  circle_t::circle_t(const fan::vec3& position, f32_t radius, const fan::color& color, render_view_t* render_view)
    : circle_t(circle_properties_t {
      .render_view = render_view,
      .position = position,
      .radius = radius,
      .color = color
    }) {}

  capsule_t::capsule_t(fan::vec3 position, fan::vec2 center0, fan::vec2 center1, f32_t radius, fan::color color)
    : capsule_t(capsule_properties_t{
      .position = position,
      .center0 = center0,
      .center1 = center1,
      .radius = radius,
      .color = color
    }) {}

  capsule_t::capsule_t(capsule_properties_t p) {
    *(fan::graphics::shapes::shape_t*)this = fan::graphics::shapes::shape_t(
      fan_init_struct(
        typename fan::graphics::shapes::capsule_t::properties_t,
        .camera = p.render_view->camera,
        .viewport = p.render_view->viewport,
        .position = p.position,
        .center0 = p.center0,
        .center1 = p.center1,
        .radius = p.radius,
        .angle = p.angle,
        .color = p.color,
        .outline_color = p.outline_color,
        .blending = p.blending,
        .flags = p.flags
      ), p.enable_culling
    );
  }

  polygon_t::polygon_t(polygon_properties_t p) {
    *(fan::graphics::shapes::shape_t*)this = fan::graphics::shapes::shape_t(
      fan_init_struct(
        typename fan::graphics::shapes::polygon_t::properties_t,
        .camera = p.render_view->camera,
        .viewport = p.render_view->viewport,
        .position = p.position,
        .vertices = p.vertices,
        .angle = p.angle,
        .rotation_point = p.rotation_point,
        .blending = p.blending,
        .draw_mode = p.draw_mode,
        .vertex_count = p.vertex_count
      ), p.enable_culling);
  }

  grid_t::grid_t(grid_properties_t p) {
    *(fan::graphics::shapes::shape_t*)this = fan::graphics::shapes::shape_t(
      fan_init_struct(
        typename fan::graphics::shapes::grid_t::properties_t,
        .camera = p.render_view->camera,
        .viewport = p.render_view->viewport,
        .position = p.position,
        .size = p.size,
        .grid_size = p.grid_size,
        .rotation_point = p.rotation_point,
        .color = p.color,
        .angle = p.angle
      ), p.enable_culling);
  }

  universal_image_renderer_t::universal_image_renderer_t(const universal_image_renderer_properties_t& p) {
    *(fan::graphics::shapes::shape_t*)this = fan::graphics::shapes::shape_t(
      fan_init_struct(
        typename fan::graphics::shapes::universal_image_renderer_t::properties_t,
        .camera = p.render_view->camera,
        .viewport = p.render_view->viewport,
        .position = p.position,
        .size = p.size,
        .tc_position = p.tc_position,
        .tc_size = p.tc_size,
        .blending = p.blending,
        .images = p.images,
        .draw_mode = p.draw_mode,
        ), p.enable_culling
    );
  }

  gradient_t::gradient_t(const gradient_properties_t& p) {
    *(fan::graphics::shapes::shape_t*)this = fan::graphics::shapes::shape_t(
      fan_init_struct(
        typename fan::graphics::shapes::gradient_t::properties_t,
        .camera = p.render_view->camera,
        .viewport = p.render_view->viewport,
        .position = p.position,
        .size = p.size,
        .color = p.color,
        .blending = p.blending,
        .angle = p.angle,
        .rotation_point = p.rotation_point
      ), p.enable_culling
    );
  }

  gradient_t::gradient_t(
    const fan::color& top, 
    const fan::color& bottom,
    const fan::vec3& position,
    const fan::vec2& size) 
      : gradient_t(gradient_properties_t{
          .position = position,
          .size = size,
          .color{top, top, bottom, bottom},
        }){}

  gradient_t::gradient_t(fan::vec3 position, fan::vec2 size, std::array<fan::color, 4> color)
    : gradient_t(gradient_properties_t{
        .position = position,
        .size = size,
        .color = color,
      }){}

  shader_shape_t::shader_shape_t(const shader_shape_properties_t& p) {
    *(fan::graphics::shapes::shape_t*)this = fan::graphics::shapes::shape_t(
      fan_init_struct(
        typename fan::graphics::shapes::shader_shape_t::properties_t,
        .camera = p.render_view->camera,
        .viewport = p.render_view->viewport,
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
      ), p.enable_culling
    );
  }

  shader_shape_t::shader_shape_t(
    const fan::str_view_t vertex_shader,
    const fan::str_view_t fragment_shader,
    const fan::vec3& position,
    const fan::vec2& size
  ) : shader_shape_t(shader_shape_properties_t{
      .position = position,
      .size = size,
      .shader = fan::graphics::shader_create("", vertex_shader, "", fragment_shader)
    }) {}
  shader_shape_t::shader_shape_t(
    const fan::str_view_t fragment_shader, 
    const fan::vec3& position,
    const fan::vec2& size
  ) : shader_shape_t(shader_shape_properties_t{
    .position = position,
    .size = size,
    .shader = fan::graphics::get_sprite_shader("", fragment_shader)
  }) {}
  shader_shape_t::shader_shape_t(
    fan::graphics::shader_t shader, 
    const fan::vec3& position,
    const fan::vec2& size
  ) : shader_shape_t(shader_shape_properties_t{
    .position = position,
    .size = size,
    .shader = shader
  }) {}

  shadow_t::shadow_t(shadow_properties_t p) {
    *(fan::graphics::shapes::shape_t*)this = fan::graphics::shapes::shape_t(
      fan_init_struct(
        typename fan::graphics::shapes::shadow_t::properties_t,
        .camera = p.render_view->camera,
        .viewport = p.render_view->viewport,
        .position = p.position,
        .shape = p.shape,
        .size = p.size,
        .rotation_point = p.rotation_point,
        .color = p.color,
        .flags = p.flags,
        .angle = p.angle,
        .light_position = p.light_position,
        .light_radius = p.light_radius
      ), p.enable_culling);
  }

  fan::graphics::shapes::shape_t& add_shape_to_immediate_draw(fan::graphics::shapes::shape_t&& s) {
    fan::graphics::get_shapes().immediate_render_list->emplace_back(std::move(s));
    return fan::graphics::get_shapes().immediate_render_list->back();
  }

  std::uint32_t add_shape_to_static_draw(fan::graphics::shapes::shape_t&& s) {
    auto ret = s.NRI;
    (*fan::graphics::get_shapes().static_render_list)[ret] = std::move(s);
    return ret;
  }

  void remove_static_shape_draw(const fan::graphics::shapes::shape_t& s) {
    fan::graphics::get_shapes().static_render_list->erase(s.NRI);
  }

  template<typename T, typename ShapeT>
  static fan::graphics::shapes::shape_t& add_or_update_immediate_shape(const T& props, int shape_type, const auto& update_fn) {
    auto& cache = fan::graphics::get_shapes().immediate_shape_caches[shape_type];
    int idx = cache.used_this_frame++;
    if (idx < (int)cache.shapes.size()) [[likely]] {
      auto& shape = cache.shapes[idx];
      update_fn(shape, props);
      if (!shape.is_visible()) { shape.set_visible(true); }
      return shape;
    }
    cache.shapes.emplace_back(ShapeT(props));
    return cache.shapes.back();
  }

  template<typename ShapeT>
  static fan::graphics::shapes::shape_t& add_or_update_immediate_shape(int shape_type, const auto& update_fn, const auto& create_props_fn) {
    auto& cache = fan::graphics::get_shapes().immediate_shape_caches[shape_type];
    int idx = cache.used_this_frame++;
    if (idx < (int)cache.shapes.size()) [[likely]] {
      auto& shape = cache.shapes[idx];
      update_fn(shape);
      if (!shape.is_visible()) { shape.set_visible(true); }
      return shape;
    }
    cache.shapes.emplace_back(ShapeT(create_props_fn()));
    return cache.shapes.back();
  }

  fan::graphics::shapes::shape_t& rectangle(const rectangle_properties_t& props) {
    return add_or_update_immediate_shape<rectangle_properties_t, rectangle_t>(props, fan::graphics::shape_type_t::rectangle, [](auto& shape, const auto& p) {
      shape.set_position(p.position);
      shape.set_size(p.size);
      shape.set_color(p.color);
      shape.set_angle(p.angle);
    });
  }

  fan::graphics::shapes::shape_t& rectangle(const fan::vec3& position, const fan::vec2& size, const fan::color& color, render_view_t* render_view) {
    return add_or_update_immediate_shape<rectangle_t>(
      fan::graphics::shape_type_t::rectangle,
      [&](auto& shape) {
        shape.set_position(position);
        shape.set_size(size);
        shape.set_color(color);
      },
      [&]() {
        return rectangle_properties_t {
          .render_view = render_view,
          .position = position,
          .size = size,
          .color = color
        };
      }
    );
  }

  fan::graphics::shapes::shape_t& sprite(const sprite_properties_t& props) {
    return add_or_update_immediate_shape<sprite_properties_t, sprite_t>(props, fan::graphics::shape_type_t::sprite, [](auto& shape, const auto& p) {
      shape.set_position(p.position);
      shape.set_size(p.size);
      shape.set_color(p.color);
      shape.set_angle(p.angle);
      shape.set_parallax_factor(p.parallax_factor);
      shape.set_flags(p.flags);
      if (p.image.valid()) { shape.set_image(p.image); }
    });
  }

  fan::graphics::shapes::shape_t& sprite(const fan::vec3& position, const fan::vec2& size, const fan::color& single_color) {
    return add_or_update_immediate_shape<sprite_t>(
      fan::graphics::shape_type_t::sprite,
      [&](auto& shape) {
        shape.set_position(position);
        shape.set_size(size);
        shape.set_color(single_color);
      },
      [&]() {
        return sprite_properties_t {
          .position = position, .size = size, .color = single_color,
          .image = fan::graphics::image_load(std::span<const fan::color>(&single_color, 1), fan::vec2ui(1, 1))
        };
      }
    );
  }
  fan::graphics::shapes::shape_t& sprite(const fan::vec3& position, const fan::vec2& size, std::initializer_list<fan::color> colors, render_view_t* render_view) {
    return add_or_update_immediate_shape<sprite_t>(
      fan::graphics::shape_type_t::sprite,
      [&](auto& shape) {
        shape.set_position(position);
        shape.set_size(size);
        if (colors.size()) { shape.set_color(colors.begin()[0]); }
      },
      [&]() {
        return sprite_properties_t {
          .render_view = render_view, .position = position, .size = size,
          .image = fan::graphics::image_load(std::span<const fan::color>(colors.begin(), colors.size()), fan::vec2ui((int)colors.size(), 1))
        };
      }
    );
  }
  fan::graphics::shapes::shape_t& sprite(const fan::vec3& position, const fan::vec2& size, const std::vector<std::uint8_t>& data, const fan::vec2ui& tex_size, render_view_t* render_view) {
    return add_shape_to_immediate_draw(sprite_t(position, size, data, tex_size, render_view));
  }
  fan::graphics::shapes::shape_t& sprite(const fan::vec3& position, const fan::vec2& size, const fan::image::info_t& info, const fan::graphics::image_load_properties_t& p, render_view_t* render_view) {
    return add_shape_to_immediate_draw(sprite_t(position, size, info, p, render_view));
  }
  fan::graphics::shapes::shape_t& sprite(const fan::vec3& position, const fan::vec2& size, const fan::graphics::image_t& image, render_view_t* render_view) {
    return add_or_update_immediate_shape<sprite_t>(
      fan::graphics::shape_type_t::sprite,
      [&](auto& shape) {
        shape.set_position(position);
        shape.set_size(size);
        if (image.valid()) { shape.set_image(image); }
      },
      [&]() {
        return sprite_properties_t {
          .render_view = render_view, .position = position, .size = size, .image = image
        };
      }
    );
  }

  fan::graphics::shapes::shape_t& unlit_sprite(const unlit_sprite_properties_t& props) {
    return add_or_update_immediate_shape<unlit_sprite_properties_t, unlit_sprite_t>(props, fan::graphics::shape_type_t::unlit_sprite, [](auto& shape, const auto& p) {
      shape.set_position(p.position);
      shape.set_size(p.size);
      shape.set_color(p.color);
      shape.set_angle(p.angle);
      if (p.image.valid()) { shape.set_image(p.image); }
    });
  }

  fan::graphics::shapes::shape_t& unlit_sprite(const fan::vec3& position, const fan::vec2& size, const fan::color& single_color) {
    return add_or_update_immediate_shape<unlit_sprite_t>(
      fan::graphics::shape_type_t::unlit_sprite,
      [&](auto& shape) {
        shape.set_position(position);
        shape.set_size(size);
        shape.set_color(single_color);
      },
      [&]() {
        return unlit_sprite_properties_t {
          .position = position, .size = size, .color = single_color,
          .image = fan::graphics::image_load(std::span<const fan::color>(&single_color, 1), fan::vec2ui(1, 1))
        };
      }
    );
  }
  fan::graphics::shapes::shape_t& unlit_sprite(const fan::vec3& position, const fan::vec2& size, std::initializer_list<fan::color> colors, render_view_t* render_view) {
    return add_or_update_immediate_shape<unlit_sprite_t>(
      fan::graphics::shape_type_t::unlit_sprite,
      [&](auto& shape) {
        shape.set_position(position);
        shape.set_size(size);
        if (colors.size()) { shape.set_color(colors.begin()[0]); }
      },
      [&]() {
        return unlit_sprite_properties_t {
          .render_view = render_view, .position = position, .size = size,
          .image = fan::graphics::image_load(std::span<const fan::color>(colors.begin(), colors.size()), fan::vec2ui((int)colors.size(), 1))
        };
      }
    );
  }
  fan::graphics::shapes::shape_t& unlit_sprite(const fan::vec3& position, const fan::vec2& size, const std::vector<std::uint8_t>& data, const fan::vec2ui& tex_size, render_view_t* render_view) {
    return add_shape_to_immediate_draw(unlit_sprite_t(position, size, data, tex_size, render_view));
  }
  fan::graphics::shapes::shape_t& unlit_sprite(const fan::vec3& position, const fan::vec2& size, const fan::image::info_t& info, const fan::graphics::image_load_properties_t& p, render_view_t* render_view) {
    return add_shape_to_immediate_draw(unlit_sprite_t(position, size, info, p, render_view));
  }
  fan::graphics::shapes::shape_t& unlit_sprite(const fan::vec3& position, const fan::vec2& size, const fan::graphics::image_t& image, render_view_t* render_view) {
    return add_or_update_immediate_shape<unlit_sprite_t>(
      fan::graphics::shape_type_t::unlit_sprite,
      [&](auto& shape) {
        shape.set_position(position);
        shape.set_size(size);
        if (image.valid()) { shape.set_image(image); }
      },
      [&]() {
        return unlit_sprite_properties_t {
          .render_view = render_view, .position = position, .size = size, .image = image
        };
      }
    );
  }

  fan::graphics::shapes::shape_t& line(const line_properties_t& props) {
    return add_or_update_immediate_shape<line_properties_t, line_t>(props, fan::graphics::shape_type_t::line, [](auto& shape, const auto& p) {
      //shape.set_position(p.src);
      shape.set_line(p.src, p.dst);
      shape.set_color(p.color);
    });
  }

  fan::graphics::shapes::shape_t& line(const fan::vec3& src, const fan::vec3& dst, const fan::color& color, f32_t thickness, render_view_t* render_view) {
    return add_or_update_immediate_shape<line_t>(
      fan::graphics::shape_type_t::line,
      [&](auto& shape) {
        //shape.set_position(src);
        shape.set_line(src, dst);
        shape.set_color(color);
      },
      [&]() {
        return line_properties_t {
          .render_view = render_view, .src = src, .dst = dst, .color = color, .thickness = thickness
        };
      }
    );
  }

  fan::graphics::shapes::shape_t& light(const light_properties_t& props) {
    return add_or_update_immediate_shape<light_properties_t, light_t>(props, fan::graphics::shape_type_t::light, [](auto& shape, const auto& p) {
      shape.set_position(p.position);
      shape.set_size(p.size);
      shape.set_color(p.color);
      shape.set_parallax_factor(p.parallax_factor);
      shape.set_flags(p.flags);
      shape.set_angle(p.angle);
    });
  }

  fan::graphics::shapes::shape_t& light(const fan::vec3& position, const fan::vec2& size, const fan::color& color, render_view_t* render_view) {
    return add_or_update_immediate_shape<light_t>(
      fan::graphics::shape_type_t::light,
      [&](auto& shape) {
        shape.set_position(position);
        shape.set_size(size);
        shape.set_color(color);
      },
      [&]() {
        return light_properties_t {
          .render_view = render_view, .position = position, .size = size, .color = color
        };
      }
    );
  }

  fan::graphics::shapes::shape_t& circle(const circle_properties_t& props) {
    return add_or_update_immediate_shape<circle_properties_t, circle_t>(props, fan::graphics::shape_type_t::circle, [](auto& shape, const auto& p) {
      shape.set_position(p.position);
      shape.set_radius(p.radius);
      shape.set_color(p.color);
      shape.set_outline_color(p.outline_color);
      shape.set_angle(p.angle);
      shape.set_flags(p.flags);
    });
  }

  fan::graphics::shapes::shape_t& circle(const fan::vec3& position, f32_t radius, const fan::color& color, render_view_t* render_view) {
    return add_or_update_immediate_shape<circle_t>(
      fan::graphics::shape_type_t::circle,
      [&](auto& shape) {
        shape.set_position(position);
        shape.set_radius(radius);
        shape.set_color(color);
      },
      [&]() {
        return circle_properties_t {
          .render_view = render_view, .position = position, .radius = radius, .color = color
        };
      }
    );
  }

  fan::graphics::shapes::shape_t& capsule(const capsule_properties_t& props) {
    return add_shape_to_immediate_draw(capsule_t(props));
  }

  fan::graphics::shapes::shape_t& polygon(const polygon_properties_t& props) {
    return add_shape_to_immediate_draw(polygon_t(props));
  }

  fan::graphics::shapes::shape_t& grid(const grid_properties_t& props) {
    return add_shape_to_immediate_draw(grid_t(props));
  }

#if defined(FAN_PHYSICS_2D)
  void aabb(const fan::physics::aabb_t& b, f32_t depth, const fan::color& c, f32_t thickness, render_view_t* render_view) {
    fan::graphics::line({.render_view = render_view, .src = {b.min, depth}, .dst = {b.max.x, b.min.y}, .color = c, .thickness = thickness});
    fan::graphics::line({.render_view = render_view, .src = {b.max.x, b.min.y, depth}, .dst = {b.max}, .color = c, .thickness = thickness});
    fan::graphics::line({.render_view = render_view, .src = {b.max, depth}, .dst = {b.min.x, b.max.y}, .color = c, .thickness = thickness});
    fan::graphics::line({.render_view = render_view, .src = {b.min.x, b.max.y, depth}, .dst = {b.min}, .color = c, .thickness = thickness});
  }

  void aabb(const fan::graphics::shapes::shape_t& s, f32_t depth, const fan::color& c, f32_t thickness, render_view_t* render_view) {
    fan::graphics::aabb(s.get_aabb(), depth, c, thickness, render_view);
  }

  void aabb(const fan::vec2& min, const fan::vec2& max, f32_t depth, const fan::color& c, f32_t thickness, render_view_t* render_view) {
    fan::physics::aabb_t aabb_;
    aabb_.min = min;
    aabb_.max = max;
    aabb(aabb_, depth, c, thickness, render_view);
  }

  void aabb(const fan::vec2& min, const fan::vec2& max, f32_t thickness, render_view_t* render_view) {
    aabb(min, max, 0xFFF0, fan::colors::white, thickness, render_view);
  }
#endif

#if defined(FAN_JSON)

  struct json_cache_t {
    std::vector<fan::graphics::shape_t> shapes;
    std::vector<fan::vec2> original_pos;
  };

  auto& get_json_cache() {
    static std::unordered_map<
      fan::ct_string<256>,
      json_cache_t,
      fan::ct_string_hash,
      fan::ct_string_equal
    > json_cache;
    return json_cache;
  }

  static void load_json_shapes(
    std::string_view json_path,
    const std::source_location& callers_path)
  {
    fan::json json_data = fan::graphics::read_json(json_path, callers_path);
    resolve_json_image_paths(json_data, json_path, callers_path);
    bool has_anim = fan::graphics::sprite_sheets_parse(json_path, json_data, callers_path);

    std::vector<fan::graphics::shape_t> shapes;

    if (json_data.contains("shapes")) {
      fan::graphics::shape_deserialize_t it;
      fan::graphics::shape_t s;
      while (it.iterate(json_data["shapes"], &s, callers_path)) {
        shapes.emplace_back(std::move(s));
        if (has_anim) shapes.back(). set_sprite_sheet_start();
      }
    }
    else if (json_data.contains("shape")) {
      shapes.emplace_back(fan::graphics::extract_single_shape(json_data, callers_path));
      if (has_anim) shapes.back(). set_sprite_sheet_start();
    }

    auto& cache = get_json_cache()[json_path];
    cache.shapes = shapes;
    cache.original_pos.resize(shapes.size());

    for (std::size_t i = 0; i < shapes.size(); ++i) {
      cache.original_pos[i] = shapes[i].get_position();
      cache.shapes[i].set_position(fan::vec2(-0xfffff));
    }
  }

  fan::graphics::shape_t shape_from_json(
    std::string_view json_path,
    const std::source_location& callers_path)
  {
    auto& cache_map = get_json_cache();
    if (!cache_map.contains(json_path)) {
      load_json_shapes(json_path, callers_path);
    }

    auto& cache = cache_map[json_path];
    fan::graphics::shape_t s = cache.shapes[0];
    s.set_position(cache.original_pos[0]);
    return s;
  }

  std::vector<fan::graphics::shape_t> shapes_from_json(
    std::string_view json_path,
    const std::source_location& callers_path)
  {
    auto& cache_map = get_json_cache();
    if (!cache_map.contains(json_path)) {
      load_json_shapes(json_path, callers_path);
    }

    auto& cache = cache_map[json_path];
    std::vector<fan::graphics::shape_t> out(cache.shapes.size());

    for (std::size_t i = 0; i < cache.shapes.size(); ++i) {
      fan::graphics::shape_t s = cache.shapes[i];
      s.set_position(cache.original_pos[i]);
      out[i] = std::move(s);
    }

    return out;
  }

  fan::graphics::shape_t shapes_children_from_json(
    std::string_view json_path,
    const std::source_location& callers_path) {
    auto children = shapes_from_json(json_path, callers_path);
    if (children.empty()) return {};
    fan::graphics::shape_t parent = std::move(children[0]);
    if (children.size() > 1) {
      parent.add_children(std::span{children}.subspan(1));
    }
    return parent;
  }

  void resolve_json_image_paths(
    fan::json& out,
    std::string_view json_path,
    const std::source_location& callers_path)
  {
    out.find_and_iterate("image_path", [&json_path, &callers_path](fan::json& value) {
      std::filesystem::path base = fan::io::file::find_relative_path(json_path, callers_path);
      base = std::filesystem::is_directory(base) ? base : base.parent_path();
      value = (base / std::filesystem::path(value.get<std::string>())).generic_string();
    });
  }

  fan::graphics::sprite_t sprite_sheet_from_json(
    const sprite_sheet_config_t config,
    const std::source_location& callers_path)
  {
    auto shape = shape_from_json(config.path, callers_path);
    shape.set_sprite_sheet_loop(shape.get_current_sprite_sheet_id(), config.loop);
    if (config.start) {
      shape.play_sprite_sheet();
    }
    return shape;
  }
#endif

  fan::graphics::shapes::polygon_t::properties_t create_hexagon(f32_t radius, const fan::color& color) {
    fan::graphics::shapes::polygon_t::properties_t pp;
    for (int i = 0; i < 6; ++i) {
      pp.vertices.push_back(fan::graphics::vertex_t {fan::vec3(0, 0, 0), color});
      f32_t angle1 = 2 * fan::math::pi * i / 6;
      pp.vertices.push_back(fan::graphics::vertex_t {fan::vec3(fan::vec2(radius * std::cos(angle1), radius * std::sin(angle1)), 0), color});
      f32_t angle2 = 2 * fan::math::pi * ((i + 1) % 6) / 6;
      pp.vertices.push_back(fan::graphics::vertex_t {fan::vec3(fan::vec2(radius * std::cos(angle2), radius * std::sin(angle2)), 0), color});
    }
    return pp;
  }

  fan::line3 get_highlight_positions(const fan::vec3& op, const fan::vec2& os, int index) {
    fan::line3 p;
    f32_t z = op.z + 1;
    switch (index) {
      case 0: p[0] = fan::vec3(op.xy() - os, z); p[1] = fan::vec3(op.x + os.x, op.y - os.y, z); break;
      case 1: p[0] = fan::vec3(op.x + os.x, op.y - os.y, z); p[1] = fan::vec3(op.xy() + os, z); break;
      case 2: p[0] = fan::vec3(op.xy() + os, z); p[1] = fan::vec3(op.x - os.x, op.y + os.y, z); break;
      case 3: p[0] = fan::vec3(op.x - os.x, op.y + os.y, z); p[1] = fan::vec3(op.xy() - os, z); break;
    }
    return p;
  }

#endif

  interactive_camera_t::operator render_view_t*() {
    return &render_view;
  }

  void interactive_camera_t::reset() {
    ignore_input = false;
    zoom_on_window_resize = true;
    pan_with_middle_mouse = false;
    reset_view();
  }

  void interactive_camera_t::reset_view() {
    set_position(get_initial_position());
    update();
  }

  void interactive_camera_t::update() {
    fan::vec2 s = fan::graphics::ctx()->viewport_get_size(
      fan::graphics::ctx(),
      render_view.viewport
    );
    fan::graphics::ctx()->camera_set_ortho(
      fan::graphics::ctx(),
      render_view.camera,
      fan::vec2(-s.x / 2.f, s.x / 2.f),
      fan::vec2(-s.y / 2.f, s.y / 2.f)
    );
  }

  void interactive_camera_t::create(
    fan::graphics::camera_t camera_nr,
    fan::graphics::viewport_t viewport_nr,
    f32_t new_zoom,
    const fan::vec2& initial_pos
  ) {
    render_view.camera = camera_nr;
    render_view.viewport = viewport_nr;
    set_zoom(new_zoom);
    auto& window = fan::graphics::get_window();
    old_window_size = window.get_size();
    if (initial_pos == -0xFAFA) {
      set_initial_position(viewport_get_size() / 2.f);
    }
    else {
      set_initial_position(initial_pos);
    }

    static auto update_ortho = [this](void* ptr) {
      update();
    };

    update();

    auto it = fan::graphics::ctx().update_callback->NewNodeLast();
    (*fan::graphics::ctx().update_callback)[it] = update_ortho;

    resize_callback_nr = window.add_resize_callback([&](const auto& d) {
      if (old_window_size.x > 0 && old_window_size.y > 0) {
        fan::graphics::viewport_set(render_view.viewport, fan::vec2(0, 0), d.size);
        fan::vec2 ratio = fan::vec2(d.size) / old_window_size;
        f32_t size_ratio = (ratio.y + ratio.x) / 2.0f;
        f32_t zoom_change = get_zoom() * (size_ratio - 1.0f);
        set_zoom(get_zoom() + zoom_change);
      }
      old_window_size = d.size;
    });

    button_cb_nr = window.add_buttons_callback([&](const auto& d) {
      if (ignore_input) {
        return;
      }

    #if defined(FAN_GUI)
      if (fan::graphics::gui::want_io()) {
        return;
      }
    #endif

      bool mouse_inside_viewport = fan::graphics::inside(
        render_view.viewport,
        fan::window::get_mouse_position()
      );
      if (mouse_inside_viewport) {
      #if defined(FAN_GUI)
        auto* context = fan::graphics::gui::get_context();
        auto* hovered_window = context->HoveredWindow;
        if (hovered_window) {
          fan::graphics::gui::set_window_focus(hovered_window->Name);
        }
      #endif
        if (d.button == fan::mouse_scroll_up) {
          set_zoom(get_zoom() * 1.2f);
        }
        else if (d.button == fan::mouse_scroll_down) {
          set_zoom(get_zoom() / 1.2f);
        }
      }
      auto state = d.window->key_state(fan::mouse_middle);
      if (state == (int)fan::mouse_state::press) {
        clicked_inside_viewport = mouse_inside_viewport;
      }
      else if (state == (int)fan::mouse_state::release) {
        clicked_inside_viewport = false;
      }
    });

    mouse_motion_nr = window.add_mouse_motion_callback([&](const auto& d) {
      auto state = d.window->key_state(fan::mouse_middle);
      if (state == (int)fan::mouse_state::press ||
        state == (int)fan::mouse_state::repeat) {
        if (pan_with_middle_mouse && clicked_inside_viewport) {
          fan::vec2 viewport_size = fan::graphics::viewport_get_size(render_view.viewport);
          camera_offset -= (d.motion * viewport_size / (viewport_size * get_zoom())) * (fan::vec2i(1) - lock_axis);
          fan::graphics::camera_set_position(render_view.camera, camera_offset);
        }
      }
    });
  }

  void interactive_camera_t::create(const fan::graphics::render_view_t& render_view, f32_t new_zoom) {
    create(render_view.camera, render_view.viewport, new_zoom);
  }

  void interactive_camera_t::create_default(f32_t zoom) {
    render_view.create_default(zoom);
    create(render_view.camera, render_view.viewport, zoom);
  }

  interactive_camera_t::interactive_camera_t(f32_t zoom) {
    create_default(zoom);
  }

  interactive_camera_t::interactive_camera_t(
    fan::graphics::camera_t camera_nr,
    fan::graphics::viewport_t viewport_nr,
    f32_t new_zoom,
    const fan::vec2& initial_pos
  ) {
    create(camera_nr, viewport_nr, new_zoom, initial_pos);
    initial_position = fan::graphics::viewport_get_size(viewport_nr) / 2.f;
    set_position(initial_position);
  }

  interactive_camera_t::interactive_camera_t(const fan::graphics::render_view_t& render_view, f32_t new_zoom)
    : interactive_camera_t(render_view.camera, render_view.viewport, new_zoom) {
    initial_position = fan::graphics::viewport_get_size(render_view.viewport) / 2.f;
    set_position(initial_position);
  }

  interactive_camera_t::~interactive_camera_t() {
    if (uc_nr.iic() == false) {
      fan::graphics::ctx().update_callback->unlrec(uc_nr);
      uc_nr.sic();
    }
  }

  fan::vec2 interactive_camera_t::get_initial_position() const {
    return initial_position;
  }

  void interactive_camera_t::set_initial_position(const fan::vec2& position) {
    initial_position = position;
  }

  fan::vec2 interactive_camera_t::get_position() const {
    return camera_offset;
  }

  void interactive_camera_t::set_position(const fan::vec2& position) {
    camera_offset = position;
    fan::graphics::camera_set_position(render_view.camera, camera_offset);
    update();
  }

  void interactive_camera_t::set_center(const fan::vec2& center) {
    fan::graphics::camera_set_center(render_view, center);
    camera_offset = camera_get_position(render_view);
    update();
  }
  void set_center(const fan::vec2& center);

  f32_t interactive_camera_t::get_zoom() const {
    return fan::graphics::camera_get_zoom(render_view.camera);
  }

  void interactive_camera_t::set_zoom(f32_t new_zoom) {
    fan::graphics::camera_set_zoom(render_view.camera, new_zoom);
    update();
  }

  fan::vec2 interactive_camera_t::get_size() const {
    return fan::graphics::ctx()->camera_get_size(
      fan::graphics::ctx(),
      render_view.camera);
  }

  fan::vec4 interactive_camera_t::get_ortho() const {
    return fan::graphics::camera_get(render_view.camera).coordinates.v;
  }

  fan::vec2 interactive_camera_t::get_viewport_size() const {
    return fan::graphics::viewport_get_size(render_view.viewport);
  }

  void interactive_camera_t::shake(f32_t intensity, f32_t duration) {
    fx_shake_intensity = intensity;
    fx_shake_duration = duration;
    fx_shake_timer = duration;
  }

  void interactive_camera_t::bump(fan::vec2 direction, f32_t distance, f32_t duration) {
    fx_bump_offset = direction.normalize() * distance;
    fx_bump_duration = duration;
    fx_bump_timer = duration;
  }

  void interactive_camera_t::bump_zoom(f32_t amount, f32_t duration) {
    if (fx_bump_zoom_timer <= 0) {
      fx_bump_zoom_base = get_zoom();
    }
    fx_bump_zoom_amount = amount;
    fx_bump_zoom_duration = duration;
    fx_bump_zoom_timer = duration;
  }

  void interactive_camera_t::flash(f32_t alpha, f32_t duration) {
    fx_flash_alpha = alpha;
    fx_flash_duration = duration;
    fx_flash_timer = duration;
  }

  void interactive_camera_t::flashbang(f32_t duration) {
    flash(1.f, duration);
  }

  f32_t interactive_camera_t::get_flash_alpha() const {
    return fx_flash_timer > 0 ? fx_flash_alpha * (fx_flash_timer / fx_flash_duration) : 0;
  }

  void interactive_camera_t::update_fx(f32_t dt) {
    fan::vec2 total_offset(0);

    if (fx_shake_timer > 0) {
      fx_shake_timer -= dt;
      f32_t t = fx_shake_timer / fx_shake_duration;
      f32_t intensity = fx_shake_intensity * t;
      total_offset += fan::vec2(
        fan::random::value(-intensity, intensity),
        fan::random::value(-intensity, intensity)
      );
    }

    if (fx_bump_timer > 0) {
      fx_bump_timer -= dt;
      f32_t t = fx_bump_timer / fx_bump_duration;
      total_offset += fx_bump_offset * t;
    }

    if (total_offset.length_squared() > 0) {
      fan::vec2 current_pos = fan::graphics::camera_get_position(render_view.camera);
      fan::graphics::camera_set_position(render_view.camera, current_pos + total_offset);
    }

    if (fx_bump_zoom_timer > 0) {
      fx_bump_zoom_timer -= dt;
      f32_t progress = 1 - fx_bump_zoom_timer / fx_bump_zoom_duration;
      f32_t offset = std::sin(progress * fan::math::pi) * fx_bump_zoom_amount;
      fan::graphics::camera_set_zoom(render_view.camera, fx_bump_zoom_base + offset);
    }

    if (fx_flash_timer > 0) {
      fx_flash_timer -= dt;
    }
  }

  world_window_t::world_window_t() : render_view(true), cam(render_view) {}

  void world_window_t::update(const fan::vec2& viewport_pos, const fan::vec2& viewport_size) {
    fan::graphics::viewport_set(
      render_view.viewport,
      viewport_pos,
      viewport_size
    );
    fan::graphics::camera_set_ortho(
      render_view.camera,
      fan::vec2(-viewport_size.x / 2, viewport_size.x / 2),
      fan::vec2(-viewport_size.y / 2, viewport_size.y / 2)
    );
  }

  world_window_t::operator render_view_t*() {
    return &render_view;
  }


#if defined(FAN_2D)

  fan::graphics::shapes::polygon_t::properties_t create_sine_ground(const fan::vec2& position, f32_t amplitude, f32_t frequency, f32_t width, f32_t groundWidth) {
    fan::graphics::shapes::polygon_t::properties_t pp;
    // for triangle strip
    for (f32_t x = 0; x < groundWidth - width; x += width) {
      f32_t y1 = position.y / 2 + amplitude * std::sin(frequency * x);
      f32_t y2 = position.y / 2 + amplitude * std::sin(frequency * (x + width));
      // top
      pp.vertices.push_back({fan::vec2(position.x + x, y1), fan::colors::red});
      // bottom
      pp.vertices.push_back({fan::vec2(position.x + x, position.y), fan::colors::white});
      // next top
      pp.vertices.push_back({fan::vec2(position.x + x + width, y2), fan::colors::red});
      // next bottom
      pp.vertices.push_back({fan::vec2(position.x + x + width, position.y), fan::colors::white});
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

  trail_t::trail_t() {
    update_callback_nr = fan::graphics::ctx().update_callback->NewNodeLast();
    (*fan::graphics::ctx().update_callback)[update_callback_nr] = [this](void* ptr) {
      update();
    };
  }

  trail_t::~trail_t() {
    if (!update_callback_nr) {
      return;
    }
    fan::graphics::ctx().update_callback->unlrec(update_callback_nr);
    update_callback_nr.sic();
  }

  void trail_t::set_point(const fan::vec3& point, f32_t drift_intensity) {
    // uses static for all trails?
    static fan::time::timer timer {(double)300000000ULL, true};
    bool should_reset = trails.empty() || timer;

    if (should_reset) {
      trails.resize(trails.size() + 1);
      trails.back().vertices.clear();
      trails.back().creation_time = fan::time::now();
      trails.back().base_alpha = 0.2f + (drift_intensity * 0.6f);
    }

    bool start_new_trail = false;
    if (!trails.empty() && !trails.back().vertices.empty()) {
      fan::vec3 last_point = trails.back().vertices.back().position;
      f32_t distance = std::sqrt(std::pow(point.x - last_point.x, 2) + std::pow(point.y - last_point.y, 2));
      if (distance > 50.0f) {
        start_new_trail = true;
      }
    }

    if (start_new_trail) {
      trails.resize(trails.size() + 1);
      trails.back().vertices.clear();
      trails.back().creation_time = fan::time::now();
      trails.back().base_alpha = 0.2f + (drift_intensity * 0.6f);
    }

    fan::vec2 direction = fan::vec2(1, 0);
    if (trails.back().vertices.size() >= 2) {
      fan::vec3 last_point = trails.back().vertices[trails.back().vertices.size() - 2].position;
      fan::vec3 diff = point - last_point;
      direction = fan::vec2(diff.x, diff.y);
      f32_t len = std::sqrt(direction.x * direction.x + direction.y * direction.y);
      if (len > 0) {
        direction.x /= len;
        direction.y /= len;
      }
    }

    fan::vec2 perp = fan::vec2(-direction.y, direction.x);
    fan::graphics::vertex_t vertex;
    vertex.color = fan::color(color.r, color.g, color.b, trails.back().base_alpha);

    vertex.position = point + fan::vec3(perp.x * thickness * 0.5f, perp.y * thickness * 0.5f, 0);
    trails.back().vertices.emplace_back(vertex);

    vertex.position = point - fan::vec3(perp.x * thickness * 0.5f, perp.y * thickness * 0.5f, 0);
    trails.back().vertices.emplace_back(vertex);

    trails.back().polygon = fan::graphics::polygon_t {{
      .position = fan::vec3(0, 0, point.z),
      .vertices = trails.back().vertices,
      .draw_mode = fan::graphics::primitive_topology_t::triangle_strip,
    }};

    timer.restart();
  }

  void trail_t::update() {
    std::uint64_t current_time = fan::time::now();

    for (auto& trail : trails) {
      std::uint64_t age = current_time - trail.creation_time;
      f32_t fade_factor = 1.0f;

      if (age > fade_duration) {
        fade_factor = std::max(0.0f, 1.0f - static_cast<f32_t>(age - fade_duration) / static_cast<f32_t>(max_trail_lifetime - fade_duration));
      }
      f32_t current_alpha = trail.base_alpha * fade_factor;

      for (std::size_t i = 0; i < trail.vertices.size(); i += 2) {
        f32_t position_factor = static_cast<f32_t>(i) / static_cast<f32_t>(trail.vertices.size() - 2);
        f32_t vertex_alpha = current_alpha * (0.2f + 0.8f * position_factor);
        trail.vertices[i].color.a = vertex_alpha;
        trail.vertices[i + 1].color.a = vertex_alpha;
      }
      fan::vec3 pos = trail.polygon.get_position();
      trail.polygon = {{
        .position = pos,
        .vertices = trail.vertices,
        .draw_mode = fan::graphics::primitive_topology_t::triangle_strip,
      }};
    }

    trails.erase(
      std::remove_if(trails.begin(), trails.end(), [&](const trail_segment_t& trail) {
        std::uint64_t age = current_time - trail.creation_time;
        return age > max_trail_lifetime;
      }),
      trails.end()
    );
  }

  f32_t get_depth_from_y(const fan::vec2& position, f32_t tile_size_y) {
    return (position.y / tile_size_y) + (0xFAAA - 2) / 2.f;
  }
  f32_t get_player_depth_from_y(const fan::vec2& position, f32_t size_y, f32_t tile_size_y) {    
    return fan::graphics::get_depth_from_y(fan::vec2(position.x, position.y + size_y), tile_size_y);
  }

  tilemap_t::tilemap_t(const fan::vec2& tile_size,
    const fan::color& color,
    const fan::vec2& area,
    const fan::vec3& offset,
    render_view_t* render_view) {
    create(tile_size, color, area, offset, render_view);
  }

  fan::vec2i tilemap_t::get_cell_count() {
    return fan::vec2i(
      shapes.empty() ? 0 : (int)shapes[0].size(),
      (int)shapes.size()
    );
  }

  template <typename T>
  fan::graphics::shapes::shape_t& tilemap_t::operator[](const fan::vec2_wrap_t<T>& v) {
    return shapes[v.y][v.x];
  }

  void tilemap_t::add_wall(const fan::vec2i& cell, fan::pathfind::generator& gen) {
    if (wall_cells.find(cell) == wall_cells.end()) {
      gen.add_collision(cell);
      wall_cells.insert(cell);
      shapes[cell.y][cell.x].set_color(fan::colors::gray * 1.5f);
    }
  }

  void tilemap_t::remove_wall(const fan::vec2i& cell, fan::pathfind::generator& gen) {
    if (wall_cells.find(cell) != wall_cells.end()) {
      gen.remove_collision(cell);
      wall_cells.erase(cell);
      shapes[cell.y][cell.x].set_color(fan::colors::gray);
    }
  }

  void tilemap_t::fill_colors(const fan::color& c) {
    for (int y = 0; y < size.y; ++y)
      for (int x = 0; x < size.x; ++x)
        shapes[y][x].set_color(c);
  }

  void tilemap_t::reset_colors(const fan::color& color) {
    for (int i = 0; i < size.y; i++) {
      for (int j = 0; j < size.x; j++) {
        fan::vec2i cell(j, i);
        if (wall_cells.contains(cell)) continue;
        shapes[i][j].set_color(color);
      }
    }
  }

  void tilemap_t::set_source(const fan::vec2i& cell, const fan::color& color) {
    if (!wall_cells.contains(cell) &&
      cell.x >= 0 && cell.x < (int)shapes[0].size() &&
      cell.y >= 0 && cell.y < (int)shapes.size()) {
      shapes[cell.y][cell.x].set_color(color);
    }
  }

  void tilemap_t::set_destination(const fan::vec2i& cell, const fan::color& color) {
    if (!wall_cells.contains(cell) &&
      cell.x >= 0 && cell.x < (int)shapes[0].size() &&
      cell.y >= 0 && cell.y < (int)shapes.size()) {
      shapes[cell.y][cell.x].set_color(color);
    }
  }

  void tilemap_t::highlight_path(
    const fan::pathfind::coordinate_list& path,
    const fan::color& color)
  {
    for (const auto& p : path) {
      if (!wall_cells.contains(p) &&
        p.x >= 0 && p.x < (int)shapes[0].size() &&
        p.y >= 0 && p.y < (int)shapes.size()) {
        shapes[p.y][p.x].set_color(color);
      }
    }
  }

  fan::pathfind::coordinate_list tilemap_t::find_path(
    const fan::vec2i& src,
    const fan::vec2i& dst,
    fan::pathfind::generator& gen,
    fan::pathfind::heuristic_function heuristic,
    bool diagonal)
  {
    gen.set_heuristic(heuristic);
    gen.set_diagonal_movement(diagonal);
    return gen.find_path(src, dst);
  }

  void tilemap_t::create(
    const fan::vec2& tile_size,
    const fan::color& color,
    const fan::vec2& area,
    const fan::vec3& offset,
    render_view_t* render_view)
  {
    fan::vec2 map_size(
      std::floor(area.x / tile_size.x),
      std::floor(area.y / tile_size.y)
    );

    size = map_size;
    this->tile_size = tile_size;
    positions.resize(map_size.y, std::vector<fan::vec2>(map_size.x));
    shapes.resize(map_size.y, std::vector<fan::graphics::shape_t>(map_size.x));
    for (int i = 0; i < map_size.y; i++) {
      for (int j = 0; j < map_size.x; j++) {
        positions[i][j] = fan::vec2(offset) + tile_size / 2 + fan::vec2(j * tile_size.x, i * tile_size.y);
        static fan::graphics::image_t img = fan::graphics::image_create(fan::colors::white);
        shapes[i][j] = fan::graphics::sprite_t {{
          .render_view = render_view,
          .position = fan::vec3(positions[i][j], offset.z),
          .size = tile_size / 2,
          .color = color,
          .image = img,
        }};
      }
    }
  }

  void tilemap_t::set_tile_color(const fan::vec2i& pos, const fan::color& c) {
    if (!in_bounds(pos)) return;
    shapes[pos.y][pos.x].set_color(c);
  }

  void tilemap_t::set_tile_image(const fan::vec2i& pos, fan::graphics::image_t image) {
    if (!in_bounds(pos)) return;
    shapes[pos.y][pos.x].set_image(image);
  }

  constexpr f32_t tilemap_t::circle_overlap(f32_t r, f32_t i0, f32_t i1) {
    if (i0 <= 0 && i1 >= 0) return r;
    f32_t y = fan::math::min(fan::math::min(std::fabs(i0), std::fabs(i1)) / r, 1.f);
    return std::sqrt(1.f - y * y) * r;
  }

  void tilemap_t::highlight_circle(const fan::graphics::shapes::shape_t& circle,
    const fan::color& highlight_color)
  {
    fan::vec2 wp = circle.get_position();
    f32_t r = circle.get_radius();
    auto gi = fan::cast<sint32_t>(decltype(wp){});

    constexpr auto recurse = []<std::uint32_t d>(const auto& self,
      tilemap_t& tilemap,
      auto& gi,
      fan::vec2 wp,
      f32_t r,
      f32_t er,
      const fan::color& hc) {
      f32_t cell_size = tilemap.tile_size[d];

      if constexpr (d + 1 < wp.size()) {
        gi[d] = (wp[d] - er) / cell_size;
        f32_t sp = (f32_t)gi[d] * cell_size;
        while (true) {
          f32_t rp = sp - wp[d];
          f32_t roff = circle_overlap(r, rp, rp + cell_size);
          self.template operator() < d + 1 > (self, tilemap, gi, wp, r, roff, hc);
          gi[d]++;
          sp += cell_size;
          if (sp > wp[d] + er) break;
        }
      }
      else if constexpr (d < wp.size()) {
        gi[d] = (wp[d] - er) / cell_size;
        sint32_t to = (wp[d] + er) / cell_size;
        for (; gi[d] <= to; gi[d]++) {
          fan::vec2i pos {gi[0], gi[1]};
          tilemap.set_tile_color(pos, hc);
        }
      }
    };

    recurse.template operator() < 0 > (recurse, *this, gi, wp, r, r, highlight_color);
  }

  void tilemap_t::highlight_line(const fan::graphics::shapes::shape_t& line,
    const fan::color& color,
    render_view_t* render_view)
  {
    fan::vec2 src = line.get_src();
    fan::vec2 dst = line.get_dst();

    auto raycast_positions = fan::graphics::algorithm::grid_raycast(
      {src, dst},
      tile_size
    );

    for (auto& pos : raycast_positions) {
      set_tile_color(pos, color);
    }
  }

  void tilemap_t::highlight(const fan::graphics::shapes::shape_t& shape, const fan::color& color) {
    using namespace fan::graphics;

    switch (shape.get_shape_type()) {
    case shapes::shape_type_t::circle:
      highlight_circle(shape, color);
      break;
    case shapes::shape_type_t::line:
      highlight_line(shape, color);
      break;
    default:
      fan::throw_error_impl("method not implemented");
      break;
    }
  }

  fan::graphics::shape_t& tilemap_t::get_tile(const fan::vec2i& pos) {
    return shapes[pos.y][pos.x];
  }

  bool tilemap_t::in_bounds(const fan::vec2i& pos) const {
    return pos.x >= 0 && pos.x < size.x &&
      pos.y >= 0 && pos.y < size.y;
  }

  fan::vec2i tilemap_t::to_grid(const fan::vec2& world_pos) const {
    return fan::vec2i(
      world_pos.x / tile_size.x,
      world_pos.y / tile_size.y
    );
  }

  fan::color terrain_palette_t::get(int value) const {
    if (value <= stops.front().first) return stops.front().second;
    if (value >= stops.back().first) return stops.back().second;

    for (std::size_t i = 0; i < stops.size() - 1; ++i) {
      if (value >= stops[i].first && value <= stops[i + 1].first) {
        int v1 = stops[i].first;
        int v2 = stops[i + 1].first;
        auto c1 = stops[i].second;
        auto c2 = stops[i + 1].second;
        f32_t factor = (value - v1) / f32_t(v2 - v1);
        return c1.lerp(c2, factor);
      }
    }
    return fan::color::rgb(0, 0, 0);
  }

  void generate_mesh(
    const vec2& noise_size,
    const std::vector<std::uint8_t>& noise_data,
    const fan::graphics::image_t& texture,
    std::vector<fan::graphics::shape_t>& out_mesh,
    const terrain_palette_t& palette,
    const sprite_properties_t& cp)
  {
    sprite_properties_t sp = cp;
    sp.size = fan::graphics::viewport_get_size(sp.render_view->viewport) / noise_size / 2;
    out_mesh.resize(noise_size.multiply());

    for (int i = 0; i < noise_size.y; ++i) {
      for (int j = 0; j < noise_size.x; ++j) {
        int index = (i * noise_size.x + j) * 3;
        int grayscale = noise_data[index];
        sp.position = fan::vec2(i, j) * sp.size * 2;
        sp.image = texture;
        sp.color = palette.get(grayscale);
        sp.color.a = 1;
        out_mesh[i * noise_size.x + j] = fan::graphics::sprite_t(sp);
      }
    }
  }

  fan::event::task_t async_generate_mesh(
    vec2 noise_size,
    const std::vector<std::uint8_t>& noise_data,
    const fan::graphics::image_t& texture,
    std::vector<fan::graphics::shape_t>& out_mesh,
    const terrain_palette_t& palette,
    sprite_properties_t cp)
  {
    sprite_properties_t sp = cp;
    sp.size = fan::graphics::viewport_get_size(sp.render_view->viewport) / noise_size / 2;
    out_mesh.resize(noise_size.multiply());
    for (int i = 0; i < noise_size.y; ++i) {
      for (int j = 0; j < noise_size.x; ++j) {
        int index = (i * noise_size.x + j) * 3;
        int grayscale = noise_data[index];
        sp.position = fan::vec2(i, j) * sp.size * 2;
        sp.image = texture;
        sp.color = palette.get(grayscale);
        sp.color.a = 1;
        out_mesh[i * noise_size.x + j] = fan::graphics::sprite_t(sp);
      }
      co_await fan::co_sleep(1);
    }
  }
#endif
}

#if defined(FAN_2D)
namespace fan::image {
  plane_split_t plane_split(void* pixel_data, const fan::vec2ui& size, std::uint32_t format) {
    plane_split_t result;
    std::uint64_t offset = 0;
    if (format == fan::graphics::image_format_e::yuv420p) {
      result.planes[0] = pixel_data;
      result.planes[1] = (std::uint8_t*)pixel_data + (offset += size.multiply());
      result.planes[2] = (std::uint8_t*)pixel_data + (offset += size.multiply() / 4);
    }
    else {
      fan::throw_error_impl("undefined");
    }
    return result;
  }
  plane_split_t plane_split(const std::vector<std::vector<std::uint8_t>>& planes) {
    plane_split_t result;
    for (std::size_t i = 0; i < planes.size() && i < 4; i++) {
      result.planes[i] = const_cast<std::uint8_t*>(planes[i].data());
    }
    return result;
  }
}

namespace fan::graphics {
  bool tile_world_generator_t::is_solid(int x, int y) {
    if (x < 0 || x >= map_size.x || y < 0 || y >= map_size.y) {
      return true;
    }
    return tiles[x + y * map_size.x];
  }

  int tile_world_generator_t::count_neighbors(int x, int y) {
    int c = 0;
    for (int j = -1; j <= 1; j++) {
      for (int i = -1; i <= 1; i++) {
        if (is_solid(x + i, y + j)) {
          c++;
        }
      }
    }
    return c;
  }

  void tile_world_generator_t::iterate() {
    std::vector<bool> nt(map_size.x * map_size.y);
    for (int y = 0; y < map_size.y; y++) {
      for (int x = 0; x < map_size.x; x++) {
        nt[x + y * map_size.x] = count_neighbors(x, y) >= 5;
      }
    }
    tiles = std::move(nt);
  }

  void tile_world_generator_t::init_tile_world() {
    tiles.resize(map_size.x * map_size.y);
    for (int i = 0; i < map_size.x * map_size.y; i++) {
      tiles[i] = fan::random::value(0.f, 1.0f) < initial_fill;
    }
  }

  void tile_world_generator_t::init() {
    init_tile_world();
  }

  void polyline_build(const polyline_properties_t& props, std::vector<vertex_t>& out) {
    out.clear();
    if (props.points.size() < 2 || props.thickness <= 0.f) {
      return;
    }

    std::vector<fan::vec2> pts;
    pts.reserve(props.points.size());
    for (auto& p : props.points) {
      pts.push_back(p);
    }

    f32_t half = props.thickness * 0.5f;
    int n = pts.size();

    auto emit_pair = [&](const fan::vec2& p, const fan::vec2& perp) {
      out.push_back({fan::vec3(p + perp * half, props.depth), props.color});
      out.push_back({fan::vec3(p - perp * half, props.depth), props.color});
    };

    auto emit_round_cap = [&](const fan::vec2& p, const fan::vec2& dir, bool start) {
      int steps = 12;
      f32_t base = std::atan2(dir.y, dir.x);
      f32_t offset = start ? fan::math::pi : 0.f;
      for (int i = 0; i <= steps; ++i) {
        f32_t a = base + offset + fan::math::pi * (i / (f32_t)steps);
        fan::vec2 perp(std::cos(a), std::sin(a));
        emit_pair(p, perp);
      }
    };

    if (props.cap_start == polyline_cap_t::square) {
      fan::vec2 dir = (pts[1] - pts[0]).normalize();
      pts.insert(pts.begin(), pts[0] - dir * half);
      n++;
    }
    else if (props.cap_start == polyline_cap_t::round) {
      fan::vec2 dir = (pts[1] - pts[0]).normalize();
      emit_round_cap(pts[0], -dir, true);
    }

    if (props.cap_end == polyline_cap_t::square) {
      fan::vec2 dir = (pts[n - 1] - pts[n - 2]).normalize();
      pts.push_back(pts[n - 1] + dir * half);
      n++;
    }

    for (int i = 0; i < n; ++i) {
      fan::vec2 p = pts[i];
      fan::vec2 dir_prev, dir_next;

      if (i > 0) {
        dir_prev = pts[i] - pts[i - 1];
        f32_t len = dir_prev.length();
        if (len > 0) dir_prev /= len;
      }
      if (i + 1 < n) {
        dir_next = pts[i + 1] - pts[i];
        f32_t len = dir_next.length();
        if (len > 0) dir_next /= len;
      }

      if (i == 0) {
        fan::vec2 perp(-dir_next.y, dir_next.x);
        emit_pair(p, perp);
        continue;
      }

      if (i == n - 1) {
        fan::vec2 perp(-dir_prev.y, dir_prev.x);
        emit_pair(p, perp);
        continue;
      }

      fan::vec2 perp_prev(-dir_prev.y, dir_prev.x);
      fan::vec2 perp_next(-dir_next.y, dir_next.x);

      if (props.join == polyline_join_t::round) {
        f32_t a0 = std::atan2(perp_prev.y, perp_prev.x);
        f32_t a1 = std::atan2(perp_next.y, perp_next.x);
        f32_t diff = a1 - a0;
        if (diff > fan::math::pi) diff -= fan::math::two_pi;
        if (diff < -fan::math::pi) diff += fan::math::two_pi;
        f32_t ad = std::abs(diff);

        if (ad < 1e-3f) {
          emit_pair(p, perp_prev);
        }
        else {
          int steps = std::clamp((int)(ad / (fan::math::pi / 16.f)), 2, 16);
          for (int j = 0; j <= steps; ++j) {
            f32_t t = j / (f32_t)steps;
            f32_t a = a0 + diff * t;
            fan::vec2 perp(std::cos(a), std::sin(a));
            emit_pair(p, perp);
          }
        }
      }
      else {
        // miter and bevel both use averaged perp
        fan::vec2 perp = perp_prev + perp_next;
        f32_t len = perp.length();
        if (len > 0) perp /= len;
        else perp = perp_prev;
        emit_pair(p, perp);
      }
    }

    if (props.cap_end == polyline_cap_t::round) {
      fan::vec2 dir = (pts[n - 1] - pts[n - 2]).normalize();
      emit_round_cap(pts[n - 1], dir, false);
    }
  }

  void update_infinite_tiled_sprite(
    shape_t& sprite,
    fan::vec2 tile_size,
    fan::vec2 world_size)
  {
    tile_size *= 2.f;
    fan::vec2 world_min = world_size * 0.5f;
    sprite.set_tc_size(world_size / tile_size);
    fan::vec2 grid_offset = (world_min.fmod(tile_size) + tile_size).fmod(tile_size);
    sprite.set_tc_position((grid_offset + tile_size / 2.f) / tile_size);
  }

  animation_frame_awaiter::animation_frame_awaiter(
    fan::graphics::shapes::shape_t* sprite_,
    const std::string& anim_,
    int frame_)
    : sprite(sprite_), animation_name(anim_), target_frame(frame_) {}

  bool animation_frame_awaiter::check_condition() const {
    return sprite && sprite->get_current_sprite_sheet_frame() >= (target_frame - 1) &&
      sprite->get_current_sprite_sheet().name == animation_name;
  }

  terrain_t::terrain_t(fan::vec2 position, fan::vec2 size, f32_t tile_size, std::initializer_list<terrain_t::tile_t> tiles) : size(size) {
    for (int y = 0; y < size.y; y++) {
      for (int x = 0; x < size.x; x++) {
        f32_t r = fan::random::value_f32(0, 1);
        f32_t acc = 0;
        for (auto& t : tiles) {
          if (r < (acc += t.chance)) {
            spawn_tile(t.sprite, {x, y}, tile_size);
            break;
          }
        }
      }
    }
  }

  void terrain_t::spawn_tile(fan::str_view_t sprite_name, fan::vec2i pos, f32_t tile_size) {
    tiles.push_back(sprite_t(
      fan::vec2(pos) * tile_size, /*position*/
      {tile_size, tile_size}, /*size*/
      sprite_name /*image path*/
    ));
  }

  void update_tiling_background(
    fan::graphics::shape_t& sprite, 
    fan::vec2 tile_size, 
    fan::graphics::render_view_t* rv) 
  {
    f32_t zoom = camera_get_zoom(*rv);
    fan::vec2 c = camera_get_center(*rv);
    fan::vec2 ws = fan::window::get_size();
    fan::vec2 half_size = ws / zoom / 2.f;

    sprite.set_position(c);
    sprite.set_size(half_size);
    sprite.set_tc_position((c - half_size) / tile_size / 2.f);
    sprite.set_tc_size(half_size / tile_size);
  }

#if defined(FAN_3D)
  line3d_t::line3d_t(line3d_properties_t p) {
    *(fan::graphics::shapes::shape_t*)this = fan::graphics::shapes::shape_t(
      fan_init_struct(
        typename fan::graphics::shapes::line3d_t::properties_t,
        .camera = p.render_view->camera,
        .viewport = p.render_view->viewport,
        .src = p.src,
        .dst = p.dst,
        .color = p.color,
        .blending = p.blending
      ));
  }

  rectangle3d_t::rectangle3d_t(rectangle3d_properties_t p) {
    *(fan::graphics::shapes::shape_t*)this = fan::graphics::shapes::shape_t(
      fan_init_struct(
        typename fan::graphics::shapes::rectangle3d_t::properties_t,
        .camera = p.render_view->camera,
        .viewport = p.render_view->viewport,
        .position = p.position,
        .size = p.size,
        .color = p.color,
        .blending = p.blending
      ));
  }
#endif

  aabb_t::aabb_t(const fan::vec3& c, const fan::vec2& hsize, f32_t d, const fan::color& col)
    : center(c), half_size(hsize), color(col), depth(d) {
    fan::vec3 bl(center.x - half_size.x, center.y - half_size.y, depth);
    fan::vec3 br(center.x + half_size.x, center.y - half_size.y, depth);
    fan::vec3 tr(center.x + half_size.x, center.y + half_size.y, depth);
    fan::vec3 tl(center.x - half_size.x, center.y + half_size.y, depth);
    edges[0] = line_t(line_properties_t {.src = bl, .dst = br, .color = color});
    edges[1] = line_t(line_properties_t {.src = br, .dst = tr, .color = color});
    edges[2] = line_t(line_properties_t {.src = tr, .dst = tl, .color = color});
    edges[3] = line_t(line_properties_t {.src = tl, .dst = bl, .color = color});
  }

  void rectangle_bordered(fan::vec3 pos, fan::vec2 outer_size, fan::color outer_col, fan::vec2 inner_size, fan::color inner_col, fan::graphics::render_view_t* rv) {
    rectangle(pos, outer_size, outer_col, rv);
    rectangle(fan::vec3(pos.x, pos.y, pos.z + 1.f), inner_size, inner_col, rv);
  }

  void polyline_t::set(const polyline_properties_t& props) {
    std::vector<vertex_t> verts;
    polyline_build(props, verts);
    mesh = polygon_t {{
      .position = fan::vec3(0, 0, props.depth),
      .vertices = verts,
      .draw_mode = primitive_topology_t::triangle_strip,
      .enable_culling = false
    }};
  }
} // namespace fan::graphics

namespace fan {
  f32_t apply_ease(ease_e easing, f32_t t) {
    switch (easing) {
    case ease_e::linear:  return t;
    case ease_e::sine:    return (std::sin((t - 0.5f) * fan::math::pi) + 1.f) * 0.5f;
    case ease_e::pulse:   return std::sin(t * fan::math::pi);
    case ease_e::ease_in: return t * t;
    case ease_e::ease_out: return 1.f - (1.f - t) * (1.f - t);
    }
    return t;
  }

  auto_color_transition_t pulse_red(f32_t duration) {
    auto_color_transition_t t;
    t.from = fan::colors::white;
    t.to = fan::color(1.f, 0.2f, 0.2f);
    t.duration = duration;
    t.loop = true;
    t.easing = ease_e::pulse;
    t.setup_lerp();
    return t;
  }

  color_transition_t fade_out(f32_t duration) {
    color_transition_t t;
    t.from = fan::colors::white;
    t.to = fan::colors::transparent;
    t.duration = duration;
    t.loop = false;
    t.easing = ease_e::ease_out;
    if constexpr (requires (fan::color a, fan::color b, f32_t u) { a.lerp(b, u); }) {
      t.lerp = [](const fan::color& a, const fan::color& b, f32_t u) { return a.lerp(b, u); };
    }
    return t;
  }

  vec2_transition_t move_linear(const fan::vec2& from, const fan::vec2& to, f32_t duration) {
    vec2_transition_t t;
    t.from = from;
    t.to = to;
    t.duration = duration;
    t.loop = false;
    t.easing = ease_e::linear;
    t.lerp = [](const fan::vec2& a, const fan::vec2& b, f32_t u) { return a + (b - a) * u; };
    return t;
  }

  vec2_transition_t move_pingpong(const fan::vec2& from, const fan::vec2& to, f32_t duration) {
    vec2_transition_t t;
    t.from = from;
    t.to = to;
    t.duration = duration;
    t.loop = true;
    t.easing = ease_e::pulse;
    t.lerp = [](const fan::vec2& a, const fan::vec2& b, f32_t u) { return a + (b - a) * u; };
    return t;
  }

  fan::event::task_t fade_transition(
    fan::graphics::lighting_t& lighting,
    bool& is_changing_flag,
    const fan::vec3& fadeout_color,
    const fan::vec3& fadein_color,
    std::function<void()> swap_cb
  ) {
    is_changing_flag = true;

    lighting.set_target(fadeout_color);
    while (!lighting.is_near_target()) {
      co_await fan::graphics::co_next_frame();
    }

    swap_cb();

    lighting.set_target(fadein_color);
    while (!lighting.is_near_target()) {
      co_await fan::graphics::co_next_frame();
    }

    is_changing_flag = false;
  }
}

#endif

#endif