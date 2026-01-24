// This file is meant to stay up-to-date. More library usage will be implemented and showcased over time

#include <fan/utility.h> // OFFSETLESS

#include <vector>
#include <string>
#include <array>////
#include <mutex>
#include <atomic>
#include <cmath>
#include <coroutine>
#include <functional>
#include <filesystem>
#include <fstream>
//
import fan;

import fan.graphics.gui.settings_menu;

// include macro extensions after import fan;
#include <fan/graphics/types.h>
//
using namespace fan::graphics;
using menu_t = fan::graphics::gui::settings_menu_t;

struct engine_demo_t {
  engine_t engine{{ // initialize before everything
    .renderer= fan::graphics::renderer_t::opengl,
  }};

  engine_demo_t() {
    create_gui();
  }

  // ------------------------SHAPES------------------------

  template<typename shape_t, typename property_func_t>
  static void init_shapes(engine_demo_t* engine_demo, property_func_t&& prop_func) {
    engine_demo->shapes.resize(engine_demo->shape_count);
    fan::vec2 viewport_size = engine_demo->engine.viewport_get_size(engine_demo->right_column_view.viewport);
    for (uint32_t i = 0; i < engine_demo->shape_count; ++i) {
      engine_demo->shapes[i] = prop_func(i, viewport_size);
    }
  }

  static void demo_shapes_init_capsule(engine_demo_t* engine_demo) {
    init_shapes<fan::graphics::capsule_t>(engine_demo, [&](uint32_t i, fan::vec2 viewport_size) {
      return fan::graphics::capsule_t {{
        .render_view = &engine_demo->right_column_view,
        .position = fan::random::vec2(-viewport_size / 2.f, viewport_size / 2.f),
        .center0 = 0,
        .center1 = fan::vec2(0, fan::random::value(10.f, 256.f)),
        .radius = fan::random::value(16.f, 64.f),
        .color = fan::random::color()
      }};
    });
  }
  static void demo_shapes_init_circle(engine_demo_t* engine_demo) {
    init_shapes<fan::graphics::circle_t>(engine_demo, [&](uint32_t i, fan::vec2 viewport_size) {
      return fan::graphics::circle_t{{
        .render_view = &engine_demo->right_column_view,
        .position = fan::random::vec2(-viewport_size / 2.f, viewport_size / 2.f),
        .radius = fan::random::value(16.f, 64.f),
        .color = fan::random::color()
      }};
    });
  }
  static void demo_shapes_init_gradient(engine_demo_t* engine_demo) {
    init_shapes<fan::graphics::gradient_t>(engine_demo, [&](uint32_t i, fan::vec2 viewport_size) {
      return fan::graphics::gradient_t{{
        .render_view = &engine_demo->right_column_view,
        .position = fan::random::vec2(-viewport_size / 2.f, viewport_size / 2.f),
        .size = fan::random::vec2(30, 200),
        .color = {
          fan::random::color(),
          fan::random::color(),
          fan::random::color(),
          fan::random::color()
        }
      }};
    });
  }

  static void demo_shapes_init_grid(engine_demo_t* engine_demo) {
    fan::vec2 viewport_size = engine_demo->engine.viewport_get_size(engine_demo->right_column_view.viewport);
    engine_demo->shapes.emplace_back(fan::graphics::grid_t{{
      .render_view = &engine_demo->right_column_view,
      .position = fan::vec3(-8, -8, 0),
      .size = viewport_size.max(),
      .grid_size = 32,
      .color = fan::colors::white
    }});
  }

  static void demo_shapes_init_universal_image_renderer(engine_demo_t* engine_demo) {
    fan::vec2 viewport_size = engine_demo->engine.viewport_get_size(engine_demo->right_column_view.viewport);
    engine_demo->shapes.emplace_back(fan::graphics::universal_image_renderer_t{{
      .render_view = &engine_demo->right_column_view,
      .position = fan::vec3(0),
      .size = viewport_size.min() / 2,//
    }});////
    std::string pixel_data_str;
    constexpr fan::vec2ui image_size = fan::vec2ui(510, 510);
    fan::io::file::read("images/rgb_510x510_420p.yuv", &pixel_data_str);
    void* pixel_data = pixel_data_str.data();
    auto split = fan::image::plane_split(pixel_data, image_size, fan::graphics::image_format::yuv420p);
    engine_demo->shapes.back().reload(fan::graphics::image_format::yuv420p, split, image_size);
  }

  fan::graphics::image_t image_tire = engine.image_load("images/tire.webp");

  static void demo_shapes_init_lighting(engine_demo_t* engine_demo) {
    fan::vec2 viewport_size = engine_demo->engine.viewport_get_size(engine_demo->right_column_view.viewport);
    
    static auto image_background = engine_demo->engine.image_create(fan::color(0.5, 0.5, 0.5, 1));
    uint32_t lighting_flags = fan::graphics::sprite_flags_e::additive | fan::graphics::sprite_flags_e::circle;

    // Background
    engine_demo->shapes.emplace_back(fan::graphics::sprite_t{{
      .render_view = &engine_demo->right_column_view,
      .position = fan::vec3(0),
      .size = viewport_size/2,
      .image = image_background,
      .flags = lighting_flags
    }});

    engine_demo->shapes.emplace_back(fan::graphics::sprite_t{{
      .render_view = &engine_demo->right_column_view,
      .position = fan::vec3(fan::vec2(0), 1),
      .size = viewport_size.min()/6,
      .image = engine_demo->image_tire,
      .flags = lighting_flags
    }});

    engine_demo->shapes.emplace_back(fan::graphics::light_t{ {
      .render_view = &engine_demo->right_column_view,
      .position = fan::vec3(fan::vec2(viewport_size.x / 3, viewport_size.y / 3) - viewport_size / 2.f , 0),
      .size = viewport_size.min() / 4,
      .color = fan::colors::red
    }});
    engine_demo->shapes.emplace_back(fan::graphics::light_t{ {
      .render_view = &engine_demo->right_column_view,
      .position =  fan::vec3(fan::vec2(viewport_size.x / 1.5, viewport_size.y / 3) - viewport_size / 2.f, 0),
      .size = viewport_size.min() / 3,
      .color = fan::colors::green
    }});
    engine_demo->shapes.emplace_back(fan::graphics::light_t{ {
      .render_view = &engine_demo->right_column_view,
      .position =  fan::vec3(fan::vec2(viewport_size.x / 2, viewport_size.y / 1.5) - viewport_size / 2.f, 0),
      .size = viewport_size.min() / 3,
      .color = fan::colors::blue
    }});
    engine_demo->shapes.emplace_back(fan::graphics::light_t{ {
      .render_view = &engine_demo->right_column_view,
      .position =  fan::vec3(fan::vec2(viewport_size.x / 2, viewport_size.y / 1.5) - viewport_size / 2.f, 0),
      .size = viewport_size.min() / 3,
      .color = fan::colors::purple
    }});
  }
  static void demo_shapes_lighting_update(engine_demo_t* engine_demo) {
    if (engine_demo->shapes.empty()) {
      return;
    }
    engine_demo->engine.lighting.set_target(0.4f);
    fan::vec2 viewport_size = engine_demo->engine.viewport_get_size(engine_demo->right_column_view.viewport);
    engine_demo->shapes[0].set_position(fan::vec2(0));
    engine_demo->shapes[0].set_size(viewport_size/2);

    engine_demo->shapes[1].set_position(fan::vec2(0));
    engine_demo->shapes[1].set_size(viewport_size.min()/6);
    engine_demo->shapes[1].set_angle(engine_demo->shapes[1].get_angle() + fan::vec3{0, 0, engine_demo->engine.delta_time});

    engine_demo->shapes[2].set_size(viewport_size.min() / 3);
    engine_demo->shapes[3].set_size(viewport_size.min() / 3);
    engine_demo->shapes[4].set_size(viewport_size.min() / 3);

    engine_demo->shapes.back().set_position(get_mouse_position(engine_demo->right_column_view));
  }
  static void demo_shapes_lighting_cleanup(engine_demo_t* engine_demo) {
    engine_demo->engine.lighting.set_target(1.f);
  }

  struct demo_particles_t {
    gui::particle_editor_t editor;
  }*demo_particles_data = 0;

  static void demo_shapes_init_particles(engine_demo_t* engine_demo) {
    fan::vec2 viewport_size = engine_demo->engine.viewport_get_size(engine_demo->right_column_view.viewport);
    engine_demo->demo_particles_data = new demo_particles_t;
    engine_demo->demo_particles_data->editor.set_particle_shape(fan::graphics::shapes::particles_t::properties_t{
      .position = fan::vec3(0.f, 0.f, 10.0f),

      .start_size = fan::vec2(32.0f),
      .end_size = fan::vec2(32.0f),

      /*.color = fan::color::from_rgba(0xFF6600FF),*/

      .alive_time = 1.0,
      .count = (uint32_t)engine_demo->shape_count,

      .start_velocity = fan::vec2(100.0f, 100.0f),
      .end_velocity = fan::vec2(100.0f, 100.0f),

      .start_angle_velocity = fan::vec3(0.0f),
      .end_angle_velocity = fan::vec3(0.0f),

      .begin_angle = 0.0f,
      .end_angle = 6.28f,
      .angle = fan::vec3(0.0f, 0.0f, 0.0f),

      .spawn_spacing = fan::vec2(0.0f, 0.0f),
      .expansion_power = 1.0f,

      .start_spread = fan::vec2(0.0f, 0.0f),
      .end_spread = fan::vec2(0.0f, 0.0f),

      .jitter_start = fan::vec2(0.0f),
      .jitter_end = fan::vec2(0.0f),
      .jitter_speed = 0.0f,

      .shape = shapes::particles_t::shapes_e::circle,

      .image = image_load("images/bubble.webp", image_presets::pixel_art()),
      .camera = engine_demo->right_column_view.camera,
      .viewport = engine_demo->right_column_view.viewport,
    });
  }
  static void demo_shapes_particles_update(engine_demo_t* engine_demo) {
    engine_demo->disable_render_gui_bg = true;
    engine_demo->demo_particles_data->editor.render();
  }
  static void demo_shapes_particles_cleanup(engine_demo_t* engine_demo) {
    delete engine_demo->demo_particles_data;
  }

  static void demo_shapes_init_polygon(engine_demo_t* engine_demo) {
    engine_demo->shapes.resize(engine_demo->shape_count);
    fan::vec2 viewport_size = engine_demo->engine.viewport_get_size(engine_demo->right_column_view.viewport);

    for (uint32_t i = 0; i < engine_demo->shape_count; ++i) {
      uint32_t sides = fan::random::value(3u, 12u);
      sides = std::max(3u, sides);

      fan::vec2 position = fan::random::vec2(-viewport_size / 2.f, viewport_size / 2.f);
      f32_t radius = fan::random::value(50.f, 200.f);
      fan::color color = fan::random::color();

      fan::graphics::shapes::polygon_t::properties_t pp;
      pp.vertices.clear();

      std::vector<fan::vec2> polygon_points;
      polygon_points.reserve(sides);

      f32_t angle_step = 2.0f * fan::math::pi / sides;
      for (uint32_t j = 0; j < sides; ++j) {
        f32_t angle = j * angle_step;
        fan::vec2 vertex_position = position + fan::vec2(std::cos(angle), std::sin(angle)) * radius;
        polygon_points.push_back(vertex_position);
      }

      fan::vec2 centroid(0, 0);
      for (const auto& point : polygon_points) {
        centroid += point;
      }
      centroid /= sides;

      for (uint32_t j = 0; j < sides; ++j) {
        pp.vertices.push_back(fan::graphics::vertex_t{
          fan::vec3(polygon_points[j], i),
          color
          });

        pp.vertices.push_back(fan::graphics::vertex_t{
          fan::vec3(polygon_points[(j + 1) % sides], i),
          color
          });

        pp.vertices.push_back(fan::graphics::vertex_t{
          fan::vec3(centroid, i),
          color
        });
      }

      engine_demo->shapes[i] = fan::graphics::polygon_t{
        {
          .render_view = &engine_demo->right_column_view,
          .position = 0,
          .vertices = pp.vertices
        }
      };
    }
  }
  static void demo_shapes_init_rectangle(engine_demo_t* engine_demo) {
    init_shapes<fan::graphics::rectangle_t>(engine_demo, [&](uint32_t i, fan::vec2 viewport_size) {
      return fan::graphics::rectangle_t{{
        .render_view = &engine_demo->right_column_view,
        .position = fan::vec3(fan::random::vec2(-viewport_size / 2.f, viewport_size / 2.f), i),
        .size = fan::random::vec2(30, 200),
        .color = fan::random::color()
      }};
    });
  }

  inline static const char* demo_shader_shape_fragment_shader = R"(#version 330
layout (location = 0) out vec4 o_attachment0;

in vec2 texture_coordinate;
in vec4 instance_color;

uniform sampler2D _t00;
uniform sampler2D _t01;
uniform sampler2D _t02;
uniform float _time;
uniform vec3 lighting_ambient;
uniform vec2 window_size;
uniform vec4 custom_color;

void main() {

  vec2 tc = texture_coordinate;

  vec4 tex_color = vec4(1, 1, 1, 1);

  tex_color = texture(_t00, tc) * instance_color;

  if (tex_color.a <= 0.25) {
    discard;
  }

  vec4 lighting_texture = vec4(texture(_t01, gl_FragCoord.xy / window_size).rgb, 1);
  vec3 base_lit = tex_color.rgb * lighting_ambient;
  vec3 additive_light = lighting_texture.rgb;
  tex_color.rgb = base_lit + additive_light;

  // demonstration of custom shader which blinks the given image and changes hue
  tex_color.rgb *= abs(sin(_time));

  float luminance = dot(tex_color.rgb, vec3(0.299, 0.587, 0.114));
  tex_color.rgb = custom_color.rgb * luminance;
  tex_color.a *= custom_color.a;

  o_attachment0 = tex_color;
})";

  fan::graphics::shader_t demo_shader_shape_shader{engine.get_sprite_vertex_shader(demo_shader_shape_fragment_shader)};
  fan::color custom_color = fan::colors::red;
  static void demo_shapes_init_shader_shape(engine_demo_t* engine_demo) {
    // if engine is initially initialized with vulkan, the nr valu doesnt update
    if (engine_demo->demo_shader_shape_shader.iic()) {
      fan::throw_error("failed to compile custom shader");
      return;
    }
    
    fan::vec2 viewport_size = engine_demo->engine.viewport_get_size(engine_demo->right_column_view.viewport);

    fan::graphics::image_t image = engine_demo->engine.image_load("images/lava_seamless.webp");
    engine_demo->shapes.emplace_back(fan::graphics::shader_shape_t{{
      .render_view = &engine_demo->right_column_view,
      .position = fan::vec3(fan::vec2(0), 3),
      .size = viewport_size / 2,
      .shader = engine_demo->demo_shader_shape_shader,
      .image = image,
    }});
    // init
    engine_demo->engine.shader_set_value(engine_demo->demo_shader_shape_shader, "custom_color", engine_demo->custom_color);
  }
  static void demo_shader_shape_update(engine_demo_t* engine_demo) {
    if (fan::graphics::gui::color_edit4("##c0", &engine_demo->custom_color)) {
      engine_demo->engine.shader_set_value(engine_demo->demo_shader_shape_shader, "custom_color", engine_demo->custom_color);
    }
  }

  static void demo_shapes_init_sprite(engine_demo_t* engine_demo) {
    init_shapes<fan::graphics::sprite_t>(engine_demo, [&](uint32_t i, fan::vec2 viewport_size) {
      return fan::graphics::sprite_t{{
        .render_view = &engine_demo->right_column_view,
        .position = fan::random::vec2(-viewport_size / 2.f, viewport_size / 2.f),
        .size = fan::random::value(30.f, 200.f),
        .color = fan::random::bright_color(),
        .image = engine_demo->image_tire
      }};
    });
  }

  struct demo_sprite_sheet_t {
    fan::graphics::shape_t sprite_with_animation;
  }*demo_sprite_sheet_data = 0;

  static void demo_shapes_init_sprite_sheet(engine_demo_t* engine_demo) {
    engine_demo->demo_sprite_sheet_data = new demo_sprite_sheet_t();
    auto& data = *engine_demo->demo_sprite_sheet_data;

    fan::vec2 viewport_size = engine_demo->engine.viewport_get_size(engine_demo->right_column_view.viewport);

    data.sprite_with_animation = fan::graphics::sprite_sheet_from_json({
      .path = "effects/light/lamp.json",
      .loop = true
    });
    data.sprite_with_animation.set_render_view(engine_demo->right_column_view);
    data.sprite_with_animation.set_position(fan::vec2(0));
    data.sprite_with_animation.set_size(viewport_size / 3.f);
    data.sprite_with_animation.play_sprite_sheet();
  }

  static void demo_sprite_sheet_cleanup(engine_demo_t* engine_demo) {
    delete engine_demo->demo_sprite_sheet_data;
  }
  // ------------------------SHAPES------------------------

  // ------------------------GUI------------------------
  struct demo_shader_live_editor_t {
    std::string shader_code = R"(#version 330
in vec2 texture_coordinate;
layout (location = 0) out vec4 o_attachment0;
uniform float m_time;
uniform sampler2D _t00;

void DrawVignette( inout vec3 color, vec2 uv ) {    
    float vignette = uv.x * uv.y * ( 1.0 - uv.x ) * ( 1.0 - uv.y );
    vignette = clamp( pow( 16.0 * vignette, 0.3 ), 0.0, 1.0 );
    color *= vignette;
}

vec2 CRTCurveUV( vec2 uv ) {
    uv = uv * 2.0 - 1.0;
    vec2 offset = abs( uv.yx ) / vec2( 6.0, 4.0 );
    uv = uv + uv * offset * offset;
    uv = uv * 0.5 + 0.5;
    return uv;
}

void DrawScanline( inout vec3 color, vec2 uv ) {
    float scanline = clamp( 0.95 + 0.05 * cos( 3.14 * ( uv.y + 0.008 * m_time ) * 240.0 * 1.0 ), 0.0, 1.0 );
    float grille = 0.85 + 0.15 * clamp( 1.5 * cos( 3.14 * uv.x * 640.0 * 1.0 ), 0.0, 1.0 );    
    color *= scanline * grille * 1.2;
}

void main() {
    vec2 tex = vec2(texture_coordinate.x, 1.0 - texture_coordinate.y);
    tex = CRTCurveUV(tex*1.05);
    vec3 actual = texture(_t00, tex).rgb;
    o_attachment0 = vec4(actual, 1);
    DrawVignette(o_attachment0.rgb, tex);
    DrawScanline(o_attachment0.rgb, tex);
})";

    fan::graphics::shader_t shader;
    fan::graphics::shape_t shader_shape;
    bool shader_compiled = true;
  }*demo_shader_live_editor_data = 0;

  static void demo_shapes_init_shader_live_editor(engine_demo_t* engine_demo) {
    engine_demo->demo_shader_live_editor_data = new demo_shader_live_editor_t();
    auto& data = *engine_demo->demo_shader_live_editor_data;

    data.shader = engine_demo->engine.get_sprite_vertex_shader(data.shader_code);
    fan::graphics::image_t image = engine_demo->engine.image_load("images/lava_seamless.webp");

    fan::vec2 viewport_pos = engine_demo->engine.viewport_get_position(engine_demo->right_column_view.viewport);
    fan::vec2 viewport_size = engine_demo->engine.viewport_get_size(engine_demo->right_column_view.viewport);
    fan::print(viewport_size);
    data.shader_shape = fan::graphics::shader_shape_t{ {
      .render_view = &engine_demo->right_column_view,
      .position = fan::vec3(fan::vec2(0), 3),
      .size = viewport_size / 2,
      .shader = data.shader,
      .image = image,
    } };
    //engine_demo->interactive_camera.set_zoom(engine_demo->engine.settings_menu.pages.front().split_ratio * 1.2f);
    engine_demo->interactive_camera.reset_view();
  }

  static void demo_shader_live_editor_update(engine_demo_t* engine_demo) {
    auto& data = *engine_demo->demo_shader_live_editor_data;

    if (!data.shader_compiled) {
      gui::text("Failed to compile shader", fan::colors::red);
    }

    fan::vec2 editor_size(gui::get_content_region_avail().x, engine_demo->panel_right_window_size.y / 2.f);
    if (gui::input_text_multiline("##Shader Code", &data.shader_code, editor_size, gui::input_text_flags_allow_tab_input)) {
      engine_demo->engine.shader_set_vertex(data.shader, engine_demo->engine.shader_list[engine_demo->engine.shapes.shaper.GetShader(fan::graphics::shape_type_t::shader_shape)].svertex);
      engine_demo->engine.shader_set_fragment(data.shader, data.shader_code);
      data.shader_compiled = engine_demo->engine.shader_compile(data.shader);
    }
  }
  static void demo_shader_live_editor_cleanup(engine_demo_t* engine_demo) {
    fan::vec2 new_viewport_size(
      engine_demo->panel_right_window_size.x,
      engine_demo->panel_right_window_size.y * (1.0 - engine_demo->engine.settings_menu.pages.front().split_ratio)
    );
    engine_demo->engine.viewport_set_size(engine_demo->right_column_view.viewport, new_viewport_size);
    engine_demo->engine.shader_erase(engine_demo->demo_shader_live_editor_data->shader);
    delete engine_demo->demo_shader_live_editor_data;
  }
  // ------------------------GUI------------------------

  // ------------------------PHYSICS------------------------

    // ------------------------MIRRORS------------------------

  struct demo_physics_mirrors_t {
    int reflect_depth = 2;
    std::vector<rectangle_t> ray_hit_point;
    std::vector<line_t> rays;
    std::vector<fan::graphics::physics::circle_t> circles;
    fan::graphics::physics::polygon_strip_t triangle;
    std::array<physics::rectangle_t, 4> walls;
    line_t user_ray;
    fan::graphics::circle_t user_ray_tips[2];
  }*demo_physics_mirrors_data=0;

  static void on_reflect_depth_resize(engine_demo_t* engine_demo) {
    engine_demo->demo_physics_mirrors_data->ray_hit_point.resize(engine_demo->demo_physics_mirrors_data->reflect_depth + 1, { {
      .render_view = &engine_demo->right_column_view,
      .size = 4,
      .color = fan::colors::red
    } });
    engine_demo->demo_physics_mirrors_data->rays.resize(engine_demo->demo_physics_mirrors_data->reflect_depth + 1, { {
      .render_view = &engine_demo->right_column_view,
      .src = {0, 0, 0xfff},
      .color = fan::colors::green,
      .thickness = 3.f
    } });
  }
  static void demo_physics_init_mirrors(engine_demo_t* engine_demo) {
    fan::vec2 viewport_size = engine_demo->engine.viewport_get_size(engine_demo->right_column_view.viewport);
    engine_demo->demo_physics_mirrors_data = new demo_physics_mirrors_t();
    auto& mirror_data = *engine_demo->demo_physics_mirrors_data;
    static std::vector<vertex_t> triangle_vertices{
     {fan::vec2(400, 400), fan::colors::orange},
     {fan::vec2(400, 600), fan::colors::orange},
     {fan::vec2(700, 600), fan::colors::orange},
    };
    mirror_data.triangle = fan::graphics::physics::polygon_strip_t{ {
      .render_view = &engine_demo->right_column_view,
      .vertices = triangle_vertices
    } };
    for (std::size_t i = 0; i < 5; ++i) {
      mirror_data.circles.push_back({ {
        .render_view = &engine_demo->right_column_view,
        .position = fan::random::vec2(-viewport_size / 2.f, viewport_size / 2.f),
        .radius = fan::random::f32(12, 84),
        .color = fan::colors::orange,
      } });
    }
    mirror_data.walls = physics::create_stroked_rectangle(fan::vec2(0), viewport_size / 2.f, 3);
    for (auto& wall : mirror_data.walls) {
      wall.set_camera(engine_demo->right_column_view.camera);
      wall.set_viewport(engine_demo->right_column_view.viewport);
    }
    mirror_data.user_ray = { {
      .render_view = &engine_demo->right_column_view,
      .src = fan::vec3(fan::vec2(viewport_size.x * 0.1, viewport_size.y * 0.9) - viewport_size / 2.f, 0xfff), /* Multiply by magic values to avoid reflection from walls*/
      .dst = fan::vec2(viewport_size.x * 0.9, viewport_size.y * 0.1) - viewport_size / 2.f,
      .color = fan::colors::white,
      .thickness = 3.f
    } };
    mirror_data.user_ray_tips[0] = { {
      .render_view = &engine_demo->right_column_view,
      .radius = 5.f,
      .color = fan::colors::green,
    } };
    mirror_data.user_ray_tips[1] = mirror_data.user_ray_tips[0];
    mirror_data.user_ray_tips[0].set_position(mirror_data.user_ray.get_src());
    mirror_data.user_ray_tips[1].set_position(mirror_data.user_ray.get_dst());

    on_reflect_depth_resize(engine_demo);
  }
  static void demo_physics_update_mirrors(engine_demo_t* engine_demo) {
    auto& mirror_data = *engine_demo->demo_physics_mirrors_data;
    fan::vec2 src = mirror_data.user_ray.get_src();
    fan::vec2 dst = mirror_data.user_ray.get_dst();
    mirror_data.user_ray.set_line(src, dst);
    
    if (fan::window::is_mouse_down(fan::mouse_right) && engine_demo->mouse_inside_demo_view) {
      mirror_data.user_ray.set_line(get_mouse_position(engine_demo->right_column_view), dst);
      mirror_data.user_ray_tips[0].set_position(mirror_data.user_ray.get_src());
    }
    if (fan::window::is_mouse_down() && engine_demo->mouse_inside_demo_view) {
      mirror_data.user_ray.set_line(src, get_mouse_position(engine_demo->right_column_view));
      mirror_data.user_ray_tips[1].set_position(mirror_data.user_ray.get_dst());
    }
    for (auto [i, d] : fan::enumerate(mirror_data.ray_hit_point)) {
      d.set_position(fan::vec3(-1000));
      mirror_data.rays[i].set_line(0, 0);
    }

    int depth = 0;
    fan::vec2 current_src = src;
    fan::vec2 current_dst = dst;

    while (depth < mirror_data.reflect_depth + 1) {
      if (auto result = fan::physics::raycast(current_src, current_dst)) {
        mirror_data.ray_hit_point[depth].set_position(result.point);

        fan::vec2 direction = (current_dst - current_src).normalized();
        fan::vec2 reflection = direction - result.normal * 2 * direction.dot(result.normal);
        mirror_data.rays[depth].set_line(current_src, result.point);
        mirror_data.rays[depth].set_color(fan::color::hsv(360.f * (depth / (f32_t)(mirror_data.reflect_depth + 1)), 100, 100));

        current_src = result.point + reflection * 0.5f;
        current_dst = result.point + reflection * 10000.f;

        depth++;
      }
      else {
        break;
      }
    }

    fan::graphics::gui::text("Hold Right Click to set ray's STARTING point\nHold Left Click to set ray's ENDING point");
    if (fan::graphics::gui::input_int("Max reflections", &engine_demo->demo_physics_mirrors_data->reflect_depth)) {
      engine_demo->demo_physics_mirrors_data->reflect_depth = std::max(0, engine_demo->demo_physics_mirrors_data->reflect_depth);
      on_reflect_depth_resize(engine_demo);
    }
  }
  static void demo_physics_cleanup_mirrors(engine_demo_t* engine_demo) {
    delete engine_demo->demo_physics_mirrors_data;
  }
    // ------------------------MIRRORS------------------------

    // ------------------------PLATFORMER------------------------

  struct demo_physics_platformer_t {
    f32_t grid_size = 64;
    fan::graphics::image_t highlight_image;
    std::array<physics::rectangle_t, 4> walls;
    fan::graphics::physics::character2d_t player;
    std::vector<physics::rectangle_t> placed_blocks;
  }*demo_physics_platformer_data = 0;

  static void demo_physics_init_platformer(engine_demo_t* engine_demo) {
    engine_demo->demo_physics_platformer_data = new demo_physics_platformer_t();
    auto& data = *engine_demo->demo_physics_platformer_data;
    fan::vec2 viewport_size = engine_demo->engine.viewport_get_size(engine_demo->right_column_view.viewport);

    // Load highlight image. Loads from cache if done repeatedly. Automatically gets freed on engine close.
    data.highlight_image = engine_demo->engine.image_load("images/highlight_hover.webp");

    // Create walls around the viewport
    // Bounds, Thickness
    data.walls = physics::create_stroked_rectangle(fan::vec2(0), viewport_size / 2.f, data.grid_size);
    for (auto& wall : data.walls) {
      wall.set_camera(engine_demo->right_column_view.camera);
      wall.set_viewport(engine_demo->right_column_view.viewport);
    }

    // Create player character
    data.player = fan::graphics::physics::character_capsule(
      { // Visual properties
        .render_view = &engine_demo->right_column_view,
        .position = fan::vec3(fan::vec2(0), 10),
        .radius = 16.f,
        .color = fan::colors::green
      },
    { // Physics properties
      .fixed_rotation = true
    }
    );

    engine_demo->demo_physics_platformer_data->player.enable_default_movement();
  }

  static void demo_physics_update_platformer(engine_demo_t* engine_demo) {
    // GUI
    fan::graphics::gui::text("Controls:", fan::colors::yellow);
    fan::graphics::gui::text("A or D - Move side ways");
    fan::graphics::gui::text("Space - Jump");
    fan::graphics::gui::text("Left Click - Place Block");
    fan::graphics::gui::text("Right Click - Delete Block");

    auto& data = *engine_demo->demo_physics_platformer_data;

    // Get mouse position and snap to grid
    fan::vec2 mouse_pos = get_mouse_position(engine_demo->right_column_view) - data.grid_size / 2.f;
    fan::vec2 place_pos = mouse_pos.snap_to_grid(data.grid_size) + data.grid_size / 2.f;

    // Draw highlight sprite. For better performance you would use fan::graphics::sprite_t object and update the position.
    // This uses fan::graphics::sprite() function to draw the sprite for simplicity and demonstration purposes
    sprite({
      .render_view = &engine_demo->right_column_view,
      .position = fan::vec3(place_pos, 0xfff),
      .size = data.grid_size / 2.f,
      .image = data.highlight_image
      });

    // Place block on mouse click
    if (fan::window::is_mouse_clicked() && engine_demo->mouse_inside_demo_view) {
      bool block_exists = false;
      // Iterate all blocks to find if block exists. 
      // In real implementation you would use constant time lookup and not iterate through all, but this is fine for demo
      for (auto& block : data.placed_blocks) {
        if (block.get_position() == place_pos) {
          block_exists = true;
          break;
        }
      }
      if (!block_exists) {
        data.placed_blocks.emplace_back(physics::rectangle_t{
          {
            .render_view = &engine_demo->right_column_view,
            .position = place_pos,
            .size = data.grid_size / 2.f,
            .color = fan::random::bright_color()
          }
        });
      }
    }

    // Iterate all blocks to delete block using right mouse click
    // In real implementation you would use constant time lookup and not iterate through all, but this is fine for demo
    if (fan::window::is_mouse_clicked(fan::mouse_right) && engine_demo->mouse_inside_demo_view) {
      for (auto it = data.placed_blocks.begin(); it != data.placed_blocks.end(); ++it) {
        if (it->get_position() == place_pos) {
          data.placed_blocks.erase(it);
          break;
        }
      }
    }
  }

  static void demo_physics_cleanup_platformer(engine_demo_t* engine_demo) {
    auto& data = *engine_demo->demo_physics_platformer_data;
    // Unload highlight image
    delete engine_demo->demo_physics_platformer_data;
  }

    // ------------------------PLATFORMER------------------------

    // ------------------------SENSORS------------------------

  struct demo_physics_sensor_t {
    fan::physics::entity_t sensors[3];
    fan::graphics::rectangle_t visuals[3];
    fan::physics::entity_t sensor1;
    fan::physics::entity_t sensor2;
  }*demo_physics_sensor_data = 0;

  static void demo_physics_init_sensor(engine_demo_t* engine_demo) {
    engine_demo->demo_physics_sensor_data = new demo_physics_sensor_t();
    auto& data = *engine_demo->demo_physics_sensor_data;

    fan::vec2 viewport_size = engine_demo->engine.viewport_get_size(engine_demo->right_column_view.viewport);

    for (int i = 0; i < 3; ++i) {
      fan::vec2 position = fan::vec2(viewport_size.x / 4 * (i + 1), viewport_size.y / 2) - viewport_size / 2.f;
      fan::vec2 size = fan::vec2(64, 64);
      data.sensors[i] = fan::physics::create_sensor_rectangle(position, size);

      data.visuals[i] = fan::graphics::rectangle_t{ {
        .render_view = &engine_demo->right_column_view,
        .position = position,
        .size = size,
        .color = fan::colors::blue.set_alpha(0.5)
      } };
    }
    data.sensor1 = engine_demo->engine.physics_context.create_box(0, 64);
    data.sensor2 = engine_demo->engine.physics_context.create_circle(fan::vec2(128, 0), 64);
  }

  static void demo_physics_update_sensor(engine_demo_t* engine_demo) {
    auto& data = *engine_demo->demo_physics_sensor_data;

    fan::graphics::gui::text(fan::colors::yellow, "Sensor Demo");

    data.sensor1.set_physics_position(get_mouse_position(engine_demo->right_column_view));

    fan::graphics::rectangle(fan::vec3(get_mouse_position(engine_demo->right_column_view), 10), 64, fan::colors::red.set_alpha(0.6), &engine_demo->right_column_view);

    fan::graphics::circle(fan::vec3(get_mouse_position(engine_demo->right_column_view)+ fan::vec2(128, 0), 10), 64, fan::colors::red.set_alpha(0.6), &engine_demo->right_column_view);

    data.sensor2.set_physics_position(get_mouse_position(engine_demo->right_column_view) + fan::vec2(128, 0));

    for (int i = 0; i < 3; ++i) {
      fan::vec2 pos = data.visuals[i].get_position();
      if (fan::physics::is_on_sensor(data.sensor1, data.sensors[i]) ||
          fan::physics::is_on_sensor(data.sensor2, data.sensors[i])) 
      {
        data.visuals[i].set_color(fan::colors::green.set_alpha(0.6));
      }
      else {
        data.visuals[i].set_color(fan::colors::blue.set_alpha(0.5));
      }
    }
  }

  static void demo_physics_cleanup_sensor(engine_demo_t* engine_demo) {
    auto& data = *engine_demo->demo_physics_sensor_data;
    for (auto& sensor : data.sensors) {
      sensor.destroy();
    }
    data.sensor1.destroy();
    data.sensor2.destroy();
    delete engine_demo->demo_physics_sensor_data;
  }

  // ------------------------SENSORS------------------------

  //TODO sensors, car, ragdoll, bouncing letters, audio buttons, audio
  // ------------------------PHYSICS------------------------


  // ------------------------ALGORITHMS------------------------

    // ------------------------GRID HIGHLIGHT------------------------

  struct demo_algorithm_grid_highlight_t {
    fan::graphics::tilemap_t tilemap;
    fan::graphics::shapes::shape_t shape;
    enum mode_e { circle, line } mode = circle;
    engine_t::mouse_down_handle_t mouse_down_handle[2];
    fan::vec2 src = 0;
    fan::vec2 dst = 300;
  }*demo_algorithm_grid_highlight_data = 0;

  static fan::graphics::circle_t make_circle(engine_demo_t* engine_demo, fan::vec2 pos) {
    return fan::graphics::circle_t{ {
      .render_view = &engine_demo->right_column_view,
      .position = fan::vec3(pos, 3),
      .radius = 128,
      .color = fan::colors::blue.set_alpha(0.7)
    } };
  }
  static fan::graphics::line_t make_line(engine_demo_t* engine_demo, fan::vec2 src, fan::vec2 dst) {
    return fan::graphics::line_t{ {
      .render_view = &engine_demo->right_column_view,
      .src = src,
      .dst = dst,
      .color = fan::colors::blue,
      .thickness = 5.f
    } };
  }

  static void demo_algorithm_init_grid_highlight(engine_demo_t* engine_demo) {
    engine_demo->demo_algorithm_grid_highlight_data = new demo_algorithm_grid_highlight_t();
    auto& data = *engine_demo->demo_algorithm_grid_highlight_data;

    fan::vec2 viewport_size = engine_demo->engine.viewport_get_size(engine_demo->right_column_view.viewport);
    engine_demo->interactive_camera.set_position(viewport_size / 2.f);

    static constexpr fan::vec2 grid_size = fan::vec2(16, 16);
    data.tilemap.create(
      grid_size,
      fan::colors::red,
      viewport_size,
      0,
      &engine_demo->right_column_view
    );

    data.shape = make_circle(engine_demo, { 0, 0 });

    data.mouse_down_handle[0] = engine_demo->engine.on_mouse_down(fan::mouse_left, [&, engine_demo](const engine_t::button_data_t& bdata) {
      data.src = fan::graphics::screen_to_world(bdata.position, engine_demo->right_column_view);
    });
    data.mouse_down_handle[1] = engine_demo->engine.on_mouse_down(fan::mouse_right, [&, engine_demo](const engine_t::button_data_t& bdata) {
      data.dst = fan::graphics::screen_to_world(bdata.position, engine_demo->right_column_view);
    });
  }
  static void demo_algorithm_update_grid_highlight(engine_demo_t* engine_demo) {
    auto& data = *engine_demo->demo_algorithm_grid_highlight_data;

    if (data.mode == demo_algorithm_grid_highlight_t::circle) {
      fan::graphics::gui::text("Move mouse to highlight grid cells with circle");
    }
    else {
      fan::graphics::gui::text("Left click sets line start");
      fan::graphics::gui::text("Right click sets line end");
    }

    fan::graphics::gui::text("Shape Mode:");

    data.tilemap.reset_colors(fan::colors::red);

    const char* modes[] = { "Circle", "Line" };
    int current = (data.mode == demo_algorithm_grid_highlight_t::circle ? 0 : 1);
    if (fan::graphics::gui::combo("Mode", &current, modes, 2)) {
      data.mode = (current == 0 ? demo_algorithm_grid_highlight_t::circle : demo_algorithm_grid_highlight_t::line);
    }

    if (data.mode == demo_algorithm_grid_highlight_t::circle) {
      fan::vec2 world_pos = get_mouse_position(engine_demo->right_column_view);
      data.shape = make_circle(engine_demo, world_pos);
    }
    else {
      data.shape = make_line(engine_demo, data.src, data.dst);
    }

    data.tilemap.highlight(data.shape, fan::colors::green);
  }

  static void demo_algorithm_cleanup_grid_highlight(engine_demo_t* engine_demo) {
    auto& data = *engine_demo->demo_algorithm_grid_highlight_data;
    delete engine_demo->demo_algorithm_grid_highlight_data;
  }

    // ------------------------GRID HIGHLIGHT------------------------

    // ------------------------PATHFIND------------------------

  struct demo_algorithm_pathfind_t {
    fan::graphics::grid_t grid_visual;
    fan::graphics::tilemap_t grid;
    fan::graphics::algorithm::pathfind::generator generator;
    fan::vec2 tile_size = fan::vec2(64, 64);
    fan::vec2i src = 0;
    fan::vec2i dst = 2;
    engine_t::mouse_down_handle_t mouse_down_handle[2];
    bool is_dragging_right = false;
    bool is_dragging_left = false;
  }*demo_algorithm_pathfind_data = 0;

  static void demo_algorithm_init_pathfind(engine_demo_t* engine_demo) {
    engine_demo->demo_algorithm_pathfind_data = new demo_algorithm_pathfind_t();
    auto& data = *engine_demo->demo_algorithm_pathfind_data;
    fan::vec2 viewport_size = engine_demo->engine.viewport_get_size(engine_demo->right_column_view.viewport);
    engine_demo->interactive_camera.set_position(viewport_size / 2.f);

    data.grid_visual = {{
      .render_view = &engine_demo->right_column_view,
      .position = fan::vec3(fan::vec2(0, 0), 0xfff0),
      .size = viewport_size.max(),
      .grid_size = viewport_size.max() / data.tile_size
    }};

    data.grid.create(
      data.tile_size,
      fan::colors::gray,
      viewport_size,
      {0, 0},
      &engine_demo->right_column_view
    );
    data.generator.set_world_size({(int)data.grid.size.x, (int)data.grid.size.y});
    data.generator.set_heuristic(fan::graphics::algorithm::pathfind::heuristic::euclidean);
    data.generator.set_diagonal_movement(false);

    auto handle_cell_click = [](engine_demo_t* engine_demo, fan::vec2 mouse_pos, bool is_shift, auto&& action) {
      mouse_pos = fan::graphics::screen_to_world(mouse_pos, engine_demo->right_column_view);
      mouse_pos -= engine_demo->demo_algorithm_pathfind_data->tile_size / 2.f;
      fan::vec2i cell = (mouse_pos / engine_demo->demo_algorithm_pathfind_data->tile_size).snap_to_grid(1).floor().clamp(fan::vec2i(0, 0), engine_demo->demo_algorithm_pathfind_data->grid.size - 1);
      action(cell, is_shift);
    };

    data.mouse_down_handle[0] = engine_demo->engine.on_mouse_down(fan::mouse_right,
      [&, engine_demo](const engine_t::mouse_down_data_t& bdata) {
      if (bdata.state == fan::mouse_state::press) {
        data.is_dragging_right = fan::graphics::inside(engine_demo->right_column_view, bdata.position);
      }
      if (bdata.state == fan::mouse_state::release) {
        data.is_dragging_right = false;
        return;
      }
      if (!data.is_dragging_right) {
        return;
      }
      handle_cell_click(engine_demo, bdata.position, fan::window::is_key_down(fan::key_left_shift),
        [&](fan::vec2i cell, bool shift) {
        if (shift) {
          data.grid.remove_wall(cell, data.generator);
        }
        else {
          data.src = cell;
          data.grid.set_source(cell, fan::colors::green);
        }
      });
    });

    data.mouse_down_handle[1] = engine_demo->engine.on_mouse_down(fan::mouse_left,
      [&, engine_demo](const engine_t::mouse_down_data_t& bdata) {
      if (bdata.state == fan::mouse_state::press) {
        data.is_dragging_left = fan::graphics::inside(engine_demo->right_column_view, bdata.position);
      }
      if (bdata.state == fan::mouse_state::release) {
        data.is_dragging_left = false;
        return;
      }
      if (!data.is_dragging_left) {
        return;
      }
      handle_cell_click(engine_demo, bdata.position, fan::window::is_key_down(fan::key_left_shift),
        [&](fan::vec2i cell, bool shift) {
        if (shift) {
          data.grid.add_wall(cell, data.generator);
        }
        else {
          data.dst = cell;
          data.grid.set_destination(cell, fan::colors::red);
        }
      });
    });
  }

  static void demo_algorithm_update_pathfind(engine_demo_t* engine_demo) {
    auto& data = *engine_demo->demo_algorithm_pathfind_data;

    data.grid.reset_colors(fan::colors::black);

    auto path = data.grid.find_path(
      data.src,
      data.dst,
      data.generator,
      fan::graphics::algorithm::pathfind::heuristic::euclidean,
      false
    );

    data.grid.highlight_path(path, fan::colors::cyan);
    data.grid.set_source(data.src, fan::colors::green);
    data.grid.set_destination(data.dst, fan::colors::red);

    fan::graphics::gui::text("Left click: set line ending point / add wall with Shift");
    fan::graphics::gui::text("Right click: set line starting point / remove wall with Shift");
  }

  static void demo_algorithm_cleanup_pathfind(engine_demo_t* engine_demo) {
    auto& data = *engine_demo->demo_algorithm_pathfind_data;
    delete engine_demo->demo_algorithm_pathfind_data;
  }

    // ------------------------PATHFIND------------------------

    // ------------------------SORTING------------------------

  struct demo_algorithm_sorting_t {
    struct node_t {
      fan::graphics::rectangle_t r;
      int value;
      fan::vec2 target_pos;
    };

    std::vector<node_t> lines;
    int step = 0;
    int i = 0;
    int comparisons_per_frame = 200;
  }*demo_sorting_data = 0;

  static void demo_algorithm_sorting_init(engine_demo_t* engine_demo) {
    engine_demo->demo_sorting_data = new demo_algorithm_sorting_t();
    auto& data = *engine_demo->demo_sorting_data;
    const fan::vec2 viewport_size = engine_demo->engine.viewport_get_size(engine_demo->right_column_view.viewport);
    const int count = engine_demo->shape_count;
    const f32_t bar_width = viewport_size.x / (f32_t)count;
    data.lines.reserve(count);
    for (int i = 0; i < count; ++i) {
      f32_t norm = (f32_t)i / (f32_t)count;
      f32_t h = viewport_size.y * (1.f - norm);
      fan::vec2 size = {bar_width * 0.5f, h * 0.5f};
      f32_t rect_x = (count - 1 - i) * bar_width - viewport_size.x * 0.5f + bar_width * 0.5f;
      f32_t rect_y = viewport_size.y * 0.5f - size.y;
      fan::vec2 rect_pos = {rect_x, rect_y};
      data.lines.push_back({
        .r = fan::graphics::rectangle_t{{
          .render_view = &engine_demo->right_column_view,
          .position = rect_pos,
          .size = size,
          .color = fan::color::hsv(norm * 360.f, 100, 100)
        }},
        .value = i,
        .target_pos = rect_pos
        });
    }
    for (int i = data.lines.size() - 1; i > 0; --i) {
      std::swap(data.lines[i].value, data.lines[fan::random::value_i64(0, i)].value);
    }
    for (int i = 0; i < data.lines.size(); ++i) {
      auto& node = data.lines[i];
      f32_t rect_x = (count - 1 - node.value) * bar_width - viewport_size.x * 0.5f + bar_width * 0.5f;
      node.target_pos = {rect_x, node.r.get_position().y};
      node.r.set_position(node.target_pos);
    }
  }
  static void demo_algorithm_sorting_update(engine_demo_t* engine_demo) {
    menus_engine_demo_render_element_count(&engine_demo->engine.settings_menu);
    auto& data = *engine_demo->demo_sorting_data;
    const fan::vec2 viewport_size = engine_demo->engine.viewport_get_size(engine_demo->right_column_view.viewport);
    const int count = data.lines.size();
    const f32_t bar_width = viewport_size.x / (f32_t)count;
    int comparisons = 0;
    while (comparisons < data.comparisons_per_frame && data.step < data.lines.size()) {
      if (data.i < data.lines.size() - 1 - data.step) {
        if (data.lines[data.i].value > data.lines[data.i + 1].value) {
          std::swap(data.lines[data.i].value, data.lines[data.i + 1].value);
          for (int idx = 0; idx < 2; ++idx) {
            auto& node = data.lines[data.i + idx];
            f32_t rect_x = (count - 1 - node.value) * bar_width - viewport_size.x * 0.5f + bar_width * 0.5f;
            node.target_pos = {rect_x, node.r.get_position().y};
          }
        }
        data.i++;
      }
      else {
        data.i = 0;
        data.step++;
      }
      comparisons++;
    }
    for (auto& node : data.lines) {
      fan::vec2 current = node.r.get_position();
      fan::vec2 delta = node.target_pos - current;
      node.r.set_position(fabs(delta.x) < 0.5f ? node.target_pos : current + delta * 0.25f);
    }
    fan::graphics::gui::text(fan::colors::yellow, "Sorting visualization (bubble sort)");
  }

  static void demo_algorithm_sorting_cleanup(engine_demo_t* engine_demo) {
    delete engine_demo->demo_sorting_data;
  }

    // ------------------------SORTING------------------------

    // ------------------------TERRAIN GENERATION------------------------

  struct demo_algorithm_terrain_t {
    fan::noise_t noise;
    fan::graphics::terrain_palette_t palette;
    std::vector<fan::graphics::shape_t> built_mesh;
    fan::vec2 noise_size = 256;
    fan::graphics::image_t dirt;
    engine_t::resize_handle_t resize_handle;
    fan::event::task_t task_gen_mesh;
    std::vector<uint8_t> noise_data;
  }*demo_algorithm_terrain_data = 0;

  static void demo_algorithm_terrain_reload(engine_demo_t* engine_demo, fan::vec2 new_size) {
    auto& data = *engine_demo->demo_algorithm_terrain_data;
    data.built_mesh.clear();
    data.noise.apply();
    data.noise_data = data.noise.generate_data(data.noise_size);
    // wait for task to finish
    data.task_gen_mesh.stop_and_join();
    data.task_gen_mesh = fan::graphics::async_generate_mesh(
      data.noise_size, 
      data.noise_data, 
      data.dirt, 
      data.built_mesh, 
      data.palette,
      {.render_view = &engine_demo->right_column_view}
    );
  }

  static void demo_algorithm_terrain_init(engine_demo_t* engine_demo) {
    engine_demo->demo_algorithm_terrain_data = new demo_algorithm_terrain_t();
    auto& data = *engine_demo->demo_algorithm_terrain_data;
    data.dirt = engine_demo->engine.image_create(fan::colors::white);
    // save noise_data so stack doesnt die in async func
    data.noise_data = data.noise.generate_data(data.noise_size);
    const fan::vec2 viewport_size = engine_demo->engine.viewport_get_size(engine_demo->right_column_view.viewport);
    engine_demo->interactive_camera.set_position(viewport_size / 2.f);
    // coroutine
    data.task_gen_mesh = fan::graphics::async_generate_mesh(
      data.noise_size, 
      data.noise_data, 
      data.dirt, 
      data.built_mesh, 
      data.palette,
      {.render_view = &engine_demo->right_column_view}
    );
    data.resize_handle = engine_demo->engine.on_resize([engine_demo](const engine_t::resize_data_t& rdata) {
      demo_algorithm_terrain_reload(engine_demo, rdata.size);
    });
  }

  static void demo_algorithm_terrain_update(engine_demo_t* engine_demo) {
    auto& data = *engine_demo->demo_algorithm_terrain_data;

    bool reload = false;
    reload |= fan::graphics::gui::drag("seed", &data.noise.seed, 1);
    reload |= fan::graphics::gui::drag("frequency", &data.noise.frequency, 0.001f);
    reload |= fan::graphics::gui::drag("gain", &data.noise.gain, 0.01f);
    reload |= fan::graphics::gui::drag("lacunarity", &data.noise.lacunarity, 0.01f);
    reload |= fan::graphics::gui::drag("octaves", &data.noise.octaves, 1);

    if (reload) {
      demo_algorithm_terrain_reload(engine_demo, fan::window::get_size());
    }
  }

  static void demo_algorithm_terrain_cleanup(engine_demo_t* engine_demo) {
    auto data = engine_demo->demo_algorithm_terrain_data;
    // wait for task to finish
    data->task_gen_mesh.stop_and_join();
    delete data;
  }

    // ------------------------TERRAIN GENERATION------------------------


  // ------------------------ALGORITHMS------------------------

  // ------------------------MULTITHREADING------------------------

  struct demo_multithreaded_image_loading_t {
    std::vector<uint8_t> rgb_data;
    fan::vec2ui image_size;
    std::atomic<uint32_t> generated_rows{0};
    std::mutex data_mutex;
    std::atomic<bool> should_quit{false};
    std::atomic<bool> generation_complete{false};
    std::atomic<bool> needs_update{false}; 
    std::string progress_message = "Generating image: 0%";

    sprite_t image_sprite;
    image_t procedural_image;
    fan::graphics::image_load_properties_t image_load_properties{
      .internal_format = fan::graphics::image_format::rgb_unorm,
      .format = fan::graphics::image_format::rgb_unorm,
    };
  }*demo_multithreaded_image_loading_data=0;

  void demo_generate_procedural_image(demo_multithreaded_image_loading_t* data) {
    data->image_size = fan::vec2ui(1024, 1024);
    size_t total_size = data->image_size.x * data->image_size.y * 3;
    {
      std::lock_guard<std::mutex> lock(data->data_mutex);
      data->rgb_data.resize(total_size, 0);
    }
    for (uint32_t y = 0; y < data->image_size.y; ++y) {
      if (data->should_quit) {
        data->generation_complete.store(true);
        return;
      }
      {
        std::lock_guard<std::mutex> lock(data->data_mutex);
        for (uint32_t x = 0; x < data->image_size.x; ++x) {
          size_t pixel_offset = (y * data->image_size.x + x) * 3;
          f32_t cx = x * 2.0f / data->image_size.x - 1.0f, cy = y * 2.0f / data->image_size.y - 1.0f;
          f32_t dist = std::sqrt(cx * cx + cy * cy), time_factor = y * 0.02f;
          f32_t r = std::sin(cx * 5.0f + time_factor) * std::cos(cy * 5.0f);
          f32_t g = std::sin(dist * 10.0f + time_factor * 2.0f) * 0.5f + 0.5f;
          f32_t b = std::sin(std::atan2(cy, cx) * 5.0f + dist * 10.0f - time_factor * 3.0f) * 0.5f + 0.5f;
          data->rgb_data[pixel_offset] = (uint8_t)((r * 0.5f + 0.5f) * 255.0f);
          data->rgb_data[pixel_offset + 1] = (uint8_t)(g * 255.0f);
          data->rgb_data[pixel_offset + 2] = (uint8_t)(b * 255.0f);
        }
      }
      fan::event::sleep(5);
      data->generated_rows.store(y + 1);
      data->needs_update.store(true);
    }
    data->generation_complete.store(true);
  }

  static void demo_init_multithreaded_image_loading(engine_demo_t* engine_demo) {
    engine_demo->demo_multithreaded_image_loading_data = new demo_multithreaded_image_loading_t;
    auto* data = engine_demo->demo_multithreaded_image_loading_data;
    fan::vec2 viewport_size = engine_demo->engine.viewport_get_size(engine_demo->right_column_view.viewport);
    int height = viewport_size.y / 2;
    height -= height % 4;
    data->image_sprite = {{
      .render_view = &engine_demo->right_column_view,
      .position = 0,
      .size = viewport_size.y / 2
    } };

    data->needs_update.store(false);

    fan::vec2ui texture_size(1024, 1024);
    std::vector<uint8_t> initial_texture(texture_size.x * texture_size.y * 3, 0);

    fan::image::info_t image_info;
    image_info.data = initial_texture.data();
    image_info.size = texture_size;
    data->procedural_image = engine_demo->engine.image_load(image_info, data->image_load_properties);

    data->image_sprite.set_image(data->procedural_image);

    fan::event::thread_create([engine_demo, data] {
      engine_demo->demo_generate_procedural_image(data);
    });
  }
  static void demo_update_multithreaded_image_loading(engine_demo_t* engine_demo) {
    auto* data = engine_demo->demo_multithreaded_image_loading_data;
    if (data->needs_update.load()) {
      uint32_t current_rows = data->generated_rows.load();

      std::vector<uint8_t> temp_data;
      {
        std::lock_guard<std::mutex> lock(data->data_mutex);
        temp_data = data->rgb_data;
      }

      fan::image::info_t update_info;
      update_info.data = temp_data.data();
      update_info.size = data->image_size;

      engine_demo->engine.image_reload(data->procedural_image, update_info, data->image_load_properties);

      data->needs_update.store(false);

      f32_t progress = ceil((f32_t)current_rows / data->image_size.y * 100.0f);
      data->progress_message = "Generating image: " + std::to_string(int(progress)) + "%";
    }

    gui::text(data->progress_message);
    gui::text_wrapped("Generates the image procedurally in a background thread while rendering it on the main thread as it's being generated");
    gui::text(fan::colors::yellow, "The image is purposefully generated slowly to simulate load");
  }
  static void demo_cleanup_multithreaded_image_loading(engine_demo_t* engine_demo) {
    auto data = engine_demo->demo_multithreaded_image_loading_data;
    data->should_quit = true;
    while (!data->generation_complete) {}
    delete data;
  }

  // ------------------------MULTITHREADING------------------------

  typedef void(*demo_function_t)(engine_demo_t*);
  typedef void(*demo_function_update_t)(engine_demo_t*);
  //
  static void default_update_function(engine_demo_t* engine_demo) {
    
  }
  static void shape_update_function(engine_demo_t* engine_demo) {
    menus_engine_demo_render_element_count(&engine_demo->engine.settings_menu);
  }

  struct demo_t {
    const char* name;
    demo_function_t init_function = 0;
    demo_function_update_t update_function = default_update_function;
    demo_function_t cleanup_function = nullptr;
  };

  #define engine_demo (*OFFSETLESS(OFFSETLESS(menu, fan::graphics::engine_t, settings_menu), engine_demo_t, engine))

  inline static auto demos = std::to_array({
    // Shapes
    demo_t{.name = "Capsule",                     .init_function = demo_shapes_init_capsule,               .update_function=shape_update_function                                                                                  },
    demo_t{.name = "Circle",                      .init_function = demo_shapes_init_circle,                .update_function=shape_update_function                                                                                  },
    demo_t{.name = "Gradient",                    .init_function = demo_shapes_init_gradient,              .update_function=shape_update_function                                                                                  },
    demo_t{.name = "Grid",                        .init_function = demo_shapes_init_grid,                                                                                                                                          },
    demo_t{.name = "Image Decoder",               .init_function = demo_shapes_init_universal_image_renderer                                                                                                                       },
    demo_t{.name = "Light",                       .init_function = demo_shapes_init_lighting,              .update_function = demo_shapes_lighting_update,             .cleanup_function = demo_shapes_lighting_cleanup            },
    demo_t{.name = "Particles",                   .init_function = demo_shapes_init_particles,             .update_function = demo_shapes_particles_update,            .cleanup_function = demo_shapes_particles_cleanup           },
    demo_t{.name = "Polygon",                     .init_function = demo_shapes_init_polygon,               .update_function = shape_update_function                                                                                },
    demo_t{.name = "Rectangle",                   .init_function = demo_shapes_init_rectangle,             .update_function = shape_update_function                                                                                },
    demo_t{.name = "Shader",                      .init_function = demo_shapes_init_shader_shape,          .update_function = demo_shader_shape_update                                                                             },
    demo_t{.name = "Sprite",                      .init_function = demo_shapes_init_sprite,                .update_function = shape_update_function                                                                                },
    demo_t{.name = "Sprite Sheet",                .init_function = demo_shapes_init_sprite_sheet,                                                                     .cleanup_function = demo_sprite_sheet_cleanup                },
    demo_t{.name = "_next"                                                                                                                                                                                                         },
    // GUI                                                                                                                                                                                                                         
    demo_t{.name = "Live Shader Editor",          .init_function = demo_shapes_init_shader_live_editor,    .update_function = demo_shader_live_editor_update,         .cleanup_function = demo_shader_live_editor_cleanup          },
    demo_t{.name = "_next"                                                                                                                                                                                                         },
    // Physics                                                                                                                                                                                                                     
    demo_t{.name = "Reflective Mirrors",          .init_function = demo_physics_init_mirrors,              .update_function = demo_physics_update_mirrors,            .cleanup_function = demo_physics_cleanup_mirrors             },
    demo_t{.name = "Platformer Builder",          .init_function = demo_physics_init_platformer,           .update_function = demo_physics_update_platformer,         .cleanup_function = demo_physics_cleanup_platformer          },
    demo_t{.name = "Sensors",                     .init_function = demo_physics_init_sensor,               .update_function = demo_physics_update_sensor,             .cleanup_function = demo_physics_cleanup_sensor              },
    demo_t{.name = "_next"                                                                                                                                                                                                         },
    // Algorithms                                                                                                                                                                                                                  
    demo_t{.name = "Grid Highlight",              .init_function = demo_algorithm_init_grid_highlight,     .update_function = demo_algorithm_update_grid_highlight,   .cleanup_function = demo_algorithm_cleanup_grid_highlight    },
    demo_t{.name = "A* Pathfind",                 .init_function = demo_algorithm_init_pathfind,           .update_function = demo_algorithm_update_pathfind,         .cleanup_function = demo_algorithm_cleanup_pathfind          },
    demo_t{.name = "Sorting visualization",       .init_function = demo_algorithm_sorting_init,            .update_function = demo_algorithm_sorting_update,          .cleanup_function = demo_algorithm_sorting_cleanup           },
    demo_t{.name = "Terrain Generation",          .init_function = demo_algorithm_terrain_init,            .update_function = demo_algorithm_terrain_update,          .cleanup_function = demo_algorithm_terrain_cleanup           },
    demo_t{.name = "_next"                                                                                                                                                                                                         },
    // Misc
    demo_t{.name = "Multithreaded image loading", .init_function = demo_init_multithreaded_image_loading, .update_function = demo_update_multithreaded_image_loading, .cleanup_function = demo_cleanup_multithreaded_image_loading }
  });

  static void menus_engine_demo_left(menu_t* menu, const fan::vec2& next_window_position, const fan::vec2& next_window_size) {
    //static std::string code;
    //static bool compiled = false;
    //gui::fragment_shader_editor(engine_t::shape_type_t::line, &code, &compiled);

    gui::set_next_window_pos(next_window_position);
    gui::set_next_window_size(next_window_size);
    if (auto wnd = gui::window("##Menu Engine Demo Left", 0, wnd_flags)) {
      render_demos(menu, {
        "SHAPES",
        "GUI",
        "PHYSICS",
        "ALGORITHMS",
        "MULTITHREADING"
      });
    }
  }
  static void menus_engine_demo_right(menu_t* menu, const fan::vec2& next_window_position, const fan::vec2& next_window_size) {
    gui::set_next_window_pos(next_window_position);
    gui::set_next_window_size(next_window_size);
    gui::push_style_color(gui::col_window_bg, fan::colors::transparent);
    engine_demo.mouse_inside_demo_view = engine_demo.engine.is_mouse_inside(engine_demo.right_column_view);
    engine_demo.interactive_camera.ignore_input = !engine_demo.mouse_inside_demo_view;

    if (auto wnd = gui::window(
      "##Menu Engine Demo Right Content Bottom",
      wnd_flags | gui::window_flags_override_input
    )) {
      gui::set_viewport(engine_demo.right_column_view.viewport);

      auto& demo = demos[engine_demo.current_demo_index];
      if (demo.update_function) {
        fan::vec2 cursor_start = gui::get_cursor_screen_pos();
        f32_t start_y = gui::get_cursor_pos_y();

        engine_demo.disable_render_gui_bg = false;
        demo.update_function(&engine_demo);

        if (!engine_demo.disable_render_gui_bg) {
          f32_t end_y = gui::get_cursor_pos_y();
          f32_t content_height = end_y - start_y;
          fan::vec2 window_size = gui::get_window_size();
  
          auto* draw_list = gui::get_background_draw_list();
          fan::vec2 rect_end = cursor_start + fan::vec2(window_size.x - gui::get_style().WindowPadding.x * 2, content_height);
          if (rect_end.y - cursor_start.y > 5.f) { // render bg if content exists
            draw_list->AddRectFilled(
              cursor_start,
              rect_end,
              fan::color(0.05f, 0.05f, 0.05f, 0.99f),
              5.0f
            );
          }
        }
      }

      if (engine_demo.new_demo_index != (uint16_t)-1) {
        engine_demo.clear_and_set_demo(engine_demo.new_demo_index);
        engine_demo.new_demo_index = -1;
      }
    }
    gui::pop_style_color();
  }

  static constexpr int wnd_flags = gui::window_flags_no_move |
    gui::window_flags_no_collapse | gui::window_flags_no_resize| gui::window_flags_no_title_bar;

  static void render_demos(menu_t* menu, const std::vector<std::string>& titles) {
    std::size_t title_index = 0;
    std::string title = titles[title_index];
    gui::text(fan::color::from_rgba(0x948c80ff) * 1.5, title);
    gui::push_style_var(gui::style_var_cell_padding, fan::vec2(0));
    if (auto tbl = gui::table(
      (title + "_settings_left_table_display").c_str(), 
      1, 
      gui::table_flags_borders_inner_h | gui::table_flags_borders_outer_h
    )) {
      {
        gui::push_style_var(gui::style_var_selectable_text_align, fan::vec2(0, 0.5));
        for (auto [demo_index, demo] : fan::enumerate(demos)) {
          if (demo.name == "_next") {
            ++title_index;
            title = titles[title_index];
            gui::end_table();
            gui::new_line();
            gui::new_line();
            gui::text(fan::color::from_rgba(0x948c80ff) * 1.5, title);
            gui::begin_table(title + "_settings_left_table_display", 1,
              gui::table_flags_borders_inner_h |
              gui::table_flags_borders_outer_h
            );
            continue;
          }
          gui::table_next_row();
          gui::table_next_column();
          f32_t row_height = gui::get_text_line_height_with_spacing() * 2;
          if (gui::selectable(demo.name, engine_demo.current_demo_index == demo_index, 0, fan::vec2(0.0f, row_height))) {
            engine_demo.new_demo_index = demo_index;
          }
        }
        gui::pop_style_var();
      }
    }
    gui::pop_style_var();
#undef make_table
  }

  static void menus_engine_demo_render_element_count(menu_t* menu) {
    if (gui::drag("Shape count", &engine_demo.shape_count, 1, 0, std::numeric_limits<int>::max())) {
      auto& demo = demos[engine_demo.current_demo_index];
      if (demo.cleanup_function) {
        demo.cleanup_function(&engine_demo);
      }
      engine_demo.shapes.clear();
      if (demo.init_function) {
        demo.init_function(&engine_demo);
      }
    }
  }


#undef engine_demo

  void create_gui() {
    engine.clear_color = 0;
    // disable actively rendering page and assign "Engine Demos" option as first
    engine.settings_menu.reset_page_selection();
    {
      menu_t::page_t page;
      page.name = "Engine Demos";
      page.toggle = 1;
      page.render_page_left = menus_engine_demo_left;
      page.render_page_right = menus_engine_demo_right;
      page.split_ratio = 0.35f;
      engine.settings_menu.pages.emplace_front(page);
    }
    right_column_view.create();
    interactive_camera.create(
      right_column_view.camera,
      right_column_view.viewport,
      1.f,
      0.f
    );
    interactive_camera.pan_with_middle_mouse = true;
  }

  void update() {
    engine.render_settings_menu = true;
    if (engine.settings_menu.current_page != 0) {
      shapes.clear();
    }
  }

  void clear_and_set_demo(size_t demo_index) {
    engine_demo_t::demo_t& demo = demos[demo_index];
    if (demos[current_demo_index].cleanup_function) {
      demos[current_demo_index].cleanup_function(this);
    }
    shapes.clear();
    interactive_camera.reset_view();
    demo.init_function(this);
    current_demo_index = demo_index;
  }

  fan::graphics::render_view_t right_column_view;
  // allows to move and zoom camera with mouse
  fan::graphics::interactive_camera_t interactive_camera;
  uint16_t current_demo_index = 0;
  uint16_t new_demo_index = current_demo_index;
  int shape_count = 100;
  std::vector<fan::graphics::shape_t> shapes;
  fan::vec2 panel_right_render_position = 0.f;
  fan::vec2 panel_right_window_size = 0.f;
  bool mouse_inside_demo_view = false;
  bool disable_render_gui_bg = false;
};

#include "library_usage_samples.h"

int main() {////
  engine_demo_t demo;

  //demo.engine.cell_size = 32;
  //demo.engine.culling_rebuild_grid();
  demo.engine.set_culling_enabled(false);
  // Update physics
  demo.engine.update_physics(true);
  fan_window_loop{
    auto camera = fan::graphics::camera_get(demo.right_column_view.camera);
    
    /*fan::vec2 s(
      (camera.coordinates.right - camera.coordinates.left) / 2.f,
      (camera.coordinates.bottom - camera.coordinates.top) / 2.f
    );*/
    //fan::graphics::aabb(0, s*2.f, 5.f, &demo.right_column_view);
    uint32_t v, c;
    demo.engine.get_culling_stats(v, c);
    fan::print_throttled(v, c);
    demo.update();
  };
/*  Optionally
  demo.engine.loop([&]{
    demo.update();
  });
*/
}