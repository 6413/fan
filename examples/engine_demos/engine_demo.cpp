// This file is meant to stay up-to-date. More library usage will be implemented and showcased over time
#include <fan/utility.h> // OFFSETLESS
#include <vector>
#include <string>
#include <array>
#include <mutex>
#include <atomic>
#include <cmath>
#include <unordered_map>
#include <coroutine>
#include <functional>
#include <filesystem>

import fan;

// include macro extensions after import fan;
#include <fan/graphics/types.h>


using namespace fan::graphics;
using menu_t = engine_t::settings_menu_t;

struct engine_demo_t {
  engine_t engine{{ // initialize before everything
    .renderer=engine_t::renderer_t::opengl,
  }};

  // ------------------------STATIC------------------------
  static void demo_shapes_init_capsule(engine_demo_t* engine_demo) {
    engine_demo->shapes.resize(engine_demo->shape_count);
    fan::vec2 viewport_size = engine_demo->engine.viewport_get(engine_demo->right_column_view.viewport).viewport_size;
    for (uint32_t i = 0; i < engine_demo->shape_count; ++i) {
      engine_demo->shapes[i] = fan::graphics::capsule_t{ {
        .render_view = &engine_demo->right_column_view,
        .position = fan::random::vec2(0, viewport_size),
        .center0 = 0,
        .center1 = fan::vec2(0, fan::random::value(10.f, 256.f)),
        .radius = fan::random::value(16.f, 64.f),
        .color = fan::random::color()
      }};
    }
  }
  static void demo_shapes_init_circle(engine_demo_t* engine_demo) {
    engine_demo->shapes.resize(engine_demo->shape_count);
    fan::vec2 viewport_size = engine_demo->engine.viewport_get(engine_demo->right_column_view.viewport).viewport_size;
    for (uint32_t i = 0; i < engine_demo->shape_count; ++i) {
      engine_demo->shapes[i] = fan::graphics::circle_t{ {
        .render_view = &engine_demo->right_column_view,
        .position = fan::random::vec2(0, viewport_size),
        .radius = fan::random::value(16.f, 64.f),
        .color = fan::random::color()
      }};
    }
  }
  static void demo_shapes_init_gradient(engine_demo_t* engine_demo) {
    for (auto& i : engine_demo->shapes) {
      i.erase();
    }
    engine_demo->shapes.resize(engine_demo->shape_count);
    fan::vec2 viewport_size = engine_demo->engine.viewport_get(engine_demo->right_column_view.viewport).viewport_size;
    for (uint32_t i = 0; i < engine_demo->shape_count; ++i) {
      engine_demo->shapes[i] = fan::graphics::gradient_t{ {
        .render_view = &engine_demo->right_column_view,
        .position = fan::random::vec2(0, viewport_size),
        .size = fan::random::vec2(30, 200),
        .color = {
          fan::random::color(),
          fan::random::color(),
          fan::random::color(),
          fan::random::color()
        }
      }};
    }
  }

  static void demo_shapes_init_grid(engine_demo_t* engine_demo) {
    fan::vec2 viewport_size = engine_demo->engine.viewport_get(engine_demo->right_column_view.viewport).viewport_size;
    
    engine_demo->shapes.emplace_back(fan::graphics::grid_t{{
    .render_view = &engine_demo->right_column_view,
    .position = fan::vec3(-8, -8, 0),
    .size = viewport_size.max(),
    .grid_size = 32,
    .color = fan::colors::white
    }});
  }

  static void demo_shapes_init_universal_image_renderer(engine_demo_t* engine_demo) {
    fan::vec2 viewport_size = engine_demo->engine.viewport_get(engine_demo->right_column_view.viewport).viewport_size;
    engine_demo->shapes.emplace_back(fan::graphics::universal_image_renderer_t{{
      .render_view = &engine_demo->right_column_view,
      .position = fan::vec3(viewport_size / 2, 0),    
      .size = viewport_size / 2,//
    }});////
    std::string pixel_data_str;
    constexpr fan::vec2ui image_size = fan::vec2ui(1920, 1080);
    fan::io::file::read("images/output1920.yuv", &pixel_data_str);
    void* pixel_data = pixel_data_str.data();//
    void* datas[3];
    uint64_t offset = 0;
    datas[0] = pixel_data;
    datas[1] = (uint8_t*)pixel_data + (offset += image_size.multiply());
    datas[2] = (uint8_t*)pixel_data + (offset += image_size.multiply() / 4);
    engine_demo->shapes.back().reload(fan::graphics::image_format::yuv420p, datas, image_size);
  }

  fan::graphics::image_t image_tire = engine.image_load("images/tire.webp");
  static void demo_shapes_init_lighting(engine_demo_t* engine_demo) {

    fan::vec2 viewport_size = engine_demo->engine.viewport_get(engine_demo->right_column_view.viewport).viewport_size;
    
    static auto image_background = engine_demo->engine.image_create(fan::color(0.5, 0.5, 0.5, 1));
    uint32_t lighting_flags = engine_t::light_flags_e::additive | engine_t::light_flags_e::circle;

    // bg
    engine_demo->shapes.emplace_back(fan::graphics::sprite_t{{
      .render_view = &engine_demo->right_column_view,
      .position = fan::vec3(viewport_size / 2, 0),
      .size = viewport_size/2,
      .image = image_background,
      .flags = lighting_flags
    }});

    engine_demo->shapes.emplace_back(fan::graphics::sprite_t{{
      .render_view = &engine_demo->right_column_view,
      .position = fan::vec3(viewport_size / 2, 1),
      .size = viewport_size.min()/6,
      .image = engine_demo->image_tire,
      .flags = lighting_flags
    }});

    engine_demo->shapes.emplace_back(fan::graphics::light_t{ {
      .render_view = &engine_demo->right_column_view,
      .position =  fan::vec3(fan::vec2(viewport_size.x / 3, viewport_size.y / 3), 0),
      .size = viewport_size.min() / 4,
      .color = fan::colors::red
    }});
    engine_demo->shapes.emplace_back(fan::graphics::light_t{ {
      .render_view = &engine_demo->right_column_view,
      .position =  fan::vec3(fan::vec2(viewport_size.x / 1.5, viewport_size.y / 3), 0),
      .size = viewport_size.min() / 3,
      .color = fan::colors::green
    }});
    engine_demo->shapes.emplace_back(fan::graphics::light_t{ {
      .render_view = &engine_demo->right_column_view,
      .position =  fan::vec3(fan::vec2(viewport_size.x / 2, viewport_size.y / 1.5), 0),
      .size = viewport_size.min() / 3,
      .color = fan::colors::blue
    }});
    engine_demo->shapes.emplace_back(fan::graphics::light_t{ {
      .render_view = &engine_demo->right_column_view,
      .position =  fan::vec3(fan::vec2(viewport_size.x / 2, viewport_size.y / 1.5), 0),
      .size = viewport_size.min() / 3,
      .color = fan::colors::purple
    }});
  }
  static void demo_lighting_update(engine_demo_t* engine_demo) {
    if (engine_demo->shapes.empty()) {
      return;
    }
    engine_demo->engine.lighting.ambient = 0.5;
    fan::vec2 viewport_size = engine_demo->engine.viewport_get(engine_demo->right_column_view.viewport).viewport_size;
    engine_demo->shapes[0].set_position(viewport_size/2);
    engine_demo->shapes[0].set_size(viewport_size/2);

    engine_demo->shapes[1].set_position(viewport_size/2);
    engine_demo->shapes[1].set_size(viewport_size.min()/6);
    engine_demo->shapes[1].set_angle(engine_demo->shapes[1].get_angle() + fan::vec3{0, 0, engine_demo->engine.delta_time});

    engine_demo->shapes[2].set_size(viewport_size.min() / 3);
    engine_demo->shapes[3].set_size(viewport_size.min() / 3);
    engine_demo->shapes[4].set_size(viewport_size.min() / 3);

    engine_demo->shapes.back().set_position(get_mouse_position(engine_demo->right_column_view));
  }

  static void demo_shapes_init_particles(engine_demo_t* engine_demo) {
    
    static auto particle_texture = engine_demo->engine.image_load("images/waterdrop.webp");

    loco_t::particles_t::properties_t p;
    p.camera = engine_demo->right_column_view.camera;
    p.viewport = engine_demo->right_column_view.viewport;
    p.position = fan::vec3(0, 0, 10);
    p.count = engine_demo->shape_count*100;
    p.size = 4;
    p.begin_angle = 4.133;
    p.end_angle = 1.0;
    p.alive_time = 1e+9;
    p.gap_size = fan::vec2(354.535, 1.0);
    p.max_spread_size = fan::vec2(2423.231, 100.0);
    p.shape = loco_t::particles_t::shapes_e::rectangle;
    p.position_velocity = fan::vec2(0,  334);
    p.image = particle_texture;
    p.color = fan::color(0.4, 0.4, 1.4);
    engine_demo->shapes.push_back(p);
  }

  static void demo_shapes_init_polygon(engine_demo_t* engine_demo) {
    engine_demo->shapes.resize(engine_demo->shape_count);
    fan::vec2 viewport_size = engine_demo->engine.viewport_get(engine_demo->right_column_view.viewport).viewport_size;

    for (uint32_t i = 0; i < engine_demo->shape_count; ++i) {
      uint32_t sides = fan::random::value(3u, 12u);
      sides = std::max(3u, sides);

      fan::vec2 position = fan::random::vec2(0, viewport_size);
      f32_t radius = fan::random::value(50.f, 200.f);
      fan::color color = fan::random::color();

      loco_t::polygon_t::properties_t pp;
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
    
    engine_demo->shapes.resize(engine_demo->shape_count);
    fan::vec2 viewport_size = engine_demo->engine.viewport_get(engine_demo->right_column_view.viewport).viewport_size;
    for (uint32_t i = 0; i < engine_demo->shape_count; ++i) {
      engine_demo->shapes[i] = fan::graphics::rectangle_t{ {
        .render_view = &engine_demo->right_column_view,
        .position = fan::random::vec2(0, viewport_size),
        .size = fan::random::vec2(30, 200),
        .color = fan::random::color()
      }};
    }
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

  loco_t::shader_t demo_shader_shape_shader{engine.get_sprite_vertex_shader(demo_shader_shape_fragment_shader)};
  fan::color custom_color = fan::colors::red;
  static void demo_shapes_init_shader_shape(engine_demo_t* engine_demo) {
    if (engine_demo->demo_shader_shape_shader.iic()) {
      fan::throw_error("failed to compile custom shader");
      return;
    }
    
    fan::vec2 viewport_size = engine_demo->engine.viewport_get(engine_demo->right_column_view.viewport).viewport_size;

    fan::graphics::image_t image = engine_demo->engine.image_load("images/lava_seamless.webp");
    engine_demo->shapes.emplace_back(fan::graphics::shader_shape_t{{
      .render_view = &engine_demo->right_column_view,
      .position = fan::vec3(viewport_size / 2, 3),
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
    engine_demo->shapes.resize(engine_demo->shape_count);
    fan::vec2 viewport_size = engine_demo->engine.viewport_get(engine_demo->right_column_view.viewport).viewport_size;
    for (uint32_t i = 0; i < engine_demo->shape_count; ++i) {
      engine_demo->shapes[i] = fan::graphics::sprite_t{ {
        .render_view = &engine_demo->right_column_view,
        .position = fan::random::vec2(0, viewport_size),
        .size = fan::random::value(30.f, 200.f),
        .color = fan::random::bright_color(), // add tint to the image
        .image = engine_demo->image_tire
      }};
    }
  }
  // ------------------------STATIC------------------------

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
    engine_demo->right_window_split_ratio = 0.5f;

    data.shader = engine_demo->engine.get_sprite_vertex_shader(data.shader_code);
    fan::graphics::image_t image = engine_demo->engine.image_load("images/lava_seamless.webp");

    fan::vec2 viewport_size = engine_demo->engine.viewport_get(engine_demo->right_column_view.viewport).viewport_size;
    data.shader_shape = fan::graphics::shader_shape_t{ {
      .render_view = &engine_demo->right_column_view,
      .position = fan::vec3(viewport_size / 2, 3),
      .size = viewport_size / 2,
      .shader = data.shader,
      .image = image,
    } };
  }

  static void demo_shader_live_editor_update(engine_demo_t* engine_demo) {
    auto& data = *engine_demo->demo_shader_live_editor_data;

    if (!data.shader_compiled) {
      gui::text("Failed to compile shader", fan::colors::red);
    }

    if (gui::input_text_multiline("##Shader Code", &data.shader_code, gui::get_content_region_avail(), gui::input_text_flags_allow_tab_input)) {
      engine_demo->engine.shader_set_vertex(data.shader, engine_demo->engine.shader_list[engine_demo->engine.shaper.GetShader(engine_t::shape_type_t::shader_shape)].svertex);
      engine_demo->engine.shader_set_fragment(data.shader, data.shader_code);
      data.shader_compiled = engine_demo->engine.shader_compile(data.shader);
    }
  }
  static void demo_shader_live_editor_cleanup(engine_demo_t* engine_demo) {
    engine_demo->right_window_split_ratio = 0.2f;
    delete engine_demo->demo_shader_live_editor_data;
  }
  // ------------------------GUI------------------------

  // ------------------------PHYSICS------------------------

  struct demo_physics_mirrors_t {
    int reflect_depth = 2;
    std::vector<rectangle_t> ray_hit_point;
    std::vector<line_t> rays;
    std::vector<fan::graphics::physics::circle_t> circles;
    fan::graphics::physics::polygon_strip_t triangle;
    std::array<physics::rectangle_t, 4> walls;
    line_t user_ray;
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
      .color = fan::colors::green
    } });
  }
  static void demo_physics_init_mirrors(engine_demo_t* engine_demo) {
    fan::vec2 viewport_size = engine_demo->engine.viewport_get(engine_demo->right_column_view.viewport).viewport_size;
    engine_demo->demo_physics_mirrors_data = new demo_physics_mirrors_t();
    static std::vector<vertex_t> triangle_vertices{
     {fan::vec2(400, 400), fan::colors::orange},
     {fan::vec2(400, 600), fan::colors::orange},
     {fan::vec2(700, 600), fan::colors::orange},
    };
    engine_demo->demo_physics_mirrors_data->triangle = fan::graphics::physics::polygon_strip_t{ {
      .render_view = &engine_demo->right_column_view,
      .vertices = triangle_vertices
    } };
    for (std::size_t i = 0; i < 5; ++i) {
      engine_demo->demo_physics_mirrors_data->circles.push_back({ {
        .render_view = &engine_demo->right_column_view,
        .position = fan::random::vec2(0, viewport_size),
        .radius = fan::random::f32(12, 84),
        .color = fan::colors::orange,
      } });
    }
    engine_demo->demo_physics_mirrors_data->walls = physics::create_stroked_rectangle(viewport_size / 2, viewport_size / 2, 3);
    for (auto& wall : engine_demo->demo_physics_mirrors_data->walls) {
      wall.set_camera(engine_demo->right_column_view.camera);
      wall.set_viewport(engine_demo->right_column_view.viewport);
    }
    engine_demo->demo_physics_mirrors_data->user_ray = { {
        .render_view = &engine_demo->right_column_view,
        .src = {0, 500, 0xfff}, 
        .color = fan::colors::white
      } };
    on_reflect_depth_resize(engine_demo);
  }
  static void demo_physics_update_mirrors(engine_demo_t* engine_demo) {
    fan::vec2 src = engine_demo->demo_physics_mirrors_data->user_ray.get_src();
    fan::vec2 dst = engine_demo->demo_physics_mirrors_data->user_ray.get_dst();
    engine_demo->demo_physics_mirrors_data->user_ray.set_line(src, dst);
    bool mouse_inside_viewport = engine_demo->engine.inside_wir(engine_demo->right_column_view.viewport, get_mouse_position(engine_demo->right_column_view));
    if (fan::window::is_mouse_down(fan::mouse_right) && mouse_inside_viewport) {
      engine_demo->demo_physics_mirrors_data->user_ray.set_line(get_mouse_position(engine_demo->right_column_view), dst);
    }
    if (fan::window::is_mouse_down() && mouse_inside_viewport) {
      engine_demo->demo_physics_mirrors_data->user_ray.set_line(src, get_mouse_position(engine_demo->right_column_view));
    }
    for (auto [i, d] : fan::enumerate(engine_demo->demo_physics_mirrors_data->ray_hit_point)) {
      d.set_position(fan::vec3(-1000));
      engine_demo->demo_physics_mirrors_data->rays[i].set_line(0, 0);
    }

    int depth = 0;
    fan::vec2 current_src = src;
    fan::vec2 current_dst = dst;

    while (depth < engine_demo->demo_physics_mirrors_data->reflect_depth + 1) {
      if (auto result = fan::physics::raycast(current_src, current_dst)) {
        engine_demo->demo_physics_mirrors_data->ray_hit_point[depth].set_position(result.point);

        fan::vec2 direction = (current_dst - current_src).normalized();
        fan::vec2 reflection = direction - result.normal * 2 * direction.dot(result.normal);
        engine_demo->demo_physics_mirrors_data->rays[depth].set_line(current_src, result.point);
        engine_demo->demo_physics_mirrors_data->rays[depth].set_color(fan::color::hsv(360.f * (depth / (f32_t)(engine_demo->demo_physics_mirrors_data->reflect_depth + 1)), 100, 100));

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

  // ------------------------PHYSICS------------------------


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
    auto& data = *engine_demo->demo_multithreaded_image_loading_data;
    fan::vec2 viewport_size = engine_demo->engine.viewport_get(engine_demo->right_column_view.viewport).viewport_size;
    int height = viewport_size.y / 2;
    height -= height % 4;
    data.image_sprite = {{
      .render_view = &engine_demo->right_column_view,
      .position = viewport_size / 2,
      .size = viewport_size.y / 2
    } };

    data.needs_update.store(false);

    fan::vec2ui texture_size(1024, 1024);
    std::vector<uint8_t> initial_texture(texture_size.x * texture_size.y * 3, 0);

    fan::image::info_t image_info;
    image_info.data = initial_texture.data();
    image_info.size = texture_size;
    data.procedural_image = engine_demo->engine.image_load(image_info, data.image_load_properties);

    data.image_sprite.set_image(data.procedural_image);

    fan::event::thread_create([engine_demo, &data] {
      engine_demo->demo_generate_procedural_image(&data);
    });
  }
  static void demo_update_multithreaded_image_loading(engine_demo_t* engine_demo) {
    auto& data = *engine_demo->demo_multithreaded_image_loading_data;
    if (data.needs_update.load()) {
      uint32_t current_rows = data.generated_rows.load();

      std::vector<uint8_t> temp_data;
      {
        std::lock_guard<std::mutex> lock(data.data_mutex);
        temp_data = data.rgb_data;
      }

      fan::image::info_t update_info;
      update_info.data = temp_data.data();
      update_info.size = data.image_size;

      engine_demo->engine.image_reload(data.procedural_image, update_info, data.image_load_properties);

      data.needs_update.store(false);

      f32_t progress = ceil((f32_t)current_rows / data.image_size.y * 100.0f);
      data.progress_message = "Generating image: " + std::to_string(int(progress)) + "%";
    }

    gui::text(data.progress_message);
    gui::text_wrapped("Generates the image procedurally in a background thread while rendering it on the main thread as it's being generated");
    gui::text("The image is purposefully generated slowly to simulate load", fan::colors::yellow);
  }
  static void demo_cleanup_multithreaded_image_loading(engine_demo_t* engine_demo) {
    auto& data = *engine_demo->demo_multithreaded_image_loading_data;
    data.should_quit = true;
    delete &data;
  }

  // ------------------------MULTITHREADING------------------------

  typedef void(*demo_function_t)(engine_demo_t*);
  typedef void(*demo_function_update_t)(engine_demo_t*);
  //
  static void default_update_function(engine_demo_t* engine_demo) {
    menus_engine_demo_render_element_count(&engine_demo->engine.settings_menu);
  }

  struct demo_t {
    const char* name;
    demo_function_t init_function = 0;
    demo_function_update_t update_function = default_update_function;
    demo_function_t cleanup_function = 0;
  };

  #define engine_demo (*OFFSETLESS(OFFSETLESS(menu, fan::graphics::engine_t, settings_menu), engine_demo_t, engine))

  inline static auto demos = std::to_array({
    demo_t{.name = "Capsule", .init_function = demo_shapes_init_capsule, .update_function = default_update_function, .cleanup_function = nullptr},
    demo_t{.name = "Circle", .init_function = demo_shapes_init_circle, .update_function = default_update_function, .cleanup_function = nullptr},
    demo_t{.name = "Gradient", .init_function = demo_shapes_init_gradient, .update_function = default_update_function, .cleanup_function = nullptr},
    demo_t{.name = "Grid", .init_function = demo_shapes_init_grid, .update_function = default_update_function, .cleanup_function = nullptr},
    demo_t{.name = "Image Decoder", .init_function = demo_shapes_init_universal_image_renderer, .update_function = default_update_function, .cleanup_function = nullptr},
    demo_t{.name = "Light", .init_function = demo_shapes_init_lighting, .update_function = demo_lighting_update, .cleanup_function = nullptr},
    demo_t{.name = "Particles", .init_function = demo_shapes_init_particles, .update_function = default_update_function, .cleanup_function = nullptr},
    demo_t{.name = "Polygon", .init_function = demo_shapes_init_polygon, .update_function = default_update_function, .cleanup_function = nullptr},
    demo_t{.name = "Rectangle", .init_function = demo_shapes_init_rectangle, .update_function = default_update_function, .cleanup_function = nullptr},
    demo_t{.name = "Shader", .init_function = demo_shapes_init_shader_shape, .update_function = demo_shader_shape_update, .cleanup_function = nullptr},
    demo_t{.name = "Sprite", .init_function = demo_shapes_init_sprite, .update_function = default_update_function, .cleanup_function = nullptr},
    demo_t{.name = "_next", .init_function = nullptr, .update_function = default_update_function, .cleanup_function = nullptr}, // skip to next title
    //GUI STUFF
    demo_t{.name = "Live Shader Editor", .init_function = demo_shapes_init_shader_live_editor, .update_function = demo_shader_live_editor_update, .cleanup_function = demo_shader_live_editor_cleanup},
    demo_t{.name = "_next", .init_function = nullptr, .update_function = default_update_function, .cleanup_function = nullptr}, // skip to next title
    demo_t{.name = "Reflective Mirrors", .init_function = demo_physics_init_mirrors, .update_function = demo_physics_update_mirrors, .cleanup_function = demo_physics_cleanup_mirrors},
    demo_t{.name = "_next", .init_function = nullptr, .update_function = default_update_function, .cleanup_function = nullptr}, // skip to next title
    demo_t{.name = "Multithreaded image loading", .init_function = demo_init_multithreaded_image_loading, .update_function = demo_update_multithreaded_image_loading, .cleanup_function = demo_cleanup_multithreaded_image_loading},
  });

  static void menus_engine_demo_left(menu_t* menu, const fan::vec2& next_window_position, const fan::vec2& next_window_size) {
    gui::set_next_window_pos(next_window_position);
    gui::set_next_window_size(next_window_size);
    fan_graphics_gui_window("##Menu Engine Demo Left", 0, wnd_flags){
      render_demos(menu, {
        "SHAPES",
        "GUI",
        "PHYSICS",
        "MULTITHREADING"
      });
    }
  }
  static void menus_engine_demo_right(menu_t* menu, const fan::vec2& next_window_position, const fan::vec2& next_window_size) {
    gui::set_next_window_pos(next_window_position);
    gui::set_next_window_size(fan::vec2(next_window_size.x, next_window_size.y * engine_demo.right_window_split_ratio));
    fan::vec2 window_size;
    fan_graphics_gui_window("##Menu Engine Demo Right Top", 0, wnd_flags) {
      engine_demo.engine.lighting.ambient = 1;
      auto& shape_info = demos[engine_demo.current_demo];
      if (shape_info.update_function) {
        shape_info.update_function(&engine_demo);
      }
      window_size = gui::get_window_size();
    }

    gui::set_next_window_pos(next_window_position + fan::vec2(0, window_size.y));
    gui::set_next_window_size(fan::vec2(next_window_size.x, next_window_size.y - window_size.y));
    gui::push_style_color(gui::col_window_bg, fan::colors::transparent);
    fan_graphics_gui_window("##Menu Engine Demo Right Content Bottom", 0, wnd_flags | gui::window_flags_no_inputs) {
      gui::set_viewport(engine_demo.right_column_view.viewport);
      fan::vec2 viewport_size = engine_demo.engine.viewport_get(engine_demo.right_column_view.viewport).viewport_size;
      engine_demo.engine.camera_set_ortho(
        engine_demo.right_column_view.camera,
        fan::vec2(0, viewport_size.x),
        fan::vec2(0, viewport_size.y)
      );
    }
    gui::pop_style_color();
  }


  static constexpr int wnd_flags = gui::window_flags_no_move| 
    gui::window_flags_no_collapse | gui::window_flags_no_resize| gui::window_flags_no_title_bar;

  static void render_demos(menu_t* menu, const std::vector<std::string>& titles) {
    std::size_t title_index = 0;
    std::string title = titles[title_index];
    gui::text(title, fan::color::from_rgba(0x948c80ff) * 1.5);
    gui::push_style_var(gui::style_var_cell_padding, fan::vec2(0));
    fan_graphics_gui_table(
      (title + "_settings_left_table_display").c_str(), 
      1, 
      gui::table_flags_borders_inner_h | gui::table_flags_borders_outer_h
    ) {
      {
        gui::push_style_var(gui::style_var_selectable_text_align, fan::vec2(0, 0.5));
        for (auto [i, shape_info] : fan::enumerate(demos)) {
          if (shape_info.name == "_next") {
            ++title_index;
            title = titles[title_index];
            gui::end_table();
            gui::new_line();
            gui::new_line();
            gui::text(title, fan::color::from_rgba(0x948c80ff) * 1.5);
            gui::begin_table(title + "_settings_left_table_display", 1,
              gui::table_flags_borders_inner_h |
              gui::table_flags_borders_outer_h
            );
            continue;
          }
          gui::table_next_row();
          gui::table_next_column();
          f32_t row_height = gui::get_text_line_height_with_spacing() * 2;
          if (gui::selectable(shape_info.name, false, 0, fan::vec2(0.0f, row_height))) {
            if (demos[engine_demo.current_demo].cleanup_function) {
              demos[engine_demo.current_demo].cleanup_function(&engine_demo);
            }
            engine_demo.shapes.clear();
            shape_info.init_function(&engine_demo);
            engine_demo.current_demo = i;
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
      auto& shape_info = demos[engine_demo.current_demo];
      if (shape_info.init_function) {
        shape_info.init_function(&engine_demo);
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
      page.page_left_render = menus_engine_demo_left;
      page.page_right_render = menus_engine_demo_right;
      engine.settings_menu.pages.emplace_front(page);
    }
    right_column_view.camera = engine.camera_create();
    right_column_view.viewport = engine.viewport_create();
  }

  void update() {
    engine.render_settings_menu = true;
    if (engine.settings_menu.current_page != 0) {
      shapes.clear();
    }
  }

  fan::graphics::render_view_t right_column_view;
  uint8_t current_demo = 0;
  int shape_count = 1;
  std::vector<engine_t::shape_t> shapes;
  f32_t right_window_split_ratio = 0.2f;
};

// Library usage includes
//-------------------------------------------------------------------
#define stage_loader_path .
#include <fan/graphics/gui/stage_maker/loader.h>
//-------------------------------------------------------------------


// Library usage samples, these structs expect there is an instance of a fan::graphics::engine_t.
// They use "gloco" to access global engine pointer
struct library_usage_t {
  struct gui {
    //-------------------------------------------------------------------
    // Demonstrates GUI console print
    struct console_print {
      console_print() {
        fan::printcl("Random number on construct(0-5):" + fan::random::value(0, 5));
      }
    };
    //-------------------------------------------------------------------
    // Demonstrates custom GUI console commands
    struct console {
      console() {
        gloco->console.commands.add("test", [&](const fan::commands_t::arg_t& args) {
          fan::printcl("test command executed");
        });
        gloco->console.commands.add("sum", [&](const fan::commands_t::arg_t& args) {
          if (args.size() == 2) {
            int a = std::stoi(args[0]);
            int b = std::stoi(args[1]);
            fan::printcl("Result: " + std::to_string(a + b));
          }
        });
      }
      void update() {

      }
      void close() {
        gloco->console.commands.remove("test");
        gloco->console.commands.remove("sum");
      }
    };
    //-------------------------------------------------------------------
  };
  struct input {
    //-------------------------------------------------------------------
    // Demonstrates input action system with key bindings and combos
    struct action {
      action() {
        gloco->input_action.add({ fan::key_space }, "jump");
        gloco->input_action.add_keycombo({ fan::key_space, fan::key_a }, "combo_test");
      }
      void update() {
        if (gloco->input_action.is_active("jump")) {
          fan::print("jump");
        }
        if (gloco->input_action.is_active("combo_test")) {
          fan::print("combo_test");
        }
      }
      void close() {
        gloco->input_action.remove("jump");
        gloco->input_action.remove("combo_test");
      }
    };
    //-------------------------------------------------------------------
    //-------------------------------------------------------------------
  // Demonstrates interactive visual elements
    struct vfi_interactive {
      fan::graphics::rectangle_t rectangle;
      fan::graphics::vfi_root_t vfi_root;

      vfi_interactive() {
        fan::graphics::vfi_t::properties_t vfip;
        vfip.shape_type = fan::graphics::vfi_t::shape_t::rectangle;
        vfip.shape.rectangle->position = fan::vec3(500, 500, 1);
        vfip.shape.rectangle->size = fan::vec2(100, 100);
        vfip.shape.rectangle->size.x /= 2; // hitbox takes half size
        vfip.shape.rectangle->camera = gloco->orthographic_render_view.camera;
        vfip.shape.rectangle->viewport = gloco->orthographic_render_view.viewport;

        vfip.mouse_button_cb = [](const fan::graphics::vfi_t::mouse_button_data_t& data) -> int {
          fan::print("Rectangle clicked.");
          return 0;
        };

        rectangle = fan::graphics::rectangle_t{ {
          .position = fan::vec3(*(fan::vec2*)&vfip.shape.rectangle->position, 0),
          .size = vfip.shape.rectangle->size,
          .color = fan::colors::red
        } };

        vfi_root.set_root(vfip);
      }

      void update() {
        fan::graphics::gui::text("Click the red rectangle");
      }
    };
    //-------------------------------------------------------------------
  };
  struct event {
    //-------------------------------------------------------------------
    // Demonstrates coroutines using fan::event::task_t by respawning rectangles every 1s
    struct task {
      fan::event::task_t spawn_rectangles() {
        while (true) {
          shapes.clear();
          shapes.push_back(fan::graphics::rectangle_t{ {
              .position = fan::vec3(fan::random::vec2(0, 400), 0),
              .size = fan::random::vec2(100, 400),
              .color = fan::random::color()
          } });
          co_await fan::co_sleep(1000); // async sleep - doesn't block main thread
        }
      }
      task() {
        task_spawn_rectangles = spawn_rectangles(); // Start the coroutine
      }
      fan::event::task_t task_spawn_rectangles;
      std::vector<fan::graphics::shape_t> shapes;
    };
    //-------------------------------------------------------------------

    //-------------------------------------------------------------------
  };
  struct io {
    //-------------------------------------------------------------------
  // Demonstrates shape serialization to JSON
    struct shape_serialization {
      shape_serialization() {
        // Test shapes
        for (int i = 0; i < 10; ++i) {
          shapes.push_back(fan::graphics::rectangle_t{ {
            .position = fan::random::vec2(0, 400),
            .size = fan::random::vec2(20, 100),
            .color = fan::random::color()
          } });
        }
      }
      void update() {
        if (fan::graphics::gui::button("Serialize to JSON")) {
          fan::json out;
          for (auto& shape : shapes) {
            out += shape;
          }
          fan::io::file::write("demo_shapes.json", out.dump(2), std::ios_base::binary);
          fan::print("JSON wrote to \"demo_shapes.json\"");
        }
      }
      std::vector<fan::graphics::rectangle_t> shapes;
    };
    //-------------------------------------------------------------------
    // Demonstrates file dialog for opening files
    struct file_dialog {
      void update() {
        if (fan::graphics::gui::button("Open File")) {
          dialog.load("png,jpg;pdf", &output_path);
        }
        if (dialog.is_finished()) {
          fan::print("Selected file: " + output_path);
          dialog.finished = false;
        }
        if (!output_path.empty()) {
          fan::graphics::gui::text("Last selected: " + output_path);
        }
      }
      std::string output_path;
      fan::graphics::file_open_dialog_t dialog;
    };
    // Demonstrates file system watching for changes
    struct file_watcher {
      file_watcher() : watcher("./") {
        watcher.start([this](const std::string& filename, int events) {
          last_event = "file: " + filename;
          if (events & fan::fs_change) {
            last_event += " (modified)";
          }
          if (events & fan::fs_rename) {
            last_event += " (renamed/deleted)";
          }
          fan::print("fs event: " + last_event);
          });
      }
      void update() {
        fan::graphics::gui::text("watching current directory for file changes");
        if (!last_event.empty()) {
          fan::graphics::gui::text("last event: " + last_event);
        }
        fan::graphics::gui::text("try creating/modifying files to see events");
      }
      fan::event::fs_watcher_t watcher;
      std::string last_event;
    };
    //-------------------------------------------------------------------
       //-------------------------------------------------------------------
    // Demonstrates async file operations with coroutines
    struct async_file {
      fan::event::task_t read_file_async(const std::string& path) {
        int fd = co_await fan::io::file::async_open(path);
        int offset = 0;
        std::string buffer;

        while (true) {
          intptr_t result = co_await fan::io::file::async_read(fd, &buffer, offset);
          if (result == 0) {
            break;
          }
          file_content += buffer;
          offset += result;
          co_await fan::co_sleep(10);
        }
        co_await fan::io::file::async_close(fd);
      }

      fan::event::task_t write_file_async(const std::string& path, const std::string& data) {
        int fd = co_await fan::io::file::async_open(path, fan::fs_out);
        size_t total_written = 0;
        size_t buffer_size = 4096;

        while (total_written < data.size()) {
          size_t remaining = data.size() - total_written;
          size_t to_write = std::min(remaining, buffer_size);
          std::string buffer(data.data() + total_written, to_write);
          size_t written = co_await fan::io::file::async_write(fd, buffer.data(), buffer.size(), total_written);
          total_written += written;
        }
        co_await fan::io::file::async_close(fd);
      }

      async_file() {
        task_handle = read_file_async("CMakeLists.txt");
      }

      void update() {
        if (fan::graphics::gui::button("Read file")) {
          file_content.clear();
          task_handle = read_file_async("CMakeLists.txt");
        }
        if (fan::graphics::gui::button("Write file")) {
          task_handle = write_file_async("output.txt", file_content);
        }
        if (!file_content.empty()) {
          fan::graphics::gui::text("Content: " + file_content.substr(0, 50) + "...");
        }
      }
      fan::event::task_t task_handle;
      std::string file_content;
    };
    //-------------------------------------------------------------------
    // Demonstrates async directory iteration with image loading and GUI display
    struct async_directory {
      async_directory() {
        iterator.sort_alphabetically = true;
        iterator.callback = [&](const std::filesystem::directory_entry& entry) -> fan::event::task_t {
          std::string path_str = entry.path().string();
          if (fan::image::valid(path_str)) {
            images.emplace_back(gloco->image_load(path_str));
            co_await fan::co_sleep(100);
          }
          co_return;
        };
        fan::io::async_directory_iterate(&iterator, "imagenet-sample-images-master");
      }
      void update() {
        fan::graphics::gui::begin("images");
        f32_t thumbnail_size = 128.0f;
        f32_t panel_width = fan::graphics::gui::get_content_region_avail().x;
        f32_t padding = 16.0f;
        int column_count = std::max((int)(panel_width / (thumbnail_size + padding)), 1);
        fan::graphics::gui::columns(column_count, 0, false);
        for (auto& i : images) {
          fan::graphics::gui::image(i, thumbnail_size);
          fan::graphics::gui::next_column();
        }
        fan::graphics::gui::end();
      }
      std::vector<fan::graphics::image_t> images;
      fan::io::async_directory_iterator_t iterator;
    };
    //-------------------------------------------------------------------
  };
  struct system {
    //-------------------------------------------------------------------
    // Demonstrates creating and loading custom stages
    struct custom_stage {
      // make custom stage
      lstd_defstruct(custom_t)
        #include <fan/graphics/gui/stage_maker/preset.h>
        static constexpr auto stage_name = "";
        fan::graphics::rectangle_t r;
        void open(void* sod) {
          fan::print("opened");
        }
        void close() {
          fan::print("closed");
        }
        void window_resize() {
          fan::print("resized");
        }
        void update() {
          //fan::print("update");
        }
      };
      stage_loader_t* stage_loader;
      stage_loader_t::nr_t stage_handle;
      custom_stage() {
        stage_loader = new stage_loader_t;
        stage_loader_t::stage_open_properties_t op;
        stage_handle = stage_loader_t::open_stage<custom_t>(op);
      }
      void close() {
        stage_loader->erase_stage(stage_handle);
        delete stage_loader;
      }
      void update() {
        fan::graphics::gui::text("Custom stage running");
      }
    };
  };
  //-------------------------------------------------------------------
};

int main() {
  engine_demo_t demo;
  demo.create_gui();

  fan_window_loop{
    demo.update();
  };
}