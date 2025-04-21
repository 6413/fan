#include <fan/pch.h>

using namespace fan::graphics;
using menu_t = engine_t::settings_menu_t;

struct engine_demo_t {
  engine_t engine{{ // initialize before everything
    .renderer=engine_t::renderer_t::opengl,
  }};

  // ------------------------STATIC------------------------
  static void demo_static_capsule(engine_demo_t* engine_demo) {
    engine_demo->shapes.resize(engine_demo->shape_count);
    fan::vec2 viewport_size = engine_demo->engine.viewport_get(engine_demo->right_column_view.viewport).viewport_size;
    for (uint32_t i = 0; i < engine_demo->shape_count; ++i) {
      engine_demo->shapes[i] = fan::graphics::capsule_t{ {
        .camera = &engine_demo->right_column_view,
        .position = fan::random::vec2(0, viewport_size),
        .center0 = 0,
        .center1 = fan::vec2(0, fan::random::value(10.f, 256.f)),
        .radius = fan::random::value(16.f, 64.f),
        .color = fan::random::color()
      }};
    }
  }
  static void demo_static_circle(engine_demo_t* engine_demo) {
    engine_demo->shapes.resize(engine_demo->shape_count);
    fan::vec2 viewport_size = engine_demo->engine.viewport_get(engine_demo->right_column_view.viewport).viewport_size;
    for (uint32_t i = 0; i < engine_demo->shape_count; ++i) {
      engine_demo->shapes[i] = fan::graphics::circle_t{ {
        .camera = &engine_demo->right_column_view,
        .position = fan::random::vec2(0, viewport_size),
        .radius = fan::random::value(16.f, 64.f),
        .color = fan::random::color()
      }};
    }
  }
  static void demo_static_gradient(engine_demo_t* engine_demo) {
    for (auto& i : engine_demo->shapes) {
      i.erase();
    }
    engine_demo->shapes.resize(engine_demo->shape_count);
    fan::vec2 viewport_size = engine_demo->engine.viewport_get(engine_demo->right_column_view.viewport).viewport_size;
    for (uint32_t i = 0; i < engine_demo->shape_count; ++i) {
      engine_demo->shapes[i] = fan::graphics::gradient_t{ {
        .camera = &engine_demo->right_column_view,
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

  static void demo_static_grid(engine_demo_t* engine_demo) {
    fan::vec2 viewport_size = engine_demo->engine.viewport_get(engine_demo->right_column_view.viewport).viewport_size;
    
    engine_demo->shapes.emplace_back(fan::graphics::grid_t{{
    .camera = &engine_demo->right_column_view,
    .position = fan::vec3(-8, -8, 0),
    .size = viewport_size.max(),
    .grid_size = 32,
    .color = fan::colors::white
    }});
  }

  static void demo_static_universal_image_renderer(engine_demo_t* engine_demo) {
    fan::vec2 viewport_size = engine_demo->engine.viewport_get(engine_demo->right_column_view.viewport).viewport_size;

    engine_demo->shapes.emplace_back(fan::graphics::universal_image_renderer_t{{
      .camera = &engine_demo->right_column_view,
      .position = fan::vec3(viewport_size / 2, 0),    
      .size = viewport_size / 2,
    }});
    fan::string pixel_data_str;
    constexpr fan::vec2ui image_size = fan::vec2ui(1920, 1080);
    fan::io::file::read("images/output1920.yuv", &pixel_data_str);
    void* pixel_data = pixel_data_str.data();
    void* datas[3];
    uint64_t offset = 0;
    datas[0] = pixel_data;
    datas[1] = (uint8_t*)pixel_data + (offset += image_size.multiply());
    datas[2] = (uint8_t*)pixel_data + (offset += image_size.multiply() / 4);
    engine_demo->shapes.back().reload(fan::pixel_format::yuv420p, datas, image_size);
  }

  loco_t::image_t image_tire = engine.image_load("images/tire.webp");
  static void demo_static_lighting(engine_demo_t* engine_demo) {
    

    fan::vec2 viewport_size = engine_demo->engine.viewport_get(engine_demo->right_column_view.viewport).viewport_size;
    
    static auto image_background = engine_demo->engine.image_create(fan::color(0, 0, 0, 1));
    // bg
    engine_demo->shapes.emplace_back(fan::graphics::sprite_t{{
      .camera = &engine_demo->right_column_view,
      .position = fan::vec3(viewport_size / 2, 0),
      .size = viewport_size/2,
      .image = image_background
    }});

    engine_demo->shapes.emplace_back(fan::graphics::sprite_t{{
      .camera = &engine_demo->right_column_view,
      .position = fan::vec3(viewport_size / 2, 1),
      .size = viewport_size.min()/6,
      .image = engine_demo->image_tire
    }});

    engine_demo->shapes.emplace_back(fan::graphics::light_t{ {
      .camera = &engine_demo->right_column_view,
      .position =  fan::vec3(fan::vec2(viewport_size.x / 3, viewport_size.y / 3), 0),
      .size = viewport_size.min() / 4,
      .color = fan::colors::red
    }});
    engine_demo->shapes.emplace_back(fan::graphics::light_t{ {
      .camera = &engine_demo->right_column_view,
      .position =  fan::vec3(fan::vec2(viewport_size.x / 1.5, viewport_size.y / 3), 0),
      .size = viewport_size.min() / 3,
      .color = fan::colors::green
    }});
    engine_demo->shapes.emplace_back(fan::graphics::light_t{ {
      .camera = &engine_demo->right_column_view,
      .position =  fan::vec3(fan::vec2(viewport_size.x / 2, viewport_size.y / 1.5), 0),
      .size = viewport_size.min() / 3,
      .color = fan::colors::blue
    }});
    engine_demo->shapes.emplace_back(fan::graphics::light_t{ {
      .camera = &engine_demo->right_column_view,
      .position =  fan::vec3(fan::vec2(viewport_size.x / 2, viewport_size.y / 1.5), 0),
      .size = viewport_size.min() / 3,
      .color = fan::colors::purple
    }});
  }
  static void demo_static_lighting_update(engine_demo_t* engine_demo) {
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

  static void demo_static_particles(engine_demo_t* engine_demo) {
    
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

  static void demo_static_polygon(engine_demo_t* engine_demo) {
    engine_demo->shapes.resize(engine_demo->shape_count);
    fan::vec2 viewport_size = engine_demo->engine.viewport_get(engine_demo->right_column_view.viewport).viewport_size;

    for (uint32_t i = 0; i < engine_demo->shape_count; ++i) {
      uint32_t sides = fan::random::value(3u, 12u);
      sides = std::max(3u, sides);

      fan::vec2 position = fan::random::vec2(0, viewport_size);
      float radius = fan::random::value(50.f, 200.f);
      fan::color color = fan::random::color();

      loco_t::polygon_t::properties_t pp;
      pp.vertices.clear();

      std::vector<fan::vec2> polygon_points;
      polygon_points.reserve(sides);

      float angle_step = 2.0f * fan::math::pi / sides;
      for (uint32_t j = 0; j < sides; ++j) {
        float angle = j * angle_step;
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
          .camera = &engine_demo->right_column_view,
          .vertices = pp.vertices
        }
      };
    }
  }
  static void demo_static_rectangle(engine_demo_t* engine_demo) {
    
    engine_demo->shapes.resize(engine_demo->shape_count);
    fan::vec2 viewport_size = engine_demo->engine.viewport_get(engine_demo->right_column_view.viewport).viewport_size;
    for (uint32_t i = 0; i < engine_demo->shape_count; ++i) {
      engine_demo->shapes[i] = fan::graphics::rectangle_t{ {
        .camera = &engine_demo->right_column_view,
        .position = fan::random::vec2(0, viewport_size),
        .size = fan::random::vec2(30, 200),
        .color = fan::random::color()
      }};
    }
  }

  inline static const char* demo_static_shader_shape_fragment_shader = R"(#version 330
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

  loco_t::shader_t demo_static_shader_shape_shader{engine.get_sprite_vertex_shader(demo_static_shader_shape_fragment_shader)};
  fan::color custom_color = fan::colors::red;
  static void demo_static_shader_shape(engine_demo_t* engine_demo) {
    if (engine_demo->demo_static_shader_shape_shader.iic()) {
      fan::throw_error("failed to compile custom shader");
    }
    
    fan::vec2 viewport_size = engine_demo->engine.viewport_get(engine_demo->right_column_view.viewport).viewport_size;

    loco_t::image_t image = engine_demo->engine.image_load("images/lava_seamless.webp");
    engine_demo->shapes.emplace_back(fan::graphics::shader_shape_t{{
      .camera = &engine_demo->right_column_view,
      .position = fan::vec3(viewport_size / 2, 3),
      .size = viewport_size / 2,
      .shader = engine_demo->demo_static_shader_shape_shader,
      .image = image,
    }});
    // init
    engine_demo->engine.shader_set_value(engine_demo->demo_static_shader_shape_shader, "custom_color", engine_demo->custom_color);
  }
  static void demo_static_shader_shape_update(engine_demo_t* engine_demo) {
    if (fan::graphics::gui::color_edit4("##c0", &engine_demo->custom_color)) {
      engine_demo->engine.shader_set_value(engine_demo->demo_static_shader_shape_shader, "custom_color", engine_demo->custom_color);
    }
  }

  static void demo_static_sprite(engine_demo_t* engine_demo) {
    engine_demo->shapes.resize(engine_demo->shape_count);
    fan::vec2 viewport_size = engine_demo->engine.viewport_get(engine_demo->right_column_view.viewport).viewport_size;
    for (uint32_t i = 0; i < engine_demo->shape_count; ++i) {
      engine_demo->shapes[i] = fan::graphics::sprite_t{ {
        .camera = &engine_demo->right_column_view,
        .position = fan::random::vec2(0, viewport_size),
        .size = fan::random::value(30.f, 200.f),
        .color = fan::random::bright_color(), // add tint to the image
        .image = engine_demo->image_tire
      }};
    }
  }
  // ------------------------STATIC------------------------

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
      .camera = &engine_demo->right_column_view,
      .size = 4,
      .color = fan::colors::red
    } });
    engine_demo->demo_physics_mirrors_data->rays.resize(engine_demo->demo_physics_mirrors_data->reflect_depth + 1, { {
      .camera = &engine_demo->right_column_view,
      .src = {0, 0, 0xfff},
      .color = fan::colors::green
    } });
  }
  static void demo_physics_mirrors(engine_demo_t* engine_demo) {
    fan::vec2 viewport_size = engine_demo->engine.viewport_get(engine_demo->right_column_view.viewport).viewport_size;
    engine_demo->demo_physics_mirrors_data = new demo_physics_mirrors_t();
    static std::vector<vertex_t> triangle_vertices{
     {fan::vec2(400, 400), fan::colors::orange},
     {fan::vec2(400, 600), fan::colors::orange},
     {fan::vec2(700, 600), fan::colors::orange},
    };
    engine_demo->demo_physics_mirrors_data->triangle = fan::graphics::physics::polygon_strip_t{ {
      .camera = &engine_demo->right_column_view,
      .vertices = triangle_vertices
    } };
    for (std::size_t i = 0; i < 5; ++i) {
      engine_demo->demo_physics_mirrors_data->circles.push_back({ {
        .camera = &engine_demo->right_column_view,
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
        .camera = &engine_demo->right_column_view,
        .src = {0, 500, 0xfff}, 
        .color = fan::colors::white
      } };
    on_reflect_depth_resize(engine_demo);
  }
  static void demo_physics_mirrors_update(engine_demo_t* engine_demo) {
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
      d.set_position(-1000);
      engine_demo->demo_physics_mirrors_data->rays[i].set_line(0, 0);
    }

    int depth = 0;
    fan::vec2 current_src = src;
    fan::vec2 current_dst = dst;

    while (depth < engine_demo->demo_physics_mirrors_data->reflect_depth + 1) {
      if (auto result = fan::physics::raycast(current_src, current_dst)) {
        engine_demo->demo_physics_mirrors_data->ray_hit_point[depth].set_position(result.point);

        fan::vec2 direction = (current_dst - current_src).normalize();
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
  static void demo_physics_mirrors_cleanup(engine_demo_t* engine_demo) {
    delete engine_demo->demo_physics_mirrors_data;
  }

  // ------------------------PHYSICS------------------------

  typedef void(*demo_function_t)(engine_demo_t*);
  typedef void(*demo_function_update_t)(engine_demo_t*);

  static void default_update_function(engine_demo_t* engine_demo) {
    menus_engine_demo_render_element_count(&engine_demo->engine.settings_menu);
  }

  struct demo_t {
    const char* name;
    demo_function_t demo_function = 0;
    demo_function_update_t update_function = default_update_function;
    demo_function_t cleanup_function = 0;
  };

  inline static auto demos = std::to_array({
    demo_t{.name = "Capsule", .demo_function = demo_static_capsule, .update_function = default_update_function, .cleanup_function = nullptr},
    demo_t{.name = "Circle", .demo_function = demo_static_circle, .update_function = default_update_function, .cleanup_function = nullptr},
    demo_t{.name = "Gradient", .demo_function = demo_static_gradient, .update_function = default_update_function, .cleanup_function = nullptr},
    demo_t{.name = "Grid", .demo_function = demo_static_grid, .update_function = default_update_function, .cleanup_function = nullptr},
    demo_t{.name = "Image Decoder", .demo_function = demo_static_universal_image_renderer, .update_function = default_update_function, .cleanup_function = nullptr},
    demo_t{.name = "Light", .demo_function = demo_static_lighting, .update_function = demo_static_lighting_update, .cleanup_function = nullptr},
    demo_t{.name = "Particles", .demo_function = demo_static_particles, .update_function = default_update_function, .cleanup_function = nullptr},
    demo_t{.name = "Polygon", .demo_function = demo_static_polygon, .update_function = default_update_function, .cleanup_function = nullptr},
    demo_t{.name = "Rectangle", .demo_function = demo_static_rectangle, .update_function = default_update_function, .cleanup_function = nullptr},
    demo_t{.name = "Shader", .demo_function = demo_static_shader_shape, .update_function = demo_static_shader_shape_update, .cleanup_function = nullptr},
    demo_t{.name = "Sprite", .demo_function = demo_static_sprite, .update_function = default_update_function, .cleanup_function = nullptr},
    demo_t{.name = "_next", .demo_function = nullptr, .update_function = default_update_function, .cleanup_function = nullptr}, // skip to next title
    demo_t{.name = "Reflective Mirrors", .demo_function = demo_physics_mirrors, .update_function = demo_physics_mirrors_update, .cleanup_function = demo_physics_mirrors_cleanup},
  });


  static constexpr int wnd_flags = gui::window_flags_no_move| 
    gui::window_flags_no_collapse | gui::window_flags_no_resize| gui::window_flags_no_title_bar;
  #define engine_demo (*OFFSETLESS(OFFSETLESS(menu, fan::graphics::engine_t, settings_menu), engine_demo_t, engine))

  static void render_demos(menu_t* menu, const std::vector<std::string>& titles) {
    std::size_t title_index = 0;
    std::string title = titles[title_index];
    gui::text(title, fan::color::hex(0x948c80ff) * 1.5);
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
            gui::text(title, fan::color::hex(0x948c80ff) * 1.5);
            gui::begin_table(title + "_settings_left_table_display", 1,
              gui::table_flags_borders_inner_h |
              gui::table_flags_borders_outer_h
            );
            continue;
          }
          gui::table_next_row();
          gui::table_next_column();
          f32_t row_height = gui::get_text_line_height_with_spacing() * 2;
          if (gui::selectable(shape_info.name, false, 0, ImVec2(0.0f, row_height))) {
            if (demos[engine_demo.current_demo].cleanup_function) {
              demos[engine_demo.current_demo].cleanup_function(&engine_demo);
            }
            engine_demo.shapes.clear();
            shape_info.demo_function(&engine_demo);
            engine_demo.current_demo = i;
          }
        }
        gui::pop_style_var();
      }
    }
    gui::pop_style_var();
#undef make_table
  }

  static void menus_engine_demo_left(menu_t* menu, const fan::vec2& next_window_position, const fan::vec2& next_window_size) {
    gui::set_next_window_pos(next_window_position);
    gui::set_next_window_size(next_window_size);
    fan_graphics_gui_window("##Menu Engine Demo Left", 0, wnd_flags){
      render_demos(menu, {
        "BASIC ENGINE SHAPES",
        "PHYSICS"
      });
    }
  }
  static void menus_engine_demo_right(menu_t* menu, const fan::vec2& next_window_position, const fan::vec2& next_window_size) {
    gui::set_next_window_pos(next_window_position);
    gui::set_next_window_size(fan::vec2(next_window_size.x, next_window_size.y / 5));
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
      gui::set_imgui_viewport(engine_demo.right_column_view.viewport);
      fan::vec2 viewport_size = engine_demo.engine.viewport_get(engine_demo.right_column_view.viewport).viewport_size;
      engine_demo.engine.camera_set_ortho(
        engine_demo.right_column_view.camera,
        fan::vec2(0, viewport_size.x),
        fan::vec2(0, viewport_size.y)
      );
    }
    gui::pop_style_color();
  }

  static void menus_engine_demo_render_element_count(menu_t* menu) {
    if (gui::drag_int("Shape count", &engine_demo.shape_count, 1, 0, std::numeric_limits<int>::max())) {
      auto& shape_info = demos[engine_demo.current_demo];
      if (shape_info.demo_function) {
        shape_info.demo_function(&engine_demo);
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

  fan::graphics::camera_t right_column_view;
  uint8_t current_demo = 0;
  int shape_count = 10;
  std::vector<engine_t::shape_t> shapes;
};

int main() {
  engine_demo_t demo;
  demo.create_gui();

  fan_window_loop{
    demo.update();
  };
}