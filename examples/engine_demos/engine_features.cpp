#include <fan/pch.h>

using namespace fan::graphics;
using menu_t = engine_t::settings_menu_t;

struct engine_demo_t {

  static void demo_static_lighting(engine_demo_t* engine_demo) {
    engine_demo->shapes.clear();

    fan::vec2 viewport_size = engine_demo->engine.viewport_get(engine_demo->right_column_view.viewport).viewport_size;
    
    static auto image_background = engine_demo->engine.image_create(fan::color(0, 0, 0, 1));
    static auto image_tire = engine_demo->engine.image_load("images/tire.webp");
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
      .image = image_tire
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
    engine_demo->shapes.clear();
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
    for (auto& i : engine_demo->shapes) {
      i.erase();
    }
    engine_demo->shapes.resize(engine_demo->shape_count);
    fan::vec2 viewport_size = engine_demo->engine.viewport_get(engine_demo->right_column_view.viewport).viewport_size;

    for (uint32_t i = 0; i < engine_demo->shape_count; ++i) {
      uint32_t sides = fan::random::value(3u, 12u);
      sides = std::max(3u, sides);

      fan::vec2 position = fan::random::vec2(0, viewport_size);
      float radius = fan::random::value(50.0f, 150.0f);
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
    for (auto& i : engine_demo->shapes) {
      i.erase();
    }
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

  typedef void(*demo_function_t)(engine_demo_t*);
  typedef void(*demo_function_update_t)(engine_demo_t*);
  struct shape_t {
    const char* name;
    demo_function_t demo_function = 0;
    demo_function_update_t update_function = 0;
  };

  inline static auto shape_infos = std::to_array({
    shape_t{.name="Capsule", .demo_function=0},
    shape_t{.name="Circle", .demo_function=0},
    shape_t{.name="Gradient", .demo_function=0},
    shape_t{.name="Grid", .demo_function=0},
    shape_t{.name="Image Renderer", .demo_function=0}, // can be used to decode specific formats straight in gpu
    shape_t{.name="Light", .demo_function=demo_static_lighting, .update_function=demo_static_lighting_update},
    shape_t{.name="Particles", .demo_function=demo_static_particles},
    shape_t{.name="Polygon", .demo_function=demo_static_polygon},
    shape_t{.name="Rectangle", .demo_function=demo_static_rectangle},
    shape_t{.name="Shader", .demo_function=0},
    shape_t{.name="Sprite", .demo_function=0}
  });

  #define engine_demo (*OFFSETLESS(OFFSETLESS(menu, fan::graphics::engine_t, settings_menu), engine_demo_t, engine))
  static void menus_engine_demo_left(menu_t* menu) {
    ImGui::PushStyleColor(ImGuiCol_ChildBg, fan::colors::transparent);
    ImGui::BeginChild("##root_left_column", ImVec2(0, 0), false, 0);
    {
      ImGui::TextColored(fan::color::hex(0x948c80ff) * 1.5, "SHAPES");
      ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(0.0f, 0.0f));
      ImGui::BeginTable("settings_left_table_display", 1,
        ImGuiTableFlags_BordersInnerH |
        ImGuiTableFlags_BordersOuterH
      );
      {
        {
          ImGui::PushStyleVar(ImGuiStyleVar_SelectableTextAlign, fan::vec2(0.00, 0.5));
          for (auto [i, shape_info] : fan::enumerate(shape_infos)) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            f32_t row_height = ImGui::GetTextLineHeightWithSpacing() * 2;
            if (ImGui::Selectable(shape_info.name, false, 0, ImVec2(0.0f, row_height))) {
              shape_info.demo_function(&engine_demo);
              engine_demo.current_demo = i;
            }
          }
          ImGui::PopStyleVar();
        }
      }
      ImGui::EndTable();
      ImGui::PopStyleVar();
    }
    ImGui::NewLine();
    ImGui::NewLine();
    {
      ImGui::TextColored(fan::color::hex(0x948c80ff) * 1.5, "POST PROCESSING");
      ImGui::BeginTable("settings_left_table_post_processing", 2,
        ImGuiTableFlags_BordersInnerH |
        ImGuiTableFlags_BordersOuterH
      );
    }
    ImGui::EndTable();

    ImGui::EndChild();
    ImGui::PopStyleColor();
  }

  static void menus_engine_demo_right(menu_t* menu) {
    engine_demo.engine.lighting.ambient = 1;
    auto& shape_info = shape_infos[engine_demo.current_demo];
    if (ImGui::DragInt("Shape count", &engine_demo.shape_count, 1, 0, std::numeric_limits<int>::max())) {
      if (shape_info.demo_function) {
        shape_info.demo_function(&engine_demo);
      }
    }
    if (shape_info.update_function) {
      shape_info.update_function(&engine_demo);
    }
    ImGui::PushStyleColor(ImGuiCol_ChildBg, fan::color(0, 0, 0, 1));
    ImGui::BeginChild("##root_right_column", ImVec2(0, 0), false, 0);
    engine_demo.engine.set_imgui_viewport(engine_demo.right_column_view.viewport);
    fan::vec2 viewport_size = engine_demo.engine.viewport_get(engine_demo.right_column_view.viewport).viewport_size;
    engine_demo.engine.camera_set_ortho(
      engine_demo.right_column_view.camera,
      fan::vec2(0, viewport_size.x),
      fan::vec2(0, viewport_size.y)
    );
    //loco_t::settings_menu_t::menu_graphics_right(menu);
    ImGui::EndChild();
    ImGui::PopStyleColor();
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
    if (engine.settings_menu.current_page != 0) {
      shapes.clear();
    }
  }

  fan::graphics::camera_t right_column_view;
  engine_t engine{{
    .render_shapes_top = 1,
    .renderer=0,
  }};
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