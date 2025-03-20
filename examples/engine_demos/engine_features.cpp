#include <fan/pch.h>

using namespace fan::graphics;
using menu_t = engine_t::settings_menu_t;

struct engine_demo_t {

  static void demo_rectangle(engine_demo_t* engine_demo) {
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
  struct shape_t {
    const char* name;
    demo_function_t demo_function;
  };

  inline static auto shape_infos = std::to_array({
    shape_t{.name="Capsule", .demo_function=0},
    shape_t{.name="Circle", .demo_function=0},
    shape_t{.name="Gradient", .demo_function=0},
    shape_t{.name="Grid", .demo_function=0},
    shape_t{.name="Image Renderer", .demo_function=0}, // can be used to decode specific formats straight in gpu
    shape_t{.name="Light", .demo_function=0},
    shape_t{.name="Particles", .demo_function=0},
    shape_t{.name="Polygon", .demo_function=0},
    shape_t{.name="Rectangle", .demo_function=demo_rectangle},
    shape_t{.name="Shader", .demo_function=0},
    shape_t{.name="Sprite", .demo_function=0}
  });

  #define engine_demo (*OFFSETLESS(OFFSETLESS(menu, fan::graphics::engine_t, settings_menu), engine_demo_t, engine))
  static void menus_engine_demo_left(menu_t* menu) {
    ImGui::PushStyleColor(ImGuiCol_ChildBg, fan::colors::transparent);
    ImGui::BeginChild("##root_left_column", ImVec2(0, 0), false, 0);
    {
      ImGui::TextColored(fan::color::hex(0x948c80ff) * 1.5, "Shapes");
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
    if (ImGui::DragInt("Shape count", &engine_demo.shape_count, 1, 0, std::numeric_limits<int>::max())) {
      auto& shape_info = shape_infos[engine_demo.current_demo];
      if (shape_info.demo_function) {
        shape_info.demo_function(&engine_demo);
      }
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
  fan::color child_bg_left = 0, child_bg_right = 0;
};

int main() {

  engine_demo_t demo;

  demo.create_gui();
  fan_window_loop{
    demo.update();
  };
}