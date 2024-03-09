#include fan_pch

int main() {
  loco_t loco;

  loco_t::texture_packe0::open_properties_t open_properties;
  loco_t::texture_packe0 e;
  e.open(open_properties);
  loco_t::texture_packe0::texture_properties_t texture_properties;
  texture_properties.visual_output = loco_t::image_t::sampler_address_mode::clamp_to_edge;
  texture_properties.min_filter = loco_t::image_t::filter::nearest;
  texture_properties.mag_filter = loco_t::image_t::filter::nearest;

  struct image_t {
    fan::vec2 uv_pos;
    fan::vec2 uv_size;
    loco_t::image_t image;
  };

  loco_t::image_t root_image;

  std::vector<std::vector<image_t>> images;

  fan::graphics::imgui_element_t element([&] {
    auto& style = ImGui::GetStyle();
    ImVec4* colors = style.Colors;
    const ImVec4 bgColor = ImVec4(0.1, 0.1, 0.1, 0.1);
    colors[ImGuiCol_WindowBg].w = bgColor.w;
    colors[ImGuiCol_ChildBg].w = bgColor.w;
    colors[ImGuiCol_TitleBg].w = bgColor.w;

    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_DockingEmptyBg, ImVec4(0, 0, 0, 0));
    ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());
    ImGui::PopStyleColor(2);
    ImGui::Begin("Editor");
    fan::vec2 window_size = gloco->window.get_size();
    fan::vec2 viewport_size = ImGui::GetContentRegionAvail();
    fan::vec2 viewport_pos = fan::vec2(ImGui::GetWindowPos() + fan::vec2(0, ImGui::GetFontSize() + ImGui::GetStyle().FramePadding.y * 2));
    fan::vec2 offset = viewport_size - viewport_size;
    fan::vec2 s = viewport_size;
    gloco->default_camera->camera.set_ortho(
      fan::vec2(-s.x, s.x),
      fan::vec2(-s.y, s.y)
    );
    gloco->default_camera->viewport.set(viewport_pos, viewport_size, window_size);

    static fan::string image_path;
    image_path.resize(40);

    static fan::vec2f cell_size = 1;
    if (ImGui::DragFloat2("cell size", cell_size.data(), 1, 1, 4096)) {
      images.clear();
      fan::vec2i divider = root_image.size / cell_size;
      fan::vec2 uv_size = root_image.size / divider / root_image.size;
      images.resize(divider.y);
      for (int i = 0; i < divider.y; ++i) {
        images[i].resize(divider.x);
        for (int j = 0; j < divider.x; ++j) {
          images[i][j] = image_t{
            .uv_pos = uv_size * fan::vec2(j, i),
            .uv_size = uv_size,
            .image = root_image
          };
        }
      }
    }

    if (ImGui::InputText(
      "image path",
      image_path.data(),
      image_path.size(),
      ImGuiInputTextFlags_EnterReturnsTrue)
      ) {

      root_image.load(image_path.c_str());
      open_properties.preferred_pack_size = root_image.size;
      /* images.push_back(image_t{
         .uv_pos =0,
         .uv_size=1,
         .image =image
       });*/
    }
    ImGui::GetStyle().ItemSpacing.x = 1;
    ImGui::GetStyle().ItemSpacing.y = 1;
    for (auto& i : images) {
      int x = 0;
      for (auto& j : i) {
        if (x) {
          ImGui::SameLine();
        }
        ImGui::Image((void*)j.image.get_texture(), j.image.size / cell_size, j.uv_pos, j.uv_pos + j.uv_size);
        x++;
      }
    }

    static fan::string texturepack_path;
    texturepack_path.resize(40);
    if (ImGui::InputText(
      "save texturepack",
      texturepack_path.data(),
      texturepack_path.size(),
      ImGuiInputTextFlags_EnterReturnsTrue)
      ) {
      int index = 0;
      for (auto& i : images) {
        for (auto& j : i) {
          texture_properties.name = fan::string("tile") + std::to_string(index);
          texture_properties.uv_pos = j.uv_pos;
          texture_properties.uv_size = j.uv_size;
          e.push_texture(j.image, texture_properties);
          ++index;
        }
      }
      e.process();
      e.save_compiled(texturepack_path.c_str());
    }

    ImGui::End();
  });
  //e.push_texture()
  loco.loop([] {

  });

}