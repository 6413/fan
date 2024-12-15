#include <fan/pch.h>

struct image_divider_t {
  struct image_t {
    fan::vec2 uv_pos;
    fan::vec2 uv_size;
    loco_t::image_t image;
  };

  loco_t::image_t root_image = gloco->default_texture;

  std::vector<std::vector<image_t>> images;
  fan::vec2 child_window_size = 1;

  struct image_click_t {
    int highlight = 0;
    int count_index;
  };

  std::vector<image_click_t> clicked_images;
  loco_t::texture_packe0::open_properties_t open_properties;
  loco_t::texture_packe0 e;
  loco_t::texture_packe0::texture_properties_t texture_properties;

  image_divider_t() {
    e.open(open_properties);
    texture_properties.visual_output = loco_t::image_sampler_address_mode::clamp_to_edge;
    texture_properties.min_filter = loco_t::image_filter::nearest;
    texture_properties.mag_filter = loco_t::image_filter::nearest;
  }

  void render() {
    auto& style = ImGui::GetStyle();
    ImVec4* colors = style.Colors;
    const ImVec4 bgColor = ImVec4(0.1, 0.1, 0.1, 0.1);
    colors[ImGuiCol_WindowBg].w = bgColor.w;
    colors[ImGuiCol_ChildBg].w = bgColor.w;
    colors[ImGuiCol_TitleBg].w = bgColor.w;

    ImGui::Begin("Editor");
    ImGui::Columns(2, "mycolumns", false);

    ImGui::SetColumnWidth(0, ImGui::GetWindowSize().x * 0.4f);

    fan::vec2 window_size = gloco->window.get_size();

    fan::vec2 viewport_size = ImGui::GetContentRegionAvail();
    fan::vec2 viewport_pos = fan::vec2(ImGui::GetWindowPos() + fan::vec2(0, ImGui::GetFontSize() + ImGui::GetStyle().FramePadding.y * 2));
    fan::vec2 offset = viewport_size - viewport_size;
    fan::vec2 s = viewport_size;
    gloco->camera_set_ortho(
      gloco->orthographic_camera.camera,
      fan::vec2(-s.x, s.x),
      fan::vec2(-s.y, s.y)
    );
    gloco->viewport_set(
      gloco->orthographic_camera.viewport,
      viewport_pos, viewport_size, window_size
    );

    static fan::string image_path;
    image_path.resize(50);

    static fan::vec2f cell_size = { 1, 1 };
    static int horizontal_line_count = 1;
    static int vertical_line_count = 1;
    bool update_drag = ImGui::InputInt("Horizontal Line Count", &horizontal_line_count, 1, 100) ||
      ImGui::InputInt("Vertical Line Count", &vertical_line_count, 1, 100);
    if (update_drag) {
      images.clear();
      if (root_image.iic() == false) {
        auto& img = gloco->image_get_data(root_image);
        fan::vec2i divider = { horizontal_line_count, vertical_line_count };
        fan::vec2 uv_size = img.size / divider / img.size;

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
        clicked_images.resize(divider.multiply());
        for (auto& i : clicked_images) {
          i.highlight = 0;
          i.count_index = 0;
        }
      }
    }

    auto& img = gloco->image_get_data(root_image);
    fan::vec2 normalized_image_size = img.size.normalize();
    cell_size.x = (child_window_size.min() * 0.95 * (normalized_image_size.x)) / horizontal_line_count;
    cell_size.y = (child_window_size.min() * 0.95 * (normalized_image_size.y)) / vertical_line_count;

    if (ImGui::InputText(
      "image path",
      image_path.data(),
      image_path.size(),
      ImGuiInputTextFlags_EnterReturnsTrue)
      ) {
      root_image = gloco->image_load(
        image_path.c_str()
      );
      auto& img = gloco->image_get_data(root_image);
      open_properties.preferred_pack_size = img.size;
    }
    ImGui::GetStyle().ItemSpacing.x = 1;
    ImGui::GetStyle().ItemSpacing.y = 1;

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
          texture_properties.image_name = fan::string("tile") + std::to_string(index);
          texture_properties.uv_pos = j.uv_pos;
          texture_properties.uv_size = j.uv_size;
          e.push_texture(j.image, texture_properties);
          ++index;
        }
      }
      e.process();
      e.save_compiled(texturepack_path.c_str());
    }

    std::size_t vec_size = 0;
    for (std::size_t i = 0; i < images.size(); ++i) {
      vec_size += images[i].size();
    }

    ImGui::NextColumn();

    ImGui::BeginChild("image");

    child_window_size = ImGui::GetWindowSize();

    int totalIndex = 0;

    int highlighted_count = 0;
    for (int k = 0; k < clicked_images.size(); ++k) {
      highlighted_count += clicked_images[k].highlight;
    }

    for (auto& i : images) {
      int x = 0;
      for (auto& j : i) {
        if (x) {
          ImGui::SameLine();
        }
        auto& jimg = gloco->image_get_data(j.image);

        if (clicked_images[totalIndex].highlight) {
          ImGui::PushStyleColor(ImGuiCol_Border, fan::color::hex(0x00e0ffff));
        }
        else {
          ImGui::PushStyleColor(ImGuiCol_Border, fan::color(0.3, 0.3, 0.3, 1));
        }

        ImGui::PushID(totalIndex);

        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, { 0, 0 });
        ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.0f);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1, 0, 0, 0.3));

        if (ImGui::ImageButton("", j.image, cell_size, j.uv_pos, j.uv_pos + j.uv_size)) {

          clicked_images[totalIndex].highlight = !clicked_images[totalIndex].highlight;
          if (clicked_images[totalIndex].highlight) {
            clicked_images[totalIndex].count_index = highlighted_count;
          }
          else {
            for (int k = 0; k < clicked_images.size(); ++k) {
              if (clicked_images[totalIndex].count_index < clicked_images[k].count_index) {
                --clicked_images[k].count_index;
              }
            }
          }
        }
        ImGui::PopStyleColor(2);
        ImGui::PopStyleVar(2);

        if (clicked_images[totalIndex].highlight) {
          ImVec2 p = ImGui::GetItemRectMin(); // Top-left of the image button
          ImVec2 size = ImGui::GetItemRectSize(); // Size of the image button
          ImVec2 text_size = ImGui::CalcTextSize(std::to_string(totalIndex).c_str());
          ImVec2 text_pos = ImVec2(p.x + 2, p.y + 2);

          ImU32 outline_col = IM_COL32(0, 0, 0, 255); // Black
          // Original text color
          ImU32 text_col = IM_COL32(255, 255, 255, 255); // White

          ImGui::GetWindowDrawList()->AddText(ImVec2(text_pos.x + 1, text_pos.y), outline_col, std::to_string(clicked_images[totalIndex].count_index).c_str());
          ImGui::GetWindowDrawList()->AddText(ImVec2(text_pos.x - 1, text_pos.y), outline_col, std::to_string(clicked_images[totalIndex].count_index).c_str());
          ImGui::GetWindowDrawList()->AddText(ImVec2(text_pos.x, text_pos.y + 1), outline_col, std::to_string(clicked_images[totalIndex].count_index).c_str());
          ImGui::GetWindowDrawList()->AddText(ImVec2(text_pos.x, text_pos.y - 1), outline_col, std::to_string(clicked_images[totalIndex].count_index).c_str());

          ImGui::GetWindowDrawList()->AddText(text_pos, text_col, std::to_string(clicked_images[totalIndex].count_index).c_str());
        }

        ImGui::PopID();

        x++;
        totalIndex++; // Increment the total index for the next image
      }
    }
    ImGui::EndChild();
    ImGui::Columns(1);

    ImGui::End();
  }

};

int main() {
  loco_t loco;

  image_divider_t image_divider;

  loco.loop([&] {
    image_divider.render();
  });

}