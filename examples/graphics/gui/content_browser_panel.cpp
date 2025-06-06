#include <fan/pch.h>

#include <fan/io/directory.h>

struct imgui_content_browser_t {

protected:
  loco_t::image_t file_icon = gloco->image_load("images/file.webp");
  loco_t::image_t directory_icon = gloco->image_load("images/folder.webp");

  std::string asset_path = "./";
  std::filesystem::path current_directory = asset_path;
public:


  void render() {
    ImGui::Begin("Content Browser");

    f32_t padding = 16;
    f32_t thumbnail_size = 128;
    f32_t cell_size = thumbnail_size + padding;

    float panel_width = ImGui::GetContentRegionAvail().x;
    int column_count = std::max((int)(panel_width / cell_size), 1);

    ImGui::Columns(column_count, 0, false);

    if (current_directory != std::filesystem::path(asset_path)) {
      if (ImGui::Button("<-")) {
        current_directory = current_directory.parent_path();
      }
    }

    fan::io::iterate_directory_sorted_by_name(current_directory, [&](const std::filesystem::directory_entry& path) {
      auto p = path.path();

      auto relative_path = std::filesystem::relative(path, asset_path);
      std::string filename_string = relative_path.filename().string();

      ImGui::PushID(filename_string.c_str());
      ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
      ImGui::ImageButton(filename_string, path.is_directory() ? directory_icon : file_icon, fan::vec2(thumbnail_size));

      if (ImGui::BeginDragDropSource()) {
        const wchar_t* item_path = relative_path.c_str();
        ImGui::SetDragDropPayload("CONTENT_BROWSER_ITEM", item_path, (wcslen(item_path) + 1) * sizeof(wchar_t));
        ImGui::Text(filename_string.c_str());
        ImGui::EndDragDropSource();
      }

      ImGui::PopStyleColor();
      if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
        if (path.is_directory()) {
          current_directory /= path.path().filename();
        }
      }
      ImGui::TextWrapped(filename_string.c_str());
      ImGui::NextColumn();
      ImGui::PopID();
      });

    ImGui::Columns(1);

    ImGui::End();
  }
  void receive_drag_drop_target(auto receive_func) {
    ImGui::Dummy(ImGui::GetContentRegionAvail());

    if (ImGui::BeginDragDropTarget()) {
      if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("CONTENT_BROWSER_ITEM")) {
        const wchar_t* path = (const wchar_t*)payload->Data;
        receive_func(std::filesystem::path(path));
        //fan::print(std::filesystem::path(path));
      }
      ImGui::EndDragDropTarget();
    }
  }
};

int main() {
  loco_t loco;

  imgui_content_browser_t content_browser;

  loco.loop([&] {
    ImGui::Begin("Scene Hierarchy");
    ImGui::End();

    ImGui::Begin("Properties");
    ImGui::End();

    ImGui::Begin("Stats");
    ImGui::End();

    content_browser.render();

    ImGui::Begin("Scene");
    content_browser.receive_drag_drop_target([](const std::filesystem::path& path) {
      fan::print(path.string());
    });
    ImGui::End();
  });
}