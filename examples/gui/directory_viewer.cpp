#include fan_pch

#include _FAN_PATH(io/directory.h)

struct directory_item_t {
  fan::string name;
  bool is_directory;
};

void sort_directory(std::vector<directory_item_t>& directory_list) {
  std::sort(directory_list.begin(), directory_list.end(), [](const directory_item_t& a, const directory_item_t& b) {
    if (a.is_directory != b.is_directory) {
      return a.is_directory > b.is_directory;
    }
    else {
      return std::lexicographical_compare(
        a.name.begin(), a.name.end(), b.name.begin(), b.name.end(),
      [](char c1, char c2) { return std::tolower(c1) < std::tolower(c2); }
      );
    }
  });
}
void update_directory_list(const fan::string& path, std::vector<directory_item_t>& directory_list) {
  directory_list.clear();
  fan::io::iterate_directory(path, [&](const fan::string& item, bool is_directory) {
      directory_item_t directory_item;
      directory_item.name = item;
      directory_item.is_directory = is_directory;
      directory_list.push_back(directory_item);
  });
  sort_directory(directory_list);
}

int main() {
  loco_t loco;

  fan::string current_path = std::filesystem::current_path().string();
  fan::string prev_path = current_path;

  std::vector<directory_item_t> directory_list;

  update_directory_list(current_path, directory_list);

  loco_t::image_t image;
  image.load("images/folder.webp");

  fan::string search_path;
  static constexpr int max_search_path = 50;
  search_path.resize(max_search_path);

  fan::string peek_file_contents;

  fan::graphics::imgui_element_t element([&] {
    ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());

    ImGui::Begin("Top panel");
    if (ImGui::Button("<-")) {
      prev_path = current_path;
      if (fan::io::is_readable_path(current_path.c_str()))  {
        std::filesystem::directory_entry directory_entry(current_path.c_str());
        fan::string new_path = directory_entry.path().parent_path().string();
        if (fan::io::is_readable_path(new_path)) {
          current_path = new_path;
          update_directory_list(current_path, directory_list);
        }
      }
    }
    ImGui::SameLine();
    if (ImGui::Button("->")) {
      fan::string temp = current_path;
      current_path = prev_path;
      prev_path = temp;
      update_directory_list(current_path, directory_list);
    }
    {
      if (ImGui::InputText("path", search_path.data(), search_path.size())) {
      }
      if (ImGui::IsKeyPressed(ImGuiKey_Enter)) {
        current_path = search_path.c_str();
        if (fan::io::directory_exists(current_path)) {
          update_directory_list(current_path, directory_list);
        }
      }
    }
   
    ImGui::End();

    ImGui::Begin("Side panel");

    ImGui::End();

    ImGui::Begin("Directory view");
    for (auto& i : directory_list) {

      fan::string item_name = fan::io::exclude_path(i.name);

      ImGui::Selectable(("##" + item_name).c_str());
      if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0)) {
        if (i.is_directory) {
          if (fan::io::is_readable_path(i.name.c_str())) {
            prev_path = current_path;
            current_path = i.name;
            update_directory_list(current_path, directory_list);
          }
        }
        else {
          peek_file_contents.clear();
          fan::io::file::read(i.name, &peek_file_contents);
        }
      }
      ImGui::SameLine();
      if (i.is_directory) {
        ImGui::Image((ImTextureID)image.get_texture(), fan::vec2(ImGui::GetItemRectSize().y - 2));
      }
      ImGui::SameLine();
      ImGui::Text(item_name.c_str());

    }

    ImGui::Begin("Peek file");
    if (peek_file_contents.size()) {
      ImGui::InputTextMultiline("##TextFileContents", peek_file_contents.data(), peek_file_contents.size(), ImVec2(-1.0f, -1.0f), ImGuiInputTextFlags_ReadOnly);
    }
    ImGui::End();


    ImGui::End();
  });

  loco.set_vsync(0);
  loco.window.set_max_fps(165);

  loco.loop([&] {

  });

  return 0;
}