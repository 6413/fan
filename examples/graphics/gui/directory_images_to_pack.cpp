#include <fan/pch.h>

#include _FAN_PATH(io/directory.h)

// images will be written to texturepack_name
// texturepack_path will include name of it
void images_to_texturepack(const std::vector<std::string>& image_paths, const std::string& texturepack_path) {

  loco_t::texture_packe0::open_properties_t open_properties;
  open_properties.preferred_pack_size = 1024;
  loco_t::texture_packe0 e;
  e.open(open_properties);
  loco_t::texture_packe0::texture_properties_t texture_properties;
  texture_properties.visual_output = loco_t::image_sampler_address_mode::clamp_to_edge;
  texture_properties.min_filter = loco_t::image_filter::nearest;
  texture_properties.mag_filter = loco_t::image_filter::nearest;
  texture_properties.group_id = 0;
  e.load_compiled(texturepack_path.c_str());

  for (std::size_t i = 0; i < image_paths.size(); ++i) {
    std::string base_filename = image_paths[i].substr(image_paths[i].find_last_of("/\\") + 1);
    std::string extension = base_filename.substr(base_filename.find_last_of('.') + 1);
    base_filename = base_filename.substr(0, base_filename.find_last_of('.'));
    texture_properties.image_name = base_filename;
    
    e.push_texture(image_paths[i], texture_properties);
  }

  e.process();
  e.save_compiled(texturepack_path.c_str());
}

void gui_images_to_texturepack(loco_t::texturepack_t& tp) {
  static fan::graphics::file_open_dialog_t open_file_dialog;


  ImGui::Begin("Import images");

  static std::vector<std::string> image_names;
  if (ImGui::Button("select models")) {
    image_names.clear();
    open_file_dialog.load("webp", &image_names);
  }
  
  if (open_file_dialog.is_finished()) {

    if (image_names.size()) {
      images_to_texturepack(image_names, tp.file_path);
      tp.open_compiled(tp.file_path);
    }
    open_file_dialog.finished = false;
  }

  ImGui::End();
}

void gui_render_texturepack(loco_t::texturepack_t& tp) {
  ImGui::Begin("render texturepack");
  int images_per_row = 4;
  int image_width = 200;
  int image_height = 200;

  int image_count = 0;
  for (auto& pd : tp.pixel_data_list) {
    ImGui::Image(pd.image, fan::vec2(image_width, image_height));
    image_count++;
    if (image_count % images_per_row != 0) {
      ImGui::SameLine();
    }
  }
  ImGui::End();
}

int main() {
  loco_t loco;


  loco_t::texturepack_t tp;
  tp.open_compiled("test.ftp");

  loco.loop([&] {
    gui_images_to_texturepack(tp);
    gui_render_texturepack(tp);
  });
}