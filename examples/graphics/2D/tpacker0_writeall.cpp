#include <fan/pch.h>

#include <fan/io/directory.h>

int main() {
  
  loco_t::texture_packe0::open_properties_t open_properties;
  open_properties.preferred_pack_size = 1024;
  loco_t::texture_packe0 e;
  e.open(open_properties);
  loco_t::texture_packe0::texture_properties_t texture_properties;
  texture_properties.visual_output = loco_t::image_sampler_address_mode::clamp_to_edge;
  texture_properties.min_filter = loco_t::image_filter::nearest;
  texture_properties.mag_filter = loco_t::image_filter::nearest;
  texture_properties.group_id = 0;
  static constexpr auto full_path = "controller_images/";

  fan::io::iterate_directory_by_image_size(full_path, [&](fan::string path) {
    //if (std::size_t found = path.find("block") == fan::string::npos) {
    //  return;
    //}
    fan::string p = path;
    auto len = strlen(full_path);
    p = p.substr(len, p.size() - len);
    texture_properties.name = p;
    texture_properties.name.replace_all(".webp", "");
    e.push_texture(path, texture_properties);
  });
  //texture_properties = loco_t::texture_packe0::texture_properties_t();
  //fan::io::iterate_directory_by_image_size(full_path, [&] (fan::string path) {
  //  //if (std::size_t found = path.find("block") != std::string::npos) {
  //  //  return;
  //  //}
  //  fan::string p = path;
  //  auto len = strlen(full_path);
  //  p = p.substr(len, p.size() - len);

  //  texture_properties.name = p;
  //  texture_properties.name.replace_all(".webp", "");
  //  e.push_texture(path, texture_properties);
  // });
  e.process();
  fan::print_no_space("pack size:", e.size());
  e.save_compiled("controller");
}
