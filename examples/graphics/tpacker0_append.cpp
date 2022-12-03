#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#if set_compile == 0
#define FAN_INCLUDE_PATH C:/libs/fan/include
#elif set_compile == 1
#define FAN_INCLUDE_PATH /usr/include
#else
#error ?
#endif
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#define loco_tp
#include _FAN_PATH(graphics/loco.h)

#include _FAN_PATH(io/directory.h)

int main() {

  loco_t::texture_packe0::open_properties_t open_properties;
  open_properties.preferred_pack_size = 1024;
  loco_t::texture_packe0 e;
  e.open(open_properties);
  if (fan::io::file::exists("../../TexturePack")) {
    e.load("../../TexturePack");
    fan::print(e.texture_list.size());
  }
  loco_t::texture_packe0::texture_properties_t texture_properties;
  texture_properties.visual_output = fan::opengl::GL_CLAMP_TO_EDGE;
  texture_properties.filter = fan::opengl::GL_NEAREST;
  texture_properties.group_id = 0;
  fan::io::iterate_directory_by_image_size("../images_out", [&](fan::string path) {
    if (std::size_t found = path.find("block") == std::string::npos) {
      return;
    }
    fan::string p = path;
    p = p.substr(strlen("../images_out/"), std::string::npos);
    texture_properties.name = p;
    e.push_texture(path, texture_properties);

    });
  texture_properties = loco_t::texture_packe0::texture_properties_t();
  fan::io::iterate_directory_by_image_size("../images_out", [&] (fan::string path) {
    if (std::size_t found = path.find("block") != std::string::npos) {
      return;
    }
    fan::string p = path;
    p = p.substr(strlen("../images_out/"), std::string::npos);

    texture_properties.name = p;
    e.push_texture(path, texture_properties);
    });
  e.process();
  fan::print_no_space("pack size:", e.size());
  e.save("../../TexturePack");
}
