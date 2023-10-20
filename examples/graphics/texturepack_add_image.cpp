#include fan_pch

#include _FAN_PATH(tp/tp.h)
#include _FAN_PATH(io/directory.h)

int main() {

  fan::tp::texture_packe e;
  e.open();
  fan::io::iterate_directory("gui_maker/", [&] (std::string path) {
    std::string p = path;
    p = p.substr(strlen("gui_maker/"), std::string::npos);
    if (p.find(".webp") != std::string::npos) {
      e.push_texture(path, p);
    }
    });
  fan::print(e.size());
  e.save("TexturePack");
}
