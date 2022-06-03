#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#if set_compile == 0
  #define FAN_INCLUDE_PATH C:/libs/fan/include
#elif set_compile == 1
  #define FAN_INCLUDE_PATH /usr/include
#else
  #error ?
#endif
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(graphics/graphics.h)

#include _FAN_PATH(tp/tp.h)

int main() {

  fan::tp::texture_packe e;
  e.open();
  e.push_pack(fan::vec2(2048, 3072));
  fan::io::iterate_directory("images", [&] (std::string path) {
    std::string p = path;
    p = p.substr(strlen("images/"), std::string::npos);
    e.push_texture(0, path, p);
  });
  e.save("TexturePack");
}
