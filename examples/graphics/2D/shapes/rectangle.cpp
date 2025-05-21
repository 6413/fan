import fan;

#include <string>

using namespace fan::graphics;

void main_loop() {//
  gui::begin("aa");//
  static std::string buffer;
  gui::input_text("input", &buffer);
  static auto hover = fan::audio::open_piece("hover");
  static auto click = fan::audio::open_piece("click");
  if (gui::audio_button("button_name", hover, click)) {

  }
  fan::printcl(buffer);
  gui::end();
}

int main() {
  engine_t engine;
  engine.loop(main_loop);
}