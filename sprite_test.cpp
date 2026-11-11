import fan;
import std;

using namespace fan::graphics;

int main() {
  engine_t engine;
  std::vector<std::uint8_t> pixels{
    255, 0, 0, 255,
    0, 255, 0, 255,
    0, 0, 255, 255,
    255, 255, 255, 255
  };
  engine.loop([&] {

    sprite(engine.whs(), engine.whs(), pixels, fan::vec2(2, 2));
  });
}
