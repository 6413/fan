#include fan_pch

int main() {
  loco_t loco;

  static constexpr f32_t count = 50;
  static constexpr f32_t size_x = 2.f / count;
  struct node_t {
    fan::graphics::rectangle_t r;
    int value;
  };

  std::vector<node_t> lines;
  lines.reserve(count);

  auto calculate_position_and_size = [](int r) {
    fan::vec2 p;
    p.x = 2.f - (f32_t)r / count * 2 - 1;
    fan::vec2 s;
    s.x = 1.f / count;
    s.y = 1.f - (f32_t)r / count;
    p.y = 1.f - s.y;
    return std::make_pair(p, s);
  };

  auto shuffle_vector = [](std::vector<node_t>& v) {

    for (int i = v.size() - 1; i > 0; --i) {
      int j = fan::random::value_i64(0, i);
      std::swap(v[i].value, v[j].value);
    }
  };

  for (int i = 0; i < count; ++i) {
    int rand_ = i;
    auto[p, s] = calculate_position_and_size(rand_);
    lines.push_back({
      .r = fan::graphics::rectangle_t{{
          .position = p,
          .size = s,
          .color = fan::color::hsv((f32_t)i / count * 360, 100, 100)
      }},
      .value = rand_
    });
  }

  shuffle_vector(lines);

  for (int i = 0; i < count; ++i) {
    fan::vec2 old = lines[i].r.get_position();
    lines[i].r.set_position(fan::vec2((f32_t)lines[i].value / count * 2 - 1, old.y));
  }

  loco.set_vsync(0);

  int step = 0;
  fan::time::clock c;
  c.start(fan::time::nanoseconds(1e+8));
  loco.loop([&] {
    loco.get_fps();
    if (step < lines.size() && c.finished()) {
      for (int i = 0; i < lines.size() - 1 - step; ++i) {
        if (lines[i].value > lines[i + 1].value) {
          std::swap(lines[i].value, lines[i + 1].value);

          fan::vec2 oldPos = lines[i].r.get_position();
          lines[i].r.set_position(fan::vec2(2.f - (f32_t)lines[i].value / count * 2 - 1, oldPos.y));

          oldPos = lines[i + 1].r.get_position();
          lines[i + 1].r.set_position(fan::vec2(2.f - (f32_t)lines[i + 1].value / count * 2 - 1, oldPos.y));
        }
      }
      step++;
      c.restart();
    }
  });
}