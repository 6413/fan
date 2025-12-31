module;

#include <sstream>

module fan.types.color;

import fan.random;

namespace fan {

  color color::hsv(f32_t H, f32_t S, f32_t V) {
    f32_t s = S / 100, v = V / 100;
    f32_t C = s * v;
    f32_t X = C * (1 - std::abs(fmod(H / 60.0, 2) - 1));
    f32_t m = v - C;

    int i = static_cast<int>(H / 60) % 6;
    f32_t rgb[6][3] = { {C,X,0},{X,C,0},{0,C,X},{0,X,C},{X,0,C},{C,0,X} };

    return fan::color(rgb[i][0] + m, rgb[i][1] + m, rgb[i][2] + m, 1.0f);
  }

  color color::to_srgb(const color& c) {
    return color(
      linear_to_srgb_channel(c.r),
      linear_to_srgb_channel(c.g),
      linear_to_srgb_channel(c.b),
      c.a
    );
  }
  color color::to_linear(const color& c) {
    return color(
      srgb_to_linear_channel(c.r),
      srgb_to_linear_channel(c.g),
      srgb_to_linear_channel(c.b),
      c.a
    );
  }

  void color::rgb_to_hsl(f32_t r, f32_t g, f32_t b, f32_t& h, f32_t& s, f32_t& l) {
    f32_t maxc = std::max(r, std::max(g, b));
    f32_t minc = std::min(r, std::min(g, b));
    f32_t delta = maxc - minc;
    l = (maxc + minc) * 0.5f;
    if (delta < 0.00001f) {
      h = 0.0f;
      s = 0.0f;
      return;
    }
    s = delta / (1.0f - std::abs(2.0f * l - 1.0f));
    if (maxc == r) {
      h = (g - b) / delta;
      if (h < 0.0f) h += 6.0f;
    }
    else if (maxc == g) {
      h = ((b - r) / delta) + 2.0f;
    }
    else {
      h = ((r - g) / delta) + 4.0f;
    }
    h /= 6.0f;
  }

  color color::hue(f32_t degrees) const {
    f32_t h, s, l;
    rgb_to_hsl(r, g, b, h, s, l);
    h += degrees / 360.0f;
    h = h - std::floor(h);
    f32_t rr, gg, bb;
    hsl_to_rgb(h, s, l, rr, gg, bb);
    return color(rr, gg, bb, a);
  }

  color color::saturation(f32_t amount) const {
    f32_t h, s, l;
    rgb_to_hsl(r, g, b, h, s, l);
    s = clamp(s * (1.0f + amount / 100.0f), 0.0f, 1.0f);
    f32_t rr, gg, bb;
    hsl_to_rgb(h, s, l, rr, gg, bb);
    return color(rr, gg, bb, a);
  }
  color color::lightness(f32_t amount) const {
    f32_t h, s, l;
    rgb_to_hsl(r, g, b, h, s, l);
    if (amount < 0) {
      l = l * (1.0f + amount / 100.0f);
    }
    else {
      l = l + (1.0f - l) * (amount / 100.0f);
    }
    l = clamp(l, 0.0f, 1.0f);
    f32_t rr, gg, bb;
    hsl_to_rgb(h, s, l, rr, gg, bb);
    return color(rr, gg, bb, a);
  }

  void color::randomize() {
    r = fan::random::value(0.f, 1.f);
    g = fan::random::value(0.f, 1.f);
    b = fan::random::value(0.f, 1.f);
    a = 1;
  }

  std::string color::to_string() const noexcept {
    return "{ " +
      std::to_string(r) + ", " +
      std::to_string(g) + ", " +
      std::to_string(b) + ", " +
      std::to_string(a) + " }";
  }

  void color::from_string(const std::string& str) {
    std::string s;
    for (char c : str) {
      if (c != '{' && c != '}' && c != ' ')
        s += c;
    }

    std::stringstream ss(s);
    std::string item;
    f32_t values[4] = { 0, 0, 0, 1 };
    size_t i = 0;

    while (std::getline(ss, item, ',') && i < 4) {
      try {
        values[i++] = std::stof(item);
      }
      catch (...) {
        values[i++] = f32_t();
      }
    }

    r = values[0];
    g = values[1];
    b = values[2];
    a = values[3];
  }

  color color::parse(const std::string& str) {
    color out;
    out.from_string(str);
    return out;
  }

  std::ostream& operator<<(std::ostream& os, const color& c) noexcept {
    os << c.to_string();
    return os;
  }

  fan::color random::color() {
    return fan::color(
      fan::random::value(0.f, 1.f),
      fan::random::value(0.f, 1.f),
      fan::random::value(0.f, 1.f),
      1
    );
  }

  fan::color random::bright_color() {
    fan::color rand_color = fan::random::color();
    f32_t max_channel = std::max({ rand_color.r, rand_color.g, rand_color.b });
    return rand_color / max_channel;
  }
}