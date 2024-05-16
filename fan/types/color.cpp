#include "color.h"

#include <random>

fan::color fan::color::hsv(f32_t H, f32_t S, f32_t V) {

  f32_t s = S / 100;
  f32_t v = V / 100;
  f32_t C = s * v;
  f32_t X = C * (1 - std::abs(fmod(H / 60.0, 2) - 1));
  f32_t m = v - C;
  f32_t r, g, b;
  if (H >= 0 && H < 60) {
    r = C, g = X, b = 0;
  }
  else if (H >= 60 && H < 120) {
    r = X, g = C, b = 0;
  }
  else if (H >= 120 && H < 180) {
    r = 0, g = C, b = X;
  }
  else if (H >= 180 && H < 240) {
    r = 0, g = X, b = C;
  }
  else if (H >= 240 && H < 300) {
    r = X, g = 0, b = C;
  }
  else {
    r = C, g = 0, b = X;
  }
  int R = (r + m) * 255;
  int G = (g + m) * 255;
  int B = (b + m) * 255;

  return fan::color::rgb(R, G, B, 255);
}

cf_t* fan::color::data() {
  return &r;
}

int64_t fan::color::value_i64(int64_t min, int64_t max) {
  static std::random_device device;
  static std::mt19937_64 random(device());

  std::uniform_int_distribution<int64_t> distance(min, max);

  return distance(random);
}

f32_t fan::color::value_f32(f32_t min, f32_t max) {
  return (f32_t)value_i64(min * float_accuracy, max * float_accuracy) / float_accuracy;
}

void fan::color::randomize() {
  *this = fan::color(
    value_f32(0, 1),
    value_f32(0, 1),
    value_f32(0, 1),
    1
  );
}

std::string fan::color::to_string() const noexcept {
  return "{ " + 
        std::to_string(r) + ", " + 
        std::to_string(g) + ", " + 
        std::to_string(b) + ", " + 
        std::to_string(a) + " }";
}