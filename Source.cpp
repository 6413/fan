#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(types/masterpiece.h)


struct rectangle_t {
  fan_masterpiece_make(
    (fan::string)name,
    (f32_t)parallax
  );
};

struct sprite_t {
  fan_masterpiece_make(
    (fan::string)name,
    (f32_t)parallax
  );
};

fan::string get_shape_out(const auto& shape) {
  fan::string str;
  shape.iterate_masterpiece([&str](const auto& field, const auto& name) {
    if constexpr (std::is_same_v<std::remove_const_t<std::remove_reference_t<decltype(field)>>, fan::string>) {
      uint32_t string_length = field.size();
      str.append((char*)&string_length, sizeof(uint32_t));
      str.append(field);
    }
    else {
      str.append((char*)&field, sizeof(field));
    }
  });
  return str;
}

fan::string get_shapes_out(auto&&... shapes) {
  fan::string str;
  ((str += get_shape_out(shapes)), ...);
  return str;
}

int main() {
  rectangle_t r;
  r.name = "hi";
  r.parallax = 2;
  sprite_t s;
  s.name = "test";
  s.parallax = 4;
  fan::string str = get_shapes_out(r, s);
}