#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

struct pile_t;

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#define loco_opengl

#define loco_window
#define loco_context

#define loco_no_inline

#define loco_sprite
#define loco_button
#include _FAN_PATH(graphics/loco.h)

std::variant<int, char, double, bool, std::string> v;

template<typename... Ts>
auto getValue(std::variant<Ts...>& v)
{
  using R = std::common_type_t<decltype(Ts::x)...>;
  return std::visit([](auto& obj) {return static_cast<R>(obj.x); }, v);
}

int main(int argc, char** argv) {

  //fan::print(typeid(k).name());
}