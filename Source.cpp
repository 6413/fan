
#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(types/vector.h)

struct network_data_t {
  fan::vec3 position;
  fan::vec2 size;
};


int main() {

  network_data_t d{.position = 10, .size = 15};
  char* byte_array = (char*)&d;

  network_data_t din = *(network_data_t*)byte_array;
}