#include <memory>


#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#define loco_vulkan

#define loco_window
#define loco_context

//fan::function_t<void()> f[5];

//void(*magic_function_array[5])(void**);

struct base_t {
  uint32_t i;
  void* data;
};

int main() {
  base_t base;
  magic_function_array[rand()](&base.data);
}