#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(graphics/graphics.h)

#include <unordered_map>
#include <iostream>
#include <memory_resource>

int main(){

  constexpr uint32_t count = 0x1000;
  std::byte stack[count * sizeof(uint32_t)];
  std::pmr::monotonic_buffer_resource rsrc(stack, sizeof(stack));
  std::pmr::unordered_map < uint32_t, uint32_t> map{ {{}}, &rsrc };

  map.reserve(count);

  fan::time::clock c;
  c.start();

  for (uint32_t i = 0; i < count; i++) {
    map.emplace(std::make_pair(i, i));
  }
  fan::print((f32_t)c.elapsed() / 1e+9);
  for (uint32_t i = 0; i < count; i++) {
    auto found = map.find(i);
    if (found == map.end()) {
      std::cout << "somethgni";
      break;
    }
    else {
      if (found->second != i) {
        std::cout << "something";
        break;
      }
    }
  }
  
}
