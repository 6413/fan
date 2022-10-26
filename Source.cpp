#include <fan/types/types.h>
//
//#if defined(__clang__)
//#error clang broken
//#endif

#if defined(fan_compiler_visual_studio)
#error clang broken
#endif
struct a_t {
  a_t() {
    fan::print("moi");
  }
};

struct b_t {
  b_t() = default;
  a_t a_;
};

int main() {
  b_t x;
}