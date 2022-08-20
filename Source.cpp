#include <fan/types/masterpiece.h>
#include <fan/types/types.h>

struct s_t {
  #define something x
  using type = int;
  type x;
  void F() {
    OFFSETLESS(0, s_t, something);
  }
  #undef something
};



int main() {
  
}