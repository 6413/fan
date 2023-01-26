#include <stdlib.h>

struct a_t {
  int z = 2511;
  int x = 5121;
  int y = 6521;
};

#define DOSOMETHING(x) var__ x

#define FOREACH_1(f, x) f(x)
#define FOREACH_2(f, x, ...)  f(x); FOREACH_1(f,__VA_ARGS__)
#define FOREACH_3(f, x, ...)  f(x); FOREACH_2(f,__VA_ARGS__)
#define FOREACH_4(f, x, ...)  f(x); FOREACH_3(f,__VA_ARGS__)
#define FOREACH_5(f, x, ...)  f(x); FOREACH_4(f,__VA_ARGS__)
#define FOREACH_N(_5,_4,_3,_2,_1,N,...) FOREACH_##N
#define FOREACH(f, ...)  FOREACH_N(__VA_ARGS__,5,4,3,2,1)(f, __VA_ARGS__)

#define init(type, ...) [] { \
  type var__; \
  FOREACH(DOSOMETHING, __VA_ARGS__); \
  return var__; \
}()

int main() {

   auto a = init(a_t, .y = 5, .x = 3);
  //auto a = a_t{ 2511, rand(), 7 };
  return a.y;

}