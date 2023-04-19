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

struct a_t {
  int cid;
};

struct b_t {
  a_t a;
  operator a_t&() {
    return a;
  }
  operator a_t*() {
    return &a/*something long here*/;
  }
  a_t* operator->() {
    return &a;
  }
  //b_t* operator->() {
  //  return &a;
  //}
};

void f(a_t& a) {

}

int main() {
  b_t b;
  f(b);
  //a_t* c = b;
  //c>
  //;
  //b->bm_id = 5;
  
  //(&b)->x;
  //b->
}