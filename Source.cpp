#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#define xy \
  int x = 5; \
  int y = 10;

//struct audio_t;

struct process_t {
  int pads[15];
};

struct out_t {
  void f();
};

struct audio_t : process_t, out_t {
  
  int pads[20];
};


void out_t::f() {
  fan::print(((process_t*)this));
}

int main() {
  audio_t d;
  fan::print("a", (out_t*)&d, (process_t*)&d, &d);
  d.f();
}
