#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(types/vector.h)

void f() {

}

struct HelloWorld {
  void main() {
    f();
  }
};

struct b_t {
  b_t();
};

struct a_t {
  b_t b;
};

b_t::b_t() {
  a_t* a = OFFSETLESS(this, a_t, b);
  fan::print("b", a);
}

struct c_t : a_t, b_t {

};


struct st_t {
  st_t(auto x) {

  }
  

  fan_init_id_t0(
    loco_t::sprite,
    id,
    .position = pos
  );
};

int main() {
  fan::vec2 pos = 0;
  auto st = new st_t(pos)
}

int main() {

}