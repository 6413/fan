#include <fan/types/types.h>

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(time/time.h)

//struct a_t {
//	fan::function_t<int(int)> f;
//};
//
//struct b_t {
//	a_t a{ .f = [](int x) -> int {  return x; } };
//};

struct a_t {
	virtual int f(int number) {
		return number * rand();
	}
};

struct b_t : a_t {

};

static struct ga_t {
	static inline struct c_t {
		void f() {

		}
	}c;
};

int main(int arg) {
	ga_t::c.f();
	fan::time::clock c;

	uint32_t v = 0;
	std::vector<b_t> bs(1e+8);
	//std::vector<b_t> bs2(1e+8);
	c.start();
	for (uint32_t i = 0; i < 1e+8; i++) {
		bs[i] = b_t();
		v += bs[i].f(arg);
	}

	fan::print(c.elapsed());
	return v;
}