#include <fan/types/types.h>

struct a_t {
	int x;
	void f() {
		fan::print("this");
	}
};

struct s_t {
	int y;
	void f() {
		fan::print("st is called");
	}
};

struct c_t : a_t, s_t {
	int c;
	void f() {
		fan::print(this);
	}
};

int main() {
	c_t c;
	((a_t*)&c)->f();
}