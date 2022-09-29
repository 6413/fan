#include <fan/types/types.h>

struct a_t {
	int x;
	int y;
};

struct b_t : a_t{
	int z;
};

int main() {
	a_t* a = new b_t;
	b_t* promoted = (b_t*)a;
}