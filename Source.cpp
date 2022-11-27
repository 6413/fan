#include <fan/types/masterpiece.h>
#include <fan/types/types.h>
#include <fan/types/vector.h>

int main(){
	uint32_t* v = (uint32_t*)malloc(2048 * 2048 * 4);
	fan::time::clock c;
	c.start();
	uint32_t i = 0;
	for (; i < 1; i++) {
		memset(v, 5, 2048 * 2048 * 4);
	}
	fan::print(c.elapsed() / i);
	for (i = 0; i < 2048 * 2048 ; i++) {
		if (v[i] != 0x05050505) {
			fan::print(i, v[i]);
			break;
		}
	}
}