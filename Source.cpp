#include <fan/types/types.h>


int main(int arg) {
	fan::time::clock c;
	c.start();
	int arr[100][100];
	uint32_t x = 0;
	for (uint32_t i = 0; i < 1000; i++) {
		if (arr[i][i / 5]) {
			arr[i][i / 10] += arg * 5;
			x *= i;
		}
		x += i * arg;
	}

	fan::print(c.elapsed());

}