#include <iostream>
#include <FAN/Alloc.hpp>

int main() {
	Alloc<int> x;
	uint64_t i = 250000000;
	while (i--) {
		x.push_back(i);
	}
	printf("im done!\n");
	getchar();
}