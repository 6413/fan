#include <iostream>
#include <FAN/DBT.hpp>

int main() {
	dbt<int> x;
	int i = 1000000;
	while (i--) {
		x.push((unsigned char*)&i, sizeof(i) * 8, i);
	}
	//Alloc<keytype<int>> x;
	//int i = 1000000;

	//while (i--) {
	//	x.push_back(keytype<int>());
	//}

	x.nodes.free();
	//getchar();
}