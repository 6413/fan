#include <iostream>
#include <FAN/DBT.hpp>

#include <ctime>

int main() {
	dbt<int> tree(1);
	int x = 5;
	tree.push((unsigned char*)&x, sizeof(x) * 8, x);

	printf("%d\n", tree.search((unsigned char*)&x, sizeof(x) * 8));
}
