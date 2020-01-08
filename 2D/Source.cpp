#include <iostream>
#include <FAN/Alloc.hpp>
#include <FAN/DBT.hpp>

#include <ctime>


int main() {
	dbt<int> db;
	int x = 100000 -1;
	for (int i = 0; i < 100000; i++) {
		db.insert((unsigned char*)&i, sizeof(i) * 8, i);
	}
	printf("%d\n", db.search((unsigned char*)&x, sizeof(x) * 8));
}
