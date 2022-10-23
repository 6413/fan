#include <fan/types/types.h>

#include <fan/io/file.h>

//f = open("test", "r")
//x = f.read();

//fan::string x =
//#include <moi>
//;

void f(int* x) {
	int& y = *x;
	fan::print("x", x);
	*x = 10;
}


int main() {
	int y = 5;
	fan::print("y", &y);
	f(&y);
	fan::print(y);
}