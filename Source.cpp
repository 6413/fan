#include <fan/types/types.h>

void assert(bool x, auto message) {
	if (!x) {
		fan::print(message);
		fan::debug::print_stacktrace();
	}
}

int main() {
	assert(!5, "error");
}