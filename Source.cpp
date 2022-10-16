#define FAN_INCLUDE_PATH C:/libs/fan/include

#include <fan/types/types.h>

int main() {
	const char* x = _FAN_PATH_QUOTE(graphics);
	fan::print(x);
}