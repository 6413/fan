#include <fan/types/types.h>

int main() {
	fan::string str("durum/likes/salsa");
	auto found = str.find_last_of("/") + 1;
	if (found == fan::string::npos) {
		// not found
	}
	str = str.substr(found, str.size() - found);
	fan::print(str);
}