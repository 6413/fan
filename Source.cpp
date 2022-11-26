#include <fan/types/masterpiece.h>
#include <fan/types/types.h>
#include <fan/types/vector.h>

struct common {
	inline static int x = 0;
	constexpr bool f() { x++; return 0; }
};

struct protocol_c {
	constexpr protocol_c(bool) {}
	
};

#define make_p(n) protocol_c n{f()};

struct protocol : common {
	
	make_p(a);
	make_p(b);
};

int main(){
	protocol p;
	fan::print(p.x);
}