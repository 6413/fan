#include <fan/types/types.h>

#define name a_t


struct 
	CONCAT(name, _)
{

};
#ifdef BLL_set_declare_rest
struct name : CONCAT(name, _) {
	functions
#endif
}

void open(name*) {

}

int main() {

}