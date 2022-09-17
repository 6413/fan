#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(time/time.h)
#include _FAN_PATH(math/random.h)

#define BLL_set_StoreFormat 1
#define BLL_set_BaseLibrary 1
#define BLL_set_prefix instance
#define BLL_set_type_node uint16_t
#define BLL_set_node_data fan::string str;
#define BLL_set_Link 1
#define BLL_set_AreWeInsideStruct 0
#define BLL_set_CPP_Node_ConstructDestruct
#define BLL_set_BaseLibrary 1
#include _FAN_PATH(BLL/BLL.h)

#include <fan/types/fstring.h>

//#pragma pack(push, 1)
// struct a_t {
//	//uint8_t begin_pad[13];
//	std::string str;
//};
//#pragma pack(pop)
//#include <Windows.h>
//
int main(int arg) {
	fan::string str;
	fan::string str2 = "test";
	str.insert(str.begin(), "str2");
	return str == "";
}