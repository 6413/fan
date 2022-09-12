#include <fan/types/types.h>

#define BLL_set_AreWeInsideStruct 0
//#define BLL_set_CPP_Node_ConstructDestruct
#define BLL_set_BaseLibrary 1
#define BLL_set_type_node uint32_t
#define BLL_set_Link 1
#define BLL_set_prefix something
#define BLL_set_node_data fan::function_t<int()> data;
#include _FAN_PATH(BLL/BLL.h)

int main() {
  something_t a;
  a.open();
  auto nr = a.NewNodeLast();
  int y = 5;
  a[nr].data = [&] { return y; };
  for (uint32_t i = 0; i < 1000; i++) {
    auto nr2 = a.NewNodeLast();
    a[nr2].data = [=] { return i; };
  }

  return a[nr].data();
}