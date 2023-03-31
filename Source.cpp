#include <fan/types/types.h>

#define BDBT_set_prefix bdbt
#define BDBT_set_type_node uint32_t
#define BDBT_set_BitPerNode 2
#define BDBT_set_declare_rest 1
#define BDBT_set_declare_Key 0
#define BDBT_set_BaseLibrary 1
#define BDBT_set_CPP_ConstructDestruct
#include _FAN_PATH(BDBT/BDBT.h)

#define BDBT_set_prefix bdbt
#define BDBT_set_type_node uint32_t
#define BDBT_set_KeySize 0
#define BDBT_set_BitPerNode 2
#define BDBT_set_declare_rest 0 
#define BDBT_set_declare_Key 1
#define BDBT_set_base_prefix bdbt
#define BDBT_set_BaseLibrary 1
#define BDBT_set_CPP_ConstructDestruct
#include _FAN_PATH(BDBT/BDBT.h)

bdbt_t bdbt;

static int x = 5;



template <int* T>
struct tt {
  static constexpr int* ptr = T;
};

int main(){
  tt<&x> vec;
  return 0;
}