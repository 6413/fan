#define _INCLUDE_TOKEN(p0, p1) <p0/p1>
#define WITCH_INCLUDE_PATH C:/libs/WITCH
#include _INCLUDE_TOKEN(WITCH_INCLUDE_PATH, WITCH.h)

#include _WITCH_PATH(WITCH.h)
#include _WITCH_PATH(IO/print.h)

#include <fan/types/types.h>

void WriteOut(const char *format, ...){
  IO_fd_t fd_stdout;
  IO_fd_set(&fd_stdout, FD_OUT);
  va_list argv;
  va_start(argv, format);
  IO_vprint(&fd_stdout, format, argv);
  va_end(argv);
}

#define set_KeySize 32

#define BDBT_set_BaseLibrary 1
#define BDBT_set_prefix BDBT
#define BDBT_set_type_node uint32_t
#define BDBT_set_KeySize set_KeySize
#define BDBT_set_BitPerNode 2
#define BDBT_set_declare_basic_types 1
#define BDBT_set_declare_rest 1
#define BDBT_set_declare_Key 0
#include _WITCH_PATH(BDBT/BDBT.h)

#define BDBT_set_BaseLibrary 1
#define BDBT_set_base_prefix BDBT
#define BDBT_set_prefix BDBT
#define BDBT_set_type_node uint32_t
#define BDBT_set_KeySize set_KeySize
#define BDBT_set_BitPerNode 2
#define BDBT_set_declare_basic_types 0
#define BDBT_set_declare_rest 0
#define BDBT_set_declare_Key 1
#include _WITCH_PATH(BDBT/BDBT.h)

#define set_test_size 1000000

#define set_test_KeyTraverse 1
#if set_test_KeyTraverse == 1
#define set_test_KeyTraverseVerify 1
#endif
#define set_test_KeyRemove 1
#define set_test_VerifyKeyRemove 1
#define set_test_KeyInAfterRemove 1
#define set_test_VerifyKeyIn 1

int main(){
  BDBT_t bdbt;
  BDBT_open(&bdbt);

  BDBT_PreAllocateNodes(&bdbt, set_test_size * set_KeySize / 2);

  BDBT_NewNode(&bdbt);

  for(uint32_t i = 0; i < set_test_size; i++){
    uint32_t key = i;
    BDBT_KeyIn(&bdbt, &key, 0, i);
  }

  #if set_test_KeyTraverse == 1
  uint8_t *KeyTable = (uint8_t *)A_resize(0, set_test_size);
  MEM_set(0, KeyTable, set_test_size);
  uint32_t Count = 0;
  BDBT_KeyTraverse_t KeyTraverse;
  BDBT_KeyTraverse_init(&KeyTraverse, 0);
  uint32_t K;
  while(BDBT_KeyTraverse(&bdbt, &KeyTraverse, &K)){
    uint32_t key = K;
    BDBT_NodeReference_t Output = KeyTraverse.Output;
    if(key != Output){
      WriteOut("fail0 %lu %lu\n", key, Output);
      return 0;
    }
    if(key >= set_test_size){
      WriteOut("fail1 %lu\n", key);
      return 0;
    }
    if(KeyTable[key] != 0){
      WriteOut("fail2 %lu\n", key);
      return 0;
    }
    KeyTable[key] = 1;
    Count++;
  }
  if(Count != set_test_size){
    WriteOut("fail\n");
    return 0;
  }
  A_resize(KeyTable, 0);
  #endif

  #if set_test_KeyRemove == 1
  for(uint32_t i = 0; i < set_test_size / 2; i++){
    uint32_t key = i;
    BDBT_KeyRemove(&bdbt, &key, 0);
  }
  #endif

  #if set_test_VerifyKeyRemove == 1
  for(uint32_t i = 0; i < set_test_size / 2; i++){
    uint32_t key = i;
    BDBT_KeySize_t KeyIndex;
    BDBT_NodeReference_t nr = 0;
    BDBT_KeyQuery(&bdbt, &key, &KeyIndex, &nr);
    if(KeyIndex == set_KeySize){
      WriteOut("fail2\n");
      break;
    }
  }
  for(uint32_t i = set_test_size / 2; i < set_test_size; i++){
    uint32_t key = i;
    BDBT_KeySize_t KeyIndex;
    BDBT_NodeReference_t nr = 0;
    BDBT_KeyQuery(&bdbt, &key, &KeyIndex, &nr);
    if(KeyIndex != set_KeySize){
      WriteOut("fail0 %lx %lx\n", i, KeyIndex);
      break;
    }
    if(nr != i){
      WriteOut("fail1 %lx %lx\n", nr, i);
      break;
    }
  }
  #endif

  #if set_test_KeyInAfterRemove == 1
  for(uint32_t i = 0; i < set_test_size / 2; i++){
    uint32_t key = i;
    BDBT_KeyIn(&bdbt, &key, 0, i);
  }
  #endif

  #if set_test_VerifyKeyIn == 1
  for(uint32_t i = 0; i < set_test_size; i++){
    uint32_t key = i;
    BDBT_KeySize_t KeyIndex;
    BDBT_NodeReference_t nr = 0;
    BDBT_KeyQuery(&bdbt, &key, &KeyIndex, &nr);
    if(KeyIndex != set_KeySize){
      WriteOut("fail0 %lx %lx\n", i, KeyIndex);
      break;
    }
    if(nr != i){
      WriteOut("fail1 %lx %lx\n", nr, i);
      break;
    }
  }
  #endif

  BDBT_close(&bdbt);

  return 0;
}
