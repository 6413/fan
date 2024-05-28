#include "CheckLogic.h"

#if defined(BLL_set_NodeData) || defined(BLL_set_NodeDataType)
  #define _BLL_HaveConstantNodeData
#endif

#define _P(p0) CONCAT3(BLL_set_prefix, _, p0)

#if BLL_set_StructFormat == 0
  #define BLL_StructBegin(n) typedef struct{
  #define BLL_StructEnd(n) }n;
#elif BLL_set_StructFormat == 1
  #define BLL_StructBegin(n) struct n{
  #define BLL_StructEnd(n) };
#endif

#if BLL_set_declare_NodeReference == 1
  #include "NodeReference.h"
#endif
#if BLL_set_declare_rest == 1
  #if BLL_set_Language == 0
    #define _BLL_this This
    #define _BLL_fdec(rtype, name, ...) static rtype name(_P(t) *This, ##__VA_ARGS__)
    #define _BLL_fcall(name, ...) name(This, ##__VA_ARGS__)
  #elif BLL_set_Language == 1
    #define _BLL_this this
    #define _BLL_fdec(rtype, name, ...) rtype name(__VA_ARGS__)
    #define _BLL_fcall(name, ...) name(__VA_ARGS__)
  #else
    #error ?
  #endif

  #if !defined(_BLL_HaveConstantNodeData) && !defined(BLL_set_MultipleType_Sizes)
    #define _BLL_fdecnds(rtype, name, ...) _BLL_fdec(rtype, name, BLL_set_NodeSizeType NodeDataSize, ##__VA_ARGS__)
  #else
    #define _BLL_fdecnds _BLL_fdec
  #endif

  #if defined(BLL_set_MultipleType_Sizes)
    #define _BLL_fdecpi(rtype, name, ...) _BLL_fdec(rtype, name, uintptr_t PointerIndex, ##__VA_ARGS__)
    #define _BLL_fcallpi(name, ...) _BLL_fcall(name, uintptr_t PointerIndex, ##__VA_ARGS__)
  #else
    #define _BLL_fdecpi _BLL_fdec
    #define _BLL_fcallpi _BLL_fcall
  #endif

  #include "rest.h"

  #undef _BLL_this
  #undef _BLL_fdec
  #undef _BLL_fcall
  #undef _BLL_fdecnds
  #undef _BLL_fdecpi
  #undef _BLL_fcallpi
#endif

#undef BLL_StructBegin
#undef BLL_StructEnd

#undef _P

#ifdef _BLL_HaveConstantNodeData
  #undef _BLL_HaveConstantNodeData
#endif
