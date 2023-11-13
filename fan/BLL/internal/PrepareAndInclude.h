#if defined(BLL_set_NodeData) || defined(BLL_set_NodeDataType)
  #if defined(BLL_set_NodeData) && defined(BLL_set_NodeDataType)
    #error no
  #endif
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
  /* _BLL_POFTWBIT; prefix of function that would be inside type */
  /* _BLL_SOFTWBIT; settings of function that would be inside type */
  /* _BLL_PBLLTFF; pass bll type for functon */
  /* _BLL_PBLLTFFC; _BLL_PBLLTFF but with comma */
  /* _BLL_DBLLTFF; declare bll type for functon */
  /* _BLL_DBLLTFFC; _BLL_DBLLTFF but with comma */
  /* _BLL_PIL0; pass if language 0 */
  #if BLL_set_Language == 0
    #define _BLL_POFTWBIT(p0) _P(p0)
    #define _BLL_SOFTWBIT static
    #define _BLL_PBLLTFF _pList
    #define _BLL_PBLLTFFC _BLL_PBLLTFF,
    #define _BLL_DBLLTFF _P(t) *_BLL_PBLLTFF
    #define _BLL_DBLLTFFC _BLL_DBLLTFF,
    #define _BLL_GetList _BLL_PBLLTFF
    #define _BLL_PIL0(p0) p0
  #elif BLL_set_Language == 1
    #define _BLL_POFTWBIT(p0) p0
    #define _BLL_SOFTWBIT
    #define _BLL_PBLLTFF
    #define _BLL_PBLLTFFC
    #define _BLL_DBLLTFF
    #define _BLL_DBLLTFFC
    #define _BLL_GetList this
    #define _BLL_PIL0(p0)
  #else
    #error ?
  #endif

  #include "rest.h"

  #undef _BLL_POFTWBIT
  #undef _BLL_SOFTWBIT
  #undef _BLL_PBLLTFF
  #undef _BLL_PBLLTFFC
  #undef _BLL_DBLLTFF
  #undef _BLL_DBLLTFFC
  #undef _BLL_GetList
  #undef _BLL_PIL0
#endif

#undef BLL_StructBegin
#undef BLL_StructEnd

#undef _P

#ifdef _BLL_HaveConstantNodeData
  #undef _BLL_HaveConstantNodeData
#endif
