#ifdef BDBT_set_namespace
  namespace BDBT_set_namespace {
#endif

typedef BDBT_set_type_node _BDBT_P(NodeReference_t);

#if _BDBT_set_ElementPerNode <= 0xff
  typedef uint8_t _BDBT_P(NodeEIT_t);
#elif _BDBT_set_ElementPerNode < 0xffff
  typedef uint16_t _BDBT_P(NodeEIT_t);
#else
  #error no
#endif

#if BDBT_set_PadNode == 0
  #pragma pack(push, 1)
#endif
typedef struct{
  _BDBT_P(NodeReference_t) n[_BDBT_set_ElementPerNode];
}_BDBT_P(Node_t);
#if BDBT_set_PadNode == 0
  #pragma pack(pop)
#endif

typedef struct{
  #if BDBT_set_StoreFormat == 0
    #if BDBT_set_BaseLibrary == 0
      VEC_t nodes;
    #elif BDBT_set_BaseLibrary == 1
      fan::hector_t<_BDBT_P(Node_t)> nodes;
    #endif
  #endif
  struct{
    _BDBT_P(NodeReference_t) c;
    _BDBT_P(NodeReference_t) p;
  }e;
}_BDBT_P(t);

#ifdef BDBT_set_namespace
  }
#endif
