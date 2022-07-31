#ifdef BLL_set_namespace
  namespace BLL_set_namespace {
#endif

typedef BLL_set_type_node _P(NodeReference_t);

#if BLL_set_PadNode == 0
  #pragma pack(push, 1)
#endif
typedef struct{
  #if BLL_set_Link == 1
    #if BLL_set_PreferNextFirst == 1
      _P(NodeReference_t) NextNodeReference;
      _P(NodeReference_t) PrevNodeReference;
    #else
      _P(NodeReference_t) PrevNodeReference;
      _P(NodeReference_t) NextNodeReference;
    #endif
  #endif
  #ifdef BLL_set_node_data
    union{
      struct{
        BLL_set_node_data
      }data;
      #if BLL_set_Link == 0
        /* used for empty next */
        _P(NodeReference_t) NextNodeReference;
      #endif
    };
  #endif
}_P(Node_t);
#if BLL_set_PadNode == 0
  #pragma pack(pop)
#endif

typedef struct{
  #if BLL_set_StoreFormat == 0
    #if BLL_set_BaseLibrary == 0
      VEC_t nodes;
    #elif BLL_set_BaseLibrary == 1
      fan::hector_t<_P(Node_t)> nodes;
    #endif
  #endif
  #if BLL_set_Link == 1
    _P(NodeReference_t) src;
    _P(NodeReference_t) dst;
  #endif
  struct{
    _P(NodeReference_t) c;
    _P(NodeReference_t) p;
  }e;
  #ifndef BLL_set_node_data
    uint32_t NodeSize;
  #endif
  #if BLL_set_SafeNext
    _P(NodeReference_t) SafeNext;
  #endif
}_P(t);

#ifdef BLL_set_namespace
  }
#endif
