#ifdef BLL_set_namespace
  namespace BLL_set_namespace {
#endif

#if BLL_set_StoreFormat == 1
  /* TODO can be more smaller */
  typedef BLL_set_type_node _P(BlockIndex_t);

  #if BLL_set_StoreFormat1_ElementPerBlock <= 0xff
    typedef uint8_t _P(BlockModulo_t);
  #elif BLL_set_StoreFormat1_ElementPerBlock <= 0xffff
    typedef uint16_t _P(BlockModulo_t);
  #elif BLL_set_StoreFormat1_ElementPerBlock <= 0xffffffff
    typedef uint32_t _P(BlockModulo_t);
  #else
    #error no
  #endif
#endif

#if BLL_set_PadNode == 0
  #pragma pack(push, 1)
#endif
  BLL_StructBegin(_P(Node_t))
    _P(Node_t)(const _P(Node_t)&) = default;
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
    //union{
      struct{
        BLL_set_node_data
      }data;
      #if BLL_set_Link == 0
        /* used for empty next */
        _P(NodeReference_t) NextNodeReference;
      #endif
   // };
  #endif
BLL_StructEnd(_P(Node_t))
#if BLL_set_PadNode == 0
  #pragma pack(pop)
#endif

#if BLL_set_StoreFormat == 0
  #define BVEC_set_prefix _P(_NodeList)
  #define BVEC_set_NodeType BLL_set_type_node
  #ifdef BLL_set_node_data
    #define BVEC_set_NodeData _P(Node_t)
  #endif
  #include _BLL_INCLUDE(BVEC/BVEC.h)
#elif BLL_set_StoreFormat == 1
  #ifndef BLL_set_node_data
    #error not yet
  #endif
  #define BVEC_set_prefix _P(_BlockList)
  #define BVEC_set_NodeType BLL_set_type_node
  #define BVEC_set_NodeData _P(Node_t) *
  #include _BLL_INCLUDE(BVEC/BVEC.h)
#endif

BLL_StructBegin(_P(t))
  #if BLL_set_StoreFormat == 0
    _P(_NodeList_t) NodeList;
  #elif BLL_set_StoreFormat == 1
    _P(_BlockList_t) BlockList;
  #endif
  #if BLL_set_StoreFormat == 1
    BLL_set_type_node NodeCurrent;
  #endif
  #if BLL_set_Link == 1
    _P(NodeReference_t) src;
    _P(NodeReference_t) dst;
  #endif
  struct{
    _P(NodeReference_t) c;
    BLL_set_type_node p;
  }e;
  #ifndef BLL_set_node_data
    uint32_t NodeSize;
  #endif
  #if BLL_set_SafeNext
    _P(NodeReference_t) SafeNext;
  #endif
BLL_StructEnd(_P(t))

#ifdef BLL_set_namespace
  }
#endif
