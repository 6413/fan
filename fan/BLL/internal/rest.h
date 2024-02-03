#ifdef BLL_set_namespace
  namespace BLL_set_namespace {
#endif

/* too much overenginer here */
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
#if defined(_BLL_HaveConstantNodeData)
  #if defined(BLL_set_NodeData)
    BLL_StructBegin(_P(NodeData_t))
      BLL_set_NodeData
    BLL_StructEnd(_P(NodeData_t))
  #elif defined(BLL_set_NodeDataType)
    typedef BLL_set_NodeDataType _P(NodeData_t);
  #else
    #error not possible
  #endif
#else
  typedef void _P(NodeData_t);
#endif
BLL_StructBegin(_P(Node_t))
  #if BLL_set_Link == 1
    #if BLL_set_PreferNextFirst == 1
      _P(NodeReference_t) NextNodeReference;
      _P(NodeReference_t) PrevNodeReference;
    #else
      _P(NodeReference_t) PrevNodeReference;
      _P(NodeReference_t) NextNodeReference;
    #endif
  #endif
  #ifdef _BLL_HaveConstantNodeData
    #ifdef BLL_set_CPP_Node_ConstructDestruct
      _P(NodeData_t) data;
      #if BLL_set_Link == 0
        uint8_t _PaddingForNextRecycled[
          sizeof(_P(NodeData_t)) < sizeof(_P(NodeReference_t)) ?
            sizeof(_P(NodeReference_t)) - sizeof(_P(NodeData_t)) :
            0
        ];
      #endif
    #else
      union{
        _P(NodeData_t) data;
        #if BLL_set_Link == 0
          /* used for empty next */
          _P(NodeReference_t) _NextRecycled;
        #endif
      };
    #endif
  #endif
BLL_StructEnd(_P(Node_t))
#if BLL_set_PadNode == 0
  #pragma pack(pop)
#endif

#if defined(BLL_set_MultipleType_Sizes)
  static constexpr auto _P(MultipleType_MakeArray)(auto... x) {
    std::array<std::tuple_element_t<0, std::tuple<decltype(x)...>>, sizeof...(x)> arr = {
      x...
    };
    #if BLL_set_Link == 1
    arr[BLL_set_MultipleType_LinkIndex] += sizeof(_P(NodeReference_t)) * 2;
    /*
    uintptr_t NeededSize = sizeof(_P(NodeReference_t)) * 2;
    bool found = false;
    for(uintptr_t i = 0; i < arr.size(); i++){
      if(arr[i] >= NeededSize){
        found = true;
      }
    }
    if(found == false){
      uintptr_t BestDiff = -1;
      uintptr_t BestIndex;
      for(uintptr_t i = 0; i < arr.size(); i++){
        if(NeededSize - arr[i])
      }
    }
    */
    #endif
    return arr;
  }
  static constexpr auto _P(_MultipleType_Sizes) = _P(MultipleType_MakeArray)(BLL_set_MultipleType_Sizes);
#endif

#if BLL_set_StoreFormat == 0
  #define BVEC_set_prefix _P(_NodeList)
  #define BVEC_set_NodeType BLL_set_type_node
  #if defined(BLL_set_MultipleType_Sizes)
    #define BVEC_set_MultipleType
    #define BVEC_set_MultipleType_SizeArray _P(_MultipleType_Sizes)
  #elif defined(_BLL_HaveConstantNodeData)
    #define BVEC_set_NodeData _P(Node_t)
  #endif
  #if defined(BLL_set_CPP_CopyAtPointerChange)
    #define BVEC_set_HandleAllocate 0
  #endif
  #define BVEC_set_alloc_open BLL_set_alloc_open
  #define BVEC_set_alloc_resize BLL_set_alloc_resize
  #define BVEC_set_alloc_close BLL_set_alloc_close
  #include "../../BVEC/BVEC.h"
#elif BLL_set_StoreFormat == 1
  #ifndef _BLL_HaveConstantNodeData
    #error not yet
  #endif
  #if defined(BLL_set_MultipleType_Sizes)
    #error not yet
  #endif
  #define BVEC_set_prefix _P(_BlockList)
  #define BVEC_set_NodeType BLL_set_type_node
  #define BVEC_set_NodeData _P(Node_t) *
  #define BVEC_set_alloc_open BLL_set_alloc_open
  #define BVEC_set_alloc_close BLL_set_alloc_close
  #include "../../BVEC/BVEC.h"
#endif

#if defined(BLL_set_MultipleType_Sizes)
  consteval
  static
  uintptr_t
  _P(_MultipleType_GetLinkIndex)(
  ){
    uintptr_t BestSize = -1;
    uintptr_t BestIndex;
    for(
      uintptr_t i = 0;
      i < sizeof(_P(_MultipleType_Sizes)) / sizeof(_P(_MultipleType_Sizes)[0]);
      i++
    ){
      if(_P(_MultipleType_Sizes)[i] < BestSize){
        BestSize = _P(_MultipleType_Sizes)[i];
        BestIndex = i;
      }
    }
    return BestIndex;
  }
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
  #ifndef _BLL_HaveConstantNodeData
    uint32_t NodeSize;
  #endif
  #if BLL_set_Link == 1
    _P(NodeReference_t) src;
    _P(NodeReference_t) dst;
  #endif
  struct{
    _P(NodeReference_t) c;
    BLL_set_type_node p;
  }e;
  #if BLL_set_SafeNext == 1
    _P(NodeReference_t) SafeNext;
  #elif BLL_set_SafeNext > 1
    uint8_t SafeNextCount;
    _P(NodeReference_t) SafeNext[BLL_set_SafeNext];
  #endif
#if BLL_set_Language == 0
  BLL_StructEnd(_P(t))
#else
  using nd_t = _P(NodeData_t);
  using nr_t = _P(NodeReference_t);
#endif

_BLL_SOFTWBIT
bool
_BLL_POFTWBIT(IsNodeReferenceEqual)
(
  _P(NodeReference_t) p0,
  _P(NodeReference_t) p1
){
  return p0.NRI == p1.NRI;
}

_BLL_SOFTWBIT
BLL_set_NodeSizeType
_BLL_POFTWBIT(GetNodeSize)
(
  _BLL_DBLLTFF
  #if defined(BLL_set_MultipleType_Sizes)
    _BLL_PIL0(__ca__) uintptr_t PointerIndex
  #endif
){
  /* TODO */
  /* this function doesnt check store format */
  /* can give compile error in future */

  #if defined(BLL_set_MultipleType_Sizes)
    return _P(_NodeList_GetNodeSize)(&_BLL_GetList->NodeList, PointerIndex);
  #elif defined(_BLL_HaveConstantNodeData)
    return sizeof(_P(Node_t));
  #else
    return _P(_NodeList_GetNodeSize)(&_BLL_GetList->NodeList);
  #endif
}

#if BLL_set_StoreFormat == 0
  _BLL_SOFTWBIT
  BLL_set_type_node
  _BLL_POFTWBIT(GetAmountOfAllocated)(
    _BLL_DBLLTFF
  ){
    return _BLL_GetList->NodeList.Possible;
  }
#endif

#if BLL_set_Language == 1
  _BLL_SOFTWBIT
  _P(NodeReference_t)
  _BLL_POFTWBIT(gnric)
  (
  ){
    return _P(gnric)();
  }
  _BLL_SOFTWBIT
  bool
  _BLL_POFTWBIT(inric)
  (
    _P(NodeReference_t) NR
  ){
    return _P(inric)(NR);
  }
#endif

/* is node reference invalid */
_BLL_SOFTWBIT
bool
_BLL_POFTWBIT(inri)
(
  _BLL_DBLLTFFC
  _P(NodeReference_t) NodeReference
){
  #if BLL_set_StoreFormat == 0
    return NodeReference.NRI >= _BLL_GetList->NodeList.Current;
  #elif BLL_set_StoreFormat == 1
    return NodeReference.NRI >= _BLL_GetList->NodeCurrent;
  #endif
}

_BLL_SOFTWBIT
_P(Node_t) *
_BLL_POFTWBIT(_GetNodeByReference)
(
  _BLL_DBLLTFFC
  _P(NodeReference_t) NodeReference
  #if defined(BLL_set_MultipleType_Sizes)
    , uintptr_t PointerIndex
  #endif
){
  #if BLL_set_StoreFormat == 0
    #if defined(BLL_set_MultipleType_Sizes)
      return (_P(Node_t) *)_P(_NodeList_GetNode)(&_BLL_GetList->NodeList, NodeReference.NRI, PointerIndex);
    #else
      return (_P(Node_t) *)_P(_NodeList_GetNode)(&_BLL_GetList->NodeList, NodeReference.NRI);
    #endif
  #elif BLL_set_StoreFormat == 1
    #if defined(_BLL_HaveConstantNodeData)
      _P(BlockIndex_t) bi = NodeReference.NRI / BLL_set_StoreFormat1_ElementPerBlock;
      _P(BlockModulo_t) bm = NodeReference.NRI % BLL_set_StoreFormat1_ElementPerBlock;
      /* TODO this looks like mess check it */
      _P(Node_t) *n = &((_P(Node_t) *)((void **)&_BLL_GetList->BlockList.ptr[0])[bi])[bm];
      return n;
    #else
      #error not implemented yet
    #endif
  #endif
}

/* get node reference th of node reference */
_BLL_SOFTWBIT
_P(NodeReference_t) *
_BLL_POFTWBIT(_GetNRTHOfNR)
(
  _BLL_DBLLTFFC
  _P(NodeReference_t) NodeReference,
  BLL_set_type_node TH
){
  #if defined(BLL_set_MultipleType_Sizes)
    _P(Node_t) *n = _BLL_POFTWBIT(_GetNodeByReference)(_BLL_PBLLTFFC NodeReference, _P(_MultipleType_GetLinkIndex)());
  #else
    _P(Node_t) *n = _BLL_POFTWBIT(_GetNodeByReference)(_BLL_PBLLTFFC NodeReference);
  #endif
  return &((_P(NodeReference_t) *)n)[TH];
}

/* get recycle node reference of node reference */
_BLL_SOFTWBIT
_P(NodeReference_t) *
_BLL_POFTWBIT(_grecnrofnr)
(
  _BLL_DBLLTFFC
  _P(NodeReference_t) NodeReference
){
  return _BLL_POFTWBIT(_GetNRTHOfNR)(_BLL_PBLLTFFC NodeReference, 0);
}

#if BLL_set_IsNodeRecycled == 1
  _BLL_SOFTWBIT
  bool
  _BLL_POFTWBIT(IsNodeReferenceRecycled)
  (
    _BLL_DBLLTFFC
    _P(NodeReference_t) NodeReference
  ){
    return _BLL_POFTWBIT(_GetNRTHOfNR)(_BLL_PBLLTFFC NodeReference, 1)->NRI == (BLL_set_type_node)-1;
  }
#endif

_BLL_SOFTWBIT
_P(Node_t) *
_BLL_POFTWBIT(GetNodeByReference)
(
  _BLL_DBLLTFFC
  _P(NodeReference_t) nr
  #if defined(BLL_set_MultipleType_Sizes)
    , uintptr_t PointerIndex
  #endif
){
  #if BLL_set_debug_InvalidAction == 1
    if(nr.NRI >= _BLL_GetList->NodeList.Current){
      __abort();
    }
  #endif
  #if defined(BLL_set_MultipleType_Sizes)
    _P(Node_t) *Node = _BLL_POFTWBIT(_GetNodeByReference)(_BLL_PBLLTFFC nr, PointerIndex);
  #else
    _P(Node_t) *Node = _BLL_POFTWBIT(_GetNodeByReference)(_BLL_PBLLTFFC nr);
  #endif
  #if BLL_set_debug_InvalidAction == 1
    do{
      #if BLL_set_debug_InvalidAction_srcAccess == 1
        if(nr.NRI == _BLL_GetList->src.NRI){
          __abort();
        }
      #endif
      #if BLL_set_debug_InvalidAction_dstAccess == 1
        if(nr.NRI == _BLL_GetList->dst.NRI){
          __abort();
        }
      #endif
    }while(0);
  #endif
  return Node;
}

_BLL_SOFTWBIT
BLL_set_type_node
_BLL_POFTWBIT(Usage)
(
  _BLL_DBLLTFF
){
  #if BLL_set_StoreFormat == 0
    #if BLL_set_Link == 0
      return _BLL_GetList->NodeList.Current - _BLL_GetList->e.p;
    #elif BLL_set_Link == 1
      return _BLL_GetList->NodeList.Current - _BLL_GetList->e.p - 2;
    #endif
  #elif BLL_set_StoreFormat == 1
    #if BLL_set_Link == 0
      return _BLL_GetList->NodeCurrent - _BLL_GetList->e.p;
    #elif BLL_set_Link == 1
      return _BLL_GetList->NodeCurrent - _BLL_GetList->e.p - 2;
    #endif
  #endif
}

#if BLL_set_StoreFormat == 1
  _BLL_SOFTWBIT
  void
  _BLL_POFTWBIT(_PushNewBlock)
  (
    _BLL_DBLLTFF
  ){
    _P(Node_t) *n = (_P(Node_t) *)BLL_set_alloc_open(
      sizeof(_P(Node_t)) * BLL_set_StoreFormat1_ElementPerBlock);
    _P(_BlockList_Add)(&_BLL_GetList->BlockList, &n);
  }
#endif

_BLL_SOFTWBIT
void
_BLL_POFTWBIT(_Node_Construct)
(
  _BLL_DBLLTFFC
  _P(NodeReference_t) NodeReference
){
  #ifdef BLL_set_CPP_Node_ConstructDestruct
    _P(Node_t) *n = _BLL_POFTWBIT(_GetNodeByReference)(_BLL_PBLLTFFC NodeReference);
    new (&n->data) _P(NodeData_t);
  #endif
}
_BLL_SOFTWBIT
void
_BLL_POFTWBIT(_Node_Destruct)
(
  _BLL_DBLLTFFC
  _P(NodeReference_t) NodeReference
){
  #ifdef BLL_set_CPP_Node_ConstructDestruct
    /* TODO _ with getnode... or what? */
    _P(Node_t) *n = _BLL_POFTWBIT(GetNodeByReference)(_BLL_PBLLTFFC NodeReference);
    ((_P(NodeData_t) *)&n->data)->~_P(NodeData_t)();
  #endif
}

_BLL_SOFTWBIT
void
_BLL_POFTWBIT(_Recycle)
(
  _BLL_DBLLTFFC
  _P(NodeReference_t) NodeReference
){
  _P(NodeReference_t) *NextRecycled = _BLL_POFTWBIT(_grecnrofnr)(_BLL_PBLLTFFC NodeReference);

  *NextRecycled = _BLL_GetList->e.c;
  #if BLL_set_IsNodeRecycled == 1
    _BLL_POFTWBIT(_GetNRTHOfNR)(_BLL_PBLLTFFC NodeReference, 1)->NRI = (BLL_set_type_node)-1;
  #endif
  _BLL_GetList->e.c = NodeReference;
  _BLL_GetList->e.p++;
}

_BLL_SOFTWBIT
void
_BLL_POFTWBIT(Recycle)
(
  _BLL_DBLLTFFC
  _P(NodeReference_t) NodeReference
){
  _BLL_POFTWBIT(_Node_Destruct)(_BLL_PBLLTFFC NodeReference);
  _BLL_POFTWBIT(_Recycle)(_BLL_PBLLTFFC NodeReference);
}

#ifdef BLL_set_CPP_Node_ConstructDestruct
  _BLL_SOFTWBIT
  void
  _BLL_POFTWBIT(Recycle_NoDestruct)
  (
    _BLL_DBLLTFFC
    _P(NodeReference_t) NodeReference
  ){
    _BLL_POFTWBIT(_Recycle)(_BLL_PBLLTFFC NodeReference);
  }
#endif

_BLL_SOFTWBIT
_P(NodeReference_t)
_BLL_POFTWBIT(_NewNode_empty_NoConstruct)
(
  _BLL_DBLLTFF
){
  _P(NodeReference_t) NodeReference = _BLL_GetList->e.c;
  _BLL_GetList->e.c = *_BLL_POFTWBIT(_grecnrofnr)(_BLL_PBLLTFFC NodeReference);
  _BLL_GetList->e.p--;
  return NodeReference;
}
_BLL_SOFTWBIT
_P(NodeReference_t)
_BLL_POFTWBIT(NewNode_empty)
(
  _BLL_DBLLTFF
){
  _P(NodeReference_t) NodeReference = _BLL_POFTWBIT(_NewNode_empty_NoConstruct)(_BLL_PBLLTFF);
  _BLL_POFTWBIT(_Node_Construct)(_BLL_PBLLTFFC NodeReference);
  return NodeReference;
}
_BLL_SOFTWBIT
_P(NodeReference_t)
_BLL_POFTWBIT(_NewNode_alloc_NoConstruct)
(
  _BLL_DBLLTFF
){
  _P(NodeReference_t) r;
  #if BLL_set_StoreFormat == 0
    r.NRI = _BLL_GetList->NodeList.Current;

    #ifdef BLL_set_CPP_CopyAtPointerChange
      if(NodeList.Current == NodeList.Possible){
        _BLL_POFTWBIT(AllocateNewBuffer)(_BLL_PBLLTFFC NodeList.Current + 1);
      }
    #endif

    _P(_NodeList_AddEmpty)(&_BLL_GetList->NodeList, 1);

  #elif BLL_set_StoreFormat == 1
    if(_BLL_GetList->NodeCurrent % BLL_set_StoreFormat1_ElementPerBlock == 0){
      _BLL_POFTWBIT(_PushNewBlock)(_BLL_PBLLTFF);
    }
    r.NRI = _BLL_GetList->NodeCurrent++;
  #endif
  return r;
}
_BLL_SOFTWBIT
_P(NodeReference_t)
_BLL_POFTWBIT(NewNode_alloc)
(
  _BLL_DBLLTFF
){
  _P(NodeReference_t) NodeReference = _BLL_POFTWBIT(_NewNode_alloc_NoConstruct)(_BLL_PBLLTFF);
  _BLL_POFTWBIT(_Node_Construct)(_BLL_PBLLTFFC NodeReference);
  return NodeReference;
}
_BLL_SOFTWBIT
_P(NodeReference_t)
_BLL_POFTWBIT(_NewNode_NoConstruct)
(
  _BLL_DBLLTFF
){
  _P(NodeReference_t) NodeReference;
  if(_BLL_GetList->e.p){
    NodeReference = _BLL_POFTWBIT(_NewNode_empty_NoConstruct)(_BLL_PBLLTFF);
  }
  else{
    NodeReference = _BLL_POFTWBIT(_NewNode_alloc_NoConstruct)(_BLL_PBLLTFF);
  }
  return NodeReference;
}
_BLL_SOFTWBIT
_P(NodeReference_t)
_BLL_POFTWBIT(NewNode)
(
  _BLL_DBLLTFF
){
  _P(NodeReference_t) NodeReference = _BLL_POFTWBIT(_NewNode_NoConstruct)(_BLL_PBLLTFF);
  #if BLL_set_IsNodeRecycled == 1
    _BLL_POFTWBIT(_GetNRTHOfNR)(_BLL_PBLLTFFC NodeReference, 1)->NRI = (BLL_set_type_node)-2;
  #endif
  _BLL_POFTWBIT(_Node_Construct)(_BLL_PBLLTFFC NodeReference);
  return NodeReference;
}

#if BLL_set_Link == 1
  /* get linked node */
  _BLL_SOFTWBIT
  _P(Node_t) *
  _BLL_POFTWBIT(gln)(
    _BLL_DBLLTFFC
    _P(NodeReference_t) nr
  ){
    #if defined(BLL_set_MultipleType_Sizes)
      return _BLL_POFTWBIT(GetNodeByReference)(_BLL_PBLLTFFC nr, BLL_set_MultipleType_LinkIndex);
    #else
      return _BLL_POFTWBIT(GetNodeByReference)(_BLL_PBLLTFFC nr);
    #endif
  }
  /* get linked node */
  _BLL_SOFTWBIT
  _P(Node_t) *
  _BLL_POFTWBIT(_gln)(
    _BLL_DBLLTFFC
    _P(NodeReference_t) nr
  ){
    #if defined(BLL_set_MultipleType_Sizes)
      return _BLL_POFTWBIT(_GetNodeByReference)(_BLL_PBLLTFFC nr, BLL_set_MultipleType_LinkIndex);
    #else
      return _BLL_POFTWBIT(_GetNodeByReference)(_BLL_PBLLTFFC nr);
    #endif
  }

  _BLL_SOFTWBIT
  _P(NodeReference_t)
  _BLL_POFTWBIT(NewNodeFirst_empty)
  (
    _BLL_DBLLTFF
  ){
    _P(NodeReference_t) NodeReference = _BLL_POFTWBIT(_NewNode_empty_NoConstruct)(_BLL_PBLLTFF);
    _P(NodeReference_t) srcNodeReference = _BLL_GetList->src;
    _BLL_POFTWBIT(_gln)(_BLL_PBLLTFFC NodeReference)->NextNodeReference = srcNodeReference;
    _BLL_POFTWBIT(_gln)(_BLL_PBLLTFFC srcNodeReference)->PrevNodeReference = NodeReference;
    _BLL_GetList->src = NodeReference;
    _BLL_POFTWBIT(_Node_Construct)(_BLL_PBLLTFFC srcNodeReference);
    return srcNodeReference;
  }
  _BLL_SOFTWBIT
  _P(NodeReference_t)
  _BLL_POFTWBIT(NewNodeFirst_alloc)
  (
    _BLL_DBLLTFF
  ){
    _P(NodeReference_t) NodeReference = _BLL_POFTWBIT(_NewNode_alloc_NoConstruct)(_BLL_PBLLTFF);
    _P(NodeReference_t) srcNodeReference = _BLL_GetList->src;
    _BLL_POFTWBIT(_gln)(_BLL_PBLLTFFC NodeReference)->NextNodeReference = srcNodeReference;
    _BLL_POFTWBIT(_gln)(_BLL_PBLLTFFC srcNodeReference)->PrevNodeReference = NodeReference;
    _BLL_GetList->src = NodeReference;
    _BLL_POFTWBIT(_Node_Construct)(_BLL_PBLLTFFC srcNodeReference);
    return srcNodeReference;
  }
  _BLL_SOFTWBIT
  _P(NodeReference_t)
  _BLL_POFTWBIT(NewNodeFirst)
  (
    _BLL_DBLLTFF
  ){
    if(_BLL_GetList->e.p){
      return _BLL_POFTWBIT(NewNodeFirst_empty)(_BLL_PBLLTFF);
    }
    else{
      return _BLL_POFTWBIT(NewNodeFirst_alloc)(_BLL_PBLLTFF);
    }
  }
  _BLL_SOFTWBIT
  _P(NodeReference_t)
  _BLL_POFTWBIT(NewNodeLast_empty)
  (
    _BLL_DBLLTFF
  ){
    _P(NodeReference_t) NodeReference = _BLL_POFTWBIT(_NewNode_empty_NoConstruct)(_BLL_PBLLTFF);
    _P(NodeReference_t) dstNodeReference = _BLL_GetList->dst;
    _BLL_POFTWBIT(_gln)(_BLL_PBLLTFFC NodeReference)->PrevNodeReference = dstNodeReference;
    _BLL_POFTWBIT(_gln)(_BLL_PBLLTFFC dstNodeReference)->NextNodeReference = NodeReference;
    _BLL_GetList->dst = NodeReference;
    _BLL_POFTWBIT(_Node_Construct)(_BLL_PBLLTFFC dstNodeReference);
    return dstNodeReference;
  }
  _BLL_SOFTWBIT
  _P(NodeReference_t)
  _BLL_POFTWBIT(NewNodeLast_alloc)
  (
    _BLL_DBLLTFF
  ){
    _P(NodeReference_t) NodeReference = _BLL_POFTWBIT(_NewNode_alloc_NoConstruct)(_BLL_PBLLTFF);
    _P(NodeReference_t) dstNodeReference = _BLL_GetList->dst;
    _BLL_POFTWBIT(_gln)(_BLL_PBLLTFFC NodeReference)->PrevNodeReference = dstNodeReference;
    _BLL_POFTWBIT(_gln)(_BLL_PBLLTFFC dstNodeReference)->NextNodeReference = NodeReference;
    _BLL_GetList->dst = NodeReference;
    _BLL_POFTWBIT(_Node_Construct)(_BLL_PBLLTFFC dstNodeReference);
    return dstNodeReference;
  }
  _BLL_SOFTWBIT
  _P(NodeReference_t)
  _BLL_POFTWBIT(NewNodeLast)
  (
    _BLL_DBLLTFF
  ){
    if(_BLL_GetList->e.p){
      return _BLL_POFTWBIT(NewNodeLast_empty)(_BLL_PBLLTFF);
    }
    else{
      return _BLL_POFTWBIT(NewNodeLast_alloc)(_BLL_PBLLTFF);
    }
  }

  _BLL_SOFTWBIT
  bool
  _BLL_POFTWBIT(IsNRSentienel)
  (
    _BLL_DBLLTFFC
    _P(NodeReference_t) NR
  ){
    if(NR.NRI == _BLL_GetList->src.NRI){
      return 1;
    }
    if(NR.NRI == _BLL_GetList->dst.NRI){
      return 1;
    }
    return 0;
  }

  _BLL_SOFTWBIT
  void
  _BLL_POFTWBIT(linkNext)
  (
    _BLL_DBLLTFFC
    _P(NodeReference_t) srcNodeReference,
    _P(NodeReference_t) dstNodeReference
  ){
    _P(Node_t) *srcNode = _BLL_POFTWBIT(gln)(_BLL_PBLLTFFC srcNodeReference);
    _P(Node_t) *dstNode = _BLL_POFTWBIT(_gln)(_BLL_PBLLTFFC dstNodeReference);
    _P(NodeReference_t) nextNodeReference = srcNode->NextNodeReference;
    _P(Node_t) *nextNode = _BLL_POFTWBIT(gln)(_BLL_PBLLTFFC nextNodeReference);
    srcNode->NextNodeReference = dstNodeReference;
    dstNode->PrevNodeReference = srcNodeReference;
    dstNode->NextNodeReference = nextNodeReference;
    nextNode->PrevNodeReference = dstNodeReference;
  }
  _BLL_SOFTWBIT
  void
  _BLL_POFTWBIT(linkPrev)
  (
    _BLL_DBLLTFFC
    _P(NodeReference_t) srcNodeReference,
    _P(NodeReference_t) dstNodeReference
  ){
    _P(Node_t) *srcNode = _BLL_POFTWBIT(gln)(_BLL_PBLLTFFC srcNodeReference);
    _P(NodeReference_t) prevNodeReference = srcNode->PrevNodeReference;
    _P(Node_t) *prevNode = _BLL_POFTWBIT(gln)(_BLL_PBLLTFFC prevNodeReference);
    prevNode->NextNodeReference = dstNodeReference;
    _P(Node_t) *dstNode = _BLL_POFTWBIT(_gln)(_BLL_PBLLTFFC dstNodeReference);
    dstNode->PrevNodeReference = prevNodeReference;
    dstNode->NextNodeReference = srcNodeReference;
    srcNode->PrevNodeReference = dstNodeReference;
  }

  _BLL_SOFTWBIT
  void
  _BLL_POFTWBIT(Unlink)
  (
    _BLL_DBLLTFFC
    _P(NodeReference_t) NodeReference
  ){
    #if BLL_set_debug_InvalidAction >= 1
      if(_BLL_POFTWBIT(IsNRSentienel)(_BLL_PBLLTFFC NodeReference) == 1){
        __abort();
      }
      if(_BLL_POFTWBIT(IsNodeReferenceRecycled)(_BLL_PBLLTFFC NodeReference) == 1){
        __abort();
      }
    #endif

    _P(Node_t) *Node = _BLL_POFTWBIT(gln)(_BLL_PBLLTFFC NodeReference);

    #if BLL_set_SafeNext == 1
      if(_BLL_GetList->SafeNext == NodeReference){
        _BLL_GetList->SafeNext = Node->PrevNodeReference;
      }
    #elif BLL_set_SafeNext > 1
      for(uint8_t i = 0; i < _BLL_GetList->SafeNextCount; i++){
        if(_BLL_GetList->SafeNext[i] == NodeReference){
          _BLL_GetList->SafeNext[i] = Node->PrevNodeReference;
        }
      }
    #endif

    _P(NodeReference_t) nextNodeReference = Node->NextNodeReference;
    _P(NodeReference_t) prevNodeReference = Node->PrevNodeReference;
    _BLL_POFTWBIT(gln)(_BLL_PBLLTFFC prevNodeReference)->NextNodeReference = nextNodeReference;
    _BLL_POFTWBIT(gln)(_BLL_PBLLTFFC nextNodeReference)->PrevNodeReference = prevNodeReference;
  }

  /* unlink recycle */
  _BLL_SOFTWBIT
  void
  _BLL_POFTWBIT(unlrec)
  (
    _BLL_DBLLTFFC
    _P(NodeReference_t) nr
  ){
    _BLL_POFTWBIT(Unlink)(_BLL_PBLLTFFC nr);
    _BLL_POFTWBIT(Recycle)(_BLL_PBLLTFFC nr);
  }

  _BLL_SOFTWBIT
  void
  _BLL_POFTWBIT(LinkAsFirst)
  (
    _BLL_DBLLTFFC
    _P(NodeReference_t) NodeReference
  ){
    _BLL_POFTWBIT(linkNext)(_BLL_PBLLTFFC _BLL_GetList->src, NodeReference);
  }
  _BLL_SOFTWBIT
  void
  _BLL_POFTWBIT(LinkAsLast)
  (
    _BLL_DBLLTFFC
    _P(NodeReference_t) NodeReference
  ){
    _BLL_POFTWBIT(linkPrev)(_BLL_PBLLTFFC _BLL_GetList->dst, NodeReference);
  }

  _BLL_SOFTWBIT
  void
  _BLL_POFTWBIT(ReLinkAsFirst)
  (
    _BLL_DBLLTFFC
    _P(NodeReference_t) NodeReference
  ){
    _BLL_POFTWBIT(Unlink)(_BLL_PBLLTFFC NodeReference);
    _BLL_POFTWBIT(LinkAsFirst)(_BLL_PBLLTFFC NodeReference);
  }
  _BLL_SOFTWBIT
  void
  _BLL_POFTWBIT(ReLinkAsLast)
  (
    _BLL_DBLLTFFC
    _P(NodeReference_t) NodeReference
  ){
    _BLL_POFTWBIT(Unlink)(_BLL_PBLLTFFC NodeReference);
    _BLL_POFTWBIT(LinkAsLast)(_BLL_PBLLTFFC NodeReference);
  }
#endif

BLL_StructBegin(_P(nrtra_t))
  _P(NodeReference_t) nr;
  #if BLL_set_IsNodeRecycled == 0
    uint8_t *_RecycledArray;
  #endif

  #if BLL_set_Language == 0
    BLL_StructEnd(_P(nrtra_t))
  #endif

  #if BLL_set_StoreFormat == 0
    #define _BLL_nrtra_count _pList->NodeList.Current
  #elif BLL_set_StoreFormat == 1
    #define _BLL_nrtra_count _pList->NodeCurrent
  #endif

  /*
    FN; function name
    PT; parameter type
    PTC; parameter type comma
    G; get
    AP; access parent
  */
  #if BLL_set_Language == 0
    #define _BLL_nrtra_FN(p0) CONCAT2(nrtra_,_BLL_POFTWBIT(p0))
    #define _BLL_nrtra_PT _P(nrtra_t) *nrtra
    #define _BLL_nrtra_PTC ,
    #define _BLL_nrtra_G nrtra
    #define _BLL_nrtra_AP(p0) _BLL_POFTWBIT(p0)
  #elif BLL_set_Language == 1
    #define _BLL_nrtra_FN(p0) _BLL_POFTWBIT(p0)
    #define _BLL_nrtra_PT
    #define _BLL_nrtra_PTC
    #define _BLL_nrtra_G this
    #define _BLL_nrtra_AP(p0) _pList->p0
  #else
    #error
  #endif

  _BLL_SOFTWBIT
  void
  _BLL_nrtra_FN(Open)
  (
    _P(t) *_pList _BLL_nrtra_PTC
    _BLL_nrtra_PT
  ){
    #if BLL_set_IsNodeRecycled == 0
      uintptr_t size = _BLL_nrtra_count * sizeof(uint8_t);
      _BLL_nrtra_G->_RecycledArray = (uint8_t *)BLL_set_alloc_open(size);
      __MemorySet(0, _BLL_nrtra_G->_RecycledArray, size);
      _P(NodeReference_t) cnr = _pList->e.c;
      for(BLL_set_type_node i = _pList->e.p; i != 0; --i){
        _BLL_nrtra_G->_RecycledArray[cnr.NRI] = 1;
        cnr = *_BLL_nrtra_AP(_grecnrofnr)(_BLL_PIL0(_pList __ca__) cnr);
      }
    #endif
    _BLL_nrtra_G->nr.NRI = (BLL_set_type_node)-1;
  }
  _BLL_SOFTWBIT
  void
  _BLL_nrtra_FN(Close)
  (
    _P(t) *_pList _BLL_nrtra_PTC
    _BLL_nrtra_PT
  ){
    #if BLL_set_IsNodeRecycled == 0
      BLL_set_alloc_close(_BLL_nrtra_G->_RecycledArray);
    #endif
  }

  _BLL_SOFTWBIT
  bool
  _BLL_nrtra_FN(Loop)
  (
    _P(t) *_pList _BLL_nrtra_PTC
    _BLL_nrtra_PT
  ){
    ++_BLL_nrtra_G->nr.NRI;
    for(; _BLL_nrtra_G->nr.NRI < _BLL_nrtra_count; ++_BLL_nrtra_G->nr.NRI){
      #if BLL_set_IsNodeRecycled == 0
        if(_BLL_nrtra_G->_RecycledArray[_BLL_nrtra_G->nr.NRI] == 1){
          continue;
        }
      #elif BLL_set_IsNodeRecycled == 1
        if(_BLL_nrtra_AP(IsNodeReferenceRecycled)(_BLL_PIL0(_pList __ca__) _BLL_nrtra_G->nr) == true){
          continue;
        }
      #else
        #error ?
      #endif

      #if BLL_set_Link == 1
        if(_BLL_nrtra_AP(IsNRSentienel)(_BLL_PIL0(_pList __ca__) _BLL_nrtra_G->nr) == true){
          continue;
        }
      #endif

      return 1;
    }
    return 0;
  }

  #undef _BLL_nrtra_FN
  #undef _BLL_nrtra_PT
  #undef _BLL_nrtra_PTC
  #undef _BLL_nrtra_G
  #undef _BLL_nrtra_AP

  #undef _BLL_nrtra_count

#if BLL_set_Language == 1
  BLL_StructEnd(_P(nrtra_t))
#endif

#ifdef BLL_set_CPP_CopyAtPointerChange
  #if !defined(_BLL_HaveConstantNodeData)
    #error those who trigger this, will burn in hell
  #endif

  _BLL_SOFTWBIT
  void
  _BLL_POFTWBIT(AllocateNewBuffer)
  (
    _BLL_DBLLTFFC
    BLL_set_type_node Amount
  ){
    NodeList.Possible = _P(_NodeList_GetBufferAmount0)(&_BLL_GetList->NodeList, Amount);
    void *np = BLL_set_alloc_open(NodeList.Possible * sizeof(_P(Node_t)));
    __MemoryCopy(NodeList.ptr, np, NodeList.Current * sizeof(_P(Node_t)));

    _P(nrtra_t) nrtra;
    nrtra.Open(_BLL_GetList);
    while(nrtra.Loop(_BLL_GetList) == true){
      new (&((_P(Node_t) *)np)[nrtra.nr.NRI].data) _P(NodeData_t)(((_P(Node_t) *)NodeList.ptr)[nrtra.nr.NRI].data);
    }
    nrtra.Close(_BLL_GetList);

    BLL_set_alloc_close(NodeList.ptr);
    _P(_NodeList_SetPointer)(&_BLL_GetList->NodeList, np);
  }
#endif

_BLL_SOFTWBIT
void
_BLL_POFTWBIT(_DestructAllNodes)
(
  _BLL_DBLLTFF
){
  #ifdef BLL_set_CPP_Node_ConstructDestruct
    _P(nrtra_t) nrtra;
    nrtra.Open(_BLL_GetList);
    while(nrtra.Loop(_BLL_GetList) == true){
      _BLL_POFTWBIT(_Node_Destruct)(_BLL_PBLLTFFC nrtra.nr);
    }
    nrtra.Close(_BLL_GetList);
  #endif
}

#if BLL_set_StoreFormat == 1
  _BLL_SOFTWBIT
  void
  _BLL_POFTWBIT(_StoreFormat1_CloseAllocatedBlocks)
  (
    _BLL_DBLLTFF
  ){
    _P(BlockIndex_t) BlockAmount = _BLL_GetList->BlockList.Current;
    for(_P(BlockIndex_t) i = 0; i < BlockAmount; i++){
      /* TODO looks ugly cast */
      void *p = (void *)((_P(Node_t) **)&_BLL_GetList->BlockList.ptr[0])[i];
      BLL_set_alloc_close(p);
    }
  }
#endif

_BLL_SOFTWBIT
void
_BLL_POFTWBIT(_AfterInitNodes)
(
  _BLL_DBLLTFF
){
  _BLL_GetList->e.p = 0;
  #if BLL_set_StoreFormat == 0
    #if BLL_set_Link == 1
      #ifdef BLL_set_CPP_CopyAtPointerChange
        if(NodeList.Possible < 2){
          _BLL_POFTWBIT(AllocateNewBuffer)(_BLL_PBLLTFFC 2);
        }
      #endif

      _P(_NodeList_AddEmpty)(&_BLL_GetList->NodeList, 2);

      _BLL_GetList->src.NRI = 0;
      _BLL_GetList->dst.NRI = 1;
    #endif
  #elif BLL_set_StoreFormat == 1
    #if BLL_set_Link == 1
      _BLL_GetList->src = _BLL_POFTWBIT(_NewNode_NoConstruct)(_BLL_PBLLTFF);
      _BLL_GetList->dst = _BLL_POFTWBIT(_NewNode_NoConstruct)(_BLL_PBLLTFF);
    #endif
  #endif

  #if BLL_set_Link == 1
    _BLL_POFTWBIT(_gln)(_BLL_PBLLTFFC _BLL_GetList->src)->NextNodeReference = _BLL_GetList->dst;
    _BLL_POFTWBIT(_gln)(_BLL_PBLLTFFC _BLL_GetList->dst)->PrevNodeReference = _BLL_GetList->src;
  #endif
}

#ifdef BLL_set_CPP_ConstructDestruct
  private:
#endif
_BLL_SOFTWBIT
void
_BLL_POFTWBIT(Open)
(
  _BLL_DBLLTFF
  #if !defined(_BLL_HaveConstantNodeData) && !defined(BLL_set_MultipleType_Sizes)
    _BLL_PIL0(__ca__) BLL_set_NodeSizeType NodeDataSize
  #endif
){
  BLL_set_NodeSizeType NodeSize = sizeof(_P(Node_t));
  #if !defined(_BLL_HaveConstantNodeData) && !defined(BLL_set_MultipleType_Sizes)
    NodeSize += NodeDataSize;
    #if BLL_set_Link == 0
      /* for NextRecycled */
      NodeSize += sizeof(_P(Node_t)) < sizeof(_P(NodeReference_t)) ? sizeof(_P(Node_t)) - _P(NodeReference_t) : 0;
    #endif
  #endif

  #if BLL_set_UseUninitialisedValues == 0
    _BLL_GetList->e.c.NRI = 0;
  #endif

  #if BLL_set_StoreFormat == 0
    #if defined(_BLL_HaveConstantNodeData) || defined(BLL_set_MultipleType_Sizes)
      _P(_NodeList_Open)(&_BLL_GetList->NodeList);
    #else
      _P(_NodeList_Open)(&_BLL_GetList->NodeList, NodeSize);
    #endif
  #elif BLL_set_StoreFormat == 1
    _BLL_GetList->NodeCurrent = 0;
    _P(_BlockList_Open)(&_BLL_GetList->BlockList);
  #endif
  _BLL_POFTWBIT(_AfterInitNodes)(_BLL_PBLLTFF);

  #if BLL_set_SafeNext == 1
    _BLL_GetList->SafeNext.NRI = (BLL_set_type_node)-1;
  #elif BLL_set_SafeNext > 1
    _BLL_GetList->SafeNextCount = 0;
  #endif
}
_BLL_SOFTWBIT
void
_BLL_POFTWBIT(Close)
(
  _BLL_DBLLTFF
){
  _BLL_POFTWBIT(_DestructAllNodes)(_BLL_PBLLTFF);
  #if BLL_set_StoreFormat == 0
    #ifdef BLL_set_CPP_CopyAtPointerChange
      BLL_set_alloc_close(NodeList.ptr);
    #endif
    _P(_NodeList_Close)(&_BLL_GetList->NodeList);
  #elif BLL_set_StoreFormat == 1
    _BLL_POFTWBIT(_StoreFormat1_CloseAllocatedBlocks)(_BLL_PBLLTFF);
    _P(_BlockList_Close)(&_BLL_GetList->BlockList);
  #endif
}
#ifdef BLL_set_CPP_ConstructDestruct
  public:
#endif

_BLL_SOFTWBIT
void
_BLL_POFTWBIT(Clear) /* TODO those 2 numbers in this function needs to be flexible */
(
  _BLL_DBLLTFF
){
  _BLL_POFTWBIT(_DestructAllNodes)(_BLL_PBLLTFF);
  #if BLL_set_ResizeListAfterClear == 0
    #if BLL_set_StoreFormat == 0
      _BLL_GetList->NodeList.Current = 0;
    #elif BLL_set_StoreFormat == 1
      _BLL_POFTWBIT(_StoreFormat1_CloseAllocatedBlocks)(_BLL_PBLLTFF);
      _BLL_GetList->BlockList.Current = 0;
    #endif
  #else
    #if BLL_set_StoreFormat == 0
      _BLL_GetList->NodeList.Current = 0;
      #if defined(BLL_set_CPP_CopyAtPointerChange)
        /* TODO */
      #else
        _P(_NodeList_Reserve)(&_BLL_GetList->NodeList, 2);
      #endif
    #elif BLL_set_StoreFormat == 1
      _BLL_POFTWBIT(_StoreFormat1_CloseAllocatedBlocks)(_BLL_PBLLTFF);
      _P(_BlockList_ClearWithBuffer)(&_BLL_GetList->BlockList);
    #endif
  #endif

  _BLL_POFTWBIT(_AfterInitNodes)(_BLL_PBLLTFF);
}

#if BLL_set_Link == 1
  _BLL_SOFTWBIT
  _P(NodeReference_t)
  _BLL_POFTWBIT(GetNodeFirst)
  (
    _BLL_DBLLTFF
  ){
    return _BLL_POFTWBIT(_gln)(_BLL_PBLLTFFC _BLL_GetList->src)->NextNodeReference;
  }
  _BLL_SOFTWBIT
  _P(NodeReference_t)
  _BLL_POFTWBIT(GetNodeLast)
  (
    _BLL_DBLLTFF
  ){
    return _BLL_POFTWBIT(_gln)(_BLL_PBLLTFFC _BLL_GetList->dst)->PrevNodeReference;
  }

  _BLL_SOFTWBIT
  bool
  _BLL_POFTWBIT(IsNodeReferenceFronter)
  (
    _BLL_DBLLTFFC
    _P(NodeReference_t) srcNodeReference,
    _P(NodeReference_t) dstNodeReference
  ){
    do{
      _P(Node_t) *srcNode = _BLL_POFTWBIT(gln)(
        _BLL_PBLLTFFC
        srcNodeReference
      );
      srcNodeReference = srcNode->NextNodeReference;
      if(_BLL_POFTWBIT(IsNodeReferenceEqual)(srcNodeReference, dstNodeReference)){
        return 0;
      }
    }while(!_BLL_POFTWBIT(IsNodeReferenceEqual)(srcNodeReference, _BLL_GetList->dst));
    return 1;
  }
#endif

#if BLL_set_SafeNext != 0
  _BLL_SOFTWBIT
  void
  _BLL_POFTWBIT(StartSafeNext)
  (
    _BLL_DBLLTFFC
    _P(NodeReference_t) NodeReference
  ){
    #if BLL_set_debug_InvalidAction == 1
      #if BLL_set_SafeNext == 1
        if(_BLL_GetList->SafeNext.NRI != (BLL_set_type_node)-1){
          __abort();
        }
      #else
        if(_BLL_GetList->SafeNextCount == BLL_set_SafeNext){
          __abort();
        }
      #endif
    #endif
    #if BLL_set_SafeNext == 1
      _BLL_GetList->SafeNext = NodeReference;
    #else
      _BLL_GetList->SafeNext[_BLL_GetList->SafeNextCount++] = NodeReference;
    #endif
  }
  _BLL_SOFTWBIT
  _P(NodeReference_t)
  _BLL_POFTWBIT(EndSafeNext)
  (
    _BLL_DBLLTFF
  ){
    #if BLL_set_debug_InvalidAction == 1
      #if BLL_set_SafeNext == 1
        if(_BLL_GetList->SafeNext.NRI == (BLL_set_type_node)-1){
          __abort();
        }
      #else
        if(_BLL_GetList->SafeNextCount == 0){
          __abort();
        }
      #endif
    #endif
    _P(NodeReference_t) nr;
    #if BLL_set_SafeNext == 1
      nr = _BLL_GetList->SafeNext;
      _BLL_GetList->SafeNext.NRI = (BLL_set_type_node)-1;
    #else
      nr = _BLL_GetList->SafeNext[--_BLL_GetList->SafeNextCount];
    #endif
    _P(Node_t) *Node = _BLL_POFTWBIT(GetNodeByReference)(
      _BLL_PBLLTFFC
      nr
    );
    return Node->NextNodeReference;
  }
  _BLL_SOFTWBIT
  _P(NodeReference_t)
  _BLL_POFTWBIT(CheckSafeNext)
  (
    _BLL_DBLLTFFC
    uint8_t Depth
  ){
    ++Depth;
    #if BLL_set_debug_InvalidAction == 1
      #if BLL_set_SafeNext == 1
        if(Depth != 1){
          __abort();
        }
      #else
        if(Depth > _BLL_GetList->SafeNextCount){
          __abort();
        }
      #endif
    #endif
    #if BLL_set_SafeNext == 1
      return _BLL_GetList->SafeNext;
    #else
      return _BLL_GetList->SafeNext[_BLL_GetList->SafeNextCount - Depth];
    #endif
  }
#endif

_BLL_SOFTWBIT
_P(NodeData_t) *
_BLL_POFTWBIT(GetNodeReferenceData)
(
  _BLL_DBLLTFFC
  _P(NodeReference_t) NodeReference
  #if defined(BLL_set_MultipleType_Sizes)
    , uintptr_t PointerIndex
  #endif
){
  _P(Node_t) *Node = _BLL_POFTWBIT(GetNodeByReference)(
    _BLL_PBLLTFFC
    NodeReference
    #if defined(BLL_set_MultipleType_Sizes)
      , PointerIndex
    #endif
  );
  #if defined(BLL_set_MultipleType_Sizes)
    #if BLL_set_Link == 1
      if(PointerIndex == BLL_set_MultipleType_LinkIndex){
        return (_P(NodeData_t) *)((uint8_t *)Node + sizeof(_P(Node_t)));
      }
      else{
        return (_P(NodeData_t) *)Node;
      }
    #else
      return (_P(NodeData_t) *)Node;
    #endif
  #elif defined(_BLL_HaveConstantNodeData)
    return (_P(NodeData_t) *)&Node->data;
  #else
    return (_P(NodeData_t) *)((uint8_t *)Node + sizeof(_P(Node_t)));
  #endif
}

#if BLL_set_Language == 1
  #if defined(BLL_set_MultipleType_Sizes)
    /* what syntax you would like */
  #elif defined(_BLL_HaveConstantNodeData)
    _P(NodeData_t) &operator[](_P(NodeReference_t) NR){
      return *_BLL_POFTWBIT(GetNodeReferenceData)(NR);
    }
  #else
    _P(NodeData_t) *operator[](_P(NodeReference_t) NR){
      return _BLL_POFTWBIT(GetNodeReferenceData)(NR);
    }
  #endif

  #ifdef BLL_set_CPP_ConstructDestruct
    _P(t)(
      #if !defined(_BLL_HaveConstantNodeData) && !defined(BLL_set_MultipleType_Sizes)
        BLL_set_NodeSizeType NodeDataSize
      #endif
    ){
      Open(
        #if !defined(_BLL_HaveConstantNodeData) && !defined(BLL_set_MultipleType_Sizes)
          NodeDataSize
        #endif
      );
    }
    ~_P(t)(
    ){
      Close();
    }
  #endif

  #ifdef BLL_set_Overload_Declare
    BLL_set_Overload_Declare
  #endif

  BLL_StructEnd(_P(t))

  #if BLL_set_Link == 1
    static _P(NodeReference_t) _P(_NodeReference_Next)(_P(NodeReference_t) *nr, _P(t) *list){
      return list->gln(*nr)->NextNodeReference;
    }
    static _P(NodeReference_t) _P(_NodeReference_Prev)(_P(NodeReference_t) *nr, _P(t) *list){
      return list->gln(*nr)->PrevNodeReference;
    }
  #endif
#endif


#ifdef BLL_set_namespace
  }
#endif
