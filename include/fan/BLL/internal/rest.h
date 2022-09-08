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
BLL_StructBegin(_P(NodeData_t))
BLL_set_node_data
BLL_StructEnd(_P(NodeData_t))
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
  #ifdef BLL_set_node_data
    #ifdef BLL_set_CPP_Node_ConstructDestruct
      _P(NodeData_t) data;
      #if BLL_set_Link == 0
        uint8_t _PaddingForNextRecycled[
          sizeof(_P(NodeData_t)) < sizeof(_P(NodeReference_t)) ?
            sizeof(_P(NodeData_t)) - sizeof(_P(NodeReference_t)) :
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
#if BLL_set_Language == 0
  BLL_StructEnd(_P(t))
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
){
  /* TODO */
  /* this function doesnt check store format */
  /* can give compile error in future */

  #ifdef BLL_set_node_data
    return sizeof(_P(Node_t));
  #else
    return _P(_NodeList_GetNodeSize)(&_BLL_GetList->NodeList);
  #endif
}

_BLL_SOFTWBIT
_P(NodeReference_t)
_BLL_POFTWBIT(GetConstantInvalidNodeReference)
(
  #if BLL_set_ConstantInvalidNodeReference_Listless == 0
    _BLL_DBLLTFF
  #endif
){
  _P(NodeReference_t) nr;
  nr.NRI = (BLL_set_type_node)-1;
  return nr;
}

_BLL_SOFTWBIT
bool
_BLL_POFTWBIT(IsNodeReferenceInvalid)
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

/* basically same with IsNodeReferenceInvalid */
/* only difference this function made for compare with GetConstantInvalidNodeReference */
/* so it can be faster in some cases */
_BLL_SOFTWBIT
bool
_BLL_POFTWBIT(IsNodeReferenceInvalidConstant)
(
  #if BLL_set_ConstantInvalidNodeReference_Listless == 0
    _BLL_DBLLTFFC
  #endif
  _P(NodeReference_t) NodeReference
){
  _P(NodeReference_t) inr = _BLL_POFTWBIT(GetConstantInvalidNodeReference)(
    #if BLL_set_ConstantInvalidNodeReference_Listless == 0
      _BLL_PBLLTFF
    #endif
  );
  return inr.NRI == NodeReference.NRI;
}

_BLL_SOFTWBIT
_P(Node_t) *
_BLL_POFTWBIT(_GetNodeByReference)
(
  _BLL_DBLLTFFC
  _P(NodeReference_t) NodeReference
){
  #if BLL_set_StoreFormat == 0
    return (_P(Node_t) *)_P(_NodeList_GetNode)(&_BLL_GetList->NodeList, NodeReference.NRI);
  #elif BLL_set_StoreFormat == 1
    #if defined(BLL_set_node_data)
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

/* get node reference th of node */
_BLL_SOFTWBIT
_P(NodeReference_t) *
_BLL_POFTWBIT(_GetNRTHOfNode)
(
  _BLL_DBLLTFFC
  _P(NodeReference_t) NodeReference,
  BLL_set_type_node TH
){
  _P(Node_t) *n = _BLL_POFTWBIT(_GetNodeByReference)(_BLL_PBLLTFFC NodeReference);
  return &((_P(NodeReference_t) *)n)[TH];
}

#if BLL_set_IsNodeUnlinked == 1
  _BLL_SOFTWBIT
  bool
  _BLL_POFTWBIT(IsNodeUnlinked)
  (
    _BLL_DBLLTFFC
    _P(Node_t) *Node
  ){
    if(Node->PrevNodeReference.NRI == (BLL_set_type_node)-1){
      return 1;
    }
    return 0;
  }
  _BLL_SOFTWBIT
  bool
  _BLL_POFTWBIT(IsNodeReferenceUnlinked)
  (
    _BLL_DBLLTFFC
    _P(NodeReference_t) NodeReference
  ){
    _P(Node_t) *Node = _BLL_POFTWBIT(_GetNodeByReference)(_BLL_PBLLTFFC NodeReference);
    return _P(IsNodeUnlinked)(_BLL_PBLLTFFC Node);
  }
#endif

_BLL_SOFTWBIT
_P(Node_t) *
_BLL_POFTWBIT(GetNodeByReference)
(
  _BLL_DBLLTFFC
  _P(NodeReference_t) NodeReference
){
  #if BLL_set_debug_InvalidAction == 1
    if(NodeReference >= _BLL_GetList->nodes.Current){
      PR_abort();
    }
  #endif
  _P(Node_t) *Node = _BLL_POFTWBIT(_GetNodeByReference)(_BLL_PBLLTFFC NodeReference);
  #if BLL_set_debug_InvalidAction == 1
    do{
      #if BLL_set_debug_InvalidAction_srcAccess == 0
        if(NodeReference == _BLL_GetList->src){
          break;
        }
      #endif
      #if BLL_set_debug_InvalidAction_dstAccess == 0
        if(NodeReference == _BLL_GetList->dst){
          break;
        }
      #endif
      if(_P(IsNodeUnlinked)(_BLL_PBLLTFFC Node)){
        PR_abort();
      }
    }while(0);
  #endif
  return Node;
}

_BLL_SOFTWBIT
BLL_set_type_node
_BLL_POFTWBIT(usage)
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
    _P(Node_t) *n = (_P(Node_t) *)BLL_set_StoreFormat1_alloc_open(
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
_P(NodeReference_t)
_BLL_POFTWBIT(NewNode_empty)
(
  _BLL_DBLLTFF
){
  _P(NodeReference_t) NodeReference = _BLL_GetList->e.c;
  _BLL_GetList->e.c = *_BLL_POFTWBIT(_GetNRTHOfNode)(_BLL_PBLLTFFC NodeReference, 0);
  _BLL_GetList->e.p--;
  _BLL_POFTWBIT(_Node_Construct)(_BLL_PBLLTFFC NodeReference);
  return NodeReference;
}
_BLL_SOFTWBIT
_P(NodeReference_t)
_BLL_POFTWBIT(NewNode_alloc)
(
  _BLL_DBLLTFF
){
  _P(NodeReference_t) r;
  #if BLL_set_StoreFormat == 0
    r.NRI = _BLL_GetList->NodeList.Current;
    _P(_NodeList_AddEmpty)(&_BLL_GetList->NodeList, 1);
  #elif BLL_set_StoreFormat == 1
    if(_BLL_GetList->NodeCurrent % BLL_set_StoreFormat1_ElementPerBlock == 0){
      _BLL_POFTWBIT(_PushNewBlock)(_BLL_PBLLTFF);
    }
    r.NRI = _BLL_GetList->NodeCurrent++;
  #endif
  _BLL_POFTWBIT(_Node_Construct)(_BLL_PBLLTFFC r);
  return r;
}
_BLL_SOFTWBIT
_P(NodeReference_t)
_BLL_POFTWBIT(NewNode)
(
  _BLL_DBLLTFF
){
  _P(NodeReference_t) NodeReference;
  if(_BLL_GetList->e.p){
    NodeReference = _BLL_POFTWBIT(NewNode_empty)(_BLL_PBLLTFF);
  }
  else{
    NodeReference = _BLL_POFTWBIT(NewNode_alloc)(_BLL_PBLLTFF);
  }
  #if BLL_set_debug_InvalidAction >= 1
    _P(Node_t) *Node = _BLL_POFTWBIT(_GetNodeByReference)(_BLL_PBLLTFFC NodeReference);
    Node->PrevNodeReference.NRI = (BLL_set_type_node)-1;
    Node->NextNodeReference.NRI = (BLL_set_type_node)-1;
  #endif
  return NodeReference;
}

#if BLL_set_Link == 1
  _BLL_SOFTWBIT
  _P(NodeReference_t)
  _BLL_POFTWBIT(NewNodeFirst_empty)
  (
    _BLL_DBLLTFF
  ){
    _P(NodeReference_t) NodeReference = _BLL_POFTWBIT(NewNode_empty)(_BLL_PBLLTFF);
    _P(NodeReference_t) srcNodeReference = _BLL_GetList->src;
    _BLL_POFTWBIT(_GetNodeByReference)(_BLL_PBLLTFFC NodeReference)->NextNodeReference = srcNodeReference;
    _BLL_POFTWBIT(_GetNodeByReference)(_BLL_PBLLTFFC srcNodeReference)->PrevNodeReference = NodeReference;
    _BLL_GetList->src = NodeReference;
    return srcNodeReference;
  }
  _BLL_SOFTWBIT
  _P(NodeReference_t)
  _BLL_POFTWBIT(NewNodeFirst_alloc)
  (
    _BLL_DBLLTFF
  ){
    _P(NodeReference_t) NodeReference = _BLL_POFTWBIT(NewNode_alloc)(_BLL_PBLLTFF);
    _P(NodeReference_t) srcNodeReference = _BLL_GetList->src;
    _BLL_POFTWBIT(_GetNodeByReference)(_BLL_PBLLTFFC NodeReference)->NextNodeReference = srcNodeReference;
    _BLL_POFTWBIT(_GetNodeByReference)(_BLL_PBLLTFFC srcNodeReference)->PrevNodeReference = NodeReference;
    _BLL_GetList->src = NodeReference;
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
    _P(NodeReference_t) NodeReference = _BLL_POFTWBIT(NewNode_empty)(_BLL_PBLLTFF);
    _P(NodeReference_t) dstNodeReference = _BLL_GetList->dst;
    _BLL_POFTWBIT(_GetNodeByReference)(_BLL_PBLLTFFC NodeReference)->PrevNodeReference = dstNodeReference;
    _BLL_POFTWBIT(_GetNodeByReference)(_BLL_PBLLTFFC dstNodeReference)->NextNodeReference = NodeReference;
    #if BLL_set_debug_InvalidAction == 1
      _BLL_POFTWBIT(_GetNodeByReference)(_BLL_PBLLTFFC dstNodeReference)->PrevNodeReference = 0;
    #endif
    _BLL_GetList->dst = NodeReference;
    return dstNodeReference;
  }
  _BLL_SOFTWBIT
  _P(NodeReference_t)
  _BLL_POFTWBIT(NewNodeLast_alloc)
  (
    _BLL_DBLLTFF
  ){
    _P(NodeReference_t) NodeReference = _BLL_POFTWBIT(NewNode_alloc)(_BLL_PBLLTFF);
    _P(NodeReference_t) dstNodeReference = _BLL_GetList->dst;
    _BLL_POFTWBIT(_GetNodeByReference)(_BLL_PBLLTFFC NodeReference)->PrevNodeReference = dstNodeReference;
    _BLL_POFTWBIT(_GetNodeByReference)(_BLL_PBLLTFFC dstNodeReference)->NextNodeReference = NodeReference;
    #if BLL_set_debug_InvalidAction == 1
      _BLL_POFTWBIT(_GetNodeByReference)(_BLL_PBLLTFFC dstNodeReference)->PrevNodeReference = 0;
    #endif
    _BLL_GetList->dst = NodeReference;
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
#endif

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
      BLL_set_StoreFormat1_alloc_close(p);
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
      _P(_NodeList_AddEmpty)(&_BLL_GetList->NodeList, 2);

      _BLL_GetList->src.NRI = 0;
      _BLL_GetList->dst.NRI = 1;
    #endif
  #elif BLL_set_StoreFormat == 1
    #if BLL_set_Link == 1
      _BLL_GetList->src = _BLL_POFTWBIT(NewNode)(_BLL_PBLLTFF);
      _BLL_GetList->dst = _BLL_POFTWBIT(NewNode)(_BLL_PBLLTFF);
    #endif
  #endif

  #if BLL_set_Link == 1
    _BLL_POFTWBIT(_GetNodeByReference)(_BLL_PBLLTFFC _BLL_GetList->src)->NextNodeReference = _BLL_GetList->dst;
    _BLL_POFTWBIT(_GetNodeByReference)(_BLL_PBLLTFFC _BLL_GetList->dst)->PrevNodeReference = _BLL_GetList->src;
  #endif
}

_BLL_SOFTWBIT
void
_BLL_POFTWBIT(open)
(
  _BLL_DBLLTFF
  #ifndef BLL_set_node_data
    _BLL_OCIBLLTFFE BLL_set_NodeSizeType NodeDataSize
  #endif
){
  BLL_set_NodeSizeType NodeSize = sizeof(_P(Node_t));
  NodeSize;
  #ifndef BLL_set_node_data
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
    #ifdef BLL_set_node_data
      _P(_NodeList_Open)(&_BLL_GetList->NodeList);
    #else
      _P(_NodeList_Open)(&_BLL_GetList->NodeList, NodeSize);
    #endif
  #elif BLL_set_StoreFormat == 1
    _BLL_GetList->NodeCurrent = 0;
    _P(_BlockList_Open)(&_BLL_GetList->BlockList);
  #endif
  _BLL_POFTWBIT(_AfterInitNodes)(_BLL_PBLLTFF);

  #if BLL_set_SafeNext
    _BLL_GetList->SafeNext.NRI = (BLL_set_type_node)-1;
  #endif
}
_BLL_SOFTWBIT
void
_BLL_POFTWBIT(close)
(
  _BLL_DBLLTFF
){
  #if BLL_set_StoreFormat == 0
    _P(_NodeList_Close)(&_BLL_GetList->NodeList);
  #elif BLL_set_StoreFormat == 1
    _BLL_POFTWBIT(_StoreFormat1_CloseAllocatedBlocks)(_BLL_PBLLTFF);
    _P(_BlockList_Close)(&_BLL_GetList->BlockList);
  #endif
}
_BLL_SOFTWBIT
void
_BLL_POFTWBIT(Clear) /* TODO those 2 numbers in this function needs to be flexible */
(
  _BLL_DBLLTFF
){
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
      _P(_NodeList_Reserve)(&_BLL_GetList->NodeList, 2);
    #elif BLL_set_StoreFormat == 1
      _BLL_POFTWBIT(_StoreFormat1_CloseAllocatedBlocks)(_BLL_PBLLTFF);
      _P(_BlockList_ClearWithBuffer)(&_BLL_GetList->BlockList);
    #endif
  #endif

  _BLL_POFTWBIT(_AfterInitNodes)(_BLL_PBLLTFF);
}

#if BLL_set_Link == 1
  _BLL_SOFTWBIT
  void
  _BLL_POFTWBIT(linkNext)
  (
    _BLL_DBLLTFFC
    _P(NodeReference_t) srcNodeReference,
    _P(NodeReference_t) dstNodeReference
  ){
    _P(Node_t) *srcNode = _BLL_POFTWBIT(GetNodeByReference)(_BLL_PBLLTFFC srcNodeReference);
    _P(Node_t) *dstNode = _BLL_POFTWBIT(_GetNodeByReference)(_BLL_PBLLTFFC dstNodeReference);
    _P(NodeReference_t) nextNodeReference = srcNode->NextNodeReference;
    _P(Node_t) *nextNode = _BLL_POFTWBIT(GetNodeByReference)(_BLL_PBLLTFFC nextNodeReference);
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
    _P(Node_t) *srcNode = _BLL_POFTWBIT(GetNodeByReference)(_BLL_PBLLTFFC srcNodeReference);
    _P(NodeReference_t) prevNodeReference = srcNode->PrevNodeReference;
    _P(Node_t) *prevNode = _BLL_POFTWBIT(GetNodeByReference)(_BLL_PBLLTFFC prevNodeReference);
    prevNode->NextNodeReference = dstNodeReference;
    _P(Node_t) *dstNode = _BLL_POFTWBIT(_GetNodeByReference)(_BLL_PBLLTFFC dstNodeReference);
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
      if(NodeReference == _BLL_GetList->src){
        PR_abort();
      }
      if(NodeReference == _BLL_GetList->dst){
        PR_abort();
      }
    #endif
    _P(Node_t) *Node = _BLL_POFTWBIT(GetNodeByReference)(_BLL_PBLLTFFC NodeReference);
    #if BLL_set_debug_InvalidAction >= 1
      if(_P(IsNodeUnlinked)(_BLL_PBLLTFFC Node)){
        PR_abort();
      }
    #endif
    #if BLL_set_SafeNext
      if(_BLL_GetList->SafeNext == NodeReference){
        _BLL_GetList->SafeNext = Node->PrevNodeReference;
      }
    #endif
    _P(NodeReference_t) nextNodeReference = Node->NextNodeReference;
    _P(NodeReference_t) prevNodeReference = Node->PrevNodeReference;
    _BLL_POFTWBIT(GetNodeByReference)(_BLL_PBLLTFFC prevNodeReference)->NextNodeReference = nextNodeReference;
    _BLL_POFTWBIT(GetNodeByReference)(_BLL_PBLLTFFC nextNodeReference)->PrevNodeReference = prevNodeReference;
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

_BLL_SOFTWBIT
void
_BLL_POFTWBIT(Recycle)
(
  _BLL_DBLLTFFC
  _P(NodeReference_t) NodeReference
){
  _BLL_POFTWBIT(_Node_Destruct)(_BLL_PBLLTFFC NodeReference);
  _P(NodeReference_t) *NextRecycled = _BLL_POFTWBIT(_GetNRTHOfNode)(_BLL_PBLLTFFC NodeReference, 0);

  *NextRecycled = _BLL_GetList->e.c;
  #if BLL_set_IsNodeUnlinked == 1
    *_P(GetNRTHOfNode)(_BLL_PBLLTFFC NodeReference, 1).NRI = (BLL_set_type_node)-1;
  #endif
  _BLL_GetList->e.c = NodeReference;
  _BLL_GetList->e.p++;
}

#if BLL_set_Link == 1
  _BLL_SOFTWBIT
  _P(NodeReference_t)
  _BLL_POFTWBIT(GetNodeFirst)
  (
    _BLL_DBLLTFF
  ){
    return _BLL_POFTWBIT(_GetNodeByReference)(_BLL_PBLLTFFC _BLL_GetList->src)->NextNodeReference;
  }
  _BLL_SOFTWBIT
  _P(NodeReference_t)
  _BLL_POFTWBIT(GetNodeLast)
  (
    _BLL_DBLLTFF
  ){
    return _BLL_POFTWBIT(_GetNodeByReference)(_BLL_PBLLTFFC _BLL_GetList->dst)->PrevNodeReference;
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
      _P(Node_t) *srcNode = _BLL_POFTWBIT(GetNodeByReference)(
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

#if BLL_set_SafeNext
  _BLL_SOFTWBIT
  void
  _BLL_POFTWBIT(StartSafeNext)
  (
    _BLL_DBLLTFFC
    _P(NodeReference_t) NodeReference
  ){
    #if BLL_set_debug_InvalidAction == 1
      if(_BLL_GetList->SafeNext.NRI != (BLL_set_type_node)-1){
        PR_abort();
      }
    #endif
    _BLL_GetList->SafeNext = NodeReference;
  }
  _BLL_SOFTWBIT
  _P(NodeReference_t)
  _BLL_POFTWBIT(EndSafeNext)
  (
    _BLL_DBLLTFF
  ){
    #if BLL_set_debug_InvalidAction == 1
      if(_BLL_GetList->SafeNext.NRI == (BLL_set_type_node)-1){
        PR_abort();
      }
    #endif
    _P(Node_t) *Node = _BLL_POFTWBIT(GetNodeByReference)(
      _BLL_PBLLTFFC
      _BLL_GetList->SafeNext
    );
    _BLL_GetList->SafeNext.NRI = (BLL_set_type_node)-1;
    return Node->NextNodeReference;
  }
#endif

_BLL_SOFTWBIT
void *
_BLL_POFTWBIT(GetNodeReferenceData)
(
  _BLL_DBLLTFFC
  _P(NodeReference_t) NodeReference
){
  _P(Node_t) *Node = _BLL_POFTWBIT(GetNodeByReference)(
    _BLL_PBLLTFFC
    NodeReference
  );
  #ifdef BLL_set_node_data
    return (void *)&Node->data;
  #else
    return (void *)((uint8_t *)Node + sizeof(_P(Node_t)));
  #endif
}

#if BLL_set_Language == 1
  _P(NodeData_t) &operator[](_P(NodeReference_t) NR){
    return _BLL_POFTWBIT(_GetNodeByReference)(NR)->data;
  }

  BLL_StructEnd(_P(t))

  #if BLL_set_Link == 1
    static _P(NodeReference_t) _P(_NodeReference_Next)(_P(NodeReference_t) *nr, _P(t) *list){
      return list->GetNodeByReference(*nr)->NextNodeReference;
    }
    static _P(NodeReference_t) _P(_NodeReference_Prev)(_P(NodeReference_t) *nr, _P(t) *list){
      return list->GetNodeByReference(*nr)->PrevNodeReference;
    }
  #endif
#endif


#ifdef BLL_set_namespace
  }
#endif
