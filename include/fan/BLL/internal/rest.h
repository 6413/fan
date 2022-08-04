#ifdef BLL_set_namespace
  namespace BLL_set_namespace {
#endif

static
bool
_P(IsNodeReferenceEqual)
(
  _P(NodeReference_t) p0,
  _P(NodeReference_t) p1
){
  return p0.NRI == p1.NRI;
}

static
uintptr_t
_P(GetNodeSize)
(
  _P(t) *list
){
  #ifdef BLL_set_node_data
    return sizeof(_P(Node_t));
  #else
    return list->NodeSize;
  #endif
}

static
bool
_P(IsNodeReferenceInvalid)
(
  _P(t) *list,
  _P(NodeReference_t) NodeReference
){
  #if BLL_set_StoreFormat == 0
    #if BLL_set_BaseLibrary == 0
      return NodeReference.NRI >= list->nodes.Current;
    #elif BLL_set_BaseLibrary == 1
      return NodeReference.NRI >= list->nodes.size();
    #endif
  #elif BLL_set_StoreFormat == 1
    return NodeReference.NRI >= list->NodeCurrent;
  #endif
}

#if BLL_set_IsNodeUnlinked == 1
  static
  bool
  _P(IsNodeUnlinked)
  (
    _P(t) *list,
    _P(Node_t) *Node
  ){
    if(Node->PrevNodeReference == _P(NodeReference_t)-1){
      return 1;
    }
    return 0;
  }
  static
  bool
  _P(IsNodeReferenceUnlinked)
  (
    _P(t) *list,
    _P(NodeReference_t) NodeReference
  ){
    _P(Node_t) *Node = &((_P(Node_t) *)&list->nodes.ptr[0])[NodeReference];
    return _P(IsNodeUnlinked)(list, Node);
  }
#endif

static
_P(Node_t) *
_PP(GetNodeByReference)
(
  _P(t) *list,
  _P(NodeReference_t) NodeReference
){
  #if BLL_set_StoreFormat == 0
    #if defined(BLL_set_node_data)
      return &((_P(Node_t) *)&list->nodes.ptr[0])[NodeReference.NRI];
    #else
      return (_P(Node_t) *)&list->nodes.ptr[NodeReference.NRI * list->NodeSize];
    #endif
  #elif BLL_set_StoreFormat == 1
    #if defined(BLL_set_node_data)
      _P(BlockIndex_t) bi = NodeReference.NRI / BLL_set_StoreFormat1_ElementPerBlock;
      _P(BlockModulo_t) bm = NodeReference.NRI % BLL_set_StoreFormat1_ElementPerBlock;
      _P(Node_t) *n = &((_P(Node_t) *)((void **)&list->Blocks.ptr[0])[bi])[bm];
      return n;
    #else
      #error not implemented yet
    #endif
    return (_P(Node_t) *)NodeReference.NRI;
  #endif
}

static
_P(Node_t) *
_P(GetNodeByReference)
(
  _P(t) *list,
  _P(NodeReference_t) NodeReference
){
  #if BLL_set_debug_InvalidAction == 1
    if(NodeReference >= list->nodes.Current){
      PR_abort();
    }
  #endif
  _P(Node_t) *Node = _PP(GetNodeByReference)(list, NodeReference);
  #if BLL_set_debug_InvalidAction == 1
    do{
      #if BLL_set_debug_InvalidAction_srcAccess == 0
        if(NodeReference == list->src){
          break;
        }
      #endif
      #if BLL_set_debug_InvalidAction_dstAccess == 0
        if(NodeReference == list->dst){
          break;
        }
      #endif
      if(_P(IsNodeUnlinked)(list, Node)){
        PR_abort();
      }
    }while(0);
  #endif
  return Node;
}

static
BLL_set_type_node
_P(usage)
(
  _P(t) *list
){
  #if BLL_set_StoreFormat == 0
    #if BLL_set_Link == 0
      #if BLL_set_BaseLibrary == 0
        return list->nodes.Current - list->e.p;
      #elif BLL_set_BaseLibrary == 1
        return list->nodes.size() - list->e.p;
      #endif
    #elif BLL_set_Link == 1
      #if BLL_set_BaseLibrary == 0
        return list->nodes.Current - list->e.p - 2;
      #elif BLL_set_BaseLibrary == 1
        return list->nodes.size() - list->e.p - 2;
      #endif
    #endif
  #elif BLL_set_StoreFormat == 1
    #if BLL_set_Link == 0
      return list->NodeCurrent - list->e.p;
    #elif BLL_set_Link == 1
      #error help
    #endif
  #endif
}

#if BLL_set_StoreFormat == 1
  static
  void
  _P(_PushNewBlock)
  (
    _P(t) *list
  ){
    _P(Node_t) *n = (_P(Node_t) *)BLL_set_StoreFormat1_alloc_open(
      sizeof(_P(Node_t)) * BLL_set_StoreFormat1_ElementPerBlock);
    #if BLL_set_BaseLibrary == 0
      VEC_handle0(&list->Blocks, 1);
      ((_P(Node_t) **)list->Blocks.ptr)[list->nodes.Current - 1] = n;
    #elif BLL_set_BaseLibrary == 1
      list->Blocks.push_back(n);
    #endif
  }
#endif

static
_P(NodeReference_t)
_P(NewNode_empty)
(
  _P(t) *list
){
  _P(NodeReference_t) NodeReference = list->e.c;
  list->e.c = _PP(GetNodeByReference)(list, NodeReference)->NextNodeReference;
  list->e.p--;
  return NodeReference;
}
static
_P(NodeReference_t)
_P(NewNode_alloc)
(
  _P(t) *list
){
  _P(NodeReference_t) r;
  #if BLL_set_StoreFormat == 0
    #if BLL_set_BaseLibrary == 0
      VEC_handle(&list->nodes);
      r.NRI = list->nodes.Current++;
    #elif BLL_set_BaseLibrary == 1
      r.NRI = list->nodes.push_back({});
    #endif
  #elif BLL_set_StoreFormat == 1
    if(list->NodeCurrent % BLL_set_StoreFormat1_ElementPerBlock == 0){
      _P(_PushNewBlock)(list);
    }
    r.NRI = list->NodeCurrent++;
  #endif
  return r;
}
static
_P(NodeReference_t)
_P(NewNode)
(
  _P(t) *list
){
  _P(NodeReference_t) NodeReference;
  if(list->e.p){
    NodeReference = _P(NewNode_empty)(list);
  }
  else{
    NodeReference = _P(NewNode_alloc)(list);
  }
  #if BLL_set_debug_InvalidAction >= 1
    _P(Node_t) *Node = _PP(GetNodeByReference)(list, NodeReference);
    Node->PrevNodeReference = _P(NodeReference_t)-1;
    Node->NextNodeReference = _P(NodeReference_t)-1;
  #endif
  return NodeReference;
}

#if BLL_set_Link == 1
  static
  _P(NodeReference_t)
  _P(NewNodeFirst_empty)
  (
    _P(t) *list
  ){
    _P(NodeReference_t) NodeReference = _P(NewNode_empty)(list);
    _P(NodeReference_t) srcNodeReference = list->src;
    _PP(GetNodeByReference)(list, NodeReference)->NextNodeReference = srcNodeReference;
    _PP(GetNodeByReference)(list, srcNodeReference)->PrevNodeReference = NodeReference;
    list->src = NodeReference;
    return srcNodeReference;
  }
  static
  _P(NodeReference_t)
  _P(NewNodeFirst_alloc)
  (
    _P(t) *list
  ){
    _P(NodeReference_t) NodeReference = _P(NewNode_alloc)(list);
    _P(NodeReference_t) srcNodeReference = list->src;
    _PP(GetNodeByReference)(list, NodeReference)->NextNodeReference = srcNodeReference;
    _PP(GetNodeByReference)(list, srcNodeReference)->PrevNodeReference = NodeReference;
    list->src = NodeReference;
    return srcNodeReference;
  }
  static
  _P(NodeReference_t)
  _P(NewNodeFirst)
  (
    _P(t) *list
  ){
    if(list->e.p){
      return _P(NewNodeFirst_empty)(list);
    }
    else{
      return _P(NewNodeFirst_alloc)(list);
    }
  }
  static
  _P(NodeReference_t)
  _P(NewNodeLast_empty)
  (
    _P(t) *list
  ){
    _P(NodeReference_t) NodeReference = _P(NewNode_empty)(list);
    _P(NodeReference_t) dstNodeReference = list->dst;
    _PP(GetNodeByReference)(list, NodeReference)->PrevNodeReference = dstNodeReference;
    _PP(GetNodeByReference)(list, dstNodeReference)->NextNodeReference = NodeReference;
    #if BLL_set_debug_InvalidAction == 1
      _PP(GetNodeByReference)(list, dstNodeReference)->PrevNodeReference = 0;
    #endif
    list->dst = NodeReference;
    return dstNodeReference;
  }
  static
  _P(NodeReference_t)
  _P(NewNodeLast_alloc)
  (
    _P(t) *list
  ){
    _P(NodeReference_t) NodeReference = _P(NewNode_alloc)(list);
    _P(NodeReference_t) dstNodeReference = list->dst;
    _PP(GetNodeByReference)(list, NodeReference)->PrevNodeReference = dstNodeReference;
    _PP(GetNodeByReference)(list, dstNodeReference)->NextNodeReference = NodeReference;
    #if BLL_set_debug_InvalidAction == 1
      _PP(GetNodeByReference)(list, dstNodeReference)->PrevNodeReference = 0;
    #endif
    list->dst = NodeReference;
    return dstNodeReference;
  }
  static
  _P(NodeReference_t)
  _P(NewNodeLast)
  (
    _P(t) *list
  ){
    if(list->e.p){
      return _P(NewNodeLast_empty)(list);
    }
    else{
      return _P(NewNodeLast_alloc)(list);
    }
  }
#endif

#if BLL_set_StoreFormat == 1
  static
  void
  _P(_StoreFormat1_CloseAllocatedBlocks)
  (
    _P(t) *list
  ){
    #if BLL_set_BaseLibrary == 0
      _P(BlockIndex_t) BlockAmount = list->Blocks.Current;
    #elif BLL_set_BaseLibrary == 1
      _P(BlockIndex_t) BlockAmount = list->Blocks.size();
    #endif
    for(_P(BlockIndex_t) i = 0; i < BlockAmount; i++){
      void *p = (void *)((_P(Node_t) **)&list->Blocks.ptr[0])[i];
      BLL_set_StoreFormat1_alloc_close(p);
    }
  }
#endif

static
void
_P(_AfterInitNodes)
(
  _P(t) *list
){
  list->e.p = 0;
  #if BLL_set_StoreFormat == 0
    #if BLL_set_Link == 1
      #if BLL_set_BaseLibrary == 0
        VEC_handle0(&list->nodes, 2);
      #elif BLL_set_BaseLibrary == 1
        list->nodes.resize(2);
      #endif

      list->src = 0;
      list->dst = 1;
    #endif
  #elif BLL_set_StoreFormat == 1
    #if BLL_set_Link == 1
      list->src = _P(NewNode)(list);
      list->dst = _P(NewNode)(list);
    #endif
  #endif

  #if BLL_set_Link == 1
    _PP(GetNodeByReference)(list, list->src)->NextNodeReference = list->dst;
    _PP(GetNodeByReference)(list, list->dst)->PrevNodeReference = list->src;
  #endif
}

static
void
_P(open)
(
  _P(t) *list
  #ifndef BLL_set_node_data
    , uintptr_t NodeDataSize
  #endif
){
  uintptr_t NodeSize = sizeof(_P(Node_t));
  #ifndef BLL_set_node_data
    NodeSize += NodeDataSize;
    list->NodeSize = NodeSize;
  #endif

  #if BLL_set_UseUninitialisedValues == 0
    list->e.c.NRI = 0;
  #endif

  #if BLL_set_StoreFormat == 0
    #if BLL_set_BaseLibrary == 0
      VEC_init(&list->nodes, NodeSize, A_resize);
    #elif BLL_set_BaseLibrary == 1
      list->nodes.open();
    #endif
  #elif BLL_set_StoreFormat == 1
    list->NodeCurrent = 0;
    #if BLL_set_BaseLibrary == 0
      VEC_init(&list->Blocks, NodeSize * BLL_set_StoreFormat1_ElementPerBlock, A_resize);
    #elif BLL_set_BaseLibrary == 1
      list->Blocks.open();
    #endif
  #endif
  _P(_AfterInitNodes)(list);

  #if BLL_set_SafeNext
    list->SafeNext = (_P(NodeReference_t))-1;
  #endif
}
static
void
_P(close)
(
  _P(t) *list
){
  #if BLL_set_StoreFormat == 0
    #if BLL_set_BaseLibrary == 0
      VEC_free(&list->nodes);
    #elif BLL_set_BaseLibrary == 1
      list->nodes.close();
    #endif
  #elif BLL_set_StoreFormat == 1
    #if BLL_set_Link == 0
      _P(_StoreFormat1_CloseAllocatedBlocks)(list);
      #if BLL_set_BaseLibrary == 0
        VEC_free(&list->Blocks);
      #elif BLL_set_BaseLibrary == 1
        list->Blocks.close();
      #endif
    #elif BLL_set_Link == 1
      #error help
    #endif
  #endif
}
static
void
_P(Clear) /* TODO those 2 numbers in this function needs to be flexible */
(
  _P(t) *list
){
  #if BLL_set_ResizeListAfterClear == 0
    #if BLL_set_StoreFormat == 0
      #if BLL_set_BaseLibrary == 0
        list->nodes.Current = 0;
      #elif BLL_set_BaseLibrary == 1
        list->nodes.resize(0);
      #endif
    #elif BLL_set_StoreFormat == 1
      _P(_StoreFormat1_CloseAllocatedBlocks)(list);
      #if BLL_set_BaseLibrary == 0
        list->Blocks.Current = 0;
      #elif BLL_set_BaseLibrary == 1
        list->Blocks.resize(0);
      #endif
    #endif
  #else
    #if BLL_set_StoreFormat == 0
      #if BLL_set_BaseLibrary == 0
        list->nodes.Current = 0;
        list->nodes.Possible = 2;
        list->nodes.ptr = list->nodes.resize(list->nodes.ptr, list->nodes.Possible * list->nodes.Type);
      #elif BLL_set_BaseLibrary == 1
        /* TODO fan hector doesnt have function for capacity change */
        list->nodes.resize(0);
      #endif
    #elif BLL_set_StoreFormat == 1
      _P(_StoreFormat1_CloseAllocatedBlocks)(list);
      #if BLL_set_BaseLibrary == 0
        list->Blocks.Current = 0;
        list->Blocks.Possible = 2;
        list->Blocks.ptr = list->nodes.resize(list->nodes.ptr, list->nodes.Possible * list->nodes.Type);
      #elif BLL_set_BaseLibrary == 1
        /* TODO fan hector doesnt have function for capacity change */
        list->nodes.resize(0);
      #endif
    #endif
  #endif
  _P(_AfterInitNodes)(list);
}

#if BLL_set_Link == 1
  static
  void
  _P(linkNext)
  (
    _P(t) *list,
    _P(NodeReference_t) srcNodeReference,
    _P(NodeReference_t) dstNodeReference
  ){
    _P(Node_t) *srcNode = _P(GetNodeByReference)(list, srcNodeReference);
    _P(Node_t) *dstNode = _PP(GetNodeByReference)(list, dstNodeReference);
    _P(NodeReference_t) nextNodeReference = srcNode->NextNodeReference;
    _P(Node_t) *nextNode = _P(GetNodeByReference)(list, nextNodeReference);
    srcNode->NextNodeReference = dstNodeReference;
    dstNode->PrevNodeReference = srcNodeReference;
    dstNode->NextNodeReference = nextNodeReference;
    nextNode->PrevNodeReference = dstNodeReference;
  }
  static
  void
  _P(linkPrev)
  (
    _P(t) *list,
    _P(NodeReference_t) srcNodeReference,
    _P(NodeReference_t) dstNodeReference
  ){
    _P(Node_t) *srcNode = _P(GetNodeByReference)(list, srcNodeReference);
    _P(NodeReference_t) prevNodeReference = srcNode->PrevNodeReference;
    _P(Node_t) *prevNode = _P(GetNodeByReference)(list, prevNodeReference);
    prevNode->NextNodeReference = dstNodeReference;
    _P(Node_t) *dstNode = _PP(GetNodeByReference)(list, dstNodeReference);
    dstNode->PrevNodeReference = prevNodeReference;
    dstNode->NextNodeReference = srcNodeReference;
    srcNode->PrevNodeReference = dstNodeReference;
  }

  static
  void
  _P(Unlink)
  (
    _P(t) *list,
    _P(NodeReference_t) NodeReference
  ){
    #if BLL_set_debug_InvalidAction >= 1
      if(NodeReference == list->src){
        PR_abort();
      }
      if(NodeReference == list->dst){
        PR_abort();
      }
    #endif
    _P(Node_t) *Node = _P(GetNodeByReference)(list, NodeReference);
    #if BLL_set_debug_InvalidAction >= 1
      if(_P(IsNodeUnlinked)(list, Node)){
        PR_abort();
      }
    #endif
    #if BLL_set_SafeNext
      if(list->SafeNext == NodeReference){
        list->SafeNext = Node->PrevNodeReference;
      }
    #endif
    _P(NodeReference_t) nextNodeReference = Node->NextNodeReference;
    _P(NodeReference_t) prevNodeReference = Node->PrevNodeReference;
    _P(GetNodeByReference)(list, prevNodeReference)->NextNodeReference = nextNodeReference;
    _P(GetNodeByReference)(list, nextNodeReference)->PrevNodeReference = prevNodeReference;
  }

  static
  void
  _P(LinkAsFirst)
  (
    _P(t) *list,
    _P(NodeReference_t) NodeReference
  ){
    _P(linkNext)(list, list->src, NodeReference);
  }
  static
  void
  _P(LinkAsLast)
  (
    _P(t) *list,
    _P(NodeReference_t) NodeReference
  ){
    _P(linkPrev)(list, list->dst, NodeReference);
  }

  static
  void
  _P(ReLinkAsFirst)
  (
    _P(t) *list,
    _P(NodeReference_t) NodeReference
  ){
    _P(Unlink)(list, NodeReference);
    _P(LinkAsFirst)(list, NodeReference);
  }
  static
  void
  _P(ReLinkAsLast)
  (
    _P(t) *list,
    _P(NodeReference_t) NodeReference
  ){
    _P(Unlink)(list, NodeReference);
    _P(LinkAsLast)(list, NodeReference);
  }
#endif

static
void
_P(Recycle)
(
  _P(t) *list,
  _P(NodeReference_t) NodeReference
){
  _P(Node_t) *Node = _P(GetNodeByReference)(list, NodeReference);

  Node->NextNodeReference = list->e.c;
  #if BLL_set_IsNodeUnlinked == 1
    Node->PrevNodeReference = _P(NodeReference_t)-1;
  #endif
  list->e.c = NodeReference;
  list->e.p++;
}

#if BLL_set_Link == 1
  static
  _P(NodeReference_t)
  _P(GetNodeFirst)
  (
    _P(t) *list
  ){
    return _PP(GetNodeByReference)(list, list->src)->NextNodeReference;
  }
  static
  _P(NodeReference_t)
  _P(GetNodeLast)
  (
    _P(t) *list
  ){
    return _PP(GetNodeByReference)(list, list->dst)->PrevNodeReference;
  }

  static
  bool
  _P(IsNodeReferenceFronter)
  (
    _P(t) *list,
    _P(NodeReference_t) srcNodeReference,
    _P(NodeReference_t) dstNodeReference
  ){
    do{
      _P(Node_t) *srcNode = _P(GetNodeByReference)(
        list,
        srcNodeReference
      );
      srcNodeReference = srcNode->NextNodeReference;
      if(_P(IsNodeReferenceEqual)(srcNodeReference, dstNodeReference)){
        return 0;
      }
    }while(!_P(IsNodeReferenceEqual)(srcNodeReference, list->dst));
    return 1;
  }
#endif

#if BLL_set_SafeNext
  static
  void
  _P(StartSafeNext)
  (
    _P(t) *list,
    _P(NodeReference_t) NodeReference
  ){
    #if BLL_set_debug_InvalidAction == 1
      if(list->SafeNext != _P(NodeReference_t)-1){
        PR_abort();
      }
    #endif
    list->SafeNext = NodeReference;
  }
  static
  _P(NodeReference_t)
  _P(EndSafeNext)
  (
    _P(t) *list
  ){
    #if BLL_set_debug_InvalidAction == 1
      if(list->SafeNext == _P(NodeReference_t)-1){
        PR_abort();
      }
    #endif
    _P(Node_t) *Node = _P(GetNodeByReference)(
      list,
      list->SafeNext
    );
    list->SafeNext = (_P(NodeReference_t))-1;
    return Node->NextNodeReference;
  }
#endif

static
void *
_P(GetNodeReferenceData)
(
  _P(t) *list,
  _P(NodeReference_t) NodeReference
){
  _P(Node_t) *Node = _P(GetNodeByReference)(
    list,
    NodeReference
  );
  #ifdef BLL_set_node_data
    return (void *)&Node->data;
  #else
    return (void *)((uint8_t *)Node + sizeof(_P(Node_t)));
  #endif
}

#ifdef BLL_set_namespace
  }
#endif
