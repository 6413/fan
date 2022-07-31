#ifdef BLL_set_namespace
  namespace BLL_set_namespace {
#endif

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

#if BLL_set_StoreFormat == 0
  static
  bool
  _P(IsNodeReferenceInvalid)
  (
    _P(t) *list,
    _P(NodeReference_t) NodeReference
  ){
    #if BLL_set_BaseLibrary == 0
      return NodeReference >= list->nodes.Current;
    #elif BLL_set_BaseLibrary == 1
      return NodeReference >= list->nodes.size();
    #endif
  }
#else
  /* pain */
#endif

#if BLL_set_IsNodeUnlinked == 1
  static
  bool
  _P(IsNodeUnlinked)
  (
    _P(t) *list,
    _P(Node_t) *Node
  ){
    if(Node->PrevNodeReference == -1){
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
      return &((_P(Node_t) *)&list->nodes.ptr[0])[NodeReference];
    #else
      return (_P(Node_t) *)&list->nodes.ptr[NodeReference * list->NodeSize];
    #endif
  #elif BLL_set_StoreFormat == 1
    return (_P(Node_t) *)NodeReference;
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

#if BLL_set_StoreFormat == 0
  static
  _P(NodeReference_t)
  _P(usage)
  (
    _P(t) *list
  ){
    #if BLL_set_BaseLibrary == 0
      return list->nodes.Current - list->e.p - 2;
    #elif BLL_set_BaseLibrary == 1
      return list->nodes.size() - list->e.p - 2;
    #endif
  }
#else
  /* pain */
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
  #if BLL_set_StoreFormat == 0
    #if BLL_set_BaseLibrary == 0
      VEC_handle(&list->nodes);
      return list->nodes.Current++;
    #elif BLL_set_BaseLibrary == 1
      return list->nodes.push_back({});
    #endif
  #elif BLL_set_StoreFormat == 1
    return (_P(NodeReference_t))BLL_set_StoreFormat1_alloc_open(_P(GetNodeSize)(list));
  #endif
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
    Node->PrevNodeReference = -3;
    Node->NextNodeReference = -3;
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
  _P(_StoreFormat1_close_UsedNodes)
  (
    _P(t) *list
  ){
    _P(NodeReference_t) nr = list->src;
    while(nr != list->dst){
      _P(NodeReference_t) nnr = _P(GetNodeByReference)(list, nr)->NextNodeReference;
      BLL_set_StoreFormat1_alloc_close((void *)nr);
      nr = nnr;
    }
  }
  static
  void
  _P(_StoreFormat1_close_EmptyNodes)
  (
    _P(t) *list
  ){
    while(list->e.p){
      _P(NodeReference_t) nr = _P(NewNode_empty)(list);
      BLL_set_StoreFormat1_alloc_close((void *)nr);
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
    #if BLL_set_BaseLibrary == 0
      VEC_handle0(&list->nodes, 2);
    #elif BLL_set_BaseLibrary == 1
      list->nodes.resize(2);
    #endif
    
    #if BLL_set_Link == 1
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
    list->e.c = 0;
  #endif

  #if BLL_set_StoreFormat == 0
    #if BLL_set_BaseLibrary == 0
      VEC_init(&list->nodes, NodeSize, A_resize);
    #elif BLL_set_BaseLibrary == 1
      list->nodes.open();
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
    _P(_StoreFormat1_close_UsedNodes)(list);
    _P(_StoreFormat1_close_EmptyNodes)(list);
  #endif
}
static
void
_P(Clear)
(
  _P(t) *list
){
  #if BLL_set_StoreFormat == 1
    _P(_StoreFormat1_close_UsedNodes)(list);
  #endif
  #if BLL_set_ResizeListAfterClear
    #if BLL_set_StoreFormat == 0
      list->nodes.Possible = 2;
      list->nodes.ptr = list->nodes.resize(list->nodes.ptr, list->nodes.Possible * list->nodes.Type);
    #elif BLL_set_StoreFormat == 1
      _P(_StoreFormat1_close_EmptyNodes)(list);
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
    Node->PrevNodeReference = -1;
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
      if(srcNodeReference == dstNodeReference){
        return 0;
      }
    }while(srcNodeReference != list->dst);
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
