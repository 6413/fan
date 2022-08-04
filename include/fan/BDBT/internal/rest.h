#ifdef BDBT_set_namespace
  namespace BDBT_set_namespace {
#endif

static
bool
_BDBT_P(IsNodeReferenceInvalid)
(
  _BDBT_P(t) *list,
  _BDBT_P(NodeReference_t) NodeReference
){
  #if BDBT_set_BaseLibrary == 0
    return NodeReference >= list->nodes.Current;
  #elif BDBT_set_BaseLibrary == 1
    return NodeReference >= list->nodes.size();
  #endif
}

static
_BDBT_P(NodeReference_t)
_BDBT_P(GetNotValidNodeReference)
(
  _BDBT_P(t) *list
){
  return (_BDBT_P(NodeReference_t))-1;
}

#if BDBT_set_IsNodeUnlinked == 1
  static
  bool
  _BDBT_P(IsNodeUnlinked)
  (
    _BDBT_P(t) *list,
    _BDBT_P(Node_t) *Node
  ){
    if(Node->n[1] == _BDBT_P(NodeReference_t)-1){
      return 1;
    }
    return 0;
  }
  static
  bool
  _BDBT_P(IsNodeReferenceUnlinked)
  (
    _BDBT_P(t) *list,
    _BDBT_P(NodeReference_t) NodeReference
  ){
    _BDBT_P(Node_t) *Node = &((_BDBT_P(Node_t) *)&list->nodes.ptr[0])[NodeReference];
    return _BDBT_P(IsNodeUnlinked)(list, Node);
  }
#endif

static
_BDBT_P(Node_t) *
_BDBT_P(_GetNodeByReference)
(
  _BDBT_P(t) *list,
  _BDBT_P(NodeReference_t) NodeReference
){
  return &((_BDBT_P(Node_t) *)&list->nodes.ptr[0])[NodeReference];
}

static
_BDBT_P(Node_t) *
_BDBT_P(GetNodeByReference)
(
  _BDBT_P(t) *list,
  _BDBT_P(NodeReference_t) NodeReference
){
  #if BDBT_set_debug_InvalidAction == 1
    if(NodeReference >= list->nodes.Current){
      PR_abort();
    }
  #endif
  _BDBT_P(Node_t) *Node = _BDBT_P(_GetNodeByReference)(list, NodeReference);
  #if BDBT_set_debug_InvalidAction == 1
    do{
      if(_BDBT_P(IsNodeUnlinked)(list, Node)){
        PR_abort();
      }
    }while(0);
  #endif
  return Node;
}

#if BDBT_set_StoreFormat == 0
  static
  _BDBT_P(NodeReference_t)
  _BDBT_P(usage)
  (
    _BDBT_P(t) *list
  ){
    #if BDBT_set_BaseLibrary == 0
      return list->nodes.Current - list->e.p;
    #elif BDBT_set_BaseLibrary == 1
      return list->nodes.size() - list->e.p;
    #endif
  }
#else
  /* pain */
#endif

static
_BDBT_P(NodeReference_t)
_BDBT_P(NewNode_empty)
(
  _BDBT_P(t) *list
){
  _BDBT_P(NodeReference_t) NodeReference = list->e.c;
  list->e.c = _BDBT_P(_GetNodeByReference)(list, NodeReference)->n[0];
  list->e.p--;
  return NodeReference;
}
static
_BDBT_P(NodeReference_t)
_BDBT_P(NewNode_alloc)
(
  _BDBT_P(t) *list
){
  #if BDBT_set_BaseLibrary == 0
    VEC_handle(&list->nodes);
    return list->nodes.Current++;
  #elif BDBT_set_BaseLibrary == 1
    return list->nodes.push_back({});
  #endif
}
static
_BDBT_P(NodeReference_t)
_BDBT_P(NewNode)
(
  _BDBT_P(t) *list
){
  _BDBT_P(NodeReference_t) NodeReference;
  if(list->e.p){
    NodeReference = _BDBT_P(NewNode_empty)(list);
  }
  else{
    NodeReference = _BDBT_P(NewNode_alloc)(list);
  }

  {
    _BDBT_P(Node_t) *Node = _BDBT_P(_GetNodeByReference)(list, NodeReference);
    for(_BDBT_P(NodeEIT_t) i = 0; i < _BDBT_set_ElementPerNode; i++){
      Node->n[i] = _BDBT_P(GetNotValidNodeReference)(list);
    }
  }

  return NodeReference;
}

static
void
_BDBT_P(_AfterInitNodes)
(
  _BDBT_P(t) *list
){
  list->e.p = 0;
}

static
void
_BDBT_P(open)
(
  _BDBT_P(t) *list
){
  #if BDBT_set_UseUninitialisedValues == 0
    list->e.c = 0;
  #endif

  #if BDBT_set_BaseLibrary == 0
    VEC_init(&list->nodes, sizeof(_BDBT_P(Node_t)), A_resize);
  #elif BDBT_set_BaseLibrary == 1
    list->nodes.open();
  #endif
  _BDBT_P(_AfterInitNodes)(list);
}
static
void
_BDBT_P(close)
(
  _BDBT_P(t) *list
){
  #if BDBT_set_BaseLibrary == 0
    VEC_free(&list->nodes);
  #elif BDBT_set_BaseLibrary == 1
    list->nodes.close();
  #endif
}
static
void
_BDBT_P(Clear)
(
  _BDBT_P(t) *list
){
  #if BDBT_set_ResizeListAfterClear
    #if BDBT_set_BaseLibrary == 0
      list->nodes.Possible = 2;
      list->nodes.ptr = list->nodes.resize(list->nodes.ptr, list->nodes.Possible * list->nodes.Type);
    #elif BDBT_set_BaseLibrary == 1
      list->nodes.resize(0);
    #endif
  #endif
  _BDBT_P(_AfterInitNodes)(list);
}

static
void
_BDBT_P(Recycle)
(
  _BDBT_P(t) *list,
  _BDBT_P(NodeReference_t) NodeReference
){
  _BDBT_P(Node_t) *Node = _BDBT_P(GetNodeByReference)(list, NodeReference);

  Node->n[0] = list->e.c;
  #if BDBT_set_IsNodeUnlinked == 1
    Node->n[1] = _BDBT_P(NodeReference_t)-1;
  #endif
  list->e.c = NodeReference;
  list->e.p++;
}

static
void
_BDBT_P(PreAllocateNodes)
(
  _BDBT_P(t) *list,
  _BDBT_P(NodeReference_t) Amount
){
  #if BDBT_set_BaseLibrary == 0
    VEC_reserve(&list->nodes, Amount);
  #elif BDBT_set_BaseLibrary == 1
    list->nodes.reserve(Amount);
  #endif
}

#ifdef BDBT_set_namespace
  }
#endif
