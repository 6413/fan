#ifdef BDBT_set_namespace
  namespace BDBT_set_namespace {
#endif

typedef BDBT_set_type_node _BDBT_P(NodeReference_t);

#if _BDBT_set_ElementPerNode < 0xff
  typedef uint8_t _BDBT_P(NodeEIT_t);
#elif _BDBT_set_ElementPerNode < 0xffff
  typedef uint16_t _BDBT_P(NodeEIT_t);
#else
  #error no
#endif

#if BDBT_set_PadNode == 0
  #pragma pack(push, 1)
#endif
BDBT_StructBegin(_BDBT_P(Node_t))
  _BDBT_P(NodeReference_t) n[_BDBT_set_ElementPerNode];
BDBT_StructEnd(_BDBT_P(Node_t))
#if BDBT_set_PadNode == 0
  #pragma pack(pop)
#endif

#define BVEC_set_prefix _BDBT_P(_NodeList)
#define BVEC_set_NodeType BDBT_set_type_node
#define BVEC_set_NodeData _BDBT_P(Node_t)
#include _BDBT_INCLUDE(BVEC/BVEC.h)

#ifdef BDBT_set_CPP_ConstructDestruct
  struct _BDBT_P(t);
  static void _BDBT_P(Open)(_BDBT_P(t) *list);
  static void _BDBT_P(Close)(_BDBT_P(t) *list);
#endif

BDBT_StructBegin(_BDBT_P(t))
  #if BDBT_set_StoreFormat == 0
    _BDBT_P(_NodeList_t) NodeList;
  #endif
  struct{
    _BDBT_P(NodeReference_t) c;
    _BDBT_P(NodeReference_t) p;
  }e;

  #ifdef BDBT_set_CPP_ConstructDestruct
    _BDBT_P(t)(){
      _BDBT_P(Open)(this);
    }
    ~_BDBT_P(t)(){
      _BDBT_P(Close)(this);
    }
  #endif
BDBT_StructEnd(_BDBT_P(t))

/* is node reference invalid */
static
bool
_BDBT_P(inri)
(
  _BDBT_P(t) *list,
  _BDBT_P(NodeReference_t) NodeReference
){
  return NodeReference >= list->NodeList.Current;
}

/* is node reference invalid constant */
static
bool
_BDBT_P(inric)
(
  _BDBT_P(t) *list,
  _BDBT_P(NodeReference_t) NodeReference
){
  return NodeReference == (_BDBT_P(NodeReference_t))-1;
}

/* get node reference invalid constant */
static
_BDBT_P(NodeReference_t)
_BDBT_P(gnric)
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
    if(Node->n[1] == (_BDBT_P(NodeReference_t))-1){
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
  return &((_BDBT_P(Node_t) *)&list->NodeList.ptr[0])[NodeReference];
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
    return list->NodeList.Current - list->e.p;
  }
#else
  #error ?
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
  _BDBT_P(_NodeList_AddEmpty)(&list->NodeList, 1);
  return list->NodeList.Current - 1;
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
      Node->n[i] = _BDBT_P(gnric)(list);
    }
  }

  return NodeReference;
}
static
_BDBT_P(NodeReference_t)
_BDBT_P(NewNodeBranchly)
(
  _BDBT_P(t) *list,
  _BDBT_P(NodeReference_t) *BNR /* branch node references */
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
      Node->n[i] = BNR[i];
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
_BDBT_P(Open)
(
  _BDBT_P(t) *list
){
  #if BDBT_set_UseUninitialisedValues == 0
    list->e.c = 0;
  #endif

  _BDBT_P(_NodeList_Open)(&list->NodeList);
  _BDBT_P(_AfterInitNodes)(list);
}
static
void
_BDBT_P(Close)
(
  _BDBT_P(t) *list
){
  _BDBT_P(_NodeList_Close)(&list->NodeList);
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
  _BDBT_P(_NodeList_Reserve)(&list->NodeList, Amount);
}

/* is node reference has child */
static
bool
_BDBT_P(inrhc)
(
  _BDBT_P(t) *list,
  _BDBT_P(NodeReference_t) nr
){
  _BDBT_P(Node_t) *n = _BDBT_P(GetNodeByReference)(list, nr);
  for(_BDBT_P(NodeEIT_t) i = 0; i < _BDBT_set_ElementPerNode; i++){
    if(_BDBT_P(inric)(list, n->n[i]) == 0){
      return 1;
    }
  }
  return 0;
}

#ifdef BDBT_set_namespace
  }
#endif
