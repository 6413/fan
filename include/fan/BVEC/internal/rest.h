#ifdef BVEC_set_NodeData
  typedef BVEC_set_NodeData _BVEC_P(Node_t);
#else
  typedef void _BVEC_P(Node_t);
#endif

typedef struct{
  BVEC_set_NodeType Current;
  BVEC_set_NodeType Possible;
  _BVEC_P(Node_t) *ptr;
  #ifndef BVEC_set_NodeData
    BVEC_set_NodeSizeType NodeSize;
    BVEC_set_NodeType BufferAmount;
  #endif
}_BVEC_P(t);

static
BVEC_set_NodeSizeType
_BVEC_P(GetNodeSize)
(
  _BVEC_P(t) *List
){
  #ifdef BVEC_set_NodeData
    return sizeof(BVEC_set_NodeData);
  #else
    return List->NodeSize;
  #endif
}

static
_BVEC_P(Node_t) *
_BVEC_P(GetNode)
(
  _BVEC_P(t) *List,
  BVEC_set_NodeType NR
){
  #ifdef BVEC_set_NodeData
    return &List->ptr[NR];
  #else
    return (_BVEC_P(Node_t) *)&((uint8_t *)List->ptr)[NR * _BVEC_P(GetNodeSize)(List)];
  #endif
}

static
BVEC_set_NodeType
_BVEC_P(GetBufferAmount)
(
  _BVEC_P(t) *List
){
  #ifdef BVEC_set_NodeData
    return BVEC_BufferAmount;
  #else
    return List->BufferAmount;
  #endif
}

static
void
_BVEC_P(Close)
(
  _BVEC_P(t) *List
){
  BVEC_set_alloc_close(List->ptr);
}
static
void
_BVEC_P(Open)
(
  _BVEC_P(t) *List
  #ifndef BVEC_set_NodeData
    , BVEC_set_NodeSizeType NodeSize
  #endif
){
  List->Current = 0;
  List->Possible = 0;
  List->ptr = 0;
  #ifndef BVEC_set_NodeData
    List->NodeSize = NodeSize;
    List->BufferAmount = BVEC_set_WantedBufferByteAmount / List->NodeSize;
    if(List->BufferAmount == 0){
      List->BufferAmount = 1;
    }
  #endif
}
static
void
_BVEC_P(Clear)
(
  _BVEC_P(t) *List
){
  List->Current = 0;
}
static
void
_BVEC_P(ClearWithBuffer)
(
  _BVEC_P(t) *List
){
  BVEC_set_alloc_close(List->ptr);
  List->Current = 0;
  List->Possible = 0;
  List->ptr = 0;
}

static
void
_BVEC_P(Reserve)
(
  _BVEC_P(t) *List,
  BVEC_set_NodeType Amount
){
  List->Possible = Amount;
  List->ptr = (_BVEC_P(Node_t) *)BVEC_set_alloc_resize(List->ptr, List->Possible * _BVEC_P(GetNodeSize)(List));
}

static
void
_BVEC_P(_AllocateBuffer)
(
  _BVEC_P(t) *List
){
	List->Possible += _BVEC_P(GetBufferAmount)(List);
	List->ptr = (_BVEC_P(Node_t) *)BVEC_set_alloc_resize(List->ptr, List->Possible * _BVEC_P(GetNodeSize)(List));
}
static
void
_BVEC_P(_AllocateBufferFromCurrent)
(
  _BVEC_P(t) *List
){
	List->Possible = List->Current + _BVEC_P(GetBufferAmount)(List);
	List->ptr = (_BVEC_P(Node_t) *)BVEC_set_alloc_resize(List->ptr, List->Possible * _BVEC_P(GetNodeSize)(List));
}

static
void
_BVEC_P(Add)
(
  _BVEC_P(t) *List,
  _BVEC_P(Node_t) *Node
){
  if(List->Current == List->Possible){
    _BVEC_P(_AllocateBuffer)(List);
  }

  #ifdef BVEC_set_NodeData
    List->ptr[List->Current] = *Node;
  #else
    BVEC_set_MemoryCopy(_BVEC_P(GetNode)(List, List->Current), Node, List->NodeSize);
  #endif
  ++List->Current;
}

static
void
_BVEC_P(AddEmpty)
(
  _BVEC_P(t) *List,
  BVEC_set_NodeType Amount
){
  List->Current += Amount;
  if(List->Current >= List->Possible){
    _BVEC_P(_AllocateBufferFromCurrent)(List);
  }
}
