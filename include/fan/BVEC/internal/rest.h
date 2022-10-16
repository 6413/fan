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
    #if BVEC_set_BufferingFormat == 0
      BVEC_set_NodeType BufferAmount;
    #endif
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

#if BVEC_set_BufferingFormat == 0
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
#endif

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
    #if BVEC_set_BufferingFormat == 0
      List->BufferAmount = BVEC_set_WantedBufferByteAmount / List->NodeSize;
      if(List->BufferAmount == 0){
        List->BufferAmount = 1;
      }
    #endif
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
  List->ptr = NULL;
}

static
void
_BVEC_P(_Resize)(
  _BVEC_P(t) *List
){
  for(uint32_t i = 0; i < BVEC_set_alloc_RetryAmount; i++){
    void *np = BVEC_set_alloc_resize(List->ptr, List->Possible * _BVEC_P(GetNodeSize)(List));
    if(np == NULL){
      continue;
    }
    List->ptr = (_BVEC_P(Node_t) *)np;
    return;
  }
  BVEC_set_abort
}

static
void
_BVEC_P(Reserve)
(
  _BVEC_P(t) *List,
  BVEC_set_NodeType Amount
){
  List->Possible = Amount;
  _BVEC_P(_Resize)(List);
}

static
void
_BVEC_P(_AllocateBuffer)
(
  _BVEC_P(t) *List
){
  #if BVEC_set_BufferingFormat == 0
  	List->Possible += _BVEC_P(GetBufferAmount)(List);
  #elif BVEC_set_BufferingFormat == 1
    List->Possible = List->Possible ? List->Possible * 2 : 1;
  #endif
	_BVEC_P(_Resize)(List);
}
static
void
_BVEC_P(_AllocateBufferFromCurrent)
(
  _BVEC_P(t) *List
){
  #if BVEC_set_BufferingFormat == 0
	  List->Possible = List->Current + _BVEC_P(GetBufferAmount)(List);
  #elif BVEC_set_BufferingFormat == 1
    List->Possible = 1 << sizeof(uintptr_t) * 8 - __clz(List->Current);
  #endif
	_BVEC_P(_Resize)(List);
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
