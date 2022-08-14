#ifndef BVEC_set_BaseLibrary
  #define BVEC_set_BaseLibrary 0
#endif

#ifndef BVEC_set_prefix
  #error ifndef BVEC_set_prefix
#endif
#ifndef BVEC_set_NodeType
  #define BVEC_set_NodeType uint32_t
#endif
#ifndef BVEC_set_NodeSizeType
  #define BVEC_set_NodeSizeType uint32_t
#endif
#ifndef BVEC_set_WantedBufferByteAmount
  #define BVEC_set_WantedBufferByteAmount 512
#endif
#ifndef BVEC_set_MemoryCopy
  #define BVEC_set_MemoryCopy __builtin_memcpy
#endif

#ifndef BVEC_set_alloc_open
  #define BVEC_set_alloc_open malloc
#endif
#ifndef BVEC_set_alloc_resize
  #define BVEC_set_alloc_resize realloc
#endif
#ifndef BVEC_set_alloc_close
  #define BVEC_set_alloc_close free
#endif

#define BVEC_BufferDivide (BVEC_set_WantedBufferByteAmount / sizeof(BVEC_set_NodeData))
#define BVEC_BufferAmount (BVEC_BufferDivide == 0 ? 1 : BVEC_BufferDivide)

#define _BVEC_P(p0) CONCAT3(BVEC_set_prefix, _, p0)

#include "internal/rest.h"

#undef _BVEC_P

#undef BVEC_BufferAmount
#undef BVEC_BufferDivide

#undef BVEC_set_alloc_close
#undef BVEC_set_alloc_resize
#undef BVEC_set_alloc_open

#undef BVEC_set_MemoryCopy
#undef BVEC_set_WantedBufferByteAmount
#ifdef BVEC_set_NodeData
  #undef BVEC_set_NodeData
#endif
#undef BVEC_set_NodeSizeType
#undef BVEC_set_NodeType
#undef BVEC_set_prefix

#undef BVEC_set_BaseLibrary
