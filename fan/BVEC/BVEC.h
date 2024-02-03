#ifdef BVEC_set_BaseLibrary
  #error BVEC_set_BaseLibrary is no longer supported. you can define BVEC_set_abort if you want.
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
#ifndef BVEC_set_MemoryCopy
  #define BVEC_set_MemoryCopy __MemoryCopy
#endif

#ifndef BVEC_set_HandleAllocate
  #define BVEC_set_HandleAllocate 1
#endif

#ifndef BVEC_set_BufferingFormat
  #define BVEC_set_BufferingFormat 1
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

#ifndef BVEC_set_alloc_RetryAmount
  #define BVEC_set_alloc_RetryAmount 0x10
#endif

#ifndef BVEC_set_abort
  #define BVEC_set_abort __abort();
#endif

#if BVEC_set_BufferingFormat == 0
  #ifndef BVEC_set_BufferingFormat0_WantedBufferByteAmount
    #define BVEC_set_BufferingFormat0_WantedBufferByteAmount 512
  #endif
  #define BVEC_BufferDivide (BVEC_set_BufferingFormat0_WantedBufferByteAmount / sizeof(BVEC_set_NodeData))
  #define BVEC_BufferAmount (BVEC_BufferDivide == 0 ? 1 : BVEC_BufferDivide)
#endif

#define _BVEC_P(p0) CONCAT3(BVEC_set_prefix, _, p0)

#include "internal/rest.h"

#undef _BVEC_P

#if BVEC_set_BufferingFormat == 0
  #undef BVEC_BufferAmount
  #undef BVEC_BufferDivide
  #undef BVEC_set_BufferingFormat0_WantedBufferByteAmount
#endif

#undef BVEC_set_alloc_RetryAmount

#undef BVEC_set_abort

#undef BVEC_set_alloc_close
#undef BVEC_set_alloc_resize
#undef BVEC_set_alloc_open

#undef BVEC_set_BufferingFormat

#undef BVEC_set_HandleAllocate

#undef BVEC_set_MemoryCopy
#ifdef BVEC_set_NodeData
  #undef BVEC_set_NodeData
#endif
#undef BVEC_set_NodeSizeType
#undef BVEC_set_NodeType
#ifdef BVEC_set_MultipleType
  #ifdef BVEC_set_MultipleType_Sizes
    #undef BVEC_set_MultipleType_Sizes
  #endif
  #ifdef BVEC_set_MultipleType_SizeArray
    #undef BVEC_set_MultipleType_SizeArray
  #endif
  #undef BVEC_set_MultipleType
#endif
#undef BVEC_set_prefix
