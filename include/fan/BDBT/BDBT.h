#ifndef BDBT_set_BaseLibrary
  #define BDBT_set_BaseLibrary 0
#endif
#if BDBT_set_BaseLibrary == 0
  #define _BDBT_INCLUDE _WITCH_PATH
#elif BDBT_set_BaseLibrary == 1
  #define _BDBT_INCLUDE _FAN_PATH
#endif

#ifndef BDBT_set_prefix
  #error ifndef BDBT_set_prefix
#endif
#ifndef BDBT_set_declare_basic_types
  #define BDBT_set_declare_basic_types 1
#endif
#ifndef BDBT_set_declare_rest
  #define BDBT_set_declare_rest 1
#endif
#ifndef BDBT_set_declare_Key
  #define BDBT_set_declare_Key 1
#endif
#ifndef BDBT_set_PadNode
  #define BDBT_set_PadNode 0
#endif
#ifndef BDBT_set_debug_InvalidAction
  #define BDBT_set_debug_InvalidAction 0
#endif
#ifndef BDBT_set_IsNodeUnlinked
  #if BDBT_set_debug_InvalidAction == 1
    #define BDBT_set_IsNodeUnlinked 1
  #else
    #define BDBT_set_IsNodeUnlinked 0
  #endif
#endif
#ifndef BDBT_set_ResizeListAfterClear
  #define BDBT_set_ResizeListAfterClear 0
#endif
#ifndef BDBT_set_UseUninitialisedValues
  #if BDBT_set_BaseLibrary == 0
    #define BDBT_set_UseUninitialisedValues WITCH_set_UseUninitialisedValues
  #elif BDBT_set_BaseLibrary == 1
    #define BDBT_set_UseUninitialisedValues fan_use_uninitialized
  #endif
#endif
#ifndef BDBT_set_type_node
  #define BDBT_set_type_node uint32_t
#endif
#ifndef BDBT_set_BitPerNode
  #define BDBT_set_BitPerNode 2
#endif

#if BDBT_set_BitPerNode == 1
  #define _BDBT_set_ElementPerNode 0x2
#elif BDBT_set_BitPerNode == 2
  #define _BDBT_set_ElementPerNode 0x4
#elif BDBT_set_BitPerNode == 4
  #define _BDBT_set_ElementPerNode 0x10
#elif BDBT_set_BitPerNode == 8
  #define _BDBT_set_ElementPerNode 0x100
#else
  #error ?
#endif

#if BDBT_set_BaseLibrary == 0
  #include _BDBT_INCLUDE(VEC/VEC.h)
#elif BDBT_set_BaseLibrary == 1
  #include _BDBT_INCLUDE(types/memory.h)
#endif

#ifdef BDBT_set_base_prefix
  #define _BDBT_BP(p0) CONCAT3(BDBT_set_base_prefix, _, p0)
#else
  #define _BDBT_BP(p0) CONCAT3(BDBT_set_prefix, _, p0)
#endif
#define _BDBT_P(p0) CONCAT3(BDBT_set_prefix, _, p0)

#if BDBT_set_declare_basic_types == 1
  #include _BDBT_INCLUDE(BDBT/internal/basic_types.h)
#endif
#if BDBT_set_declare_rest == 1
  #include _BDBT_INCLUDE(BDBT/internal/rest.h)
#endif
#if BDBT_set_declare_Key == 1
  #include _BDBT_INCLUDE(BDBT/internal/Key/Key.h)
#endif

#undef _BDBT_P
#undef _BDBT_BP

#undef _BDBT_set_ElementPerNode
#undef BDBT_set_BitPerNode
#undef BDBT_set_prefix
#undef BDBT_set_declare_Key
#undef BDBT_set_declare_basic_types
#undef BDBT_set_declare_rest
#undef BDBT_set_type_node
#undef BDBT_set_PadNode
#undef BDBT_set_debug_InvalidAction
#undef BDBT_set_IsNodeUnlinked
#undef BDBT_set_ResizeListAfterClear
#undef BDBT_set_UseUninitialisedValues
#ifdef BDBT_set_base_namespace
  #undef BDBT_set_base_namespace
#endif
#ifdef BDBT_set_namespace
  #undef BDBT_set_namespace
#endif

#undef _BDBT_INCLUDE
#undef BDBT_set_BaseLibrary
