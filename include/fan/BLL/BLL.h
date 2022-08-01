#ifndef BLL_set_BaseLibrary
  #define BLL_set_BaseLibrary 0
#endif

#ifndef BLL_set_KeepSettings
  #define BLL_set_KeepSettings 0
#endif
#ifndef BLL_set_prefix
  #error ifndef BLL_set_prefix
#endif
#ifndef BLL_set_StructFormat
  #define BLL_set_StructFormat 0
#endif
#ifndef BLL_set_declare_basic_types
  #define BLL_set_declare_basic_types 1
#endif
#ifndef BLL_set_declare_rest
  #define BLL_set_declare_rest 1
#endif
/* if you access next more than prev it can make performance difference */
#ifndef BLL_set_PreferNextFirst
  #define BLL_set_PreferNextFirst 1
#endif
#ifndef BLL_set_PadNode
  #define BLL_set_PadNode 0
#endif
#ifndef BLL_set_debug_InvalidAction
  #define BLL_set_debug_InvalidAction 0
#endif
#ifndef BLL_set_IsNodeUnlinked
  #if BLL_set_debug_InvalidAction == 1
    #define BLL_set_IsNodeUnlinked 1
  #else
    #define BLL_set_IsNodeUnlinked 0
  #endif
#endif
#ifndef BLL_set_SafeNext
  #define BLL_set_SafeNext 0
#endif
#ifndef BLL_set_ResizeListAfterClear
  #define BLL_set_ResizeListAfterClear 0
#endif
#ifndef BLL_set_UseUninitialisedValues
  #if BLL_set_BaseLibrary == 0
    #define BLL_set_UseUninitialisedValues WITCH_set_UseUninitialisedValues
  #elif BLL_set_BaseLibrary == 1
    #define BLL_set_UseUninitialisedValues fan_use_uninitialized
  #endif
#endif
#ifndef BLL_set_Link
  #define BLL_set_Link 1
#endif
#ifndef BLL_set_StoreFormat
  #define BLL_set_StoreFormat 0
#endif
#ifndef BLL_set_type_node
  #if BLL_set_StoreFormat == 1
    #define BLL_set_type_node uintptr_t
  #else
    #define BLL_set_type_node uint32_t
  #endif
#else
  #if BLL_set_StoreFormat == 1
    #error when (BLL_set_StoreFormat == 1) you cant set BLL_set_StoreFormat.
  #endif
#endif
#ifndef BLL_set_SyntaxStyle
  #define BLL_set_SyntaxStyle 0
#endif

#if BLL_set_Link == 0
  #if BLL_set_SafeNext != 0
    #error SafeNext is not possible when there is not linking.
  #endif
  #if BLL_set_IsNodeUnlinked != 0
    #error (IsNodeUnlinked != 0) is not available with (Link == 0) yet.
  #endif
#endif

#if BLL_set_StoreFormat == 1
  #ifndef BLL_set_StoreFormat1_alloc_open
    #error ?
  #endif
  #ifndef BLL_set_StoreFormat1_alloc_close
    #error ?
  #endif
#endif

#if BLL_set_debug_InvalidAction == 1
  #if BLL_set_IsNodeUnlinked == 0
    #error BLL_set_IsNodeUnlinked cant be 0 when BLL_set_debug_InvalidAction is 1
  #endif
  #ifndef BLL_set_debug_InvalidAction_srcAccess
    #define BLL_set_debug_InvalidAction_srcAccess 1
  #endif
  #ifndef BLL_set_debug_InvalidAction_dstAccess
    #define BLL_set_debug_InvalidAction_dstAccess 1
  #endif
#else
  #ifndef BLL_set_debug_InvalidAction_srcAccess
    #define BLL_set_debug_InvalidAction_srcAccess 0
  #endif
  #ifndef BLL_set_debug_InvalidAction_dstAccess
    #define BLL_set_debug_InvalidAction_dstAccess 0
  #endif
#endif

#if BLL_set_IsNodeUnlinked == 1
  #if BLL_set_Link == 0
    #error ?
  #endif
#endif

#if BLL_set_BaseLibrary == 0
  #define _BLL_INCLUDE _WITCH_PATH
#elif BLL_set_BaseLibrary == 1
  #define _BLL_INCLUDE _FAN_PATH
#endif

#if BLL_set_StoreFormat == 0
  #if BLL_set_BaseLibrary == 0
    #include _BLL_INCLUDE(VEC/VEC.h)
  #elif BLL_set_BaseLibrary == 1
    #include _BLL_INCLUDE(types/memory.h)
  #endif
#endif

#define _P(p0) CONCAT3(BLL_set_prefix, _, p0)
#define _PP(p0) CONCAT4(_, BLL_set_prefix, _, p0)

#if BLL_set_StructFormat == 0
  #define BLL_StructBegin(n) typedef struct{
  #define BLL_StructEnd(n) }n;
#elif BLL_set_StructFormat == 1
  #define BLL_StructBegin(n) struct n{
  #define BLL_StructEnd(n) };
#endif

#if BLL_set_declare_basic_types == 1
  #include _BLL_INCLUDE(BLL/internal/basic_types.h)
#endif
#if BLL_set_declare_rest == 1
  #include _BLL_INCLUDE(BLL/internal/rest.h)
#endif

#undef BLL_StructBegin
#undef BLL_StructEnd

#undef _P
#undef _PP

#undef _BLL_INCLUDE

#if BLL_set_KeepSettings == 0
  #undef BLL_set_KeepSettings
  #undef BLL_set_StructFormat
  #undef BLL_set_prefix
  #undef BLL_set_declare_basic_types
  #undef BLL_set_declare_rest
  #undef BLL_set_type_node
  #ifdef BLL_set_node_data
    #undef BLL_set_node_data
  #endif
  #undef BLL_set_PreferNextFirst
  #undef BLL_set_PadNode
  #undef BLL_set_debug_InvalidAction
  #undef BLL_set_debug_InvalidAction_srcAccess
  #undef BLL_set_debug_InvalidAction_dstAccess
  #undef BLL_set_IsNodeUnlinked
  #undef BLL_set_SafeNext
  #undef BLL_set_ResizeListAfterClear
  #undef BLL_set_UseUninitialisedValues
  #undef BLL_set_Link
  #undef BLL_set_StoreFormat
  #undef BLL_set_SyntaxStyle
  #ifdef BLL_set_NodeReference_Overload_Declare
    #undef BLL_set_NodeReference_Overload_Declare
  #endif
  #ifdef BLL_set_namespace
    #undef BLL_set_namespace
  #endif

  #undef BLL_set_BaseLibrary
#endif
