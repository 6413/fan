/* +++ outdated +++ */

#ifdef BLL_set_KeepSettings
  #error outdated setting. dont use it
#endif
#ifdef BLL_set_declare_basic_types
  #error outdated setting. now it shipped with BLL_set_declare_rest
#endif
#ifdef BLL_set_ConstantInvalidNodeReference_Listless
  #error outdated setting.
#endif

/* --- outdated --- */

/* can be used for debug internal code inside bll with checking preprocessor */
#ifndef BLL_set_Mark
  #define BLL_set_Mark 0
#endif

#ifndef BLL_set_BaseLibrary
  #define BLL_set_BaseLibrary 0
#endif

#ifndef BLL_set_Language
  #if BLL_set_BaseLibrary == 0
    #define BLL_set_Language 0
  #elif BLL_set_BaseLibrary == 1
    #define BLL_set_Language 1
  #else
    #error ?
  #endif
#endif

#ifndef BLL_set_AreWeInsideStruct
  #define BLL_set_AreWeInsideStruct 0
#endif

#ifndef BLL_set_prefix
  #error ifndef BLL_set_prefix
#endif
#ifndef BLL_set_StructFormat
  #if BLL_set_Language == 0
    #define BLL_set_StructFormat 0
  #elif BLL_set_Language == 1
    #define BLL_set_StructFormat 1
  #else
    #error ?
  #endif
#endif
#ifndef BLL_set_declare_NodeReference
  #define BLL_set_declare_NodeReference 1
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
  #define BLL_set_type_node uint32_t
#endif
#ifndef BLL_set_NodeSizeType
  #define BLL_set_NodeSizeType uint32_t
#endif

#if BLL_set_Link == 0
  #if BLL_set_SafeNext != 0
    #error SafeNext is not possible when there is not linking.
  #endif
  #if BLL_set_IsNodeUnlinked != 0
    #error (IsNodeUnlinked != 0) is not available with (Link == 0) yet.
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

#if BLL_set_StoreFormat == 0
  #ifndef BLL_set_StoreFormat0_alloc_open
    #define BLL_set_StoreFormat0_alloc_open malloc
  #endif
  #ifndef BLL_set_StoreFormat0_alloc_resize
    #define BLL_set_StoreFormat0_alloc_resize realloc
  #endif
  #ifndef BLL_set_StoreFormat0_alloc_close
    #define BLL_set_StoreFormat0_alloc_close free
  #endif
#elif BLL_set_StoreFormat == 1
  #ifndef BLL_set_StoreFormat1_alloc_open
    #define BLL_set_StoreFormat1_alloc_open malloc
  #endif
  #ifndef BLL_set_StoreFormat1_alloc_close
    #define BLL_set_StoreFormat1_alloc_close free
  #endif
  #ifndef BLL_set_StoreFormat1_ElementPerBlock
    #define BLL_set_StoreFormat1_ElementPerBlock 1
  #endif
#endif

#if BLL_set_BaseLibrary == 0
  #define _BLL_INCLUDE _WITCH_PATH
#elif BLL_set_BaseLibrary == 1
  #define _BLL_INCLUDE _FAN_PATH
#endif

#define _P(p0) CONCAT3(BLL_set_prefix, _, p0)

#if BLL_set_StructFormat == 0
  #define BLL_StructBegin(n) typedef struct{
  #define BLL_StructEnd(n) }n;
#elif BLL_set_StructFormat == 1
  #define BLL_StructBegin(n) struct n{
  #define BLL_StructEnd(n) };
#endif

#if BLL_set_declare_NodeReference == 1
  #include "internal/NodeReference.h"
#endif
#if BLL_set_declare_rest == 1
  /* _BLL_POFTWBIT; prefix of function that would be inside type */
  /* _BLL_SOFTWBIT; settings of function that would be inside type */
  /* _BLL_PBLLTFF; pass bll type for functon */
  /* _BLL_PBLLTFFC; _BLL_PBLLTFF but with comma */
  /* _BLL_DBLLTFF; declare bll type for functon */
  /* _BLL_DBLLTFFC; _BLL_DBLLTFF but with comma */
  /* _BLL_OCIBLLTFFE; only comma if bll type for function exists */
  #if BLL_set_Language == 0
    #define _BLL_POFTWBIT(p0) _P(p0)
    #define _BLL_SOFTWBIT static
    #define _BLL_PBLLTFF _pList
    #define _BLL_PBLLTFFC _BLL_PBLLTFF,
    #define _BLL_DBLLTFF _P(t) *_BLL_PBLLTFF
    #define _BLL_DBLLTFFC _BLL_DBLLTFF,
    #define _BLL_GetList _BLL_PBLLTFF
    #define _BLL_OCIBLLTFFE ,
  #elif BLL_set_Language == 1
    #define _BLL_POFTWBIT(p0) p0
    #define _BLL_SOFTWBIT
    #define _BLL_PBLLTFF
    #define _BLL_PBLLTFFC
    #define _BLL_DBLLTFF
    #define _BLL_DBLLTFFC
    #define _BLL_GetList this
    #define _BLL_OCIBLLTFFE
  #endif

  #include "internal/rest.h"

  #undef _BLL_POFTWBIT
  #undef _BLL_SOFTWBIT
  #undef _BLL_PBLLTFF
  #undef _BLL_PBLLTFFC
  #undef _BLL_DBLLTFF
  #undef _BLL_DBLLTFFC
  #undef _BLL_GetList
  #undef _BLL_OCIBLLTFFE
#endif

#undef BLL_StructBegin
#undef BLL_StructEnd

#undef _P

#undef _BLL_INCLUDE

#ifdef BLL_set_NodeReference_Overload_Declare
  #undef BLL_set_NodeReference_Overload_Declare
#endif
#ifdef BLL_set_Overload_Declare
  #undef BLL_set_Overload_Declare
#endif

#ifdef BLL_set_StoreFormat1_ElementPerBlock
  #undef BLL_set_StoreFormat1_ElementPerBlock
#endif
#ifdef BLL_set_CPP_ConstructDestruct
  #undef BLL_set_CPP_ConstructDestruct
#endif
#ifdef BLL_set_CPP_Node_ConstructDestruct
  #undef BLL_set_CPP_Node_ConstructDestruct
#endif
#ifdef BLL_set_node_data
  #undef BLL_set_node_data
#endif
#ifdef BLL_set_MultipleType_LinkIndex
  #undef BLL_set_MultipleType_LinkIndex
#endif
#ifdef BLL_set_MultipleType_Sizes
  #undef BLL_set_MultipleType_Sizes
#endif
#undef BLL_set_debug_InvalidAction_dstAccess
#undef BLL_set_debug_InvalidAction_srcAccess
#undef BLL_set_NodeSizeType
#undef BLL_set_type_node
#undef BLL_set_StoreFormat
#undef BLL_set_Link
#undef BLL_set_UseUninitialisedValues
#undef BLL_set_ResizeListAfterClear
#undef BLL_set_SafeNext
#undef BLL_set_IsNodeUnlinked
#undef BLL_set_debug_InvalidAction
#undef BLL_set_PreferNextFirst
#undef BLL_set_declare_rest
#undef BLL_set_declare_NodeReference
#undef BLL_set_StructFormat
#undef BLL_set_AreWeInsideStruct
#undef BLL_set_prefix
#ifdef BLL_set_namespace
  #undef BLL_set_namespace
#endif
#undef BLL_set_Language
#undef BLL_set_BaseLibrary
#undef BLL_set_Mark
