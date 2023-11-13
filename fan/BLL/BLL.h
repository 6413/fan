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
#ifdef BLL_set_node_data
  #error outdated setting. use BLL_set_NodeData
#endif
#ifdef BLL_set_IsNodeUnlinked
  #error outdated setting. now there is only BLL_set_IsNodeRecycled
#endif
#if defined(BLL_set_StoreFormat0_alloc_open) || defined(BLL_set_StoreFormat1_alloc_open)
  #error outdated setting. now it's BLL_set_alloc_open
#endif
#if defined(BLL_set_StoreFormat0_alloc_resize)
  #error outdated setting. now it's BLL_set_alloc_resize
#endif
#if defined(BLL_set_StoreFormat0_alloc_close) || defined(BLL_set_StoreFormat1_alloc_close)
  #error outdated setting. now it's BLL_set_alloc_close
#endif
#ifdef BLL_set_BaseLibrary
  #error BLL_set_BaseLibrary is no longer supported. look at down line of this error to see the baselibrary config.
  /*
  #if BLL_set_BaseLibrary == 0
    #define BLL_set_Language 0
  #elif BLL_set_BaseLibrary == 1
    #ifndef BLL_set_Language
      #define BLL_set_Language 1
    #endif
    #ifndef BLL_set_CPP_ConstructDestruct
      #define BLL_set_CPP_ConstructDestruct
    #endif
    #ifndef BLL_set_CPP_Node_ConstructDestruct
      #define BLL_set_CPP_Node_ConstructDestruct
    #endif
    #ifndef BLL_set_CPP_nrsic
      #define BLL_set_CPP_nrsic 1
    #endif
  #else
    #error ?
  #endif
  */
#endif

/* --- outdated --- */

/* can be used for debug internal code inside bll with checking preprocessor */
#ifndef BLL_set_Mark
  #define BLL_set_Mark 0
#endif

#ifndef BLL_set_Language
  #error BLL_set_Language needs to be set. 0 is C and 1 is C++
#endif

#ifndef BLL_set_CPP_nrsic
  #define BLL_set_CPP_nrsic 0
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
#ifndef BLL_set_SafeNext
  #define BLL_set_SafeNext 0
#endif
#ifndef BLL_set_ResizeListAfterClear
  #define BLL_set_ResizeListAfterClear 0
#endif
#ifndef BLL_set_UseUninitialisedValues
  #if defined(WITCH_set_UseUninitialisedValues)
    #define BLL_set_UseUninitialisedValues WITCH_set_UseUninitialisedValues
  #elif defined(fan_use_uninitialized)
    #define BLL_set_UseUninitialisedValues fan_use_uninitialized
  #else
    #error BLL_set_UseUninitialisedValues needs to be set.
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

#ifndef BLL_set_IsNodeRecycled
  #if (BLL_set_debug_InvalidAction == 1 || defined(BLL_set_CPP_Node_ConstructDestruct)) && BLL_set_Link == 1
    #define BLL_set_IsNodeRecycled 1
  #else
    #define BLL_set_IsNodeRecycled 0
  #endif
#endif

#if BLL_set_Link == 0
  #if BLL_set_SafeNext != 0
    #error SafeNext is not possible when there is not linking.
  #endif
  #if BLL_set_IsNodeRecycled == 1
    #error BLL_set_IsNodeRecycled requires BLL_set_Link 1
  #endif
#endif

#if BLL_set_debug_InvalidAction == 1
  #if BLL_set_IsNodeRecycled == 0
    #error BLL_set_IsNodeRecycled cant be 0 when BLL_set_debug_InvalidAction is 1
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

#ifndef BLL_set_alloc_open
  #define BLL_set_alloc_open malloc
#endif
#ifndef BLL_set_alloc_resize
  #define BLL_set_alloc_resize realloc
#endif
#ifndef BLL_set_alloc_close
  #define BLL_set_alloc_close free
#endif
#if BLL_set_StoreFormat == 0
#elif BLL_set_StoreFormat == 1
  #if defined(BLL_set_CPP_CopyAtPointerChange)
    #error StoreFormat 1 doesn't change pointers.
  #endif
  #ifndef BLL_set_StoreFormat1_ElementPerBlock
    #define BLL_set_StoreFormat1_ElementPerBlock 1
  #endif
#endif

#include "internal/PrepareAndInclude.h"

#ifdef BLL_set_NodeReference_Overload_Declare
  #undef BLL_set_NodeReference_Overload_Declare
#endif
#ifdef BLL_set_Overload_Declare
  #undef BLL_set_Overload_Declare
#endif

#ifdef BLL_set_StoreFormat1_ElementPerBlock
  #undef BLL_set_StoreFormat1_ElementPerBlock
#endif

#ifdef BLL_set_CPP_nrsic
  #undef BLL_set_CPP_nrsic
#endif
#ifdef BLL_set_CPP_CopyAtPointerChange
  #undef BLL_set_CPP_CopyAtPointerChange
#endif
#ifdef BLL_set_CPP_ConstructDestruct
  #undef BLL_set_CPP_ConstructDestruct
#endif
#ifdef BLL_set_CPP_Node_ConstructDestruct
  #undef BLL_set_CPP_Node_ConstructDestruct
#endif

#ifdef BLL_set_NodeData
  #undef BLL_set_NodeData
#endif
#ifdef BLL_set_NodeDataType
  #undef BLL_set_NodeDataType
#endif

#ifdef BLL_set_MultipleType_LinkIndex
  #undef BLL_set_MultipleType_LinkIndex
#endif
#ifdef BLL_set_MultipleType_Sizes
  #undef BLL_set_MultipleType_Sizes
#endif

#undef BLL_set_alloc_close
#undef BLL_set_alloc_resize
#undef BLL_set_alloc_open

#undef BLL_set_debug_InvalidAction_dstAccess
#undef BLL_set_debug_InvalidAction_srcAccess
#undef BLL_set_NodeSizeType
#undef BLL_set_type_node
#undef BLL_set_StoreFormat
#undef BLL_set_Link
#undef BLL_set_UseUninitialisedValues
#undef BLL_set_ResizeListAfterClear
#undef BLL_set_SafeNext
#undef BLL_set_IsNodeRecycled
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
#undef BLL_set_Mark
