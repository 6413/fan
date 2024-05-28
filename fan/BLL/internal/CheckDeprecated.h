#ifdef BLL_set_KeepSettings
  #error deprecated setting. dont use it
#endif
#ifdef BLL_set_declare_basic_types
  #error deprecated setting. now it shipped with BLL_set_declare_rest
#endif
#ifdef BLL_set_ConstantInvalidNodeReference_Listless
  #error deprecated setting.
#endif
#ifdef BLL_set_node_data
  #error deprecated setting BLL_set_node_data. use BLL_set_NodeData
#endif
#ifdef BLL_set_IsNodeUnlinked
  #error deprecated setting. now there is only BLL_set_IsNodeRecycled
#endif
#if defined(BLL_set_StoreFormat0_alloc_open) || defined(BLL_set_StoreFormat1_alloc_open)
  #error deprecated setting. now it's BLL_set_alloc_open
#endif
#if defined(BLL_set_StoreFormat0_alloc_resize)
  #error deprecated setting. now it's BLL_set_alloc_resize
#endif
#if defined(BLL_set_StoreFormat0_alloc_close) || defined(BLL_set_StoreFormat1_alloc_close)
  #error deprecated setting. now it's BLL_set_alloc_close
#endif
#ifdef BLL_set_BaseLibrary
  #error BLL_set_BaseLibrary is no longer supported.
  #error look at down line of this error to see the baselibrary config.
  /*
  #if BLL_set_BaseLibrary == 0
    #define BLL_set_Language 0
  #elif BLL_set_BaseLibrary == 1
    #ifndef BLL_set_Language
      #define BLL_set_Language 1
    #endif
    #ifndef BLL_set_CPP_ConstructDestruct
      #define BLL_set_CPP_ConstructDestruct 1
    #endif
    #ifndef BLL_set_CPP_Node_ConstructDestruct
      #define BLL_set_CPP_Node_ConstructDestruct 1
    #endif
    #ifndef BLL_set_CPP_nrsic
      #define BLL_set_CPP_nrsic 1
    #endif
  #else
    #error ?
  #endif
  */
#endif
#ifdef BLL_set_UseUninitialisedValues
  #error deprecated setting BLL_set_UseUninitialisedValues. __sanit used instead.
#endif
#ifdef BLL_set_namespace
  #error deprecated setting BLL_set_namespace. just put include bll inside namespace.
#endif
#if defined(BLL_set_debug_InvalidAction) || \
  defined(BLL_set_debug_InvalidAction_srcAccess) || \
  defined(BLL_set_debug_InvalidAction_dstAccess)

  #error debug stuff will be implemented in future
#endif
#ifdef BLL_set_NoSentinel
  #error deprecated setting BLL_set_NoSentinel. now its BLL_set_LinkSentinel 1 or 0.
#endif
