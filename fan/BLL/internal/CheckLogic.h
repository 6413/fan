#if BLL_set_AreWeInsideStruct < 0 || BLL_set_AreWeInsideStruct > 1
  #error invalid BLL_set_AreWeInsideStruct
#endif
#if BLL_set_Language < 0 || BLL_set_Language > 1
  #error invalid BLL_set_Language
#endif
#if BLL_set_StructFormat < 0 || BLL_set_StructFormat > 1
  #error invalid BLL_set_StructFormat
#endif
#if BLL_set_declare_NodeReference < 0 || BLL_set_declare_NodeReference > 1
  #error invalid BLL_set_declare_NodeReference
#endif
#if BLL_set_declare_rest < 0 || BLL_set_declare_rest > 1
  #error invalid BLL_set_declare_rest
#endif
#if BLL_set_IntegerNR < 0 || BLL_set_IntegerNR > 1
  #error invalid BLL_set_IntegerNR
#endif
#if BLL_set_PreferNextFirst < 0 || BLL_set_PreferNextFirst > 1
  #error invalid BLL_set_PreferNextFirst
#endif
#if BLL_set_Recycle < 0 || BLL_set_Recycle > 1
  #error invalid BLL_set_Recycle
#endif
#if BLL_set_PadNode < 0 || BLL_set_PadNode > 1
  #error invalid BLL_set_PadNode
#endif
#if BLL_set_SafeNext < 0
  #error invalid BLL_set_SafeNext
#endif
#if BLL_set_ResizeListAfterClear < 0 || BLL_set_ResizeListAfterClear > 1
  #error invalid BLL_set_ResizeListAfterClear
#endif
#if BLL_set_Link < 0 || BLL_set_Link > 1
  #error invalid BLL_set_Link
#endif
#if BLL_set_LinkSentinel < 0 || BLL_set_LinkSentinel > 1
  #error invalid BLL_set_LinkSentinel
#endif
#if BLL_set_StoreFormat < 0 || BLL_set_StoreFormat > 1
  #error invalid BLL_set_StoreFormat
#endif
#if BLL_set_IsNodeRecycled < 0 || BLL_set_IsNodeRecycled > 1
  #error invalid BLL_set_IsNodeRecycled
#endif
#if BLL_set_StoreFormat1_ElementPerBlock < 0
  #error invalid BLL_set_StoreFormat1_ElementPerBlock
#endif
#if BLL_set_CPP_nrsic < 0 || BLL_set_CPP_nrsic > 1
  #error invalid BLL_set_CPP_nrsic
#endif
#if BLL_set_CPP_Node_ConstructDestruct < 0 || BLL_set_CPP_Node_ConstructDestruct > 1
  #error invalid BLL_set_CPP_Node_ConstructDestruct
#endif
#if BLL_set_CPP_ConstructDestruct < 0 || BLL_set_CPP_ConstructDestruct > 1
  #error invalid BLL_set_CPP_ConstructDestruct
#endif
#if BLL_set_CPP_CopyAtPointerChange < 0 || BLL_set_CPP_CopyAtPointerChange > 1
  #error invalid BLL_set_CPP_CopyAtPointerChange
#endif

/* ------------------------------------------------------------------------------------------ */

#if BLL_set_LinkSentinel && BLL_set_Link == 0
  #error BLL_set_LinkSentinel 1 is not possible with BLL_set_Link 0
#endif

#if BLL_set_CPP_CopyAtPointerChange && BLL_set_StoreFormat == 1
  #error StoreFormat 1 doesn't change pointers.
#endif

#if BLL_set_SafeNext > 0 && !BLL_set_Link
  #error SafeNext is not possible when there is no linking.
#endif
#if BLL_set_IsNodeRecycled && !BLL_set_Link
  #error BLL_set_IsNodeRecycled requires BLL_set_Link 1
#endif

#if defined(BLL_set_NodeData) && defined(BLL_set_NodeDataType)
  #error BLL_set_NodeData and BLL_set_NodeDataType cant be defined same time.
#endif