#ifndef ETC_BCOL_set_prefix
  #error ifndef ETC_BCOL_set_prefix
#endif
#ifndef ETC_BCOL_set_PreferredFloatSize
  #define ETC_BCOL_set_PreferredFloatSize 32
#endif
#ifndef ETC_BCOL_set_StoreExtraDataInsideObject
  #define ETC_BCOL_set_StoreExtraDataInsideObject 0
#endif
#ifndef ETC_BCOL_set_SupportGrid
  #define ETC_BCOL_set_SupportGrid 0
#endif
#ifndef ETC_BCOL_set_DynamicDeltaFunction
  #define ETC_BCOL_set_DynamicDeltaFunction
#endif
#ifndef ETC_BCOL_set_DynamicToDynamic
  #define ETC_BCOL_set_DynamicToDynamic 1
#endif
#ifndef ETC_BCOL_set_StepNumber
  #define ETC_BCOL_set_StepNumber 0
#endif

#if ETC_BCOL_set_StoreExtraDataInsideObject == 0
#elif ETC_BCOL_set_StoreExtraDataInsideObject == 1
  #ifndef ETC_BCOL_set_ExtraDataInsideObject
    #error ?
  #endif
#else
  #error ?
#endif

#if ETC_BCOL_set_SupportGrid == 0
#elif ETC_BCOL_set_SupportGrid == 1
#else
  #error ?
#endif

#include _WITCH_PATH(ETC/BCOL/internal/rest.h)

#ifndef ETC_BCOL_set_PostSolve_Grid_CollisionNormal
  #undef ETC_BCOL_set_PostSolve_Grid_CollisionNormal
#endif
#ifndef ETC_BCOL_set_PostSolve_Grid
  #undef ETC_BCOL_set_PostSolve_Grid
#endif
#ifndef ETC_BCOL_set_ConstantBumpFriction
  #undef ETC_BCOL_set_ConstantBumpFriction
#endif
#ifndef ETC_BCOL_set_ConstantFriction
  #undef ETC_BCOL_set_ConstantFriction
#endif
#undef ETC_BCOL_set_StepNumber
#undef ETC_BCOL_set_DynamicToDynamic
#undef ETC_BCOL_set_DynamicDeltaFunction
#undef ETC_BCOL_set_SupportGrid
#if ETC_BCOL_set_StoreExtraDataInsideObject == 0
#elif ETC_BCOL_set_StoreExtraDataInsideObject == 1
  #undef ETC_BCOL_set_ExtraDataInsideObject
#endif
#undef ETC_BCOL_set_PreferredFloatSize
#undef ETC_BCOL_set_prefix
