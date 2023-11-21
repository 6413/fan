#ifndef BCOL_set_prefix
  #error ifndef BCOL_set_prefix
#endif
#ifndef BCOL_set_Dimension
  #error BCOL_set_Dimension needs to be defined.
#endif
#ifndef BCOL_set_PreferredFloatSize
  #define BCOL_set_PreferredFloatSize 32
#endif
#ifndef BCOL_set_StoreExtraDataInsideObject
  #define BCOL_set_StoreExtraDataInsideObject 0
#endif
#ifndef BCOL_set_HaveDefaultCB
  #define BCOL_set_HaveDefaultCB 1
#endif
#ifndef BCOL_set_SupportGrid
  #define BCOL_set_SupportGrid 0
#endif
#ifndef BCOL_set_DynamicDeltaFunction
  #define BCOL_set_DynamicDeltaFunction
#endif
#ifndef BCOL_set_DynamicToDynamic
  #define BCOL_set_DynamicToDynamic 1
#endif
#ifndef BCOL_set_StepNumber
  #define BCOL_set_StepNumber 0
#endif
#ifndef BCOL_set_VisualSolve
  #define BCOL_set_VisualSolve 0
#endif

#if BCOL_set_VisualSolve != 0
  #ifndef BCOL_set_VisualSolve_dmin
    #define BCOL_set_VisualSolve_dmin 0.1
  #endif
  #ifndef BCOL_set_VisualSolve_dmax
    #define BCOL_set_VisualSolve_dmax 99999999
  #endif
  #if BCOL_set_SupportGrid != 0
    #ifndef BCOL_set_VisualSolve_GridContact
      #error define BCOL_set_VisualSolve_GridContact
    #endif
  #endif
#endif

#if defined(BCOL_set_IncludePath)
  #define BCOL_Include(p) <BCOL_set_IncludePath/p>
#else
  #error BCOL_set_IncludePath needs to be defined.
#endif

#if BCOL_set_StoreExtraDataInsideObject == 0
#elif BCOL_set_StoreExtraDataInsideObject == 1
  #ifndef BCOL_set_ExtraDataInsideObject
    #error ?
  #endif
#else
  #error ?
#endif

#if BCOL_set_SupportGrid == 0
#elif BCOL_set_SupportGrid == 1
#else
  #error ?
#endif

#include "internal/rest.h"

#ifndef BCOL_set_PostSolve_Grid_CollisionNormal
  #undef BCOL_set_PostSolve_Grid_CollisionNormal
#endif
#ifndef BCOL_set_PostSolve_Grid
  #undef BCOL_set_PostSolve_Grid
#endif
#ifndef BCOL_set_ConstantBumpFriction
  #undef BCOL_set_ConstantBumpFriction
#endif
#ifndef BCOL_set_ConstantFriction
  #undef BCOL_set_ConstantFriction
#endif
#undef BCOL_Include
#if BCOL_set_VisualSolve != 0
  #if BCOL_set_SupportGrid != 0
    #undef BCOL_set_VisualSolve_GridContact
  #endif
  #undef BCOL_set_VisualSolve_dmax
  #undef BCOL_set_VisualSolve_dmin
#endif
#undef BCOL_set_VisualSolve
#undef BCOL_set_StepNumber
#undef BCOL_set_DynamicToDynamic
#undef BCOL_set_DynamicDeltaFunction
#undef BCOL_set_SupportGrid
#if BCOL_set_StoreExtraDataInsideObject == 0
#elif BCOL_set_StoreExtraDataInsideObject == 1
  #undef BCOL_set_ExtraDataInsideObject
#endif
#undef BCOL_set_HaveDefaultCB
#undef BCOL_set_StoreExtraDataInsideObject
#undef BCOL_set_PreferredFloatSize
#ifdef BCOL_set_IncludePath
  #undef BCOL_set_IncludePath
#endif
#undef BCOL_set_Dimension
#undef BCOL_set_prefix
