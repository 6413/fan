void
__ETC_BCOL_P(close)
(
  __ETC_BCOL_P(t) *bcol
){
  bcol->ObjectList.Close();
  bcol->ShapeList_Circle.Close();
  bcol->ShapeList_Square.Close();
  bcol->ShapeList_Rectangle.Close();
}

typedef struct{
  uint32_t filler;
  #if ETC_BCOL_set_SupportGrid == 1
    __pfloat GridBlockSize;
    __ETC_BCOL_P(PreSolve_Grid_cb_t) PreSolve_Grid_cb;
  #endif
  #ifdef ETC_BCOL_set_PostSolve_Grid
    __ETC_BCOL_P(PostSolve_Grid_cb_t) PostSolve_Grid_cb;
  #endif

  #if ETC_BCOL_set_DynamicToDynamic == 1
    __ETC_BCOL_P(PreSolve_Shape_cb_t) PreSolve_Shape_cb;
  #endif
}__ETC_BCOL_P(OpenProperties_t);

void
__ETC_BCOL_P(open)
(
  __ETC_BCOL_P(t) *bcol,
  __ETC_BCOL_P(OpenProperties_t) *Properties
){
  bcol->ObjectList.Open();
  bcol->ShapeList_Circle.Open();
  bcol->ShapeList_Square.Open();
  bcol->ShapeList_Rectangle.Open();

  #if ETC_BCOL_set_SupportGrid == 1
    bcol->GridBlockSize = Properties->GridBlockSize;
    bcol->PreSolve_Grid_cb = Properties->PreSolve_Grid_cb;
  #endif
  #ifdef ETC_BCOL_set_PostSolve_Grid
    bcol->PostSolve_Grid_cb = Properties->PostSolve_Grid_cb;
  #endif

  #if ETC_BCOL_set_DynamicToDynamic == 1
    bcol->PreSolve_Shape_cb = Properties->PreSolve_Shape_cb;
  #endif
  #if ETC_BCOL_set_StepNumber == 1
    bcol->StepNumber = 0;
  #endif
}

#if ETC_BCOL_set_StepNumber == 1
  uint64_t
  __ETC_BCOL_P(GetStepNumber)
  (
    __ETC_BCOL_P(t) *bcol
  ){
    return bcol->StepNumber;
  }
#endif
