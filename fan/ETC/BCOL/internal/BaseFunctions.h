struct OpenProperties_t{
  uint32_t filler;
  #if BCOL_set_SupportGrid == 1
    _f GridBlockSize;
    PreSolve_Grid_cb_t PreSolve_Grid_cb;
  #endif
  #ifdef BCOL_set_PostSolve_Grid
    PostSolve_Grid_cb_t PostSolve_Grid_cb;
  #endif

  #if BCOL_set_DynamicToDynamic == 1
    PreSolve_Shape_cb_t PreSolve_Shape_cb;
  #endif
};

void Open(OpenProperties_t *Properties){
  this->ObjectList.Open();
  this->ShapeList_Circle.Open();
  this->ShapeList_Rectangle.Open();

  #if BCOL_set_SupportGrid == 1
    this->GridBlockSize = Properties->GridBlockSize;
    this->PreSolve_Grid_cb = Properties->PreSolve_Grid_cb;
  #endif
  #ifdef BCOL_set_PostSolve_Grid
    this->PostSolve_Grid_cb = Properties->PostSolve_Grid_cb;
  #endif

  #if BCOL_set_DynamicToDynamic == 1
    this->PreSolve_Shape_cb = Properties->PreSolve_Shape_cb;
  #endif
  #if BCOL_set_StepNumber == 1
    this->StepNumber = 0;
  #endif
}

void Close(){
  this->ObjectList.Close();
  this->ShapeList_Circle.Close();
  this->ShapeList_Rectangle.Close();
}

#if BCOL_set_StepNumber == 1
  uint64_t GetStepNumber(){
    return this->StepNumber;
  }
#endif
