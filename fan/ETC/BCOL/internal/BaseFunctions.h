struct OpenProperties_t{
  #if BCOL_set_SupportGrid == 1
    _f GridBlockSize;
  #endif
};

void Open(const OpenProperties_t &p){
  this->ObjectList.Open();
  this->ShapeList_Circle.Open();
  this->ShapeList_Rectangle.Open();

  #if BCOL_set_SupportGrid == 1
    this->GridBlockSize = p.GridBlockSize;
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
