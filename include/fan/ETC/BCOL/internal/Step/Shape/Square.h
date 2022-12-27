__ETC_BCOL_PP(ShapeData_Square_t) *SquareData = __ETC_BCOL_PP(ShapeData_Square_Get)(bcol, ShapeData->ShapeID);

__pfloat NewPositionY = NewObjectPositionY + SquareData->PositionY;
__pfloat NewPositionX = NewObjectPositionX + SquareData->PositionX;

__pfloat WantedPositionY = 0;
__pfloat WantedPositionX = 0;
__pfloat WantedDirectionY = 0;
__pfloat WantedDirectionX = 0;
__pfloat WantedCollisionRequesters = 0;

#if ETC_BCOL_set_SupportGrid == 1
  const __pfloat GridBlockSize = bcol->GridBlockSize;
  const __pfloat GridBlockSize_D2 = GridBlockSize / 2;

  sint32_t SquareMiddleGridY = __floorf(NewPositionY / GridBlockSize);

  {
    __pfloat SquareLeftX = NewPositionX - SquareData->Size;
    __pfloat SquareRightX = NewPositionX + SquareData->Size;
    sint32_t SquareLeftGridX = __floorf(SquareLeftX / GridBlockSize);
    sint32_t SquareRightGridX = __floorf(SquareRightX / GridBlockSize);
    for(sint32_t SquareGridX = SquareLeftGridX; SquareGridX <= SquareRightGridX; SquareGridX++){
      __ETC_BCOL_P(Contact_Grid_t) Contact;
      bcol->PreSolve_Grid_cb(
        bcol,
        ObjectID,
        __ETC_BCOL_P(ShapeEnum_Square),
        ShapeData->ShapeID,
        SquareMiddleGridY,
        SquareGridX,
        &Contact);
      if(Contact.Flag & __ETC_BCOL_PP(Contact_Grid_Flag_EnableContact_e)); else{
        continue;
      };
      f32_t oSquareY;
      f32_t oSquareX;
      f32_t oDirectionY;
      f32_t oDirectionX;
      __ETC_BCOL_PP(CPC_Square_Square)(
        NewPositionY,
        NewPositionX,
        SquareData->Size,
        SquareMiddleGridY * GridBlockSize + GridBlockSize_D2,
        SquareGridX * GridBlockSize + GridBlockSize_D2,
        GridBlockSize_D2,
        &oSquareY,
        &oSquareX,
        &oDirectionY,
        &oDirectionX);
      WantedPositionY += oSquareY;
      WantedPositionX += oSquareX;
      WantedDirectionY += oDirectionY;
      WantedDirectionX += oDirectionX;
      WantedCollisionRequesters++;
    }
  }

  {
    __pfloat SquareTopY = NewPositionY - SquareData->Size;
    sint32_t SquareTopGridY = __floorf(SquareTopY / GridBlockSize);
    for(sint32_t SquareGridY = SquareMiddleGridY; SquareGridY > SquareTopGridY;){
      __pfloat SquareLeftX = NewPositionX - SquareData->Size;
      __pfloat SquareRightX = NewPositionX + SquareData->Size;
      sint32_t SquareLeftGridX = __floorf(SquareLeftX / GridBlockSize);
      sint32_t SquareRightGridX = __floorf(SquareRightX / GridBlockSize);
      SquareGridY--;
      for(sint32_t SquareGridX = SquareLeftGridX; SquareGridX <= SquareRightGridX; SquareGridX++){
        __ETC_BCOL_P(Contact_Grid_t) Contact;
        bcol->PreSolve_Grid_cb(
          bcol,
          ObjectID,
          __ETC_BCOL_P(ShapeEnum_Square),
          ShapeData->ShapeID,
          SquareGridY,
          SquareGridX,
          &Contact);
        if(Contact.Flag & __ETC_BCOL_PP(Contact_Grid_Flag_EnableContact_e)); else{
          continue;
        };
        f32_t oSquareY;
        f32_t oSquareX;
        f32_t oDirectionY;
        f32_t oDirectionX;
        __ETC_BCOL_PP(CPC_Square_Square)(
          NewPositionY,
          NewPositionX,
          SquareData->Size,
          SquareGridY * GridBlockSize + GridBlockSize_D2,
          SquareGridX * GridBlockSize + GridBlockSize_D2,
          GridBlockSize_D2,
          &oSquareY,
          &oSquareX,
          &oDirectionY,
          &oDirectionX);
        WantedPositionY += oSquareY;
        WantedPositionX += oSquareX;
        WantedDirectionY += oDirectionY;
        WantedDirectionX += oDirectionX;
        WantedCollisionRequesters++;
      }
    }
  }

  {
    __pfloat SquareBottomY = NewPositionY + SquareData->Size;
    sint32_t SquareBottomGridY = __floorf(SquareBottomY / GridBlockSize);
    for(sint32_t SquareGridY = SquareMiddleGridY; SquareGridY < SquareBottomGridY;){
      SquareGridY++;
      __pfloat SquareLeftX = NewPositionX - SquareData->Size;
      __pfloat SquareRightX = NewPositionX + SquareData->Size;
      sint32_t SquareLeftGridX = __floorf(SquareLeftX / GridBlockSize);
      sint32_t SquareRightGridX = __floorf(SquareRightX / GridBlockSize);
      for(sint32_t SquareGridX = SquareLeftGridX; SquareGridX <= SquareRightGridX; SquareGridX++){
        __ETC_BCOL_P(Contact_Grid_t) Contact;
        bcol->PreSolve_Grid_cb(
          bcol,
          ObjectID,
          __ETC_BCOL_P(ShapeEnum_Square),
          ShapeData->ShapeID,
          SquareGridY,
          SquareGridX,
          &Contact);
        if(Contact.Flag & __ETC_BCOL_PP(Contact_Grid_Flag_EnableContact_e)); else{
          continue;
        };
        f32_t oSquareY;
        f32_t oSquareX;
        f32_t oDirectionY;
        f32_t oDirectionX;
        __ETC_BCOL_PP(CPC_Square_Square)(
          NewPositionY,
          NewPositionX,
          SquareData->Size,
          SquareGridY * GridBlockSize + GridBlockSize_D2,
          SquareGridX * GridBlockSize + GridBlockSize_D2,
          GridBlockSize_D2,
          &oSquareY,
          &oSquareX,
          &oDirectionY,
          &oDirectionX);
        WantedPositionY += oSquareY;
        WantedPositionX += oSquareX;
        WantedDirectionY += oDirectionY;
        WantedDirectionX += oDirectionX;
        WantedCollisionRequesters++;
      }
    }
  }
#endif

#if ETC_BCOL_set_DynamicToDynamic == 1
  __ETC_BCOL_P(TraverseObjects_t) TraverseObjects_;
  __ETC_BCOL_P(TraverseObjects_init)(bcol, &TraverseObjects_);
  while(__ETC_BCOL_P(TraverseObjects_loop)(bcol, &TraverseObjects_)){
    __ETC_BCOL_P(ObjectID_t) ObjectID_ = TraverseObjects_.ObjectID;
    if(ObjectID_ == ObjectID){
      continue;
    }
    __ETC_BCOL_PP(ObjectData_t) *ObjectData_ = __ETC_BCOL_PP(GetObjectData)(bcol, ObjectID_);
    for(uint32_t ShapeID_ = 0; ShapeID_ < ObjectData_->ShapeList.Current; ShapeID_++){
      __ETC_BCOL_PP(ShapeData_t) *ShapeData_ = &((__ETC_BCOL_PP(ShapeData_t) *)ObjectData_->ShapeList.ptr)[ShapeID_];

      switch(ShapeData_->ShapeEnum){
        case __ETC_BCOL_P(ShapeEnum_Circle):{
          __ETC_BCOL_PP(ShapeData_Circle_t) *CircleData_ =
            __ETC_BCOL_PP(ShapeData_Circle_Get)(bcol, ShapeData_->ShapeID);

          __pfloat WorldPositionY = ObjectData_->PositionY + CircleData_->PositionY;
          __pfloat WorldPositionX = ObjectData_->PositionX + CircleData_->PositionX;

          __ETC_BCOL_PP(CPCU_Square_Circle_t) CData;
          __ETC_BCOL_PP(CPCU_Square_Circle_Pre)(
            NewPositionY,
            NewPositionX,
            SquareData->Size,
            WorldPositionY,
            WorldPositionX,
            CircleData_->Size,
            &CData);

          if(!__ETC_BCOL_PP(CPCU_Square_Circle_IsThereCollision)(&CData)){
            break;
          }

          __ETC_BCOL_P(Contact_Shape_t) Contact;
          bcol->PreSolve_Shape_cb(
            bcol,
            ObjectID,
            __ETC_BCOL_P(ShapeEnum_Square),
            ShapeData->ShapeID,
            ObjectID_,
            __ETC_BCOL_P(ShapeEnum_Circle),
            ShapeID_,
            &Contact);
          if(Contact.Flag & __ETC_BCOL_PP(Contact_Shape_Flag_EnableContact_e)); else{
            break;
          };

          f32_t oPositionY;
          f32_t oPositionX;
          f32_t oDirectionY;
          f32_t oDirectionX;
          __ETC_BCOL_PP(CPCU_Square_Circle_Solve)(
            NewPositionY,
            NewPositionX,
            SquareData->Size,
            WorldPositionY,
            WorldPositionX,
            CircleData_->Size,
            &CData,
            &oPositionY,
            &oPositionX,
            &oDirectionY,
            &oDirectionX);

          WantedPositionY += oPositionY;
          WantedPositionX += oPositionX;
          WantedDirectionY += oDirectionY;
          WantedDirectionX += oDirectionX;
          WantedCollisionRequesters++;

          break;
        }
        case __ETC_BCOL_P(ShapeEnum_Square):{
          /* TODO */
          break;
        }
        case __ETC_BCOL_P(ShapeEnum_Rectangle):{
          /* TODO */
          break;
        }
      }
    }
  }
#endif

if(WantedCollisionRequesters){
  WantedObjectPositionY += WantedPositionY - WantedCollisionRequesters * SquareData->PositionY;
  WantedObjectPositionX += WantedPositionX - WantedCollisionRequesters * SquareData->PositionX;
  WantedObjectDirectionY += WantedDirectionY;
  WantedObjectDirectionX += WantedDirectionX;
  WantedObjectCollisionRequesters += WantedCollisionRequesters;
}

break;
