__ETC_BCOL_PP(ShapeData_Rectangle_t) *RectangleData = __ETC_BCOL_PP(ShapeData_Rectangle_Get)(bcol, ShapeData->ShapeID);

__pfloat NewPositionY = NewObjectPositionY + RectangleData->PositionY;
__pfloat NewPositionX = NewObjectPositionX + RectangleData->PositionX;

__pfloat WantedPositionY = 0;
__pfloat WantedPositionX = 0;
__pfloat WantedDirectionY = 0;
__pfloat WantedDirectionX = 0;
__pfloat WantedCollisionRequesters = 0;

#if ETC_BCOL_set_SupportGrid == 1
  const __pfloat GridBlockSize = bcol->GridBlockSize;
  const __pfloat GridBlockSize_D2 = GridBlockSize / 2;

  sint32_t RectangleMiddleGridY = __floorf(NewPositionY / GridBlockSize);

  {
    __pfloat RectangleLeftX = NewPositionX - RectangleData->SizeX;
    __pfloat RectangleRightX = NewPositionX + RectangleData->SizeX;
    sint32_t RectangleLeftGridX = __floorf(RectangleLeftX / GridBlockSize);
    sint32_t RectangleRightGridX = __floorf(RectangleRightX / GridBlockSize);
    for(sint32_t RectangleGridX = RectangleLeftGridX; RectangleGridX <= RectangleRightGridX; RectangleGridX++){
      __ETC_BCOL_P(Contact_Grid_t) Contact;
      Contact.Flag = 0;
      bcol->PreSolve_Grid_cb(
        bcol,
        ObjectID,
        __ETC_BCOL_P(ShapeEnum_Rectangle),
        ShapeData->ShapeID,
        RectangleMiddleGridY,
        RectangleGridX,
        &Contact);
      if(Contact.Flag & __ETC_BCOL_PP(Contact_Grid_Flag_EnableContact_e)); else{
        continue;
      };
      f32_t oRectangleY;
      f32_t oRectangleX;
      f32_t oDirectionY;
      f32_t oDirectionX;
      __ETC_BCOL_PP(CPC_Rectangle_Square)(
        NewPositionY,
        NewPositionX,
        RectangleData->SizeY,
        RectangleData->SizeX,
        RectangleMiddleGridY * GridBlockSize + GridBlockSize_D2,
        RectangleGridX * GridBlockSize + GridBlockSize_D2,
        GridBlockSize_D2,
        &oRectangleY,
        &oRectangleX,
        &oDirectionY,
        &oDirectionX);
      WantedPositionY += oRectangleY;
      WantedPositionX += oRectangleX;
      WantedDirectionY += oDirectionY;
      WantedDirectionX += oDirectionX;
      WantedCollisionRequesters++;
    }
  }

  {
    __pfloat RectangleTopY = NewPositionY - RectangleData->SizeY;
    sint32_t RectangleTopGridY = __floorf(RectangleTopY / GridBlockSize);
    for(sint32_t RectangleGridY = RectangleMiddleGridY; RectangleGridY > RectangleTopGridY;){
      __pfloat RectangleLeftX = NewPositionX - RectangleData->SizeX;
      __pfloat RectangleRightX = NewPositionX + RectangleData->SizeX;
      sint32_t RectangleLeftGridX = __floorf(RectangleLeftX / GridBlockSize);
      sint32_t RectangleRightGridX = __floorf(RectangleRightX / GridBlockSize);
      RectangleGridY--;
      for(sint32_t RectangleGridX = RectangleLeftGridX; RectangleGridX <= RectangleRightGridX; RectangleGridX++){
        __ETC_BCOL_P(Contact_Grid_t) Contact;
        bcol->PreSolve_Grid_cb(
          bcol,
          ObjectID,
          __ETC_BCOL_P(ShapeEnum_Rectangle),
          ShapeData->ShapeID,
          RectangleGridY,
          RectangleGridX,
          &Contact);
        if(Contact.Flag & __ETC_BCOL_PP(Contact_Grid_Flag_EnableContact_e)); else{
          continue;
        };
        f32_t oRectangleY;
        f32_t oRectangleX;
        f32_t oDirectionY;
        f32_t oDirectionX;
        __ETC_BCOL_PP(CPC_Rectangle_Square)(
          NewPositionY,
          NewPositionX,
          RectangleData->SizeY,
          RectangleData->SizeX,
          RectangleGridY * GridBlockSize + GridBlockSize_D2,
          RectangleGridX * GridBlockSize + GridBlockSize_D2,
          GridBlockSize_D2,
          &oRectangleY,
          &oRectangleX,
          &oDirectionY,
          &oDirectionX);
        WantedPositionY += oRectangleY;
        WantedPositionX += oRectangleX;
        WantedDirectionY += oDirectionY;
        WantedDirectionX += oDirectionX;
        WantedCollisionRequesters++;
      }
    }
  }

  {
    __pfloat RectangleBottomY = NewPositionY + RectangleData->SizeY;
    sint32_t RectangleBottomGridY = __floorf(RectangleBottomY / GridBlockSize);
    for(sint32_t RectangleGridY = RectangleMiddleGridY; RectangleGridY < RectangleBottomGridY;){
      RectangleGridY++;
      __pfloat RectangleLeftX = NewPositionX - RectangleData->SizeX;
      __pfloat RectangleRightX = NewPositionX + RectangleData->SizeX;
      sint32_t RectangleLeftGridX = __floorf(RectangleLeftX / GridBlockSize);
      sint32_t RectangleRightGridX = __floorf(RectangleRightX / GridBlockSize);
      for(sint32_t RectangleGridX = RectangleLeftGridX; RectangleGridX <= RectangleRightGridX; RectangleGridX++){
        __ETC_BCOL_P(Contact_Grid_t) Contact;
        bcol->PreSolve_Grid_cb(
          bcol,
          ObjectID,
          __ETC_BCOL_P(ShapeEnum_Rectangle),
          ShapeData->ShapeID,
          RectangleGridY,
          RectangleGridX,
          &Contact);
        if(Contact.Flag & __ETC_BCOL_PP(Contact_Grid_Flag_EnableContact_e)); else{
          continue;
        };
        f32_t oRectangleY;
        f32_t oRectangleX;
        f32_t oDirectionY;
        f32_t oDirectionX;
        __ETC_BCOL_PP(CPC_Rectangle_Square)(
          NewPositionY,
          NewPositionX,
          RectangleData->SizeY,
          RectangleData->SizeX,
          RectangleGridY * GridBlockSize + GridBlockSize_D2,
          RectangleGridX * GridBlockSize + GridBlockSize_D2,
          GridBlockSize_D2,
          &oRectangleY,
          &oRectangleX,
          &oDirectionY,
          &oDirectionX);
        WantedPositionY += oRectangleY;
        WantedPositionX += oRectangleX;
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

          __ETC_BCOL_PP(CPCU_Rectangle_Circle_t) CData;
          __ETC_BCOL_PP(CPCU_Rectangle_Circle_Pre)(
            NewPositionY,
            NewPositionX,
            RectangleData->SizeY,
            RectangleData->SizeX,
            WorldPositionY,
            WorldPositionX,
            CircleData_->Size,
            &CData);

          if(!__ETC_BCOL_PP(CPCU_Rectangle_Circle_IsThereCollision)(&CData)){
            break;
          }

          __ETC_BCOL_P(Contact_Shape_t) Contact;
          bcol->PreSolve_Shape_cb(
            bcol,
            ObjectID,
            __ETC_BCOL_P(ShapeEnum_Rectangle),
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
          __ETC_BCOL_PP(CPCU_Rectangle_Circle_Solve)(
            NewPositionY,
            NewPositionX,
            RectangleData->SizeY,
            RectangleData->SizeX,
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
  WantedObjectPositionY += WantedPositionY - WantedCollisionRequesters * RectangleData->PositionY;
  WantedObjectPositionX += WantedPositionX - WantedCollisionRequesters * RectangleData->PositionX;
  WantedObjectDirectionY += WantedDirectionY;
  WantedObjectDirectionX += WantedDirectionX;
  WantedObjectCollisionRequesters += WantedCollisionRequesters;
}

break;
