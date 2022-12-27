__ETC_BCOL_PP(ShapeData_Circle_t) *CircleData = __ETC_BCOL_PP(ShapeData_Circle_Get)(bcol, ShapeData->ShapeID);

__pfloat NewPositionY = NewObjectPositionY + CircleData->PositionY;
__pfloat NewPositionX = NewObjectPositionX + CircleData->PositionX;

__pfloat WantedPositionY = 0;
__pfloat WantedPositionX = 0;
__pfloat WantedDirectionY = 0;
__pfloat WantedDirectionX = 0;
__pfloat WantedCollisionRequesters = 0;

#if ETC_BCOL_set_SupportGrid == 1
  const __pfloat GridBlockSize = bcol->GridBlockSize;
  const __pfloat GridBlockSize_D2 = GridBlockSize / 2;

  sint32_t CircleMiddleGridY = __floorf(NewPositionY / GridBlockSize);

  {
    __pfloat CircleLeftX = NewPositionX - CircleData->Size;
    __pfloat CircleRightX = NewPositionX + CircleData->Size;
    sint32_t CircleLeftGridX = __floorf(CircleLeftX / GridBlockSize);
    sint32_t CircleRightGridX = __floorf(CircleRightX / GridBlockSize);
    for(sint32_t CircleGridX = CircleLeftGridX; CircleGridX <= CircleRightGridX; CircleGridX++){
      __ETC_BCOL_P(Contact_Grid_t) Contact;
      Contact.Flag = 0;
      bcol->PreSolve_Grid_cb(
        bcol,
        ObjectID,
        __ETC_BCOL_P(ShapeEnum_Circle),
        ShapeData->ShapeID,
        CircleMiddleGridY,
        CircleGridX,
        &Contact);
      if(Contact.Flag & __ETC_BCOL_PP(Contact_Grid_Flag_EnableContact_e)); else{
        continue;
      };

      f32_t oCircleY;
      f32_t oCircleX;
      f32_t oDirectionY;
      f32_t oDirectionX;
      __ETC_BCOL_PP(CPC_Circle_Square)(
        NewPositionY,
        NewPositionX,
        CircleData->Size,
        CircleMiddleGridY * GridBlockSize + GridBlockSize_D2,
        CircleGridX * GridBlockSize + GridBlockSize_D2,
        GridBlockSize_D2,
        &oCircleY,
        &oCircleX,
        &oDirectionY,
        &oDirectionX);

      #ifdef ETC_BCOL_set_PostSolve_Grid
        __ETC_BCOL_P(ContactResult_Grid_t) ContactResult;
      #endif
      #ifdef ETC_BCOL_set_PostSolve_Grid_CollisionNormal
        ContactResult.NormalY = oDirectionY;
        ContactResult.NormalX = oDirectionX;
      #endif
      #ifdef ETC_BCOL_set_PostSolve_Grid
        bcol->PostSolve_Grid_cb(
          bcol,
          ObjectID,
          __ETC_BCOL_P(ShapeEnum_Circle),
          ShapeData->ShapeID,
          CircleMiddleGridY,
          CircleGridX,
          &ContactResult);
      #endif

      WantedPositionY += oCircleY;
      WantedPositionX += oCircleX;
      WantedDirectionY += oDirectionY;
      WantedDirectionX += oDirectionX;
      WantedCollisionRequesters++;
    }
  }

  {
    __pfloat CircleTopY = NewPositionY - CircleData->Size;
    sint32_t CircleTopGridY = __floorf(CircleTopY / GridBlockSize);
    for(sint32_t CircleGridY = CircleMiddleGridY; CircleGridY > CircleTopGridY;){
      __pfloat CircleY = (__pfloat)CircleGridY * GridBlockSize;
      __pfloat CircleOffsetY = CircleY - NewPositionY;
      __pfloat Magic = __sqrt(__absf(CircleData->Size * CircleData->Size - CircleOffsetY * CircleOffsetY));
      __pfloat CircleLeftX = NewPositionX - Magic;
      __pfloat CircleRightX = NewPositionX + Magic;
      sint32_t CircleLeftGridX = __floorf(CircleLeftX / GridBlockSize);
      sint32_t CircleRightGridX = __floorf(CircleRightX / GridBlockSize);
      CircleGridY--;
      for(sint32_t CircleGridX = CircleLeftGridX; CircleGridX <= CircleRightGridX; CircleGridX++){
        __ETC_BCOL_P(Contact_Grid_t) Contact;
        bcol->PreSolve_Grid_cb(
          bcol,
          ObjectID,
          __ETC_BCOL_P(ShapeEnum_Circle),
          ShapeData->ShapeID,
          CircleGridY,
          CircleGridX,
          &Contact);
        if(Contact.Flag & __ETC_BCOL_PP(Contact_Grid_Flag_EnableContact_e)); else{
          continue;
        };
        f32_t oCircleY;
        f32_t oCircleX;
        f32_t oDirectionY;
        f32_t oDirectionX;
        __ETC_BCOL_PP(CPC_Circle_Square)(
          NewPositionY,
          NewPositionX,
          CircleData->Size,
          CircleGridY * GridBlockSize + GridBlockSize_D2,
          CircleGridX * GridBlockSize + GridBlockSize_D2,
          GridBlockSize_D2,
          &oCircleY,
          &oCircleX,
          &oDirectionY,
          &oDirectionX);

        #ifdef ETC_BCOL_set_PostSolve_Grid
          __ETC_BCOL_P(ContactResult_Grid_t) ContactResult;
        #endif
        #ifdef ETC_BCOL_set_PostSolve_Grid_CollisionNormal
          ContactResult.NormalY = oDirectionY;
          ContactResult.NormalX = oDirectionX;
        #endif
        #ifdef ETC_BCOL_set_PostSolve_Grid
          bcol->PostSolve_Grid_cb(
            bcol,
            ObjectID,
            __ETC_BCOL_P(ShapeEnum_Circle),
            ShapeData->ShapeID,
            CircleGridY,
            CircleGridX,
            &ContactResult);
        #endif

        WantedPositionY += oCircleY;
        WantedPositionX += oCircleX;
        WantedDirectionY += oDirectionY;
        WantedDirectionX += oDirectionX;
        WantedCollisionRequesters++;
      }
    }
  }

  {
    __pfloat CircleBottomY = NewPositionY + CircleData->Size;
    sint32_t CircleBottomGridY = __floorf(CircleBottomY / GridBlockSize);
    for(sint32_t CircleGridY = CircleMiddleGridY; CircleGridY < CircleBottomGridY;){
      CircleGridY++;
      __pfloat CircleY = (__pfloat)CircleGridY * GridBlockSize;
      __pfloat CircleOffsetY = CircleY - NewPositionY;
      __pfloat Magic = __sqrt(__absf(CircleData->Size * CircleData->Size - CircleOffsetY * CircleOffsetY));
      __pfloat CircleLeftX = NewPositionX - Magic;
      __pfloat CircleRightX = NewPositionX + Magic;
      sint32_t CircleLeftGridX = __floorf(CircleLeftX / GridBlockSize);
      sint32_t CircleRightGridX = __floorf(CircleRightX / GridBlockSize);
      for(sint32_t CircleGridX = CircleLeftGridX; CircleGridX <= CircleRightGridX; CircleGridX++){
        __ETC_BCOL_P(Contact_Grid_t) Contact;
        bcol->PreSolve_Grid_cb(
          bcol,
          ObjectID,
          __ETC_BCOL_P(ShapeEnum_Circle),
          ShapeData->ShapeID,
          CircleGridY,
          CircleGridX,
          &Contact);
        if(Contact.Flag & __ETC_BCOL_PP(Contact_Grid_Flag_EnableContact_e)); else{
          continue;
        };
        f32_t oCircleY;
        f32_t oCircleX;
        f32_t oDirectionY;
        f32_t oDirectionX;
        __ETC_BCOL_PP(CPC_Circle_Square)(
          NewPositionY,
          NewPositionX,
          CircleData->Size,
          CircleGridY * GridBlockSize + GridBlockSize_D2,
          CircleGridX * GridBlockSize + GridBlockSize_D2,
          GridBlockSize_D2,
          &oCircleY,
          &oCircleX,
          &oDirectionY,
          &oDirectionX);

        #ifdef ETC_BCOL_set_PostSolve_Grid
          __ETC_BCOL_P(ContactResult_Grid_t) ContactResult;
        #endif
        #ifdef ETC_BCOL_set_PostSolve_Grid_CollisionNormal
          ContactResult.NormalY = oDirectionY;
          ContactResult.NormalX = oDirectionX;
        #endif
        #ifdef ETC_BCOL_set_PostSolve_Grid
          bcol->PostSolve_Grid_cb(
            bcol,
            ObjectID,
            __ETC_BCOL_P(ShapeEnum_Circle),
            ShapeData->ShapeID,
            CircleGridY,
            CircleGridX,
            &ContactResult);
        #endif

        WantedPositionY += oCircleY;
        WantedPositionX += oCircleX;
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

          __pfloat DifferenceY = NewPositionY - WorldPositionY;
          __pfloat DifferenceX = NewPositionX - WorldPositionX;
          __pfloat Hypotenuse = __sqrt(DifferenceY * DifferenceY + DifferenceX * DifferenceX);
          __pfloat CombinedSize = CircleData->Size + CircleData_->Size;
          if(Hypotenuse >= CombinedSize){
            break;
          }

          __ETC_BCOL_P(Contact_Shape_t) Contact;
          bcol->PreSolve_Shape_cb(
            bcol,
            ObjectID,
            __ETC_BCOL_P(ShapeEnum_Circle),
            ShapeData->ShapeID,
            ObjectID_,
            __ETC_BCOL_P(ShapeEnum_Circle),
            ShapeID_,
            &Contact);
          if(Contact.Flag & __ETC_BCOL_PP(Contact_Shape_Flag_EnableContact_e)); else{
            break;
          };

          if(Hypotenuse != 0){
            __pfloat DirectionY = DifferenceY / Hypotenuse;
            __pfloat DirectionX = DifferenceX / Hypotenuse;
            WantedPositionY += NewPositionY + DirectionY * (CombinedSize - Hypotenuse);
            WantedPositionX += NewPositionX + DirectionX * (CombinedSize - Hypotenuse);
            WantedDirectionY += DirectionY;
            WantedDirectionX += DirectionX;
            WantedCollisionRequesters++;
          }
          break;
        }
        case __ETC_BCOL_P(ShapeEnum_Square):{
          __ETC_BCOL_PP(ShapeData_Square_t) *SquareData_ =
            __ETC_BCOL_PP(ShapeData_Square_Get)(bcol, ShapeData_->ShapeID);

          __pfloat WorldPositionY = ObjectData_->PositionY + SquareData_->PositionY;
          __pfloat WorldPositionX = ObjectData_->PositionX + SquareData_->PositionX;

          __ETC_BCOL_PP(CPCU_Circle_Square_t) CData;
          __ETC_BCOL_PP(CPCU_Circle_Square_Pre)(
            NewPositionY,
            NewPositionX,
            CircleData->Size,
            WorldPositionY,
            WorldPositionX,
            SquareData_->Size,
            &CData);

          if(!__ETC_BCOL_PP(CPCU_Circle_Square_IsThereCollision)(&CData)){
            break;
          }

          __ETC_BCOL_P(Contact_Shape_t) Contact;
          bcol->PreSolve_Shape_cb(
            bcol,
            ObjectID,
            __ETC_BCOL_P(ShapeEnum_Circle),
            ShapeData->ShapeID,
            ObjectID_,
            __ETC_BCOL_P(ShapeEnum_Square),
            ShapeID_,
            &Contact);
          if(Contact.Flag & __ETC_BCOL_PP(Contact_Shape_Flag_EnableContact_e)); else{
            break;
          };

          f32_t oPositionY;
          f32_t oPositionX;
          f32_t oDirectionY;
          f32_t oDirectionX;
          __ETC_BCOL_PP(CPCU_Circle_Square_Solve)(
            NewPositionY,
            NewPositionX,
            CircleData->Size,
            WorldPositionY,
            WorldPositionX,
            SquareData_->Size,
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
        case __ETC_BCOL_P(ShapeEnum_Rectangle):{
          __ETC_BCOL_PP(ShapeData_Rectangle_t) *RectangleData_ =
            __ETC_BCOL_PP(ShapeData_Rectangle_Get)(bcol, ShapeData_->ShapeID);

          __pfloat WorldPositionY = ObjectData_->PositionY + RectangleData_->PositionY;
          __pfloat WorldPositionX = ObjectData_->PositionX + RectangleData_->PositionX;

          __ETC_BCOL_PP(CPCU_Circle_Rectangle_t) CData;
          __ETC_BCOL_PP(CPCU_Circle_Rectangle_Pre)(
            NewPositionY,
            NewPositionX,
            CircleData->Size,
            WorldPositionY,
            WorldPositionX,
            RectangleData_->SizeY,
            RectangleData_->SizeX,
            &CData);

          if(!__ETC_BCOL_PP(CPCU_Circle_Rectangle_IsThereCollision)(&CData)){
            break;
          }

          __ETC_BCOL_P(Contact_Shape_t) Contact;
          bcol->PreSolve_Shape_cb(
            bcol,
            ObjectID,
            __ETC_BCOL_P(ShapeEnum_Circle),
            ShapeData->ShapeID,
            ObjectID_,
            __ETC_BCOL_P(ShapeEnum_Rectangle),
            ShapeID_,
            &Contact);
          if(Contact.Flag & __ETC_BCOL_PP(Contact_Shape_Flag_EnableContact_e)); else{
            break;
          };

          f32_t oPositionY;
          f32_t oPositionX;
          f32_t oDirectionY;
          f32_t oDirectionX;
          __ETC_BCOL_PP(CPCU_Circle_Rectangle_Solve)(
            NewPositionY,
            NewPositionX,
            CircleData->Size,
            WorldPositionY,
            WorldPositionX,
            RectangleData_->SizeY,
            RectangleData_->SizeX,
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
      }
    }
  }
#endif

if(WantedCollisionRequesters){
  WantedObjectPositionY += WantedPositionY - WantedCollisionRequesters * CircleData->PositionY;
  WantedObjectPositionX += WantedPositionX - WantedCollisionRequesters * CircleData->PositionX;
  WantedObjectDirectionY += WantedDirectionY;
  WantedObjectDirectionX += WantedDirectionX;
  WantedObjectCollisionRequesters += WantedCollisionRequesters;
}

break;
