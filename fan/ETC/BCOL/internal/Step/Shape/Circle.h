auto CircleData = ShapeData_Circle_Get(ObjectData0->ShapeList.ptr[sip0.ShapeID.ID].ShapeID);

_vf NewPosition = NewObjectPosition + CircleData->Position;

_vf WantedPosition = 0;
_vf WantedDirection = 0;
_f WantedCollisionRequesters = 0;

#if ETC_BCOL_set_SupportGrid == 1
  const _f GridBlockSize = this->GridBlockSize;
  const _f GridBlockSize_D2 = GridBlockSize / 2;

  sint32_t CircleMiddleGridY = fan::math::floor(NewPosition.y / GridBlockSize);

  {
    _f CircleLeftX = NewPosition.x - CircleData->Size;
    _f CircleRightX = NewPosition.x + CircleData->Size;
    sint32_t CircleLeftGridX = fan::math::floor(CircleLeftX / GridBlockSize);
    sint32_t CircleRightGridX = fan::math::floor(CircleRightX / GridBlockSize);
    for(sint32_t CircleGridX = CircleLeftGridX; CircleGridX <= CircleRightGridX; CircleGridX++){
      Contact_Grid_t Contact;
      Contact.Flag = 0;
      this->PreSolve_Grid_cb(
        this,
        &sip0,
        {CircleGridX, CircleMiddleGridY},
        &Contact);
      if(this->ObjectList.CheckSafeNext(0) != sip0.ObjectID){
        goto gt_Object0Unlinked;
      }
      if(Contact.Flag & Contact_Grid_Flag::EnableContact); else{
        continue;
      };

      _vf oCircle;
      _vf oDirection;
      CPC_Circle_Square(
        NewPosition,
        CircleData->Size,
        {CircleGridX * GridBlockSize + GridBlockSize_D2, CircleMiddleGridY * GridBlockSize + GridBlockSize_D2},
        GridBlockSize_D2,
        &oCircle,
        &oDirection);

      #ifdef ETC_BCOL_set_PostSolve_Grid
        ContactResult_Grid_t ContactResult;
      #endif
      #ifdef ETC_BCOL_set_PostSolve_Grid_CollisionNormal
        ContactResult.Normal = oDirection;
      #endif
      #ifdef ETC_BCOL_set_PostSolve_Grid
        this->PostSolve_Grid_cb(
          this,
          &sip0,
          {CircleGridX, CircleMiddleGridY},
          &ContactResult);
        if(this->ObjectList.CheckSafeNext(0) != sip0.ObjectID){
          goto gt_Object0Unlinked;
        }
      #endif

      WantedPosition += oCircle;
      WantedDirection += oDirection;
      WantedCollisionRequesters++;
    }
  }

  {
    _f CircleTopY = NewPosition.y - CircleData->Size;
    sint32_t CircleTopGridY = fan::math::floor(CircleTopY / GridBlockSize);
    for(sint32_t CircleGridY = CircleMiddleGridY; CircleGridY > CircleTopGridY;){
      _f CircleY = (_f)CircleGridY * GridBlockSize;
      _f CircleOffsetY = CircleY - NewPosition.y;
      _f Magic = fan::math::sqrt(fan::math::abs(CircleData->Size * CircleData->Size - CircleOffsetY * CircleOffsetY));
      _f CircleLeftX = NewPosition.x - Magic;
      _f CircleRightX = NewPosition.x + Magic;
      sint32_t CircleLeftGridX = fan::math::floor(CircleLeftX / GridBlockSize);
      sint32_t CircleRightGridX = fan::math::floor(CircleRightX / GridBlockSize);
      CircleGridY--;
      for(sint32_t CircleGridX = CircleLeftGridX; CircleGridX <= CircleRightGridX; CircleGridX++){
        Contact_Grid_t Contact;
        this->PreSolve_Grid_cb(
          this,
          &sip0,
          {CircleGridX, CircleGridY},
          &Contact);
        if(this->ObjectList.CheckSafeNext(0) != sip0.ObjectID){
          goto gt_Object0Unlinked;
        }
        if(Contact.Flag & Contact_Grid_Flag::EnableContact); else{
          continue;
        };

        _vf oCircle;
        _vf oDirection;
        CPC_Circle_Square(
          NewPosition,
          CircleData->Size,
          {CircleGridX * GridBlockSize + GridBlockSize_D2, CircleGridY * GridBlockSize + GridBlockSize_D2},
          GridBlockSize_D2,
          &oCircle,
          &oDirection);

        #ifdef ETC_BCOL_set_PostSolve_Grid
          ContactResult_Grid_t ContactResult;
        #endif
        #ifdef ETC_BCOL_set_PostSolve_Grid_CollisionNormal
          ContactResult.Normal = oDirection;
        #endif
        #ifdef ETC_BCOL_set_PostSolve_Grid
          this->PostSolve_Grid_cb(
            this,
            &sip0,
            {CircleGridX, CircleGridY},
            &ContactResult);
          if(this->ObjectList.CheckSafeNext(0) != sip0.ObjectID){
            goto gt_Object0Unlinked;
          }
        #endif

        WantedPosition += oCircle;
        WantedDirection += oDirection;
        WantedCollisionRequesters++;
      }
    }
  }

  {
    _f CircleBottomY = NewPosition.y + CircleData->Size;
    sint32_t CircleBottomGridY = fan::math::floor(CircleBottomY / GridBlockSize);
    for(sint32_t CircleGridY = CircleMiddleGridY; CircleGridY < CircleBottomGridY;){
      CircleGridY++;
      _f CircleY = (_f)CircleGridY * GridBlockSize;
      _f CircleOffsetY = CircleY - NewPosition.y;
      _f Magic = fan::math::sqrt(fan::math::abs(CircleData->Size * CircleData->Size - CircleOffsetY * CircleOffsetY));
      _f CircleLeftX = NewPosition.x - Magic;
      _f CircleRightX = NewPosition.x + Magic;
      sint32_t CircleLeftGridX = fan::math::floor(CircleLeftX / GridBlockSize);
      sint32_t CircleRightGridX = fan::math::floor(CircleRightX / GridBlockSize);
      for(sint32_t CircleGridX = CircleLeftGridX; CircleGridX <= CircleRightGridX; CircleGridX++){
        Contact_Grid_t Contact;
        this->PreSolve_Grid_cb(
          this,
          &sip0,
          {CircleGridX, CircleGridY},
          &Contact);
        if(this->ObjectList.CheckSafeNext(0) != sip0.ObjectID){
          goto gt_Object0Unlinked;
        }
        if(Contact.Flag & Contact_Grid_Flag::EnableContact); else{
          continue;
        };

        _vf oCircle;
        _vf oDirection;
        CPC_Circle_Square(
          NewPosition,
          CircleData->Size,
          {CircleGridX * GridBlockSize + GridBlockSize_D2, CircleGridY * GridBlockSize + GridBlockSize_D2},
          GridBlockSize_D2,
          &oCircle,
          &oDirection);

        #ifdef ETC_BCOL_set_PostSolve_Grid
          ContactResult_Grid_t ContactResult;
        #endif
        #ifdef ETC_BCOL_set_PostSolve_Grid_CollisionNormal
          ContactResult.Normal = oDirection;
        #endif
        #ifdef ETC_BCOL_set_PostSolve_Grid
          this->PostSolve_Grid_cb(
            this,
            &sip0,
            {CircleGridX, CircleGridY},
            &ContactResult);
          if(this->ObjectList.CheckSafeNext(0) != sip0.ObjectID){
            goto gt_Object0Unlinked;
          }
        #endif

        WantedPosition += oCircle;
        WantedDirection += oDirection;
        WantedCollisionRequesters++;
      }
    }
  }
#endif

#if ETC_BCOL_set_DynamicToDynamic == 1
  ShapeInfoPack_t sip1;
  sip1.ObjectID = this->ObjectList.GetNodeFirst();
  while(sip1.ObjectID != this->ObjectList.dst){
    this->ObjectList.StartSafeNext(sip1.ObjectID);
    auto ObjectData1 = this->GetObjectData(sip1.ObjectID);
    if(sip1.ObjectID == sip0.ObjectID){
      goto gt_Object1_Circle_Unlinked;
    }

    for(sip1.ShapeID.ID = 0; sip1.ShapeID.ID < ObjectData1->ShapeList.Current; sip1.ShapeID.ID++){
      auto ShapeData1 = this->GetObject_ShapeData(sip1.ObjectID, sip1.ShapeID);
      sip1.ShapeEnum = ShapeData1->ShapeEnum;

      switch(sip1.ShapeEnum){
        case ShapeEnum_t::Circle:{
          auto CircleData1 = this->ShapeData_Circle_Get(ShapeData1->ShapeID);

          _vf WorldPosition = ObjectData1->Position + CircleData1->Position;

          _vf Difference = NewPosition - WorldPosition;
          _f Hypotenuse = Difference.length();
          _f CombinedSize = CircleData->Size + CircleData1->Size;
          if(Hypotenuse >= CombinedSize){
            break;
          }

          Contact_Shape_t Contact[2];
          this->PreSolve_Shape_cb(
            this,
            &sip0,
            &sip1,
            &Contact[0]);
          this->PreSolve_Shape_cb(
            this,
            &sip1,
            &sip0,
            &Contact[1]);

          Contact[0].AfterCB(
            this,
            &sip0,
            &sip1
          );
          Contact[1].AfterCB(
            this,
            &sip1,
            &sip0
          );
          if(ObjectList.CheckSafeNext(1) != sip0.ObjectID){
            this->ObjectList.EndSafeNext();
            goto gt_Object0Unlinked;
          }
          if(ObjectList.CheckSafeNext(0) != sip1.ObjectID){
            goto gt_Object1_Circle_Unlinked;
          }
          if(Contact[0].Flag & Contact[1].Flag & Contact_Shape_Flag::EnableContact); else{
            break;
          };

          if(Hypotenuse != 0){
            _vf Direction = Difference / Hypotenuse;
            WantedPosition += NewPosition + Direction * (CombinedSize - Hypotenuse);
            WantedDirection += Direction;
            WantedCollisionRequesters++;
          }
          break;
        }
        case ShapeEnum_t::Rectangle:{
          auto RectangleData_ = this->ShapeData_Rectangle_Get(ShapeData1->ShapeID);

          _vf WorldPosition = ObjectData1->Position + RectangleData_->Position;

          CPCU_Circle_Rectangle_t CData;
          CPCU_Circle_Rectangle_Pre(
            NewPosition,
            CircleData->Size,
            WorldPosition,
            RectangleData_->Size,
            &CData);

          if(!CPCU_Circle_Rectangle_IsThereCollision(&CData)){
            break;
          }

          Contact_Shape_t Contact;
          this->PreSolve_Shape_cb(
            this,
            &sip0,
            &sip1,
            &Contact);
          if(ObjectList.CheckSafeNext(1) != sip0.ObjectID){
            this->ObjectList.EndSafeNext();
            goto gt_Object0Unlinked;
          }
          if(ObjectList.CheckSafeNext(0) != sip1.ObjectID){
            goto gt_Object1_Circle_Unlinked;
          }
          if(Contact.Flag & Contact_Shape_Flag::EnableContact); else{
            break;
          };

          _vf oPosition;
          _vf oDirection;
          CPCU_Circle_Rectangle_Solve(
            NewPosition,
            CircleData->Size,
            WorldPosition,
            RectangleData_->Size,
            &CData,
            &oPosition,
            &oDirection);

          WantedPosition += oPosition;
          WantedDirection += oDirection;
          WantedCollisionRequesters++;

          break;
        }
      }
    }

    gt_Object1_Circle_Unlinked:
    sip1.ObjectID = this->ObjectList.EndSafeNext();
  }
#endif

if(WantedCollisionRequesters){
  WantedObjectPosition += WantedPosition - CircleData->Position * WantedCollisionRequesters;
  WantedObjectDirection += WantedDirection;
  WantedObjectCollisionRequesters += WantedCollisionRequesters;
}

break;
