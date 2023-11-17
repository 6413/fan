auto CircleData = ShapeData_Circle_Get(ObjectData0->ShapeList.ptr[sip0.ShapeID.ID].ShapeID);

_vf NewPosition = NewObjectPosition + CircleData->Position;

_vf WantedPosition = 0;
_vf WantedDirection = 0;
_f WantedCollisionRequesters = 0;

#if BCOL_set_SupportGrid == 1
  const _f GridBlockSize = this->GridBlockSize;
  const _f GridBlockSize_D2 = GridBlockSize / 2;

  _vf gbs(GridBlockSize);
  for(iterate_grid_for_circle_t<gbs.size()> ig; ig.it(gbs, NewPosition, CircleData->Size);){
    Contact_Grid_t Contact;
    this->PreSolve_Grid_cb(
      this,
      &sip0,
      ig.gs,
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
      _vf(ig.gs) * GridBlockSize + GridBlockSize_D2,
      GridBlockSize_D2,
      &oCircle,
      &oDirection);

    #ifdef BCOL_set_PostSolve_Grid
      ContactResult_Grid_t ContactResult;
    #endif
    #ifdef BCOL_set_PostSolve_Grid_CollisionNormal
      ContactResult.Normal = oDirection;
    #endif
    #ifdef BCOL_set_PostSolve_Grid
      this->PostSolve_Grid_cb(
        this,
        &sip0,
        ig.gs,
        &ContactResult);
      if(this->ObjectList.CheckSafeNext(0) != sip0.ObjectID){
        goto gt_Object0Unlinked;
      }
    #endif

    WantedPosition += oCircle;
    WantedDirection += oDirection;
    WantedCollisionRequesters++;
  }
#endif

#if BCOL_set_DynamicToDynamic == 1
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
