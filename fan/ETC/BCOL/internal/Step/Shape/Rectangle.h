auto RectangleData = ShapeData_Rectangle_Get(ObjectData0->ShapeList.ptr[sip0.ShapeID.ID].ShapeID);

_vf NewPosition = NewObjectPosition + RectangleData->Position;

_vf WantedPosition = 0;
_vf WantedDirection = 0;
_f WantedCollisionRequesters = 0;

#if BCOL_set_SupportGrid == 1
  const _f GridBlockSize = this->GridBlockSize;
  const _f GridBlockSize_D2 = GridBlockSize / 2;

  _vf gbs(GridBlockSize);
  for(iterate_grid_for_rectangle_t ig; ig.it(gbs, NewPosition, RectangleData->Size);){
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

    _vf oRectangle;
    _vf oDirection;
    CPC_Rectangle_Square(
      NewPosition,
      RectangleData->Size,
      _vf(ig.gs) * GridBlockSize + GridBlockSize_D2,
      GridBlockSize_D2,
      &oRectangle,
      &oDirection);
    WantedPosition += oRectangle;
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
      goto gt_Object1_Rectangle_Unlinked;
    }

    for(sip1.ShapeID.ID = 0; sip1.ShapeID.ID < ObjectData1->ShapeList.Current; sip1.ShapeID.ID++){
      auto ShapeData1 = this->GetObject_ShapeData(sip1.ObjectID, sip1.ShapeID);
      sip1.ShapeEnum = ShapeData1->ShapeEnum;

      switch(sip1.ShapeEnum){
        case ShapeEnum_t::Circle:{
          auto CircleData_ = this->ShapeData_Circle_Get(ShapeData1->ShapeID);

          _vf WorldPosition = ObjectData1->Position + CircleData_->Position;

          CPCU_Rectangle_Circle_t CData;
          CPCU_Rectangle_Circle_Pre(
            NewPosition,
            RectangleData->Size,
            WorldPosition,
            CircleData_->Size,
            &CData);

          if(!CPCU_Rectangle_Circle_IsThereCollision(&CData)){
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
            goto gt_Object1_Rectangle_Unlinked;
          }
          if(Contact.Flag & Contact_Shape_Flag::EnableContact); else{
            break;
          };

          _vf oPosition;
          _vf oDirection;
          CPCU_Rectangle_Circle_Solve(
            NewPosition,
            RectangleData->Size,
            WorldPosition,
            CircleData_->Size,
            &CData,
            &oPosition,
            &oDirection);

          WantedPosition += oPosition;
          WantedDirection += oDirection;
          WantedCollisionRequesters++;

          break;
        }
        case ShapeEnum_t::Rectangle:{
          /* TODO */
          break;
        }
      }
    }

    gt_Object1_Rectangle_Unlinked:
    sip1.ObjectID = this->ObjectList.EndSafeNext();
  }
#endif

if(WantedCollisionRequesters){
  WantedObjectPosition += WantedPosition - RectangleData->Position * WantedCollisionRequesters;
  WantedObjectDirection += WantedDirection;
  WantedObjectCollisionRequesters += WantedCollisionRequesters;
}

break;
