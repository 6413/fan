void Step(
  _f delta
){
  ShapeInfoPack_t sip0;
  sip0.ObjectID = this->ObjectList.GetNodeFirst();
  while(sip0.ObjectID != this->ObjectList.dst){
    this->ObjectList.StartSafeNext(sip0.ObjectID);

    auto ObjectData0 = this->GetObjectData(sip0.ObjectID);

    /* bad way */
    if(ObjectData0->Flag & ObjectFlag::Constant){
      sip0.ObjectID = this->ObjectList.EndSafeNext();
      continue;
    }

    {
      BCOL_set_DynamicDeltaFunction
    }

    const _vf StepVelocity = ObjectData0->Velocity * delta;

    _vf NewObjectPosition = ObjectData0->Position + StepVelocity;

    _vf WantedObjectPosition = 0;
    _vf WantedObjectDirection = 0;
    _f WantedObjectCollisionRequesters = 0;

    for(sip0.ShapeID.ID = 0; sip0.ShapeID.ID < ObjectData0->ShapeList.Current; sip0.ShapeID.ID++){
      sip0.ShapeEnum = ObjectData0->ShapeList.ptr[sip0.ShapeID.ID].ShapeEnum;
      switch(sip0.ShapeEnum){
        case ShapeEnum_t::Circle:{
          #include "Shape/Circle.h"
        }
        case ShapeEnum_t::Rectangle:{
          #include "Shape/Rectangle.h"
        }
      }

      /* user is able to change pointers by adding or removing objects inside callbacks */
      ObjectData0 = this->GetObjectData(sip0.ObjectID);
    }

    if(WantedObjectCollisionRequesters){
      ObjectData0->Position = WantedObjectPosition / WantedObjectCollisionRequesters;

      _vf DirectionAverage = WantedObjectDirection / WantedObjectCollisionRequesters;

      _f VelocityHypotenuse = ObjectData0->Velocity.length();
      if(VelocityHypotenuse != 0){
        _vf VelocityNormal = ObjectData0->Velocity / VelocityHypotenuse;

        _vf CollidedVelocity = VelocityNormal.reflect(DirectionAverage);

        ObjectData0->Velocity = CollidedVelocity * VelocityHypotenuse;

        #if defined(BCOL_set_ConstantFriction) || defined(BCOL_set_ConstantBumpFriction)
          _f ForceThroughNormal = fan::math::dot2(DirectionAverage, VelocityNormal);
          ForceThroughNormal = abs(ForceThroughNormal) * VelocityHypotenuse;
        #endif
        #ifdef BCOL_set_ConstantFriction
          ObjectData0->Velocity /= ForceThroughNormal * BCOL_set_ConstantFriction * delta + 1;
        #endif
        #ifdef BCOL_set_ConstantBumpFriction
          ObjectData0->Velocity -= fan::math::copysign(
            min(abs(ObjectData0->Velocity),
            ForceThroughNormal * BCOL_set_ConstantBumpFriction * delta),
            ObjectData0->Velocity);
        #endif
      }
    }
    else{
      ObjectData0->Position = NewObjectPosition;
    }

    gt_Object0Unlinked:
    sip0.ObjectID = this->ObjectList.EndSafeNext();
  }

  #if BCOL_set_StepNumber == 1
    this->StepNumber++;
  #endif
}
