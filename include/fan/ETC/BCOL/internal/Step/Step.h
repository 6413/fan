void
__ETC_BCOL_P(Step)
(
  __ETC_BCOL_P(t) *bcol,
  __pfloat delta
){
  __ETC_BCOL_P(TraverseObjects_t) TraverseObjects;
  __ETC_BCOL_P(TraverseObjects_init)(bcol, &TraverseObjects);
  while(__ETC_BCOL_P(TraverseObjects_loop)(bcol, &TraverseObjects)){
    __ETC_BCOL_P(ObjectID_t) ObjectID = TraverseObjects.ObjectID;
    __ETC_BCOL_PP(ObjectData_t) *ObjectData = __ETC_BCOL_PP(GetObjectData)(bcol, ObjectID);

    /* bad */
    if(ObjectData->Flag & __ETC_BCOL_P(ObjectFlag_Constant)){
      continue;
    }

    {
      ETC_BCOL_set_DynamicDeltaFunction
    }

    const __pfloat StepVelocityY = ObjectData->VelocityY * delta;
    const __pfloat StepVelocityX = ObjectData->VelocityX * delta;

    __pfloat NewObjectPositionY = ObjectData->PositionY + StepVelocityY;
    __pfloat NewObjectPositionX = ObjectData->PositionX + StepVelocityX;

    __pfloat WantedObjectPositionY = 0;
    __pfloat WantedObjectPositionX = 0;
    __pfloat WantedObjectDirectionY = 0;
    __pfloat WantedObjectDirectionX = 0;
    __pfloat WantedObjectCollisionRequesters = 0;

    for(uint32_t ShapeID = 0; ShapeID < ObjectData->ShapeList.Current; ShapeID++){
      __ETC_BCOL_PP(ShapeData_t) *ShapeData = &((__ETC_BCOL_PP(ShapeData_t) *)ObjectData->ShapeList.ptr)[ShapeID];

      switch(ShapeData->ShapeEnum){
        case __ETC_BCOL_P(ShapeEnum_Circle):{
          #include _WITCH_PATH(ETC/BCOL/internal/Step/Shape/Circle.h)
        }
        case __ETC_BCOL_P(ShapeEnum_Square):{
          #include _WITCH_PATH(ETC/BCOL/internal/Step/Shape/Square.h)
        }
        case __ETC_BCOL_P(ShapeEnum_Rectangle):{
          #include _WITCH_PATH(ETC/BCOL/internal/Step/Shape/Rectangle.h)
        }
      }
    }

    if(WantedObjectCollisionRequesters){
      ObjectData->PositionY = WantedObjectPositionY / WantedObjectCollisionRequesters;
      ObjectData->PositionX = WantedObjectPositionX / WantedObjectCollisionRequesters;

      __pfloat DirectionAverageY = WantedObjectDirectionY / WantedObjectCollisionRequesters;
      __pfloat DirectionAverageX = WantedObjectDirectionX / WantedObjectCollisionRequesters;

      __pfloat VelocityHypotenuse = __hypotenuse(ObjectData->VelocityY, ObjectData->VelocityX);
      if(VelocityHypotenuse != 0){
        __pfloat VelocityNormalY = ObjectData->VelocityY / VelocityHypotenuse;
        __pfloat VelocityNormalX = ObjectData->VelocityX / VelocityHypotenuse;

        __pfloat CollidedVelocityY;
        __pfloat CollidedVelocityX;
        __NormalResolve(
          VelocityNormalY,
          VelocityNormalX,
          DirectionAverageY,
          DirectionAverageX,
          1,
          &CollidedVelocityY,
          &CollidedVelocityX);

        ObjectData->VelocityY = CollidedVelocityY * VelocityHypotenuse;
        ObjectData->VelocityX = CollidedVelocityX * VelocityHypotenuse;

        #if defined(ETC_BCOL_set_ConstantFriction) || defined(ETC_BCOL_set_ConstantBumpFriction)
          __pfloat ForceThroughNormal = __dot2(DirectionAverageY, DirectionAverageX, VelocityNormalY, VelocityNormalX);
          ForceThroughNormal = __absf(ForceThroughNormal) * VelocityHypotenuse;
        #endif
        #ifdef ETC_BCOL_set_ConstantFriction
          ObjectData->VelocityY /= ForceThroughNormal * ETC_BCOL_set_ConstantFriction * delta + 1;
          ObjectData->VelocityX /= ForceThroughNormal * ETC_BCOL_set_ConstantFriction * delta + 1;
        #endif
        #ifdef ETC_BCOL_set_ConstantBumpFriction
          ObjectData->VelocityY -= MATH_copysign_f32(
            MATH_min_f32(MATH_abs_f32(ObjectData->VelocityY),
            ForceThroughNormal * ETC_BCOL_set_ConstantBumpFriction * delta),
            ObjectData->VelocityY);
          ObjectData->VelocityX -= MATH_copysign_f32(
            MATH_min_f32(MATH_abs_f32(ObjectData->VelocityX),
            ForceThroughNormal * ETC_BCOL_set_ConstantBumpFriction * delta),
            ObjectData->VelocityX);
        #endif
      }
    }
    else{
      ObjectData->PositionY = NewObjectPositionY;
      ObjectData->PositionX = NewObjectPositionX;
    }
  }

  #if ETC_BCOL_set_StepNumber == 1
    bcol->StepNumber++;
  #endif
}
