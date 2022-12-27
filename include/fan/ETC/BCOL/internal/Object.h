__ETC_BCOL_PP(ObjectData_t) *
__ETC_BCOL_PP(GetObjectData)
(
  __ETC_BCOL_P(t) *bcol,
  __ETC_BCOL_P(ObjectID_t) ObjectID
){
  return &bcol->ObjectList[ObjectID];
}

void
__ETC_BCOL_P(UnlinkObject)
(
  __ETC_BCOL_P(t) *bcol,
  __ETC_BCOL_P(ObjectID_t) ObjectID
){
  bcol->ObjectList.Unlink(ObjectID);
}
void
__ETC_BCOL_P(LinkObject)
(
  __ETC_BCOL_P(t) *bcol,
  __ETC_BCOL_P(ObjectID_t) ObjectID
){
  bcol->ObjectList.linkPrev(bcol->ObjectList.dst, ObjectID);
}

__ETC_BCOL_P(ObjectID_t)
__ETC_BCOL_P(NewObject)
(
  __ETC_BCOL_P(t) *bcol,
  __ETC_BCOL_P(ObjectProperties_t) *ObjectProperties,
  uint32_t Flag
){
  auto ObjectID = bcol->ObjectList.NewNodeLast();
  auto ObjectData = __ETC_BCOL_PP(GetObjectData)(bcol, ObjectID);

  ObjectData->Flag = Flag;

  ObjectData->PositionY = ObjectProperties->PositionY;
  ObjectData->PositionX = ObjectProperties->PositionX;

  VEC_init(&ObjectData->ShapeList, sizeof(__ETC_BCOL_PP(ShapeData_t)), A_resize);

  #if ETC_BCOL_set_StoreExtraDataInsideObject == 1
    ObjectData->ExtraData = ObjectProperties->ExtraData;
  #endif

  return ObjectID;
}

#if ETC_BCOL_set_StoreExtraDataInsideObject == 1
  __ETC_BCOL_P(ObjectExtraData_t) *
  __ETC_BCOL_P(GetObjectExtraData)
  (
    __ETC_BCOL_P(t) *bcol,
    __ETC_BCOL_P(ObjectID_t) ObjectID
  ){
    __ETC_BCOL_PP(ObjectData_t) *ObjectData = __ETC_BCOL_PP(GetObjectData)(bcol, ObjectID);
    return &ObjectData->ExtraData;
  }
#endif

void
__ETC_BCOL_P(SetObject_Position)
(
  __ETC_BCOL_P(t) *bcol,
  __ETC_BCOL_P(ObjectID_t) ObjectID,
  __pfloat PositionY,
  __pfloat PositionX
){
  __ETC_BCOL_PP(ObjectData_t) *ObjectData = __ETC_BCOL_PP(GetObjectData)(bcol, ObjectID);
  ObjectData->PositionY = PositionY;
  ObjectData->PositionX = PositionX;
}
void
__ETC_BCOL_P(GetObject_Position)
(
  __ETC_BCOL_P(t) *bcol,
  __ETC_BCOL_P(ObjectID_t) ObjectID,
  __pfloat *PositionY,
  __pfloat *PositionX
){
  __ETC_BCOL_PP(ObjectData_t) *ObjectData = __ETC_BCOL_PP(GetObjectData)(bcol, ObjectID);
  *PositionY = ObjectData->PositionY;
  *PositionX = ObjectData->PositionX;
}

void
__ETC_BCOL_P(SetObject_Velocity)
(
  __ETC_BCOL_P(t) *bcol,
  __ETC_BCOL_P(ObjectID_t) ObjectID,
  __pfloat VelocityY,
  __pfloat VelocityX
){
  __ETC_BCOL_PP(ObjectData_t) *ObjectData = __ETC_BCOL_PP(GetObjectData)(bcol, ObjectID);
  ObjectData->VelocityY = VelocityY;
  ObjectData->VelocityX = VelocityX;
}
void
__ETC_BCOL_P(GetObject_Velocity)
(
  __ETC_BCOL_P(t) *bcol,
  __ETC_BCOL_P(ObjectID_t) ObjectID,
  __pfloat *VelocityY,
  __pfloat *VelocityX
){
  __ETC_BCOL_PP(ObjectData_t) *ObjectData = __ETC_BCOL_PP(GetObjectData)(bcol, ObjectID);
  *VelocityY = ObjectData->VelocityY;
  *VelocityX = ObjectData->VelocityX;
}
