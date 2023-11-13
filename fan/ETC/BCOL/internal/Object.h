ObjectData_t *GetObjectData(ObjectID_t ObjectID){
  return &this->ObjectList[ObjectID];
}

ShapeData_t *GetObject_ShapeData(ObjectID_t ObjectID, ShapeID_t ShapeID){
  auto ObjectData = this->GetObjectData(ObjectID);
  return &ObjectData->ShapeList.ptr[ShapeID.ID];
}

void UnlinkObject(ObjectID_t ObjectID){
  this->ObjectList.Unlink(ObjectID);
}
void LinkObject(ObjectID_t ObjectID){
  this->ObjectList.linkPrev(this->ObjectList.dst, ObjectID);
}

ObjectID_t NewObject(ObjectProperties_t *ObjectProperties, uint32_t Flag){
  auto ObjectID = this->ObjectList.NewNodeLast();
  auto ObjectData = GetObjectData(ObjectID);

  ObjectData->Flag = Flag;

  ObjectData->Position = ObjectProperties->Position;

  ShapeList_Open(&ObjectData->ShapeList);

  #if ETC_BCOL_set_StoreExtraDataInsideObject == 1
    ObjectData->ExtraData = ObjectProperties->ExtraData;
  #endif

  return ObjectID;
}

#if ETC_BCOL_set_StoreExtraDataInsideObject == 1
  ObjectExtraData_t *GetObjectExtraData(ObjectID_t ObjectID){
    auto ObjectData = this->GetObjectData(ObjectID);
    return &ObjectData->ExtraData;
  }
#endif

void SetObject_Position(ObjectID_t ObjectID, _vf Position){
  auto ObjectData = this->GetObjectData(ObjectID);
  ObjectData->Position = Position;
}
_vf GetObject_Position(ObjectID_t ObjectID){
  auto ObjectData = this->GetObjectData(ObjectID);
  return ObjectData->Position;
}

void SetObject_Velocity(ObjectID_t ObjectID, _vf Velocity){
  auto ObjectData = this->GetObjectData(ObjectID);
  ObjectData->Velocity = Velocity;
}
_vf GetObject_Velocity(ObjectID_t ObjectID){
  auto ObjectData = this->GetObjectData(ObjectID);
  return ObjectData->Velocity;
}
