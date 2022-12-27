typedef struct{
  uint8_t ShapeEnum;
  __ETC_BCOL_P(ShapeID_t) ShapeID;
  struct{
    uint32_t ShapeIndex;
  }priv;
}__ETC_BCOL_P(TraverseObject_t);

void
__ETC_BCOL_P(TraverseObject_init)
(
  __ETC_BCOL_P(TraverseObject_t) *TraverseObject
){
  TraverseObject->priv.ShapeIndex = 0;
}

bool
__ETC_BCOL_P(TraverseObject_loop)
(
  __ETC_BCOL_P(t) *bcol,
  __ETC_BCOL_P(ObjectID_t) ObjectID,
  __ETC_BCOL_P(TraverseObject_t) *TraverseObject
){
  __ETC_BCOL_PP(ObjectData_t) *ObjectData = __ETC_BCOL_PP(GetObjectData)(bcol, ObjectID);
  if(TraverseObject->priv.ShapeIndex == ObjectData->ShapeList.Current){
    return 0;
  }

  __ETC_BCOL_PP(ShapeData_t) *ShapeData = &((__ETC_BCOL_PP(ShapeData_t) *)ObjectData->ShapeList.ptr)[TraverseObject->priv.ShapeIndex];
  TraverseObject->ShapeEnum = ShapeData->ShapeEnum;
  TraverseObject->ShapeID = ShapeData->ShapeID;

  TraverseObject->priv.ShapeIndex++;
  return 1;
}

typedef struct{
  __ETC_BCOL_P(ObjectID_t) ObjectID;
  struct{
    __ETC_BCOL_P(ObjectID_t) NextObjectID;
  }priv;
}__ETC_BCOL_P(TraverseObjects_t);

void
__ETC_BCOL_P(TraverseObjects_init)
(
  __ETC_BCOL_P(t) *bcol,
  __ETC_BCOL_P(TraverseObjects_t) *TraverseObjects
){
  TraverseObjects->ObjectID = bcol->ObjectList.src;
  auto ObjectNode = bcol->ObjectList.GetNodeByReference(TraverseObjects->ObjectID);
  TraverseObjects->priv.NextObjectID = ObjectNode->NextNodeReference;
}

bool
__ETC_BCOL_P(TraverseObjects_loop)
(
  __ETC_BCOL_P(t) *bcol,
  __ETC_BCOL_P(TraverseObjects_t) *TraverseObjects
){
  if(TraverseObjects->priv.NextObjectID == bcol->ObjectList.dst){
    return 0;
  }
  TraverseObjects->ObjectID = TraverseObjects->priv.NextObjectID;
  auto ObjectNode = bcol->ObjectList.GetNodeByReference(TraverseObjects->ObjectID);
  TraverseObjects->priv.NextObjectID = ObjectNode->NextNodeReference;
  return 1;
}
