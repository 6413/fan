struct ShapeProperties_Circle_t{
  _vf Position;
  _f Size;
};

ShapeData_Circle_t *ShapeData_Circle_Get(
  ShapeID_t ShapeID
){
  return &this->ShapeList_Circle[ShapeID];
}

ShapeID_t NewShape_Circle(
  ObjectID_t ObjectID,
  ShapeProperties_Circle_t *Properties
){
  auto ShapeID = this->ShapeList_Circle.NewNodeLast();
  auto SData = this->ShapeData_Circle_Get(ShapeID);
  SData->Position = Properties->Position;
  SData->Size = Properties->Size;
  this->AddShapeToObject(ObjectID, ShapeEnum_t::Circle, ShapeID);
  return ShapeID;
}

_vf GetShape_Circle_WorldPosition(
  ObjectID_t ObjectID,
  ShapeID_t ShapeID
){
  auto OData = this->GetObjectData(ObjectID);
  auto SData = this->ShapeData_Circle_Get(ShapeID);
  return OData->Position + SData->Position;
}

_f GetShape_Circle_Size(ShapeID_t ShapeID){
  auto SData = this->ShapeData_Circle_Get(ShapeID);
  return SData->Size;
}

void UnlinkRecycleOrphanShape_Circle(
  ShapeID_t ShapeID
){
  this->ShapeList_Circle.unlrec(ShapeID);
}
