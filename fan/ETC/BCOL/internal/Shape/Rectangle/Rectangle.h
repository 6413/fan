struct ShapeProperties_Rectangle_t{
  _vf Position;
  _vf Size;
};

ShapeData_Rectangle_t *ShapeData_Rectangle_Get(
  ShapeID_t ShapeID
){
  return &this->ShapeList_Rectangle[ShapeID];
}

ShapeID_t NewShape_Rectangle(
  ObjectID_t ObjectID,
  ShapeProperties_Rectangle_t *Properties
){
  auto ShapeID = this->ShapeList_Rectangle.NewNodeLast();
  auto SData = this->ShapeData_Rectangle_Get(ShapeID);
  SData->Position = Properties->Position;
  SData->Size = Properties->Size;
  this->AddShapeToObject(ObjectID, ShapeEnum_t::Rectangle, ShapeID);
  return ShapeID;
}

_vf GetShape_Rectangle_WorldPosition(
  ObjectID_t ObjectID,
  ShapeID_t ShapeID
){
  auto OData = this->GetObjectData(ObjectID);
  auto SData = this->ShapeData_Rectangle_Get(ShapeID);
  return OData->Position + SData->Position;
}

_vf GetShape_Rectangle_Size(ShapeID_t ShapeID){
  auto SData = this->ShapeData_Rectangle_Get(ShapeID);
  return SData->Size;
}

void UnlinkRecycleOrphanShape_Rectangle(
  ShapeID_t ShapeID
){
  this->ShapeList_Rectangle.unlrec(ShapeID);
}
