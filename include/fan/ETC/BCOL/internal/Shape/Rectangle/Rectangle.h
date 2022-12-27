typedef struct{
  __pfloat PositionY;
  __pfloat PositionX;
  __pfloat SizeY;
  __pfloat SizeX;
}__ETC_BCOL_P(ShapeProperties_Rectangle_t);

__ETC_BCOL_PP(ShapeData_Rectangle_t) *
__ETC_BCOL_PP(ShapeData_Rectangle_Get)
(
  __ETC_BCOL_P(t) *bcol,
  __ETC_BCOL_P(ShapeID_t) ShapeID
){
  return &bcol->ShapeList_Rectangle[ShapeID];
}

__ETC_BCOL_P(ShapeID_t)
__ETC_BCOL_P(NewShape_Rectangle)
(
  __ETC_BCOL_P(t) *bcol,
  __ETC_BCOL_P(ObjectID_t) ObjectID,
  __ETC_BCOL_P(ShapeProperties_Rectangle_t) *Properties
){
  __ETC_BCOL_P(ShapeID_t) ShapeID = bcol->ShapeList_Rectangle.NewNodeLast();
  __ETC_BCOL_PP(ShapeData_Rectangle_t) *SData = __ETC_BCOL_PP(ShapeData_Rectangle_Get)(bcol, ShapeID);
  SData->PositionY = Properties->PositionY;
  SData->PositionX = Properties->PositionX;
  SData->SizeY = Properties->SizeY;
  SData->SizeX = Properties->SizeX;
  __ETC_BCOL_PP(AddShapeToObject)(bcol, ObjectID, __ETC_BCOL_P(ShapeEnum_Rectangle), ShapeID);
  return ShapeID;
}

void
__ETC_BCOL_P(GetShape_Rectangle_WorldPosition)
(
  __ETC_BCOL_P(t) *bcol,
  __ETC_BCOL_P(ObjectID_t) ObjectID,
  __ETC_BCOL_P(ShapeID_t) ShapeID,
  __pfloat *PositionY,
  __pfloat *PositionX
){
  __ETC_BCOL_PP(ObjectData_t) *OData = __ETC_BCOL_PP(GetObjectData)(bcol, ObjectID);
  __ETC_BCOL_PP(ShapeData_Rectangle_t) *SData = __ETC_BCOL_PP(ShapeData_Rectangle_Get)(bcol, ShapeID);
  *PositionY = OData->PositionY + SData->PositionY;
  *PositionX = OData->PositionX + SData->PositionX;
}

void
__ETC_BCOL_P(GetShape_Rectangle_Size)
(
  __ETC_BCOL_P(t) *bcol,
  __ETC_BCOL_P(ShapeID_t) ShapeID,
  __pfloat *SizeY,
  __pfloat *SizeX
){
  __ETC_BCOL_PP(ShapeData_Rectangle_t) *SData = __ETC_BCOL_PP(ShapeData_Rectangle_Get)(bcol, ShapeID);
  *SizeY = SData->SizeY;
  *SizeX = SData->SizeX;
}

void
__ETC_BCOL_PP(UnlinkRecycleOrphanShape_Rectangle)
(
  __ETC_BCOL_P(t) *bcol,
  __ETC_BCOL_P(ShapeID_t) ShapeID
){
  bcol->ShapeList_Rectangle.unlrec(ShapeID);
}
