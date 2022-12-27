typedef struct{
  __pfloat PositionY;
  __pfloat PositionX;
  __pfloat Size;
}__ETC_BCOL_P(ShapeProperties_Circle_t);

__ETC_BCOL_PP(ShapeData_Circle_t) *
__ETC_BCOL_PP(ShapeData_Circle_Get)
(
  __ETC_BCOL_P(t) *bcol,
  __ETC_BCOL_P(ShapeID_t) ShapeID
){
  return &bcol->ShapeList_Circle[ShapeID];
}

__ETC_BCOL_P(ShapeID_t)
__ETC_BCOL_P(NewShape_Circle)
(
  __ETC_BCOL_P(t) *bcol,
  __ETC_BCOL_P(ObjectID_t) ObjectID,
  __ETC_BCOL_P(ShapeProperties_Circle_t) *Properties
){
  __ETC_BCOL_P(ShapeID_t) ShapeID = bcol->ShapeList_Circle.NewNodeLast();
  __ETC_BCOL_PP(ShapeData_Circle_t) *SData = __ETC_BCOL_PP(ShapeData_Circle_Get)(bcol, ShapeID);
  SData->PositionY = Properties->PositionY;
  SData->PositionX = Properties->PositionX;
  SData->Size = Properties->Size;
  __ETC_BCOL_PP(AddShapeToObject)(bcol, ObjectID, __ETC_BCOL_P(ShapeEnum_Circle), ShapeID);
  return ShapeID;
}

void
__ETC_BCOL_P(GetShape_Circle_WorldPosition)
(
  __ETC_BCOL_P(t) *bcol,
  __ETC_BCOL_P(ObjectID_t) ObjectID,
  __ETC_BCOL_P(ShapeID_t) ShapeID,
  __pfloat *PositionY,
  __pfloat *PositionX
){
  __ETC_BCOL_PP(ObjectData_t) *OData = __ETC_BCOL_PP(GetObjectData)(bcol, ObjectID);
  __ETC_BCOL_PP(ShapeData_Circle_t) *SData = __ETC_BCOL_PP(ShapeData_Circle_Get)(bcol, ShapeID);
  *PositionY = OData->PositionY + SData->PositionY;
  *PositionX = OData->PositionX + SData->PositionX;
}

void
__ETC_BCOL_P(GetShape_Circle_Size)
(
  __ETC_BCOL_P(t) *bcol,
  __ETC_BCOL_P(ShapeID_t) ShapeID,
  __pfloat *Size
){
  __ETC_BCOL_PP(ShapeData_Circle_t) *SData = __ETC_BCOL_PP(ShapeData_Circle_Get)(bcol, ShapeID);
  *Size = SData->Size;
}

void
__ETC_BCOL_PP(UnlinkRecycleOrphanShape_Circle)
(
  __ETC_BCOL_P(t) *bcol,
  __ETC_BCOL_P(ShapeID_t) ShapeID
){
  bcol->ShapeList_Circle.unlrec(ShapeID);
}
