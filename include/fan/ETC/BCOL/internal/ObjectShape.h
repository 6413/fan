void
__ETC_BCOL_P(RecycleObject)
(
  __ETC_BCOL_P(t) *bcol,
  __ETC_BCOL_P(ObjectID_t) ObjectID
){
  auto ObjectData = &bcol->ObjectList[ObjectID];
  for(uint32_t i = 0; i < ObjectData->ShapeList.Current; i++){
    __ETC_BCOL_PP(ShapeData_t) *ShapeData = &((__ETC_BCOL_PP(ShapeData_t *))ObjectData->ShapeList.ptr)[i];
    switch(ShapeData->ShapeEnum){
      case __ETC_BCOL_P(ShapeEnum_Circle):{
        __ETC_BCOL_PP(UnlinkRecycleOrphanShape_Circle)(bcol, ShapeData->ShapeID);
        break;
      }
      case __ETC_BCOL_P(ShapeEnum_Square):{
        __ETC_BCOL_PP(UnlinkRecycleOrphanShape_Square)(bcol, ShapeData->ShapeID);
        break;
      }
      case __ETC_BCOL_P(ShapeEnum_Rectangle):{
        __ETC_BCOL_PP(UnlinkRecycleOrphanShape_Rectangle)(bcol, ShapeData->ShapeID);
        break;
      }
    }
  }
  VEC_free(&ObjectData->ShapeList);
  bcol->ObjectList.Recycle(ObjectID);
}
