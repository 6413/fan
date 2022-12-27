void
__ETC_BCOL_PP(AddShapeToObject)
(
  __ETC_BCOL_P(t) *bcol,
  __ETC_BCOL_P(ObjectID_t) ObjectID,
  uint8_t ShapeEnum,
  __ETC_BCOL_P(ShapeID_t) ShapeID
){
  __ETC_BCOL_PP(ObjectData_t) *ObjectData = __ETC_BCOL_PP(GetObjectData)(bcol, ObjectID);
  VEC_handle(&ObjectData->ShapeList);
  __ETC_BCOL_PP(ShapeData_t) *ShapeData =
    &((__ETC_BCOL_PP(ShapeData_t) *)ObjectData->ShapeList.ptr)[ObjectData->ShapeList.Current];
  ShapeData->ShapeEnum = ShapeEnum;
  ShapeData->ShapeID = ShapeID;
  ObjectData->ShapeList.Current++;
}

#include _WITCH_PATH(ETC/BCOL/internal/Shape/Circle/Circle.h)
#include _WITCH_PATH(ETC/BCOL/internal/Shape/Square/Square.h)
#include _WITCH_PATH(ETC/BCOL/internal/Shape/Rectangle/Rectangle.h)
