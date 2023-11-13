void AddShapeToObject(
  ObjectID_t ObjectID,
  ShapeEnum_t ShapeEnum,
  ShapeID_t ShapeID
){
  auto ObjectData = this->GetObjectData(ObjectID);
  ShapeList_AddEmpty(&ObjectData->ShapeList, 1);
  auto ShapeData = &ObjectData->ShapeList.ptr[ObjectData->ShapeList.Current - 1];
  ShapeData->ShapeEnum = ShapeEnum;
  ShapeData->ShapeID = ShapeID;
}

#include "Circle/Circle.h"
#include "Rectangle/Rectangle.h"
