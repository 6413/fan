struct CompiledShapes_Circle_t{
  _vf Position;
  _f Size;
};
struct CompiledShapes_Rectangle_t{
  _vf Position;
  _vf Size;
};

#define BVEC_set_prefix uint8Vector
#define BVEC_set_NodeType uint32_t
#define BVEC_set_NodeData uint8_t
#include ETC_BCOL_Include(BVEC/BVEC.h)
struct CompiledShapes_t{
  uint8Vector_t ShapeData;
};

void CompiledShapes_close(
  CompiledShapes_t *CompiledShapes
){
  uint8Vector_Close(&CompiledShapes->ShapeData);
}
void CompiledShapes_open(
  CompiledShapes_t *CompiledShapes
){
  uint8Vector_Open(&CompiledShapes->ShapeData);
}

void CompiledShapes_Write_Circle(
  CompiledShapes_t *CompiledShapes,
  CompiledShapes_Circle_t *Circle
){
  uintptr_t co = CompiledShapes->ShapeData.Current;
  uint8Vector_AddEmpty(&CompiledShapes->ShapeData, sizeof(uint8_t) + sizeof(CompiledShapes_Circle_t));
  *(ShapeEnum_t *)&CompiledShapes->ShapeData.ptr[co] = ShapeEnum_t::Circle;
  co += sizeof(ShapeEnum_t);
  *(CompiledShapes_Circle_t *)&CompiledShapes->ShapeData.ptr[co] = *Circle;
}
void CompiledShapes_Write_Rectangle(
  CompiledShapes_t *CompiledShapes,
  CompiledShapes_Rectangle_t *Rectangle
){
  uintptr_t co = CompiledShapes->ShapeData.Current;
  uint8Vector_AddEmpty(&CompiledShapes->ShapeData, sizeof(uint8_t) + sizeof(CompiledShapes_Rectangle_t));
  *(ShapeEnum_t *)&CompiledShapes->ShapeData.ptr[co] = ShapeEnum_t::Rectangle;
  co += sizeof(ShapeEnum_t);
  *(CompiledShapes_Rectangle_t *)&CompiledShapes->ShapeData.ptr[co] = *Rectangle;
}

void CompiledShapes_ToObject(
  CompiledShapes_t *CompiledShapes,
  ObjectID_t ObjectID
){
  for(uintptr_t i = 0; i < CompiledShapes->ShapeData.Current;){
    auto Type = *(ShapeEnum_t *)&CompiledShapes->ShapeData.ptr[i];
    i += sizeof(ShapeEnum_t);
    switch(Type){
      case ShapeEnum_t::Circle:{
        auto s = (CompiledShapes_Circle_t *)&CompiledShapes->ShapeData.ptr[i];
        ShapeProperties_Circle_t sp;
        sp.Position = s->Position;
        sp.Size = s->Size;
        auto ShapeID = this->NewShape_Circle(ObjectID, &sp);
        i += sizeof(CompiledShapes_Circle_t);
        break;
      }
      case ShapeEnum_t::Rectangle:{
        auto s = (CompiledShapes_Rectangle_t *)&CompiledShapes->ShapeData.ptr[i];
        ShapeProperties_Rectangle_t sp;
        sp.Position = s->Position;
        sp.Size = s->Size;
        auto ShapeID = this->NewShape_Rectangle(ObjectID, &sp);
        i += sizeof(CompiledShapes_Rectangle_t);
        break;
      }
    }
  }
}
