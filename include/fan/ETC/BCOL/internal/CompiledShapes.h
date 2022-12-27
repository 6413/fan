typedef struct{
  __pfloat PositionY;
  __pfloat PositionX;
  __pfloat Size;
}__ETC_BCOL_PP(CompiledShapes_Circle_t);
typedef struct{
  __pfloat PositionY;
  __pfloat PositionX;
  __pfloat Size;
}__ETC_BCOL_PP(CompiledShapes_Square_t);
typedef struct{
  __pfloat PositionY;
  __pfloat PositionX;
  __pfloat SizeY;
  __pfloat SizeX;
}__ETC_BCOL_PP(CompiledShapes_Rectangle_t);

typedef struct{
  VEC_t ShapeData; /* uint8_t */
}__ETC_BCOL_P(CompiledShapes_t);

void
__ETC_BCOL_P(CompiledShapes_close)(
  __ETC_BCOL_P(t) *bcol,
  __ETC_BCOL_P(CompiledShapes_t) *CompiledShapes
){
  VEC_free(&CompiledShapes->ShapeData);
}
void
__ETC_BCOL_P(CompiledShapes_open)(
  __ETC_BCOL_P(t) *bcol,
  __ETC_BCOL_P(CompiledShapes_t) *CompiledShapes
){
  VEC_init(&CompiledShapes->ShapeData, sizeof(uint8_t), A_resize);
}

void
__ETC_BCOL_P(CompiledShapes_Write_Circle)(
  __ETC_BCOL_P(t) *bcol,
  __ETC_BCOL_P(CompiledShapes_t) *CompiledShapes,
  __ETC_BCOL_PP(CompiledShapes_Circle_t) *Circle
){
  uintptr_t co = CompiledShapes->ShapeData.Current;
  VEC_handle0(&CompiledShapes->ShapeData, sizeof(uint8_t) + sizeof(__ETC_BCOL_PP(CompiledShapes_Circle_t)));
  ((uint8_t *)CompiledShapes->ShapeData.ptr)[co] = __ETC_BCOL_P(ShapeEnum_Circle);
  co += sizeof(uint8_t);
  __ETC_BCOL_PP(CompiledShapes_Circle_t *) c =
    (__ETC_BCOL_PP(CompiledShapes_Circle_t) *)&((uint8_t *)CompiledShapes->ShapeData.ptr)[co];
  *c = *Circle;
}
void
__ETC_BCOL_P(CompiledShapes_Write_Square)(
  __ETC_BCOL_P(t) *bcol,
  __ETC_BCOL_P(CompiledShapes_t) *CompiledShapes,
  __ETC_BCOL_PP(CompiledShapes_Square_t) *Square
){
  uintptr_t co = CompiledShapes->ShapeData.Current;
  VEC_handle0(&CompiledShapes->ShapeData, sizeof(uint8_t) + sizeof(__ETC_BCOL_PP(CompiledShapes_Square_t)));
  ((uint8_t *)CompiledShapes->ShapeData.ptr)[co] = __ETC_BCOL_P(ShapeEnum_Square);
  co += sizeof(uint8_t);
  __ETC_BCOL_PP(CompiledShapes_Square_t *) s =
    (__ETC_BCOL_PP(CompiledShapes_Square_t) *)&((uint8_t *)CompiledShapes->ShapeData.ptr)[co];
  *s = *Square;
}
void
__ETC_BCOL_P(CompiledShapes_Write_Rectangle)(
  __ETC_BCOL_P(t) *bcol,
  __ETC_BCOL_P(CompiledShapes_t) *CompiledShapes,
  __ETC_BCOL_PP(CompiledShapes_Rectangle_t) *Rectangle
){
  uintptr_t co = CompiledShapes->ShapeData.Current;
  VEC_handle0(&CompiledShapes->ShapeData, sizeof(uint8_t) + sizeof(__ETC_BCOL_PP(CompiledShapes_Rectangle_t)));
  ((uint8_t *)CompiledShapes->ShapeData.ptr)[co] = __ETC_BCOL_P(ShapeEnum_Rectangle);
  co += sizeof(uint8_t);
  __ETC_BCOL_PP(CompiledShapes_Rectangle_t *) r =
    (__ETC_BCOL_PP(CompiledShapes_Rectangle_t) *)&((uint8_t *)CompiledShapes->ShapeData.ptr)[co];
  *r = *Rectangle;
}

void
__ETC_BCOL_P(CompiledShapes_ToObject)(
  __ETC_BCOL_P(t) *bcol,
  __ETC_BCOL_P(CompiledShapes_t) *CompiledShapes,
  __ETC_BCOL_P(ObjectID_t) ObjectID
){
  for(uintptr_t i = 0; i < CompiledShapes->ShapeData.Current;){
    uint8_t Type = ((uint8_t *)CompiledShapes->ShapeData.ptr)[i];
    ++i;
    switch(Type){
      case __ETC_BCOL_P(ShapeEnum_Circle):{
        __ETC_BCOL_PP(CompiledShapes_Circle_t *) s =
          (__ETC_BCOL_PP(CompiledShapes_Circle_t) *)&((uint8_t *)CompiledShapes->ShapeData.ptr)[i];
        __ETC_BCOL_P(ShapeProperties_Circle_t) sp;
        sp.PositionY = s->PositionY;
        sp.PositionX = s->PositionX;
        sp.Size = s->Size;
        __ETC_BCOL_P(ShapeID_t) ShapeID = __ETC_BCOL_P(NewShape_Circle)(bcol, ObjectID, &sp);
        i += sizeof(__ETC_BCOL_PP(CompiledShapes_Circle_t));
        break;
      }
      case __ETC_BCOL_P(ShapeEnum_Square):{
        __ETC_BCOL_PP(CompiledShapes_Square_t *) s =
          (__ETC_BCOL_PP(CompiledShapes_Square_t) *)&((uint8_t *)CompiledShapes->ShapeData.ptr)[i];
        __ETC_BCOL_P(ShapeProperties_Square_t) sp;
        sp.PositionY = s->PositionY;
        sp.PositionX = s->PositionX;
        sp.Size = s->Size;
        __ETC_BCOL_P(ShapeID_t) ShapeID = __ETC_BCOL_P(NewShape_Square)(bcol, ObjectID, &sp);
        i += sizeof(__ETC_BCOL_PP(CompiledShapes_Square_t));
        break;
      }
      case __ETC_BCOL_P(ShapeEnum_Rectangle):{
        __ETC_BCOL_PP(CompiledShapes_Rectangle_t *) s =
          (__ETC_BCOL_PP(CompiledShapes_Rectangle_t) *)&((uint8_t *)CompiledShapes->ShapeData.ptr)[i];
        __ETC_BCOL_P(ShapeProperties_Rectangle_t) sp;
        sp.PositionY = s->PositionY;
        sp.PositionX = s->PositionX;
        sp.SizeY = s->SizeY;
        sp.SizeX = s->SizeX;
        __ETC_BCOL_P(ShapeID_t) ShapeID = __ETC_BCOL_P(NewShape_Rectangle)(bcol, ObjectID, &sp);
        i += sizeof(__ETC_BCOL_PP(CompiledShapes_Rectangle_t));
        break;
      }
    }
  }
}
