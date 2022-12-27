#pragma pack(push, 1)

typedef struct{
  f32_t PositionY;
  f32_t PositionX;
  f32_t SizeY;
  f32_t SizeX;
}__ETC_BCOL_PP(ImportHM_Shape_Rectangle_t);
typedef struct{
  f32_t PositionY;
  f32_t PositionX;
  f32_t Size;
}__ETC_BCOL_PP(ImportHM_Shape_Circle_t);

#pragma pack(pop)

enum{
  __ETC_BCOL_P(ImportHM_ErrorEnum_OK) = 0
};

sint32_t
__ETC_BCOL_P(ImportHM)
(
  __ETC_BCOL_P(t) *bcol,
  void *Data,
  uintptr_t Size,
  __ETC_BCOL_P(CompiledShapes_t) *CompiledShapes
){
  uintptr_t Offset = 0;

  {
    uint32_t TriangleCount = *(uint32_t *)&((uint8_t *)Data)[Offset];
    if(TriangleCount != 0){
      /* not supported yet */
      return 0;
    }
    Offset += sizeof(uint32_t);
  }
  {
    uint32_t RectangleCount = *(uint32_t *)&((uint8_t *)Data)[Offset];
    Offset += sizeof(uint32_t);
    for(uint32_t i = 0; i < RectangleCount; i++){
      __ETC_BCOL_PP(ImportHM_Shape_Rectangle_t) *s =
        (__ETC_BCOL_PP(ImportHM_Shape_Rectangle_t) *)
        &((uint8_t *)Data)[Offset + i * sizeof(__ETC_BCOL_PP(ImportHM_Shape_Rectangle_t))];
      __ETC_BCOL_PP(CompiledShapes_Rectangle_t) r;
      r.PositionY = s->PositionY;
      r.PositionX = s->PositionX;
      r.SizeY = s->SizeY;
      r.SizeX = s->SizeX;
      __ETC_BCOL_P(CompiledShapes_Write_Rectangle)(bcol, CompiledShapes, &r);
    }
    Offset += RectangleCount * sizeof(__ETC_BCOL_PP(ImportHM_Shape_Rectangle_t));
  }
  {
    uint32_t CircleCount = *(uint32_t *)&((uint8_t *)Data)[Offset];
    Offset += sizeof(uint32_t);
    for(uint32_t i = 0; i < CircleCount; i++){
      __ETC_BCOL_PP(ImportHM_Shape_Circle_t) *s =
        (__ETC_BCOL_PP(ImportHM_Shape_Circle_t) *)
        &((uint8_t *)Data)[Offset + i * sizeof(__ETC_BCOL_PP(ImportHM_Shape_Circle_t))];
      __ETC_BCOL_PP(CompiledShapes_Circle_t) c;
      c.PositionY = s->PositionY;
      c.PositionX = s->PositionX;
      c.Size = s->Size;
      __ETC_BCOL_P(CompiledShapes_Write_Circle)(bcol, CompiledShapes, &c);
    }
    Offset += CircleCount * sizeof(__ETC_BCOL_PP(ImportHM_Shape_Circle_t));
  }
  return __ETC_BCOL_P(ImportHM_ErrorEnum_OK);
}
