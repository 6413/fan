#pragma pack(push, 1)

struct ImportHM_Shape_Rectangle_t{
  _vf Position;
  _vf Size;
};
struct ImportHM_Shape_Circle_t{
  _vf Position;
  _f Size;
};

#pragma pack(pop)

enum class ImportHM_ErrorEnum{
  OK,
  NotSupportedShape
};

ImportHM_ErrorEnum ImportHM(
  void *Data,
  uintptr_t Size,
  CompiledShapes_t *CompiledShapes
){
  uintptr_t Offset = 0;

  {
    uint32_t TriangleCount = *(uint32_t *)&((uint8_t *)Data)[Offset];
    if(TriangleCount != 0){
      /* not supported yet */
      return ImportHM_ErrorEnum::NotSupportedShape;
    }
    Offset += sizeof(uint32_t);
  }
  {
    uint32_t RectangleCount = *(uint32_t *)&((uint8_t *)Data)[Offset];
    Offset += sizeof(uint32_t);
    for(uint32_t i = 0; i < RectangleCount; i++){
      auto s = (ImportHM_Shape_Rectangle_t *)
        &((uint8_t *)Data)[Offset + i * sizeof(ImportHM_Shape_Rectangle_t)];
      CompiledShapes_Rectangle_t r;
      r.Position = s->Position;
      r.Size = s->Size;
      this->CompiledShapes_Write_Rectangle(CompiledShapes, &r);
    }
    Offset += RectangleCount * sizeof(ImportHM_Shape_Rectangle_t);
  }
  {
    uint32_t CircleCount = *(uint32_t *)&((uint8_t *)Data)[Offset];
    Offset += sizeof(uint32_t);
    for(uint32_t i = 0; i < CircleCount; i++){
      auto s = (ImportHM_Shape_Circle_t *)
        &((uint8_t *)Data)[Offset + i * sizeof(ImportHM_Shape_Circle_t)];
      CompiledShapes_Circle_t c;
      c.Position = s->Position;
      c.Size = s->Size;
      this->CompiledShapes_Write_Circle(CompiledShapes, &c);
    }
    Offset += CircleCount * sizeof(ImportHM_Shape_Circle_t);
  }
  return ImportHM_ErrorEnum::OK;
}
