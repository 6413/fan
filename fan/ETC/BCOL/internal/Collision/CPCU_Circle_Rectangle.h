struct CPCU_Circle_Rectangle_t{
  _vf Sp0;
  uint8_t Stage;
  union{
    struct{
      _vf CircleOffset;
    }Corner;
    struct{
      _vf SPp0;
    }Side;
  }StageData;
};

bool CPCU_Circle_Rectangle_IsThereCollision(
  CPCU_Circle_Rectangle_t *Data
){
  return Data->Stage != (uint8_t)-1;
}

void CPCU_Circle_Rectangle_Pre(
  _vf p0,
  _f p0Size,
  _vf p1,
  _vf p1Size,
  CPCU_Circle_Rectangle_t *Data
){
  Data->Sp0 = p0 - p1;
  _vf SPp0 = Data->Sp0.abs();
  if(SPp0.x > p1Size.x && SPp0.y > p1Size.y){
    _vf Corner = SPp0 - p1Size;
    _f Divider = fan::math::sqrt(Corner.y * Corner.y + Corner.x * Corner.x);
    Data->StageData.Corner.CircleOffset = Corner / Divider;
    _vf DD = Corner - Data->StageData.Corner.CircleOffset * p0Size;
    if(DD.y > 0 || DD.x > 0){
      Data->Stage = (uint8_t)-1;
    }
    else{
      Data->Stage = 0;
    }
  }
  else{
    Data->StageData.Side.SPp0 = SPp0;
    _vf CombinedSize = p1Size + p0Size;
    if(SPp0.y > CombinedSize.y || SPp0.x > CombinedSize.x){
      Data->Stage = (uint8_t)-1;
    }
    else{
      Data->Stage = 1;
    }
  }
}

void CPCU_Circle_Rectangle_Solve(
  _vf p0,
  _f p0Size,
  _vf p1,
  _vf p1Size,
  CPCU_Circle_Rectangle_t *Data,
  _vf *op0,
  _vf *oDirection
){
  switch(Data->Stage){
    case 0:{
      *op0 = p1 + fan::vec2(p1Size + Data->StageData.Corner.CircleOffset * p0Size).copysign(Data->Sp0);
      *oDirection = fan::vec2(Data->StageData.Corner.CircleOffset).copysign(Data->Sp0);
      break;
    }
    case 1:{
      if(Data->StageData.Side.SPp0.x <= p1Size.x){
        op0->y = p1.y + fan::math::copysign(p1Size.y + p0Size, Data->Sp0.y);
        op0->x = p0.x;
        oDirection->y = fan::math::copysign(1, Data->Sp0.y);
        oDirection->x = 0;
      }
      else if(Data->StageData.Side.SPp0.y <= p1Size.y){
        op0->y = p0.y;
        op0->x = p1.x + fan::math::copysign(p1Size.x + p0Size, Data->Sp0.x);
        oDirection->y = 0;
        oDirection->x = fan::math::copysign(1, Data->Sp0.x);
      }
      break;
    }
  }
}
