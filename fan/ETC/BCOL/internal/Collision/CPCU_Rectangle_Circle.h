struct CPCU_Rectangle_Circle_t{
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

bool CPCU_Rectangle_Circle_IsThereCollision(
  CPCU_Rectangle_Circle_t *Data
){
  return Data->Stage != (uint8_t)-1;
}

void CPCU_Rectangle_Circle_Pre(
  _vf p0,
  _vf p0Size,
  _vf p1,
  _f p1Size,
  CPCU_Rectangle_Circle_t *Data
){
  Data->Sp0 = p0 - p1;
  _vf SPp0 = Data->Sp0.abs();
  if(SPp0.x > p0Size.x && SPp0.y > p0Size.y){
    _vf Corner = SPp0 - p0Size;
    _f Divider = Corner.length();
    Data->StageData.Corner.CircleOffset = Corner / Divider;
    _vf DD = Corner - Data->StageData.Corner.CircleOffset * p1Size;
    if(DD.y > 0 || DD.x > 0){
      Data->Stage = (uint8_t)-1;
    }
    else{
      Data->Stage = 0;
    }
  }
  else{
    Data->StageData.Side.SPp0 = SPp0;
    _vf CombinedSize = p0Size + p1Size;
    if(SPp0.y > CombinedSize.y || SPp0.x > CombinedSize.x){
      Data->Stage = (uint8_t)-1;
    }
    else{
      Data->Stage = 1;
    }
  }
}

void CPCU_Rectangle_Circle_Solve(
  _vf p0,
  _vf p0Size,
  _vf p1,
  _f p1Size,
  CPCU_Rectangle_Circle_t *Data,
  _vf *op0,
  _vf *oDirection
){
  switch(Data->Stage){
    case 0:{
      *op0 = p1 + fan::vec2(p0Size + Data->StageData.Corner.CircleOffset * p1Size).copysign(Data->Sp0);
      *oDirection = fan::vec2(Data->StageData.Corner.CircleOffset).copysign(Data->Sp0);
      break;
    }
    case 1:{
      if(Data->StageData.Side.SPp0.x <= p0Size.x){
        op0->y = p1.y + fan::math::copysign(p0Size.y + p1Size, Data->Sp0.y);
        op0->x = p0.x;
        oDirection->y = fan::math::copysign(1, Data->Sp0.y);
        oDirection->x = 0;
      }
      else if(Data->StageData.Side.SPp0.y <= p0Size.y){
        op0->y = p0.y;
        op0->x = p1.x + fan::math::copysign(p0Size.x + p1Size, Data->Sp0.x);
        oDirection->y = 0;
        oDirection->x = fan::math::copysign(1, Data->Sp0.x);
      }
      break;
    }
  }
}
