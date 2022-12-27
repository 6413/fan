void
__ETC_BCOL_PP(CPC_Circle_Square)
(
  f32_t p0Y,
  f32_t p0X,
  f32_t p0Size,
  f32_t p1Y,
  f32_t p1X,
  f32_t p1Size,
  f32_t *op0Y,
  f32_t *op0X,
  f32_t *oDirectionY,
  f32_t *oDirectionX
){
  f32_t Sp0Y = p0Y - p1Y;
  f32_t Sp0X = p0X - p1X;
  f32_t SPp0Y = __absf(Sp0Y);
  f32_t SPp0X = __absf(Sp0X);
  if(SPp0X > p1Size && SPp0Y > p1Size){
    f32_t CornerY = SPp0Y - p1Size;
    f32_t CornerX = SPp0X - p1Size;
    f32_t Divider = __sqrt(CornerY * CornerY + CornerX * CornerX);
    f32_t CircleOffsetY = CornerY / Divider;
    f32_t CircleOffsetX = CornerX / Divider;
    *op0Y = p1Y + __copysignf(p1Size + CircleOffsetY * p0Size, Sp0Y);
    *op0X = p1X + __copysignf(p1Size + CircleOffsetX * p0Size, Sp0X);
    *oDirectionY = __copysignf(CircleOffsetY, Sp0Y);
    *oDirectionX = __copysignf(CircleOffsetX, Sp0X);
  }
  else{
    if(SPp0X <= p1Size){
      *op0Y = p1Y + __copysignf(p1Size + p0Size, Sp0Y);
      *op0X = p0X;
      *oDirectionY = __copysignf(1, Sp0Y);
      *oDirectionX = 0;
    }
    else if(SPp0Y <= p1Size){
      *op0Y = p0Y;
      *op0X = p1X + __copysignf(p1Size + p0Size, Sp0X);
      *oDirectionY = 0;
      *oDirectionX = __copysignf(1, Sp0X);
    }
  }
}
