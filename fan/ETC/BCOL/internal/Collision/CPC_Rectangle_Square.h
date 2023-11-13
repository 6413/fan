void CPC_Rectangle_Square(
  _vf p0,
  _vf p0Size,
  _vf p1,
  _f p1Size,
  _vf *op0,
  _vf *oDirection
){
  _vf Sp0 = p0 - p1;
  _vf SPp0 = Sp0.abs();
  _vf Diff = SPp0 / (p0Size + p1Size);
  if(Diff.y > Diff.x){
    op0->y = p1.y + fan::math::copysign(p0Size.y + p1Size, Sp0.y);
    op0->x = p0.x;
    oDirection->y = fan::math::copysign(1, Sp0.y);
    oDirection->x = 0;
  }
  else{
    op0->y = p0.y;
    op0->x = p1.x + fan::math::copysign(p0Size.x + p1Size, Sp0.x);
    oDirection->y = 0;
    oDirection->x = fan::math::copysign(1, Sp0.x);
  }
}
