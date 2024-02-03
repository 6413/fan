void CPC_Circle_Square(
  _vf p0,
  _f p0Size,
  _vf p1,
  _f p1Size,
  _vf *op0,
  _vf *oDirection
){
  _vf p0_p1 = p0 - p1;
  _vf dirsign = (p0_p1 * 9999999).clamp(_f(-1), _f(+1));
  _vf outdir = (p0_p1.abs() - p1Size).max(_vf(0));
  *op0 = p0 + (outdir * (p0Size / outdir.length()) - outdir) * dirsign;
  *oDirection = (outdir / p0Size * dirsign).normalize();
}
