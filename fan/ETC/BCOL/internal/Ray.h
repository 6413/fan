#if BCOL_set_SupportGrid == 1 && BCOL_set_VisualSolve == 1
  VisualSolve_t Ray(
    _vf p, /* position */
    _vf d /* direction */
  ){
    for(uint32_t i = 0; i < _vf::size(); i++){
      if(d[i] == 0){
        d[i] = 0.00001;
      }
    }

    p /= GridBlockSize;
    _vf at = p;
    _vsi32 gi;
    for(uint32_t d = 0; d < _vf::size(); d++){
      gi[d] = at[d] + (at[d] < _f(0) ? _f(-1) : _f(0));
    }
    _vf r = at - gi;
    while(1){
      {
        bool Contact;
        BCOL_set_VisualSolve_GridContact
        if(Contact == true){
          VisualSolve_t VisualSolve;
          this->VisualSolve_Grid_cb(
            this,
            gi,
            p,
            at,
            &VisualSolve);
          return VisualSolve;
        }
      }

      _vf left;
      #if 0
        for(uint32_t i = 0; i < _vf::size(); i++){
          if(d[i] > 0){
            left[i] = f32_t(1) - r[i];
          }
          else{
            left[i] = r[i];
          }
        }
        _vf multiplers = left / d.abs();
      #elif 1
        left = ((d * 9999999).clamp(_f(0), _f(1)) - r).abs();
        _vf multiplers = left / d.abs();
      #elif 0
        left = (d * 9999999).clamp(_f(-0.0000001), _f(1)) - r;
        _vf multiplers = left / d;
      #endif

      f32_t min_multipler = multiplers.min();
      for(uint32_t i = 0; i < _vf::size(); i++){
        if(multiplers[i] == min_multipler){
          gi[i] += copysign((sint32_t)1, d[i]);
          r[i] -= copysign((f32_t)1, d[i]);
        }
      }
      _vf min_dir = d * min_multipler;
      at += min_dir;
      r += min_dir;
    }
  }
#endif
