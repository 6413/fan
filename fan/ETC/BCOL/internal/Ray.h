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

    _vf at = p / GridBlockSize;
    _vsi32 gi = at;
    _vf r = at - _vsi32(at);
    while(1){
      Contact_Grid_t Contact;
      VisualSolve_t VisualSolve;
      this->VisualSolve_Grid_cb(
        this,
        gi,
        at,
        &Contact,
        &VisualSolve);
      if(Contact.Flag & Contact_Grid_Flag::EnableContact){
        return VisualSolve;
      };

      _vf left;
      for(uint32_t i = 0; i < _vf::size(); i++){
        if(d[i] > 0){
          left[i] = f32_t(1) - r[i];
        }
        else{
          left[i] = r[i];
        }
      }
      _vf multiplers = left / d;

      f32_t min_multipler = multiplers.abs().min();
      _vsi32 iterate_result = _vsi32(0);
      for(uint32_t i = 0; i < _vf::size(); i++){
        if(std::abs(multiplers[i]) == min_multipler){
          iterate_result[i] = copysign(1, multiplers[i]);
        }
      }
      _vf min_dir = d * min_multipler;
      at += min_dir;
      r += min_dir;
      gi += iterate_result;
      r -= iterate_result;
    }
  }
#endif