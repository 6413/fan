#if BCOL_set_SupportGrid == 1
  _vsi32 Ray(
    _vf p, /* position */
    _vf d, /* direction */
    _vf *at
  ){
    for(uint32_t i = 0; i < _vf::size(); i++){
      if(d[i] == 0){
        d[i] = 0.00001;
      }
    }

    *at = p / GridBlockSize;
    _vsi32 gi = *at;
    _vf r = *at - _vsi32(*at);
    while(1){
      Contact_Grid_t Contact;
      this->PreSolveUnknown_Grid_cb(
        this,
        gi,
        &Contact);
      if(Contact.Flag & Contact_Grid_Flag::EnableContact){
        *at *= GridBlockSize;
        return gi;
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
      *at += min_dir;
      r += min_dir;
      gi += iterate_result;
      r -= iterate_result;
    }
  }
#endif