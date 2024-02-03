#if BCOL_set_VisualSolve == 1
  /* this function is thread safe */
  VisualSolve_t Ray(
    _vf position,
    _vf direction
  ){
    for(uint32_t i = 0; i < _dc; i++){
      if(direction[i] == 0){
        direction[i] = 0.00001;
      }
    }

    #if BCOL_set_SupportGrid == 1
      struct{
        _vf np; /* normalized position */
        _vf at;
        _vsi32 gi;
      }grid_result;
      grid_result.np = position / GridBlockSize;
      grid_result.at = grid_result.np;
      for(uint32_t d = 0; d < _dc; d++){
        grid_result.gi[d] = grid_result.at[d] + (grid_result.at[d] < _f(0) ? _f(-1) : _f(0));
      }

      _vf r = grid_result.at - grid_result.gi;
      while(1){
        {
          bool Contact;
          BCOL_set_VisualSolve_GridContact
          if((position - grid_result.at * GridBlockSize).length() < BCOL_set_VisualSolve_dmin){
            Contact = false;
          }
          if(Contact == true){
            break;
          }
        }

        _vf left;
        #if 0
          for(uint32_t i = 0; i < _dc; i++){
            if(direction[i] > 0){
              left[i] = f32_t(1) - r[i];
            }
            else{
              left[i] = r[i];
            }
          }
          _vf multiplers = left / direction.abs();
        #elif 1
          left = ((direction * 9999999).clamp(_f(0), _f(1)) - r).abs();
          _vf multiplers = left / direction.abs();
        #elif 0
          left = (direction * 9999999).clamp(_f(-0.0000001), _f(1)) - r;
          _vf multiplers = left / direction;
        #endif

        f32_t min_multipler = multiplers.min();
        for(uint32_t i = 0; i < _dc; i++){
          if(multiplers[i] == min_multipler){
            grid_result.gi[i] += copysign((sint32_t)1, direction[i]);
            r[i] -= copysign((f32_t)1, direction[i]);
          }
        }
        _vf min_dir = direction * min_multipler;
        grid_result.at += min_dir;
        r += min_dir;
      }
    #endif

    struct{
      ShapeInfoPack_t sip;
      _vf intersection_pos;
    }closest_shape;
    closest_shape.sip.ObjectID.sic();
    closest_shape.intersection_pos = position + 999999999;

    {
      ShapeInfoPack_t sip;
      sip.ObjectID = this->ObjectList.GetNodeFirst();
      while(sip.ObjectID != this->ObjectList.dst){
        auto ObjectData = this->GetObjectData(sip.ObjectID);

        for(sip.ShapeID.ID = 0; sip.ShapeID.ID < ObjectData->ShapeList.Current; sip.ShapeID.ID++){
          sip.ShapeEnum = ObjectData->ShapeList.ptr[sip.ShapeID.ID].ShapeEnum;
          switch(sip.ShapeEnum){
            case ShapeEnum_t::Circle:{
              auto CircleData = ShapeData_Circle_Get(ObjectData->ShapeList.ptr[sip.ShapeID.ID].ShapeID);
              auto circle_pos = ObjectData->Position + CircleData->Position;
              _vf intersection_pos;
              if(ray_circle_intersection(position, direction, circle_pos, CircleData->Size, intersection_pos) == false){
                break;
              }
              if((intersection_pos - position).length() < BCOL_set_VisualSolve_dmin){
                break;
              }
              if((intersection_pos - position).length() < (closest_shape.intersection_pos - position).length()){
                closest_shape.sip = sip;
                closest_shape.intersection_pos = intersection_pos;
              }
            }
            case ShapeEnum_t::Rectangle:{
              break;
            }
          }
        }

        sip.ObjectID = sip.ObjectID.Next(&this->ObjectList);
      }
    }

    VisualSolve_t VisualSolve;

    #if BCOL_set_SupportGrid == 1
      if((grid_result.at * GridBlockSize - position).length() < (closest_shape.intersection_pos - position).length()){
        this->VisualSolve_Grid_cb(
          this,
          grid_result.gi,
          grid_result.np,
          grid_result.at,
          &VisualSolve);
      }
      else
    #endif
    {
      if(closest_shape.sip.ObjectID.iic()){
        return VisualSolve_t(0);
      }
      VisualSolve_Shape_cb(this, &closest_shape.sip, position, closest_shape.intersection_pos, &VisualSolve);
    }

    return VisualSolve;
  }
#endif
