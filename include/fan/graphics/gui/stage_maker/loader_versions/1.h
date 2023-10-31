fan::string in;
fan::io::file::read(filename, &in);
uint64_t off = 0;
uint32_t version = fan::read_data<uint32_t>(in, off);
if (version != current_version) {
  fan::print_format("invalid file version, file:{}, current:{}", version, current_version);
  return;
}
fan::mp_t<current_version_t::shapes_t> shapes;
while (off != in.size()) {
  bool ignore = true;
  uint32_t byte_count = 0;
  uint16_t shape_type = fan::read_data<uint16_t>(in, off);
  byte_count = fan::read_data<uint32_t>(in, off);
  shapes.iterate([&]<auto i0, typename T>(T & v0) {
    if (!(shape_type == loco_t::shape_type_t::rectangle && T::shape_type == loco_t::shape_type_t::mark)) {
      if (shape_type != T::shape_type) {
        return;
      }
    }

    ignore = false;


    #if !defined(stage_maker_loader)  && !defined(model_maker_loader)
    auto it = shape_list.NewNodeLast();
    #endif

    fan::mp_t<T> shape;
    shape.iterate([&]<auto i, typename T2>(T2 & v) {
      v = fan::read_data<T2>(in, off);
    });


  #if defined(stage_maker_loader) 
    auto it = stage->stage_common.cid_list.NewNodeLast();
    stage->stage_common.cid_list[it] = shape.get_shape(texturepack);
    fan::string string_type;
    switch (stage->stage_common.cid_list[it]->shape_type) {
      case loco_t::shape_type_t::rectangle: {
        string_type = "rectangle_";
        break;
      }
      case loco_t::shape_type_t::sprite: {
        string_type = "sprite_";
        break;
      }
      case loco_t::shape_type_t::button: {
        string_type = "button_";
        break;
      }
    }
    cid_map[std::make_pair(stage, string_type + shape.id)] = it;
  #elif defined(model_maker_loader)
    lambda(shape);
  #else
    shape_list[it] = shape.get_shape(this);
  #endif
  });
  // if shape is not part of version
  if (ignore) {
    off += byte_count;
  }
}

#undef stage_maker_loader
#undef model_maker_loader