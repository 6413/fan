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
    if (!((loco_t::shape_type_t)shape_type == loco_t::shape_type_t::rectangle && T::shape_type == loco_t::shape_type_t::mark)) {
      if ((loco_t::shape_type_t)shape_type != T::shape_type) {
        return;
      }
    }

    ignore = false;


    #if !defined(tile_map_editor_loader)
    auto it = shape_list.NewNodeLast();
    #endif

    fan::mp_t<T> shape;
    shape.iterate([&]<auto i, typename T2>(T2 & v) {
      v = fan::read_data<T2>(in, off);
    });

    #if defined(tile_map_editor_loader) 

    #else
    shape_list[it] = shape.get_shape(this);
    #endif
  });
  // if shape is not part of version
  if (ignore) {
    off += byte_count;
  }
}

#undef tile_map_editor_loader