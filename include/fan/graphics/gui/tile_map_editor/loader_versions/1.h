fan::string in;
if (fan::io::file::read(filename, &in)) {
  #if defined(tile_map_editor_loader) 
    fan::throw_error_format(
  #else
    fan::print_format(
  #endif
  "failed to load file, file:{}", filename);
  #if !defined(tile_map_editor_loader) 
  return;
  #endif
}
uint64_t off = 0;
uint32_t version = fan::read_data<uint32_t>(in, off);
if (version != current_version) {
  #if defined(tile_map_editor_loader) 
    fan::throw_error_format(
  #else
    fan::print_format(
  #endif
  "invalid file version, file version:{}, current:{}", version, current_version);
  #if !defined(tile_map_editor_loader) 
  return;
  #endif
}
map_size = fan::read_data<fan::vec2ui>(in, off);
tile_size = fan::read_data<fan::vec2ui>(in, off);

#if !defined(tile_map_editor_loader) 
map_tiles.resize(map_size.y);
for (auto& i : map_tiles) {
  i.resize(map_size.x);
}
#endif

fan::mp_t<current_version_t::shapes_t> shapes;
while (off != in.size()) {
  bool ignore = true;
  shapes.iterate([&]<auto i0, typename T>(T & v0) {

    fan::mp_t<T> shape;
    shape.iterate([&]<auto i, typename T2>(T2 & v) {
      uint32_t byte_count = 0;
      byte_count = fan::read_data<uint32_t>(in, off);

      if constexpr (fan_requires_rule(T2, typename T2::value_type)) {
        if constexpr (std::is_same_v<T2, std::vector<T2::value_type>>) {
          uint32_t element_count = fan::read_data<uint32_t>(in, off);
          for (int k = 0; k < element_count; ++k) {
            v.push_back(fan::read_data<T2::value_type>(in, off));
          }
        }
      }
      else {
        v = fan::read_data<T2>(in, off);
      }
    });

    #if defined(tile_map_editor_loader) 
    v0 = shape;
    #else
    shape.get_shape(this);
    #endif
  });
  #if defined(tile_map_editor_loader)
  compiled_map.compiled_shapes.push_back(shapes);
  #endif
}


#undef tile_map_editor_loader