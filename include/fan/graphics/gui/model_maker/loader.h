struct model_loader_t {

  struct private_ {
    #define fgm_build_model_maker
    #include _FAN_PATH(graphics/gui/fgm/common.h)
    #undef fgm_build_model_maker
  };

  using sprite_t = private_::stage_maker_shape_format::shape_sprite_t;
  using mark_t = private_::stage_maker_shape_format::shape_mark_t;

  void load(loco_t::texturepack_t* tp, const fan::string& path, auto lambda) {
    fan::string f;

    if (!fan::io::file::exists(path)) {
      return;
    }
    fan::io::file::read(path, &f);

    if (f.empty()) {
      return;
    }

    struct iterator_t {
      fan_masterpiece_make(
        (private_::stage_maker_shape_format::shape_sprite_t)sprite,
        (private_::stage_maker_shape_format::shape_mark_t)mark
      );
    }iterator;

    uint64_t offset = 0;
    // read header
    uint32_t header = fan::read_data<uint32_t>(f, offset);
    iterator.iterate_masterpiece([&](auto& d) {
      // read the type
      auto type = fan::read_data<loco_t::shape_type_t::_t>(f, offset);
      uint32_t instance_count = fan::read_data<uint32_t>(f, offset);
      for (uint32_t i = 0; i < instance_count; ++i) {
        d.iterate_masterpiece([&](auto& o) {
          o = fan::read_data<std::remove_reference_t<decltype(o)>>(f, offset);
        });
        lambda(d);
      }
    });
  }
};