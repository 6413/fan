struct version_001_t {

  struct sprite_t {

    static constexpr typename loco_t::shape_type_t::_t shape_type = loco_t::shape_type_t::sprite;

    fan::vec3 position;
    fan::vec2 size;
    fan::color color;
    fan::string image_name;

    // global
    fan::string id;
    uint32_t group_id;

  #ifndef stage_maker_loader

    void init(loco_t::shape_t& shape) {
      position = shape.get_position();
      size = shape.get_size();
      color = shape.get_color();
    }

    void from_string(const fan::string& str, uint64_t& off) {
      position = fan::read_data<fan::vec3>(str, off);
      size = fan::read_data<fan::vec3>(str, off);
      color = fan::read_data<fan::color>(str, off);
      position = fan::read_data<fan::vec3>(str, off);
    }

    shapes_t::global_t* get_shape(fgm_t* fgm) {
      fgm_t::shapes_t::global_t* ret = new fgm_t::shapes_t::global_t(
        fgm,
       fan::graphics::sprite_t{{
           .position = position,
           .size = size,
           .color = color
         }}
      );

      ret->shape_data.sprite.image_name = image_name;
      ret->id = id;
      ret->group_id = group_id;

      if (image_name.empty()) {
        return ret;
      }

      loco_t::texturepack_t::ti_t ti;
      if (fgm->texturepack.qti(image_name, &ti)) {
        fan::print_no_space("failed to load texture:", image_name);
      }
      else {
        auto& data = fgm->texturepack.get_pixel_data(ti.pack_id);
        gloco->sprite.load_tp(ret->children[0], &ti);
      }

      return ret;
    }
  #else
    loco_t::shape_t get_shape(loco_t::texturepack_t* tp) {
      fan::graphics::sprite_t s{{
          .position = position,
          .size = size,
          .color = color
        }};
      if (image_name.empty()) {
        return *dynamic_cast<loco_t::shape_t*>(&s);
      }

      loco_t::texturepack_t::ti_t ti;
      if (tp->qti(image_name, &ti)) {
        fan::print_no_space("failed to load texture:", image_name);
      }
      else {
        auto& data = tp->get_pixel_data(ti.pack_id);
        gloco->sprite.load_tp(s, &ti);
      }
      return *dynamic_cast<loco_t::shape_t*>(&s);
    }
  #endif
  };

  struct shapes_t {
    sprite_t sprite;
  };
};

static constexpr uint32_t version_001 = 1;

static constexpr uint32_t current_version = version_001;
using current_version_t = version_001_t;

#undef only_struct_data