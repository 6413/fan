struct version_001_t {

  struct sprite_t {

    static constexpr typename loco_t::shape_type_t shape_type = loco_t::shape_type_t::sprite;
    using type_t = fan::graphics::sprite_t;

    fan::vec3 position;
    fan::vec2 size;
    fan::color color;
    fan::string image_name;

    // global
    fan::string id;
    uint32_t group_id;

  #if !defined(stage_maker_loader) && !defined(model_maker_loader)

    void init(auto& shape) {
      position = shape->children[0].get_position();
      size = shape->children[0].get_size();
      color = shape->children[0].get_color();
      image_name = shape->shape_data.sprite.image_name;
      id = shape->id;
      group_id = shape->group_id;
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
        gloco->shapes.sprite.load_tp(ret->children[0], &ti);
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
        gloco->shapes.sprite.load_tp(s, &ti);
      }
      return *dynamic_cast<loco_t::shape_t*>(&s);
    }
  #endif
  };

  struct unlit_sprite_t {
    static constexpr typename loco_t::shape_type_t shape_type = loco_t::shape_type_t::unlit_sprite;
    using type_t = fan::graphics::unlit_sprite_t;

    fan::vec3 position;
    fan::vec2 size;
    fan::color color;
    fan::string image_name;

    // global
    fan::string id;
    uint32_t group_id;

    #if !defined(stage_maker_loader) && !defined(model_maker_loader)

    void init(auto& shape) {
      position = shape->children[0].get_position();
      size = shape->children[0].get_size();
      color = shape->children[0].get_color();
      image_name = shape->shape_data.sprite.image_name;
      id = shape->id;
      group_id = shape->group_id;
    }

    shapes_t::global_t* get_shape(fgm_t* fgm) {
      fgm_t::shapes_t::global_t* ret = new fgm_t::shapes_t::global_t(
        fgm,
       fan::graphics::unlit_sprite_t{{
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
        gloco->shapes.unlit_sprite.load_tp(ret->children[0], &ti);
      }

      return ret;
    }
    #else
    loco_t::shape_t get_shape(loco_t::texturepack_t* tp) {
      fan::graphics::unlit_sprite_t s{{
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
        gloco->shapes.unlit_sprite.load_tp(s, &ti);
      }
      return *dynamic_cast<loco_t::shape_t*>(&s);
    }
    #endif
  };

  struct mark_t {
    static constexpr typename loco_t::shape_type_t shape_type = loco_t::shape_type_t::mark;
    using type_t = fan::graphics::rectangle_t;

    fan::vec3 position;

    // global
    fan::string id;
    uint32_t group_id;

    #if !defined(stage_maker_loader) && !defined(model_maker_loader)

    void init(auto& shape) {
      position = shape->children[0].get_position();
      id = shape->id;
      group_id = shape->group_id;
    }

    shapes_t::global_t* get_shape(fgm_t* fgm) {
      fgm_t::shapes_t::global_t* ret = new fgm_t::shapes_t::global_t(
        fgm,
       fan::graphics::rectangle_t{{
           .position = position,
           .size = 10,
           .color = fan::colors::white
         }}
      );
      ret->id = id;
      ret->group_id = group_id;
      return ret;
    }
    #else
    loco_t::shape_t get_shape(loco_t::texturepack_t* tp) {
      fan::graphics::rectangle_t s{{
          .position = position,
          .size = 50,
          .color = fan::colors::white
        }};
      return *dynamic_cast<loco_t::shape_t*>(&s);
    }
    #endif
  };

  struct shapes_t {
    sprite_t sprite;
    unlit_sprite_t unlit_sprite;
    mark_t mark;
  };
};

static constexpr uint32_t version_001 = 1;

static constexpr uint32_t current_version = version_001;
using current_version_t = version_001_t;

#undef only_struct_data
#undef model_maker_loader