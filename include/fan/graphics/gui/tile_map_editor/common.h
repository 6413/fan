struct version_001_t {
  struct shapes_t {
    struct tile_t {

      #if !defined(tile_map_editor_loader)

      void init(ftme_t* ftme, ftme_t::shapes_t::global_t* g) {
        for (auto& i : dynamic_cast<fan::graphics::vfi_root_custom_t<ftme_t::tile_t>*>(g)->children) {
          ftme_t::tile_t layer;
          layer.position = i.get_position();
          layer.image_hash = i.image_hash;
          layer.mesh_property = i.mesh_property;
          layers.push_back(layer);
        }
      }

      void get_shape(ftme_t* ftme) {
        fan::vec2ui grid_idx(
          (layers[0].position.x - ftme->tile_size.x) / ftme->tile_size.x / 2,
          (layers[0].position.y - ftme->tile_size.y) / ftme->tile_size.y / 2
        );
        auto& instance = ftme->map_tiles[grid_idx.y][grid_idx.x];
        instance->set_position(fan::vec3(
          fan::vec2(layers[0].position.x,
            layers[0].position.y
          ),
          0));
        instance->children[0].set_size(ftme->tile_size);

        for (int i = 1; i < layers.size(); ++i) {
          ftme_t::tile_t& layer = layers[i];
          fan::graphics::vfi_root_custom_t<ftme_t::tile_t>::child_data_t cd = {{}, layer};
          instance->children.push_back(cd);
          auto& shape = *dynamic_cast<loco_t::shape_t*>(&instance->children.back());
          shape = fan::graphics::sprite_t{{
              .position = layer.position,
              .size = ftme->tile_size
          }};
          fan::vec3 p = shape.get_position();
          shape.set_position(p);
          loco_t::texturepack_t::ti_t ti;
          if (ftme->texturepack.qti(layer.image_hash, &ti)) {
            fan::throw_error("failed to read image from .ftme - editor save file corrupted");
          }
          gloco->shapes.sprite.load_tp(
            shape,
            &ti
          );
        }
        layers.clear();
      }
      #endif

      std::vector<ftme_t::tile_t> layers;
    }tile; // dummy for iterating struct
  };
};

using current_version_t = version_001_t;

static constexpr int current_version = 001;

#undef tile_map_editor_loader