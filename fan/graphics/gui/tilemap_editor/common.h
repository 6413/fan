struct version_001_t {
  struct shapes_t {
    struct tile_t {

      #if !defined(tilemap_editor_loader)

      void init(fte_t* fte, fte_t::shapes_t::global_t* g) {
       /* for (auto& i : dynamic_cast<fan::graphics::vfi_root_custom_t<fte_t::tile_t>*>(g)->children) {
          fte_t::tile_t layer;
          layer.position = i.get_position();
          layer.image_hash = i.image_hash;
          layer.mesh_property = i.mesh_property;
          layers.push_back(layer);
        }*/
      }

      void get_shape(fte_t* fte) {
        /*fan::vec2ui grid_idx(
          (layers[0].position.x - fte->tile_size.x) / fte->tile_size.x / 2,
          (layers[0].position.y - fte->tile_size.y) / fte->tile_size.y / 2
        );
        auto& instance = fte->map_tiles[grid_idx.y][grid_idx.x];
        instance->set_position(fan::vec3(
          fan::vec2(layers[0].position.x,
            layers[0].position.y
          ),
          0));
        instance->children[0].set_size(fte->tile_size);

        for (int i = 1; i < layers.size(); ++i) {
          fte_t::tile_t& layer = layers[i];
          fan::graphics::vfi_root_custom_t<fte_t::tile_t>::child_data_t cd = {{}, layer};
          instance->children.push_back(cd);
          auto& shape = *dynamic_cast<loco_t::shape_t*>(&instance->children.back());
          shape = fan::graphics::sprite_t{{
              .position = layer.position,
              .size = fte->tile_size
          }};
          fan::vec3 p = shape.get_position();
          shape.set_position(p);
          loco_t::texturepack_t::ti_t ti;
          if (fte->texturepack.qti(layer.image_hash, &ti)) {
            fan::throw_error("failed to read image from .fte - editor save file corrupted");
          }
          gloco->shapes.sprite.load_tp(
            shape,
            &ti
          );
        }
        layers.clear();*/
      }
      #endif

      std::vector<fte_t::tile_t> layers;
    }tile; // dummy for iterating struct
  };
};

using current_version_t = version_001_t;

static constexpr int current_version = 001;

#undef tilemap_editor_loader