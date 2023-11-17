struct version_001_t {
  struct shapes_t {
    struct tile_t {

      #if !defined(tilemap_editor_loader)

      void init(fte_t* fte, fte_t::shapes_t::global_t* g) {
        for (auto& i : g->layers) {
          fte_t::tile_t layer;
          layer.position = i.shape.get_position();
          layer.angle = i.shape.get_angle();
          layer.image_hash = i.tile.image_hash;
          layer.mesh_property = i.tile.mesh_property;
          static int z = 0;
          if (layer.mesh_property == fte_t::mesh_property_t::collider) {
            z++;
            fan::print(z);
          }
          layers.push_back(layer);
        }
      }

      void get_shape(fte_t* fte) {
        for (int i = 0; i < layers.size(); ++i) {
          fte_t::tile_t& layer = layers[i];
          auto found = fte->map_tiles.find(fan::vec2i(layer.position.x / fte->tile_size.x, layer.position.y / fte->tile_size.y));
          if (found != fte->map_tiles.end()) {
            auto& map_tile = found->second;
            loco_t::texturepack_t::ti_t ti;
            if (fte->texturepack.qti(layer.image_hash, &ti)) {
              fan::throw_error("failed to read image from .fte - editor save file corrupted");
            }

            fte_t::shapes_t::global_t::layer_t map_layer;
            map_layer.tile = layer;
            map_layer.shape = fan::graphics::sprite_t{{
                .position = layer.position,
                .size = fte->tile_size,
                .angle = layer.angle,
                .blending = true
              }};

            map_tile.layers.push_back(std::move(map_layer));
            switch (layer.mesh_property) {
              case fte_t::mesh_property_t::none: {
                map_tile.layers.back().shape.set_tp(&ti);
                break;
              }
              case fte_t::mesh_property_t::collider: {
                map_tile.layers.back().shape.set_image(&fte->grid_visualize.collider_color);
                break;
              }
            }
          }
        }
        layers.clear();
      }
      #endif

      std::vector<fte_t::tile_t> layers;
    }tile; // dummy for iterating struct
  };
};

using current_version_t = version_001_t;

static constexpr int current_version = 001;

#undef tilemap_editor_loader