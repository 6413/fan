struct version_001_t {
  struct shapes_t {
    struct tile_t {

      #if !defined(tile_map_editor_loader)

      void init(ftme_t::shapes_t::global_t* g) {
        position = g->get_position();
        image_hash = g->shape_data.tile.image_hash;
        mesh_property = g->shape_data.tile.mesh_property;
        color_idx = g->shape_data.tile.color_idx;
      }

      void get_shape(ftme_t* ftme) {
        fan::vec2ui grid_idx(
          (position.x - ftme->tile_size.x) / ftme->tile_size.x / 2,
          (position.y - ftme->tile_size.y) / ftme->tile_size.y / 2
        );
        auto& instance = ftme->map_tiles[grid_idx.y][grid_idx.x];
        instance = std::make_unique<ftme_t::shapes_t::global_t>(ftme, fan::graphics::sprite_t{{
          .position = fan::vec3(position, 0),
          .size = ftme->tile_size
        }});

        loco_t::texturepack_t::ti_t ti;
        if (ftme->texturepack.qti(image_hash, &ti)) {
          fan::throw_error("failed to read image from .ftme - editor save file corrupted");
        }
        gloco->shapes.sprite.load_tp(
          instance->children[0],
          &ti
        );
        instance->shape_data.tile.image_hash = image_hash;
        instance->shape_data.tile.mesh_property = mesh_property;
        instance->shape_data.tile.color_idx = color_idx;
      }
      #endif

      fan::vec2ui position;
      uint64_t image_hash;
      uint8_t mesh_property;
      uint8_t color_idx;
    }tile; // dummy for iterating struct
  };
};

using current_version_t = version_001_t;

static constexpr int current_version = 001;

#undef tile_map_editor_loader