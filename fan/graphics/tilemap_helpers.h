namespace fan::graphics {
  namespace tilemap::helpers {
    static fan::vec3 get_spawn_position_or_default(tilemap_renderer_t& renderer,
      tilemap_renderer_t::id_t map_id,
      const std::string& id = "",
      const fan::vec3& default_pos = 0) {
      try {
        return renderer.get_spawn_position(map_id, id);
      }
      catch (...) {
        return default_pos;
      }
    }

    static void setup_lights(tilemap_renderer_t& renderer,
      tilemap_renderer_t::id_t map_id,
      const std::string& light_prefix,
      auto callback) {
      int index = 0;
      while (auto* light = renderer.get_light_by_id(map_id, light_prefix + std::to_string(index))) {
        callback(light, index);
        ++index;
      }
    }
  };
}