module;

export module fan.graphics.voxel;

#if defined (FAN_WINDOW)

#if defined(FAN_3D)

import std;
import fan.noise;
import fan.types;
import fan.types.vector;
import fan.types.flat_hash_map;
import fan.graphics.shapes;

export namespace fan::graphics {

  struct terrain_noise_t {
    int   seed          = 42;
    f32_t frequency     = 0.01f;
    f32_t height_scale  = 5.f;
    f32_t height_offset = 0.f;

    void apply() {
      noise.seed = seed;
      noise.frequency = frequency;
      noise.apply();
    }

    int sample_height(int x, int z) const {
      return static_cast<int>((noise.get_noise(x, z) + 1.f) * height_scale + height_offset);
    }

    fan::noise_t noise;
  };

  struct voxel_world_t {
    using generator_t = std::function<std::optional<fan::graphics::shapes::rectangle3d_t::properties_t>(fan::vec3i)>;

    void set_generator(generator_t gen);
    void clear();
    void update(const fan::vec3& camera_pos, int view_dist);

    f32_t block_size = 10.f;
    fan::vec3 last_center{};
    fan::flat_map_t<fan::vec3i, fan::graphics::shape_t> blocks;
    generator_t generator;
  };

} // namespace fan::graphics

#endif

#endif