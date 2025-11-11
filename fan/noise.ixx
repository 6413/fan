module;

#include <vector>
#include <cstdint>
#include <algorithm>

export module fan.noise;

import fan.types;
import fan.graphics;
import fan.random;

#include <fan/graphics/algorithm/FastNoiseLite.h>

export namespace fan {

  struct noise_t {
    FastNoiseLite fn;

    int seed = fan::random::value((uint32_t)0, ((uint32_t)-1) / 2);
    f32_t frequency = 0.01f;
    f32_t gain = 0.5f;
    f32_t lacunarity = 2.0f;
    int octaves = 5;

    void apply() {
      fn.SetSeed(seed);
      fn.SetFrequency(frequency);
      fn.SetFractalGain(gain);
      fn.SetFractalLacunarity(lacunarity);
      fn.SetFractalOctaves(octaves);
      fn.SetFractalType(FastNoiseLite::FractalType_FBm);
      fn.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
      fn.SetFractalPingPongStrength(2.0f);
    }

    f32_t get_noise(f32_t x, f32_t y) const {
      return fn.GetNoise(x, y);
    }

    std::vector<uint8_t> generate_data(
      const vec2& size,
      f32_t tex_min = -1.0f,
      f32_t tex_max = 1.0f
    ) const {
      std::vector<uint8_t> data(size.multiply() * 3);
      f32_t scale = 255.0f / (tex_max - tex_min);

      int idx = 0;
      for (int y = 0; y < size.y; ++y) {
        for (int x = 0; x < size.x; ++x) {
          f32_t val = get_noise((f32_t)x, (f32_t)y);
          unsigned char c = (unsigned char)std::clamp((val - tex_min) * scale, 0.0f, 255.0f);
          data[idx * 3 + 0] = c;
          data[idx * 3 + 1] = c;
          data[idx * 3 + 2] = c;
          ++idx;
        }
      }
      return data;
    }

    graphics::image_t to_image(const vec2& size,
                               f32_t tex_min = -1.0f,
                               f32_t tex_max = 0.1f) const {
      auto data = generate_data(size, tex_min, tex_max);
      return from_data(size, data);
    }

    static graphics::image_t from_data(const vec2& size,
                                       const std::vector<uint8_t>& data) {
      graphics::image_load_properties_t lp;
      image::info_t ii;
      ii.data = (void*)data.data();
      ii.size = size;
      ii.channels = 3;
      return image_load(ii, lp);
    }
  };

}